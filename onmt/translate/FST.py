import torch
import fst
from collections import defaultdict, deque
from toposort import toposort_flatten

#from ops import FST, FSTCostsOp, FSTTransitionOp, MAX_STATES, NOT_STATE

def read_symbols(fname):
    syms = fst.SymbolTable('eps')
    with open(fname) as sf:
        for line in sf:
            s,i = line.strip().split()
            syms[s] = int(i)
    return syms

class FST(object):

    """Picklable wrapper around FST."""
    def __init__(self, path):
        self.path = path

    def load(self):
        self.fst = fst.read(self.path)
        self.isyms = dict(self.fst.isyms.items())

    def __getitem__(self, state):
        """Returns all arcs of the state i"""
        return self.fst[state]

    def combine_weights(self, *args):
        # Protection from underflow when -x is too small
        m = max(args)
        return m - math.log(sum(math.exp(m - x) for x in args if x is not None))

    def get_arcs(self, state, character):
        return [(state, arc.nextstate, arc.ilabel, float(arc.weight))
                for arc in self[state] if arc.ilabel == character]

    def transition(self, states, character):
        arcs = list(itertools.chain(
            *[self.get_arcs(state, character) for state in states]))
        next_states = {}
        for next_state in {arc[1] for arc in arcs}:
            next_states[next_state] = self.combine_weights(
                *[states[arc[0]] + arc[3] for arc in arcs
                  if arc[1] == next_state])
        return next_states

    def expand(self, states):
        seen = set()
        depends = defaultdict(list)
        queue = deque()
        for state in states:
            queue.append(state)
            seen.add(state)
        while len(queue):
            state = queue.popleft()
            for arc in self.get_arcs(state, EPSILON):
                depends[arc[1]].append((arc[0], arc[3]))
                if arc[1] in seen:
                    continue
                queue.append(arc[1])
                seen.add(arc[1])

        depends_for_toposort = {key: {state for state, weight in value}
                                for key, value in depends.items()}
        order = toposort_flatten(depends_for_toposort)

        next_states = states
        for next_state in order:
            next_states[next_state] = self.combine_weights(
                *([next_states.get(next_state)] +
                  [next_states[prev_state] + weight
                   for prev_state, weight in depends[next_state]]))

        return next_states

    def explain(self, input_):
        input_ = list(input_)
        states = {self.fst.start: 0}
        print("Initial states: {}".format(states))
        states = self.expand(states)
        print("Expanded states: {}".format(states))

        for char, ilabel in zip(input_, [self.isyms[char] for char in input_]):
            states = self.transition(states, ilabel)
            print("{} consumed: {}".format(char, states))
            states = self.expand(states)
            print("Expanded states: {}".format(states))

        result = None
        for state, weight in states.items():
            if numpy.isfinite(weight + float(self.fst[state].final)):
                print("Finite state {} with path weight {} and its own weight {}".format(
                    state, weight, self.fst[state].final))
                result = self.combine_weights(
                    result, weight + float(self.fst[state].final))

        print("Total weight: {}".format(result))
        return result
    

class FSTTransition(BaseRecurrent, Initializable):
    def __init__(self, fst, remap_table, no_transition_cost, **kwargs):
        """Wrap FST in a recurrent brick.

        Parameters
        ----------
        fst : FST instance
        remap_table : dict
            Maps neutral network characters to FST characters.
        no_transition_cost : float
            Cost of going to the start state when no arc for an input
            symbol is available.

        """
        super(FSTTransition, self).__init__(**kwargs)
        self.fst = fst
        self.transition = FSTTransitionOp(fst, remap_table)
        self.probability_computer = FSTCostsOp(
            fst, remap_table, no_transition_cost)

        self.out_dim = len(remap_table)

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'weights', 'add'],
               outputs=['states', 'weights', 'add'], contexts=[])
    def apply(self, inputs, states, weights, add,
              mask=None):
        new_states, new_weights = self.transition(states, weights, inputs)
        if mask:
            # In fact I don't really understand why we do this:
            # anyway states not covered by masks should have no effect
            # on the cost...
            new_states = tensor.cast(mask * new_states +
                                     (1. - mask) * states, 'int64')
            new_weights = mask * new_weights + (1. - mask) * weights
        new_add = self.probability_computer(new_states, new_weights)
        return new_states, new_weights, new_add

    @application(outputs=['states', 'weights', 'add'])
    def initial_states(self, batch_size, *args, **kwargs):
        states_dict = self.fst.expand({self.fst.fst.start: 0.0})
        states = tensor.as_tensor_variable(
            self.transition.pad(states_dict.keys(), NOT_STATE))
        states = tensor.tile(states[None, :], (batch_size, 1))
        weights = tensor.as_tensor_variable(
            self.transition.pad(states_dict.values(), 0))
        weights = tensor.tile(weights[None, :], (batch_size, 1))
        add = self.probability_computer(states, weights)
        return states, weights, add

    def get_dim(self, name):
        if name == 'states' or name == 'weights':
            return MAX_STATES
        if name == 'add':
            return self.out_dim
        if name == 'inputs':
            return 0
        return super(FSTTransition, self).get_dim(name)


class ShallowFusionReadout(Readout):
    def __init__(self, lm_costs_name, lm_weight,
                 normalize_am_weights=False,
                 normalize_lm_weights=False,
                 normalize_tot_weights=True,
                 am_beta=1.0,
                 **kwargs):
        super(ShallowFusionReadout, self).__init__(**kwargs)
        self.lm_costs_name = lm_costs_name
        self.lm_weight = lm_weight
        self.normalize_am_weights = normalize_am_weights
        self.normalize_lm_weights = normalize_lm_weights
        self.normalize_tot_weights = normalize_tot_weights
        self.am_beta = am_beta
        self.softmax = NDimensionalSoftmax()
        self.children += [self.softmax]

    @application
    def readout(self, **kwargs):
        lm_costs = -kwargs.pop(self.lm_costs_name)
        if self.normalize_lm_weights:
            lm_costs = self.softmax.log_probabilities(
                lm_costs, extra_ndim=lm_costs.ndim - 2)
        am_pre_softmax = self.am_beta * super(ShallowFusionReadout, self).readout(**kwargs)
        if self.normalize_am_weights:
            am_pre_softmax = self.softmax.log_probabilities(
                am_pre_softmax, extra_ndim=am_pre_softmax.ndim - 2)
        x = am_pre_softmax + self.lm_weight * lm_costs
        if self.normalize_tot_weights:
            x = self.softmax.log_probabilities(x, extra_ndim=x.ndim - 2)
        return x


class LanguageModel(SequenceGenerator):
    def __init__(self, path, nn_char_map, no_transition_cost=1e12, **kwargs):
        # Since we currently support only type, it is ignored.
        # if type_ != 'fst':
        #    raise ValueError("Supports only FST's so far.")
        fst = FST(path)
        fst_char_map = dict(fst.fst.isyms.items())
        del fst_char_map['<eps>']
        if not len(fst_char_map) == len(nn_char_map):
            raise ValueError()
        remap_table = {nn_char_map[character]: fst_code
                       for character, fst_code in fst_char_map.items()}
        transition = FSTTransition(fst, remap_table, no_transition_cost)

        # This SequenceGenerator will be used only in a very limited way.
        # That's why it is sufficient to equip it with a completely
        # fake readout.
        dummy_readout = Readout(
            source_names=['add'], readout_dim=len(remap_table),
            merge=Merge(input_names=['costs'], prototype=Identity()),
            post_merge=Identity(),
            emitter=SoftmaxEmitter())
        super(LanguageModel, self).__init__(
            transition=transition,
            fork=Fork(output_names=[name for name in transition.apply.sequences
                                    if name != 'mask'],
                      prototype=Identity()),
            readout=dummy_readout, **kwargs)


class SelectInEachRow(Brick):
    @application(inputs=['matrix', 'indices'], outputs=['output_'])
    def apply(self, matrix, indices):
        return matrix[tensor.arange(matrix.shape[0]), indices]


class SelectInEachSubtensor(SelectInEachRow):
    decorators = [WithExtraDims()]


class LMEmitter(AbstractEmitter):
    """Emitter to use when language model is used.

    Since with the language model all normalization is
    done in ShallowFusionReadout, we need this no-op brick to
    interface it with BeamSearch.

    """
    @lazy(allocation=['readout_dim'])
    def __init__(self, readout_dim, **kwargs):
        super(LMEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim
        self.select = SelectInEachSubtensor()
        self.children = [self.select]

    @application
    def emit(self, readouts):
        # Non-sense, but the returned result should never be used.
        return tensor.zeros_like(readouts[:, 0], dtype='int64')

    @application
    def cost(self, readouts, outputs):
        return -self.select.apply(
            readouts, outputs, extra_ndim=readouts.ndim - 2)

    @application
    def costs(self, readouts):
        return -readouts

    @application
    def initial_outputs(self, batch_size):
        # As long as we do not use the previous character, can be anything
        return tensor.zeros((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(LMEmitter, self).get_dim(name)
