import torch
import fst
import numpy
import math
import itertools
from picklable_itertools.extras import equizip
from collections import defaultdict, deque
from toposort import toposort_flatten
#import FST#, FSTCostsOp, FSTTransitionOp, MAX_STATES, NOT_STATE

EPSILON = 0
MAX_STATES = 200
NOT_STATE = -1

class FST(object):

    """Picklable wrapper around FST."""
    def __init__(self, path):
        self.path = path
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

class LanguageModel(object):

    def __init__(self, path, nn_char_map):
        self.no_transition_cost = 1e12
        self.fst = FST(path)
        self.fst_char_map = dict(self.fst.fst.isyms.items())

        print "FST (before):", self.fst_char_map
        dels = ['<eps>'] #, '-'] #'+', '<noise>'
        for d in dels:
            del self.fst_char_map[d]
            
        reps = [('<bol>', '<s>'), ('<eol>', '</s>'), ('<spc>', 'SPACE'), ('+', '\'')] #, ('<eps>', '<blank>')]
        for (orig, rep) in reps:
            t = self.fst_char_map[orig]
            del self.fst_char_map[orig]
            self.fst_char_map[rep] = t
            
        print "NN:", nn_char_map
        print "FST:", self.fst_char_map
        #if not len(self.fst_char_map) == len(nn_char_map):
        #    raise ValueError()
        for character, fst_code in self.fst_char_map.items():
            print character, fst_code, nn_char_map[character]
        self.remap_table = {nn_char_map[character]: fst_code
                       for character, fst_code in self.fst_char_map.items()}
        print self.remap_table
        
    def pad(self, arr, value):
        return numpy.pad(arr, (0, MAX_STATES - len(arr)),
                         mode='constant', constant_values=value)
    
    # from FST.FSTTransition
    # JD FIX THE THEANO STUFF!!!
    def initial_states(self, batch_size):
        states_dict = self.fst.expand({self.fst.fst.start: 0.0})
        #states = torch.Tensor(self.pad(states_dict.keys(), NOT_STATE))
        #states = states.repeat(batch_size, 1)
        #weights = torch.Tensor(self.pad(states_dict.values(), 0))
        #weights = weights.repeat(batch_size, 1)
        states = numpy.expand_dims(self.pad(states_dict.keys(), NOT_STATE), 0)
        #print states.shape
        states = states.repeat(batch_size, 0)
        #print states.shape
        weights = numpy.expand_dims(self.pad(states_dict.values(), 0), 0)
        weights = weights.repeat(batch_size, 0)
        costs = self.probability_computer(states, weights)
        #print len(costs)
        return states, weights, costs


    def transition(self, all_inputs, all_states, all_weights):
        # Each row of all_states contains a set of states
        # padded with NOT_STATE.
        print len(all_inputs), len(all_states), len(all_weights)
        all_next_states = []
        all_next_weights = []
        for states, weights, input_ in equizip(all_states, all_weights, all_inputs):
            states_dict = dict(zip(states, weights))
            try:
                del states_dict[NOT_STATE]
            except:
                print "couldn't delete"
            
            #print input_
            #print states_dict
            try:
                next_states_dict = self.fst.transition(states_dict, self.remap_table[input_])
            except:
                print "TRANSITION ERROR:", input_
                next_states_dict = self.fst.transition(states_dict, self.remap_table[2])
            next_states_dict = self.fst.expand(next_states_dict)
            
            if next_states_dict:
                next_states, next_weights = zip(*next_states_dict.items())
            else:
                # No adequate state when no arc exists for now
                next_states, next_weights = [], []

            #print next_states
            #print next_weights
            #print
            
            all_next_states.append(self.pad(next_states, NOT_STATE))
            all_next_weights.append(self.pad(next_weights, 0))

        return all_next_states, all_next_weights

    def probability_computer(self, all_states, all_weights):
        all_costs = []
        #print "pc:", len(all_states)
        for states, weights in zip(all_states, all_weights):
            states_dict = dict(zip(states, weights))
            try:
                del states_dict[NOT_STATE]
            except:
                print "couldn't delete"
                
            costs = (numpy.ones(len(self.remap_table)+2, dtype=numpy.float32)
                     * self.no_transition_cost)
            if states_dict:
                #print "dict:", states_dict
                total_weight = self.fst.combine_weights(*states_dict.values())
                #print total_weight
                for nn_character, fst_character in self.remap_table.items():
                    #print "chars:", nn_character, fst_character
                    next_states_dict = self.fst.transition(states_dict, fst_character)
                    #print "trans:", next_states_dict
                    next_states_dict = self.fst.expand(next_states_dict)
                    #print "expand:", next_states_dict
                    if next_states_dict:
                        next_total_weight = self.fst.combine_weights(*next_states_dict.values())
                        #print "next weight:", next_total_weight
                        costs[nn_character] = next_total_weight - total_weight
            all_costs.append(costs)
            
        return all_costs

