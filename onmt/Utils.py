import torch


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def sequence_mask_window(start, ends, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = ends.numel()
    max_len = max_len or ends.max()
    end_mask = (torch.arange(0, max_len)
                .type_as(ends)
                .repeat(batch_size, 1)
                .lt(ends.unsqueeze(1)))

    start_mask = (torch.arange(0, max_len)
                  .type_as(ends)
                  .repeat(batch_size, 1)
                  .gt(start))

    total_mask = (end_mask + start_mask).eq(2)
    return total_mask

def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)
