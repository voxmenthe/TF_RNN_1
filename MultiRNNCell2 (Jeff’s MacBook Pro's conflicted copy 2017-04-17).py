from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    zeros_flat = [
        array_ops.zeros(
            array_ops.stack(_state_size_with_prefix(
                s, prefix=[batch_size])),
            dtype=dtype) for s in state_size_flat
    ]
    for s, z in zip(state_size_flat, zeros_flat):
      z.set_shape(_state_size_with_prefix(s, prefix=[None]))
    zeros = nest.pack_sequence_as(structure=state_size,
                                  flat_sequence=zeros_flat)
  else:
    zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
    zeros = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)
    zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

  return zeros

class MultiRNNCell2(object):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        # return super(MultiRNNCell, self).zero_state(batch_size, dtype) # original line
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
          state_size = self.state_size
          return _zero_state_tensors(state_size, batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or "multi_rnn_cell"):
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("cell_%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    return cur_inp, new_states
