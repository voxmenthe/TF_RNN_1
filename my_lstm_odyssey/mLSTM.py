import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
# from tf.contrib.rnn import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell



def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.
  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.
  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.
  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    zeros_flat = [tf.zeros(tf.stack(_state_size_with_prefix(s, prefix=[batch_size])),dtype=dtype) for s in state_size_flat]
    for s, z in zip(state_size_flat, zeros_flat):
      z.set_shape(_state_size_with_prefix(s, prefix=[None]))
    zeros = nest.pack_sequence_as(structure=state_size,
                                  flat_sequence=zeros_flat)
  else:
    zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
    zeros = tf.zeros(array_ops.stack(zeros_size), dtype=dtype)
    zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

  return zeros

class MILSTMCell(RNNCell):
    def __init__(self, num_blocks):
        self._num_blocks = num_blocks
        
    @property
    def input_size(self):
        return self._num_blocks
    
    @property
    def output_size(self):
        return self._num_blocks
    
    @property
    def state_size(self):
        return 2 * self._num_blocks

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size x s]` for each s in `state_size`.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
          state_size = self.state_size
          return _zero_state_tensors(state_size, batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            
            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)
            
            c_prev, y_prev = tf.split(state, 2, 1)
            
            # initialize all params using `get_variable` so we can reuse vars at each time-step
            # instead of creating new params at each step
            # also, all params are transposed from the paper's definitions to avoid additional graph operations
            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_i = get_variable("W_i", [self.input_size, self._num_blocks])
            W_f = get_variable("W_f", [self.input_size, self._num_blocks])
            W_o = get_variable("W_o", [self.input_size, self._num_blocks])
            
            R_z = get_variable("R_z", [self.input_size, self._num_blocks])
            R_i = get_variable("R_i", [self.input_size, self._num_blocks])
            R_f = get_variable("R_f", [self.input_size, self._num_blocks])
            R_o = get_variable("R_o", [self.input_size, self._num_blocks])
            
            b_z = get_variable("b_z", [1, self._num_blocks])
            b_i = get_variable("b_i", [1, self._num_blocks])
            b_f = get_variable("b_f", [1, self._num_blocks])
            b_o = get_variable("b_o", [1, self._num_blocks])


            p_i = get_variable("p_i", [self._num_blocks])
            p_f = get_variable("p_f", [self._num_blocks])
            p_o = get_variable("p_o", [self._num_blocks])
            
            # define each equation as operations in the graph
            # many have reversed inputs so that matmuls produce correct dimensionality
            
            # for mLSTM we first need to define an intermediate state m_t
            # first need to implement the W_hh_(xt) diagonal matrix among other things

            m_t = 0 # TO BE IMPLEMENTED

            # AND THEN IN THE IMPLEMENTATIONS BELOW, I THINK ALL THE y_prev
            # NEED TO BE REPLACED WITH m_t

            # z = tf.tanh(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            z = tf.tanh(tf.matmul(inputs, W_z) + tf.matmul(m_t, R_z) + b_z)
         
            # i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.multiply(c_prev, p_i) + b_i)

            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(m_t, R_i) + tf.multiply(c_prev, p_i) + b_i)


            # f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f) + tf.multiply(c_prev, p_f) + b_f)

            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(m_t, R_f) + tf.multiply(c_prev, p_f) + b_f)


            # c remains unchanged for mLSTM ? ??????? not sure yet
            c = tf.multiply(i, z) + tf.multiply(f, c_prev)

            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.multiply(c, p_o) + b_o)

            # changed in mLSTM:
            # y = tf.multiply(tf.tanh(c), o)
            y = tf.tanh(tf.multiply(c, o))

            
            return y, tf.concat([c,y],1)






# class RNNCell(object):
#   """Abstract object representing an RNN cell.
#   Every `RNNCell` must have the properties below and implement `__call__` with
#   the following signature.
#   This definition of cell differs from the definition used in the literature.
#   In the literature, 'cell' refers to an object with a single scalar output.
#   This definition refers to a horizontal array of such units.
#   An RNN cell, in the most abstract setting, is anything that has
#   a state and performs some operation that takes a matrix of inputs.
#   This operation results in an output matrix with `self.output_size` columns.
#   If `self.state_size` is an integer, this operation also results in a new
#   state matrix with `self.state_size` columns.  If `self.state_size` is a
#   tuple of integers, then it results in a tuple of `len(state_size)` state
#   matrices, each with a column size corresponding to values in `state_size`.
#   """

#   def __call__(self, inputs, state, scope=None):
#     """Run this RNN cell on inputs, starting from the given state.
#     Args:
#       inputs: `2-D` tensor with shape `[batch_size x input_size]`.
#       state: if `self.state_size` is an integer, this should be a `2-D Tensor`
#         with shape `[batch_size x self.state_size]`.  Otherwise, if
#         `self.state_size` is a tuple of integers, this should be a tuple
#         with shapes `[batch_size x s] for s in self.state_size`.
#       scope: VariableScope for the created subgraph; defaults to class name.
#     Returns:
#       A pair containing:
#       - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
#       - New state: Either a single `2-D` tensor, or a tuple of tensors matching
#         the arity and shapes of `state`.
#     """
#     raise NotImplementedError("Abstract method")

#   @property
#   def state_size(self):
#     """size(s) of state(s) used by this cell.
#     It can be represented by an Integer, a TensorShape or a tuple of Integers
#     or TensorShapes.
#     """
#     raise NotImplementedError("Abstract method")

#   @property
#   def output_size(self):
#     """Integer or TensorShape: size of outputs produced by this cell."""
#     raise NotImplementedError("Abstract method")

#   def zero_state(self, batch_size, dtype):
#     """Return zero-filled state tensor(s).
#     Args:
#       batch_size: int, float, or unit Tensor representing the batch size.
#       dtype: the data type to use for the state.
#     Returns:
#       If `state_size` is an int or TensorShape, then the return value is a
#       `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
#       If `state_size` is a nested list or tuple, then the return value is
#       a nested list or tuple (of the same structure) of `2-D` tensors with
#       the shapes `[batch_size x s]` for each s in `state_size`.
#     """
#     with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
#       state_size = self.state_size
#       return _zero_state_tensors(state_size, batch_size, dtype)

