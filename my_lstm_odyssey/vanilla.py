import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

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

class VanillaLSTMCell(object):
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
            
            b_z = get_variable("b_z", [self.input_size, self._num_blocks])
            b_i = get_variable("b_i", [self.input_size, self._num_blocks])
            b_f = get_variable("b_f", [self.input_size, self._num_blocks])
            b_o = get_variable("b_o", [self.input_size, self._num_blocks])
            
            p_i = get_variable("p_i", [self.input_size, self._num_blocks])
            p_f = get_variable("p_f", [self.input_size, self._num_blocks])
            p_o = get_variable("p_o", [self.input_size, self._num_blocks])
            
            # define each equation as operations in the graph
            # many have reversed inputs so that matmuls produce correct dimensionality
            z = tf.tanh(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.mul(c_prev, p_i) + b_i)
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f), + tf.mul(c_prev, p_f) + b_f)
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.mul(c, p_o) + b_o)
            y = tf.mul(tf.tanh(c), o)
            
            return y, tf.concat([c,y],1)
