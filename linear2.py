def linear2(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  weights = vs.get_variable(
      _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
  if len(args) == 1:
    res = math_ops.matmul(args[0], weights)
  else:
    res = math_ops.matmul(array_ops.concat(args, 1), weights)
  if not bias:
    return res
  biases = vs.get_variable(
      _BIAS_VARIABLE_NAME, [output_size],
      dtype=dtype,
      initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return nn_ops.bias_add(res, biases)