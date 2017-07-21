import tensorflow as tf


def weight_variable_for_selu(shape, name=None):
    units = shape[0]
    if len(shape) == 4:
        units *= shape[1]*shape[2]
    else:
        assert "Invalid shape for weights of selu"

    return tf.get_variable(
        name+"/weight",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=tf.sqrt(1.0/units)),  # initialization for S-ELU
        dtype=tf.float32
    )


def bias_variable_for_selu(shape, name=None):
    return tf.get_variable(
        name+"/bias",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=.0),  # initialization for S-ELU
        dtype=tf.float32
    )


def elu(input_t, alpha=1.0, name="elu"):
    with tf.variable_scope(name):
        return tf.where(input_t >= 0, input_t, alpha*(tf.exp(input_t) - 1.0))


def selu(input_t, name="selu"):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # return scale*elu(input_t, alpha)
        return scale*tf.where(input_t >= 0.0, input_t, alpha*tf.nn.elu(input_t))

