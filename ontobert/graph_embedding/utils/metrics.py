import tensorflow as tf

def distance_func(x: tf.Tensor, y: tf.Tensor, distance_type='L2', squared=True) -> tf.Tensor:
    if distance_type == "L1":
        dist = tf.reduce_sum(tf.abs(x - y), axis=-1)
    elif distance_type == "L2":
        if squared:
            dist = tf.reduce_sum(tf.square(x - y), axis=-1)
        else:
            dist = tf.pow(tf.reduce_sum(tf.square(x - y), axis=-1), 1/2)
    else:
        raise NotImplementedError(
            "Only distance functions supported are 'L1' and 'L2'"
        )
    return dist
