import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def sample(logits):
    noise = tf.random.uniform(tf.shape(logits))
    return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()
