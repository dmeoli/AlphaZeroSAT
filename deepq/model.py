import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def layer_norm_fn(x, relu=True):
    x = tf.layers.layer_norm(x, scale=True, center=True)
    if relu:
        x = tf.nn.relu(x)
    return x


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model(img_in, num_actions, scope, reuse=False, layer_norm=False, keep_prob=1.0):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # add max_pool layers
            out = tf.layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            # out = max_pool_2x2(out)
            out = tf.layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            # out = max_pool_2x2(out)
            out = tf.layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            # out = max_pool_2x2(out)
        conv_out = tf.layers.flatten(out)

        with tf.variable_scope("action_value"):
            # change num_outputs to 1024 (was 512)
            value_out = tf.layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                value_out = layer_norm_fn(value_out, relu=True)
            else:
                value_out = tf.nn.relu(value_out)
            # add dropout here
            print("keep_prob is {}".format(keep_prob))
            value_out = tf.nn.dropout(value_out, keep_prob)
            value_out = tf.layers.fully_connected(value_out, num_outputs=num_actions, activation_fn=None)
        return value_out


def dueling_model(img_in, num_actions, scope, reuse=False, layer_norm=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = tf.layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = tf.layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = tf.layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = tf.layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = tf.layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                state_hidden = layer_norm_fn(state_hidden, relu=True)
            else:
                state_hidden = tf.nn.relu(state_hidden)
            state_score = tf.layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = tf.layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                actions_hidden = layer_norm_fn(actions_hidden, relu=True)
            else:
                actions_hidden = tf.nn.relu(actions_hidden)
            action_scores = tf.layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return state_score + action_scores
