import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.regularizers import L2


def model(X, nact, args):
    """
    This model function takes input of n_batch * max_clause * max_var * 1, values are 1 or -1 or 0.
    """
    # h = conv(tf.cast(X, tf.float32), nf=32, rf=8, stride=1, init_scale=np.sqrt(2))
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=8,
                               activation='relu',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(X)
    # TODO: when upgraded to batch run, add layer_norm to conv
    # h = layers.layer_norm(x)
    # if args.layer_norm:
    #     h = tf.keras.layers.LayerNormalization()(h)
    # h2 = conv(h, nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
    h2 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=4,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h)
    # h3 = conv(h2, nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
    h3 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h2)
    # h3_flat = conv_to_fc(h3)
    h3_flat = tf.keras.layers.Flatten()(h3)
    # h4 = fc(h3_flat, 512, init_scale=np.sqrt(2))
    h4 = tf.keras.layers.Dense(units=512,
                               activation='relu',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h3_flat)
    # pi = fc(h4, nact, act=lambda x: x)
    pi = tf.keras.layers.Dense(units=nact,
                               activation='linear',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h4)
    # vf = fc(h4, 1, act=lambda x: tf.tanh(x))
    vf = tf.keras.layers.Dense(units=1,
                               activation='tanh',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h4)

    # get 1 if the positive variable exists in any clauses, otherwise 0
    pos = tf.reduce_max(X, axis=1)
    # get -1 if the negative variables exists in any clauses, otherwise 0
    neg = tf.reduce_min(X, axis=1)
    # get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
    ind = tf.concat([pos, neg], axis=2)
    # this is n_batch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
    ind_flat = tf.reshape(ind, [-1, nact])
    # this is n_batch * nact, with 0 values labeling non_valid actions, 1 for other
    ind_flat_filter = tf.abs(tf.cast(ind_flat, tf.float32))
    # pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32
    pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32

    return pi_fil, vf[:, 0]


def model2(X, nact, args):
    """
    This model function takes input of n_batch * max_clause * max_var * 2 (boolean),
    values are 1 or 0 (can be of type boolean).
    """
    # h = conv(tf.cast(X, tf.float32), nf=32, rf=8, stride=1, init_scale=np.sqrt(2))
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=8,
                               activation='relu',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(X)
    # TODO: when upgraded to batch run, add layer_norm to conv
    # h = layers.layer_norm(x)
    # if args.layer_norm:
    #     h = tf.keras.layers.LayerNormalization()(h)
    # h2 = conv(h, nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
    h2 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=4,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h)
    # h3 = conv(h2, nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
    h3 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h2)
    # h3_flat = conv_to_fc(h3)
    h3_flat = tf.keras.layers.Flatten()(h3)
    # h4 = fc(h3_flat, 512, init_scale=np.sqrt(2))
    h4 = tf.keras.layers.Dense(units=512,
                               activation='relu',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h3_flat)
    # pi = fc(h4, nact, act=lambda x: x)
    pi = tf.keras.layers.Dense(units=nact,
                               activation='linear',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h4)
    # vf = fc(h4, 1, act=lambda x: tf.tanh(x))
    vf = tf.keras.layers.Dense(units=1,
                               activation='tanh',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h4)

    # filter out non-valid actions from pi
    valid = tf.reduce_max(tf.cast(X, tf.float32), axis=1)
    valid_flat = tf.reshape(valid, [-1, nact])  # this is the equivalent of "ind_flat_filter"
    pi_fil = pi + (valid_flat - tf.ones(tf.shape(valid_flat))) * 1e32

    return pi_fil, vf[:, 0]


def model3(X, nact, args):
    """
    This model function takes the same input as model2, but there is some simplification.
    """
    # h = conv(tf.cast(X, tf.float32), nf=32, rf=8, stride=1, init_scale=np.sqrt(2))
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=8,
                               activation='relu',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(X)
    # TODO: when upgraded to batch run, add layer_norm to conv
    # h = layers.layer_norm(x)
    # if args.layer_norm:
    #     h = tf.keras.layers.LayerNormalization()(h)
    # h2 = conv(h, nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
    h2 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=4,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h)
    # h3 = conv(h2, nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
    h3 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                kernel_regularizer=L2(args.l2_coeff),
                                bias_regularizer=L2(args.l2_coeff),
                                kernel_initializer=orthogonal(np.sqrt(2)))(h2)
    # h_pi = conv(h3, nf=2, rf=1, stride=1, init_scale=np.sqrt(2))
    h_pi = tf.keras.layers.Conv2D(filters=2,
                                  kernel_size=1,
                                  activation='relu',
                                  kernel_regularizer=L2(args.l2_coeff),
                                  bias_regularizer=L2(args.l2_coeff),
                                  kernel_initializer=orthogonal(np.sqrt(2)))(h3)
    # h_pi_flat = conv_to_fc(h_pi)
    h_pi_flat = tf.keras.layers.Flatten()(h_pi)
    # pi = fc(h_pi_flat, nact, act=lambda x: x)
    pi = tf.keras.layers.Dense(units=nact,
                               activation='linear',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h_pi_flat)
    # h_v = conv(h3, nf=1, rf=1, stride=1, init_scale=np.sqrt(2))
    h_v = tf.keras.layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 activation='relu',
                                 kernel_regularizer=L2(args.l2_coeff),
                                 bias_regularizer=L2(args.l2_coeff),
                                 kernel_initializer=orthogonal(np.sqrt(2)))(h3)
    # h_v_flat = conv_to_fc(h_v)
    h_v_flat = tf.keras.layers.Flatten()(h_v)
    # h_v_flat256 = fc(h_v_flat, 256, init_scale=np.sqrt(2))
    h_v_flat256 = tf.keras.layers.Dense(units=512,
                                        activation='relu',
                                        kernel_regularizer=L2(args.l2_coeff),
                                        bias_regularizer=L2(args.l2_coeff),
                                        kernel_initializer=orthogonal(np.sqrt(2)))(h_v_flat)
    # vf = fc(h_v_flat256, 1, act=lambda x: tf.tanh(x))
    vf = tf.keras.layers.Dense(units=1,
                               activation='tanh',
                               kernel_regularizer=L2(args.l2_coeff),
                               bias_regularizer=L2(args.l2_coeff),
                               kernel_initializer=orthogonal(np.sqrt(2)))(h_v_flat256)

    # filter out non-valid actions from pi
    valid = tf.reduce_max(tf.cast(X, tf.float32), axis=1)
    valid_flat = tf.reshape(valid, [-1, nact])
    pi_fil = pi + (valid_flat - tf.ones(tf.shape(valid_flat))) * 1e32

    return pi_fil, vf[:, 0]


def load(params, load_path):
    """
    Load function returns a list of tensorflow actions,
    that needs to be run in a session
    """
    load_file = os.path.join(load_path, "saved")
    with open(load_file, "rb") as file_to_load:
        loaded_params = pickle.load(file_to_load)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    return restores


def save(ps, save_path):
    """
    Save function saves the parameters as a side effect
    """
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "saved")
    with open(save_file, "wb") as file_to_save:
        pickle.dump(ps, file_to_save)
