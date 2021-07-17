import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf

from utils import conv, fc, conv_to_fc

tf.disable_v2_behavior()


def model(X, nact, scope, reuse=False, layer_norm=False):
    """
    This model function takes input of nbatch * max_clause * max_var * 1, values are 1 or -1 or 0
    """
    with tf.variable_scope(scope, reuse=reuse):
        h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=8, stride=1,
                 init_scale=np.sqrt(2))  # TODO: when upgraded to batch run, add layer_norm to conv
        # x = layers.layer_norm(x, scale=True, center=True)
        h2 = conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
        h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
        h3 = conv_to_fc(h3)
        h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
        pi = fc(h4, 'pi', nact, act=lambda x: x)
        vf = fc(h4, 'v', 1, act=lambda x: tf.tanh(x))
        #  get 1 if the positive variable exists in any clauses, otherwise 0
        pos = tf.reduce_max(X, axis=1)
        #  get -1 if the negative variables exists in any clauses, otherwise 0
        neg = tf.reduce_min(X, axis=1)
        #  get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
        ind = tf.concat([pos, neg], axis=2)
        #  this is nbatch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
        ind_flat = tf.reshape(ind, [-1, nact])
        #  this is nbatch * nact, with 0 values labeling non_valid actions, 1 for other
        ind_flat_filter = tf.abs(tf.cast(ind_flat, tf.float32))
        # pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32
        pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32
    return pi_fil, vf[:, 0]


def model2(X, nact, scope, reuse=False, layer_norm=False):
    """
    This model function takes input of n_batch * max_clause * max_var * 2,
    values are 1 or 0 (can be of type boolean)
    """
    # X should be n_batch * ncol * nrow * 2 (boolean)
    with tf.variable_scope(scope, reuse=reuse):
        h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=8, stride=1, init_scale=np.sqrt(2))
        # x = layers.layer_norm(x, scale = True, center = True)
        h2 = conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
        h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
        h3 = conv_to_fc(h3)
        h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
        pi = fc(h4, 'pi', nact, act=lambda x: x)
        vf = fc(h4, 'v', 1, act=lambda x: tf.tanh(x))

        # filter out non-valid actions from pi
        valid = tf.reduce_max(tf.cast(X, tf.float32), axis=1)
        valid_flat = tf.reshape(valid, [-1, nact])  # this is the equivalent of "ind_flat_filter"
        pi_fil = pi + (valid_flat - tf.ones(tf.shape(valid_flat))) * 1e32
    return pi_fil, vf[:, 0]


def model3(X, nact, scope, reuse=False, layer_norm=False):
    """
    This model function takes the same input as model2,
    but there is some simplification
    """
    with tf.variable_scope(scope, reuse=reuse):
        h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=8, stride=1, init_scale=np.sqrt(2))
        h2 = conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2))
        h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
        # for pi
        h_pi = conv(h3, 'c_pi', nf=2, rf=1, stride=1, init_scale=np.sqrt(2))
        h_pi_flat = conv_to_fc(h_pi)
        pi = fc(h_pi_flat, 'pi', nact, act=lambda x: x)
        # for v
        h_v = conv(h3, 'c_v1', nf=1, rf=1, stride=1, init_scale=np.sqrt(2))
        h_v_flat = conv_to_fc(h_v)
        h_v_flat256 = fc(h_v_flat, 'c_v2', 256, init_scale=np.sqrt(2))
        vf = fc(h_v_flat256, 'v', 1, act=lambda x: tf.tanh(x))

        # filter out non-valid actions from pi
        valid = tf.reduce_max(tf.cast(X, tf.float32), axis=1)
        valid_flat = tf.reshape(valid, [-1, nact])
        pi_fil = pi + (valid_flat - tf.ones(tf.shape(valid_flat))) * 1e32
    return pi_fil, vf[:, 0]


def load(params, load_path):
    """
    Load function returns a list of tensorflow actions,
    that needs to be ran in a session
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
