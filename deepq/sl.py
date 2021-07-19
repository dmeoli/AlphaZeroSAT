import argparse
import os
import pickle
import time
from random import shuffle

import numpy as np
import tensorflow.compat.v1 as tf

import GameSAT.common.tf_util as U
from GameSAT import logger
from GameSAT.common.misc_util import (boolean_flag)
from GameSAT.deepq.model import model

tf.disable_v2_behavior()


def parse_args():
    parser = argparse.ArgumentParser("Supervised Learning for deepq neural network in SAT solving")
    # Environment
    parser.add_argument("--env",
                        type=str,
                        default=None,
                        help="name of the game")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="which seed to use")
    # about environment, are we in test mode with a test_path?
    parser.add_argument("--test_path",
                        type=str,
                        default=None,
                        help="if in the test mode, give the directory of SAT problems for testing")
    parser.add_argument("--dump_pair_into",
                        type=str,
                        default=None,
                        help="if in the test mode, give the directory of saving state-action pairs")
    parser.add_argument("--permute_training",
                        default=False,
                        help="if true, the training data will be permuted after every round of usage")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size",
                        type=int,
                        default=int(1e6),
                        help="replay buffer size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps",
                        type=int,
                        default=int(1e5),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq",
                        type=int,
                        default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq",
                        type=int,
                        default=40000,
                        help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq",
                        type=int,
                        default=50,
                        help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq",
                        type=int,
                        default=10000,
                        help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    # Bells and whistles
    boolean_flag(parser, "double-q",
                 default=True,
                 help="whether or not to use double q learning")
    boolean_flag(parser, "dueling",
                 default=False,
                 help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized",
                 default=False,
                 help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise",
                 default=False,
                 help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm",
                 default=False,
                 help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor",
                 default=False,
                 help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha",
                        type=float,
                        default=0.6,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0",
                        type=float,
                        default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps",
                        type=float,
                        default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    parser.add_argument("--L2_coeff",
                        type=float,
                        default=0.0,
                        help="coefficiency for L2 regularization")
    parser.add_argument("--keep_prob",
                        type=float,
                        default=1.0,
                        help="the probability of dropout")
    # Checkpointing
    parser.add_argument("--save-dir",
                        type=str,
                        default=None,
                        help="directory in which training model should be saved.")
    parser.add_argument("--save-freq",
                        type=int,
                        default=1e4,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--model_name_freq",
                        type=int,
                        default=1e4,
                        help="save model with a different name every time this many iterations are completed")
    parser.add_argument("--model-dir",
                        type=str,
                        default=None,
                        help="load model at this directory")
    # parser.add_argument("--load-model",
    # type=int,
    # default=-1,
    # help="if not negative, load model number load-model before supervised training")
    boolean_flag(parser, "load-on-start",
                 default=True,
                 help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


class data_shuffler:
    """
        this object shuffles the data in pickle_dump, and divide them into training and testing
    """

    def __init__(self, input_dumpName, sortM=False, nbatch=-1):
        with open(input_dumpName, "rb") as pull:
            data_dict = pickle.load(pull)
        dataX = np.asarray(data_dict["states"])  # should be numpy array of ndata * max_clause * max_var * 1
        dataY = np.asarray(
            data_dict["actions"])  # should be numpy array of ndata, each content is int of choice (index)
        (self.num_data, self.max_clause, self.max_var, _) = dataX.shape

        # divide data into training and testing (May want to optimize on training size)
        ratio = 0.99
        if nbatch > 0:  # nbatch is given at construction time, smartly adjust num_train to be a multiple of nbatch
            self.num_train = int(self.num_data * ratio) // nbatch * nbatch
        if nbatch == -1:  # no nbatch information given
            self.num_train = int(self.num_data * ratio)

        # shuffle training data
        index = [i for i in range(self.num_train)]
        shuffle(index)
        dataX_train = dataX[index]
        dataY_train = dataY[index]
        dataX_test = dataX[self.num_train:]
        dataY_test = dataY[self.num_train:]

        self.train = {"trainX": dataX_train, "trainY": dataY_train}
        self.test = {"testX": dataX_test, "testY": dataY_test}
        self.lastUsed = 0

    """
        this function return a batch of trainig data
    """

    def next_batch(self, size, permute=False):
        if self.lastUsed + size > self.num_train:
            self.lastUsed = 0
            if permute:
                print(self.train["trainX"][500, 4, :, 0])  # just to prove that permutation happend. Can remove.
                print(self.train["trainY"][500])
                self.permute_train_col()
        x = self.train["trainX"][self.lastUsed: self.lastUsed + size, :, :, :]
        y = self.train["trainY"][self.lastUsed: self.lastUsed + size]
        self.lastUsed += size
        return [x, y]

    """
        this function permutes the states (X input) of the training data
    """

    def permute_train(self):
        toPermute = self.train["trainX"]
        # toPermute is of (self.num_train, max_clause, max_var, 1). We need to permute dim 1, within valid range
        for i in range(self.num_train):
            toPermutePer = toPermute[i, :, :, :]  # toPermutePer is (max_clause, max_var, 1)
            valid_ind = np.any(toPermutePer != 0,
                               axis=1)  # valid_ind should be (max_clause, 1) with 1 for valid, 0 for not
            valid_lim = np.sum(valid_ind)  # valid_lim is a scalar
            np.random.shuffle(toPermutePer[:valid_lim])  # this shuffles toPermutePer in place (only the valid rows)
        # changing toPermutePer also changes toPermute, and also changes self.train["trainX"]. This is IMPORTANT!

    """
        this function permutes the states (X input) and the actions, in columns, of the training data
    """

    def permute_train_col(self):
        toPermuteX = self.train["trainX"]  # (self.num_train, max_clause, max_var, 1), we need to permute dim 2
        toPermuteY = self.train["trainY"]  # (self.num_train,), we need to permute "dim 2" by the same order
        for i in range(self.num_train):
            toPermutePerX = toPermuteX[i, :, :, :]
            toPermutePerY = toPermuteY[i]
            s = np.arange(self.max_var)  # this is the seed of random shuffle TODO: can be a field
            np.random.shuffle(s)

            onehot = np.zeros(2 * self.max_var)  # cast toPermutePerY as onehot encoding
            onehot[toPermutePerY] = 1
            onehot = np.reshape(onehot, (2, self.max_var), order='F')  # reshape onehot as (2, self.max_var)
            onehot = onehot[:, s]  # shuffle onehot by seed
            onehot = np.reshape(onehot, (2 * self.max_var), order='F')  # reshape onehot back to 1 d

            self.train["trainY"][i] = np.argmax(onehot)  # get the new Y label
            self.train["trainX"][i] = toPermutePerX[:, s, :]  # get the new X state


class data_shuffler_v2:
    """
        this object shuffles the data in pickle_dump (actions are saved as q_values),
        and divide them into training and testing
    """

    def __init__(self, input_dumpName, sortM=False, nbatch=-1):
        with open(input_dumpName, "rb") as pull:
            data_dict = pickle.load(pull)

        # data_dict["states"] is a list of scipy.sparse csc_matrix (2-D).
        temp = list(map(lambda x: x.toarray()[:, :, None],
                        data_dict["states"]))  # now the states are list of full numpy matrix (3-D)
        dataX = np.asarray(temp)  # now the states are numpy array of ndata * max_clause * max_var * 1

        # data_dict["actions"] is a list of numpy array (1_D), as the q_value of the corresponding state.
        dataY = np.asarray(
            data_dict["actions"])  # now the actions are numpy array of ndata * (nact, which is max_var * 2)
        (self.num_data, self.max_clause, self.max_var, _) = dataX.shape

        # divide data into training and testing (May want to optimize on training size)
        ratio = 1.0  # too few training data for graph_coloring small data set, so just use everything as training.
        if nbatch > 0:  # nbatch is given at construction time, smartly adjust num_train to be a multiple of nbatch
            self.num_train = int(self.num_data * ratio) // nbatch * nbatch
        if nbatch == -1:  # no nbatch information given
            self.num_train = int(self.num_data * ratio)

        # shuffle training data
        index = [i for i in range(self.num_train)]
        shuffle(index)
        dataX_train = dataX[index]
        dataY_train = dataY[index]
        dataX_test = dataX[self.num_train:]
        dataY_test = dataY[self.num_train:]

        self.train = {"trainX": dataX_train, "trainY": dataY_train}
        self.test = {"testX": dataX_test, "testY": dataY_test}
        self.lastUsed = 0

    """
        this function return a batch of trainig data
    """

    def next_batch(self, size, permute=False):
        if self.lastUsed + size > self.num_train:
            self.lastUsed = 0
            if permute:
                print(self.train["trainX"][100, 4, :, 0])  # just to prove that permutation happend. Can remove.
                print(self.train["trainY"][100, 3])
                self.permute_train_col()
        x = self.train["trainX"][self.lastUsed: self.lastUsed + size, :, :, :]
        y = self.train["trainY"][self.lastUsed: self.lastUsed + size]
        self.lastUsed += size
        return [x, y]

    """
        this function permutes the states (X input) of the training data
    """

    def permute_train(self):
        toPermute = self.train["trainX"]
        # toPermute is of (self.num_train, max_clause, max_var, 1). We need to permute dim 1, within valid range
        for i in range(self.num_train):
            toPermutePer = toPermute[i, :, :, :]  # toPermutePer is (max_clause, max_var, 1)
            valid_ind = np.any(toPermutePer != 0,
                               axis=1)  # valid_ind should be (max_clause, 1) with 1 for valid, 0 for not
            valid_lim = np.sum(valid_ind)  # valid_lim is a scalar
            np.random.shuffle(toPermutePer[:valid_lim])  # this shuffles toPermutePer in place (only the valid rows)
        # changing toPermutePer also changes toPermute, and also changes self.train["trainX"]. This is IMPORTANT!

    """
        this function permutes the states (X input) and the actions, in columns, of the training data
    """

    def permute_train_col(self):
        toPermuteX = self.train["trainX"]  # (self.num_train, max_clause, max_var, 1), we need to permute dim 2
        toPermuteY = self.train["trainY"]  # (self.num_train,), we need to permute "dim 2" by the same order
        for i in range(self.num_train):
            toPermutePerX = toPermuteX[i, :, :, :]
            toPermutePerY = toPermuteY[i]
            s = np.arange(self.max_var)  # this is the seed of random shuffle TODO: can be a field
            np.random.shuffle(s)

            reshape = np.reshape(toPermutePerY, (2, self.max_var), order='F')  # reshape as (2, self.max_var)
            permute = reshape[:, s]  # shuffle by seed
            reshape = np.reshape(permute, (2 * self.max_var), order='F')  # reshape back to 1 d

            self.train["trainY"][i] = reshape  # get the new q_values
            self.train["trainX"][i] = toPermutePerX[:, s, :]  # get the new X state

    """
        this function permutes the state and actions in columns. 
        It is a more compacted form that should replace the function permute_train_col
        WARNNING: for some reason this function DID NOT WORK!!!!
    """

    def permute_train_var(self):
        s = np.arange(self.max_var)
        np.random.shuffle(s)
        # now permute both trainX and trainY by s! # this line is for trainX
        self.train["trainX"] = self.train["trainX"][:, :, s, :]  # permute on the dim of vars
        # these lines are for trainY
        reshape = np.reshape(self.train["trainY"], (self.num_train, 2, self.max_var), order='F')
        permute = reshape[:, :, s]
        self.train["trainY"] = np.reshape(permute, (self.num_train, 2 * self.max_var), order='F')


def maybe_save_model(savedir, model_num):
    """This function checkpoints the model of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(int(model_num))
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, model_dir):
    """Load model if present at the specified path."""
    if (savedir is None) or (model_dir is None):
        return
    U.load_state(os.path.join(savedir, model_dir, "saved"))


def deepq_act_model(observations_ph, num_actions, layer_norm):
    # construct the model
    kwargs = {}
    with tf.variable_scope("deepq", reuse=None):
        # this is nbatch * nact, with values as q values
        q_values = model(observations_ph, num_actions, scope="q_func", layer_norm=layer_norm, **kwargs)

        # filter out non-valid actions
        pos = tf.reduce_max(observations_ph, axis=1)  # get 1 if the postive variable exists in any clauses, otherwise 0
        neg = tf.reduce_min(observations_ph,
                            axis=1)  # get -1 if the negative variables exists in any clauses, otherwise 0
        ind = tf.concat([pos, neg],
                        axis=2)  # get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
        ind_flat = tf.reshape(ind, [-1,
                                    num_actions])  # this is nbatch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
        ind_flat_filter = tf.abs(tf.cast(ind_flat,
                                         tf.float32))  # this is nbatch * nact, with 0 values labeling non_valid actions, 1 for other
        q_min = tf.reduce_min(q_values, axis=1)
        q_values_adjust = q_values - tf.expand_dims(q_min, axis=1)  # make sure the maximal values are positive
        q_values_filter = q_values_adjust * ind_flat_filter  # zero-fy non-valid values, unchange valid values
        q_values_filter_adjust = q_values_filter + tf.expand_dims(q_min, axis=1)  # adjust back the q values.
        return q_values_filter_adjust


def main_test_performance():
    """
        this function test the performance of the supervised learning model in gym environment
    """
    # set up environment
    from GameSAT.deepq.minisat import (gym_sat_Env, gym_sat_sort_Env, gym_sat_permute_Env,
                                       gym_sat_graph_Env, gym_sat_graph2_Env)
    env_type = args.env
    test_path = args.test_path
    if env_type is None or test_path is None:
        print("Error: both env_type and test_path need to be provided, as defaults or command line arguments")
        return
    if env_type == "gym_sat_Env-v0":
        env = gym_sat_Env(test_path=test_path)
    elif env_type == "gym_sat_Env-v1":
        env = gym_sat_sort_Env(test_path=test_path)
    elif env_type == "gym_sat_Env-v2":
        env = gym_sat_permute_Env(test_path=test_path)
    elif env_type == "gym_sat_Env-v3":
        env = gym_sat_graph_Env(test_path=test_path)
    elif env_type == "gym_sat_Env-v4":
        env = gym_sat_graph2_Env(test_path=test_path)
    else:
        print("ERROR: env is not one of the pre-defined mode")
        return
    test_file_num = env.test_file_num
    print("there are {} files to test".format(test_file_num))

    # build and load model
    observations_ph = tf.placeholder(tf.float32, shape=(env.observation_space[None]).shape, name="states")
    q_values_filter = deepq_act_model(observations_ph, env.action_space.n, args.layer_norm)
    action = tf.argmax(q_values_filter, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_dir = args.save_dir
        model_dir = args.model_dir
        if model_dir is not None:
            print("load model at {} before testing".format(model_dir))
            maybe_load_model(save_dir, model_dir)
        # load_model_num = args.load_model
        # if load_model_num >= 0: # there is a model to load, otherwise we will be testing based on a random initialization
        #	print("load model number {} before testing".format(load_model_num))
        #	maybe_load_model(save_dir, load_model_num)

        # test model with environment
        score = 0.0
        reward = 0
        for i in range(test_file_num):
            obs = env.reset()  # this reset is in test mode (because we passed test_path at the construction of env)
            # so the reset will iterate all test files in test_path, instead of randomly picking a file
            while True:
                act = sess.run(action, feed_dict={observations_ph: np.array(obs)[None]})
                new_obs, rew, done, info = env.step(act)
                obs = new_obs
                reward += 1
                if done:
                    score = (score * i + reward) / (i + 1)
                    reward = 0
                    break
        print("the average performance is {}".format(score))


def super_train(filename, num_steps, nbatch, num_report, layer_norm, num_procs, num_model, permute=False):
    """
        this function trains a CNN by supervised learning
        filename: the name of file that pickle dumped the training and testing data ind_flat_filter
        num_steps: total number of steps of supervised training
        nbatch: number of samples used per training step
        num_report: by how many steps should we output the training and testing accuracy
        layer_norm: boolean flag of whether we want to normalize each layer
        num_procs: number of processors used in this training
        num_model: how often should model name change
        permute: if we want the dataX to be permuted after using, make permute True
    """
    print("read files from pickle_dump file %s" % filename)
    data = data_shuffler_v2(filename, nbatch)
    num_var = int(data.max_var)
    num_clause = int(data.max_clause)
    num_actions = num_var * 2

    observations_ph = tf.placeholder(tf.float32, shape=[None, num_clause, num_var, 1], name="states")
    y_ = tf.placeholder(tf.float32, shape=[None, num_actions], name="actions")

    # maybe in supervised learning, we should train q_func, not f_act. i.e. we should train without filtering
    kwargs = {}
    q_values = model(observations_ph, num_actions, scope="deepq/q_func",
                     layer_norm=layer_norm, keep_prob=args.keep_prob, **kwargs)
    q_func_vars = U.scope_vars("deepq/q_func")
    print("found trainable vars of {} many".format(len(q_func_vars)))
    # Train and evaluate
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(y_),
                                                                               logits=q_values))
        # L2 regularization
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in q_func_vars])
        loss = cross_entropy + lossL2 * args.L2_coeff
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(q_values, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=num_procs,
                            inter_op_parallelism_threads=num_procs)
    config.gpu_options.allow_growth = True

    # run training in session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # may load model if given
        save_dir = args.save_dir
        if args.model_dir is not None:
            print("load model at {} before supervised learning".format(args.model_dir))
            maybe_load_model(save_dir, model_dir)

        # if load_model_num >= 0: # there is a model to load before training
        #	print("load model number {} before supervised learning".format(load_model_num))
        #	maybe_load_model(save_dir, load_model_num)

        # supervised training cycle
        for i in range(num_steps + 1):
            batch = data.next_batch(nbatch,
                                    permute=permute)  # training data will be permuted after using one round of them!
            feed_dict = {observations_ph: batch[0], y_: batch[1]}
            sess.run(train_step, feed_dict)
            if i > 0 and i % num_report == 0:  # report accuracy
                train_accuracy = sess.run(accuracy, feed_dict)
                print('step %d, training accuracy %g' % (i, train_accuracy))
                feed_dict_test = {observations_ph: data.test["testX"], y_: data.test["testY"]}
                test_accuracy = sess.run(accuracy, feed_dict_test)
                print('step %d, testing accuracy %g' % (i, test_accuracy))
                # save model
                maybe_save_model(save_dir, i // num_model)


if __name__ == '__main__':
    args = parse_args()

    # if test_path is given, we should be in the test mode
    if args.test_path is not None:
        main_test_performance()
        exit(0)

    # otherwise, enter the training mode
    filename = args.dump_pair_into
    num_steps = args.num_steps
    nbatch = args.batch_size
    num_report = args.save_freq
    layer_norm = args.layer_norm
    num_procs = 16
    num_model = args.model_name_freq
    # load_model_num = args.load_model
    permute_training = args.permute_training
    if permute_training:
        print("PERMUTE_TRAINING!")
    super_train(filename=filename, num_steps=num_steps, nbatch=nbatch, num_report=num_report,
                layer_norm=layer_norm, num_procs=num_procs, num_model=num_model, permute=permute_training)
