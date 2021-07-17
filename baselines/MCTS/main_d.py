import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf

from mct_d import MCT
from models import load, save
from models import model3 as model
from sl_buffer_d import slBuffer_allFile
from utils import find_trainable_variables

tf.disable_v2_behavior()


class Status:

    def __init__(self):
        """
            create a Status object that keeps track of the status of MCTS RL algorithm
            NOTE: any changes to this Status object is updated in hardware by pickle dump
            NOTE: a new instance of Status can be created by reading from a pickle dump
        """
        self.best_model = -1  # the index of the best model (if -1, means no model available)
        self.n_start = 0  # the index of the training file to start with for self_play
        self.ev_hist = []  # the list of evaluation history for all models
        self.length_hist = 0  # the length of models generated (or to evaluate)
        # NOTE: if length_hist is larger than len(ev_hist), it means some number of models have not been evaluated
        self.status_file = None  # the file name of this object
        self.args = None  # the arguments list about this status file

    def retrete(self):
        """
        This function decrease length_hist and ev_hist to the best model
        """
        self.length_hist = self.best_model + 1
        del self.ev_hist[self.length_hist:]
        self.write_to_disc(update_hist=True)

    def reset_ev(self, resetTo):
        """
        This function reset ev_hist to resetTo
        """
        self.ev_hist = self.ev_hist[:resetTo]
        self.write_to_disc(update_hist=True)

    def reset_n_start(self, n_start):
        """
        This function reset n_start value
        """
        self.n_start = n_start
        self.write_to_disc()

    def reset_best_model(self, best_model):
        """
        This function reset best_model
        """
        self.best_model = best_model
        self.write_to_disc()

    def reset_length_hist(self, length_hist):
        """
            this function reset length_hist (only used if the model is not saved completely)
        """
        self.length_hist = length_hist
        self.write_to_disc()

    def reset_args(self, args):
        """
        This function reset the args field
        """
        self.args = args
        self.write_to_disc()

    def set_same_length_hist(self, other):
        """
        This function set the same length_hist has the "other"
        """
        self.length_hist = other.length_hist
        self.write_to_disc()

    def self_check(self):
        if self.best_model == -1:
            assert len(self.ev_hist) == 0, "when self.best_model is -1, self.ev_hist should be empty"
        assert self.best_model <= len(self.ev_hist), "self.best_model should be less than or equal to len(self.ev_hist)"
        assert len(
            self.ev_hist) <= self.length_hist, "self.ev_hist should have length less than or equal to self.length_hist"

    def write_to_disc(self, update_hist=False):
        """
        This method write the model to disc at self.status_file (every change should be updated)
        TODO : optimize so that the vast content of ev_hist is not unnecessarily updated
        """
        with open(self.status_file, "wb") as f:
            pickle.dump((self.best_model, self.n_start, self.length_hist, self.status_file, self.args), f)
        if update_hist:
            with open(self.status_file + ".hist", "wb") as d:
                pickle.dump(self.ev_hist, d)

    def start_with(self, status_file):
        """
        This method fill the fields of Status object with information stored in a status pickle dump
        """
        with open(status_file, "rb") as f:
            self.best_model, self.n_start, self.length_hist, self.status_file, self.args = pickle.load(f)
        with open(status_file + ".hist", "rb") as d:
            self.ev_hist = pickle.load(d)
        self.self_check()

    def init_with(self, best_model, n_start, ev_hist, length_hist, status_file, args=None):
        """
        This method initialize the fields of Status object with given parameters
        """
        self.best_model = best_model
        self.n_start = n_start
        self.ev_hist = ev_hist
        self.length_hist = length_hist
        self.status_file = status_file
        self.args = args
        self.self_check()
        self.write_to_disc(update_hist=True)

    def get_model_dir(self):
        """
        This method returns the dir of the best model as indicated by self.best_model (or None of best_model == -1)
        """
        if self.best_model == -1:
            return None
        return "model-" + str(self.best_model)

    def get_nbatch_index(self, nbatch, ntotal):
        """
        This method update n_start field, with the help of nbatch and ntotal, and return the correct indexes of batch
        """
        indexes = np.asarray(range(self.n_start, self.n_start + nbatch)) % ntotal
        self.n_start += nbatch
        self.n_start %= ntotal
        self.write_to_disc()
        return indexes

    def get_sl_starter(self):
        """
        This method returns the starting model of the supervised learning
        """
        assert self.length_hist > 0, "at supervised training stage, there should exist at least one model"
        # NOTE: if recent two models are worse, instead of better, use the "best_model" as the starting model for sl
        if self.length_hist - 1 <= self.best_model + 2:
            return "model-" + str(self.length_hist - 1)
        else:
            return "model-" + str(max(self.best_model, 0))

    def generate_new_model(self):
        """
        This function is used by sl_train (or the initial phase of self_play) to put more models in hard drive
        returns the new model dir name for this new model
        """
        self.length_hist += 1
        if self.best_model == -1:
            assert self.length_hist == 1, "this should be the first model"
            self.best_model = 0
        self.write_to_disc()
        return "model-" + str(self.length_hist - 1)

    def which_model_to_evaluate(self):
        """
        This function returns the dir name of the model to be evaluated
        returns None if no such model dir exists
        """
        index = len(self.ev_hist)
        if index < self.length_hist:
            return "model-" + str(index)
        else:
            return None

    def write_performance(self, performance):
        """
        This function report the performance of the last evaluated model,
        then compare with the best model and possibly update it
        """
        self.ev_hist.append(performance)
        if self.best_model == -1:
            assert len(self.ev_hist) == 1, "this must be the first model evaluated"
            self.best_model = 0
        else:
            if self.better_than(performance, self.ev_hist[self.best_model]):
                self.best_model = len(self.ev_hist) - 1
        self.write_to_disc(update_hist=True)

    def better_than(self, per1, per2):
        if (per1 <= per2).sum() >= per1.shape[0] * 0.95 and np.mean(per1) < np.mean(per2) * 0.99:
            return True
        if (per1 <= per2).sum() >= per1.shape[0] * 0.65 and np.mean(per1) < np.mean(per2) * 0.95:
            return True
        if (per1 <= per2).sum() >= per1.shape[0] * 0.50 and np.mean(per1) < np.mean(per2) * 0.90:
            return True
        return False

    def show_itself(self):
        """
        This function print the information in this object
        """
        print("best_model is {}".format(self.best_model))
        print("n_start is {}".format(self.n_start))
        print("ev_hist has length {}".format(len(self.ev_hist)))
        print("length_hist is {}".format(self.length_hist))
        print("status_file is {}".format(self.status_file))
        if self.args is not None:
            print("args is {}__{}__{}".format(self.args.save_dir, self.args.train_path, self.args.test_path))

    def print_all_models_performance(self):
        """
        This function print the performance of all models (all average values in ev_hist)
        """
        for i in range(len(self.ev_hist)):
            print(np.mean(self.ev_hist[i]), end=", ")
        print("\n")


def build_model(args, scope):
    """
    This function builds the model that is used by all three functions below
    """
    nh = args.max_clause
    nw = args.max_var
    nc = 2
    nact = nc * nw
    ob_shape = (None, nh, nw, nc * args.nstack)
    X = tf.placeholder(tf.float32, ob_shape)
    Y = tf.placeholder(tf.float32, (None, nact))
    Z = tf.placeholder(tf.float32, None)

    p, v = model(X, nact, scope)
    params = find_trainable_variables(scope)
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=p))
        value_loss = tf.losses.mean_squared_error(labels=Z, predictions=v)
        lossL2 = tf.add_n([tf.nn.l2_loss(vv) for vv in params])
        loss = cross_entropy + value_loss + args.l2_coeff * lossL2

    return X, Y, Z, p, v, params, loss


def self_play(args, built_model, status_track):
    """
    c_act (exploration parameter of MCTS) and num_mcts (the full size of MCTS tree)
    are determined in minisat.core.Const.h
    NOTE: max_clause, max_var and nc are define in both here (in args for model)
    and in minisat.core.Const.h (for writing states).
    They need to BE the same.
    nbatch is the degree of parallel for neural net
    nstack is the number of history for a state
    """
    # take out the parts that self_play need from the model
    X, _, _, p, v, params, _ = built_model

    # within a tensorflow session, run MCT objects with model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = status_track.get_model_dir()
        if (args.save_dir is not None) and (model_dir is not None):
            sess.run(load(params, os.path.join(args.save_dir, model_dir)))
            print("loaded model {} at dir {} for selfplay".format(args.save_dir, model_dir))
        else:
            # this is the initial random parameter! let's save it in hard drive!
            ps = sess.run(params)
            model_dir = status_track.generate_new_model()
            save(ps, os.path.join(args.save_dir, model_dir))

        # initialize a list of MCT and run self_play
        MCTList = []
        for i in status_track.get_nbatch_index(args.nbatch, args.n_train_files):
            MCTList.append(MCT(args.train_path, i, args.max_clause, args.max_var, args.nrepeat,
                               tau=lambda x: 1.0 if x <= 30 else 0.0001, resign=400))
        pi_matrix = np.zeros((args.nbatch, 2 * args.max_var), dtype=np.float32)
        v_array = np.zeros((args.nbatch,), dtype=np.float32)
        needMore = np.ones((args.nbatch,), dtype=np.bool)
        while True:
            states = []
            pi_v_index = 0
            for i in range(args.nbatch):
                if needMore[i]:
                    temp = MCTList[i].get_state(pi_matrix[pi_v_index], v_array[pi_v_index])
                    pi_v_index += 1
                    if temp is None:
                        needMore[i] = False
                    else:
                        states.append(temp)
            if not np.any(needMore):
                break
            pi_matrix, v_array = sess.run([p, v], feed_dict={X: np.asarray(states, dtype=np.float32)})

        print("loop finished and save Pi graph to slBuffer")
        # bring sl_buffer to memory
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_trace = os.path.join(args.dump_dir, args.dump_file)
        if os.path.isfile(dump_trace):
            with open(dump_trace, 'rb') as sl_file:
                sl_Buffer = pickle.load(sl_file)
        else:
            sl_Buffer = slBuffer_allFile(args.sl_buffer_size, args.train_path, args.n_train_files)
        # write in sl_buffer
        for i in range(args.nbatch):
            MCTList[i].write_data_to_buffer(sl_Buffer)
        # write sl_buffer back to disk
        with open(dump_trace, 'wb') as sl_file:
            pickle.dump(sl_Buffer, sl_file, -1)


def super_train(args, built_model, status_track):
    """
    This function does supervised training
    """
    # take out the parts that self_play needs from the model
    X, Y, Z, _, _, params, loss = built_model
    with tf.name_scope("train"):
        if args.which_cycle == 0:
            lr = 1e-2
        else:
            lr = 1e-3
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = status_track.get_sl_starter()
        assert (args.save_dir is not None) and (
                model_dir is not None), "save_dir and model_dir needs to be specified for super_training"
        sess.run(load(params, os.path.join(args.save_dir, model_dir)))
        print("loaded model {} at dir {} as super_training starter".format(args.save_dir, model_dir))

        # data for supervised training
        dump_trace = os.path.join(args.dump_dir, args.dump_file)
        with open(dump_trace, 'rb') as sl_file:
            sl_Buffer = pickle.load(sl_file)

        # supervised training cycle
        for i in range(args.sl_num_steps + 1):
            batch = sl_Buffer.sample(args.sl_nbatch)
            feed_dict = {X: batch[0], Y: batch[1], Z: batch[2]}
            sess.run(train_step, feed_dict)
            if i > 0 and i % args.sl_ncheckpoint == 0:
                new_model_dir = status_track.generate_new_model()
                print("checkpoint model {}".format(new_model_dir))
                ps = sess.run(params)
                save(ps, os.path.join(args.save_dir, new_model_dir))


def model_ev(args, built_model, status_track, ev_testing=False):
    """
    This function evaluates all unevaluated model, as indicated in the status_track object
    """
    # there may be a few number of unevaluated models, and this function evaluate them all
    model_dir = status_track.which_model_to_evaluate()
    if model_dir is None:
        return

    # add this layer of indirection so that the function is fit for both evaluating training files and testing files
    if ev_testing:
        sat_path = args.test_path
        sat_num = args.n_test_files
    else:
        sat_path = args.train_path
        sat_num = args.n_train_files

    # take out the parts that self_play needs from the model
    X, _, _, p, v, params, _ = built_model

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # may run this multiple times because there maybe multiple models to evaluate
        while model_dir is not None:
            sess.run(load(params, os.path.join(args.save_dir, model_dir)))
            print("loaded model {} at dir {} for evaluation".format(args.save_dir, model_dir))

            MCTList = []
            for i in range(args.nbatch):
                # tau is small for testing, and evaluation only solve a problem once.
                MCTList.append(MCT(sat_path, i, args.max_clause, args.max_var, 1,
                                   tau=lambda x: 0.001, resign=400))
            pi_matrix = np.zeros((args.nbatch, 2 * args.max_var), dtype=np.float32)
            v_array = np.zeros((args.nbatch,), dtype=np.float32)
            needMore = np.ones((args.nbatch,), dtype=np.bool)
            next_file_index = args.nbatch
            assert (next_file_index <= sat_num), "this is a convention"
            all_files_done = next_file_index == sat_num
            performance = np.zeros(sat_num)
            while True:
                states = []
                pi_v_index = 0
                for i in range(args.nbatch):
                    if needMore[i]:
                        temp = MCTList[i].get_state(pi_matrix[pi_v_index], v_array[pi_v_index])
                        pi_v_index += 1
                        while temp is None:
                            idx, rep, scr = MCTList[i].report_performance()
                            performance[idx] = scr / rep
                            if all_files_done:
                                break
                            MCTList[i] = MCT(sat_path, next_file_index, args.max_clause, args.max_var, 1,
                                             tau=lambda x: 0.001, resign=400)
                            next_file_index += 1
                            if next_file_index >= sat_num:
                                all_files_done = True
                            temp = MCTList[i].get_state(pi_matrix[pi_v_index - 1], v_array[
                                pi_v_index - 1])  # the pi and v are not used (for new MCT object)
                        if temp is None:
                            needMore[i] = False
                        else:
                            states.append(temp)
                if not np.any(needMore):
                    break
                pi_matrix, v_array = sess.run([p, v], feed_dict={X: np.asarray(states, dtype=np.float32)})

            # write performance to the status_track
            print(performance)
            status_track.write_performance(performance)
            model_dir = status_track.which_model_to_evaluate()


def ev_ss(args, built_model, status_track, file_no):
    model_dir = status_track.get_model_dir()
    sat_path = args.train_path
    sat_num = args.n_train_files

    if model_dir is None:
        return
    X, _, _, p, v, params, _ = built_model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(load(params, os.path.join(args.save_dir, model_dir)))
        print("load model {} at dir {}".format(args.save_dir, model_dir))
        MCT58 = MCT(sat_path, file_no, args.max_clause, args.max_var, 1,
                    tau=lambda x: 0.9, resign=80)
        pi_matrix = np.zeros((1, 2 * args.max_var,), dtype=np.float32)
        v_array = np.zeros([1, ], dtype=np.float32)
        while True:
            temp = MCT58.get_state(pi_matrix[0], v_array[0])
            if temp is None: break
            states = []
            states.append(temp)
            pi_matrix, v_array = sess.run([p, v], feed_dict={X: np.asarray(states, dtype=np.float32)})
        idx, rep, scr = MCT58.report_performance()
        print("performance is {}".format(scr / rep))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir',
                        type=str,
                        help='where is the model saved',
                        default='runs/uf50-218')
    parser.add_argument('--best_model',
                        type=int,
                        help='the index of the best model (-1 for unknown)',
                        default=-1)
    parser.add_argument('--status_file',
                        type=str,
                        help='which file keeps a record of the status',
                        default='status.pkl')
    parser.add_argument('--result_file',
                        type=str,
                        help='this file keeps the performance of models on testing files',
                        default='result.pkl')
    parser.add_argument('--dump_dir',
                        type=str,
                        help='where to save (state, Pi, num_step) for SL',
                        default='parameters/')
    parser.add_argument('--dump_file',
                        type=str,
                        help='what is the filename to save (state, Pi, num_step) for SL',
                        default='sl.pkl')
    parser.add_argument('--train_path',
                        type=str,
                        help='where are training files',
                        default='../data/uniform-random-3-sat/train/uf50-218')
    parser.add_argument('--test_path',
                        type=str,
                        help='where are test files',
                        default='../data/uniform-random-3-sat/test/uf50-218')
    parser.add_argument('--max_clause',
                        type=int,
                        help='what is the max_clause',
                        default=120)
    parser.add_argument('--max_var',
                        type=int,
                        help='what is the max_var',
                        default=20)
    parser.add_argument('--sl_buffer_size',
                        type=int,
                        help='max size of sl buffer',
                        default=1000000)
    parser.add_argument('--nbatch',
                        type=int,
                        help='what is the batch size to use',
                        default=32)
    parser.add_argument('--nstack',
                        type=int,
                        help='how many layers of states to use',
                        default=1)
    parser.add_argument('--nrepeat',
                        type=int,
                        help='how many times to repeat a SAT problem',
                        default=100)
    parser.add_argument('--n_start',
                        type=int,
                        help='which file index to start with (for running)',
                        default=0)
    parser.add_argument('--n_train_files',
                        type=int,
                        help='total number of training files',
                        default=0)  # calculated later
    parser.add_argument('--n_test_files',
                        type=int,
                        help='total number of testing files',
                        default=0)  # calculated later
    parser.add_argument('--l2_coeff',
                        type=float,
                        help='the coefficient for l2 regularization',
                        default=0.0001)
    parser.add_argument('--sl_num_steps',
                        type=int,
                        help='how many times to do supervised training',
                        default=64000)
    parser.add_argument('--sl_nbatch',
                        type=int,
                        help='what is the batch size for supervised training',
                        default=32)
    parser.add_argument('--sl_ncheckpoint',
                        type=int,
                        help='how often to checkpoint a supervised trained model',
                        default=32000)
    parser.add_argument('--n_cycles',
                        type=int,
                        help='how many cycles of self_play -> super_train -> model_ev do we want to run',
                        default=2)
    parser.add_argument('--show_only',
                        type=str,
                        help='if only show the result',
                        default='No')
    parser.add_argument('--which_cycle',
                        type=int,
                        help='which cycle are we in now',
                        default=0)

    args = parser.parse_args()
    args.n_train_files = len([f for f in os.listdir(args.train_path) if
                              os.path.isfile(os.path.join(args.train_path, f))])  # total number of training files
    args.n_test_files = len([f for f in os.listdir(args.test_path) if
                             os.path.isfile(os.path.join(args.test_path, f))])  # total number of testing files
    args.dump_dir = args.save_dir  # all files related to this project are saved in save_dir, so dump_dir is useless
    os.makedirs(args.save_dir, exist_ok=True)

    # start the status_track for these operations
    status_track = Status()
    if os.path.isfile(os.path.join(args.save_dir, args.status_file)):
        status_track.start_with(os.path.join(args.save_dir, args.status_file))
    else:  # otherwise the initial values in Status object fits with the default values here;
        status_track.init_with(args.best_model, args.n_start, [], 0, os.path.join(args.save_dir, args.status_file),
                               args)
    status_track.show_itself()

    # following code evaluates the performance of models on testing files
    result_track = Status()
    if os.path.isfile(os.path.join(args.save_dir, args.result_file)):
        result_track.start_with(os.path.join(args.save_dir, args.result_file))
    else:  # initialize values in Status object with the "total model number" --> "length_hist field" of status_track
        result_track.init_with(-1, 0, [], 0, os.path.join(args.save_dir, args.result_file))

    if args.show_only == "Yes":
        status_track.show_itself()
        status_track.print_all_models_performance()
        result_track.show_itself()
        result_track.print_all_models_performance()

    # build the model for all three functions
    built_model = build_model(args, scope='mcts')

    # run a specific file that has bugs
    #    ev_ss(args, built_model, status_track, 0)
    #    return
    #    model_ev(args, built_model, status_track)
    #    status_track.show_itself()
    #    return
    #    result_track.set_same_length_hist(status_track)
    #    model_ev(args, built_model, result_track, ev_testing = True)
    #    result_track.show_itself()
    #    return

    # run args.n_cycles number of iteration (self_play -> super_train -> model_ev)
    for i in range(args.n_cycles):
        args.which_cycle = i
        self_play(args, built_model, status_track)
        status_track.show_itself()
        super_train(args, built_model, status_track)
        status_track.show_itself()
        model_ev(args, built_model, status_track)
        result_track.set_same_length_hist(status_track)
        model_ev(args, built_model, result_track, ev_testing=True)
        status_track.show_itself()

    # print the performance of all models we have so far:
    status_track.print_all_models_performance()
    result_track.print_all_models_performance()
