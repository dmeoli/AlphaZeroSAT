import random
from os import listdir
from os.path import isfile, join

import numpy as np

from .GymSolver import GymSolver


class Sat:
    """
    This class is a simple wrapper of minisat instance, used in MCTS training as perfect information
    """

    def __init__(self, sat_dir, max_clause=100, max_var=20, mode='random'):
        """
        sat_dir: directory to the sat problems
        max_clause: number of rows for the final state
        max_var: number of columns for the final state
        mode: 'random' => at reset, randomly pick a file from directory
              'iterate' => at reset, iterate each file one by one
              'repeat^n' => at reset, give the same problem n times before iterates to the next one
              'filename' => at reset, repeatedly use the given filename
        """
        print("SAT-v0: at dir {} max_clause {} max_var {} mode {}".format(sat_dir, max_clause, max_var, mode))
        self.sat_dir = sat_dir
        self.sat_files = [join(self.sat_dir, f)
                          for f in listdir(self.sat_dir)
                          if isfile(join(self.sat_dir, f))]
        self.sat_file_num = len(self.sat_files)

        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 2), dtype=bool)
        self.action_space = max_var * 2
        self.mode = mode
        if mode.startswith("repeat^"):
            self.repeat_limit = int(mode.split('^')[1])
        elif mode == "random" or mode == "iterate":
            pass
        else:
            try:
                self.file_index = self.sat_files.index(join(self.sat_dir, self.mode))
            except ValueError:
                assert False, "file {} in not in dir {}".format(mode, sat_dir)
        # this class is stateful, by these fields
        self.repeat_counter = 0
        self.iterate_counter = 0

    def reset(self):
        """
        This function reset the minisat by the rule of mode
        """
        if self.mode == "random":
            pickfile = self.sat_files[random.randint(0, self.sat_file_num - 1)]
            self.repeat_counter += 1
        elif self.mode == "iterate":
            pickfile = self.sat_files[self.iterate_counter]
            self.iterate_counter += 1
            if self.iterate_counter >= self.sat_file_num:
                self.iterate_counter = 0
                self.repeat_counter += 1
                print("WARNING: iteration of all files in dir {} "
                      "is done, will restart iteration".format(self.sat_dir))
        elif self.mode.startswith("repeat^"):
            pickfile = self.sat_files[self.iterate_counter]
            self.repeat_counter += 1
            if self.repeat_counter >= self.repeat_limit:
                self.repeat_counter = 0
                self.iterate_counter += 1
                if self.iterate_counter >= self.sat_file_num:
                    self.iterate_counter = 0
                    print("WARNING: repeated iteration of all files in dir {} "
                          "is done, will restart iteration".format(self.sat_dir))
        else:
            pickfile = self.sat_files[self.file_index]
            self.repeat_counter += 1
        state = np.zeros((self.max_clause, self.max_var, 2), dtype=np.float32)
        self.S = GymSolver(pickfile)
        self.S.init(np.reshape(state, (self.max_clause * self.max_var * 2,)))
        return state

    def resetAt(self, file_no):
        """
        This function reset the minisat by the file_no
        """
        assert (file_no >= 0) and (file_no < self.sat_file_num), "file_no has to be a valid file list index"
        pickfile = self.sat_files[file_no]
        state = np.zeros((self.max_clause, self.max_var, 2), dtype=np.float32)
        self.S = GymSolver(pickfile)
        if self.S.init(np.reshape(state, (self.max_clause * self.max_var * 2,))):
            return state
        else:
            return None

    def step(self, decision):
        """
        This function makes a step based on the parameter input
        return true if the SAT problem is finished.
        """
        self.S.set_decision(decision)
        state = np.zeros((self.max_clause, self.max_var, 2), dtype=np.float32)
        self.S.step(np.reshape(state, (self.max_clause * self.max_var * 2,)))
        return self.S.getDone(), state

    def simulate(self, pi, v):
        """
        This function makes a simulation step, while providing the pi and v
        from neural net for the state from the last simulation
        return state (next state to evaluate), bool (need evaluate state, not empty), bool (need more MCTS steps)
        """
        state = np.zeros((self.max_clause, self.max_var, 2), dtype=np.float32)
        code = self.S.simulate(np.reshape(state, (self.max_clause * self.max_var * 2,)), pi,
                               np.asarray([v], dtype=np.float32))
        if code == 0:
            return state, False, False
        if code == 1:
            return state, True, False
        if code == 2:
            return state, False, True
        if code == 3:
            return state, True, True
        assert False, "return code of simulation {} is not one of the designed value".format(code)

    def get_visit_count(self):
        """
        This function gets the visit count of the root node of MCTS
        """
        count = np.zeros((self.action_space,), dtype=np.float32)
        self.S.get_visit_count(count)
        return count
