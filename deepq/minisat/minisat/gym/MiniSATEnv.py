import _thread
import random
from os import listdir
from os.path import isfile, join

import gym
import numpy as np
from gym import spaces

from baseline.minisat.minisat.gym.GymSolver import GymSolver


class gym_sat_Env(gym.Env):
    """
    This class is a gym environment for Reinforcement Learning algorithms
    It doesn't do any preprocessing (sorting matrix) or postprocessing (permute the training file)
    max_clause: the number of rows in state representation
    max_var: the number of columns in state representation
    """

    def __init__(self, max_clause=100, max_var=20, test_path=None):
        if test_path == None:
            self.test_mode = False
            # BE AWARE, the directory of the training files is statically determined in __init__function
            self.test_path = "uf20-91_train_v0"
            print("SAT-v0: We are in the training mode of path {}".format(self.test_path))
        else:
            self.test_mode = True
            self.test_path = test_path
            print("SAT-v0: We are in the test mode of path {}".format(self.test_path))
        # Get all test files
        self.test_files = [join(self.test_path, f)
                           for f in listdir(self.test_path)
                           if isfile(join(self.test_path, f))]
        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 1))
        self.action_space = spaces.Discrete(2 * self.max_var)
        self.score = 0
        self.exp_av_score = 15  # some randomly initialized initial average value

    def parse_state(self):
        """
        This function parse the state into sparse matrix with -1 or 1 values
        Can handle the case when state is empty and the SAT is either broken or solved already
        """
        curr_state = np.zeros((self.max_clause, self.max_var, 1), dtype=np.int8)
        clause_counter = 0  # this tracks the current row-to-write (which is permutable!)
        actionSet = set()  # this set tracks all allowed actions for this state
        # if S is already Done, should return here.
        if self.S.getDone():
            return curr_state, clause_counter, True, actionSet
        # S is not yet Done, parse and return real state
        for line in self.S.getState().split('\n'):
            if line.startswith("p cnf"):  # this is the header of a cnf problem # p cnf 20 90
                header = line.split(" ")
                num_var = int(header[2])
                num_clause = int(header[3])
                assert (num_var <= self.max_var)
            # remove this assert (might be wrong if we learnt too many clauses and restarted)
            # assert (num_clause <= self.max_clause)
            elif line.startswith("c"):
                continue
            else:  # clause data line # -11 -17 20 0
                literals = line.split(" ")
                n = len(literals)
                for j in range(n - 1):
                    number = int(literals[j])
                    value = 1 if number > 0 else -1
                    curr_state[clause_counter, abs(number) - 1] = value
                    actionSet.add(number)
                clause_counter += 1
                if clause_counter >= self.max_clause:  # add a safe guard for overflow of number of clauses
                    break
        return curr_state, clause_counter, False, actionSet

    def random_pick_satProb(self):
        """
        This function randomly pick a file from the training file set
        """
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def report_to_agent(self):
        """
        This function reports to the agent about the environment
        """
        return self.curr_state, self.S.getReward(), self.isSolved, {}

    def reset(self):
        """
        This function reset the environment and return the initial state
        """
        if self.test_mode:  # in test mode, we print the actual score of each SAT problem in test files
            print(self.score, end=".", flush=True)
        else:  # in training mode, we print an exponential average of scores of randomly picked files
            self.exp_av_score = self.exp_av_score * 0.98 + self.score * 0.02
            print(round(self.exp_av_score), end=".", flush=True)
        self.score = 0
        filename = self.random_pick_satProb()
        self.S = GymSolver(filename)
        self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
        return self.curr_state

    def step(self, decision):
        """
        This function make a step based on parameter input
        """
        self.score += 1
        if decision < 0:  # this is to say that let minisat pick the decision
            decision = 32767
        elif decision % 2 == 0:  # this is to say that pick decision and assign positive value
            decision = int(decision / 2 + 1)
        else:  # this is to say that pick decision and assign negative value
            decision = 0 - int(decision / 2 + 1)
        if (decision in self.actionSet) or (decision == 32767):
            self.S.step(decision)
            self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
            return self.report_to_agent()
        else:
            return self.report_to_agent()

    def render(self, mode='human', close=False):
        """
        This function renders the sat problem
        """
        pass


class gym_sat_sort_Env(gym.Env):
    """
    This class is a gym environment for Reinforcement Learning algorithms
    It always sort the rows of matrix that represent each state!
    max_clause: the number of rows in state representation
    max_var: the number of columns in state representation
    """

    def __init__(self, max_clause=100, max_var=20, test_path=None):
        if test_path == None:
            self.test_mode = False
            # BE AWARE, the directory of training files is statically defined in __init__function!
            self.test_path = "uf20-91_train_v1"
            print("SAT-v1 (sort): We are in the training mode of path {}".format(self.test_path))
        else:
            self.test_mode = True
            self.test_path = test_path
            print("SAT-v1 (sort): We are in the test mode of path {}".format(self.test_path))
        # Get all test files
        self.test_files = [join(self.test_path, f)
                           for f in listdir(self.test_path)
                           if isfile(join(self.test_path, f))]
        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 1))
        self.action_space = spaces.Discrete(2 * self.max_var)
        self.score = 0
        self.exp_av_score = 15  # some randomly initialized initial average value

    def parse_state(self):
        """
        This function parse the state into sparse matrix with -1 or 1 values
        Can handle the case when state is empty and the SAT is either broken or solved already
        """
        curr_state = np.zeros((self.max_clause, self.max_var, 1), dtype=np.int8)
        clause_counter = 0  # this tracks the current row-to-write (which is permutable!)
        actionSet = set()  # this set tracks all allowed actions for this state
        # if S is already Done, should return here.
        if self.S.getDone():
            return curr_state, clause_counter, True, actionSet
        # S is not yet Done, parse and return real state
        for line in self.S.getState().split('\n'):
            if line.startswith("p cnf"):  # this is the header of a cnf problem # p cnf 20 90
                header = line.split(" ")
                num_var = int(header[2])
                num_clause = int(header[3])
                assert (num_var <= self.max_var)
            # remove this assert (might be wrong if we learnt too many clauses and restarted)
            # assert (num_clause <= self.max_clause)
            elif line.startswith("c"):
                continue
            else:  # clause data line # -11 -17 20 0
                literals = line.split(" ")
                n = len(literals)
                for j in range(n - 1):
                    number = int(literals[j])
                    value = 1 if number > 0 else -1
                    curr_state[clause_counter, abs(number) - 1] = value
                    actionSet.add(number)
                clause_counter += 1
                if clause_counter >= self.max_clause:  # add a safe guard for overflow of number of clauses
                    break
        curr_state = self.sortMatrix(
            curr_state)  # this is to sort the state representation by rows (every time we parse the state)
        return curr_state, clause_counter, False, actionSet

    def sortMatrix(self, M):
        """
        This function return the sorted Matrix
        """
        [row, col, _] = M.shape
        Morder = np.zeros(row)
        for i in range(col):
            Morder = Morder * 2 + np.absolute(M[:, i, 0])
        index = np.argsort(-1 * Morder)
        return M[index, :, :]

    def random_pick_satProb(self):
        """
        This function randomly pick a file from the training file set
        """

        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def report_to_agent(self):
        """
        This function reports to the agent about the environment
        """
        return self.curr_state, self.S.getReward(), self.isSolved, {}

    def reset(self):
        """
        This function reset the environment and return the initial state
        """
        if self.test_mode:  # in test mode, we print the actual score of each SAT problem in test files
            print(self.score, end=".", flush=True)
        else:  # in training mode, we print an exponential average of scores of randomly picked files
            self.exp_av_score = self.exp_av_score * 0.98 + self.score * 0.02
            print(round(self.exp_av_score), end=".", flush=True)
        self.score = 0
        filename = self.random_pick_satProb()
        self.S = GymSolver(filename)
        self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
        return self.curr_state

    def step(self, decision):
        """
        This function make a step based on parameter input
        """
        self.score += 1
        if decision < 0:  # this is to say that let minisat pick the decision
            decision = 32767
        elif decision % 2 == 0:  # this is to say that pick decision and assign positive value
            decision = int(decision / 2 + 1)
        else:  # this is to say that pick decision and assign negative value
            decision = 0 - int(decision / 2 + 1)
        if (decision in self.actionSet) or (decision == 32767):
            self.S.step(decision)
            self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
            return self.report_to_agent()
        else:
            return self.report_to_agent()

    def render(self, mode='human', close=False):
        """
        This function renders the sat problem
        """
        pass


class gym_sat_permute_Env(gym.Env):
    """
    This class is a gym environment for Reinforcement Learning algorithms
    It always permute the rows after reading a file
    max_clause: the number of rows in state representation
    max_var: the number of columns in state representation
    """

    def __init__(self, max_clause=100, max_var=20, test_path=None):
        if test_path == None:
            self.test_mode = False
            # BE AWARE, the training data directory in statically determined in __init__function
            self.test_path = "uf20-91_train_v2"
            print("SAT-v2 (permute): We are in the training mode of path {}".format(self.test_path))
        else:
            self.test_mode = True
            self.test_path = test_path
            print("SAT-v2 (permute): We are in the test mode of path {}".format(self.test_path))
        # Get all test files
        self.test_files = [join(self.test_path, f)
                           for f in listdir(self.test_path)
                           if isfile(join(self.test_path, f))]
        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 1))
        self.action_space = spaces.Discrete(2 * self.max_var)
        self.score = 0
        self.exp_av_score = 15  # some randomly initialized initial average value

    def parse_state(self):
        """
        This function parse the state into sparse matrix with -1 or 1 values
        Can handle the case when state is empty and the SAT is either broken or solved already
        """
        curr_state = np.zeros((self.max_clause, self.max_var, 1), dtype=np.int8)
        clause_counter = 0  # this tracks the current row-to-write (which is permutable!)
        actionSet = set()  # this set tracks all allowed actions for this state
        # if S is already Done, should return here.
        if self.S.getDone():
            return curr_state, clause_counter, True, actionSet
        # S is not yet Done, parse and return real state
        for line in self.S.getState().split('\n'):
            if line.startswith("p cnf"):  # this is the header of a cnf problem # p cnf 20 90
                header = line.split(" ")
                num_var = int(header[2])
                num_clause = int(header[3])
                assert (num_var <= self.max_var)
            # remove this assert (might be wrong if we learnt too many clauses and restarted)
            # assert (num_clause <= self.max_clause)
            elif line.startswith("c"):
                continue
            else:  # clause data line # -11 -17 20 0
                literals = line.split(" ")
                n = len(literals)
                for j in range(n - 1):
                    number = int(literals[j])
                    value = 1 if number > 0 else -1
                    curr_state[clause_counter, abs(number) - 1] = value
                    actionSet.add(number)
                clause_counter += 1
                if clause_counter >= self.max_clause:  # add a safe guard for overflow of number of clauses
                    break
        return curr_state, clause_counter, False, actionSet

    def random_pick_satProb(self):
        """
        This function randomly pick a file from the training file set
        """
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def permute_row(self, filename):
        """
        This function permute the rows (clauses) of a given file, and rewrite that file with the permuted one
        """
        clauses = []
        header = None
        with open(filename, "r") as read_in:
            for line in read_in:
                if line.startswith("p cnf"):
                    header = line
                elif line.startswith("c"):
                    # comments line, skip
                    continue
                elif any(char.isdigit() and (not char == '0') for char in line):
                    # clause data line
                    # put them in a list first, then permute and write to write_out
                    clauses.append(line)
        with open(filename, 'w') as write_out:
            if header is None:
                print("file {} has no header".format(filename))
            else:
                write_out.write(header)
                # permute the clauses and write to write_out
                random.shuffle(clauses)
                for line in clauses:
                    write_out.write(line)

    def report_to_agent(self):
        """
        This function reports to the agent about the environment
        """
        return self.curr_state, self.S.getReward(), self.isSolved, {}

    def reset(self):
        """
        This function reset the environment and return the initial state
        """
        if self.test_mode:  # in test mode, we print the actual score of each SAT problem in test files
            print(self.score, end=".", flush=True)
        else:  # in training mode, we print an exponential average of scores of randomly picked files
            self.exp_av_score = self.exp_av_score * 0.98 + self.score * 0.02
            print(round(self.exp_av_score), end=".", flush=True)
        self.score = 0
        filename = self.random_pick_satProb()
        self.S = GymSolver(filename)
        # since we just used the "filename", we should permute rows of this file in a separate thread
        _thread.start_new_thread(self.permute_row, (filename,))
        self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
        return self.curr_state

    def step(self, decision):
        """
        This function make a step based on parameter input
        """
        self.score += 1
        if decision < 0:  # this is to say that let minisat pick the decision
            decision = 32767
        elif decision % 2 == 0:  # this is to say that pick decision and assign positive value
            decision = int(decision / 2 + 1)
        else:  # this is to say that pick decision and assign negative value
            decision = 0 - int(decision / 2 + 1)
        if (decision in self.actionSet) or (decision == 32767):
            self.S.step(decision)
            self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
            return self.report_to_agent()
        else:
            return self.report_to_agent()

    def render(self, mode='human', close=False):
        """
        This function renders the sat problem
        """
        pass


class gym_sat_graph_Env(gym.Env):
    """
    This class is a gym environment for Reinforcement Learning algorithms of SAT problems for graph coloring
    It doesn't do any preprocessing (sorting matrix) or postprocessing (permute the training file)
    max_clause: the number of rows in state representation
    max_var: the number of columns in state representation
    """

    def __init__(self, max_clause=300, max_var=90, test_path=None):
        if test_path == None:
            self.test_mode = False
            # BE AWARE, the directory of the training files is statically determined in __init__function
            self.test_path = "graph_rand_train"
            print("SAT-v3: We are in the training mode of path {}".format(self.test_path))
        else:
            self.test_mode = True
            self.test_path = test_path
            print("SAT-v3: We are in the test mode of path {}".format(self.test_path))
        # Get all test files
        self.test_files = [join(self.test_path, f)
                           for f in listdir(self.test_path)
                           if isfile(join(self.test_path, f))]
        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 1))
        self.action_space = spaces.Discrete(2 * self.max_var)
        self.score = 0
        self.exp_av_score = 15  # some randomly initialized initial average value

    def parse_state(self):
        """
        This function parse the state into sparse matrix with -1 or 1 values
        Can handle the case when state is empty and the SAT is either broken or solved already
        """
        curr_state = np.zeros((self.max_clause, self.max_var, 1), dtype=np.int8)
        clause_counter = 0  # this tracks the current row-to-write (which is permutable!)
        actionSet = set()  # this set tracks all allowed actions for this state
        # if S is already Done, should return here.
        if self.S.getDone():
            return curr_state, clause_counter, True, actionSet
        # S is not yet Done, parse and return real state
        for line in self.S.getState().split('\n'):
            if line.startswith("p cnf"):  # this is the header of a cnf problem # p cnf 20 90
                header = line.split(" ")
                num_var = int(header[2])
                num_clause = int(header[3])
                assert (num_var <= self.max_var)
            # remove this assert (might be wrong if we learnt too many clauses and restarted)
            # assert (num_clause <= self.max_clause)
            elif line.startswith("c"):
                continue
            elif any(char.isdigit() and (not char == '0') for char in line):
                # clause data line # -11 -17 20 0
                literals = line.split(" ")
                n = len(literals)
                for j in range(n - 1):
                    number = int(literals[j])
                    value = 1 if number > 0 else -1
                    curr_state[clause_counter, abs(number) - 1] = value
                    actionSet.add(number)
                clause_counter += 1
                if clause_counter >= self.max_clause:  # add a safe guard for overflow of number of clauses
                    print("total number of clauses overflowed max_clause, and were not all represented as state")
                    break
        return curr_state, clause_counter, False, actionSet

    def random_pick_satProb(self):
        """
        This function randomly pick a file from the training file set
        """
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def report_to_agent(self):
        """
        This function reports to the agent about the environment
        """
        return self.curr_state, self.S.getReward(), self.isSolved, {}

    def reset(self):
        """
        This function reset the environment and return the initial state
        """
        if self.test_mode:  # in test mode, we print the actual score of each SAT problem in test files
            print(self.score, end=".", flush=True)
        else:  # in training mode, we print an exponential average of scores of randomly picked files
            self.exp_av_score = self.exp_av_score * 0.98 + self.score * 0.02
            print(round(self.exp_av_score), end=".", flush=True)
        self.score = 0
        filename = self.random_pick_satProb()
        self.S = GymSolver(filename)
        self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
        return self.curr_state

    def step(self, decision):
        """
        This function make a step based on parameter input
        """
        self.score += 1
        if decision < 0:  # this is to say that let minisat pick the decision
            decision = 32767
        elif decision % 2 == 0:  # this is to say that pick decision and assign positive value
            decision = int(decision / 2 + 1)
        else:  # this is to say that pick decision and assign negative value
            decision = 0 - int(decision / 2 + 1)
        if (decision in self.actionSet) or (decision == 32767):
            self.S.step(decision)
            self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
            return self.report_to_agent()
        else:
            return self.report_to_agent()

    def render(self, mode='human', close=False):
        """
        This function renders the sat problem
        """
        pass


class gym_sat_graph2_Env(gym.Env):
    """
    This class is a gym environment for Reinforcement Learning algorithms of SAT problems for graph coloring 2
    It doesn't do any preprocessing (sorting matrix) or postprocessing (permute the training file)
    max_clause: the number of rows in state representation
    max_var: the number of columns in state representation
    """

    def __init__(self, max_clause=550, max_var=150, test_path=None):
        if test_path == None:
            self.test_mode = False
            # BE AWARE, the directory of the training files is statically determined in __init__function
            self.test_path = "graph2_rand_train"
            print("SAT-v4: We are in the training mode of path {}".format(self.test_path))
        else:
            self.test_mode = True
            self.test_path = test_path
            print("SAT-v4: We are in the test mode of path {}".format(self.test_path))
        # Get all test files
        self.test_files = [join(self.test_path, f)
                           for f in listdir(self.test_path)
                           if isfile(join(self.test_path, f))]
        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.max_clause = max_clause
        self.max_var = max_var
        self.observation_space = np.zeros((max_clause, max_var, 1))
        self.action_space = spaces.Discrete(2 * self.max_var)
        self.score = 0
        self.exp_av_score = 15  # some randomly initialized initial average value

    def parse_state(self):
        """
        This function parse the state into sparse matrix with -1 or 1 values
        Can handle the case when state is empty and the SAT is either broken or solved already
        """
        curr_state = np.zeros((self.max_clause, self.max_var, 1), dtype=np.int8)
        clause_counter = 0  # this tracks the current row-to-write (which is permutable!)
        actionSet = set()  # this set tracks all allowed actions for this state
        # if S is already Done, should return here.
        if self.S.getDone():
            return curr_state, clause_counter, True, actionSet
        # S is not yet Done, parse and return real state
        for line in self.S.getState().split('\n'):
            if line.startswith("p cnf"):  # this is the header of a cnf problem # p cnf 20 90
                header = line.split(" ")
                num_var = int(header[2])
                num_clause = int(header[3])
                assert (num_var <= self.max_var)
            # remove this assert (might be wrong if we learnt too many clauses and restarted)
            # assert (num_clause <= self.max_clause)
            elif line.startswith("c"):
                continue
            elif any(char.isdigit() and (not char == '0') for char in line):
                # clause data line # -11 -17 20 0
                literals = line.split(" ")
                n = len(literals)
                for j in range(n - 1):
                    number = int(literals[j])
                    value = 1 if number > 0 else -1
                    curr_state[clause_counter, abs(number) - 1] = value
                    actionSet.add(number)
                clause_counter += 1
                if clause_counter >= self.max_clause:  # add a safe guard for overflow of number of clauses
                    print("total number of clauses overflowed max_clause, and were not all represented as state")
                    break
        return curr_state, clause_counter, False, actionSet

    def random_pick_satProb(self):
        """
        This function randomly pick a file from the training file set
        """
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def report_to_agent(self):
        """
        This function reports to the agent about the environment
        """
        return self.curr_state, self.S.getReward(), self.isSolved, {}

    def reset(self):
        """
        This function reset the environment and return the initial state
        """
        if self.test_mode:  # in test mode, we print the actual score of each SAT problem in test files
            print(self.score, end=".", flush=True)
        else:  # in training mode, we print an exponential average of scores of randomly picked files
            self.exp_av_score = self.exp_av_score * 0.98 + self.score * 0.02
            print(round(self.exp_av_score), end=".", flush=True)
        self.score = 0
        filename = self.random_pick_satProb()
        self.S = GymSolver(filename)
        self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
        return self.curr_state

    def step(self, decision):
        """
        This function make a step based on parameter input
        """
        self.score += 1
        if decision < 0:  # this is to say that let minisat pick the decision
            decision = 32767
        elif decision % 2 == 0:  # this is to say that pick decision and assign positive value
            decision = int(decision / 2 + 1)
        else:  # this is to say that pick decision and assign negative value
            decision = 0 - int(decision / 2 + 1)
        if (decision in self.actionSet) or (decision == 32767):
            self.S.step(decision)
            self.curr_state, self.clause_counter, self.isSolved, self.actionSet = self.parse_state()
            return self.report_to_agent()
        else:
            return self.report_to_agent()

    def render(self, mode='human', close=False):
        """
        This function renders the sat problem
        """
        pass
