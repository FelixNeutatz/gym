"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

class ColumnSelectionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.number_of_columns = 20 #number_of_columns
        self.probability_of_empty_column = 0.3 #probability_of_empty_column
        self.use_delta_fscore_as_reward = True

        self.viewer = None

        # define minimum and maximum of all observations
        self.low = np.zeros(224 * self.number_of_columns, dtype = float)
        self.high = np.ones(224 * self.number_of_columns, dtype = float)

        #number of labels can be infinite
        for i in range(self.number_of_columns):
            self.high[(224 * i) + 1] = np.finfo(np.float32).max
            self.high[(224 * i) + 113] = np.finfo(np.float32).max

        self.action_space = spaces.Discrete(self.number_of_columns)
        self.observation_space = spaces.Box(self.low, self.high)

        self.c_matrix, self.t_matrix, self.maximal_number_of_rounds = self.load_data()

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        reward = 0.0

        done = False

        if self.column_ids[action] != -1:
            if self.game_state[action] + 1 < self.maximal_number_of_rounds:
                # go to next round for column that was chosen
                self.game_state[action] = self.game_state[action] + 1

                # create new state
                self.state = self.set_state()

                # calculate reward
                if self.use_delta_fscore_as_reward:
                    reward = self.calculate_reward_delta(action)
                else:
                    reward = self.calculate_reward()
            else:
                done = True
        else:
            reward = -1.0

        return np.array(self.state), reward, done, {}

    # calculate delta fscore
    def calculate_reward_delta(self, action):
        current_column_id = self.column_ids[action]
        current_column_round = self.game_state[action]

        delta_fscore = 0.0
        if not current_column_id == -1:
            if current_column_round == 0: # should not happen
                delta_fscore = self.t_matrix[current_column_id][0]
            else:
                delta_fscore = self.t_matrix[current_column_id][current_column_round] - self.t_matrix[current_column_id][current_column_round - 1]
        else: # should not happen
            delta_fscore = -1.0

        return delta_fscore

    def calculate_reward(self):
        sum = 0.0
        i = 0
        for c in range(self.number_of_columns):
            current_column_id = self.column_ids[c]
            current_column_round = self.game_state[c]

            if not current_column_id == -1:
                sum += self.t_matrix[current_column_id][current_column_round]
                i += 1

        return sum / i

    def read_csv1(self, path, header):
        data = pd.read_csv(path, header=header)

        x = data[data.columns[0:(data.shape[1] - 1)]].values
        y = data[data.columns[data.shape[1] - 1]].values

        return x, y

    def add_history(self, x, y, nr_columns):
        x_with_history = np.hstack((x[nr_columns:len(x), :], x[0:len(x) - nr_columns, :]))
        y_with_history = y[nr_columns:len(x)]

        return x_with_history, y_with_history


    def create_matrix(self, x, y, number_of_columns):
        number_of_rounds = int(len(x) / number_of_columns)

        columns_matrix = np.empty((number_of_columns, number_of_rounds, x.shape[1]))
        target_matrix = np.empty((number_of_columns, number_of_rounds))

        for i in range(len(x)):
            current_column = i % number_of_columns
            current_round = int((i - (i % number_of_columns)) / number_of_columns)

            columns_matrix[current_column, current_round] = x[i]
            target_matrix[current_column, current_round] = y[i]

        return columns_matrix, target_matrix



    def load_data(self):
        train_x, train_y = self.read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_hospital.csv", None)
        #train_x1, train_y1 = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_blackoak.csv", None)
        train_x1, train_y1 = self.read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_flight.csv", None)

        #remove prediction change features
        train_x = train_x[:, 0:train_x.shape[1] - 4]
        train_x1 = train_x1[:, 0:train_x1.shape[1] - 4]

        col1 = 17
        col2 = 4

        train_x, train_y = self.add_history(train_x, train_y, col1)
        train_x1, train_y1 = self.add_history(train_x1, train_y1, col2)

        column_matrix1, target_matrix1 = self.create_matrix(train_x, train_y, col1)
        column_matrix2, target_matrix2 = self.create_matrix(train_x1, train_y1, col2)

        column_matrix_all = np.concatenate((column_matrix1, column_matrix2), axis=0)
        target_matrix_all = np.concatenate((target_matrix1, target_matrix2), axis=0)

        maximal_number_of_rounds = int(len(train_x) / col1)

        return column_matrix_all, target_matrix_all, maximal_number_of_rounds

    def set_state(self):
        features = []
        for c in range(self.number_of_columns):
            if self.column_ids[c] == -1:
                features.extend(np.zeros(224))
            else:
                current_column_id = self.column_ids[c]
                current_column_round = self.game_state[c]
                features.extend(self.c_matrix[current_column_id][current_column_round])

        return np.array(features)


    def _reset(self):
        self.game_state = np.zeros(self.number_of_columns, dtype=int)

        # sample from all columns
        c_ids = np.arange(len(self.c_matrix))
        self.np_random.shuffle(c_ids)
        self.column_ids = c_ids[0:self.number_of_columns]

        # randomly select columns to be empty
        is_empty = self.np_random.rand(self.number_of_columns) < self.probability_of_empty_column

        #print("number empty columns: " + str(np.sum(is_empty)))

        self.column_ids[is_empty] = -1

        self.state = self.set_state()

        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 10
        screen_height = 10


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ == '__main__':
    cs = ColumnSelectionEnv()

    done = False
    i = 1
    while not done:
        action = np.random.randint(0,20)
        _, reward, done, _ = cs._step(action)

        print("round: " +  str(i) +" action:"  + str(action) +" reward: " + str(reward))
        i += 1
