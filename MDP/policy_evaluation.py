import gym
import numpy as np
import time

from utils.data_utils import read_file


class PolicyEvaluation(object):
    """
    Class with policy evaluation implementation.
    """
    def __init__(self, cfg):
        """
        Initializes class.
        :param cfg: config
        """
        self.cfg = cfg

        self.init_env()
        self.show_env_info()

        self.transition_matrix = self.env.transition_matrix
        self.states_num = len(self.transition_matrix)
        self.actions_num = len(self.transition_matrix[0])
        self.init_matrices()

        self.solutions_types = ['direct', 'iterative']

    def init_env(self):
        """
        Initializes environment with parameters from config.
        """
        self.env = gym.make('frozen_lake:default-v0', map_name=self.cfg.map_name, action_set_name=self.cfg.action_set)
        self.env.reset(start_state_index=0)

    def show_env_info(self):
        """
        Prints information about current environment.
        """
        if self.cfg.verbose:
            self.env.render(object_type="environment")
            self.env.render(object_type="actions")
            self.env.render(object_type="states")
            print(f'\nMap type: {self.cfg.map_name}\n'
                  f'Policy type: {self.cfg.policy_type}\n'
                  f'Action Set: {self.cfg.action_set}\n'
                  f'Discount Factor: {self.cfg.discount_factor}')

    def init_matrices(self):
        """
        init transition probability matrix, reward matrix.
        """
        self.transition_prob_matrix = np.zeros((self.states_num, self.states_num, self.actions_num))
        self.reward_matrix = np.zeros((self.states_num, self.actions_num, self.states_num))

        for s in range(self.states_num):
            for a in range(self.actions_num):
                for new_s_tuple in self.transition_matrix[s][a]:
                    transition_prob, new_s, reward, _ = new_s_tuple
                    self.transition_prob_matrix[s][new_s][a] += transition_prob
                    self.reward_matrix[s][a][new_s] += reward

        self.reward_matrix = np.max(self.reward_matrix, axis=-1)  # mean

    def get_policy(self):
        """
        Initializes policy within config params.
        """
        if self.cfg.policy_type == 'stochastic':
            self.policy = np.random.uniform(0, 1, (self.states_num, self.actions_num))
            self.policy = [self.policy[i, :] / np.sum(self.policy, 1)[i]
                           for i in range(self.states_num)]

        elif self.cfg.policy_type == 'optimal':
            data = read_file(self.cfg.optimal_policies_file_path)
            self.policy = np.asarray(data[self.cfg.map_name][self.cfg.action_set])

        else:
            raise Exception

    def get_model_with_policy(self):
        """
        Gets transition probability matrix, reward matrix within chosen policy.
        :return:
        """
        self.transition_prob_matrix_pi = np.zeros((self.states_num, self.states_num))
        self.reward_matrix_pi = np.zeros(self.states_num)

        for s in range(self.states_num):
            for new_s in range(self.states_num):
                self.transition_prob_matrix_pi[s][new_s] = self.policy[s] @ self.transition_prob_matrix[s][new_s]

            self.reward_matrix_pi[s] = self.policy[s] @ self.reward_matrix[s]

    def get_direct_solution(self):
        """
        Gets direct solution with given formula V = (I − γP)^−1 R.
        :return: v vector
        """
        start_time = time.time()

        eye_matrix = np.eye(self.states_num)
        v_pi = np.linalg.inv(
            eye_matrix - self.cfg.discount_factor * self.transition_prob_matrix_pi) @ self.reward_matrix_pi

        self.direct_solution_time = time.time() - start_time
        if self.cfg.verbose:
            print(f'\nDirect solution time: {self.direct_solution_time} s')
        return v_pi

    def get_iterative_solution(self):
        """
        Gets iterative solution.
        :return: v vector
        """
        start_time = time.time()

        v = np.zeros(self.states_num)
        step = 0

        while True:  # step <= self.cfg.max_iter:
            dif = 0
            for s in range(self.states_num):
                prev_v = v[s]
                v[s] = self.reward_matrix_pi[s] + self.cfg.discount_factor * (self.transition_prob_matrix_pi[s] @ v)
                dif = np.max([dif, abs(prev_v - v[s])])
            # print(step, dif, v')
            step += 1

            if dif < self.cfg.estimation_accuracy_thr:
                break

        self.iterative_solution_time = time.time() - start_time
        if self.cfg.verbose:
            print(f'Iterative solution time: {self.iterative_solution_time} s\n')
        return v

    def run(self):
        """
        Runs direct and iterative policy evaluation algorithms.
        """

        self.get_policy()
        self.get_model_with_policy()

        self.v_direct = self.get_direct_solution()
        self.v_iterative = self.get_iterative_solution()

        assert np.allclose(self.v_iterative, self.v_direct, atol=1e-1)

        if self.cfg.verbose:
            print(f'Direct solution: v={self.v_direct}\n')
            print(f'Iterative solution: v={self.v_iterative}\n')

            fast_solution_type = self.solutions_types[np.argmin([self.direct_solution_time,
                                                                 self.iterative_solution_time])]
            time_dif = abs(self.direct_solution_time - self.iterative_solution_time)
            print(f'Faster solution: {fast_solution_type}, is faster by: {time_dif} seconds')
