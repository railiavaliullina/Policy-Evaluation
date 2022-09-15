import numpy as np
import pandas as pd
import decimal
from collections import Counter

from MDP.policy_evaluation import PolicyEvaluation
from utils.data_utils import write_file
from configs.config import cfg
from enums.enums import *


class Executor(object):
    def __init__(self):
        """
        Class for running policy evaluation algorithms with different params several times and analysing results.
        """
        self.cfg = cfg
        np.random.seed(0)
        self.seeds = np.random.choice(int(5e5), self.cfg.runs_num, replace=False)
        self.map_names = [m.name for m in MapName]
        self.policy_types = [p.name for p in PolicyType]
        self.action_sets = [a.name for a in ActionSet]
        self.direct_solution_avg_time = {m: [] for m in self.map_names}
        self.iterative_solution_avg_time = {m: [] for m in self.map_names}
        self.policy_evaluation = PolicyEvaluation(cfg)
        self.part_of_states_with_better_acc_direct = {m: [] for m in self.map_names}
        self.mean_accuracy_direct = {m: [] for m in self.map_names}
        self.mean_accuracy_iterative = {m: [] for m in self.map_names}
        self.diff_between_solutions = {m: [] for m in self.map_names}

    @staticmethod
    def get_solution_acc(num):
        """
        Gets floating point accuracies
        :param num: vector of nums
        :return: accuracies array
        """
        num = [decimal.Decimal(n) for n in num]
        acc = []
        for n in num:
            split_num = str(n).split('.')
            split_len = 0 if len(split_num) < 2 else len(split_num[-1])
            acc.append(split_len)
        return acc

    def compare_accuracies(self):
        """
        Compares and saves accuracies for direct and iterative solutions
        """
        solutions_acc = np.zeros((2, self.policy_evaluation.states_num))
        self.v_direct_acc = self.get_solution_acc(self.policy_evaluation.v_direct)
        self.v_iterative_acc = self.get_solution_acc(self.policy_evaluation.v_iterative)
        best_accuracies = Counter(np.argmax(solutions_acc, 0)).most_common()
        self.part_of_states_with_better_acc_direct[self.cfg.map_name].append(
            best_accuracies[0][1] / self.policy_evaluation.states_num)
        self.mean_accuracy_direct[self.cfg.map_name].append(np.mean(self.v_direct_acc))
        self.mean_accuracy_iterative[self.cfg.map_name].append(np.mean(self.v_iterative_acc))
        self.diff_between_solutions[self.cfg.map_name].append(np.sum(abs(self.policy_evaluation.v_direct -
                                                                         self.policy_evaluation.v_iterative)))

    @staticmethod
    def get_mean_values_in_dict(d, a=1, b=0):
        """
        Gets dict with mean values of given dict values
        :param d: dict
        :param a: mean value factor
        :param b: mean value bias
        :return: dict with mean values of given dict values
        """
        return {k: a * np.mean(v) + b for k, v in d.items()}

    def write_stats_to_file(self):
        """
        Aggregates and writes stats as dict to /data/runs_stats.json file
        """
        direct_solution_avg_time = self.get_mean_values_in_dict(self.direct_solution_avg_time)
        iterative_solution_avg_time = self.get_mean_values_in_dict(self.iterative_solution_avg_time)

        direct_solution_part_of_states_with_better_acc = self.get_mean_values_in_dict(
            self.part_of_states_with_better_acc_direct)
        iterative_solution_part_of_states_with_better_acc = self.get_mean_values_in_dict(
            direct_solution_part_of_states_with_better_acc, a=-1, b=1)

        mean_accuracy_direct = self.get_mean_values_in_dict(self.mean_accuracy_direct)
        mean_accuracy_iterative = self.get_mean_values_in_dict(self.mean_accuracy_iterative)
        mean_diff_between_solutions = self.get_mean_values_in_dict(self.diff_between_solutions)

        write_file(self.cfg.runs_stats_file_path, {'time': {'direct_solution_avg_time': direct_solution_avg_time,
                                                            'iterative_solution_avg_time': iterative_solution_avg_time},
                                                   'acc': {
                                                       'direct_solution_part_of_states_with_better_acc':
                                                           direct_solution_part_of_states_with_better_acc,
                                                       'iterative_solution_part_of_states_with_better_acc':
                                                           iterative_solution_part_of_states_with_better_acc,
                                                       'mean_accuracy_direct': mean_accuracy_direct,
                                                       'mean_accuracy_iterative': mean_accuracy_iterative,
                                                       'mean_diff_between_solutions': mean_diff_between_solutions
                                                   }})

    def measure_time_and_acc(self):
        """
        Runs loop for different env sizes several times
        """
        cfg.verbose = False

        for m_id, map_name in enumerate(self.map_names):
            self.cfg.map_name = map_name

            for s_id, seed in enumerate(self.seeds):
                np.random.seed(seed)

                print(f'Map: {m_id}/{len(self.map_names)}, seed: {s_id}/{len(self.seeds)}')

                self.policy_evaluation = PolicyEvaluation(self.cfg)
                self.policy_evaluation.run()

                self.direct_solution_avg_time[map_name].append(self.policy_evaluation.direct_solution_time)
                self.iterative_solution_avg_time[map_name].append(self.policy_evaluation.iterative_solution_time)

                self.compare_accuracies()
                self.write_stats_to_file()

    def write_results_to_csv(self, action_set, policy_type, direct_solution, iterative_solution, discount_factor,
                             mean_time_direct, mean_accuracy_direct, mean_time_iterative, mean_accuracy_iterative):
        """
        Writes all experiments results to csv file.

        :param action_set: list of action_set (default or slippery)
        :param policy_type: list of policy_type (stochastic or optimal)
        :param direct_solution: list of solutions, given by direct algorithm
        :param iterative_solution: list of solutions, given by iterative algorithm
        :param discount_factor: list of discount_factor values
        :param mean_time_direct: list of mean time per experiment (run 200 times to get mean time)
        :param mean_accuracy_direct: list of mean accuracy per experiment (run 200 times to get mean accuracy)
        :param mean_time_iterative: list of mean time per experiment (run 200 times to get mean time)
        :param mean_accuracy_iterative: list of mean accuracy per experiment (run 200 times to get mean accuracy)
        """
        df = pd.DataFrame()
        df['action_set'] = action_set
        df['policy_type'] = policy_type
        df['direct_solution'] = direct_solution
        df['iterative_solution'] = iterative_solution
        df['discount_factor'] = discount_factor
        df['mean_time_direct'] = mean_time_direct
        df['mean_time_iterative'] = mean_time_iterative
        df['time_difference'] = abs(np.asarray(mean_time_direct) - np.asarray(mean_time_iterative))
        df['mean_accuracy_direct'] = mean_accuracy_direct
        df['mean_accuracy_iterative'] = mean_accuracy_iterative
        df['accuracy_difference'] = abs(np.asarray(mean_accuracy_direct) - np.asarray(mean_accuracy_iterative))
        df.to_csv(self.cfg.results_file_path)
        print(f'Saved csv file with results to {self.cfg.results_file_path}.')

    def run_sequence_of_experiments(self):
        """
        Runs sequence of experiments with different params.
        """
        action_set_list, policy_type_list, direct_solution, iterative_solution, discount_factor_list = \
            [], [], [], [], []
        mean_time_direct, mean_accuracy_direct, mean_time_iterative, mean_accuracy_iterative = [], [], [], []

        for p_id, policy_type in enumerate(self.policy_types):
            self.cfg.policy_type = policy_type

            for a_id, action_set in enumerate(self.action_sets):
                self.cfg.action_set = action_set

                for discount_factor in np.arange(0, 1, 0.05):
                    self.cfg.discount_factor = discount_factor

                    time_direct, time_iterative = [], []
                    for _ in range(self.cfg.runs_num):

                        self.policy_evaluation = PolicyEvaluation(self.cfg)
                        self.policy_evaluation.run()

                        time_direct.append(self.policy_evaluation.direct_solution_time)
                        time_iterative.append(self.policy_evaluation.iterative_solution_time)

                        # accuracy_direct.append(np.mean(self.get_solution_acc(self.policy_evaluation.v_direct)))
                        # accuracy_iterative.append(np.mean(self.get_solution_acc(self.policy_evaluation.v_iterative)))

                    mean_time_direct.append(np.mean(time_direct))
                    mean_time_iterative.append(np.mean(time_iterative))

                    mean_accuracy_direct.append(np.mean(self.get_solution_acc(self.policy_evaluation.v_direct)))
                    mean_accuracy_iterative.append(np.mean(self.get_solution_acc(self.policy_evaluation.v_iterative)))

                    action_set_list.append(self.cfg.action_set)
                    policy_type_list.append(self.cfg.policy_type)
                    direct_solution.append(self.policy_evaluation.v_direct)
                    iterative_solution.append(self.policy_evaluation.v_iterative)
                    discount_factor_list.append(self.cfg.discount_factor)

        self.write_results_to_csv(action_set_list, policy_type_list, direct_solution, iterative_solution,
                                  discount_factor_list, mean_time_direct, mean_accuracy_direct, mean_time_iterative,
                                  mean_accuracy_iterative)

    def run(self):
        """
        Runs whole pipeline.
        """
        if self.cfg.run_single_exp:
            self.policy_evaluation.run()
        else:
            self.run_sequence_of_experiments()

        if self.cfg.measure_time_and_acc:
            self.measure_time_and_acc()


if __name__ == '__main__':
    executor = Executor()
    executor.run()
