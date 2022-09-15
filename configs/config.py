from easydict import EasyDict

from enums.enums import *

cfg = EasyDict()

cfg.optimal_policies_file_path = '../data/optimal_policies.json'
cfg.results_file_path = '../data/results.csv'
cfg.runs_stats_file_path = '../data/runs_stats.json'

cfg.discount_factor = 0.9
cfg.estimation_accuracy_thr = 1e-12

cfg.map_name = MapName.small.name
cfg.policy_type = PolicyType.optimal.name
cfg.action_set = ActionSet.default.name
cfg.verbose = True

cfg.measure_time_and_acc = False
cfg.runs_num = 200

cfg.run_single_exp = True
