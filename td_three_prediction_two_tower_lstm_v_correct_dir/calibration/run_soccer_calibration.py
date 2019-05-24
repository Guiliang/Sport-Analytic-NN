import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')
from td_three_prediction_two_tower_lstm_v_correct_dir.calibration.calibration import Calibration

if __name__ == '__main__':
    calibration_features = ['period', 'score_differential', 'pitch', 'manpower']
    calibration_bins = {'period': {'feature_name': ('sec', 'min'), 'range': (1, 2)},
                        'score_differential': {'feature_name': ('scoreDiff'), 'range': range(-8, 8)},
                        'pitch': {'feature_name': ('x'), 'range': ('left', 'right')},
                        'manpower': {'feature_name': ('manPower'), 'range': (-3, -2, -1, 0, 1, 2, 3)}
                        }
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data"
    tt_lstm_config_path = "../soccer-config.yaml"
    Cali = Calibration(bins=calibration_bins, data_path=data_path,
                       calibration_features=calibration_features, tt_lstm_config_path=tt_lstm_config_path,
                       soccer_data_store_dir=soccer_data_store_dir,
                       result_dir='./cali_results/calibration_result.txt')
    Cali.construct_bin_dicts()
    Cali.aggregate_calibration_values()
    Cali.compute_kld()
