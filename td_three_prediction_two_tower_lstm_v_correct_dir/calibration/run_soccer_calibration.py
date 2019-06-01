import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')
from td_three_prediction_two_tower_lstm_v_correct_dir.calibration.calibration import Calibration


def generate_cali_latex_table(result_file_dir):
    calibration_features = ['period', 'score_differential', 'pitch', 'manpower']
    calibration_bins = {'period': {'feature_name': ('sec', 'min'), 'range': (1, 2)},
                        'score_differential': {'feature_name': ('scoreDiff'), 'range': range(-1, 2)},
                        'pitch': {'feature_name': ('x'), 'range': ('left', 'right')},
                        'manpower': {'feature_name': ('manPower'), 'range': range(-1, 2)}
                        }
    with open(result_file_dir) as f:
        data = f.readlines()
    str_all = ''
    ref_dict = {'score_differential': 0, 'manpower': 0, 'period': 0, 'pitch': 0}
    for score_diff in calibration_bins['manpower']['range']:
        ref_dict['manpower'] = score_diff
        for manpower in calibration_bins['score_differential']['range']:
            ref_dict['score_differential'] = manpower
            for period in calibration_bins['period']['range']:
                ref_dict['period'] = period
                for pitch in calibration_bins['pitch']['range']:
                    ref_dict['pitch'] = pitch
                    ref_str = ''
                    for feature in calibration_features:
                        ref_str = ref_str + feature + '_' + str(ref_dict[feature]) + '-'

                    for line in data:
                        eles = line.split('\t')
                        red_str = eles[0].split(':')[1]

                        if ref_str == red_str:
                            number = eles[1].split(':')[1]
                            h_cali = round(float(eles[2].split(':')[1]), 4)
                            h_model = round(float(eles[3].split(':')[1]), 4)
                            a_cali = round(float(eles[5].split(':')[1]), 4)
                            a_model = round(float(eles[6].split(':')[1]), 4)
                            kld = round(float(eles[10].split(':')[1].replace('\n', '')), 4)
                            mae = round(float(eles[11].split(':')[1].replace('\n', '')), 4)

                            str_all += '{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} & {10} \\\\ \n'.format(
                                str(score_diff), str(manpower), str(period), str(pitch),
                                str(number), str(h_model), str(a_model), str(h_cali), str(a_cali), str(kld), str(mae)
                            )

    print str_all + '\hline'


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
    apply_old = True
    apply_difference = False
    Cali = Calibration(bins=calibration_bins, data_path=data_path,
                       calibration_features=calibration_features, tt_lstm_config_path=tt_lstm_config_path,
                       soccer_data_store_dir=soccer_data_store_dir, apply_old=apply_old,
                       apply_difference=apply_difference,
                       focus_actions_list=['shot', 'pass'])
    Cali.construct_bin_dicts()
    Cali.aggregate_calibration_values()
    Cali.compute_distance()
    # save_calibration_dir = "/Local-Scratch/PycharmProjects/Sport-Analytic-NN/td_three_prediction_two_tower_lstm_v_correct_dir/calibration/calibration_results/calibration-['shot', 'pass']-2019May29.txt"
    generate_cali_latex_table(Cali.save_calibration_dir)
    # generate_cali_latex_table(save_calibration_dir)
