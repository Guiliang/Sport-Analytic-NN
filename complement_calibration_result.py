import ast
import csv
import math
from scipy.stats.stats import pearsonr

project_dir = "/cs/oschulte/Galen/PycharmProjects/Sport-Analytic-NN/"

calibration_result_dirs = [
    "td_three_prediction_simple_v_correct_dir/calibration_result/home_td_three_cut_together_calibration_entire_feature_5_V3_Iter50_batch32_v_correct_sum_2017-8-05.csv",
    "td_three_prediction_lstm_v_correct_dir/calibration_result/home_td_three_lstm_cut_together_calibration_entire_feature_5_v3_Iter50_lr5_batch32_v_correct_sum_2017-8-09.csv",
    "mc_three_prediction_simple_v_correct_dir/calibration_result/home_mc_three_cut_together_calibration_entire_feature_5_V3_Iter50_batch32_v_correct_sum_2017-8-05.csv",
    "td_three_prediction_fix_rnn_v_correct_dir/calibration_result/home_fix_rnn_td_three_cut_together_calibration_entire_feature_5_v1_Iter50_batch32_v_correct_sum_2017-8-05.csv",
    "td_three_prediction_simple_c4_v_correct_dir/calibration_result/home_c4_td_three_cut_together_calibration_entire_feature_5_V3_Iter50_batch32_v_correct_sum_2017-8-05.csv"
]

csv_sequence = ['Goal Different', 'Manpower', 'Period', 'game_count', 'state_count',
                'model_predict_home_value_record_average', 'model_poss_predict_home_value_record_average',
                'calibration_home_value_record_average',
                'abs_home_difference',
                'model_predict_away_value_record_average', 'model_poss_predict_away_value_record_average',
                'calibration_away_value_record_average',
                'abs_away_difference',
                'model_predict_end_value_record_average', 'model_poss_predict_end_value_record_average',
                'calibration_end_value_record_average', 'abs_end_difference',
                'kld'
                ]


def read_calibration_csv(calibration_result_dir):
    csv_read_dict_list = []
    csv_road_name = []
    first_row_flag = True
    with open(calibration_result_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if first_row_flag:
                csv_road_name = row
                first_row_flag = False
                continue
            else:
                csv_row_read_dict = {}
                for item_index in range(0, len(row)):
                    csv_row_read_dict.update({csv_road_name[item_index]: row[item_index]})
                csv_read_dict_list.append(csv_row_read_dict)
    return csv_read_dict_list


def compute_abs_difference(csv_read_dict_list):
    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        csv_row_read_dict = csv_read_dict_list[csv_row_read_dict_index]
        model_predict_home_value_record_average = float(
            csv_row_read_dict.get("model_predict_home_value_record_average"))
        model_predict_away_value_record_average = float(
            csv_row_read_dict.get("model_predict_away_value_record_average"))
        model_predict_end_value_record_average = float(csv_row_read_dict.get("model_predict_end_value_record_average"))
        calibration_home_value_record_average = float(csv_row_read_dict.get("calibration_home_value_record_average"))
        calibration_away_value_record_average = float(csv_row_read_dict.get("calibration_away_value_record_average"))
        calibration_end_value_record_average = float(csv_row_read_dict.get("calibration_end_value_record_average"))

        abs_home_difference = abs(model_predict_home_value_record_average - calibration_home_value_record_average)
        abs_away_difference = abs(model_predict_away_value_record_average - calibration_away_value_record_average)
        abs_end_difference = abs(model_predict_end_value_record_average - calibration_end_value_record_average)

        csv_row_read_dict.update({"abs_home_difference": abs_home_difference})
        csv_row_read_dict.update({"abs_away_difference": abs_away_difference})
        csv_row_read_dict.update({"abs_end_difference": abs_end_difference})
        csv_read_dict_list[csv_row_read_dict_index] = csv_row_read_dict

    return csv_read_dict_list


def compute_kld(csv_read_dict_list):
    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        csv_row_read_dict = csv_read_dict_list[csv_row_read_dict_index]
        model_predict_home_value_record_average = float(
            csv_row_read_dict.get("model_predict_home_value_record_average"))
        model_predict_away_value_record_average = float(
            csv_row_read_dict.get("model_predict_away_value_record_average"))
        model_predict_end_value_record_average = float(csv_row_read_dict.get("model_predict_end_value_record_average"))
        calibration_home_value_record_average = float(csv_row_read_dict.get("calibration_home_value_record_average"))
        calibration_away_value_record_average = float(csv_row_read_dict.get("calibration_away_value_record_average"))
        calibration_end_value_record_average = float(csv_row_read_dict.get("calibration_end_value_record_average"))

        if model_predict_end_value_record_average < 0 or calibration_end_value_record_average == 0:
            model_predict_home_poss = model_predict_home_value_record_average / (
                model_predict_home_value_record_average + model_predict_away_value_record_average)
            model_predict_away_poss = model_predict_away_value_record_average / (
                model_predict_home_value_record_average + model_predict_away_value_record_average)
            model_predict_end_poss = 1000  # denominator must bigger then 0
            calibration_home_poss = calibration_home_value_record_average / (
                calibration_home_value_record_average + calibration_away_value_record_average)
            calibration_away_poss = calibration_away_value_record_average / (
                calibration_home_value_record_average + calibration_away_value_record_average)
            calibration_end_poss = 0
            kld = calibration_home_poss * math.log(
                calibration_home_poss / model_predict_home_poss) + calibration_away_poss * math.log(
                calibration_away_poss / model_predict_away_poss)
        else:
            model_predict_home_poss = model_predict_home_value_record_average / (
                model_predict_home_value_record_average + model_predict_away_value_record_average + model_predict_end_value_record_average)
            model_predict_away_poss = model_predict_away_value_record_average / (
                model_predict_home_value_record_average + model_predict_away_value_record_average + model_predict_end_value_record_average)
            model_predict_end_poss = model_predict_end_value_record_average / (
                model_predict_home_value_record_average + model_predict_away_value_record_average + model_predict_end_value_record_average)
            calibration_home_poss = calibration_home_value_record_average / (
                calibration_home_value_record_average + calibration_away_value_record_average + calibration_end_value_record_average)
            calibration_away_poss = calibration_away_value_record_average / (
                calibration_home_value_record_average + calibration_away_value_record_average + calibration_end_value_record_average)
            calibration_end_poss = calibration_end_value_record_average / (
                calibration_home_value_record_average + calibration_away_value_record_average + calibration_end_value_record_average)
            kld = calibration_home_poss * math.log(
                calibration_home_poss / model_predict_home_poss) + calibration_away_poss * math.log(
                calibration_away_poss / model_predict_away_poss) + calibration_end_poss * math.log(
                calibration_end_poss / model_predict_end_poss)

        csv_row_read_dict.update({"kld": kld})
        csv_read_dict_list[csv_row_read_dict_index] = csv_row_read_dict
    return csv_read_dict_list


def compute_average(csv_read_dict_list):
    average_target_dict = {'abs_home_difference': 0, 'abs_away_difference': 0, 'abs_end_difference': 0, 'kld': 0}
    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        csv_row_read_dict = csv_read_dict_list[csv_row_read_dict_index]
        for key in average_target_dict.keys():
            value = csv_row_read_dict.get(key)
            value_new = average_target_dict.get(key) + value
            average_target_dict.update({key: value_new})

    append_dict = {}
    for key_dict in csv_read_dict_list[0].keys():
        append_dict.update({key_dict: ""})

    for key_target in average_target_dict.keys():
        value_average = average_target_dict.get(key_target) / len(csv_read_dict_list)
        append_dict.update({key_target: value_average})

    return append_dict


def compute_correlated_coefficient(csv_read_dict_list):
    coe_target_dict = {'model_predict_home_value_record_average': [], 'calibration_home_value_record_average': [],
                       'model_predict_away_value_record_average': [], 'calibration_away_value_record_average': [],
                       'model_predict_end_value_record_average': [], 'calibration_end_value_record_average': []}

    coe_save_dict = ['abs_home_difference', 'abs_away_difference', 'abs_end_difference']
    coe_save_value_list = []

    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        csv_row_read_dict = csv_read_dict_list[csv_row_read_dict_index]
        for key in coe_target_dict.keys():
            value = float(csv_row_read_dict.get(key))
            value_new = (coe_target_dict.get(key))
            value_new.append(value)
            coe_target_dict.update({key: value_new})

    coe_save_value_list.append(pearsonr(coe_target_dict.get('model_predict_home_value_record_average'),
                                        coe_target_dict.get('calibration_home_value_record_average')))
    coe_save_value_list.append(pearsonr(coe_target_dict.get('model_predict_away_value_record_average'),
                                        coe_target_dict.get('calibration_away_value_record_average')))
    coe_save_value_list.append(pearsonr(coe_target_dict.get('model_predict_end_value_record_average'),
                                        coe_target_dict.get('calibration_end_value_record_average')))

    append_dict = {}
    for key_dict in csv_read_dict_list[0].keys():
        append_dict.update({key_dict: ""})

    i = 0  # TODO: wrong sequence
    for key_save in coe_save_dict:
        append_dict.update({key_save: coe_save_value_list[i][0]})
        i += 1

    return append_dict


def count_csv_calibration_game_number_state_count():
    csv_calibration_dir = "/cs/oschulte/Galen/PycharmProjects/Sport-Analytic-NN" \
                          "/mc_three_prediction_simple_v_correct_dir/calibration_result" \
                          "/home_mc_three_cut_together_game_record_entire_feature_5_V3_Iter50_batch32_v_correct_sum_2017-8-05.csv"
    calibration_record_all_dict_list = []
    with open(csv_calibration_dir, 'rb') as f:
        reader = csv.reader(f)
        the_list = map(tuple, reader)
        calibration_item_name_list = the_list[0]
        for cali_num in range(0, len(calibration_item_name_list)):
            game_state_count_total = 0
            game_name_count = 0
            try:
                for item in the_list[1:]:
                    if item[cali_num] == '':
                        break
                    else:
                        game_name_state_count = (item[cali_num]).split(",")
                        game_state_count = game_name_state_count[1]
                        game_state_count_total = game_state_count_total + int(game_state_count)
                        game_name_count += 1
            except:
                print "wrong for debug"
            calibration_item_name_dict = ast.literal_eval(calibration_item_name_list[cali_num])
            calibration_item_name_dict.update(
                {"calibration_game_count": game_name_count, "game_state_count_total": game_state_count_total})
            calibration_record_all_dict_list.append(calibration_item_name_dict)

    return calibration_record_all_dict_list


def write_sequenced_csv(csv_name, data_record):
    with open(csv_name, 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(csv_sequence)
        for record in data_record:
            record_sequence_row = []
            for csv_item in csv_sequence:
                record_sequence_row.append(record.get(csv_item))
            writer.writerow(record_sequence_row)


def combine_dict_list(calibration_record_all_dict_list, csv_read_dict_list):
    find_count = 0
    for calibration_record_all_dict in calibration_record_all_dict_list:
        manpower = calibration_record_all_dict.get("MD")
        goal_diff = calibration_record_all_dict.get("GD")
        period = calibration_record_all_dict.get("P")
        game_count = calibration_record_all_dict.get("calibration_game_count")
        state_count = calibration_record_all_dict.get("game_state_count_total")

        for csv_read_dict_index in range(0, len(csv_read_dict_list)):
            if int(csv_read_dict_list[csv_read_dict_index].get("Manpower")) == manpower and int(
                    csv_read_dict_list[csv_read_dict_index].get("Goal Different")) == goal_diff and int(
                csv_read_dict_list[csv_read_dict_index].get("Period")) == period:
                csv_read_dict_list[csv_read_dict_index].update({
                    "game_count": game_count})
                csv_read_dict_list[csv_read_dict_index].update({
                    "state_count": state_count})
                find_count += 1

    if len(calibration_record_all_dict_list) == find_count and find_count == len(csv_read_dict_list):
        None
    else:
        raise ValueError("combine wrong!")
    return csv_read_dict_list


def compute_weighted_kld(csv_read_dict_list):
    weight_by_game = True

    game_count_sum = 0
    state_count_sum = 0
    append_dict = {}
    weighted_kld = 0
    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        game_count = float(csv_read_dict_list[csv_row_read_dict_index].get("game_count"))
        state_count = float(csv_read_dict_list[csv_row_read_dict_index].get("state_count"))
        game_count_sum += game_count
        state_count_sum += state_count

    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        kld = float(csv_read_dict_list[csv_row_read_dict_index].get("kld"))
        game_count = float(csv_read_dict_list[csv_row_read_dict_index].get("game_count"))
        state_count = float(csv_read_dict_list[csv_row_read_dict_index].get("state_count"))
        if weight_by_game:
            weighted_kld = weighted_kld + kld * (game_count / game_count_sum)
        else:
            weighted_kld = weighted_kld + kld * (state_count / state_count_sum)

    for key_dict in csv_read_dict_list[0].keys():
        append_dict.update({key_dict: ""})

    append_dict.update({"kld": weighted_kld})

    return append_dict


def compute_possibility_csv(csv_read_dict_list):
    for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
        predict_home = float(csv_read_dict_list[csv_row_read_dict_index].get("model_predict_home_value_record_average"))
        predict_away = float(csv_read_dict_list[csv_row_read_dict_index].get("model_predict_away_value_record_average"))
        predict_end = float(csv_read_dict_list[csv_row_read_dict_index].get("model_predict_end_value_record_average"))
        if predict_end < 0:
            predict_end = 0
        predict_poss_home = predict_home / (predict_home + predict_away + predict_end)
        predict_poss_away = predict_away / (predict_home + predict_away + predict_end)
        predict_poss_end = predict_end / (predict_home + predict_away + predict_end)
        csv_read_dict_list[csv_row_read_dict_index].update(
            {"model_poss_predict_home_value_record_average": predict_poss_home})
        csv_read_dict_list[csv_row_read_dict_index].update(
            {"model_poss_predict_away_value_record_average": predict_poss_away})
        csv_read_dict_list[csv_row_read_dict_index].update(
            {"model_poss_predict_end_value_record_average": predict_poss_end})

    return csv_read_dict_list


if __name__ == '__main__':
    calibration_record_all_dict_list = count_csv_calibration_game_number_state_count()
    calibration_result_dir = calibration_result_dirs[1]
    csv_read_dict_list = read_calibration_csv(calibration_result_dir)
    csv_read_dict_list = compute_abs_difference(csv_read_dict_list)
    csv_read_dict_list = compute_possibility_csv(csv_read_dict_list)
    csv_read_dict_list = compute_kld(csv_read_dict_list)
    csv_read_dict_list = combine_dict_list(calibration_record_all_dict_list, csv_read_dict_list)
    append_average_dict = compute_average(csv_read_dict_list)
    append_coe_dict = compute_correlated_coefficient(csv_read_dict_list)
    append_weighted_kld = compute_weighted_kld(csv_read_dict_list)
    csv_read_dict_list.append(append_average_dict)
    csv_read_dict_list.append(append_coe_dict)
    csv_read_dict_list.append(append_weighted_kld)
    write_sequenced_csv(calibration_result_dir, csv_read_dict_list)
