import csv
import os
import scipy.io as sio
import unicodedata
import ast

FEATURE_TYPE = 5
MODEL_TYPE = "V8"
ITERATE_NUM = 75
BATCH_SIZE = 32
calibration_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Test100-Hockey-Training-All-feature{0}-scale-neg_reward".format(str(FEATURE_TYPE))
# calibration_store_dir = "/cs/oschulte/Galen/Hockey-data/td_calibrate_all_feature_5_2017-6-01"
# store_name = ""


save_csv_name = "td_testing_cut_calibration_entire_feature_" + str(
    FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(
    BATCH_SIZE) + "_sum_2017-6-29.csv"
save_game_csv_name = "td_testing_cut_game_record_entire_feature_" + str(
    FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(
    BATCH_SIZE) + "_sum_2017-6-29.csv"


data_name = "model_cut_predict_feature_" + str(
    FEATURE_TYPE) + "_" + MODEL_TYPE + "_Iter" + str(ITERATE_NUM) + "_batch" + str(BATCH_SIZE)


def agg2calibrate_model(check_target):
    model_predict_value_record = []
    calibration_value_record = []
    found_game_name_record = []

    for calibration_dir_game in os.listdir(calibration_store_dir):
        game_found_flag = False
        for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
            if "training_data_dict_all_name" in file_name:
                calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif data_name in file_name:
                model_predict_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "summation_cut_goal_home" in file_name:
                summation_cut_goal_home_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "summation_cut_goal_away" in file_name:
                summation_cut_goal_away_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            else:
                continue

        calibrate_names = ((sio.loadmat(calibrate_name_name))["training_data_dict_all_name"]).tolist()
        model_predict = ((sio.loadmat(model_predict_name))[data_name]).tolist()
        summation_cut_goal_home = (((sio.loadmat(summation_cut_goal_home_name))["summation_cut_goal_home"]).tolist())[0]
        summation_cut_goal_away = (((sio.loadmat(summation_cut_goal_away_name))["summation_cut_goal_away"]).tolist())[0]

        # calibration_value_game_record = []
        # model_predict_value_game_record = []

        if not len(calibrate_names) == len(model_predict) and len(model_predict) == len(
                summation_cut_goal_home) and len(summation_cut_goal_away) == len(
                summation_cut_goal_home):
            raise ValueError("lens of data don't consist")

        for calibrate_name_index in range(0, len(calibrate_names)):
            calibrate_name = calibrate_names[calibrate_name_index]
            calibrate_name_str = unicodedata.normalize('NFKD', calibrate_name).encode('ascii', 'ignore')
            calibrate_name_dict = ast.literal_eval(calibrate_name_str)
            goal_diff = calibrate_name_dict.get("scoreDifferential")
            manpower_diff = calibrate_name_dict.get("Penalty")
            time_remain = calibrate_name_dict.get("time remained")
            home = calibrate_name_dict.get("home")
            if FEATURE_TYPE == 5:
                if time_remain > 2400:
                    period = 1.0
                elif time_remain > 1200:
                    period = 2.0
                elif time_remain > 0:
                    period = 3.0
                else:
                    period = 4.0
            elif FEATURE_TYPE == 9:
                period = calibrate_name_dict.get("period")
            else:
                raise ValueError("Unsupported FEATURE_TYPE:{}".format(str(FEATURE_TYPE)))

            if float(check_target.get("GD")) == float(goal_diff) and float(check_target.get("MD")) == float(
                    manpower_diff) and float(check_target.get("P")) == float(period):
                game_found_flag = True
                # if ISHOME and home_identifier[calibrate_name_index]:  # TODO delete home_identifier[calibrate_name_index]
                try:
                    model_predict_value_record.append((model_predict[calibrate_name_index])[0])
                    calibration_value_record.append(float(summation_cut_goal_home[calibrate_name_index])+float(summation_cut_goal_away[calibrate_name_index]))
                except:
                    print "error"

        if game_found_flag:
            found_game_name_record.append(calibration_dir_game)

        # try:
        #     calibration_value_game_average = float(sum(calibration_value_record)) / len(
        #         calibration_value_record)
        #     calibration_value_record.append(calibration_value_game_average)
        # except:
        #     calibration_value_game_average = 0.0
        #
        # try:
        #     model_predict_value_game_average = float(sum(model_predict_value_record)) / len(
        #         model_predict_value_record)
        #     model_predict_value_record.append(model_predict_value_game_average)
        # except:
        #     model_predict_value_game_average = 0.0

    # model_predict_value_record = [value[0] for value in model_predict_value_record]
    # calibration_value_record = [value[0] for value in calibration_value_record]
    try:
        model_predict_value_average = float(sum(model_predict_value_record)) / len(model_predict_value_record)
    except:
        model_predict_value_average = 0

    try:
        calibration_value_record_average = float(sum(calibration_value_record)) / len(calibration_value_record)
    except:
        calibration_value_record_average = 0

    print "model_predict_value_average with " + str(check_target) + " is " + str(model_predict_value_average)
    print "calibration_value_record_average with " + str(check_target) + " is " + str(calibration_value_record_average)

    return model_predict_value_average, calibration_value_record_average, found_game_name_record


def start_calibration():
    store_dict_list = []
    game_name_dict = {}
    for goal_diff in [-3, -2, -1, 0, 1, 2, 3]:
        # for goal_diff in [0]:
        for manpower in [-1, 0, 1]:
            # for manpower in [0]:
            for period in [1, 2, 3]:
                store_dict = {"Goal Different": goal_diff, "Manpower": manpower, "Period": period}
                check_target = {"GD": goal_diff, "MD": manpower, "P": period}
                model_predict_average, calibration_average, found_game_list = agg2calibrate_model(
                    check_target=check_target)
                store_dict.update({"model_predict_average": model_predict_average})
                store_dict.update({"calibration_averagev": calibration_average})
                store_dict_list.append(store_dict)
                game_name_dict.update({str(check_target): found_game_list})

    write_csv(save_csv_name, store_dict_list)
    game_name_pad_list = padding_dict(game_name_dict)
    write_csv(save_game_csv_name, game_name_pad_list)


def write_csv(csv_name, data_record):
    with open(csv_name, 'w') as csvfile:
        fieldnames = (data_record[0]).keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in data_record:
            writer.writerow(record)


def padding_dict(dict_to_pad):
    # dict_to_pad = {}
    values_list = dict_to_pad.values()

    max_value_len = 0
    for values in values_list:
        if len(values) > max_value_len:
            max_value_len = len(values)

    record_dict_list = []
    for index in range(0, max_value_len):
        record_dict = {}
        for key in dict_to_pad.keys():
            try:
                record_dict.update({key: (dict_to_pad.get(key))[index]})
            except:
                record_dict.update({key: ""})

        record_dict_list.append(record_dict)
    return record_dict_list


if __name__ == '__main__':
    start_calibration()
