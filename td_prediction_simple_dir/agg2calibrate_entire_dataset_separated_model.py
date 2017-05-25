import os
import scipy.io as sio
import unicodedata
import ast

FEATURE_TYPE = 5
calibration_store_dir = "/media/gla68/Windows/Hockey-data/calibrate_all_feature_" + str(FEATURE_TYPE)
ISHOME = True
check_target = {"GD": 0, "MD": 0, "P": 1}


def agg2calibrate_model():
    model_predict_value_record = []
    calibration_value_record = []

    for calibration_dir_game in os.listdir(calibration_store_dir):
        for file_name in os.listdir(calibration_store_dir + "/" + calibration_dir_game):
            if "training_data_dict_all_value" in file_name:
                calibrate_value_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "training_data_dict_all_name" in file_name:
                calibrate_name_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "model_predict_home" in file_name:
                model_predict_home_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "model_predict_away" in file_name:
                model_predict_away_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "home_identifier" in file_name:
                home_identifier_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "summation_goal_home" in file_name:
                summation_goal_home_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            elif "summation_goal_away" in file_name:
                summation_goal_away_name = calibration_store_dir + "/" + calibration_dir_game + "/" + file_name
            else:
                continue

        calibrate_values = ((sio.loadmat(calibrate_value_name))["training_data_dict_all_value"]).tolist()
        calibrate_names = ((sio.loadmat(calibrate_name_name))["training_data_dict_all_name"]).tolist()
        model_predict_home = ((sio.loadmat(model_predict_home_name))["model_predict_home"]).tolist()
        model_predict_away = ((sio.loadmat(model_predict_away_name))["model_predict_away"]).tolist()
        home_identifier = (((sio.loadmat(home_identifier_name))["home_identifier"])[0]).tolist()
        summation_goal_home = ((sio.loadmat(summation_goal_home_name))["summation_goal_home"]).tolist()
        summation_goal_away = ((sio.loadmat(summation_goal_away_name))["summation_goal_away"]).tolist()

        for calibrate_name_index in range(0, len(calibrate_names)):
            calibrate_name = calibrate_names[calibrate_name_index]
            calibrate_name_str = unicodedata.normalize('NFKD', calibrate_name).encode('ascii', 'ignore')
            calibrate_name_dict = ast.literal_eval(calibrate_name_str)
            goal_diff = calibrate_name_dict.get("scoreDifferential")
            manpower_diff = calibrate_name_dict.get("Penalty")
            time_remain = calibrate_name_dict.get("time remained")
            if time_remain > 2400:
                period = 1.0
            elif time_remain > 1200:
                period = 2.0
            elif time_remain > 0:
                period = 3.0
            else:
                period = 4.0

            if float(check_target.get("GD")) == float(goal_diff) and float(check_target.get("MD")) == float(
                    manpower_diff) and float(check_target.get("P")) == float(period):
                if ISHOME and home_identifier[calibrate_name_index]:  # TODO delete home_identifier[calibrate_name_index]
                    # print "Found home"
                    model_predict_value_record.append(model_predict_home[calibrate_name_index])
                    calibration_value_record.append(float(summation_goal_home[calibrate_name_index]))
                elif not ISHOME and not home_identifier[calibrate_name_index]:  # TODO delete home_identifier[calibrate_name_index]
                    # print "Found away"
                    model_predict_value_record.append(model_predict_away[calibrate_name_index])
                    calibration_value_record.append(float(summation_goal_away[calibrate_name_index]))

    model_predict_value_average = float(sum(model_predict_value_record)) / len(model_predict_value_record)
    calibration_value_record_average = float(sum(calibration_value_record)) / len(calibration_value_record)

    print "model_predict_value_average: " + str(model_predict_value_average)
    print "calibration_value_record_average: " + str(calibration_value_record_average)


if __name__ == '__main__':
    agg2calibrate_model()
