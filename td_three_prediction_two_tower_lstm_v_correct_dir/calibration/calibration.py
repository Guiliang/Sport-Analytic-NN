import os
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import read_features_within_events, \
    read_feature_within_events
import json
import math
import datetime


class Calibration:
    def __init__(self, bins, data_path, calibration_features,
                 tt_lstm_config_path, soccer_data_store_dir,
                 apply_old, apply_difference, focus_actions_list=[]):
        self.bins = bins
        # self.bins_names = bins.keys()
        self.apply_old = apply_old
        self.apply_difference = apply_difference
        self.data_path = data_path
        self.calibration_features = calibration_features
        if self.apply_difference:
            self.calibration_values_all_dict = {'all': {'cali_sum': [0], 'model_sum': [0], 'number': 0}}
        else:
            self.calibration_values_all_dict = {'all': {'cali_sum': [0, 0, 0], 'model_sum': [0, 0, 0], 'number': 0}}
        self.soccer_data_store_dir = soccer_data_store_dir
        self.tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
        self.focus_actions_list = focus_actions_list
        if self.apply_difference:
            self.save_calibration_dir = './calibration_results/difference-calibration-{0}-{1}.txt'. \
                format(str(self.focus_actions_list), datetime.date.today().strftime("%Y%B%d"))
        else:
            self.save_calibration_dir = './calibration_results/calibration-{0}-{1}.txt'. \
                format(str(self.focus_actions_list), datetime.date.today().strftime("%Y%B%d"))
        self.save_calibration_file = open(self.save_calibration_dir, 'w')
        if apply_difference:
            self.teams = ['home-away']
        else:
            self.teams = ['home', 'away', 'end']
            # learning_rate = tt_lstm_config.learn.learning_rate
            # pass

    def __del__(self):
        print 'ending calibration'
        print self.save_calibration_file.close()

    def recursive2construct(self, store_dict_str, depth):
        feature_number = len(self.calibration_features)
        if depth >= feature_number:
            if self.apply_difference:
                self.calibration_values_all_dict.update({store_dict_str: {'cali_sum': [0],
                                                                          'model_sum': [0],
                                                                          'number': 0}})
            else:
                self.calibration_values_all_dict.update({store_dict_str: {'cali_sum': [0, 0, 0],
                                                                          'model_sum': [0, 0, 0],
                                                                          'number': 0}})
            return
        calibration_feature = self.calibration_features[depth]
        feature_range = self.bins.get(calibration_feature).get('range')
        for value in feature_range:
            # store_dict_str = '-' + store_dict_str if len(store_dict_str) > 0 else store_dict_str
            store_dict_str_update = store_dict_str + calibration_feature + '_' + str(value) + '-'
            self.recursive2construct(store_dict_str_update, depth + 1)

    def construct_bin_dicts(self):
        """create calibration dict"""
        self.recursive2construct('', 0)

    def compute_calibration_values(self, actions_team_all):
        """ground truth value for each game"""
        pre_index = 0
        cali_home = [0] * len(actions_team_all)
        cali_away = [0] * len(actions_team_all)
        cali_end = [0] * len(actions_team_all)
        for index in range(0, len(actions_team_all)):
            actions_team = actions_team_all[index]
            if actions_team['action'] == 'goal':
                if actions_team['home_away'] == 'H':
                    cali_home[pre_index:index] = [1] * (index - pre_index)
                elif actions_team['home_away'] == 'A':
                    cali_away[pre_index:index] = [1] * (index - pre_index)
                pre_index = index
            if index == len(actions_team_all) - 1:
                cali_end[pre_index:index] = [1] * (index - pre_index)
        return zip(cali_home, cali_away, cali_end)

    def obtain_model_prediction(self, directory):
        """model predicted value for each game"""

        if self.apply_old:
            old_string = 'ijcai_'
        else:
            old_string = ''

        if self.tt_lstm_config.learn.merge_tower:
            merge_model_msg = '_merge'
        else:
            merge_model_msg = ''

        learning_rate = self.tt_lstm_config.learn.learning_rate
        if learning_rate == 1e-5:
            learning_rate_write = '5'
        elif learning_rate == 1e-4:
            learning_rate_write = '4'
        elif learning_rate == 0.0005:
            learning_rate_write = '5_5'
        data_name = "{6}model{7}_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}.json".format(
            str(self.tt_lstm_config.learn.feature_type),
            str(self.tt_lstm_config.learn.iterate_num),
            str(learning_rate_write),
            str(self.tt_lstm_config.learn.batch_size),
            str(self.tt_lstm_config.learn.max_trace_length),
            str(self.tt_lstm_config.learn.model_type),
            str(old_string),
            merge_model_msg
        )
        # directory = '917811'
        print('model name is {0}'.format(data_name))
        with open(self.soccer_data_store_dir + "/" + directory + "/" + data_name) as outfile:
            model_output = json.load(outfile)

        return model_output

    def aggregate_calibration_values(self):
        """update calibration dict by each game"""
        dir_all = os.listdir(self.data_path)
        # dir_all = ['919069.json']  # TODO: test
        # self.data_path = '/Users/liu/Desktop/'
        for json_dir in dir_all:
            features_all = []
            for calibration_feature in self.calibration_features:
                features = self.bins.get(calibration_feature).get('feature_name')
                if isinstance(features, str):
                    features_all.append(features)
                else:
                    for feature in features:
                        features_all.append(feature)

            model_values = self.obtain_model_prediction(directory=json_dir.split('.')[0])
            # model_values = [[1, 0, 0]] * 1519  # TODO: test
            actions_team_all = read_features_within_events(feature_name_list=['action', 'home_away'],
                                                           data_path=self.data_path, directory=json_dir)
            calibration_values = self.compute_calibration_values(actions_team_all)

            features_values_dict_all = read_features_within_events(feature_name_list=features_all,
                                                                   data_path=self.data_path,
                                                                   directory=json_dir)
            for index in range(0, len(features_values_dict_all)):

                action = actions_team_all[index]['action']  # find the action we focus
                continue_flag = False if len(self.focus_actions_list) == 0 else True
                for f_action in self.focus_actions_list:
                    if f_action in action:
                        # print action
                        continue_flag = False
                if continue_flag:
                    continue

                features_values_dict = features_values_dict_all[index]
                cali_dict_str = ''
                for calibration_feature in self.calibration_features:
                    if calibration_feature == 'period':
                        min = features_values_dict.get('min')
                        sec = features_values_dict.get('sec')
                        if min <= 45:
                            value = 1
                        else:
                            value = 2
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    elif calibration_feature == 'score_differential':
                        value = features_values_dict.get('scoreDiff')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    elif calibration_feature == 'pitch':
                        xccord = features_values_dict.get('x')
                        if xccord <= 50:
                            value = 'left'
                        else:
                            value = 'right'
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + value + '-'

                    elif calibration_feature == 'manpower':
                        value = features_values_dict.get('manPower')
                        cali_dict_str = cali_dict_str + calibration_feature + '_' + str(value) + '-'
                    else:
                        raise ValueError('unknown feature' + calibration_feature)

                calibration_value = calibration_values[index]
                model_value = model_values[str(index)]

                cali_bin_info = self.calibration_values_all_dict.get(cali_dict_str)
                # print cali_dict_str
                assert cali_bin_info is not None
                cali_sum = cali_bin_info.get('cali_sum')
                model_sum = cali_bin_info.get('model_sum')
                number = cali_bin_info.get('number')
                number += 1
                if self.apply_difference:
                    cali_sum[0] = cali_sum[0] + (calibration_value[0] - calibration_value[1])
                    model_sum[0] = model_sum[0] + (model_value['home'] - model_value['away'])
                else:
                    for i in range(len(self.teams)):  # [home, away,end]
                        cali_sum[i] = cali_sum[i] + calibration_value[i]
                        model_sum[i] = model_sum[i] + model_value[self.teams[i]]

                self.calibration_values_all_dict.update({cali_dict_str: {'cali_sum': cali_sum,
                                                                         'model_sum': model_sum,
                                                                         'number': number}})

                cali_bin_info = self.calibration_values_all_dict.get('all')
                cali_sum = cali_bin_info.get('cali_sum')
                model_sum = cali_bin_info.get('model_sum')
                number = cali_bin_info.get('number')
                number += 1
                if self.apply_difference:
                    cali_sum[0] = cali_sum[0] + (calibration_value[0] - calibration_value[1])
                    model_sum[0] = model_sum[0] + (model_value['home'] - model_value['away'])
                else:
                    for i in range(len(self.teams)):  # [home, away,end]
                        cali_sum[i] = cali_sum[i] + calibration_value[i]
                        model_sum[i] = model_sum[i] + model_value[self.teams[i]]

                self.calibration_values_all_dict.update({'all': {'cali_sum': cali_sum,
                                                                 'model_sum': model_sum,
                                                                 'number': number}})

                # break

    def compute_distance(self):
        cali_dict_strs = self.calibration_values_all_dict.keys()
        for cali_dict_str in cali_dict_strs:
            cali_bin_info = self.calibration_values_all_dict.get(cali_dict_str)
            kld_sum = 0
            mae_sum = 0
            if cali_bin_info['number'] == 0:
                print "number of bin {0} is 0".format(cali_dict_str)
                continue
            cali_record_dict = 'Bin:' + cali_dict_str
            for i in range(len(self.teams)):  # [home, away,end]
                cali_prob = float(cali_bin_info['cali_sum'][i]) / cali_bin_info['number']
                model_prob = float(cali_bin_info['model_sum'][i]) / cali_bin_info['number']
                cali_record_dict += '\t{0}_number'.format(self.teams[i]) + ":" + str(cali_bin_info['number'])
                cali_record_dict += '\t{0}_cali'.format(self.teams[i]) + ":" + str(cali_prob)
                cali_record_dict += '\t{0}_model'.format(self.teams[i]) + ":" + str(model_prob)
                model_prob = model_prob + 1e-10
                cali_prob = cali_prob + 1e-10
                try:
                    kld = cali_prob * math.log(cali_prob / model_prob)
                except:
                    print 'kld is ' + str(cali_prob / model_prob)
                    kld = 0
                kld_sum += kld
                ae = abs(cali_prob - model_prob)
                mae_sum = mae_sum + ae
            cali_record_dict += '\tkld:' + str(kld_sum)
            cali_record_dict += '\tmae:' + str(float(mae_sum) / len(self.teams))
            self.save_calibration_file.write(str(cali_record_dict) + '\n')
