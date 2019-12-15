import os
import json
import random

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig

from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name


def gather_impact_values(soccer_data_store_path, tt_lstm_config, game_data_path, action_selected_list):
    data_name = get_data_name(config=tt_lstm_config, league_name='')
    model_data = None
    home_identifier = None
    dir_all = os.listdir(soccer_data_store_path)

    impact_time_remain = []

    random_game_indices = random.sample(range(len(dir_all)), 10)

    # for game_name_dir in np.asarray(dir_all)[random_game_indices]:
    for game_name_dir in np.asarray(dir_all)[:10]:
        for file_name in os.listdir(soccer_data_store_path + "/" + game_name_dir):
            # print file_name
            if file_name == data_name:
                model_data_name = soccer_data_store_path + "/" + game_name_dir + "/" + file_name
                with open(model_data_name) as f:
                    model_data = json.load(f)
            elif file_name.startswith("home_away"):
                home_identifier_name = soccer_data_store_path + "/" + game_name_dir + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_away"][0]
        assert model_data is not None
        assert home_identifier is not None
        actions = tools.read_feature_within_events(data_path=game_data_path, directory=game_name_dir + '.json',
                                                   feature_name='action')
        game_time = tools.read_feature_within_events(data_path=game_data_path, directory=game_name_dir + '.json',
                                                     feature_name='gameTimeRemain')
        skip_number = 0
        for event_Index in range(0, len(model_data)):
            if action_selected_list is not None:
                continue_flag = False if len(action_selected_list) == 0 else True
                for f_action in action_selected_list:
                    if f_action in actions[event_Index]:
                        # print action
                        continue_flag = False
                if continue_flag:
                    skip_number += 1
                    continue
            model_value = model_data[str(event_Index)]
            if event_Index - 1 >= 0:  # define model pre
                if actions[event_Index - 1] == "goal":
                    model_value_pre = model_data[str(event_Index)]  # the goal cancel out here, just as we cut the game
                else:
                    model_value_pre = model_data[str(event_Index - 1)]
            else:
                model_value_pre = model_data[str(event_Index)]
            ishome = home_identifier[event_Index]
            home_model_value = model_value['home']
            away_model_value = model_value['away']
            home_model_value_pre = model_value_pre['home']
            away_model_value_pre = model_value_pre['away']

            if ishome:
                impact = (home_model_value - home_model_value_pre)
            else:
                impact = (away_model_value - away_model_value_pre)

            impact_time_remain.append([impact, game_time[event_Index]])

    impact_time_remain = np.asarray(impact_time_remain)
    return impact_time_remain[impact_time_remain[:, 1].argsort()]


def plot_impact_partial_dependency(impact_features_array):
    plt.figure(figsize=(9, 6))
    plt.scatter(impact_features_array[:, 1], impact_features_array[:, 0])
    # plt.figure(figsize=(9, 6))
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Game Time Remain', fontsize=15)
    plt.ylabel('Impact Values', fontsize=15)
    plt.ylim([-0.10, 0.10])
    plt.show()


if __name__ == '__main__':
    soccer_data_store_path = "/cs/oschulte/Galen/Soccer-data/"
    game_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    action_selected_list = ['pass']
    tt_lstm_config_path = "../soccer-config-v5.yaml"
    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    impact_features_array = gather_impact_values(soccer_data_store_path,
                                                 tt_lstm_config,
                                                 game_data_path,
                                                 action_selected_list)
    plot_impact_partial_dependency(impact_features_array)
