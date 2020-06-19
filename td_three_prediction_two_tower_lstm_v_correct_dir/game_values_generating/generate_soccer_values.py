import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')
import json
import os
import scipy.io as sio
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools

from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from soccer_data_config import interested_compute_features, interested_raw_features, action_all
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name, \
    handle_trace_length


def gather_values_by_games(model_data_store_dir, dir_game, data_name, game_data_store, write_file, target_action=None):
    for file_name in os.listdir(model_data_store_dir + "/" + dir_game):
        # print file_name
        if file_name == data_name:
            model_data_name = model_data_store_dir + "/" + dir_game + "/" + file_name
            with open(model_data_name) as f:
                model_data = json.load(f)
        elif file_name.startswith("home_away"):
            home_identifier_name = model_data_store_dir + "/" + dir_game + "/" + file_name
            home_identifier = (sio.loadmat(home_identifier_name))["home_away"][0]
        else:
            continue

    state_input_name = 'state_{0}.mat'.format(dir_game)
    state_input = sio.loadmat(model_data_store_dir + "/" + dir_game + "/" + state_input_name)['state']
    state_trace_length_name = 'trace_{0}.mat'.format(dir_game)
    state_trace_length = sio.loadmat(
        model_data_store_dir + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
    state_trace_length = handle_trace_length(state_trace_length)
    actions = tools.read_feature_within_events(data_path=game_data_store, directory=dir_game + '.json',
                                               feature_name='action')
    # playerIds = tools.read_feature_within_events(data_path=game_data_dir, directory=dir_game + '.json',
    #                                              feature_name='playerId')

    for event_index in range(0, len(actions)):
        ishome = home_identifier[event_index]

        if target_action not in actions[event_index]:
            continue
        if not ishome:
            continue

        model_value = model_data[str(event_index)]
        if event_index - 1 >= 0:  # define model pre
            if actions[event_index - 1] == "goal":
                model_value_pre = model_data[str(event_index)]  # the goal cancel out here, just as we cut the game
            else:
                model_value_pre = model_data[str(event_index - 1)]
        else:
            model_value_pre = model_data[str(event_index)]
        # if event_index + 1 < len(actions):  # define model next
        #     if actions[event_index + 1] == "goal":
        #         model_value_nex = model_data[str(event_index)]
        #     else:
        #         model_value_nex = model_data[str(event_index + 1)]
        # else:
        #     model_value_nex = model_data[str(event_index)]
        # ishome = home_identifier[event_index]
        home_model_value = model_value['home']
        away_model_value = model_value['away']
        # end_model_value = abs(model_value[2])
        home_model_value_pre = model_value_pre['home']
        away_model_value_pre = model_value_pre['away']
        # end_model_value_pre = abs(model_value_pre[2])
        if ishome:
            impact_value = (home_model_value - home_model_value_pre)

        else:
            impact_value = (away_model_value - away_model_value_pre)

        if event_index + 1 < len(actions):
            if 'goal' in actions[event_index + 1]:
                print(event_index)
                print('{0} value is {1}, {4} value is {2}, {0} impact is {3}'.format(target_action,
                                                                                      str(model_value['home']),
                                                                                      str(model_data[
                                                                                              str(event_index + 1)][
                                                                                              'home']),
                                                                                      str(impact_value),
                                                                                     actions[event_index + 1]
                                                                                     ))

        write_file.write(str(impact_value))
        write_file.write(',' + str(model_value['home']))
        tl = state_trace_length[event_index] if state_trace_length[event_index] <= 10 else 10
        if tl < 10:
            for j in range(0, 10 - tl):
                for zero_index in range(0, len(interested_raw_features + interested_compute_features + action_all)+2):
                    write_file.write(',0')
        for i in range(0, tl, 1):
            for state_value in state_input[event_index][i][
                               0:len(interested_raw_features) + len(interested_compute_features)]:
                write_file.write(',' + str(state_value))
            player_action = actions[event_index - tl + 1 + i]
            for action_name in action_all:
                if action_name == player_action:
                    write_file.write(',1')
                else:
                    write_file.write(',0')
            if home_identifier[event_index - tl + 1 + i]:
                write_file.write(',1,0')
            else:
                write_file.write(',0,1')

        write_file.write('\n')


def gather_all_values(data_path, model_data_store_dir, data_name, target_action):
    soccer_dir_all = os.listdir(data_path)
    with open('./generated_values_store/{0}_impact_Q_states_features_history_soccer.csv'.format(target_action),
              'w') as write_file:
        items_name_all = interested_raw_features + interested_compute_features + action_all
        write_file.write('impact')
        write_file.write(',Q')
        for i in range(9, -1, -1):
            for item in items_name_all:
                write_file.write(',' + item + str(i))
            write_file.write(',' + 'home' + str(i))
            write_file.write(',' + 'away' + str(i))
        write_file.write('\n')
        for game_name_dir in soccer_dir_all:
            print('working on game {0}'.format(game_name_dir))
            dir_game = game_name_dir.split('.')[0]
            gather_values_by_games(model_data_store_dir, dir_game, data_name, data_path, write_file, target_action)


if __name__ == '__main__':
    test_flag = False
    if not test_flag:
        data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
        model_data_save_dir = "/cs/oschulte/Galen/Soccer-data/"
    else:
        data_path = "/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/"
        model_data_save_dir = "/Users/liu/Desktop/soccer-data-sample/Soccer-data/"

    player_id_name_pair_dir = '/Local-Scratch/PycharmProjects/Sport-Analytic-NN/' \
                              'td_three_prediction_two_tower_lstm_v_correct_dir/resource/soccer_id_name_pair.json'

    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config-v5.yaml"
    # difference_type = 'back_difference_'
    target_action = 'pass'
    soccer_dir_all = os.listdir(data_path)
    fine_tune_flag = False

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    data_name = get_data_name(config=tt_lstm_config)
    gather_all_values(data_path, model_data_save_dir, data_name, target_action)
