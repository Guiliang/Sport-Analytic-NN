import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import scipy.io as sio
import os
import tensorflow as tf
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import find_game_dir, \
    find_soccer_game_dir_by_team, get_team_name, get_action_position
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import plot_game_value
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.print_tools import print_mark_info
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import normalize_data, \
    get_game_time, \
    get_network_dir
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import compute_game_values, read_plot_model

if __name__ == '__main__':
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config-v5.yaml"
    # dir_all = os.listdir(data_path)
    dir_all = [ '922081.json']
    # find_soccer_game_dir_by_team(dir_all, data_path)
    # game_name_dir = find_game_dir(dir_all, data_path, target_game_id, sports='Soccer')
    # game_name_dir = '922075'

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4

    sess_nn = tf.InteractiveSession()

    model_nn = td_prediction_tt_embed(
        feature_number=tt_lstm_config.learn.feature_number,
        home_h_size=tt_lstm_config.Arch.HomeTower.home_h_size,
        away_h_size=tt_lstm_config.Arch.AwayTower.away_h_size,
        max_trace_length=tt_lstm_config.learn.max_trace_length,
        learning_rate=tt_lstm_config.learn.learning_rate,
        embed_size=tt_lstm_config.learn.embed_size,
        output_layer_size=tt_lstm_config.learn.output_layer_size,
        home_lstm_layer_num=tt_lstm_config.Arch.HomeTower.lstm_layer_num,
        away_lstm_layer_num=tt_lstm_config.Arch.AwayTower.lstm_layer_num,
        dense_layer_num=tt_lstm_config.learn.dense_layer_num,
        apply_softmax=tt_lstm_config.learn.apply_softmax
    )
    model_nn.initialize_ph()
    model_nn.build()
    model_nn.call()

    _, saved_network_path = get_network_dir('', tt_lstm_config, train_msg='')
    # data_store = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
    # tt_lstm_config.learn.feature_type) + "-scale-neg_reward" + tt_lstm_config.learn.if_correct_velocity + "_length-dynamic"
    read_plot_model(model_path=saved_network_path, sess_nn=sess_nn)
    for game_name_dir in dir_all:
        game_name = game_name_dir.split('.')[0]
        game_time_all = get_game_time(data_path, game_name_dir)
        home_team, away_team, date = get_team_name(data_path, game_name_dir)
        action_xy_return = get_action_position(data_path, game_name_dir)

        if not 'Fulham' in away_team and not 'Fulham' in home_team:
            continue

        print 'game:{3}, home_team:{0}, away_team:{1}, date:{2} \n'.format(home_team, away_team, date, game_name_dir)

        game_value = compute_game_values(sess_nn=sess_nn,
                                         model=model_nn,
                                         data_store=data_store_dir,
                                         dir_game=game_name,
                                         config=tt_lstm_config,
                                         sport='Soccer')
        for value_index in range(0, len(game_value)):
            if game_value[value_index][0] > 0.5 or game_value[value_index][1] > 0.5:
                print '{0} {1}'.format(str(value_index), str(action_xy_return[value_index]))

        save_image_name = './soccer-image/{0} v.s. {1} value_ticker.png'.format(home_team, away_team)

        home_max_index, away_max_index, home_maxs, away_maxs = plot_game_value(game_value=game_value,
                                                                               game_time_all=game_time_all,
                                                                               save_image_name=save_image_name,
                                                                               normalize_data=normalize_data,
                                                                               home_team=home_team,
                                                                               away_team=away_team)
        # break
        # print_mark_info(data_store_dir, game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs)
