import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import scipy.io as sio
import os
import json
import tensorflow as tf
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import compute_game_values, read_plot_model
from td_three_prediction_two_tower_lstm_v_correct_dir.compute_impact.player_impact import PlayerImpact
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name, \
    get_network_dir


def compute_values_for_all_games(config, data_store_dir, dir_all,
                                 model_number=None,
                                 league_name=None,
                                 write_values=True):
    sess_nn = tf.InteractiveSession()

    model_nn = td_prediction_tt_embed(
        feature_number=config.learn.feature_number,
        home_h_size=config.Arch.HomeTower.home_h_size,
        away_h_size=config.Arch.AwayTower.away_h_size,
        max_trace_length=config.learn.max_trace_length,
        learning_rate=config.learn.learning_rate,
        embed_size=config.learn.embed_size,
        output_layer_size=config.learn.output_layer_size,
        home_lstm_layer_num=config.Arch.HomeTower.lstm_layer_num,
        away_lstm_layer_num=config.Arch.AwayTower.lstm_layer_num,
        dense_layer_num=config.learn.dense_layer_num,
        apply_softmax=config.learn.apply_softmax
    )
    model_nn.initialize_ph()
    model_nn.build()
    model_nn.call()

    _, saved_network_path = get_network_dir(league_name, tt_lstm_config, train_msg='')

    data_name = get_data_name(config=config, league_name=league_name)
    if model_number is not None:
        saver = tf.train.Saver()
        model_path = saved_network_path + '/Soccer-game--{0}'.format(model_number)
        saver.restore(sess_nn, model_path)
        print 'successfully load data from' + model_path
    else:
        read_plot_model(model_path=saved_network_path, sess_nn=sess_nn)
    for game_name_dir in dir_all:
        game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir)
        model_value = compute_game_values(sess_nn=sess_nn,
                                          model=model_nn,
                                          data_store=data_store_dir,
                                          dir_game=game_name,
                                          config=tt_lstm_config,
                                          sport='Soccer')

        if write_values:
            model_value_json = {}
            for value_index in range(0, len(model_value)):
                model_value_json.update({value_index: {'home': float(model_value[value_index][0]),
                                                       'away': float(model_value[value_index][1]),
                                                       'end': float(model_value[value_index][2])}})

            game_store_dir = game_name_dir.split('.')[0]
            with open(data_store_dir + "/" + game_store_dir + "/" + data_name, 'w') as outfile:
                json.dump(model_value_json, outfile)

            # sio.savemat(data_store_dir + "/" + game_name_dir + "/" + data_name,
            #             {'model_value': np.asarray(model_value)})

    return data_name


def compute_impact(soccer_data_store_dir, game_data_dir,
                   data_name, player_id_name_pair_dir,
                   difference_type, league_name):
    PI = PlayerImpact(data_name=data_name, game_data_dir=game_data_dir,
                      model_data_store_dir=soccer_data_store_dir, difference_type=difference_type)
    dir_all = os.listdir(soccer_data_store_dir)
    for game_name_dir in dir_all:
        if difference_type == 'expected_goal':
            PI.aggregate_match_diff_values(game_name_dir, action_selected_list=['shot'])
        else:
            PI.aggregate_match_diff_values(game_name_dir, action_selected_list=['shot', 'goal'])  # 'shot', 'goal'
    PI.transfer2player_name_dict(player_id_name_pair_dir)
    PI.save_player_impact(league_name=league_name)


if __name__ == '__main__':
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
    player_id_name_pair_dir = '/Local-Scratch/PycharmProjects/Sport-Analytic-NN/' \
                              'td_three_prediction_two_tower_lstm_v_correct_dir/resource/soccer_id_name_pair.json'

    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config-v5.yaml"
    difference_type = 'back_difference_'
    soccer_dir_all = os.listdir(data_path)
    fine_tune_flag = False

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = '5'
    elif learning_rate == 1e-4:
        learning_rate_write = '4'
    elif learning_rate == 0.0005:
        learning_rate_write = '5_5'
    if fine_tune_flag:
        model_number = 24601
        league_name = "_English_Barclays_Premier_League"
        data_name = compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
                                                 dir_all=soccer_dir_all, model_number=model_number,
                                                 league_name=league_name)
    else:
        model_number = 2101  # 2101, 7201, 7801 ,10501 ,13501 ,15301 ,18301*, 20701*
        league_name = ''
        data_name = compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
                                                 dir_all=soccer_dir_all, model_number=model_number, league_name=None)
    # if difference_type == 'back_difference_':
    #     data_name = get_data_name(config=tt_lstm_config, league_name=league_name)
    # else:
    #     data_name = get_data_name(config=tt_lstm_config, league_name=league_name)
    compute_impact(data_name=data_name, game_data_dir=data_path, soccer_data_store_dir=soccer_data_store_dir,
                   player_id_name_pair_dir=player_id_name_pair_dir, difference_type=difference_type,
                   league_name=league_name)
