import sys

import datetime

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
from td_three_prediction_lstm_v_correct_dir import td_three_prediction_lstm_v_correct_cut_together
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_soccer_game_data_old


def get_data_name(config):
    data_name = "ijcai_model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}.json".format(
        str(tt_lstm_config.learn.feature_type),
        str(tt_lstm_config.learn.iterate_num),
        str(learning_rate_write),
        str(tt_lstm_config.learn.batch_size),
        str(tt_lstm_config.learn.max_trace_length),
        str(tt_lstm_config.learn.model_type))

    return data_name


def compute_values_for_all_games(config, data_store_dir, model_data_save_dir, dir_all):
    sess_nn = tf.InteractiveSession()
    td_three_prediction_lstm_v_correct_cut_together.FEATURE_NUMBER = 61
    model_nn = td_three_prediction_lstm_v_correct_cut_together.td_prediction_lstm_V4()
    # model_nn.initialize_ph()
    # model_nn.build()
    # model_nn.call()
    # saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
    #     tt_lstm_config.learn.feature_type) + "_batch" + str(
    #     tt_lstm_config.learn.batch_size) + "_iterate" + str(
    #     tt_lstm_config.learn.iterate_num) + "_lr" + str(
    #     tt_lstm_config.learn.learning_rate) + "_" + str(
    #     tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
    #     tt_lstm_config.learn.max_trace_length)
    #
    saved_network_path = "/cs/oschulte/Galen/miyunLuo/Code/Soccer_DRL/models/hybrid_sl_saved_NN/" \
                         "Scale-three-cut_together_saved_networks_feature5_batch32_iterate30_lr0.0001_v4_MaxTL10"

    data_name = get_data_name(config)

    read_plot_model(model_path=saved_network_path, sess_nn=sess_nn)
    for game_name_dir in dir_all:
        game_name = game_name_dir.split('.')[0]
        # game_time_all = get_game_time(data_path, game_name_dir)
        state_trace_length, state_input, reward = get_soccer_game_data_old(data_store=data_store_dir,
                                                                                  dir_game=game_name,
                                                                                  config=config, )

        [model_value] = sess_nn.run([model_nn.read_out],
                                    feed_dict={model_nn.trace_lengths: state_trace_length,
                                               model_nn.rnn_input: state_input
                                               })
        model_value_json = {}
        for value_index in range(0, len(model_value)):
            model_value_json.update({value_index: {'home': float(model_value[value_index][0]),
                                                   'away': float(model_value[value_index][1]),
                                                   'end': float(model_value[value_index][2])}})

        game_store_dir = game_name_dir.split('.')[0]
        with open(model_data_save_dir + "/" + game_store_dir + "/" + data_name, 'w') as outfile:
            json.dump(model_value_json, outfile)

            # sio.savemat(data_store_dir + "/" + game_name_dir + "/" + data_name,
            #             {'model_value': np.asarray(model_value)})
    return data_name


def compute_impact(soccer_data_store_dir, game_data_dir, data_name, player_id_name_pair_dir):
    PI = PlayerImpact(data_name=data_name, game_data_dir=game_data_dir, model_data_store_dir=soccer_data_store_dir)
    dir_all = os.listdir(soccer_data_store_dir)
    for game_name_dir in dir_all:
        PI.aggregate_match_diff_values(game_name_dir)
    PI.transfer2player_name_dict(player_id_name_pair_dir)
    PI.save_player_impact('ijcai_soccer_player_GIM_{0}.json'.format(datetime.date.today().strftime("%Y%B%d")))


if __name__ == '__main__':
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    soccer_data_store_dir = "/cs/oschulte/miyunLuo/Documents/Data/Soccer_for_DRL"
    model_data_save_dir = "/cs/oschulte/Galen/Soccer-data/"
    player_id_name_pair_dir = '/Local-Scratch/PycharmProjects/Sport-Analytic-NN/td_three_prediction_two_tower_lstm_v_correct_dir/resource/soccer_id_name_pair.json'

    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config.yaml"
    soccer_dir_all = os.listdir(data_path)

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4

    # compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
    #                             dir_all=soccer_dir_all, model_data_save_dir=model_data_save_dir)
    data_name = get_data_name(config=tt_lstm_config)
    compute_impact(data_name=data_name, game_data_dir=data_path, soccer_data_store_dir=model_data_save_dir,
                   player_id_name_pair_dir=player_id_name_pair_dir)
