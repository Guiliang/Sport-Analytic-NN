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


def get_data_name(config):
    data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
        str(tt_lstm_config.learn.feature_type),
        str(tt_lstm_config.learn.iterate_num),
        str(learning_rate_write),
        str(tt_lstm_config.learn.batch_size),
        str(tt_lstm_config.learn.max_trace_length),
        str(tt_lstm_config.learn.model_type))

    return data_name


def compute_ranking(soccer_data_store_dir, game_data_dir, data_name, player_summary_info_dir, rank_store_file_dir,
                    action_selected):
    PI = PlayerImpact(data_name=data_name, game_data_dir=game_data_dir, model_data_store_dir=soccer_data_store_dir)
    dir_all = os.listdir(soccer_data_store_dir)
    for game_name_dir in dir_all:
        PI.aggregate_match_diff_values(game_name_dir, action_selected=action_selected)
    # PI.transfer2player_name_dict(player_id_name_pair_dir)
    PI.rank_player_by_impact(player_summary_info_dir, rank_store_file_dir)
    # PI.save_player_impact()


if __name__ == '__main__':
    action_selected = None
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
    player_id_name_pair_dir = '/Local-Scratch/PycharmProjects/Sport-Analytic-NN/' \
                              'td_three_prediction_two_tower_lstm_v_correct_dir/resource/soccer_id_name_pair.json'
    player_summary_info_dir = '../resource/Soccer_summary_info.csv'
    rank_store_file_dir = './player_ranking/soccer_player_ranking'

    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config.yaml"
    soccer_dir_all = os.listdir(data_path)

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4

    # data_name = compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
    #                                          dir_all=soccer_dir_all)
    data_name = get_data_name(config=tt_lstm_config)
    compute_ranking(data_name=data_name, game_data_dir=data_path, soccer_data_store_dir=soccer_data_store_dir,
                    player_summary_info_dir=player_summary_info_dir, rank_store_file_dir=rank_store_file_dir,
                    action_selected=action_selected)
