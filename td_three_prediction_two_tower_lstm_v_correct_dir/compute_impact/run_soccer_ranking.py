import csv
import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import scipy.io as sio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import tensorflow as tf
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import compute_game_values, read_plot_model
from td_three_prediction_two_tower_lstm_v_correct_dir.compute_impact.player_impact import PlayerImpact
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name


def compute_ranking(soccer_data_store_dir, game_data_dir,
                    data_name, player_summary_info_dir,
                    action_selected_lists, game_info_all,
                    league_number):
    PI = PlayerImpact(data_name=data_name, game_data_dir=game_data_dir,
                      model_data_store_dir=soccer_data_store_dir)
    dir_all = os.listdir(soccer_data_store_dir)
    for action_selected_list in action_selected_lists:
        print ("working on action {0}".format(str(action_selected_list)))
        if action_selected_list is not None:
            write_file = open('./player_ranking/soccer_player_ranking_{0}{1}'.format(action_selected_list[0],
                                                                                     league_name), 'w')
        else:
            write_file = open('./player_ranking/soccer_player_ranking_all{0}'.format(league_name), 'w')
        for game_name_dir in dir_all:
            if game_name_dir == '.DS_Store':
                continue
            PI.aggregate_match_diff_values(game_name_dir,
                                           action_selected_list=action_selected_list,
                                           league_id=league_number)
        # PI.transfer2player_name_dict(player_id_name_pair_dir)
        action_selected = action_selected_list[0] if action_selected_list is not None else None
        PI.rank_player_by_impact(player_summary_info_dir,
                                 write_file=write_file,
                                 action_selected=action_selected,
                                 game_info_all=game_info_all)
        write_file.close()


if __name__ == '__main__':
    test_flag = False
    fine_tune_flag = True
    action_selected_lists = [['shot'], ['pass'], None]
    if test_flag:
        data_path = '/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/'
        soccer_data_store_dir = "/Users/liu/Desktop/soccer-data-sample/Soccer-data/"
    else:
        data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
        soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
    player_id_name_pair_dir = '../resource/soccer_id_name_pair.json'
    player_summary_info_dir = '../resource/Soccer_summary.csv'

    tt_lstm_config_path = "../soccer-config-v5.yaml"
    soccer_dir_all = os.listdir(data_path)

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4

    # data_name = compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
    #                                          dir_all=soccer_dir_all)
    game_info_path = '../resource/player_team_id_name_value.csv'
    game_info_file = open(game_info_path)
    game_reader = csv.DictReader(game_info_file)
    game_info_all = []
    for r in game_reader:
        p_name = r['playerName']
        t_name = r['teamName']
        id = r['playerId']
        game_info_all.append([p_name, t_name, id])

    if fine_tune_flag:
        model_number = 4801
        league_number = 8
        league_name = "_English_Barclays_Premier_League"
        player_summary_info_dir = '../resource/whoScored/PremierLeague/Premier_League_summary.csv'
    else:
        model_number = 2101  # 2101, 7201, 7801 ,10501 ,13501 ,15301 ,18301*, 20701*
        league_number = None
        league_name = ''

    data_name = get_data_name(config=tt_lstm_config, league_name=league_name)
    compute_ranking(data_name=data_name, game_data_dir=data_path, soccer_data_store_dir=soccer_data_store_dir,
                    player_summary_info_dir=player_summary_info_dir,
                    action_selected_lists=action_selected_lists, game_info_all=game_info_all,
                    league_number=league_number)
