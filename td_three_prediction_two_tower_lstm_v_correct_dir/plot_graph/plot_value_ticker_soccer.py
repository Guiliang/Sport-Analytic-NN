import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import scipy.io as sio
import os
import tensorflow as tf
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import find_game_dir, \
    find_soccer_game_dir_by_team
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import plot_game_value
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.print_tools import print_mark_info
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import normalize_data, get_game_time
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import compute_game_values, read_plot_model

if __name__ == '__main__':
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
    # tt_lstm_config_path = '../icehockey-config.yaml'
    tt_lstm_config_path = "../soccer-config-v2.yaml"
    home_team = 'Manchester City'
    away_team = 'Arsenal'
    # 919164
    # dir_all = os.listdir(data_path)
    dir_all = ['918996.json']
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

    saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
        tt_lstm_config.learn.feature_type) + "_batch" + str(
        tt_lstm_config.learn.batch_size) + "_iterate" + str(
        tt_lstm_config.learn.iterate_num) + "_lr" + str(
        tt_lstm_config.learn.learning_rate) + "_" + str(
        tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
        tt_lstm_config.learn.max_trace_length)

    # data_store = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
    # tt_lstm_config.learn.feature_type) + "-scale-neg_reward" + tt_lstm_config.learn.if_correct_velocity + "_length-dynamic"
    read_plot_model(model_path=saved_network_path, sess_nn=sess_nn)
    for game_name_dir in dir_all:
        game_name = game_name_dir.split('.')[0]
        game_time_all = get_game_time(data_path, game_name_dir)

        game_value = compute_game_values(sess_nn=sess_nn,
                                         model=model_nn,
                                         data_store=data_store_dir,
                                         dir_game=game_name,
                                         config=tt_lstm_config,
                                         sport='Soccer')
        for value_index in range(0, len(game_value)):

            if game_value[value_index][1] > 0.9:
                print value_index


        # data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}_v_correct_". \
        #     format(str(tt_lstm_config.learn.feature_type),
        #            str(tt_lstm_config.learn.iterate_num),
        #            str(learning_rate_write),
        #            str(tt_lstm_config.learn.batch_size),
        #            str(tt_lstm_config.learn.max_trace_length),
        #            str(tt_lstm_config.learn.model_type))
        # game_value = sio.loadmat(data_store_dir + game_name_dir + "/" + data_name)
        # game_value = game_value[data_name]

        save_image_name = './soccer-image/{0} v.s. {1} value_ticker.png'.format(home_team, away_team)

        home_max_index, away_max_index, home_maxs, away_maxs = plot_game_value(game_value=game_value,
                                                                               game_time_all=game_time_all,
                                                                               save_image_name=save_image_name,
                                                                               normalize_data=normalize_data,
                                                                               home_team=home_team,
                                                                               away_team=away_team)
        # break
        # print_mark_info(data_store_dir, game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs)
