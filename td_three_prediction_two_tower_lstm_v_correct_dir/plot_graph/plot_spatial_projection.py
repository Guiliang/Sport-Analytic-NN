import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import construct_simulation_data, \
    padding_hybrid_feature_input
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import image_blending
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure import td_tt_lstm
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.config.icehockey_feature_setting import select_feature_setting


def start_lstm_generate_spatial_simulation(history_action_type, history_action_type_coord,
                                           action_type, data_simulation_dir, simulation_type,
                                           feature_type, max_trace_length, features_num, is_home=True):
    simulated_data_all = []

    features_train, features_mean, features_scale, actions = select_feature_setting(feature_type=feature_type)

    for history_index in range(0, len(history_action_type) + 1):
        state_ycoord_list = []
        for ycoord in np.linspace(-42.5, 42.5, 171):
            state_xcoord_list = []
            for xcoord in np.linspace(-100.0, 100.0, 401):
                set_dict = {'xAdjCoord': xcoord, 'yAdjCoord': ycoord}
                state_generated = construct_simulation_data(
                    features_train=features_train,
                    features_mean=features_mean,
                    features_scale=features_scale,
                    feature_type=feature_type,
                    is_home=is_home,
                    action_type=action_type,
                    actions=actions,
                    set_dict=set_dict)
                state_generated_list = [state_generated]
                for inner_history in range(0, history_index):
                    xAdjCoord = history_action_type_coord[inner_history].get('xAdjCoord')
                    yAdjCoord = history_action_type_coord[inner_history].get('yAdjCoord')
                    action = history_action_type[inner_history]
                    if action != action_type:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1, action_type: 0}
                    else:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1}
                    state_generated_history = construct_simulation_data(
                        features_train=features_train,
                        features_mean=features_mean,
                        features_scale=features_scale,
                        feature_type=feature_type,
                        is_home=is_home,
                        action_type=action_type,
                        actions=actions,
                        set_dict=set_dict_history, )
                    state_generated_list = [state_generated_history] + state_generated_list

                state_generated_padding = padding_hybrid_feature_input(
                    hybrid_feature_input=state_generated_list,
                    max_trace_length=max_trace_length,
                    features_num=features_num)
                state_xcoord_list.append(state_generated_padding)
            state_ycoord_list.append(np.asarray(state_xcoord_list))

        store_data_dir = data_simulation_dir + '/' + simulation_type

        if not os.path.isdir(store_data_dir):
            os.makedirs(store_data_dir)
        # else:
        #     raise Exception
        if is_home:
            sio.savemat(
                store_data_dir + "/LSTM_Home_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        else:
            sio.savemat(
                store_data_dir + "/LSTM_Away_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        simulated_data_all.append(np.asarray(state_ycoord_list))

    return simulated_data_all


def nn_simulation(simulate_data,
                  model_path,
                  model_nn,
                  history_action_type,
                  action_type,
                  sess_nn,
                  nn_save_image_dir,
                  nn_half_save_image_dir,
                  draw_target='Q_home'):
    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)

    else:
        print model_path
        raise Exception("can't restore network")

    value_spatial_home = []
    value_spatial_away = []
    # value_spatial_home_dict_list = []
    # value_spatial_away_dict_list = []

    # y_count = 0
    for x_coord_states in simulate_data:
        trace_length = np.ones(len(x_coord_states)) * (len(history_action_type) + 1)
        if "home" in draw_target:
            home_away_indicator = np.ones(len(x_coord_states))
        else:
            home_away_indicator = np.zeros(len(x_coord_states))
        readout_x_coord_values = model_nn.readout.eval(
            feed_dict={model_nn.rnn_input_ph: x_coord_states,
                       model_nn.trace_lengths_ph: trace_length,
                       model_nn.home_away_indicator_ph: home_away_indicator})

        # y_coord = -42.5 + y_count
        # y_count += float(85) / float(simulate_data.shape[0] - 1)

        # for x_coord in np.linspace(-100.0, 100.0, x_coord_states.shape[0]):
        #     readout_x_label = 0 + float(x_coord_states.shape[0] - 1) / 200 * (x_coord + 100)
        #     value_spatial_home_dict_list.append(
        #         {'x_coord': x_coord, 'y_coord': y_coord, 'q_home': readout_x_coord_values[int(readout_x_label), 0]})
        #     value_spatial_away_dict_list.append(
        #         {'x_coord': x_coord, 'y_coord': y_coord, 'q_home': readout_x_coord_values[int(readout_x_label), 1]})

        value_spatial_home.append((readout_x_coord_values[:, 0]).tolist())
        value_spatial_away.append((readout_x_coord_values[:, 1]).tolist())

    # value_spatial_home_df = pd.DataFrame(value_spatial_home_dict_list)
    # value_spatial_away_df = pd.DataFrame(value_spatial_away_dict_list)

    if draw_target == "Q_home":
        value_spatial = value_spatial_home
        if action_type == "shot":
            vmin_set = 0.55
            vmax_set = 0.80
        else:
            vmin_set = None
            vmax_set = None
    elif draw_target == "Q_away":
        value_spatial = value_spatial_away
        if action_type == "shot":
            vmin_set = 0.16
            vmax_set = 0.50
        else:
            vmin_set = None
            vmax_set = None
    else:
        raise ValueError("wrong type of DRAW_TARGET")

    print "heat map"
    plt.figure(figsize=(15, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(value_spatial, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r",
                     vmin=vmin_set,
                     vmax=vmax_set)
    # plt.xlabel('XAdjcoord', fontsize=18)
    # plt.ylabel('YAdjcoord', fontsize=18)
    if len(history_action_type) != 0:

        plt.title("PT-LSTM {2} for {0}\n with history:{1} on right rink"
                  .format(action_type, str(history_action_type), draw_target), fontsize=30)
    elif len(history_action_type) == 0:
        plt.title("PT-LSTM {1} for {0} without history".format(action_type, draw_target), fontsize=20)
    else:
        raise ValueError("undefined HIS_ACTION_TYPE{0}:".format(history_action_type))

    plt.savefig(nn_save_image_dir)

    value_spatial_home_half = [v[200:402] for v in value_spatial]
    plt.figure(figsize=(15, 12))
    sns.set()
    ax = sns.heatmap(value_spatial_home_half, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r", vmin=vmin_set,
                     vmax=vmax_set)
    # plt.xlabel('XAdjcoord', fontsize=26)
    # plt.ylabel('YAdjcoord', fontsize=26)
    if len(history_action_type) != 0:
        plt.title("PT-LSTM {2} for {0}\n with history:{1} on right rink".format(action_type, str(history_action_type),
                                                                                draw_target),
                  fontsize=30)
    elif len(history_action_type) == 0:
        plt.title("PT-LSTM {2} for {0}\n with history:{1} on right rink".format(action_type, "[]", draw_target),
                  fontsize=30)
    else:
        raise ValueError("undefined HIS_ACTION_TYPE{0}:".format(history_action_type))

    plt.savefig(nn_half_save_image_dir)


if __name__ == '__main__':
    tt_lstm_config_path = "../icehockey-config.yaml"
    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    history_action_type = ['reception', 'pass', 'reception']
    history_action_type_coord = [{'xAdjCoord': 50.18904442739472, 'yAdjCoord': 0.47699011276943787},
                                 {'xAdjCoord': 48.06645981534736, 'yAdjCoord': 0.7993870137732708},
                                 {'xAdjCoord': 38.898981773048014, 'yAdjCoord': 1.1692141494472155}]

    # history_action_type = []
    # history_action_type_coord = []

    feature_type = tt_lstm_config.learn.feature_type
    batch_size = tt_lstm_config.learn.batch_size
    iterate_num = tt_lstm_config.learn.iterate_num
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4
    if_correct_velocity = tt_lstm_config.learn.if_correct_velocity
    action_type = 'shot'
    simulation_type = 'entire_spatial_simulation'
    data_simulation_dir = '../simulated_data/'
    draw_target = 'Q_home'
    model_type = tt_lstm_config.learn.model_type
    is_diff = True
    if is_diff:
        diff_str = "_diff"
    else:
        diff_str = ""

    simulated_data_all = start_lstm_generate_spatial_simulation(history_action_type=history_action_type,
                                                                history_action_type_coord=history_action_type_coord,
                                                                action_type=action_type,
                                                                data_simulation_dir=data_simulation_dir,
                                                                simulation_type=simulation_type,
                                                                feature_type=feature_type,
                                                                features_num=tt_lstm_config.learn.feature_number,
                                                                max_trace_length=tt_lstm_config.learn.max_trace_length
                                                                )

    sess_nn = tf.InteractiveSession()

    model_nn = td_tt_lstm.td_prediction_tt_embed(
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
    # saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/soccer-models/hybrid_sl_saved_NN" \
    #                                                             "/Scale-tt-three-cut_together_saved_networks_feature" \
    #                      + str(tt_lstm_config.learn.feature_type) + "_batch" \
    #                      + str(tt_lstm_config.learn.batch_size) + "_iterate" \
    #                      + str(tt_lstm_config.learn.iterate_num) + "_lr" \
    #                      + str(tt_lstm_config.learn.learning_rate) + "_" \
    #                      + str(tt_lstm_config.learn.model_type) + \
    #                      tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + \
    #                      str(tt_lstm_config.learn.max_trace_length)

    saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
        tt_lstm_config.learn.feature_type) + "_batch" + str(
        tt_lstm_config.learn.batch_size) + "_iterate" + str(
        tt_lstm_config.learn.iterate_num) + "_lr" + str(
        tt_lstm_config.learn.learning_rate) + "_" + str(
        tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
        tt_lstm_config.learn.max_trace_length)

    for data_index in range(0, len(simulated_data_all)):
        nn_image_save_dir = "./icehockey-image/{7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png". \
            format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        nn_half_image_save_dir = "./icehockey-image/right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        blend_image_save_dir = "./icehockey-image/blend {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        blend_half_image_save_dir = "./icehockey-image/blend right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)

        simulate_data = simulated_data_all[data_index]
        nn_simulation(simulate_data=simulate_data,
                      model_path=saved_network_path,
                      model_nn=model_nn,
                      history_action_type=history_action_type[:data_index],
                      action_type=action_type[:data_index],
                      sess_nn=sess_nn,
                      nn_save_image_dir=nn_image_save_dir,
                      nn_half_save_image_dir=nn_half_image_save_dir)
        image_blending(nn_image_save_dir, blend_image_save_dir, nn_half_image_save_dir, blend_half_image_save_dir)
