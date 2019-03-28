import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_icehockey_game_data
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_soccer_game_data


def image_blending(value_Img_dir, save_dir, value_Img_half_dir, half_save_dir):
    value_Img = cv2.imread(
        value_Img_dir)
    value_Img_half = cv2.imread(
        value_Img_half_dir)
    background = cv2.imread("../resource/hockey-field.png")
    # v_rows, v_cols, v_channels = value_Img.shape
    # v_h_rows, v_h_cols, v_h_channels = value_Img_half.shape

    focus_Img = value_Img[60:540, 188:1118]
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255 / 2)
    blend_all = value_Img
    blend_all[60:540, 188:1118] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('res', focus_Img)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)

    focus_Img_half = value_Img_half[120:1090, 190:1125]
    f_h_rows, f_h_cols, f_h_channels = focus_Img_half.shape
    focus_background_half = cv2.resize(background[:, 899:1798, :], (f_h_cols, f_h_rows), interpolation=cv2.INTER_CUBIC)
    blend_half_focus = cv2.addWeighted(focus_Img_half, 1, focus_background_half, 0.5, -255 / 2)
    blend_half_all = value_Img_half
    blend_half_all[120:1090, 190:1125] = blend_half_focus
    cv2.imwrite(half_save_dir, blend_half_all)


def compute_game_values(model_path, sess_nn, model, data_store, dir_game, config, sport):
    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)

    else:
        print model_path
        raise Exception("can't restore network")

    if sport == "IceHockey":
        state_trace_length, state_input, reward, ha_id = get_icehockey_game_data(data_store=data_store,
                                                                                 dir_game=dir_game,
                                                                                 config=config, )
    elif sport == 'Soccer':
        state_trace_length, state_input, reward, ha_id = get_soccer_game_data(data_store=data_store,
                                                                              dir_game=dir_game,
                                                                              config=config, )
    else:
        raise ValueError("unknown sport")

    [readout] = sess_nn.run([model.readout],
                            feed_dict={model.trace_lengths_ph: state_trace_length,
                                       model.rnn_input_ph: state_input,
                                       model.home_away_indicator_ph: ha_id
                                       })
    return readout


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


def plot_game_value(game_value, save_image_name,
                    normalize_data, home_team, away_team,
                    if_normalized=True, draw_three=True):
    # game_value_home = game_value[:, 0]/(game_value[:, 0]+game_value[:, 1])
    # game_value_away = game_value[:, 1]/(game_value[:, 0]+game_value[:, 1])
    game_value_home = game_value[:, 0]
    game_value_away = game_value[:, 1]
    game_value_end = game_value[:, 2]

    if if_normalized:
        game_value_home, game_value_away, game_value_end = normalize_data(game_value_home, game_value_away,
                                                                          game_value_end)

    # find the index of max home and away
    home_max_index = game_value_home.argsort()[-20:][::-1]
    away_max_index = game_value_away.argsort()[-20:][::-1]
    home_maxs = game_value_home[home_max_index]
    away_maxs = game_value_away[away_max_index]

    game_value_diff = game_value_home - game_value_away
    game_value_rate = game_value_home / game_value_away

    event_numbers = [d for d in range(1, len(game_value_diff) + 1)]

    plt.figure(figsize=(15, 6))
    if draw_three:
        plt.plot(event_numbers, game_value_home, label="Q for Home".format(home_team))
        plt.plot(event_numbers, game_value_away, label="Q for Away".format(away_team))
        plt.plot(event_numbers[0:len(game_value_end) - 1], game_value_end[0:len(game_value_end) - 1],
                 label="Q for Game End")
    else:
        plt.plot(event_numbers, game_value_diff, label="q_home-q_away")
    plt.title("2015-2016 NHL regular season {0}(Away) vs {1}(Home)".format(away_team, home_team), fontsize=15)
    plt.xlabel("event number", fontsize=13)
    plt.ylabel("Q Value", fontsize=13)
    plt.legend(loc='upper right', fontsize=13)
    # plt.show()
    plt.savefig(save_image_name)

    return home_max_index, away_max_index, home_maxs, away_maxs
