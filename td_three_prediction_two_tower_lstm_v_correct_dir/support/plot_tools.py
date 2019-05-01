import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_icehockey_game_data
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_soccer_game_data


def image_blending(value_Img_dir, save_dir, value_Img_half_dir, half_save_dir, background_image_dir,
                   sport='ice-hockey'):
    value_Img = cv2.imread(
        value_Img_dir)
    value_Img_half = cv2.imread(
        value_Img_half_dir)
    # background = cv2.imread("../resource/hockey-field.png")
    background = cv2.imread(background_image_dir)
    # v_rows, v_cols, v_channels = value_Img.shape
    # v_h_rows, v_h_cols, v_h_channels = value_Img_half.shape
    print sport
    if sport == 'soccer':
        image_y = [66, 539]
        image_x = [138, 902]
        image_h_y = [134, 1079]
        image_h_x = [190, 1148]
        background_half = [470, 944]
    elif sport == 'ice-hockey':
        image_y = [60, 540]
        image_x = [188, 1118]
        image_h_y = [120, 1090]
        image_h_x = [190, 1125]
        background_half = [899, 1798]
    else:
        raise ValueError("unknown sport")
    focus_Img = value_Img[image_y[0]:image_y[1], image_x[0]:image_x[1]]
    # cv2.imshow('res', focus_Img)
    # cv2.waitKey(0)
    f_rows, f_cols, f_channels = focus_Img.shape
    focus_background = cv2.resize(background, (f_cols, f_rows), interpolation=cv2.INTER_CUBIC)
    blend_focus = cv2.addWeighted(focus_Img, 1, focus_background, 0.5, -255 / 2)
    blend_all = value_Img
    blend_all[image_y[0]:image_y[1], image_x[0]:image_x[1]] = blend_focus
    # final_rows = v_rows * float(b_rows) / float(f_rows)
    # final_cols = v_cols * float(b_cols) / float(f_cols)
    # blend_all_final = cv2.resize(blend_all, (int(final_cols), int(final_rows)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('res', focus_Img)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir, blend_all)

    focus_Img_half = value_Img_half[image_h_y[0]:image_h_y[1], image_h_x[0]:image_h_x[1]]
    # cv2.imshow('res', focus_Img_half)
    # cv2.waitKey(0)
    f_h_rows, f_h_cols, f_h_channels = focus_Img_half.shape
    focus_background_half = cv2.resize(background[:, background_half[0]:background_half[1], :], (f_h_cols, f_h_rows),
                                       interpolation=cv2.INTER_CUBIC)
    blend_half_focus = cv2.addWeighted(focus_Img_half, 1, focus_background_half, 0.5, -255 / 2)
    blend_half_all = value_Img_half
    blend_half_all[image_h_y[0]:image_h_y[1], image_h_x[0]:image_h_x[1]] = blend_half_focus
    cv2.imwrite(half_save_dir, blend_half_all)


def read_plot_model(sess_nn, model_path):
    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        print model_path
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print model_path
        raise Exception("can't restore network")


def compute_game_values(sess_nn, model, data_store, dir_game, config, sport):
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
                  draw_target='Q_home',
                  sport='ice-hockey'):
    value_spatial_home = []
    value_spatial_away = []
    # value_spatial_home_dict_list = []
    # value_spatial_away_dict_list = []

    # y_count = 0
    for x_coord_states in simulate_data:
        trace_length = np.ones(len(x_coord_states)) * (len(history_action_type) + 1)
        # trace_length = np.ones(len(x_coord_states)) * 2
        if "home" in draw_target:
            home_away_indicator = np.ones(len(x_coord_states))
        else:
            home_away_indicator = np.zeros(len(x_coord_states))
        readout_x_coord_values = model_nn.readout.eval(
            feed_dict={model_nn.rnn_input_ph: x_coord_states,
                       model_nn.trace_lengths_ph: trace_length,
                       model_nn.home_away_indicator_ph: home_away_indicator})

        value_spatial_home.append((readout_x_coord_values[:, 0]).tolist())
        value_spatial_away.append((readout_x_coord_values[:, 1]).tolist())

    if sport == 'ice-hockey':
        half_x = [200, 402]
    elif sport == 'soccer':
        half_x = [100, 200]
    else:
        raise ValueError('unknown sport')

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
    plt.figure(figsize=(12, 6))
    sns.set(font_scale=1.6)
    ax = sns.heatmap(value_spatial, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r",
                     vmin=vmin_set,
                     vmax=vmax_set)
    # plt.show()
    # plt.xlabel('XAdjcoord', fontsize=18)
    # plt.ylabel('YAdjcoord', fontsize=18)
    if len(history_action_type) != 0:
        plt.title("{2} for {0}\n with history:{1}"
                  .format(action_type, str(history_action_type), draw_target), fontsize=30)
    elif len(history_action_type) == 0:
        plt.title("{1} for {0}".format(action_type, draw_target), fontsize=20)
    else:
        raise ValueError("undefined HIS_ACTION_TYPE{0}:".format(history_action_type))

    plt.savefig(nn_save_image_dir)

    value_spatial_home_half = [v[half_x[0]:half_x[1]] for v in value_spatial]
    plt.figure(figsize=(15, 12))
    sns.set(font_scale=2.5)
    ax = sns.heatmap(value_spatial_home_half, xticklabels=False, yticklabels=False,
                     cmap="RdYlBu_r", vmin=vmin_set,
                     vmax=vmax_set)
    # plt.xlabel('XAdjcoord', fontsize=26)
    # plt.ylabel('YAdjcoord', fontsize=26)
    if len(history_action_type) != 0:
        plt.title("{2} for {0}\n with history:{1}".format(action_type, str(history_action_type),
                                                                        draw_target), fontsize=30)
    elif len(history_action_type) == 0:
        plt.title("{2} for {0}".format(action_type, "[]", draw_target),
                  fontsize=30)
    else:
        raise ValueError("undefined HIS_ACTION_TYPE{0}:".format(history_action_type))

    plt.savefig(nn_half_save_image_dir)


def plot_game_value(game_value, save_image_name,
                    normalize_data, home_team, away_team,
                    if_normalized=True, draw_three=True, game_time_all=None):
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

    if game_time_all is not None:
        x = game_time_all
    else:
        x = event_numbers
    plt.figure(figsize=(15, 6))
    plt.xticks(size=15)
    plt.yticks(size=15)
    if draw_three:
        plt.plot(x, game_value_home, label="Q for Home".format(home_team))
        plt.plot(x, game_value_away, label="Q for Away".format(away_team))
        plt.plot(x[0:len(game_value_end) - 1], game_value_end[0:len(game_value_end) - 1],
                 label="Q for Game End")
    else:
        plt.plot(x, game_value_diff, label="q_home-q_away")
    # plt.title("2015-2016 NHL regular season {0}(Away) vs {1}(Home)".format(away_team, home_team), fontsize=15)
    plt.xlabel("Game Time", fontsize=15)
    plt.ylabel("Q Values", fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    # plt.show()
    plt.savefig(save_image_name)

    return home_max_index, away_max_index, home_maxs, away_maxs
