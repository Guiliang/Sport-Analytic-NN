import os

import math
import scipy.io as sio
import tensorflow as tf
import td_prediction_simple_separated
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

FEATURE_TYPE = 5
MODEL_TYPE = "V3"
ITERATE_NUM = 25


SIMPLE_HOME_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/saved_entire_Home_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate"+str(ITERATE_NUM)+"-NEG_REWARD_GAMMA1_"+MODEL_TYPE+"-Sequenced"
SIMPLE_AWAY_SAVED_NETWORK_PATH = "/cs/oschulte/Galen/models/saved_NN/saved_entire_Away_networks_feature" + str(
    FEATURE_TYPE) + "_batch16_iterate"+str(ITERATE_NUM)+"-NEG_REWARD_GAMMA1_"+MODEL_TYPE+"-Sequenced"


def iterate_over_games(model_iter, sess_iter):
    test_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward"
    dir_test_all = os.listdir(test_dir)
    for dir_game in dir_test_all:
        game_files = os.listdir(test_dir + "/" + dir_game)
        for filename in game_files:
            if filename.startswith("reward"):
                reward_name = filename
            elif filename.startswith("state"):
                state_name = filename

        reward = sio.loadmat(test_dir + "/" + dir_game + "/" + reward_name)
        reward = (reward['reward'][0]).tolist()
        state = sio.loadmat(test_dir + "/" + dir_game + "/" + state_name)
        state = state['state']

        nn_real_game(SIMPLE_HOME_SAVED_NETWORK_PATH, sess_iter)
        readout_t_batch_home = model_iter.read_out.eval(feed_dict={model_iter.x: state})
        readout_record_home = (readout_t_batch_home[:, 0]).tolist()
        print readout_record_home
        # readout_record_home = [value[0] for value in readout_record_home]

        nn_real_game(SIMPLE_AWAY_SAVED_NETWORK_PATH, sess_iter)
        readout_t_batch_away = model_iter.read_out.eval(feed_dict={model_iter.x: state})
        readout_record_away = (readout_t_batch_away[:, 0]).tolist()
        print readout_record_away
        # readout_record_away = [value[0] for value in readout_record_away]
        # readout_record_away_abs = map(abs, readout_record_away)

        stimulate_value_rate = [(float(c)/(float(c) - float(d))) for c, d in zip(readout_record_home, readout_record_away)]
        draw_game_predict(stimulate_value_rate=stimulate_value_rate, reward=reward)


def draw_game_predict(stimulate_value_rate, reward):
    r_pos_index = [i for i in range(0, len(reward)) if reward[i] == 1]
    r_neg_index = [i for i in range(0, len(reward)) if reward[i] == -1]
    x_axis = range(0, len(stimulate_value_rate))
    plt.plot(x_axis, stimulate_value_rate, '-')
    plt.scatter(r_pos_index, [0] * len(r_pos_index), color='b')
    plt.scatter(r_neg_index, [0] * len(r_neg_index), color='r')
    plt.xlabel("events number (time)", fontsize=15)
    plt.ylabel("value", fontsize=15)
    plt.title("Season 2015-2016 regular match", fontsize=15)
    plt.show()
    plt.close()


def nn_real_game(SIMPLE_SAVED_NETWORK_PATH, sess):
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        except Exception:
            print "Can't load model, wrong model type"
        print "Successfully loaded:" + str(checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

        # readout_record = []
        # reward_record = []
        # for dir_game in dir_test_all:
        #     game_files = os.listdir(test_dir + "/" + dir_game)
        #     for filename in game_files:
        #         if filename.startswith("reward"):
        #             reward_name = filename
        #         elif filename.startswith("state"):
        #             state_name = filename
        #
        #     reward = sio.loadmat(test_dir + "/" + dir_game + "/" + reward_name)
        #     reward = (reward['reward'][0]).tolist()
        #     state = sio.loadmat(test_dir + "/" + dir_game + "/" + state_name)
        #     state = state['state']
        #
        # readout_t_batch = model_nn.read_out.eval(feed_dict={model_nn.x: state})
        #     readout_record = readout_record + (readout_t_batch[:, 0]).tolist()
        #     reward_record = reward_record + reward
        #
        #     r_pos_index = [i for i in range(0, len(reward)) if reward[i] == 1]
        #     r_neg_index = [i for i in range(0, len(reward)) if reward[i] == -1]
        #
        #     print(dir_game)
        #     x_axis = range(0, len(state))
        #     plt.plot(x_axis, (readout_t_batch[:, 0]).tolist(), '-')
        #     plt.scatter(r_pos_index, [0] * len(r_pos_index), color='b')
        #     plt.scatter(r_neg_index, [0] * len(r_neg_index), color='r')
        #     plt.xlabel("events number (time)", fontsize=15)
        #     plt.ylabel("value", fontsize=15)
        #     plt.title("Season 2015-2016 regular match", fontsize=15)
        #     plt.show()
        #     plt.close()
        #
        # print(len(readout_record))
        # print pearsonr(readout_record, reward_record)


if __name__ == '__main__':
    sess_nn = tf.InteractiveSession()
    if MODEL_TYPE == "V1":
        model_nn = td_prediction_simple_separated.td_prediction_simple()
    elif MODEL_TYPE == "V2":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V2()
    elif MODEL_TYPE == "V3":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V3()
    elif MODEL_TYPE == "V4":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V4()
    elif MODEL_TYPE == "V5":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V5()
    elif MODEL_TYPE == "V6":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V6()
    elif MODEL_TYPE == "V7":
        model_nn = td_prediction_simple_separated.td_prediction_simple_V7()
    else:
        raise ValueError("Unclear model type")
    # model_nn = td_prediction_simple_separated.td_prediction_simple_V3()
    iterate_over_games(model_nn, sess_nn)

    # # stimulate_value_home = nn_real_game(SIMPLE_HOME_SAVED_NETWORK_PATH, sess=sess_nn, model=model_nn_home)
    # # stimulate_value_away = nn_real_game(SIMPLE_HOME_SAVED_NETWORK_PATH, sess=sess_nn, model=model_nn_away)
    #
    # stimulate_value_home = [value[0] for value in stimulate_value_home]
    # stimulate_value_away = [value[0] for value in stimulate_value_away]
    # stimulate_value_away_abs = map(abs, stimulate_value_away)
    # stimulate_value_away_abs.reverse()
    # stimulate_value_rate = [float(c) / float(d) for c, d in zip(stimulate_value_home, stimulate_value_away_abs)]
    # draw_value_over_position(stimulate_value_rate)
