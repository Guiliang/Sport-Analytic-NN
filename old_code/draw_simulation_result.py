import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import td_prediction_simple
import tensorflow as tf
from scipy.stats import pearsonr

import td_prediction_eligibility_trace as et

STIMULATE_TYPE = "angel"
FEATURE_TYPE = 8
SIMPLE_SAVED_NETWORK_PATH = "./saved_NN/saved_networks_feature8_batch16_iterate2Converge-NEG_REWARD_GAMMA1_V3"
ET_SAVED_NETWORK_PATH = et.checkpoint_path

SIMULATION_DATA_PATH = "/media/gla68/Windows/Hockey-data/Simulation-data-feature" + str(
    FEATURE_TYPE) + "/" + STIMULATE_TYPE + "_simulation/"


def nn_simulation():
    simulate_data_name = ["Away_angel_simulation-shot-feature3-['time remained'][0].mat",
                          "Away_angel_simulation-shot-feature3-1.mat",
                          "Away_angel_simulation-shot-feature3-['time remained'][3600].mat",
                          "angel_simulation-shot-feature3-['time remained'][0].mat",
                          "angel_simulation-shot-feature3-1.mat",
                          "angel_simulation-shot-feature3-['time remained'][3600].mat"]
    name_list = ["away shot, remain 0",
                 "away shot, remain 1800",
                 "away shot, remain 3600",
                 "home shot, remain 0",
                 "home shot, remain 1800",
                 "home shot, remain 3600"]
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple.td_prediction_simple()

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    plt.subplots()
    for i in range(len(simulate_data_name)):
        specify_path = simulate_data_name[i]
        simulate_data = sio.loadmat(SIMULATION_DATA_PATH + specify_path)
        simulate_data = (simulate_data['simulate_data'])
        readout_t1_batch = model_nn.read_out.eval(feed_dict={model_nn.x: simulate_data})
        y_list = readout_t1_batch.tolist()
        y_deal = []
        for y_data in y_list:
            # y_data_deal = ((y_data[0] - min_data) * scale) - 40
            y_deal.append(y_data[0])

        if STIMULATE_TYPE == "angel":
            x = np.arange(-0, 360, float(360) / 120)
        elif STIMULATE_TYPE == "position":
            x = np.arange(-100, 100, 2)
        plt.plot(x, y_deal, '-', linewidth=3, label=str(name_list[i]))
        plt.legend(loc='upper left')

    plt.title("Simulation around the gate with different time remain", fontweight="bold", fontsize=15)
    plt.ylabel("Value", fontweight="bold", fontsize=15)
    plt.xlabel("Theta around the gate", fontweight="bold", fontsize=15)

    plt.show()


def et_simulation():
    simulate_data_name = ["Away_angel_simulation-shot-feature3-['time remained'][0].mat",
                          "Away_angel_simulation-shot-feature3-1.mat",
                          "Away_angel_simulation-shot-feature3-['time remained'][3600].mat",
                          "angel_simulation-shot-feature3-['time remained'][0].mat",
                          "angel_simulation-shot-feature3-1.mat",
                          "angel_simulation-shot-feature3-['time remained'][3600].mat"]
    # simulate_data_name = ["Away_angel_simulation-shot-feature3-['Penalty'][-1].mat",
    #                       "Away_angel_simulation-shot-feature3-['Penalty'][0].mat",
    #                       "Away_angel_simulation-shot-feature3-['Penalty'][1].mat",
    #                       "angel_simulation-shot-feature3-['Penalty'][-1].mat",
    #                       "angel_simulation-shot-feature3-['Penalty'][0].mat",
    #                       "angel_simulation-shot-feature3-['Penalty'][1].mat"]
    name_list = ["away shot, remain 0",
                 "away shot, remain 1800",
                 "away shot, remain 3600",
                 "home shot, remain 0",
                 "home shot, remain 1800",
                 "home shot, remain 3600"]
    # name_list = ["away shot, shortHanded",
    #              "away shot, evenStrength",
    #              "away shot, powerPlay",
    #              "home shot, shortHanded",
    #              "home shot, evenStrength",
    #              "home shot, powerPlay"]
    sess_et = tf.InteractiveSession()
    model_et = et.Model(sess_et, et.model_path, et.summary_path, et.checkpoint_path)

    saver = tf.train.Saver()
    sess_et.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(ET_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_et, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    plt.subplots()
    for i in range(len(simulate_data_name)):
        specify_path = simulate_data_name[i]
        simulate_data = sio.loadmat(SIMULATION_DATA_PATH + specify_path)
        simulate_data = (simulate_data['simulate_data'])
        train_len = len(simulate_data)
        game_step = 0
        output_record = []
        while game_step < train_len:
            s_t = np.array([simulate_data[game_step]])
            V = model_et.get_output(s_t)
            output_record.append(V[0][0])
            game_step += 1

        y_deal = output_record

        if STIMULATE_TYPE == "angel":
            x = np.arange(-0, 360, float(360) / 120)
        elif STIMULATE_TYPE == "position":
            x = np.arange(-100, 100, 2)
        plt.plot(x, y_deal, '-', linewidth=3, label=str(name_list[i]))
        plt.legend(loc='upper left')

    plt.title("Simulation around the gate with different time remain", fontweight="bold", fontsize=15)
    plt.ylabel("Value", fontweight="bold", fontsize=15)
    plt.xlabel("Theta Around the Gate", fontweight="bold", fontsize=15)

    plt.show()


def nn_correlation():
    test_dir = "/media/gla68/Windows/Hockey-data/Hockey-Training-All-feature" + str(
        FEATURE_TYPE) + "-scale-neg_reward_Test"
    dir_test_all = os.listdir(test_dir)
    sess_nn = tf.InteractiveSession()
    model_nn = td_prediction_simple.td_prediction_simple_V3()

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        except Exception:
            print "Can't load model, wrong model type"
        print "Successfully loaded:" + str(checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    readout_record = []
    reward_record = []
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

        readout_t_batch = model_nn.read_out.eval(feed_dict={model_nn.x: state})
        readout_record = readout_record + (readout_t_batch[:, 0]).tolist()
        reward_record = reward_record + reward

        r_pos_index = [i for i in range(0, len(reward)) if reward[i] == 1]
        r_neg_index = [i for i in range(0, len(reward)) if reward[i] == -1]

        print(dir_game)
        x_axis = range(0, len(state))
        plt.plot(x_axis, (readout_t_batch[:, 0]).tolist(), '-')
        plt.scatter(r_pos_index, [0] * len(r_pos_index), color='b')
        plt.scatter(r_neg_index, [0] * len(r_neg_index), color='r')
        plt.xlabel("events number (time)", fontsize=15)
        plt.ylabel("value", fontsize=15)
        plt.title("Season 2015-2016 regular match", fontsize=15)
        plt.show()
        plt.close()

    print(len(readout_record))
    print pearsonr(readout_record, reward_record)


def get_testing_batch(state, reward, train_number, train_len):
    """
    combine testing data to a batch
    """
    batch_return = []
    current_batch_length = 0
    while current_batch_length < td_prediction_simple.BATCH_SIZE:
        s_t = state[train_number]
        r_t = reward[train_number]
        train_number += 1
        if train_number + 1 == train_len:
            batch_return.append((s_t, r_t))
            return batch_return, 1
        else:
            batch_return.append((s_t, r_t))
        current_batch_length += 1

    return batch_return, 0


if __name__ == '__main__':
    nn_correlation()
