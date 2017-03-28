import scipy.io as sio
import tensorflow as tf
import td_prediction_RNN
import td_prediction_simple
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

SIMULATION_DATA_PATH = "/home/gla68/Documents/Hockey-data/Simulation-data/position_simulation/position_simulation-shot-1.mat"
RNN_SAVED_NETWORK_PATH = "./"
SIMPLE_SAVED_NETWORK_PATH = "./saved_networks_feature2"
# SIMPLE_SAVED_NETWORK_PATH = "./saved_networks_feature2_FORWARD"


def rnn_simulation():
    sess_nn = tf.InteractiveSession()
    rnn_input_nn, read_out_nn, y_nn, train_step_nn, cost_nn = td_prediction_RNN.create_network_RNN_type2()

    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = simulate_data['simulate_data']

    checkpoint = tf.train.get_checkpoint_state(RNN_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        sess_nn.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    return None


def nn_simulation():
    sess_nn = tf.InteractiveSession()
    x, read_out, y, train_step, cost, W1_print, y1_print, b1_print, W2_print, y2_print, b2_print = td_prediction_simple.create_network()

    simulate_data = sio.loadmat(SIMULATION_DATA_PATH)
    simulate_data = (simulate_data['simulate_data'])

    saver = tf.train.Saver()
    sess_nn.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SIMPLE_SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess_nn, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("can't restore network")

    readout_t1_batch = read_out.eval(feed_dict={x: simulate_data})

    draw_value_over_position(readout_t1_batch)

    print(max(readout_t1_batch))
    print(min(readout_t1_batch))

    return readout_t1_batch


def draw_value_over_position(y):
    max_data = (max(y))[0]
    min_data = (min(y))[0]
    scale = float(80) / (max_data - min_data)
    y_list = y.tolist()
    y_deal = []
    y_deal
    for y_data in y_list:
        # y_data_deal = ((y_data[0] - min_data) * scale) - 40
        y_deal.append(y_data[0])

    print(y_deal)

    x = np.arange(-100, 100, 2)
    plt.plot(x, y_deal)
    # img = imread('./hockey-field.png')
    # plt.imshow(img, extent=[-100, 100, -50, 50])
    plt.show()
    return None


if __name__ == '__main__':
    nn_simulation()
