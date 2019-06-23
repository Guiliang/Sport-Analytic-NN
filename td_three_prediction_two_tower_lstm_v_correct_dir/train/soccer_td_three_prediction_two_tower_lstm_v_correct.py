import json
import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import scipy.io as sio
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import handle_trace_length, \
    compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv, get_network_dir

tt_lstm_config_path = "../soccer-config-v5.yaml"
tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
DATA_PATH = "/cs/oschulte/soccer-data/sequences_append_goal/"
SOCCER_DATA_STORE_DIR = "/cs/oschulte/Galen/Soccer-data/"
DIR_GAMES_ALL = os.listdir(SOCCER_DATA_STORE_DIR)
DIR_DATA_ALL = os.listdir(DATA_PATH)
number_of_total_game = len(DIR_GAMES_ALL)
TRAIN_FLAG = False
if TRAIN_FLAG:
    train_msg = 'Train_'
    DIR_GAMES_ALL = DIR_GAMES_ALL[:2400]
else:
    train_msg = ''

# fine-tuning testing
FINE_TUNING = True
if FINE_TUNING:
    league_number = 10
    league_name = "_English_Npower_Championship"
    model_train_continue = True
    print('fine-tuning on the {0} league'.format(league_name))
else:
    model_train_continue = False
    league_name = ''


def train_network(sess, model, print_parameters=False):
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver(max_to_keep=300)
    merge = tf.summary.merge_all()
    load_log_dir, load_network_dir = get_network_dir(league_name='', tt_lstm_config=tt_lstm_config, train_msg=train_msg)
    save_log_dir, save_network_dir = get_network_dir(league_name=league_name, tt_lstm_config=tt_lstm_config,
                                                     train_msg=train_msg)
    train_writer = tf.summary.FileWriter(save_log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    if model_train_continue:  # resume the training
        checkpoint = tf.train.get_checkpoint_state(load_network_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
            game_number_checkpoint = check_point_game_number % number_of_total_game
            game_number = check_point_game_number
            game_starting_point = 0
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    game_diff_record_all = []

    while True:
        game_diff_record_dict = {}
        iteration_now = game_number / number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        if converge_flag:
            break
        elif game_number >= number_of_total_game * tt_lstm_config.learn.iterate_num:
            break
        else:
            converge_flag = True

        for index in range(0, len(DIR_GAMES_ALL)):

            if FINE_TUNING:
                with open(DATA_PATH + DIR_DATA_ALL[index]) as f:
                    data_lines = json.load(f)
                competitionId = data_lines.get('competitionId')
                # print(competitionId)
                # print(league_number)
                if competitionId != league_number:
                    continue

            dir_game = DIR_GAMES_ALL[index]

            if tt_lstm_config.learn.model_train_continue:
                # if checkpoint and checkpoint.model_checkpoint_path:
                if tt_lstm_config.learn.model_train_continue:  # go the check point data
                    game_starting_point += 1
                    if game_number_checkpoint + 1 > game_starting_point:
                        continue

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            game_files = os.listdir(SOCCER_DATA_STORE_DIR + "/" + dir_game)
            for filename in game_files:
                if "reward" in filename:
                    reward_name = filename
                elif "state" in filename:
                    state_input_name = filename
                elif "trace" in filename:
                    state_trace_length_name = filename
                elif "home_away" in filename:
                    ha_id_name = filename

            reward = sio.loadmat(SOCCER_DATA_STORE_DIR + "/" + dir_game + "/" + reward_name)['reward']
            state_input = sio.loadmat(SOCCER_DATA_STORE_DIR + "/" + dir_game + "/" + state_input_name)['state']
            ha_id = sio.loadmat(SOCCER_DATA_STORE_DIR + "/" + dir_game + "/" + ha_id_name)["home_away"][0].astype(int)
            # state_input = (state_input['dynamic_feature_input'])
            # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
            # state_output = state_output['hybrid_output_state']
            state_trace_length = sio.loadmat(
                SOCCER_DATA_STORE_DIR + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
            # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
            state_trace_length = handle_trace_length(state_trace_length)
            state_trace_length, state_input, reward = compromise_state_trace_length(
                state_trace_length=state_trace_length,
                state_input=state_input,
                reward=reward,
                max_trace_length=tt_lstm_config.learn.max_trace_length,
                features_num=tt_lstm_config.learn.feature_number
            )

            print ("\n load file" + str(dir_game) + " success")
            reward_count = sum(reward)
            print ("reward number" + str(reward_count))
            if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
            print train_len
            train_number = 0
            s_t0 = state_input[train_number]
            train_number += 1

            while True:
                # try:
                batch_return, \
                train_number, \
                s_tl, \
                print_flag = get_together_training_batch(s_t0=s_t0,
                                                         state_input=state_input,
                                                         reward=reward,
                                                         train_number=train_number,
                                                         train_len=train_len,
                                                         state_trace_length=state_trace_length,
                                                         ha_id=ha_id,
                                                         batch_size=tt_lstm_config.learn.batch_size)

                # get the batch variables
                s_t0_batch = [d[0] for d in batch_return]
                s_t1_batch = [d[1] for d in batch_return]
                r_t_batch = [d[2] for d in batch_return]
                trace_t0_batch = [d[3] for d in batch_return]
                trace_t1_batch = [d[4] for d in batch_return]
                ha_id_t0_batch = [d[5] for d in batch_return]
                ha_id_t1_batch = [d[6] for d in batch_return]
                y_batch = []

                # readout_t1_batch = model.read_out.eval(
                #     feed_dict={model.trace_lengths: trace_t1_batch, model.rnn_input: s_t1_batch})  # get value of s

                [readout_t1_batch] = sess.run([model.read_out],
                                              feed_dict={model.trace_lengths_ph: trace_t1_batch,
                                                         model.rnn_input_ph: s_t1_batch,
                                                         model.home_away_indicator_ph: ha_id_t1_batch
                                                         })

                for i in range(0, len(batch_return)):
                    terminal = batch_return[i][7]
                    cut = batch_return[i][8]
                    # if terminal, only equals reward
                    if terminal or cut:
                        y_home = float((r_t_batch[i])[0])
                        y_away = float((r_t_batch[i])[1])
                        y_end = float((r_t_batch[i])[2])
                        y_batch.append([y_home, y_away, y_end])
                        break
                    else:
                        y_home = float((r_t_batch[i])[0]) + tt_lstm_config.learn.gamma * \
                                 ((readout_t1_batch[i]).tolist())[0]
                        y_away = float((r_t_batch[i])[1]) + tt_lstm_config.learn.gamma * \
                                 ((readout_t1_batch[i]).tolist())[1]
                        y_end = float((r_t_batch[i])[2]) + tt_lstm_config.learn.gamma * \
                                ((readout_t1_batch[i]).tolist())[2]
                        y_batch.append([y_home, y_away, y_end])

                # perform gradient step
                y_batch = np.asarray(y_batch)
                [diff, read_out, cost_out, summary_train, _] = sess.run(
                    [model.diff, model.read_out, model.cost, merge, model.train_step],
                    feed_dict={model.y_ph: y_batch,
                               model.trace_lengths_ph: trace_t0_batch,
                               model.rnn_input_ph: s_t0_batch,
                               model.home_away_indicator_ph: ha_id_t0_batch})

                v_diff_record.append(diff)

                if cost_out > 0.0001:
                    converge_flag = False
                global_counter += 1
                game_cost_record.append(cost_out)
                train_writer.add_summary(summary_train, global_step=global_counter)
                s_t0 = s_tl

                # print info
                # if print_flag:
                print ("cost of the network is" + str(cost_out))
                # if terminal or ((train_number - 1) / tt_lstm_config.learn.batch_size) % 5 == 1:
                # print ("TIMESTEP:", train_number, "Game:", game_number)
                home_avg = sum(read_out[:, 0]) / len(read_out[:, 0])
                away_avg = sum(read_out[:, 1]) / len(read_out[:, 1])
                end_avg = sum(read_out[:, 2]) / len(read_out[:, 2])
                print "home average:{0}, away average:{1}, end average:{2}".format(str(home_avg), str(away_avg),
                                                                                   str(end_avg))
                # if print_flag:
                #     print ("cost of the network is" + str(cost_out))

                if terminal:
                    print 'game{0} finishing'.format(str(game_number))
                    if (game_number - 1) % 300 == 0:
                        # save progress after a game
                        print 'saving game', game_number
                        saver.save(sess, save_network_dir + '/' + tt_lstm_config.learn.sport + '-game-',
                                   global_step=game_number)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
            write_game_average_csv([{"iteration": str(game_number / number_of_total_game + 1), "game": game_number,
                                     "cost_per_game_average": cost_per_game_average}],
                                   log_dir=save_log_dir)

        game_diff_record_all.append(game_diff_record_dict)


def run():
    sess = tf.Session()
    if FINE_TUNING:
        lr = tt_lstm_config.learn.learning_rate/10
    else:
        lr = tt_lstm_config.learn.learning_rate
    print tt_lstm_config.learn.learning_rate
    model = td_prediction_tt_embed(
        feature_number=tt_lstm_config.learn.feature_number,
        home_h_size=tt_lstm_config.Arch.HomeTower.home_h_size,
        away_h_size=tt_lstm_config.Arch.AwayTower.away_h_size,
        max_trace_length=tt_lstm_config.learn.max_trace_length,
        learning_rate=lr,
        embed_size=tt_lstm_config.learn.embed_size,
        output_layer_size=tt_lstm_config.learn.output_layer_size,
        home_lstm_layer_num=tt_lstm_config.Arch.HomeTower.lstm_layer_num,
        away_lstm_layer_num=tt_lstm_config.Arch.AwayTower.lstm_layer_num,
        dense_layer_num=tt_lstm_config.learn.dense_layer_num,
        apply_softmax=tt_lstm_config.learn.apply_softmax)
    model.initialize_ph()
    model.build()
    model.call()
    train_network(sess=sess, model=model, )
    sess.close()


if __name__ == '__main__':
    run()
