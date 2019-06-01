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
    get_together_training_batch, write_game_average_csv

tt_lstm_config_path = "../soccer-config-v3.yaml"
tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
DATA_STORE = "/cs/oschulte/Galen/Soccer-data/"
DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)
TRAIN_FLAG = False
if TRAIN_FLAG:
    train_msg = 'Train_'
    DIR_GAMES_ALL = DIR_GAMES_ALL[:2400]
else:
    train_msg = ''


if tt_lstm_config.learn.merge_tower:
    merge_msg = 'm'
else:
    merge_msg = 's'

LOG_DIR = "{0}/oschulte/Galen/soccer-models/hybrid_sl_log_NN" \
          "/{1}Scale-tt{9}-three-cut_together_log_feature{2}" \
          "_batch{3}_iterate{4}_lr{5}_{6}{7}_MaxTL{8}".format(tt_lstm_config.learn.save_mother_dir,
                                                              train_msg,
                                                              str(tt_lstm_config.learn.feature_type),
                                                              str(tt_lstm_config.learn.batch_size),
                                                              str(tt_lstm_config.learn.iterate_num),
                                                              str(tt_lstm_config.learn.learning_rate),
                                                              str(tt_lstm_config.learn.model_type),
                                                              str(tt_lstm_config.learn.if_correct_velocity),
                                                              str(tt_lstm_config.learn.max_trace_length),
                                                              merge_msg)

SAVED_NETWORK = "{0}/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/" \
                "{1}Scale-tt{9}-three-cut_together_saved_networks_feature{2}" \
                "_batch{3}_iterate{4}_lr{5}_{6}{7}_MaxTL{8}".format(tt_lstm_config.learn.save_mother_dir,
                                                                    train_msg,
                                                                    str(tt_lstm_config.learn.feature_type),
                                                                    str(tt_lstm_config.learn.batch_size),
                                                                    str(tt_lstm_config.learn.iterate_num),
                                                                    str(tt_lstm_config.learn.learning_rate),
                                                                    str(tt_lstm_config.learn.model_type),
                                                                    str(tt_lstm_config.learn.if_correct_velocity),
                                                                    str(tt_lstm_config.learn.max_trace_length),
                                                                    merge_msg)


def train_network(sess, model, print_parameters=False):
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver()
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    if tt_lstm_config.learn.model_train_continue:  # resume the training
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
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
        for dir_game in DIR_GAMES_ALL:

            if tt_lstm_config.learn.model_train_continue:
                # if checkpoint and checkpoint.model_checkpoint_path:
                if tt_lstm_config.learn.model_train_continue:  # go the check point data
                    game_starting_point += 1
                    if game_number_checkpoint + 1 > game_starting_point:
                        continue

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            game_files = os.listdir(DATA_STORE + "/" + dir_game)
            for filename in game_files:
                if "reward" in filename:
                    reward_name = filename
                elif "state" in filename:
                    state_input_name = filename
                elif "trace" in filename:
                    state_trace_length_name = filename
                elif "home_away" in filename:
                    ha_id_name = filename

            reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)['reward']
            state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)['state']
            ha_id = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + ha_id_name)["home_away"][0].astype(int)
            # state_input = (state_input['dynamic_feature_input'])
            # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
            # state_output = state_output['hybrid_output_state']
            state_trace_length = sio.loadmat(
                DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
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
                    if (game_number - 1) % 5000 == 0:
                        # save progress after a game
                        print 'saving game', game_number
                        saver.save(sess, SAVED_NETWORK + '/' + tt_lstm_config.learn.sport + '-game-',
                                   global_step=game_number)
                    v_diff_record_average = sum(v_diff_record) / len(v_diff_record)
                    game_diff_record_dict.update({dir_game: v_diff_record_average})
                    break

                    # break
            cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
            write_game_average_csv([{"iteration": str(game_number / number_of_total_game + 1), "game": game_number,
                                     "cost_per_game_average": cost_per_game_average}],
                                   log_dir=LOG_DIR)

        game_diff_record_all.append(game_diff_record_dict)


def run():
    sess = tf.Session()
    model = td_prediction_tt_embed(
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
        apply_softmax=tt_lstm_config.learn.apply_softmax)
    model.initialize_ph()
    model.build()
    model.call()
    train_network(sess=sess, model=model, )
    sess.close()


if __name__ == '__main__':
    run()
