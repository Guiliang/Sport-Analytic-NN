import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure.td_tt_lstm import td_prediction_tt_embed
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import handle_trace_length, \
    compromise_state_trace_length, \
    get_together_training_batch, write_game_average_csv
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_icehockey_game_data

tt_lstm_config_path = "./icehockey-config.yaml"
tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)

LOG_DIR = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_log_NN/Scale-tt-three-cut_together_log_train_feature" + str(
    tt_lstm_config.learn.feature_type) + "_batch" + str(
    tt_lstm_config.learn.batch_size) + "_iterate" + str(
    tt_lstm_config.learn.iterate_num) + "_lr" + str(
    tt_lstm_config.learn.learning_rate) + "_" + str(
    tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
    tt_lstm_config.learn.max_trace_length)
SAVED_NETWORK = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
    tt_lstm_config.learn.feature_type) + "_batch" + str(
    tt_lstm_config.learn.batch_size) + "_iterate" + str(
    tt_lstm_config.learn.iterate_num) + "_lr" + str(
    tt_lstm_config.learn.learning_rate) + "_" + str(
    tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
    tt_lstm_config.learn.max_trace_length)
DATA_STORE = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature" + str(
    tt_lstm_config.learn.feature_type) + "-scale-neg_reward" + tt_lstm_config.learn.if_correct_velocity + "_length-dynamic"

DIR_GAMES_ALL = os.listdir(DATA_STORE)
number_of_total_game = len(DIR_GAMES_ALL)


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

            if checkpoint and checkpoint.model_checkpoint_path:
                if tt_lstm_config.learn.model_train_continue:  # go the check point data
                    game_starting_point += 1
                    if game_number_checkpoint + 1 > game_starting_point:
                        continue

            v_diff_record = []
            game_number += 1
            game_cost_record = []
            state_trace_length, state_input, reward, ha_id = get_icehockey_game_data(
                data_store=DATA_STORE, dir_game=dir_game, config=tt_lstm_config)

            print ("\n load file" + str(dir_game) + " success")
            reward_count = sum(reward)
            print ("reward number" + str(reward_count))
            if len(state_input) != len(reward) or len(state_trace_length) != len(reward):
                raise Exception('state length does not equal to reward length')

            train_len = len(state_input)
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

                [readout_t1_batch] = sess.run([model.readout],
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
                    [model.diff, model.readout, model.cost, merge, model.train_step],
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
                    # save progress after a game
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
        apply_softmax=tt_lstm_config.learn.apply_softmax
    )
    model.initialize_ph()
    model.build()
    model.call()
    train_network(sess=sess, model=model, )
    sess.close()


if __name__ == '__main__':
    run()
