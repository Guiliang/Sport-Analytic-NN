import csv

import math
import matplotlib.pyplot as plt

MAX_ITERATION = 50
BATCH_SIZE = 32

cost_data_directory_dict = {
    # 'Action State MC_NN': 'log_NN/mc-Scale-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'Action State TD_NN': 'log_NN/Scale-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'Action State TD_c4_NN': 'log_NN/Scale-c4-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'State MC_NN': 'log_NN/State-mc-Scale-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'State TD_NN': 'log_NN/Scale-state-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'State TD_c4_NN': 'log_NN/State-Scale-c4-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    'Action State FT_LSTM': 'log_NN/Scale-fix_rnn_cut_together_log_train_feature5_batch{0}_iterate50_v1'.format(str(BATCH_SIZE)),
    # 'State FT_LSTM': 'log_NN/State-Scale-fix_rnn_cut_together_log_train_feature5_batch{0}_iterate50_v1'.format(str(BATCH_SIZE)),
    # 'State DT_LSTM': 'hybrid_sl_log_NN/State-Scale-cut_together_log_train_feature5_batch{0}_iterate30_lr1e-05_v3'.format(str(BATCH_SIZE)),
    'Action State PT-LSTM': 'hybrid_sl_log_NN/Scale-three-cut_together_saved_networks_feature5_batch32_iterate50_lr1e-05_v3_v_correct_'.format(str(BATCH_SIZE))
}


def read_csv(csv_name):
    csv_dict_list = []
    with open(csv_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_dict_list.append(row)
    return csv_dict_list


def compute_average_over_iteration(csv_dict_list):
    iteration_average_dict = {}
    for dict in csv_dict_list:
        iteration_num = int(dict.get('iteration'))
        cost_per_game_average = float(dict.get('cost_per_game_average'))
        game = int(dict.get('game'))
        if game % 1140 == 0:
            iteration_num -= 1
        cost_sum_and_game_number = iteration_average_dict.get(iteration_num)
        if cost_sum_and_game_number is not None:
            cost_sum = cost_sum_and_game_number.get('cost_sum')
            cost_sum = cost_sum + cost_per_game_average
            game_number = cost_sum_and_game_number.get('game_number')
            game_number += 1
            cost_sum_and_game_number.update({'cost_sum': cost_sum, 'game_number': game_number})
            iteration_average_dict.update({iteration_num: cost_sum_and_game_number})
        else:
            cost_sum = cost_per_game_average
            game_number = 1
            cost_sum_and_game_number = {'cost_sum': cost_sum, 'game_number': game_number}
            iteration_average_dict.update({iteration_num: cost_sum_and_game_number})
    return iteration_average_dict


def plot_cost_average(iteration_average_dict, cost_data_type):
    iteration_length = len(iteration_average_dict.keys())
    iterations = []
    average_cost = []
    for iteration_num in range(1, iteration_length + 1):
        if iteration_num > MAX_ITERATION:
            break
        iterations.append(iteration_num)
        cost_sum_and_game_number = iteration_average_dict.get(iteration_num)
        cost_sum = cost_sum_and_game_number.get('cost_sum')
        game_number = cost_sum_and_game_number.get('game_number')
        cost_average = math.sqrt(float(cost_sum) / (game_number * 32))
        average_cost.append(cost_average)

    plt.plot(iterations, average_cost, label=cost_data_type)


if __name__ == '__main__':
    for cost_data_type in cost_data_directory_dict.keys():
        cost_data_directory_whole = '/cs/oschulte/Galen/models/' + cost_data_directory_dict.get(
            cost_data_type) + '/avg_cost_record.csv'
        csv_dict_list = read_csv(cost_data_directory_whole)
        iteration_average_dict = compute_average_over_iteration(csv_dict_list)
        plot_cost_average(iteration_average_dict, cost_data_type)
    plt.legend(loc='upper right')
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Cost", fontsize=15)
    plt.show()
