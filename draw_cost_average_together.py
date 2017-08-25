import csv

import math
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.pyplot
label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

MAX_ITERATION = 50
BATCH_SIZE = 32

color_and_marker_dict = {
    'c1-MC': ['r', (3, 0), '-'],
    'c1-Sarsa': ['y', (4, 0), '-'],
    'c4-Sarsa': ['b', (4, 1), '-'],
    'FT-LSTM': ['g', (5, 0), '-'],
    'DP-LSTM': ['m', (5, 1), '-']
}

cost_data_directory_dict = {
    'c1-MC': 'log_NN/mc-Scale-three-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced_v_correct_',
    'c1-Sarsa': 'log_NN/Scale-three-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced_v_correct_',
    'c4-Sarsa': 'log_NN/Scale-three-c4-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced_v_correct_',
    # 'State MC_NN': 'log_NN/State-mc-Scale-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'State TD_NN': 'log_NN/Scale-state-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    # 'State TD_c4_NN': 'log_NN/State-Scale-c4-cut_log_entire_together_train_feature5_batch32_iterate50_lr1e-05-NEG_REWARD_GAMMA1_V3-Sequenced',
    'FT-LSTM': 'log_NN/Scale-fix_rnn_cut_together_log_train_feature5_batch{0}_iterate50_v1'.format(str(BATCH_SIZE)),
    # 'State FT_LSTM': 'log_NN/Scale-three-fix_rnn_cut_together_log_train_feature5_batch32_iterate50_v1_v_correct_'.format(str(BATCH_SIZE)),
    # 'State DT_LSTM': 'hybrid_sl_log_NN/State-Scale-cut_together_log_train_feature5_batch{0}_iterate30_lr1e-05_v3'.format(str(BATCH_SIZE)),
    'DP-LSTM': 'hybrid_sl_log_NN/Scale-three-cut_together_saved_networks_feature5_batch32_iterate50_lr1e-05_v3_v_correct_'.format(
        str(BATCH_SIZE))
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
        average_cost.append(math.log(cost_average))
    color_and_marker = color_and_marker_dict.get(cost_data_type)
    plt.plot(iterations, average_cost, label=cost_data_type, marker=color_and_marker[1], color=color_and_marker[0], linestyle = color_and_marker[2])


if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    plt.ticklabel_format(useOffset=False)
    for cost_data_type in cost_data_directory_dict.keys():
        cost_data_directory_whole = '/cs/oschulte/Galen/models/' + cost_data_directory_dict.get(
            cost_data_type) + '/avg_cost_record.csv'
        csv_dict_list = read_csv(cost_data_directory_whole)
        iteration_average_dict = compute_average_over_iteration(csv_dict_list)
        plot_cost_average(iteration_average_dict, cost_data_type)

    plt.legend(loc='upper right',fontsize=16)
    plt.title("Error Signal vs Epochs", fontsize=17)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.grid()
    plt.savefig("./cost_graph_dir/{0}".format(str(cost_data_directory_dict.keys())))
    plt.show()
