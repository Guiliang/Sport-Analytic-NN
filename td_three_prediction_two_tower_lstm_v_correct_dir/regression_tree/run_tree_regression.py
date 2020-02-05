import os
import pickle
import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import datetime
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name
from tree_regression_impact import TreeRegression

if __name__ == '__main__':
    test_flag = False
    run_cv = False
    action_selected = 'pass'
    # apply_cv = True

    # for min_sample_leaf in [5, 10, 30, 40, 50]:
    if test_flag:
        data_path = '/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/'
        soccer_data_store_dir = "/Users/liu/Desktop/soccer-data-sample/Soccer-data/"
        # min_sample_leaf = 1
    else:
        data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
        soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
        # min_sample_leaf = 20
    tt_lstm_config_path = '../soccer-config-v5.yaml'
    difference_type = 'back_difference_'
    print 'selected action is {0}'.format(action_selected)
    cart_model_name = 'CART_soccer_impact_{0}_{1}.json'. \
        format(action_selected, difference_type)

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = '5'
    elif learning_rate == 1e-4:
        learning_rate_write = '4'
    elif learning_rate == 0.0005:
        learning_rate_write = '5_5'

    model_number = 2101  # 2101, 7201, 7801 ,10501 ,13501 ,15301 ,18301*, 20701*
    # data_name = compute_values_for_all_games(config=tt_lstm_config, data_store_dir=soccer_data_store_dir,
    #                                          dir_all=soccer_dir_all, model_number=model_number

    data_name = get_data_name(config=tt_lstm_config)
    TR = TreeRegression(cart_model_name=cart_model_name, data_name=data_name,
                        model_data_store_dir=soccer_data_store_dir, game_data_dir=data_path,
                        difference_type=difference_type, action_selected=action_selected)
    # all_data_dir = os.listdir(soccer_data_store_dir)
    all_input_list, all_impact_list = TR.gather_impact_data()

    if run_cv:
        for min_sample_leaf in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print('min sample leaf is {0}'.format(str(min_sample_leaf)))
            cv_results_record_name = 'cv_{1}_running_record_leaf{0}.txt'.format(str(min_sample_leaf), action_selected)
            all_data_dir = os.listdir(soccer_data_store_dir)
            with open('./dt_record/' + cv_results_record_name, 'w') as f:
                f.close()

            apply_cv = True
            with open('./dt_record/' + cv_results_record_name, 'w') as f:
                f.close()
            for i in range(0, 5):
                # all_training_data_dir = all_data_dir[i*len(all_data_dir)/5:(i+1)*len(all_data_dir)/5]
                # all_testing_data_dir = all_data_dir[:i*len(all_data_dir)/5] + all_data_dir[(i + 1) * len(all_data_dir)/5:]
                # training_input_list, training_impact_list = TR.gather_impact_data(apply_cv, all_training_data_dir)
                # testing_input_list, testing_impact_list = TR.gather_impact_data(apply_cv, all_testing_data_dir)
                training_input_list = all_input_list[:i * len(all_input_list) / 5] + all_input_list[
                                                                                     (i + 1) * len(all_input_list) / 5:]
                testing_input_list = all_input_list[i * len(all_input_list) / 5:(i + 1) * len(all_input_list) / 5]
                training_impact_list = all_impact_list[:i * len(all_impact_list) / 5] + all_impact_list[(i + 1) * len(
                    all_impact_list) / 5:]
                testing_impact_list = all_impact_list[i * len(all_impact_list) / 5:(i + 1) * len(all_impact_list) / 5]
                print 'total training data collected length is {0}'.format(len(training_input_list))
                print 'total testing data collected length is {0}\n'.format(len(testing_input_list))
                mae, var_mae, mse, var_mse = TR.train_cart_validation_model(np.asarray(training_input_list),
                                                                            np.asarray(training_impact_list),
                                                                            np.asarray(testing_input_list),
                                                                            np.asarray(testing_impact_list),
                                                                            read_model=False,
                                                                            test_flag=test_flag,
                                                                            running_number=i,
                                                                            min_sample_leaf=min_sample_leaf)
                with open('./dt_record/' + cv_results_record_name, 'a') as f:
                    f.write('The {4} running, mae:{0}, var_mae:{1}, mse:{2}, var_mse:{3}\n'.
                            format(str(mae), str(var_mae), str(mse), str(var_mse), str(i)))
                with open('/cs/oschulte/Galen/soccer-models/dt_datas/dt_training_data_{0}_{1}.pkl'.
                                  format(action_selected, str(i)), 'wb') as f:
                    pickle.dump([training_input_list, training_impact_list], f)
                with open('/cs/oschulte/Galen/soccer-models/dt_datas/dt_testing_data_{0}_{1}.pkl'.
                                  format(action_selected, str(i)), 'wb') as f:
                    pickle.dump([testing_input_list, testing_impact_list], f)

    else:
        min_sample_leaf = 90
        mae, var_mae, mse, var_mse = TR.train_cart_validation_model(np.asarray(all_input_list),
                                                                    np.asarray(all_impact_list),
                                                                    np.asarray(all_input_list),
                                                                    np.asarray(all_impact_list),
                                                                    read_model=False,
                                                                    test_flag=test_flag,
                                                                    running_number='all',
                                                                    min_sample_leaf=min_sample_leaf)
        TR.read_cart_model(cart_model_name=cart_model_name + '_rn_all' + '_msf_' + str(min_sample_leaf))
        TR.print_decision_path()
