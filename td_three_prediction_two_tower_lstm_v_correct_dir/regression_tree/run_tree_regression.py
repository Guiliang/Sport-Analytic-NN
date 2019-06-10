import datetime
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name
from tree_regression_impact import TreeRegression

if __name__ == '__main__':
    test_flag = True
    if test_flag:
        data_path = '/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/'
        soccer_data_store_dir = "/Users/liu/Desktop/soccer-data-sample/Soccer-data/"
        min_sample_leaf = 1
    else:
        data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
        soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"
        min_sample_leaf = 20
    tt_lstm_config_path = '../soccer-config-v5.yaml'
    difference_type = 'back_difference_'
    action_selected = 'shot'
    cart_model_name = 'CART_soccer_impact_{0}_{1}.json'. \
        format(datetime.date.today().strftime("%Y%B%d"), difference_type)

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
    #                                          dir_all=soccer_dir_all, model_number=model_number)
    data_name = get_data_name(config=tt_lstm_config)

    TR = TreeRegression(cart_model_name=cart_model_name, data_name=data_name,
                        model_data_store_dir=soccer_data_store_dir, game_data_dir=data_path,
                        difference_type=difference_type, action_selected=action_selected,
                        min_sample_leaf=min_sample_leaf)
    all_input_list, all_impact_list = TR.gather_all_training_data()
    TR.cart_validation_model(np.asarray(all_input_list), np.asarray(all_impact_list),
                             np.asarray(all_input_list), np.asarray(all_impact_list),
                             read_model=False, test_flag=test_flag)
    TR.print_decision_path()
