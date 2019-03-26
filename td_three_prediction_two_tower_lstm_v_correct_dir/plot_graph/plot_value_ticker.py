import scipy.io as sio
import os
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import find_game_dir
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import plot_game_value
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.print_tools import print_mark_info
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import normalize_data


if __name__ == '__main__':
    data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature5-scale" \
                     "-neg_reward_v_correct__length-dynamic/"
    data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data"
    tt_lstm_config_path = '../icehockey-config.yaml'
    home_team = 'Penguins'
    away_team = 'Canadiens'
    target_game_id = str(1403)
    dir_all = os.listdir(data_path)
    game_name_dir = find_game_dir(dir_all, data_path, target_game_id)

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4

    tt_lstm_config.learn.model_type = 'v3'  # TODO: comment me in the future
    tt_lstm_config.learn.batch_size = 32  # TODO: comment me in the future

    data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}_v_correct_". \
        format(str(tt_lstm_config.learn.feature_type),
               str(tt_lstm_config.learn.iterate_num),
               str(learning_rate_write),
               str(tt_lstm_config.learn.batch_size),
               str(tt_lstm_config.learn.max_trace_length),
               str(tt_lstm_config.learn.model_type))

    save_image_name = './icehockey-image/{0} v.s. {1} value_ticker.png'.format(home_team, away_team)
    home_max_index, away_max_index, home_maxs, away_maxs = plot_game_value(game_name_dir=game_name_dir,
                                                                           data_store_dir=data_store_dir,
                                                                           data_name=data_name,
                                                                           save_image_name=save_image_name,
                                                                           normalize_data=normalize_data,
                                                                           home_team=home_team,
                                                                           away_team=away_team)
    print_mark_info(data_store_dir, game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs)
