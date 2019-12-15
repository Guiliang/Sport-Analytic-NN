import numpy as np
from scipy import stats
from td_three_prediction_two_tower_lstm_v_correct_dir.correlation.correlation import Correlation
if __name__ == '__main__':
    game_info_path = '../resource/player_team_id_name_value.csv'
    correlation = Correlation(game_info_path=game_info_path, online_info_path_dict_all={})
    GIM_value_dict = correlation.get_rank_value_dict(rank_value_name='GIM2t')
    for name in ['PM', 'ALG', 'GIM', 'SI', 'EG']:
        rank_value_dict = correlation.get_rank_value_dict(rank_value_name=name)
        rank_value_list = []
        GIM_value_list = []
        for playerId in rank_value_dict.keys():
            if GIM_value_dict.get(playerId) is None or rank_value_dict.get(playerId) is None:
                continue
            rank_value_list.append(rank_value_dict.get(playerId))
            GIM_value_list.append(GIM_value_dict.get(playerId))
        rank_value_list = np.asarray(rank_value_list)
        GIM_value_list = np.asarray(GIM_value_list)
        print stats.ttest_rel(rank_value_list, GIM_value_list)
