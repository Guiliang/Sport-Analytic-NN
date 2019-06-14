import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

from td_three_prediction_two_tower_lstm_v_correct_dir.correlation.correlation import Correlation

if __name__ == '__main__':
    game_info_path = '../resource/player_team_id_name_value.csv'
    online_info_path_list = ['../resource/Soccer_summary.csv',
                             '../resource/Soccer_defensive.csv',
                             '../resource/Soccer_offensive.csv'
                             ]
    correlation = Correlation(game_info_path=game_info_path, online_info_path_list=online_info_path_list)
    correlation.compute_all_correlations()
