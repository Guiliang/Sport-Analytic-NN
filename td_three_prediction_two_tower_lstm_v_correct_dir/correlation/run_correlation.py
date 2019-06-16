import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

from td_three_prediction_two_tower_lstm_v_correct_dir.correlation.correlation import Correlation

if __name__ == '__main__':
    game_info_path = '../resource/player_team_id_name_value.csv'
    online_info_path_list = {
        'Championship': ['../resource/whoScored/Championship/Championship_summary.csv',
                         '../resource/whoScored/Championship/Championship_defensive.csv',
                         '../resource/whoScored/Championship/Championship_offensive.csv',
                         '../resource/whoScored/Championship/Championship_passing.csv'
                         ],
        'PremierLeague': ['../resource/whoScored/PremierLeague/Premier_League_summary.csv',
                          '../resource/whoScored/PremierLeague/Premier_League_defensive.csv',
                          '../resource/whoScored/PremierLeague/Premier_League_offensive.csv',
                          '../resource/whoScored/PremierLeague/Premier_League_passing.csv'
                          ]
    }
    correlation = Correlation(game_info_path=game_info_path, online_info_path_dict_all=online_info_path_list)
    correlation.compute_all_correlations()
