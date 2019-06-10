import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')
import pickle
import json
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name
from td_three_prediction_two_tower_lstm_v_correct_dir.round_by_round_correlations.rbr_correlation import \
    RoundByRoundCorrelation

if __name__ == "__main__":
    raw_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    model_data_store_dir = "/cs/oschulte/Galen/Soccer-data"
    interested_metric = ['GIM2t', 'GIM', 'EG', 'SI']  # ['GIM', 'SI']
    player_summary_dir = '../resource/Soccer_summary.csv'
    game_info_path = '../resource/player_team_id_name_value.csv'
    rbr_correlation = RoundByRoundCorrelation(raw_data_path, interested_metric, player_summary_dir,
                                              model_data_store_dir, game_info_path)
    # team_game_dict = rbr_correlation.read_team_by_date()
    # pickle.dump(team_game_dict, open('./tmp_stores/team_game_dict.pkl', 'w'))
    team_game_dict = pickle.load(open('./tmp_stores/team_game_dict.pkl', 'r'))
    game_by_round_dict = rbr_correlation.compute_game_by_round(team_game_dict=team_game_dict)
    total_game = 0
    game_ste = {}
    for values in game_by_round_dict.values():
        total_game += len(values)
        for value in values:
            if game_ste.get(value.split('$')[-1]) is not None:
                number = game_ste.get(value.split('$')[-1]) + 1
            else:
                number = 1
            game_ste.update({value.split('$')[-1]: number})
    print total_game / 2
    player_id_info_dict = rbr_correlation.compute_player_season_totals()
    correlated_coefficient_round_by_round_all = {}
    for metric in interested_metric:
        if metric == 'GIM2t':
            rbr_correlation.difference_type = 'back_difference_'
            tt_lstm_config_path = "../soccer-config-v3.yaml"
            tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
            data_name = get_data_name(config=tt_lstm_config)
            rbr_correlation.data_name = data_name
        elif metric == 'EG':
            rbr_correlation.difference_type = 'expected_goal'
            tt_lstm_config_path = "../soccer-config-v3.yaml"
            tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
            data_name = get_data_name(config=tt_lstm_config)
            rbr_correlation.data_name = data_name
        elif metric == 'GIM':
            rbr_correlation.difference_type = 'back_difference_'
            tt_lstm_config_path = "../soccer-config.yaml"
            tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
            data_name = get_data_name(config=tt_lstm_config, if_old=True)
            rbr_correlation.data_name = data_name
        elif metric == 'SI':
            rbr_correlation.difference_type = 'expected_goal'
            data_name = 'markov_impact_values.json'
            rbr_correlation.data_name = data_name
        correlated_coefficient_round_by_round = rbr_correlation.compute_correlations_by_round(
            player_id_info_dict=player_id_info_dict, game_by_round_dict=game_by_round_dict,
            metric_name=metric)
        correlated_coefficient_round_by_round_all.update({metric: correlated_coefficient_round_by_round})
    with open('round_by_round_correlation.json', 'w') as outfile:
        json.dump(obj=correlated_coefficient_round_by_round_all, fp=outfile)

    print 'still working'