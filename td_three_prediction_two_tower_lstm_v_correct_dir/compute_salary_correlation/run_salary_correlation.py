import numpy as np
import td_three_prediction_two_tower_lstm_v_correct_dir.resource.salary_1617 as salary_1617_dict
import td_three_prediction_two_tower_lstm_v_correct_dir.resource.salary_1718 as salary_1718_dict
import td_three_prediction_two_tower_lstm_v_correct_dir.resource.salary_1819 as salary_1819_dict
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_markov_rank_value, \
    get_GIM_rank_value

ranking_dir_dict = {
    'GIM': ['GIM', '../compute_impact/player_impact/ijcai_soccer_player_GIM_2019June01.json'],
    'SI': ['', '../resource/bak_soccer_player_markov_impact-2019June04.json'],
    'GIM2t': ['GIM', '../compute_impact/player_impact/soccer_player_GIM_2019June01.json'],
    'EG': ['GIM', '../compute_impact/player_impact/bak-soccer_player_GIM_2019June05_expected_goal.json']
    # 'ALG': ''
}
salary_1617_dict = salary_1617_dict.salary
salary_1718_dict = salary_1718_dict.salary
salary_1819_dict = salary_1819_dict.salary
for rank_value_name in ranking_dir_dict.keys():
    if rank_value_name == 'GIM' or rank_value_name == 'GIM2t' or rank_value_name == 'EG':
        rank_value_dict = get_GIM_rank_value(rank_value_name, ranking_dir_dict)
    else:
        rank_value_dict = get_markov_rank_value(rank_value_name, ranking_dir_dict)
    salary_list = []
    rank_value_list = []
    for player_id in rank_value_dict.keys():
        try:
            salary = salary_1718_dict.get(int(player_id))
        except:
            salary = salary_1718_dict.get(player_id)
        if salary is None:
            continue
        else:
            salary_list.append(salary)
            rank_value_list.append(rank_value_dict.get(player_id))
    correlation = np.corrcoef(salary_list, rank_value_list)
    print correlation

print 'still working'
