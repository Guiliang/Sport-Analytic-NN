import json
import os
from data_processing_tools import read_features_within_events

if __name__ == '__main__':
    test_flag = True
    store_pm_dir = '../resource/pm_player_all.json'
    if test_flag:
        data_path = '/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/'
        soccer_data_store_dir = "/Users/liu/Desktop/soccer-data-sample/Soccer-data/"
    else:
        data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
        soccer_data_store_dir = "/cs/oschulte/Galen/Soccer-data/"

    global_player_pm = {}

    data_dir_all = os.listdir(data_path)

    for json_dir in data_dir_all:
        features_all = read_features_within_events(directory=json_dir, data_path=data_path,
                                                   feature_name_list=['action', 'playerId', 'home_away'])

        home_player_set = set()
        away_player_set = set()
        for features_dict in features_all:
            if features_dict.get('home_away') == 'H':
                home_player_set.add(features_dict.get('playerId'))
            else:
                away_player_set.add(features_dict.get('playerId'))

        home_player_list = list(home_player_set)
        away_player_list = list(away_player_set)

        for features_dict in features_all:
            if features_dict.get('action') == 'goal':
                if features_dict.get('home_away') == 'H':
                    player_list_plus = home_player_list
                    player_list_minus = away_player_list
                else:
                    player_list_plus = away_player_list
                    player_list_minus = home_player_list
                for player_id in player_list_plus:
                    if global_player_pm.get(player_id):
                        player_pm = global_player_pm.get(player_id)
                        player_pm += 1
                        global_player_pm.update({player_id:player_pm})
                    else:
                        player_pm = 1
                        global_player_pm.update({player_id: player_pm})
                for player_id in player_list_minus:
                    if global_player_pm.get(player_id):
                        player_pm = global_player_pm.get(player_id)
                        player_pm -= 1
                        global_player_pm.update({player_id:player_pm})
                    else:
                        player_pm = -1
                        global_player_pm.update({player_id: player_pm})

    with open(store_pm_dir, 'w') as f:
        json.dump(global_player_pm, f)

