import os
import scipy.io as sio
import unicodedata
import matplotlib.pyplot as plt

FEATURE_TYPE = 5
ITERATE_NUM = 30
BATCH_SIZE = 48
MAX_LENGTH = 10
MODEL_TYPE = "v3"
DRAW_THREE = True
learning_rate = 1e-4
if learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

TARGET_GAMEID = str(1403)
HOME_TEAM = 'Penguins'
AWAY_TEAM = 'Canadiens'

if DRAW_THREE:
    save_image_dir = "./image/Three 2015-2016 NHL regular season {0} vs {1}_Iter{2}_lr{3}_Batch{4}.png".format(AWAY_TEAM, HOME_TEAM, str(ITERATE_NUM), str(learning_rate_write), str(BATCH_SIZE),)
else:
    save_image_dir = "./image/2015-2016 NHL regular season {0} vs {1}_Iter{2}_lr{3}_Batch{4}.png".format(AWAY_TEAM, HOME_TEAM, str(ITERATE_NUM), str(learning_rate_write), str(BATCH_SIZE))

data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data"
data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature5-scale-neg_reward_v_correct__length-dynamic/"
# state_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/State-Hybrid-RNN-Hockey-Training-All-feature5-scale-neg_reward_length-dynamic/"
dir_all = os.listdir(data_path)
data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}_v_correct_".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(learning_rate_write),
    str(BATCH_SIZE),
    str(MAX_LENGTH),
    str(MODEL_TYPE))
state_data_name = "model_three_state_cut_together_predict_Fea{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}_v_correct_".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(6),
    str(8),
    str(MAX_LENGTH),
    str(MODEL_TYPE))


def find_game_dir():
    for directory in dir_all:
        game = sio.loadmat(data_path + "/" + str(directory))
        gameId = (game['x'])['gameId'][0][0][0]
        gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
        if gameId == TARGET_GAMEID:
            game_name = directory
            print directory
            break
    return game_name.split(".")[0]


def plot_game_value(game_name_dir):
    game_value = sio.loadmat(data_store_dir + game_name_dir + "/" + data_name)
    game_value = game_value[data_name]
    # game_value_home = game_value[:, 0]/(game_value[:, 0]+game_value[:, 1])
    # game_value_away = game_value[:, 1]/(game_value[:, 0]+game_value[:, 1])
    game_value_home = game_value[:, 0]
    game_value_away = game_value[:, 1]
    game_value_end = game_value[:, 2]

    # find the index of max home and away
    home_max_index = game_value_home.argsort()[-20:][::-1]
    away_max_index = game_value_away.argsort()[-20:][::-1]
    home_maxs = game_value_home[home_max_index]
    away_maxs = game_value_away[away_max_index]

    game_value_diff = game_value_home - game_value_away
    game_value_rate = game_value_home / game_value_away

    event_numbers = [d for d in range(1, len(game_value_diff) + 1)]

    plt.figure(figsize=(15, 6))
    if DRAW_THREE:
        plt.plot(event_numbers, game_value_home, label="q_home")
        plt.plot(event_numbers, game_value_away, label="q_away")
        plt.plot(event_numbers[0:len(game_value_end)-1], game_value_end[0:len(game_value_end)-1], label="q_end")
    else:
        plt.plot(event_numbers, game_value_diff, label="q_home-q_away")
    plt.title("2015-2016 NHL regular season {0}(Away) vs {1}(Home)".format(AWAY_TEAM, HOME_TEAM))
    plt.xlabel("event number")
    plt.ylabel("value")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(save_image_dir)

    return home_max_index, away_max_index, home_maxs, away_maxs


def print_mark_info(game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs):
    training_data_info = sio.loadmat(data_store_dir + game_name_dir + "/" + "training_data_dict_all_name.mat")
    training_data_info = training_data_info["training_data_dict_all_name"]
    home_training_data_info = training_data_info[home_max_index]
    away_training_data_info = training_data_info[away_max_index]
    print "\nhome_training_data_info"
    print zip(home_maxs.tolist(), home_max_index.tolist())
    print home_training_data_info
    print "\naway_training_data_info"
    print zip(away_maxs.tolist(), away_max_index.tolist())
    print away_training_data_info


if __name__ == '__main__':
    game_name_dir = find_game_dir()
    home_max_index, away_max_index, home_maxs, away_maxs = plot_game_value(game_name_dir)
    print_mark_info(game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs)

"""
home_training_data_info
[(1.0078580379486084, 1340), (1.0039684772491455, 1576), (0.7500014901161194, 1339), (0.7144783139228821, 627), (0.7122133374214172, 628), (0.7084869742393494, 625), (0.7059627175331116, 626), (0.7045314311981201, 630), (0.6995928287506104, 1956), (0.6979097723960876, 629)]
[ u"{'xAdjCoord': 59.0, 'scoreDifferential': -1.0, 'yAdjCoord': 18.5, 'away': 0.0, 'time remained': 2068.54, 'Penalty': 0.0, 'duration': 0.43, 'angel2gate': 0.55, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 1.0, 'event_outcome': 1.0}                  "
 u"{'xAdjCoord': 55.5, 'scoreDifferential': -1.0, 'yAdjCoord': 13.0, 'away': 0.0, 'time remained': 1794.29, 'Penalty': 0.0, 'duration': 0.07, 'angel2gate': 0.37, 'velocity_x': 67.43, 'velocity_y': 7.49, 'action': 'goal', 'home': 1.0, 'event_outcome': 1.0}               "
 u"{'xAdjCoord': 59.0, 'scoreDifferential': -1.0, 'yAdjCoord': 18.5, 'away': 0.0, 'time remained': 2068.97, 'Penalty': 0.0, 'duration': 1.1, 'angel2gate': 0.55, 'velocity_x': 30.88, 'velocity_y': -9.99, 'action': 'shot', 'home': 1.0, 'event_outcome': 1.0}               "
 u"{'xAdjCoord': 31.0, 'scoreDifferential': -1.0, 'yAdjCoord': 20.5, 'away': 0.0, 'time remained': 2851.35, 'Penalty': 1.0, 'duration': 0.77, 'angel2gate': 0.34, 'velocity_x': 3.91, 'velocity_y': -0.65, 'action': 'pass', 'home': 1.0, 'event_outcome': 1.0}               "
 u"{'xAdjCoord': 49.0, 'scoreDifferential': -1.0, 'yAdjCoord': 40.5, 'away': 0.0, 'time remained': 2850.32, 'Penalty': 1.0, 'duration': 1.03, 'angel2gate': 0.79, 'velocity_x': 17.4, 'velocity_y': 19.34, 'action': 'reception', 'home': 1.0, 'event_outcome': 1.0}          "
 u"{'xAdjCoord': 21.5, 'scoreDifferential': -1.0, 'yAdjCoord': 0.5, 'away': 0.0, 'time remained': 2852.75, 'Penalty': 1.0, 'duration': 1.6, 'angel2gate': 0.01, 'velocity_x': 5.0, 'velocity_y': 19.36, 'action': 'pass', 'home': 1.0, 'event_outcome': 1.0}                  "
 u"{'xAdjCoord': 28.0, 'scoreDifferential': -1.0, 'yAdjCoord': 21.0, 'away': 0.0, 'time remained': 2852.12, 'Penalty': 1.0, 'duration': 0.63, 'angel2gate': 0.33, 'velocity_x': 10.25, 'velocity_y': 32.34, 'action': 'reception', 'home': 1.0, 'event_outcome': 1.0}         "
 u"{'xAdjCoord': 49.5, 'scoreDifferential': -1.0, 'yAdjCoord': 38.0, 'away': 0.0, 'time remained': 2849.92, 'Penalty': 1.0, 'duration': 0.37, 'angel2gate': 0.77, 'velocity_x': 1.36, 'velocity_y': -1.36, 'action': 'pass', 'home': 1.0, 'event_outcome': -1.0}              "
 u"{'xAdjCoord': 78.5, 'scoreDifferential': 0.0, 'yAdjCoord': -1.5, 'away': 0.0, 'time remained': 1358.32, 'Penalty': 0.0, 'duration': 0.27, 'angel2gate': 0.14, 'velocity_x': 1.87, 'velocity_y': 1.87, 'action': 'shot', 'home': 1.0, 'event_outcome': 1.0}                 "
 u"{'xAdjCoord': -49.0, 'scoreDifferential': 1.0, 'yAdjCoord': -38.5, 'away': 1.0, 'time remained': 2850.28, 'Penalty': -1.0, 'duration': 0.03, 'angel2gate': 0.27, 'velocity_x': 0.0, 'velocity_y': 59.94, 'action': 'block', 'home': 0.0, 'event_outcome': 1.0}             "]

away_training_data_info
[(1.0155373811721802, 314), (1.0096704959869385, 1496), (0.9936325550079346, 2361), (0.7936543226242065, 1478), (0.7879369258880615, 2047), (0.787600040435791, 1477), (0.7829372882843018, 1495), (0.7646687030792236, 1479), (0.7582378387451172, 437), (0.7527756690979004, 1483)]
[ u"{'xAdjCoord': 42.5, 'scoreDifferential': 0.0, 'yAdjCoord': 7.5, 'away': 1.0, 'time remained': 3254.45, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.16, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                     "
 u"{'xAdjCoord': 62.0, 'scoreDifferential': 0.0, 'yAdjCoord': 30.0, 'away': 1.0, 'time remained': 1876.08, 'Penalty': 1.0, 'duration': 0.0, 'angel2gate': 0.84, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                    "
 u"{'xAdjCoord': 74.0, 'scoreDifferential': 0.0, 'yAdjCoord': -10.5, 'away': 1.0, 'time remained': 884.12, 'Penalty': 0.0, 'duration': 0.07, 'angel2gate': 0.61, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                   "
 u"{'xAdjCoord': 46.5, 'scoreDifferential': 0.0, 'yAdjCoord': 6.5, 'away': 1.0, 'time remained': 1901.23, 'Penalty': 1.0, 'duration': 1.03, 'angel2gate': 0.15, 'velocity_x': 11.12, 'velocity_y': 30.45, 'action': 'reception', 'home': 0.0, 'event_outcome': 1.0}           "
 u"{'xAdjCoord': 79.0, 'scoreDifferential': 0.0, 'yAdjCoord': 4.0, 'away': 1.0, 'time remained': 1257.12, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.38, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}                     "
 u"{'xAdjCoord': -35.0, 'scoreDifferential': 0.0, 'yAdjCoord': 25.0, 'away': 0.0, 'time remained': 1902.27, 'Penalty': -1.0, 'duration': 0.27, 'angel2gate': 0.2, 'velocity_x': 50.57, 'velocity_y': -35.59, 'action': 'block', 'home': 1.0, 'event_outcome': -1.0}           "
 u"{'xAdjCoord': 62.0, 'scoreDifferential': 0.0, 'yAdjCoord': 30.0, 'away': 1.0, 'time remained': 1876.08, 'Penalty': 1.0, 'duration': 1.53, 'angel2gate': 0.84, 'velocity_x': -10.1, 'velocity_y': -0.33, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}               "
 u"{'xAdjCoord': 44.5, 'scoreDifferential': 0.0, 'yAdjCoord': 7.5, 'away': 1.0, 'time remained': 1901.1, 'Penalty': 1.0, 'duration': 0.13, 'angel2gate': 0.17, 'velocity_x': -14.99, 'velocity_y': 7.49, 'action': 'pass', 'home': 0.0, 'event_outcome': 1.0}                 "
 u"{'xAdjCoord': 77.5, 'scoreDifferential': 1.0, 'yAdjCoord': -4.5, 'away': 1.0, 'time remained': 3083.68, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.37, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}                    "
 u"{'xAdjCoord': 79.5, 'scoreDifferential': 0.0, 'yAdjCoord': -36.0, 'away': 1.0, 'time remained': 1892.56, 'Penalty': 1.0, 'duration': 4.44, 'angel2gate': 1.31, 'velocity_x': 5.63, 'velocity_y': -0.79, 'action': 'pass', 'home': 0.0, 'event_outcome': 1.0} 
"""
