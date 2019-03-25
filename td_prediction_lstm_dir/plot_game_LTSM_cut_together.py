import os
import scipy.io as sio
import unicodedata
import matplotlib.pyplot as plt

FEATURE_TYPE = 5
ITERATE_NUM = 30
BATCH_SIZE = 32
MAX_LENGTH = 10
MODEL_TYPE = "v3"
learning_rate = 1e-5
if learning_rate == 1e-5:
    learning_rate_write = 5
elif learning_rate == 1e-4:
    learning_rate_write = 4

TARGET_GAMEID = str(1403)
HOME_TEAM = 'Penguins'
AWAY_TEAM = 'Canadiens'

save_image_dir = "./icehockey-image/2015-2016 NHL regular season {0} vs {1}.png".format(AWAY_TEAM, HOME_TEAM)
data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data"
data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature5-scale-neg_reward_length-dynamic/"
state_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/State-Hybrid-RNN-Hockey-Training-All-feature5-scale-neg_reward_length-dynamic/"
dir_all = os.listdir(data_path)
data_name = "model_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
    str(FEATURE_TYPE),
    str(ITERATE_NUM),
    str(learning_rate_write),
    str(BATCH_SIZE),
    str(MAX_LENGTH),
    str(MODEL_TYPE))
state_data_name = "model_state_cut_together_predict_Fea{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
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
            break
    return game_name.split(".")[0]


def plot_game_value(game_name_dir):
    game_value = sio.loadmat(data_store_dir + game_name_dir + "/" + data_name)
    game_value = game_value[data_name]
    # game_value_home = game_value[:, 0]/(game_value[:, 0]+game_value[:, 1])
    # game_value_away = game_value[:, 1]/(game_value[:, 0]+game_value[:, 1])
    game_value_home = game_value[:, 0]
    game_value_away = game_value[:, 1]

    # find the index of max home and away
    home_max_index = game_value_home.argsort()[-10:][::-1]
    away_max_index = game_value_away.argsort()[-10:][::-1]
    home_maxs = game_value_home[home_max_index]
    away_maxs = game_value_away[away_max_index]

    game_value_diff = game_value_home - game_value_away
    game_value_rate = game_value_home / game_value_away

    game_state_value = sio.loadmat(state_data_store_dir + game_name_dir + "/" + state_data_name)
    game_state_value = game_state_value[state_data_name]
    # game_state_value_home = game_state_value[:, 0]/(game_state_value[:, 0]+game_state_value[:, 1])
    # game_state_value_away = game_state_value[:, 1]/(game_state_value[:, 0]+game_state_value[:, 1])
    game_state_value_home = game_state_value[:, 0]
    game_state_value_away = game_state_value[:, 1]
    game_state_value_diff = game_state_value_home - game_state_value_away
    game_state_value_rate = game_state_value_home / game_state_value_away

    event_numbers = [d for d in range(1, len(game_value_diff) + 1)]

    plt.figure(figsize=(15, 6))
    plt.plot(event_numbers, game_state_value_diff, label="v_home-v_away")
    plt.plot(event_numbers, game_value_diff, label="q_home-q_away")
    plt.title("2015-2016 NHL regular season {0} vs {1}".format(AWAY_TEAM, HOME_TEAM))
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
[(1.0080910921096802, 1340), (0.9888582825660706, 1576), (0.717399537563324, 627), (0.7112974524497986, 625), (0.7082575559616089, 626), (0.7075169682502747, 628), (0.7051160931587219, 629), (0.6981307864189148, 623), (0.6973187327384949, 613), (0.6952045559883118, 630)]
[ u"{'xAdjCoord': 59.0, 'scoreDifferential': -1.0, 'yAdjCoord': 18.5, 'away': 0.0, 'time remained': 2068.54, 'Penalty': 0.0, 'duration': 0.43, 'angel2gate': 0.55, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 1.0, 'event_outcome': 1.0}                     "
 u"{'xAdjCoord': 55.5, 'scoreDifferential': -1.0, 'yAdjCoord': 13.0, 'away': 0.0, 'time remained': 1794.29, 'Penalty': 0.0, 'duration': 0.07, 'angel2gate': 0.37, 'velocity_x': 1595.9, 'velocity_y': 382.12, 'action': 'goal', 'home': 1.0, 'event_outcome': 1.0}               "
 u"{'xAdjCoord': 31.0, 'scoreDifferential': -1.0, 'yAdjCoord': 20.5, 'away': 0.0, 'time remained': 2851.35, 'Penalty': 1.0, 'duration': 0.77, 'angel2gate': 0.34, 'velocity_x': 3.91, 'velocity_y': -0.65, 'action': 'pass', 'home': 1.0, 'event_outcome': 1.0}                  "
 u"{'xAdjCoord': 21.5, 'scoreDifferential': -1.0, 'yAdjCoord': 0.5, 'away': 0.0, 'time remained': 2852.75, 'Penalty': 1.0, 'duration': 1.6, 'angel2gate': 0.01, 'velocity_x': 5.0, 'velocity_y': 19.36, 'action': 'pass', 'home': 1.0, 'event_outcome': 1.0}                     "
 u"{'xAdjCoord': 28.0, 'scoreDifferential': -1.0, 'yAdjCoord': 21.0, 'away': 0.0, 'time remained': 2852.12, 'Penalty': 1.0, 'duration': 0.63, 'angel2gate': 0.33, 'velocity_x': 10.25, 'velocity_y': 32.34, 'action': 'reception', 'home': 1.0, 'event_outcome': 1.0}            "
 u"{'xAdjCoord': 49.0, 'scoreDifferential': -1.0, 'yAdjCoord': 40.5, 'away': 0.0, 'time remained': 2850.32, 'Penalty': 1.0, 'duration': 1.03, 'angel2gate': 0.79, 'velocity_x': 17.4, 'velocity_y': 19.34, 'action': 'reception', 'home': 1.0, 'event_outcome': 1.0}             "
 u"{'xAdjCoord': -49.0, 'scoreDifferential': 1.0, 'yAdjCoord': -38.5, 'away': 1.0, 'time remained': 2850.28, 'Penalty': -1.0, 'duration': 0.03, 'angel2gate': 0.27, 'velocity_x': -2937.06, 'velocity_y': -2367.63, 'action': 'block', 'home': 0.0, 'event_outcome': 1.0}        "
 u"{'xAdjCoord': -9.0, 'scoreDifferential': -1.0, 'yAdjCoord': 8.0, 'away': 0.0, 'time remained': 2855.09, 'Penalty': 1.0, 'duration': 0.43, 'angel2gate': 0.08, 'velocity_x': 10.37, 'velocity_y': 10.37, 'action': 'pass', 'home': 1.0, 'event_outcome': 1.0}                  "
 u"{'xAdjCoord': 20.5, 'scoreDifferential': -1.0, 'yAdjCoord': -31.0, 'away': 0.0, 'time remained': 2868.6, 'Penalty': 1.0, 'duration': 0.8, 'angel2gate': 0.42, 'velocity_x': 25.6, 'velocity_y': 1.87, 'action': 'puckprotection', 'home': 1.0, 'event_outcome': 1.0}          "
 u"{'xAdjCoord': 49.5, 'scoreDifferential': -1.0, 'yAdjCoord': 38.0, 'away': 0.0, 'time remained': 2849.92, 'Penalty': 1.0, 'duration': 0.37, 'angel2gate': 0.77, 'velocity_x': 268.37, 'velocity_y': 208.43, 'action': 'pass', 'home': 1.0, 'event_outcome': -1.0}              "]

away_training_data_info
[(1.0205165147781372, 2361), (1.0147039890289307, 1496), (1.0023695230484009, 314), (0.7907000780105591, 1477), (0.787431538105011, 437), (0.7855710983276367, 1478), (0.779913067817688, 2047), (0.7733984589576721, 1495), (0.7574598789215088, 1479), (0.7471904754638672, 1483)]
[ u"{'xAdjCoord': 74.0, 'scoreDifferential': 0.0, 'yAdjCoord': -10.5, 'away': 1.0, 'time remained': 884.12, 'Penalty': 0.0, 'duration': 0.07, 'angel2gate': 0.61, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                      "
 u"{'xAdjCoord': 62.0, 'scoreDifferential': 0.0, 'yAdjCoord': 30.0, 'away': 1.0, 'time remained': 1876.08, 'Penalty': 1.0, 'duration': 0.0, 'angel2gate': 0.84, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                       "
 u"{'xAdjCoord': 42.5, 'scoreDifferential': 0.0, 'yAdjCoord': 7.5, 'away': 1.0, 'time remained': 3254.45, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.16, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'goal', 'home': 0.0, 'event_outcome': 1.0}                        "
 u"{'xAdjCoord': -35.0, 'scoreDifferential': 0.0, 'yAdjCoord': 25.0, 'away': 0.0, 'time remained': 1902.27, 'Penalty': -1.0, 'duration': 0.27, 'angel2gate': 0.2, 'velocity_x': -312.81, 'velocity_y': 222.9, 'action': 'block', 'home': 1.0, 'event_outcome': -1.0}             "
 u"{'xAdjCoord': 77.5, 'scoreDifferential': 1.0, 'yAdjCoord': -4.5, 'away': 1.0, 'time remained': 3083.68, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.37, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}                       "
 u"{'xAdjCoord': 46.5, 'scoreDifferential': 0.0, 'yAdjCoord': 6.5, 'away': 1.0, 'time remained': 1901.23, 'Penalty': 1.0, 'duration': 1.03, 'angel2gate': 0.15, 'velocity_x': 78.79, 'velocity_y': -17.89, 'action': 'reception', 'home': 0.0, 'event_outcome': 1.0}             "
 u"{'xAdjCoord': 79.0, 'scoreDifferential': 0.0, 'yAdjCoord': 4.0, 'away': 1.0, 'time remained': 1257.12, 'Penalty': 0.0, 'duration': 0.0, 'angel2gate': 0.38, 'velocity_x': 0.0, 'velocity_y': 0.0, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}                        "
 u"{'xAdjCoord': 62.0, 'scoreDifferential': 0.0, 'yAdjCoord': 30.0, 'away': 1.0, 'time remained': 1876.08, 'Penalty': 1.0, 'duration': 1.53, 'angel2gate': 0.84, 'velocity_x': -10.1, 'velocity_y': -0.33, 'action': 'shot', 'home': 0.0, 'event_outcome': 1.0}                  "
 u"{'xAdjCoord': 44.5, 'scoreDifferential': 0.0, 'yAdjCoord': 7.5, 'away': 1.0, 'time remained': 1901.1, 'Penalty': 1.0, 'duration': 0.13, 'angel2gate': 0.17, 'velocity_x': -14.99, 'velocity_y': 7.49, 'action': 'pass', 'home': 0.0, 'event_outcome': 1.0}                    "
 u"{'xAdjCoord': 79.5, 'scoreDifferential': 0.0, 'yAdjCoord': -36.0, 'away': 1.0, 'time remained': 1892.56, 'Penalty': 1.0, 'duration': 4.44, 'angel2gate': 1.31, 'velocity_x': 5.63, 'velocity_y': -0.79, 'action': 'pass', 'home': 0.0, 'event_outcome': 1.0}                  "]
"""