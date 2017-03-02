import scipy.io
import gameInfo_until
import numpy as np
import math
import os
from sklearn import preprocessing

NEGATIVE_REWARD_FLAG = False

features_name = ["manpowerSituation", "shorthand", "playerPosition", "period", "yAdjCoord", "xAdjCoord",
                 "scoreDifferential", "zone",
                 "gameTime", "frame"]

# add time elapse and team penalty
features_want = ['gameTime', 'frame', 'period', 'xAdjCoord', 'yAdjCoord', 'scoreDifferential', 'manpowerSituation',
                 'teamId', 'name', 'type', 'outcome']

event_dict = {}

features_train = ['velocity', 'time remained', 'scoreDifferential', 'Penalty', 'duration', 'event_id', 'event_outcome']
"""
1. velocity
2. time remained
3. scoreDifferential
4. Penalty
5. duration
6. event_id
7. event_outcome
"""

# data_path = "/Users/liu/Desktop/sport-analytic/Hockey-Match-All-data"
data_path = "/home/gla68/Documents/Hockey-data/Hockey-Match-All-data"
data_store = "/home/gla68/Documents/Hockey-data/Hockey-Training-All"
dir_all = os.listdir(data_path)


def print_general_data(game_dir):
    game = scipy.io.loadmat(data_path + "/" + str(game_dir))
    gameId = (game['x'])['gameId'][0][0][0]
    print("process game" + str(gameId))
    events = (game['x'])['events'][0][0]

    gameInfo = []

    for j in range(events.size):  # the number of event
        eve = events[0, j]
        teamId = (((eve['teamId'])[0])[0])[0]
        home_boolean = gameInfo_until.judge_home(gameId, teamId)
        event_array = {}
        # print("event number:"+str(j)+" name:"+str(eve['name'][0][0][0]) + " frame_num:" + str(eve['frame'][0][0][0][0]) + " gameTime:" + str(eve['gameTime'][0][0][0][0])+ " period:" + str(eve['period'][0][0][0][0]))
        for feature in features_want:
            info_record = eve[feature][0][0][0]
            if isinstance(info_record, np.ndarray):
                info_record = info_record[0]
            # if feature == 'manpowerSituation':
            #     info_record = str(info_record)
            #     print (info_record)
            if feature == 'manpowerSituation' or feature == 'name' or feature == 'type' or feature == 'outcome':
                info_record = str(info_record)
            else:
                info_record = float(info_record)

            event_array.update({feature: info_record})
        if home_boolean:
            event_array.update({"home": 1})
        else:
            event_array.update({"home": 0})
            temp_score_diff = event_array["scoreDifferential"]
            if int(temp_score_diff) == 0:
                event_array.update({"scoreDifferential": temp_score_diff})
            else:
                event_array.update({"scoreDifferential": - temp_score_diff})
        # print(event_array)
        gameInfo.append(event_array)

    return gameInfo


def draw_puck_trajectories(game_dir):
    game = scipy.io.loadmat(data_path + "/" + str(game_dir))
    gameId = (game['x'])['gameId'][0][0][0]
    print("process game" + str(gameId))
    events = (game['x'])['events'][0][0]

    gameInfo = []
    home_game_info = []
    away_game_info = []

    for j in range(events.size):  # the number of event
        eve = events[0, j]
        teamId = (((eve['teamId'])[0])[0])[0]
        home_boolean = gameInfo_until.judge_home(gameId, teamId)
        record_coord = []
        record_coord_home = []
        record_coord_away = []

        for feature in ["xAdjCoord", "yAdjCoord"]:
            try:
                info_record = eve[feature][0][0][0]
            except:
                info_record = eve[feature][0][0]

            if isinstance(info_record, np.ndarray):
                info_record = info_record[0]
            record_coord.append(float(info_record))
            if home_boolean:
                record_coord_home.append(info_record)
            else:
                record_coord_away.append(info_record)

        if home_boolean:
            home_game_info.append(record_coord_home)
            record_coord.append("home")
        else:
            away_game_info.append(record_coord_away)
            record_coord.append("away")
        gameInfo.append(record_coord)

    return home_game_info, away_game_info, gameInfo


def compute_puck_velocity(gameInfo_dict):
    prev_gametime = (gameInfo_dict[0])['gameTime']
    prev_x_coord = (gameInfo_dict[0])['xAdjCoord']
    prev_y_coord = (gameInfo_dict[0])['yAdjCoord']

    for event_array in gameInfo_dict:
        gametime = event_array['gameTime']
        time_diff = gametime - prev_gametime
        if time_diff == 0:
            velocity = 0
        else:
            x_coord = event_array['xAdjCoord']
            y_coord = event_array['yAdjCoord']
            space_diff = math.sqrt(math.pow((prev_x_coord - x_coord), 2) + math.pow((prev_y_coord - y_coord), 2))
            velocity = float(space_diff) / time_diff
        event_array.update({'velocity': velocity})
        # print(event_array)

    return gameInfo_dict


def compute_time_remained(gameInfo_dict):
    for event_array in gameInfo_dict:
        gametime = event_array['gameTime']
        time_remained = 3600 - gametime
        event_array.update({'time remained': time_remained})
    return gameInfo_dict


def compute_time_duration(gameInfo_dict):
    prev_gametime = 0
    for event_array in gameInfo_dict:
        gametime = event_array['gameTime']
        time_duration = gametime - prev_gametime
        event_array.update({'duration': time_duration})
    return gameInfo_dict


def select_train_features(gameInfo_dict):
    train_data_array = []
    for event_array in gameInfo_dict:
        state = []
        for feature in features_train:
            if feature == 'Penalty':
                feature_value = event_array['manpowerSituation']
                manpower = {
                    'shortHanded': -1,
                    'powerPlay': 1,
                    'evenStrength': 0
                }
                state.append(manpower[feature_value])
            elif feature == 'event_id':
                feature_value = event_array['name']
                actions = {'block': 0,
                           'carry': 1,
                           'check': 2,
                           'dumpin': 3,
                           'dumpout': 4,
                           'goal': 5,
                           'lpr': 6,
                           'offside': 7,
                           'pass': 8,
                           'puckprotection': 9,
                           'reception': 10,
                           'shot': 11,
                           'shotagainst': 12}
                state.append(actions[feature_value])
            elif feature == 'event_outcome':
                feature_value = event_array['outcome']
                outcome = {'successful': 1,
                           'failed': -1}
                state.append(outcome[feature_value])
            else:
                state.append(event_array[feature])
        train_data_array.append(np.asarray(state))
    return np.asarray(train_data_array)


def reward_add(gameInfo_dict):
    reward = np.zeros(len(gameInfo_dict))
    for event_num in range(0, len(gameInfo_dict)):
        event_name = (gameInfo_dict[event_num])['name']
        if event_name == 'goal':
            if NEGATIVE_REWARD_FLAG:
                if (gameInfo_dict[event_num])['home'] == 1:
                    reward[event_num] = 1
                else:
                    reward[event_num] = -1
            else:
                if (gameInfo_dict[event_num])['home'] == 1:
                    reward[event_num] = 1
                    # print gameInfo_dict[event_num - 1]
                    # print gameInfo_dict[event_num]
                    # print gameInfo_dict[event_num + 1]
                    # print gameInfo_dict[event_num + 2]
                    # print ("\n")

    return reward


def deal_al_data():
    problem_message = []
    for directory in dir_all:
        try:
            gameInfo = print_general_data(directory)
            gameInfo = compute_puck_velocity(gameInfo)
            gameInfo = compute_time_remained(gameInfo)
            gameInfo = compute_time_duration(gameInfo)
            reward = reward_add(gameInfo)
            training_data = select_train_features(gameInfo)
            training_data_scale = preprocessing.scale(training_data)
            game_dir = data_store+"/"+directory[:-4]
            os.makedirs(game_dir)
            gameInfo_until.write_pickle(game_dir+"/"+"reward_"+directory[:-4]+".pickle", reward)
            gameInfo_until.write_pickle(game_dir+"/"+"general_"+directory[:-4]+".pickle", gameInfo)
            gameInfo_until.write_pickle(game_dir+"/"+"state_"+directory[:-4]+".pickle", training_data_scale)
        except:
            print ("problem with game"+str(directory))
            problem_message.append("problem with game: "+str(directory))
            continue
    print problem_message

if __name__ == "__main__":
    deal_al_data()
    # print(dir[0])
    # draw_puck_trajectories(dir[0])
    # gameInfo = print_general_data("game000340.mat")
    # gameInfo = compute_puck_velocity(gameInfo)
    # gameInfo = compute_time_remained(gameInfo)
    # gameInfo = compute_time_duration(gameInfo)
    # reward = reward_add(gameInfo)
    # training_data = select_train_features(gameInfo)
    # training_data_scale = preprocessing.scale(training_data)
    # print (training_data)
