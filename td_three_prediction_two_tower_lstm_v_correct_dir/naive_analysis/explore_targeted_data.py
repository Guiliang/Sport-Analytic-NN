import json
import os
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_soccer_game_data
import scipy.io as sio


def find_icehockey_target_data(directory, data_path):
    data = sio.loadmat(data_path + "/" + str(directory))
    events = data['x']['events'][0][0][0]
    for j in range(events.size - 1):
        event = events[j]
        action = event['name'][0][0][0].encode('utf-8')
        outcome = event['outcome'][0][0][0].encode('utf-8')
        team = event['teamId'][0][0][0].encode('utf-8')
        event_nex = events[j + 1]
        action_nex = event_nex['name'][0][0][0].encode('utf-8')
        outcome_nex = event_nex['outcome'][0][0][0].encode('utf-8')
        team_nex = event_nex['teamId'][0][0][0].encode('utf-8')
        if action == 'goal':
            print 'action:{0}, outcome:{1}, team:{2}'.format(action, outcome, team)
            print 'action_nex:{0}, outcome_nex:{1}, team:{2}\\'.format(action_nex, outcome_nex, team_nex)
    print 'ok'


def find_soccer_target_data(directory, data_path):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    # print "game time is:" + str(data.get('gameDate'))
    events = data.get('events')
    number = 0
    for event in events:
        reward = float(event.get('reward'))
        x = str(event.get('x'))
        y = str(event.get('y'))
        action = str(event.get('action'))
        outcome = str(event.get('outcome'))
        h_a = str(event.get('home_away'))
        if action == 'goal':
            print  'team'+h_a
            print "x:{0}, y:{1}, action:{2}, reward:{3}, event_number:{4},outcome:{5}".format(x, y, action, str(reward),
                                                                                              str(number), outcome)
        number += 1


def read_training_data(data_store, dir_game):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "state_add" in filename:
            state_input_name = filename

    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['state']
    print "find"


def check_soccer_data():
    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    # train_data_path = "/cs/oschulte/miyunLuo/Documents/data/"
    dir_all = os.listdir(data_path)
    for dir in dir_all:
        # read_training_data(train_data_path, dir.split('.')[0])
        find_soccer_target_data(dir, data_path)
        # break


def find_events_with_idx():
    idxs = [1787, 1788, 1789]
    game_dir = "922070.json"
    train_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    with open(train_data_path + str(game_dir)) as f:
        data = json.load(f)
    events = data.get('events')
    event_found = [events[idx] for idx in idxs]
    for event in event_found:
        print event


if __name__ == '__main__':
    # check_soccer_data()
    find_events_with_idx()
    # check_soccer_data()
    # data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data/"
    # dir_all = os.listdir(data_path)
    #
    # for dir in dir_all:
    #     find_icehockey_target_data(dir, data_path)
