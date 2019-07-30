import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')
import json
import os
import numpy as np
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_soccer_game_data
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_shot_scatter(suc_shot_action_all):
    plt.figure(figsize=(12, 6))
    plt.scatter(suc_shot_action_all[:, 0], suc_shot_action_all[:, 1])
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    # plt.show()
    plt.savefig('./suc_shot_scatter.png')


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


def gather_score_data(directory, data_path, suc_shot_action_all):
    with open(data_path + str(directory)) as f:
        data = json.load(f)

    events = data.get('events')
    for j in range(len(events)):
        event = events[j]
        action = str(event.get('action'))
        if action == 'goal':
            if 'own' not in events[j - 1].get('action'):
                x_ = float(events[j - 1].get('x'))
                y_ = float(events[j - 1].get('y'))
                print str(events[j - 1].get('action'))
                suc_shot_action_all.append([x_, y_])

    return suc_shot_action_all


def find_soccer_target_data(directory, data_path, print_latex):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    # print "game time is:" + str(data.get('gameDate'))
    events = data.get('events')
    number = 0
    if print_latex:
        print("GTR & X & Y & MP & GD & Action & OC  & Velocity & ED & Angle & H/A & Reward \\\\ \hline")
    for j in range(len(events)):
        event = events[j]
        action = str(event.get('action'))

        if action == 'goal':

            x_pre = str(events[j - 10].get('x'))
            y_pre = str(events[j - 10].get('y'))
            min_pre = str(events[j - 10].get('min'))
            sec_pre = str(events[j - 10].get('sec'))
            h_a_pre = str(events[j - 10].get('home_away'))

            print('\n')
            for i in np.arange(7, -1, -1):
                event_print = events[j - i]
                manpower = str(event_print.get('manPower'))
                if int(manpower) > 0:
                    manpower_str = 'PowerP'
                elif int(manpower) < 0:
                    manpower_str = 'ShortedH'
                else:
                    manpower_str = 'Even'
                angle = event_print.get('angle')
                min = event_print.get('min')
                sec = event_print.get('sec')
                action = event_print.get('action').replace('_', ' ').replace('-', ' ')
                reward = event_print.get('reward')
                x = event_print.get('x')
                y = event_print.get('y')
                outcome = event_print.get('outcome')
                h_a = event_print.get('home_away')
                sd = event_print.get('scoreDiff')
                if print_latex:
                    outcome = 'S' if int(outcome) == 1 else 'F'
                    duration = (float(min) - float(min_pre)) * 60 + (float(sec) - float(sec_pre))
                    angle = round(angle, 2)
                    if h_a == 'A':
                        x = 100 - float(x)
                        y = 100 - float(y)
                    if float(duration) > 0:
                        # if h_a_pre == h_a:
                        velocity_x = round((float(x) - float(x_pre)) / duration, 1)
                        velocity_y = round((float(y) - float(y_pre)) / duration, 1)
                        # else:
                        #     velocity_x = round((float(x) - (100 - float(x_pre))) / float(duration), 3)
                        #     velocity_y = round((float(y) - (100 - float(y_pre))) / float(duration), 3)
                    else:
                        velocity_x = 0
                        velocity_y = 0
                    GameTimeRemainMin = int(((90 - float(min)) - 1))
                    GameTimeRemainSec = int((60 - float(sec)))
                    # X & Y & MP & GD & Action & OC  & Velocity & GTR & ED & Angle & H/A
                    print("{8}m{9}s & {0} & {1} & {2} & {3} & {4} & {5}  "
                          "& ({6}, {7}) & {10} & {11} & {12} & {13} \\\\".format(
                        x, y, manpower_str, sd, action, outcome, velocity_x, velocity_y, GameTimeRemainMin,
                        GameTimeRemainSec, duration, angle, h_a, reward
                    ))
                    x_pre = x
                    y_pre = y
                    min_pre = min
                    sec_pre = sec
                    h_a_pre = h_a

                else:
                    print "h_a:{6}, x:{0}, y:{1}, action:{2}, " \
                          "reward:{3}, event_number:{4},outcome:{5}, " \
                          "min:{7}, sec:{8}, manpower:{9}, angle{10}".format(x,
                                                                             y,
                                                                             action,
                                                                             str(reward),
                                                                             str(number),
                                                                             outcome,
                                                                             h_a,
                                                                             min,
                                                                             sec,
                                                                             manpower,
                                                                             angle)


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


def count_event_number():
    raw_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    dir_all = os.listdir(raw_data_path)
    event_all_number = 0
    for directory in dir_all:
        with open(raw_data_path + str(directory)) as f:
            data = json.load(f)
        # print "game time is:" + str(data.get('gameDate'))/Users/liu/Desktop/soccer-data-sample/sequences_append_goal
        events = data.get('events')

        event_all_number += len(events)
    print(event_all_number)


if __name__ == '__main__':
    # check_soccer_data()
    # count_event_number()
    # find_events_with_idx()
    # check_soccer_data()
    # data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data/"
    # data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    data_path = '/Users/liu/Desktop/soccer-data-sample/sequences_append_goal/'
    dir_all = os.listdir(data_path)
    #
    suc_shot_action_all = []
    for dir in dir_all:
        gather_score_data(dir, data_path, suc_shot_action_all)
        # find_soccer_target_data(dir, data_path, print_latex=True)
    plot_shot_scatter(np.asarray(suc_shot_action_all))
    # print suc_shot_action_all

    print np.asarray(suc_shot_action_all).mean(axis=0)