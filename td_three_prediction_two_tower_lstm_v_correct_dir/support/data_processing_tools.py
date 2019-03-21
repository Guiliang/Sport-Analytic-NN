import numpy as np
import os
import csv
import math


def handle_trace_length(state_trace_length):
    """
    transform format of trace length
    :return:
    """
    trace_length_record = []
    for length in state_trace_length:
        for sub_length in range(0, int(length)):
            trace_length_record.append(sub_length + 1)

    return trace_length_record


def compromise_state_trace_length(state_trace_length, state_input, reward, max_trace_length, features_num):
    state_trace_length_output = []
    for index in range(0, len(state_trace_length)):
        tl = state_trace_length[index]
        if tl >= 10:
            tl = 10
        if tl > max_trace_length:
            state_input_change_list = []
            state_input_org = state_input[index]
            reward_change_list = []
            reward_org = reward[index]
            for i in range(0, max_trace_length):
                state_input_change_list.append(state_input_org[tl - max_trace_length + i])
                temp = reward_org[tl - max_trace_length + i]
                # if temp != 0:
                #     print 'find miss reward'
                reward_change_list.append(reward_org[tl - max_trace_length + i])

            state_input_update = padding_hybrid_feature_input(state_input_change_list,
                                                              max_trace_length=max_trace_length,
                                                              features_num=features_num)
            state_input[index] = state_input_update
            reward_update = padding_hybrid_reward(reward_change_list)
            reward[index] = reward_update

            tl = max_trace_length
        state_trace_length_output.append(tl)
    return state_trace_length_output, state_input, reward


# def padding_hybrid_feature_input(hybrid_feature_input):
#     current_list_length = len(hybrid_feature_input)
#     padding_list_length = 10 - current_list_length
#     for i in range(0, padding_list_length):
#         hybrid_feature_input.append(np.asarray([float(0)] * 25))
#     return np.asarray(hybrid_feature_input)


def padding_hybrid_reward(hybrid_reward):
    current_list_length = len(hybrid_reward)
    padding_list_length = 10 - current_list_length
    for i in range(0, padding_list_length):
        hybrid_reward.append(0)
    return np.asarray(hybrid_reward)


def get_together_training_batch(s_t0, state_input, reward, train_number, train_len, state_trace_length, ha_id,
                                batch_size):
    """
    combine training data to a batch
    :return:
    """
    batch_return = []
    print_flag = False
    current_batch_length = 0
    while current_batch_length < batch_size:
        s_t1 = state_input[train_number]
        if len(s_t1) < 10 or len(s_t0) < 10:
            raise ValueError("wrong length of s")
            # train_number += 1
            # continue
        s_length_t1 = state_trace_length[train_number]
        s_length_t0 = state_trace_length[train_number - 1]
        home_away_id_t1 = ha_id[train_number]
        home_away_id_t0 = ha_id[train_number - 1]
        if s_length_t1 > 10:  # if trace length is too long
            s_length_t1 = 10
        if s_length_t0 > 10:  # if trace length is too long
            s_length_t0 = 10
        try:
            s_reward_t1 = reward[train_number]
            s_reward_t0 = reward[train_number - 1]
        except IndexError:
            raise IndexError("s_reward wrong with index")
        train_number += 1
        if train_number + 1 == train_len:
            trace_length_index_t1 = s_length_t1 - 1
            trace_length_index_t0 = s_length_t0 - 1
            r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
            r_t1 = np.asarray([s_reward_t1[trace_length_index_t1]])
            if r_t0 == [float(0)]:
                r_t0_combine = [float(0), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, home_away_id_t1, home_away_id_t1, 1, 0))

            elif r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, home_away_id_t1, home_away_id_t1, 1, 0))

            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 0))

                if r_t1 == float(0):
                    r_t1_combine = [float(0), float(0), float(1)]
                elif r_t1 == float(-1):
                    r_t1_combine = [float(0), float(1), float(1)]
                elif r_t1 == float(1):
                    r_t1_combine = [float(1), float(0), float(1)]
                else:
                    raise ValueError("incorrect r_t1")
                batch_return.append(
                    (s_t1, s_t1, r_t1_combine, s_length_t1, s_length_t1, home_away_id_t1, home_away_id_t1, 1, 0))
            else:
                raise ValueError("r_t0 wrong value")

            s_t0 = s_t1
            break

        trace_length_index_t0 = s_length_t0 - 1  # we want the reward of s_t0, so -2
        r_t0 = np.asarray([s_reward_t0[trace_length_index_t0]])
        if r_t0 != [float(0)]:
            print 'find no-zero reward', r_t0
            print_flag = True
            if r_t0 == [float(-1)]:
                r_t0_combine = [float(0), float(1), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 1))
            elif r_t0 == [float(1)]:
                r_t0_combine = [float(1), float(0), float(0)]
                batch_return.append(
                    (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 1))
            else:
                raise ValueError("r_t0 wrong value")
            s_t0 = s_t1
            break
        r_t0_combine = [float(0), float(0), float(0)]
        batch_return.append(
            (s_t0, s_t1, r_t0_combine, s_length_t0, s_length_t1, home_away_id_t0, home_away_id_t1, 0, 0))
        current_batch_length += 1
        s_t0 = s_t1

    return batch_return, train_number, s_t0, print_flag


def write_game_average_csv(data_record, log_dir):
    if os.path.exists(log_dir + '/avg_cost_record.csv'):
        with open(log_dir + '/avg_cost_record.csv', 'a') as csvfile:
            fieldnames = (data_record[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for record in data_record:
                writer.writerow(record)
    else:
        with open(log_dir + '/avg_cost_record.csv', 'w') as csvfile:
            fieldnames = (data_record[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in data_record:
                writer.writerow(record)


def judge_feature_in_action(feature_input, actions):
    for action in actions:
        if feature_input == action:
            return True
    return False


def construct_simulation_data(features_train, features_mean, features_scale,
                              feature_type, is_home, action_type, set_dict={}):
    state = []
    for feature in features_train:
        if feature == 'xAdjCoord':
            xAdjCoord = set_dict.get('xAdjCoord')
            scale_xAdjCoord = float(xAdjCoord - features_mean['xAdjCoord']) / features_scale['xAdjCoord']
            state.append(scale_xAdjCoord)
        elif feature == 'yAdjCoord':
            yAdjCoord = set_dict.get('yAdjCoord')
            scale_yAdjCoord = float(yAdjCoord - features_mean['yAdjCoord']) / features_scale['yAdjCoord']
            state.append(scale_yAdjCoord)
        elif feature in set_dict:
            temp = set_dict[feature]
            scale_temp = float(temp - features_mean[feature]) / features_scale[feature]
            state.append(scale_temp)
        elif feature_type < 5 and feature == 'event_id':
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
            scale_actions = float(actions[action_type] - features_mean['event_id']) / features_scale['event_id']
            state.append(scale_actions)

        elif feature_type >= 5 and action_type == feature:
            scale_action = float(1 - features_mean[action_type]) / features_scale[action_type]
            state.append(scale_action)
        elif feature_type >= 5 and judge_feature_in_action(feature):
            scale_action = float(0 - features_mean[feature]) / features_scale[feature]
            state.append(scale_action)
        elif feature == 'event_outcome':
            scale_event_outcome = float(1 - features_mean['event_outcome']) / features_scale['event_outcome']
            state.append(scale_event_outcome)
        elif feature == 'angel2gate':
            gate_x_coord = 89
            gate_y_coord = 0
            y_diff = abs(yAdjCoord - gate_y_coord)
            x_diff = gate_x_coord - xAdjCoord
            z = math.sqrt(math.pow(y_diff, 2) + math.pow(x_diff, 2))
            try:
                angel2gate = math.acos(float(x_diff) / z)
            except:
                print ("exception point with x:{0} and y:{1}".format(str(xAdjCoord), str(yAdjCoord)))
                angel2gate = math.pi
            scale_angel2gate = float(angel2gate - features_mean['angel2gate']) / features_scale['angel2gate']
            state.append(scale_angel2gate)
        elif feature == 'home':
            if is_home:
                scale_home = float(1 - features_mean['home']) / features_scale['home']
                state.append(scale_home)
            else:
                scale_home = float(0 - features_mean['home']) / features_scale['home']
                state.append(scale_home)
        elif feature == 'away':
            if is_home:
                scale_away = float(0 - features_mean['away']) / features_scale['away']
                state.append(scale_away)
            else:
                scale_away = float(1 - features_mean['away']) / features_scale['away']
                state.append(scale_away)
        else:
            state.append(0)

    return np.asarray(state)


def padding_hybrid_feature_input(hybrid_feature_input, max_trace_length, features_num):
    current_list_length = len(hybrid_feature_input)
    padding_list_length = max_trace_length - current_list_length
    for i in range(0, padding_list_length):
        hybrid_feature_input.append(np.asarray([float(0)] * features_num))
    return hybrid_feature_input
