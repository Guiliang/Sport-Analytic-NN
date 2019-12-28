import numpy as np
import os
import csv
import math
import json
import scipy.io as sio
import unicodedata
import td_three_prediction_two_tower_lstm_v_correct_dir.config.icehockey_feature_setting as icehockey_feature_setting
import td_three_prediction_two_tower_lstm_v_correct_dir.config.soccer_feature_setting as soccer_feature_setting


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


def get_team_name(data_path, directory):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    # print "game time is:" + str(data.get('gameDate'))
    home_name = data['homeTeamName']
    away_name = data['awayTeamName']
    game_date = data['gameDate']

    return unicodedata.normalize('NFKD', home_name).encode('ascii', 'ignore'), \
           unicodedata.normalize('NFKD', away_name).encode('ascii', 'ignore'), \
           unicodedata.normalize('NFKD', game_date).encode('ascii', 'ignore')


def count_players_by_league(data_path, directory, competitionId, player_id_list):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    game_competition_id = data.get('competitionId')
    if competitionId != game_competition_id:
        return player_id_list
    else:
        events = data.get('events')
        for event in events:
            playerId = event.get('playerId')
            if playerId not in player_id_list:
                player_id_list.append(playerId)
        return player_id_list


def get_game_time(data_path, directory):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    # print "game time is:" + str(data.get('gameDate'))
    events = data.get('events')
    time_all = 95
    game_time_all = []
    for event in events:
        game_minutes = event.get('min')
        game_seconds = event.get('sec')
        game_time = game_minutes * 60 + game_seconds
        # time_remain = float(event.get('gameTimeRemain'))
        # if time_all is None:
        #     time_all = time_remain
        # game_time = (time_all - time_remain)
        game_time_all.append(game_time)
    game_time_all.sort(reverse=False)
    return game_time_all


def get_action_position(data_path, directory):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    # print "game time is:" + str(data.get('gameDate'))
    events = data.get('events')
    time_all = 95
    action_xy_return = []
    for event in events:
        action_name = unicodedata.normalize('NFKD', event.get('action')).encode('ascii', 'ignore')
        x = event.get('x')
        y = event.get('y')
        action_xy_return.append({'action': action_name, 'x': x, 'y': y})
    return action_xy_return


def get_soccer_game_data_old(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "reward_seq_" in filename:
            reward_name = filename
        elif "state_feature_seq_" in filename:
            state_input_name = filename
        elif "trace_drl_" in filename:
            state_trace_length_name = filename

    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)
    reward = reward['reward']
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)
    state_input = (state_input['state_feature_seq'])
    state_trace_length = sio.loadmat(data_store + "/" + dir_game + "/" + state_trace_length_name)
    state_trace_length = (state_trace_length['trace_length'])[0]
    state_trace_length = handle_trace_length(state_trace_length)
    # ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_away"][0].astype(int)
    # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    # state_trace_length = sio.loadmat(
    #     data_store + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
    # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
    # state_trace_length = handle_trace_length(state_trace_length)
    state_trace_length, state_input, reward = compromise_state_trace_length(
        state_trace_length=state_trace_length,
        state_input=state_input,
        reward=reward,
        max_trace_length=config.learn.max_trace_length,
        features_num=config.learn.feature_number
    )
    return state_trace_length, state_input, reward


def get_soccer_game_data(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "reward" in filename:
            reward_name = filename
        elif "state" in filename:
            state_input_name = filename
        elif "trace" in filename:
            state_trace_length_name = filename
        elif "home_away" in filename:
            ha_id_name = filename

    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)['reward']
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['state']
    ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_away"][0].astype(int)
    # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    state_trace_length = sio.loadmat(
        data_store + "/" + dir_game + "/" + state_trace_length_name)['trace_length'][0]
    # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
    state_trace_length = handle_trace_length(state_trace_length)
    state_trace_length, state_input, reward = compromise_state_trace_length(
        state_trace_length=state_trace_length,
        state_input=state_input,
        reward=reward,
        max_trace_length=config.learn.max_trace_length,
        features_num=config.learn.feature_number
    )
    return state_trace_length, state_input, reward, ha_id


def get_icehockey_game_data(data_store, dir_game, config):
    game_files = os.listdir(data_store + "/" + dir_game)
    for filename in game_files:
        if "dynamic_rnn_reward" in filename:
            reward_name = filename
        elif "dynamic_rnn_input" in filename:
            state_input_name = filename
        elif "trace" in filename:
            state_trace_length_name = filename
        elif "home_identifier" in filename:
            ha_id_name = filename

    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)
    ha_id = sio.loadmat(data_store + "/" + dir_game + "/" + ha_id_name)["home_identifier"][0]
    try:
        reward = reward['dynamic_rnn_reward']
    except:
        print "\n" + dir_game
        raise ValueError("reward wrong")
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['dynamic_feature_input']
    # state_input = (state_input['dynamic_feature_input'])
    # state_output = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_output_name)
    # state_output = state_output['hybrid_output_state']
    state_trace_length = sio.loadmat(
        data_store + "/" + dir_game + "/" + state_trace_length_name)['hybrid_trace_length'][0]
    # state_trace_length = (state_trace_length['hybrid_trace_length'])[0]
    state_trace_length = handle_trace_length(state_trace_length)
    state_trace_length, state_input, reward = compromise_state_trace_length(
        state_trace_length=state_trace_length,
        state_input=state_input,
        reward=reward,
        max_trace_length=config.learn.max_trace_length,
        features_num=config.learn.feature_number
    )

    return state_trace_length, state_input, reward, ha_id


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
            print 'terminating'
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


def get_data_name(config, if_old=False, league_name=''):
    learning_rate = config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = '5'
    elif learning_rate == 1e-4:
        learning_rate_write = '4'
    elif learning_rate == 0.0005:
        learning_rate_write = '5_5'
    if config.learn.merge_tower:
        merge_model_msg = '_merge'
    else:
        merge_model_msg = ''

    if if_old:
        extra_model_msg = 'ijcai_'
    else:
        extra_model_msg = ''

    data_name = "{7}model{6}_three_cut_together_predict_Feature{0}_Iter{1}" \
                "_lr{2}_Batch{3}_MaxLength{4}_Type{5}{8}.json".format(
        str(config.learn.feature_type),
        str(config.learn.iterate_num),
        str(learning_rate_write),
        str(config.learn.batch_size),
        str(config.learn.max_trace_length),
        str(config.learn.model_type),
        merge_model_msg,
        extra_model_msg,
        league_name
    )

    return data_name


def construct_simulation_data(features_train, features_mean, features_scale,
                              feature_type, is_home, action_type, actions, set_dict={}, gate_x_coord=None,
                              gate_y_coord=None):
    state = []
    for feature in features_train:
        if feature == 'xAdjCoord' or feature == 'x':
            xAdjCoord = set_dict.get('xAdjCoord')
            scale_xAdjCoord = float(xAdjCoord - features_mean[feature]) / features_scale[feature]
            state.append(scale_xAdjCoord)
        elif feature == 'yAdjCoord' or feature == 'y':
            yAdjCoord = set_dict.get('yAdjCoord')
            scale_yAdjCoord = float(yAdjCoord - features_mean[feature]) / features_scale[feature]
            state.append(scale_yAdjCoord)
        elif feature in set_dict:
            temp = set_dict[feature]
            scale_temp = float(temp - features_mean[feature]) / features_scale[feature]
            state.append(scale_temp)
        elif feature_type < 5 and feature == 'event_id':
            raise ValueError("feature type<5 in inapplicable")
        # actions = {'block': 0,
        #                'carry': 1,
        #                'check': 2,
        #                'dumpin': 3,
        #                'dumpout': 4,
        #                'goal': 5,
        #                'lpr': 6,
        #                'offside': 7,
        #                'pass': 8,
        #                'puckprotection': 9,
        #                'reception': 10,
        #                'shot': 11,
        #                'shotagainst': 12}
        #     scale_actions = float(actions[action_type] - features_mean['event_id']) / features_scale['event_id']
        #     state.append(scale_actions)

        elif feature_type >= 5 and action_type == feature:
            scale_action = float(1 - features_mean[action_type]) / features_scale[action_type]
            state.append(scale_action)
        elif feature_type >= 5 and judge_feature_in_action(feature, actions):
            scale_action = float(0 - features_mean[feature]) / features_scale[feature]
            state.append(scale_action)
        elif feature == 'event_outcome':  # ignore the outcome for soccer
            scale_event_outcome = float(0.6 - features_mean[feature]) / features_scale[feature]
            state.append(scale_event_outcome)
        elif feature == 'outcome':
            scale_event_outcome = float(0.6 - features_mean[feature]) / features_scale[feature]
            state.append(scale_event_outcome)
        # elif feature == 'angel2gate' or feature == 'angle': # TODO: temporally ignore angle
        #     xAdjCoord = set_dict.get('xAdjCoord')
        #     yAdjCoord = set_dict.get('yAdjCoord')
        #     y_diff = abs(yAdjCoord - gate_y_coord)
        #     x_diff = gate_x_coord - xAdjCoord
        #     z = math.sqrt(math.pow(y_diff, 2) + math.pow(x_diff, 2))
        #     try:
        #         angel2gate = math.acos(float(x_diff) / z)
        #     except:
        #         print ("exception point with x:{0} and y:{1}".format(str(xAdjCoord), str(yAdjCoord)))
        #         angel2gate = math.pi
        #     scale_angel2gate = float(angel2gate - features_mean[feature]) / features_scale[feature]
        #     state.append(scale_angel2gate)
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
        elif feature == 'distance_x':
            xAdjCoord = set_dict.get('xAdjCoord')
            distance_x = gate_x_coord - xAdjCoord
            assert distance_x >= 0
            scale_distance_x = float(distance_x - features_mean[feature]) / features_scale[feature]
            state.append(scale_distance_x)
        elif feature == 'distance_y':
            yAdjCoord = set_dict.get('yAdjCoord')
            distance_y = abs(gate_y_coord - yAdjCoord)
            # assert distance_y >= 0
            scale_distance_y = float(distance_y - features_mean[feature]) / features_scale[feature]
            state.append(scale_distance_y)
        elif feature == 'distance_to_goal':
            xAdjCoord = set_dict.get('xAdjCoord')
            distance_x = gate_x_coord - xAdjCoord
            yAdjCoord = set_dict.get('yAdjCoord')
            distance_y = abs(gate_y_coord - yAdjCoord)
            distance_to_goal = (distance_x ** 2 + distance_y ** 2) ** (1 / 2)
            # assert distance_y >= 0
            scale_distance_to_goal = float(distance_to_goal - features_mean[feature]) / features_scale[feature]
            state.append(scale_distance_to_goal)
        else:
            # print feature
            state.append(0)

    return np.asarray(state)


def padding_hybrid_feature_input(hybrid_feature_input, max_trace_length, features_num):
    current_list_length = len(hybrid_feature_input)
    padding_list_length = max_trace_length - current_list_length
    for i in range(0, padding_list_length):
        hybrid_feature_input.append(np.asarray([float(0)] * features_num))
    return hybrid_feature_input


def start_lstm_generate_spatial_simulation(history_action_type, history_action_type_coord,
                                           action_type, data_simulation_dir, simulation_type,
                                           feature_type, max_trace_length, features_num, is_home=True,
                                           sports='ice-hockey'):
    if sports == 'ice-hockey':
        x_min = -42.5
        x_max = 42.5
        x_section_num = 171
        y_min = -100
        y_max = 100
        y_section_num = 401
        features_train, features_mean, features_scale, actions = icehockey_feature_setting.select_feature_setting(
            feature_type=feature_type)
        gate_x_coord = 89
        gate_y_coord = 0
    elif sports == 'soccer':
        x_min = 0
        x_max = 100
        x_section_num = 100
        y_min = 0
        y_max = 100
        y_section_num = 100
        features_train, features_mean, features_scale, actions = soccer_feature_setting.select_feature_setting(
            feature_type=feature_type)
        gate_x_coord = 100
        gate_y_coord = 0
    else:
        raise ValueError('unknown sport')

    simulated_data_all = []

    for history_index in range(0, len(history_action_type) + 1):
        state_ycoord_list = []
        for ycoord in np.linspace(x_min, x_max, x_section_num):
            state_xcoord_list = []
            for xcoord in np.linspace(y_min, y_max, y_section_num):
                set_dict = {'xAdjCoord': xcoord, 'yAdjCoord': ycoord}
                state_generated = construct_simulation_data(
                    features_train=features_train,
                    features_mean=features_mean,
                    features_scale=features_scale,
                    feature_type=feature_type,
                    is_home=is_home,
                    action_type=action_type,
                    actions=actions,
                    set_dict=set_dict,
                    gate_x_coord=gate_x_coord,
                    gate_y_coord=gate_y_coord, )
                state_generated_list = [state_generated]
                for inner_history in range(0, history_index):
                    xAdjCoord = history_action_type_coord[inner_history].get('xAdjCoord')
                    yAdjCoord = history_action_type_coord[inner_history].get('yAdjCoord')
                    action = history_action_type[inner_history]
                    if action != action_type:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1, action_type: 0}
                    else:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1}
                    state_generated_history = construct_simulation_data(
                        features_train=features_train,
                        features_mean=features_mean,
                        features_scale=features_scale,
                        feature_type=feature_type,
                        is_home=is_home,
                        action_type=action_type,
                        actions=actions,
                        set_dict=set_dict_history,
                        gate_x_coord=gate_x_coord,
                        gate_y_coord=gate_y_coord,
                    )
                    state_generated_list = [state_generated_history] + state_generated_list

                state_generated_padding = padding_hybrid_feature_input(
                    hybrid_feature_input=state_generated_list,
                    max_trace_length=max_trace_length,
                    features_num=features_num)
                state_xcoord_list.append(state_generated_padding)
            state_ycoord_list.append(np.asarray(state_xcoord_list))

        store_data_dir = data_simulation_dir + '/' + simulation_type

        if not os.path.isdir(store_data_dir):
            os.makedirs(store_data_dir)
        # else:
        #     raise Exception
        if is_home:
            sio.savemat(
                store_data_dir + "/LSTM_Home_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        else:
            sio.savemat(
                store_data_dir + "/LSTM_Away_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        simulated_data_all.append(np.asarray(state_ycoord_list))

    return simulated_data_all


def read_feature_within_events(directory, data_path, feature_name):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    events = data.get('events')
    features_all = []
    for event in events:
        try:
            value = str(event.get(feature_name).encode('utf-8'))
        except:
            value = event.get(feature_name)
        features_all.append(value)

    return features_all


def read_features_within_events(directory, data_path, feature_name_list):
    with open(data_path + str(directory)) as f:
        data = json.load(f)
    try:
        events = data.get('events')
    except:
        events = data[0].get('events')
    features_all = []
    for event in events:
        feature_values = {}
        for feature_name in feature_name_list:
            try:
                value = str(event.get(feature_name).encode('utf-8'))
            except:
                value = event.get(feature_name)
            feature_values.update({feature_name: value})
        features_all.append(feature_values)

    return features_all


def find_soccer_game_dir_by_team(dir_all, data_path, team="Arsenal"):
    number = 0
    for directory in dir_all:
        number += 1
        print number
        try:
            with open(data_path + str(directory)) as f:
                data = json.load(f)
        except:
            print "can't read dir {0}".format(str(directory))
            continue
        gameId = str(data.get('gameId'))
        # gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
        homename = str(data.get('homeTeamName').encode('utf-8'))
        awayname = str(data.get('awayTeamName').encode('utf-8'))
        if team in homename or team in awayname:
            print "game time is:" + str(data.get('gameDate'))
            print "Home:{0} v.s. Away:{1}".format(homename, awayname)
            print gameId
            print directory
            # break


def find_game_dir(dir_all, data_path, target_game_id, sports):
    if sports == 'IceHockey':
        game_name = None
        for directory in dir_all:
            game = sio.loadmat(data_path + "/" + str(directory))
            gameId = (game['x'])['gameId'][0][0][0]
            gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
            if gameId == target_game_id:
                game_name = directory
                print directory
                break
    elif sports == 'Soccer':
        for directory in dir_all:
            with open(data_path + str(directory)) as f:
                data = json.load(f)[0]
            gameId = str(data.get('gameId'))
            # gameId = unicodedata.normalize('NFKD', gameId).encode('ascii', 'ignore')
            print "game time is:" + str(data.get('gameDate'))
            homename = str(data.get('homeTeamName'))
            awayname = str(data.get('awayTeamName'))
            print "Home:{0} v.s. Away:{1}".format(homename, awayname)
            if gameId == target_game_id:
                game_name = directory
                print directory
                break
    else:
        raise ValueError('Unknown sports game')

    if game_name:
        return game_name.split(".")[0]
    else:
        raise ValueError("can't find the game {0}".format(str(target_game_id)))


def normalize_data(game_value_home, game_value_away, game_value_end):
    game_value_home_normalized = []
    game_value_away_normalized = []
    game_value_end_normalized = []
    for index in range(0, len(game_value_home)):
        home_value = game_value_home[index]
        away_value = game_value_away[index]
        end_value = game_value_end[index]
        if end_value < 0:
            end_value = 0
        if away_value < 0:
            away_value = 0
        if home_value < 0:
            home_value = 0
        game_value_home_normalized.append(float(home_value) / (home_value + away_value + end_value))
        game_value_away_normalized.append(float(away_value) / (home_value + away_value + end_value))
        game_value_end_normalized.append(float(end_value) / (home_value + away_value + end_value))
    return np.asarray(game_value_home_normalized), np.asarray(game_value_away_normalized), np.asarray(
        game_value_end_normalized)


def combine_player_data(player_info_csv, player_stats, player_info_stats):
    player_id_info_dict = {}
    with open(player_info_csv) as f:
        data_lines = f.readlines()
    for line in data_lines[1:]:
        items = line.split(',')
        playerId = items[0]
        playerName = items[1]
        teamId = items[2]
        teamName = items[3]
        value = items[4]
        player_id_info_dict.update({playerName: [playerId, teamId, teamName]})

    with open(player_stats) as f:
        data_lines = f.readlines()
    record_file = open(player_info_stats, 'w')
    record_file.write('name,playerId,team,teamId,Apps,Mins,Goals,Assists,Yel,Red,SpG,PS,AeriaisWon,MotM,Rating\n')
    for line in data_lines[1:]:
        items = line.split(',')
        name = items[0]
        info = player_id_info_dict.get(name)
        if info is None:
            print name
            continue
        store_line = '{0},{1},{2},{3}'.format(items[0], str(info[0]), str(info[2]), str(info[1]))
        for item in items[3:]:
            store_line += ',' + item
        record_file.write(store_line)


def flat_state_input(state_input):
    flat_input = []
    for state in state_input:
        zero_flag = True
        for value in state:
            if value != 0:
                zero_flag = False
                break
        if zero_flag:
            flat_input = flat_input + len(state) * [0]
        else:
            flat_input = state.tolist() + flat_input
    return flat_input


def get_markov_rank_value(metric_name, ranking_dir_dict):
    metric_info = ranking_dir_dict.get(metric_name)
    with open(metric_info[1]) as f:
        d = json.load(f)
    return d


def get_GIM_rank_value(metric_name, ranking_dir_dict):
    metric_info = ranking_dir_dict.get(metric_name)
    rank_value_dict = {}
    with open(metric_info[1]) as f:
        d = json.load(f)
        for k in d.keys():
            dic = d[k]
            gim = dic[metric_info[0]]
            id = k
            if gim is None:
                continue
            value = gim['value']
            rank_value_dict[str(id)] = value
    return rank_value_dict


def get_network_dir(league_name, tt_lstm_config, train_msg):
    if tt_lstm_config.learn.merge_tower:
        merge_msg = 'm'
    else:
        merge_msg = 's'
    if len(league_name) > 0:
        lr = tt_lstm_config.learn.learning_rate / 10
    else:
        lr = tt_lstm_config.learn.learning_rate

    log_dir = "{0}/oschulte/Galen/soccer-models/hybrid_sl_log_NN" \
              "/{1}Scale-tt{9}-three-cut_together_log_feature{2}" \
              "_batch{3}_iterate{4}_lr{5}_{6}{7}_MaxTL{8}{10}".format(tt_lstm_config.learn.save_mother_dir,
                                                                      train_msg,
                                                                      str(tt_lstm_config.learn.feature_type),
                                                                      str(tt_lstm_config.learn.batch_size),
                                                                      str(tt_lstm_config.learn.iterate_num),
                                                                      str(lr),
                                                                      str(tt_lstm_config.learn.model_type),
                                                                      str(tt_lstm_config.learn.if_correct_velocity),
                                                                      str(tt_lstm_config.learn.max_trace_length),
                                                                      merge_msg,
                                                                      league_name)

    save_network_dir = "{0}/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/" \
                       "{1}Scale-tt{9}-three-cut_together_saved_networks_feature{2}" \
                       "_batch{3}_iterate{4}_lr{5}_{6}{7}_MaxTL{8}{10}".format(tt_lstm_config.learn.save_mother_dir,
                                                                               train_msg,
                                                                               str(tt_lstm_config.learn.feature_type),
                                                                               str(tt_lstm_config.learn.batch_size),
                                                                               str(tt_lstm_config.learn.iterate_num),
                                                                               str(lr),
                                                                               str(tt_lstm_config.learn.model_type),
                                                                               str(
                                                                                   tt_lstm_config.learn.if_correct_velocity),
                                                                               str(
                                                                                   tt_lstm_config.learn.max_trace_length),
                                                                               merge_msg,
                                                                               league_name)
    return log_dir, save_network_dir


def compute_cv_average_performance(cv_results_record_dir):
    with open(cv_results_record_dir, 'r') as f:
        results_by_line = f.readlines()
    all_results_record = []
    for line in results_by_line:
        results = line.split(':')[1:]
        results_line_record = []
        for result in results:
            results_line_record.append(float(result.split(',')[0]))
        all_results_record.append(results_line_record)
    print(np.mean(np.asarray(all_results_record), axis=0))
    print(np.var(np.asarray(all_results_record), axis=0))




if __name__ == '__main__':
    # combine_player_data(player_info_csv='../resource/player_team_id_name_value.csv',
    #                     player_stats='../resource/Soccer_summary.csv',
    #                     player_info_stats='../resource/Soccer_summary_info.csv')

    # compute_cv_average_performance(
    #     cv_results_record_dir='../regression_tree/dt_record/cv_pass_running_record_leaf100.txt')

    data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    dir_all = os.listdir(data_path)
    player_id_list = []
    competitionId = 10
    for game_name_dir in dir_all:
        game_name = game_name_dir.split('.')[0]
        player_id_list = count_players_by_league(data_path, game_name_dir, competitionId, player_id_list)
    print (player_id_list)
    print(len(player_id_list))
