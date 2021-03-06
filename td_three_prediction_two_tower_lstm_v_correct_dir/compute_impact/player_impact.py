import csv
import os

import datetime
import scipy.io as sio
import json
import operator
import unicodedata
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools
import td_three_prediction_two_tower_lstm_v_correct_dir.resource.salary_1718 as salary_1718


class PlayerImpact:
    def __init__(self, data_name, model_data_store_dir, game_data_dir, difference_type='back_difference_'):
        self.player_id_dict_all = {}
        self.player_name_dict_all = {}
        # self.PLAYER_ID_DICT_ALL = {}
        self.difference_type = difference_type
        self.data_name = data_name
        self.game_data_dir = game_data_dir
        self.model_data_store_dir = model_data_store_dir

    def save_player_impact(self, save_name_write=None, league_name=''):
        assert len(self.player_name_dict_all.keys()) > 0
        if not save_name_write:
            save_name_write = 'soccer_player_GIM_{0}_{1}{2}.json'. \
                format(datetime.date.today().strftime("%Y%B%d"), self.difference_type, league_name)
        with open('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/'
                  'td_three_prediction_two_tower_lstm_v_correct_dir/'
                  'compute_impact/player_impact/' + save_name_write, 'w') as f:
            json.dump(self.player_name_dict_all, f)

    def get_id(self, playername, teamname, game_info_all):
        for info in game_info_all:
            p_name = info[0]
            t_name = info[1]
            # if playername == p_name:
            #     print('find name')
            if playername in p_name and teamname in t_name:
                return True, info[2]
        return False, None

    def rank_player_by_impact(self, player_summary_info_dir, write_file, action_selected, game_info_all):
        player_id_info_dir = {}
        with open(player_summary_info_dir) as online_info_file:
            online_reader = csv.DictReader(online_info_file)
            for r in online_reader:
                playername = r['name']
                teamname = r['team']
                if teamname[0] == '"':
                    teamname = teamname[1:-1]
                teamname = teamname.split(',')[0]
                Goals = r['Goals']
                Assist = r['Assists']
                Flag, id = self.get_id(playername, teamname, game_info_all)
                if id is None:
                    continue
                if int(id) not in self.player_id_dict_all:
                    continue
                player_id_info_dir.update({id: [playername, teamname, Goals, Assist]})
                print 'find player {0}'.format(playername)

        sorted_id_GIM = sorted(self.player_id_dict_all.items(), key=operator.itemgetter(1), reverse=True)
        if action_selected == 'shot':
            write_file.write('name & team & GIM & Goal \\\\ \n')
        elif action_selected == 'pass':
            write_file.write('name & team & GIM & Assist \\\\ \n')
        else:
            write_file.write('name & team & GIM & Goal & Assist $ Salary \\\\ \n')
        for (id, GIM) in sorted_id_GIM:
            if id is None:
                print (id, GIM)
                continue
            salary = salary_1718.salary.get(int(id))
            salary = str(salary) if salary is not None else '-'
            info = player_id_info_dir.get(str(id))
            if info is None:
                continue
            name = info[0]
            team = info[1]
            goals = info[2]
            assist = info[3]
            if action_selected == 'shot':
                # write_file.write('name & team & GIM & Goal \\\\')
                write_file.write('{0} & {1} & {2} & {3} \\\\  \n'.format(name, team, round(GIM['value'], 3), goals))
            elif action_selected == 'pass':
                # write_file.write('name & team & GIM & Assist \\\\')
                write_file.write('{0} & {1} & {2} & {3} \\\\ \n'.format(name, team, round(GIM['value'], 3), assist))
            else:
                # write_file.write('name & team & GIM & Goal & Assist \\\\')
                write_file.write('{0} & {1} & {2} & {3} & {4} & {5} \\\\ \n'.format(name, team, round(GIM['value'], 3),
                                                                                    goals, assist, salary))

    def transfer2player_name_dict(self, player_id_name_pair_dir):
        with open(player_id_name_pair_dir, 'r') as f:
            player_id_name_pair = json.load(f)

        ids = player_id_name_pair.keys()
        # player_name_GIM_dict = {}
        for id in ids:
            id = int(id)
            GIM = self.player_id_dict_all.get(id)
            name = player_id_name_pair.get(str(id))
            name_str = name.get('first_name') + ' ' + name.get('last_name')
            self.player_name_dict_all.update({id: {'name': name_str, 'GIM': GIM}})

    def aggregate_match_diff_values(self, dir_game, action_selected_list=None, league_id=None):
        """compute impact"""

        for file_name in os.listdir(self.model_data_store_dir + "/" + dir_game):
            # print file_name
            if file_name == self.data_name:
                model_data_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                with open(model_data_name) as f:
                    model_data = json.load(f)
                    # model_data = (sio.loadmat(model_data_name))[self.data_name]
                    # elif file_name.startswith("playerId"):
                    #     playerIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                    #     playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
                    # elif file_name.startswith("teamId"):
                    #     teamIds_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                    # teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_away"):
                home_identifier_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_away"][0]
            else:
                continue
        # TODO: fix the names of features
        if league_id is not None:
            game_league_id = tools.read_game_league_info(data_path=self.game_data_dir, directory=dir_game + '.json')
            if game_league_id != league_id:
                return

        actions = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
                                                   feature_name='action')
        # print actions
        playerIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
                                                     feature_name='playerId')
        # teamIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game+'.json',
        #                                           feature_name='teamIds')
        # print len(playerIds)
        # print len(actions)
        skip_number = 0
        for player_Index in range(0, len(playerIds)):
            if action_selected_list is not None:
                continue_flag = False if len(action_selected_list) == 0 else True
                for f_action in action_selected_list:
                    if f_action in actions[player_Index]:
                        # print action
                        continue_flag = False
                if continue_flag:
                    skip_number += 1
                    continue

            playerId = playerIds[player_Index]
            # teamId = teamIds[player_Index]
            # if int(teamId_target) == int(teamId):
            # print model_data
            model_value = model_data[str(player_Index)]
            if player_Index - 1 >= 0:  # define model pre
                if actions[player_Index - 1] == "goal":
                    model_value_pre = model_data[str(player_Index)]  # the goal cancel out here, just as we cut the game
                else:
                    model_value_pre = model_data[str(player_Index - 1)]
            else:
                model_value_pre = model_data[str(player_Index)]
            if player_Index + 1 < len(playerIds):  # define model next
                if actions[player_Index + 1] == "goal":
                    model_value_nex = model_data[str(player_Index)]
                else:
                    model_value_nex = model_data[str(player_Index + 1)]
            else:
                model_value_nex = model_data[str(player_Index)]

            ishome = home_identifier[player_Index]
            player_value = self.player_id_dict_all.get(playerId)

            home_model_value = model_value['home']
            away_model_value = model_value['away']
            # end_model_value = abs(model_value[2])
            home_model_value_pre = model_value_pre['home']
            away_model_value_pre = model_value_pre['away']
            # end_model_value_pre = abs(model_value_pre[2])
            home_model_value_nex = model_value_nex['home']
            away_model_value_nex = model_value_nex['away']
            # end_model_value_nex = abs(model_value_nex[2])

            if ishome:
                if self.difference_type == "back_difference_":
                    value = (home_model_value - home_model_value_pre)
                    # - (away_model_value - away_model_value_pre)
                elif self.difference_type == "front_difference_":
                    value = (home_model_value_nex - home_model_value)
                    # - (away_model_value_nex - away_model_value)
                elif self.difference_type == "skip_difference_":
                    value = (home_model_value_nex - home_model_value_pre)
                    # - (away_model_value_nex - away_model_value_pre)
                elif self.difference_type == "expected_goal":
                    value = home_model_value
                else:
                    raise ValueError('unknown difference type')
                if player_value is None:
                    self.player_id_dict_all.update({playerId: {"value": value}})
                else:
                    player_value_number = player_value.get("value") + value
                    self.player_id_dict_all.update(
                        {playerId: {"value": player_value_number}})
                    # "state value": model_state_value[0] - model_state_value[1]}})
            else:

                if self.difference_type == "back_difference_":
                    value = (away_model_value - away_model_value_pre)
                    # - (home_model_value - home_model_value_pre)
                elif self.difference_type == "front_difference_":
                    value = (away_model_value_nex - away_model_value)
                    # - (home_model_value_nex - home_model_value)
                elif self.difference_type == "skip_difference_":
                    value = (away_model_value_nex - away_model_value_pre)
                    # - (home_model_value_nex - home_model_value_pre)
                elif self.difference_type == "expected_goal":
                    value = away_model_value
                else:
                    raise ValueError('unknown difference type')

                if player_value is None:
                    self.player_id_dict_all.update({playerId: {"value": value}})
                else:
                    player_value_number = player_value.get("value") + value
                    self.player_id_dict_all.update(
                        {playerId: {"value": player_value_number}})

        print('finish game {0} and solve {1} events'.format(dir_game, str(len(playerIds) - skip_number)))
