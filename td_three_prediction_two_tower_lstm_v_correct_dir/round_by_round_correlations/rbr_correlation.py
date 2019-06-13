import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import csv
import json
import os
import pickle

import numpy as np
import operator
from scipy.stats import pearsonr
import scipy.io as sio
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import get_data_name


class RoundByRoundCorrelation:
    def __init__(self, raw_data_path, interested_metric, player_summary_dir,
                 model_data_store_dir, game_info_path):
        self.player_summary_dir = player_summary_dir
        self.action_selected_list = None
        self.raw_data_path = raw_data_path
        self.interested_metric = interested_metric
        self.model_data_store_dir = model_data_store_dir
        self.difference_type = None
        self.data_name = None
        self.round_number = 50
        self.game_info_path = game_info_path
        self.game_info_file = open(self.game_info_path)
        game_reader = csv.DictReader(self.game_info_file)
        self.game_info_all = []
        for r in game_reader:
            p_name = r['playerName']
            t_name = r['teamName']
            id = r['playerId']
            self.game_info_all.append([p_name, t_name, id])

    def read_team_by_date(self):
        dir_all = os.listdir(self.raw_data_path)
        team_game_dict = {}
        for directory in dir_all:
            print('processing date data {0}'.format(directory))
            with open(self.raw_data_path + str(directory)) as f:

                data = json.load(f)
                game_date = int(data.get('gameDate').split(' ')[0].replace('-', ''))
                home_teamId = data.get('homeTeamId')
                away_teamId = data.get('awayTeamId')

                if team_game_dict.get(home_teamId) is None:
                    game_date_dict = {}
                else:
                    game_date_dict = team_game_dict.get(home_teamId)
                game_date_dict.update({game_date: 'home$' + str(directory)})
                team_game_dict.update({home_teamId: game_date_dict})

                if team_game_dict.get(away_teamId) is None:
                    game_date_dict = {}
                else:
                    game_date_dict = team_game_dict.get(away_teamId)
                game_date_dict.update({game_date: 'away$' + str(directory)})
                team_game_dict.update({away_teamId: game_date_dict})
        return team_game_dict

    def aggregate_partial_impact_values(self, dir_game, ha_id, partial_player_value_dict, action_selected=None):
        ha_id = 1 if ha_id == 'home' else 0
        print('processing game {0} with ha_id {1}'.format(dir_game, str(ha_id)))
        """compute impact"""
        for file_name in os.listdir(self.model_data_store_dir + "/" + dir_game):
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

        actions = tools.read_feature_within_events(data_path=self.raw_data_path, directory=dir_game + '.json',
                                                   feature_name='action')
        playerIds = tools.read_feature_within_events(data_path=self.raw_data_path, directory=dir_game + '.json',
                                                     feature_name='playerId')

        for event_Index in range(0, len(playerIds)):

            if action_selected is not None:
                if action_selected not in actions[event_Index]:
                    continue

            if home_identifier[event_Index] != ha_id:
                # print 'skip event {0} as id is {1}'.format(str(event_Index), str(home_identifier[event_Index]))
                continue

            playerId = playerIds[event_Index]
            # teamId = teamIds[event_Index]
            # if int(teamId_target) == int(teamId):
            # print model_data
            model_value = model_data[str(event_Index)]
            if event_Index - 1 >= 0:  # define model pre
                if actions[event_Index - 1] == "goal":
                    model_value_pre = model_data[str(event_Index)]  # the goal cancel out here, just as we cut the game
                else:
                    model_value_pre = model_data[str(event_Index - 1)]
            else:
                model_value_pre = model_data[str(event_Index)]
            if event_Index + 1 < len(playerIds):  # define model next
                if actions[event_Index + 1] == "goal":
                    model_value_nex = model_data[str(event_Index)]
                else:
                    model_value_nex = model_data[str(event_Index + 1)]
            else:
                model_value_nex = model_data[str(event_Index)]

            ishome = home_identifier[event_Index]
            player_value = partial_player_value_dict.get(playerId)

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
                    partial_player_value_dict.update({playerId: {"value": value}})
                else:
                    player_value_number = player_value.get("value") + value
                    partial_player_value_dict.update(
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
                    partial_player_value_dict.update({playerId: {"value": value}})
                else:
                    player_value_number = player_value.get("value") + value
                    partial_player_value_dict.update(
                        {playerId: {"value": player_value_number}})
        return partial_player_value_dict

    def compute_game_by_round(self, team_game_dict):
        teams = team_game_dict.keys()
        game_by_round_dict = {}
        for round_num in range(0, self.round_number):
            game_by_round_dict.update({round_num + 1: []})

        for teamId in teams:
            date_dict = team_game_dict.get(teamId)

            if len(date_dict) < self.round_number:
                print("team round is {0},round_number ({1}) is too large".format(str(len(date_dict)),
                                                                                 str(self.round_number)))

            sorted_date_dict = sorted(date_dict.items(), key=operator.itemgetter(0))

            for round_num in range(1, self.round_number + 1):
                games_list = game_by_round_dict.get(round_num)

                if len(sorted_date_dict) < round_num:
                    continue

                date = sorted_date_dict[round_num - 1]
                games_list.append(str(date[0]) + '$' + date[1])
                game_by_round_dict.update({round_num: games_list})

        return game_by_round_dict

    def compute_player_season_totals(self):
        with open(self.player_summary_dir) as f:
            lines = f.readlines()
        player_id_info_dict = {}
        for line in lines[1:]:
            # name,playerId,team,teamId,Apps,Mins,Goals,Assists,Yel,Red,SpG,PS,AeriaisWon,MotM,Rating
            items = line.split(',')
            name = items[0]
            playerId = items[1]
            team = items[2]
            Goals = items[6] if items[6] != '-' else 0
            Assist = items[7] if items[7] != '-' else 0
            player_id_info_dict.update({playerId: [name, team, Goals, Assist]})
        return player_id_info_dict

    def compute_correlations_by_round(self, game_by_round_dict,
                                      player_id_info_dict, metric_name):
        correlated_coefficient_round_by_round = {}

        playerIds = player_id_info_dict.keys()
        partial_player_value_dict_goal = {}
        partial_player_value_dict_assist = {}
        for round_num in range(1, self.round_number + 1):  # TODO: too slow, fix it
            # partial_player_value_dict = {}
            game_info_lists = game_by_round_dict.get(round_num)
            game_dir_all = []
            game_ha_id_all = []
            for game_info in game_info_lists:
                game_info_items = game_info.split('$')
                h_a_id = game_info_items[1]
                game_ha_id_all.append(h_a_id)
                game_dir = game_info_items[2]
                game_dir_all.append(game_dir)
                action_selected = None if self.action_selected_list is None else self.action_selected_list[0]
                partial_player_value_dict_goal = \
                    self.aggregate_partial_impact_values(dir_game=game_dir.split('.')[0],
                                                         ha_id=h_a_id,
                                                         partial_player_value_dict=partial_player_value_dict_goal,
                                                         action_selected=action_selected)
                action_selected = None if self.action_selected_list is None else self.action_selected_list[1]
                partial_player_value_dict_assist = \
                    self.aggregate_partial_impact_values(dir_game=game_dir.split('.')[0],
                                                         ha_id=h_a_id,
                                                         partial_player_value_dict=partial_player_value_dict_assist,
                                                         action_selected=action_selected)

            # player_assist_list = []
            # player_goal_list = []
            # partial_player_GIM_list = []
            # for playerId in playerIds:
            #     player_gim = partial_player_value_dict.get(int(playerId))
            #     if player_gim is not None:
            #         player_gim_value = player_gim['value']
            #     else:
            #         # print 'continue'
            #         continue
            #     player_assist_list.append(int(player_id_info_dict.get(playerId)[3]))
            #     player_goal_list.append(int(player_id_info_dict.get(playerId)[2]))
            #     partial_player_GIM_list.append(player_gim_value)
            # goal_correlation = self.compute_correlated_coefficient(partial_player_GIM_list,
            #                                                        player_goal_list)
            # assistant_correlation = self.compute_correlated_coefficient(partial_player_GIM_list,
            #                                                             player_assist_list)
            # print ('matched player number is {0}'.format(len(partial_player_GIM_list)))

            goal_correlation = self.compute_correlation(rank_value_dict=partial_player_value_dict_goal,
                                                        interest_metric='Goals')
            assistant_correlation = self.compute_correlation(rank_value_dict=partial_player_value_dict_assist,
                                                             interest_metric='Assists')
            rate = -1 if metric_name == 'SI' else 1
            correlated_coefficient_round_by_round.update({round_num: {'assistant': rate * assistant_correlation,
                                                                      'goal': rate * goal_correlation}})
            print 'correlation for round {0} is assist:{1} and goal:{2}'.format(str(round_num),
                                                                                str(assistant_correlation),
                                                                                str(goal_correlation))
        return correlated_coefficient_round_by_round

    def compute_correlated_coefficient(self, listA, listB):
        return np.corrcoef(listA, listB)[0][1]
        # return pearsonr(listA, listB)[0]

    def get_id(self, playername, teamname):
        for info in self.game_info_all:
            p_name = info[0]
            t_name = info[1]
            # if playername == p_name:
            #     print('find name')
            if playername in p_name and teamname in t_name:
                return True, info[2]
        return False, ''

    def compute_correlation(self, rank_value_dict, interest_metric):
        online_value_list = []
        game_value_list = []
        with open(self.player_summary_dir) as online_info_file:
            online_reader = csv.DictReader(online_info_file)
            i = 0
            for r in online_reader:
                playername = r['name']
                teamname = r['team']
                if teamname[0] == '"':
                    teamname = teamname[1:-1]
                teamname = teamname.split(',')[0]
                standard_value = r[interest_metric]
                if standard_value == '-':
                    continue
                # print(playername, ' ', teamname)
                Flag, id = self.get_id(playername, teamname)
                if not Flag:
                    continue
                # print type(id)
                id = int(id)
                # print rank_value_dict
                if id not in rank_value_dict:
                    continue
                value = rank_value_dict[id]['value']
                # print(value)
                online_value_list.append(float(standard_value))
                # print(value)
                game_value_list.append(float(value))
                i += 1

        # print(len(online_value_list))
        # print(len(game_value_list))
        # print('matched number is ' + str(i))
        return np.corrcoef(online_value_list, game_value_list)[0][1]

    def normalization(self, player_impacts):
        impact_all = []
        for impact in player_impacts.values():
            impact_all.append(impact)
        variance = np.var(impact_all)
        mean = np.mean(impact_all)

        for player_Id in player_impacts.keys():
            player_impact = player_impacts.get(player_Id)
            player_impact = (player_impact - mean) / variance
            player_impacts.update({player_Id: player_impact})

        return player_impacts

    def write_round_correlation(self, correlated_coefficient_round_by_round,
                                csv_name='./season_ranking_result/statistic_round_correlation.csv'):

        standard_statistic_fields = ['assistant', 'goal', 'point']

        with open(csv_name, 'w') as csvfile:
            fieldnames = standard_statistic_fields
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for round_number in range(1, self.round_number + 1):
                round_correlated_coefficients = correlated_coefficient_round_by_round.get(round_number)
                round_correlated_coefficients_record = {}
                for standard_statistic in standard_statistic_fields:
                    round_correlated_coefficients_record.update(
                        {standard_statistic: round_correlated_coefficients.get(standard_statistic)})
                writer.writerow(round_correlated_coefficients_record)


if __name__ == "__main__":
    tt_lstm_config_path = "../soccer-config-v3.yaml"

    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    raw_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    model_data_store_dir = "/cs/oschulte/Galen/Soccer-data"
    interested_metric = ['Goals', 'Assists', 'Auto']
    player_summary_dir = '../resource/Soccer_summary.csv'
    game_info_path = '../resource/player_team_id_name_value.csv'
    data_name = get_data_name(config=tt_lstm_config)
    rbr_correlation = RoundByRoundCorrelation(raw_data_path, interested_metric, player_summary_dir,
                                              model_data_store_dir, data_name, game_info_path)
    # team_game_dict = rbr_correlation.read_team_by_date()
    # pickle.dump(team_game_dict, open('./tmp_stores/team_game_dict.pkl', 'w'))
    team_game_dict = pickle.load(open('./tmp_stores/team_game_dict.pkl', 'r'))
    game_by_round_dict = rbr_correlation.compute_game_by_round(team_game_dict=team_game_dict)
    total_game = 0
    game_ste = {}
    for values in game_by_round_dict.values():
        total_game += len(values)
        for value in values:
            if game_ste.get(value.split('$')[-1]) is not None:
                number = game_ste.get(value.split('$')[-1]) + 1
            else:
                number = 1
            game_ste.update({value.split('$')[-1]: number})
    print total_game / 2
    player_id_info_dict = rbr_correlation.compute_player_season_totals()
    correlated_coefficient_round_by_round = rbr_correlation.compute_correlations_by_round(
        player_id_info_dict=player_id_info_dict, game_by_round_dict=game_by_round_dict)
    with open('round_by_round_correlation.json', 'w') as outfile:
        json.dump(obj=correlated_coefficient_round_by_round, fp=outfile)

    print 'still working'
