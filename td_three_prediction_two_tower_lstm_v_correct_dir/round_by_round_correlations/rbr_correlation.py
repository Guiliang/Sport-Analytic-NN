import csv
import json
import os
import numpy as np
import operator
from scipy.stats import pearsonr
import scipy.io as sio
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools


class RoundByRoundCorrelation:
    def __init__(self, raw_data_path, interested_metric, player_summary_info_dir,
                 model_data_store_dir, data_name, difference_type='back_difference_'):
        self.player_summary_info_dir = player_summary_info_dir
        self.raw_data_path = raw_data_path
        self.interested_metric = interested_metric
        self.model_data_store_dir = model_data_store_dir
        self.difference_type = difference_type
        self.data_name = data_name
        self.ROUND_NUMBER = 60

    def read_team_by_date(self):
        dir_all = os.listdir(self.raw_data_path)
        team_game_dict = {}
        for directory in dir_all:
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
                if actions[event_Index] != action_selected:
                    continue

            if home_identifier[event_Index] != ha_id:
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
                    q_value = (home_model_value - home_model_value_pre)
                    # - (away_model_value - away_model_value_pre)
                elif self.difference_type == "front_difference_":
                    q_value = (home_model_value_nex - home_model_value)
                    # - (away_model_value_nex - away_model_value)
                elif self.difference_type == "skip_difference_":
                    q_value = (home_model_value_nex - home_model_value_pre)
                    # - (away_model_value_nex - away_model_value_pre)
                else:
                    raise ValueError('unknown difference type')
                if player_value is None:
                    partial_player_value_dict.update({playerId: {"value": q_value}})
                else:
                    player_value_number = player_value.get("value") + q_value
                    partial_player_value_dict.update(
                        {playerId: {"value": player_value_number}})
                    # "state value": model_state_value[0] - model_state_value[1]}})
            else:

                if self.difference_type == "back_difference_":
                    q_value = (away_model_value - away_model_value_pre)
                    # - (home_model_value - home_model_value_pre)
                elif self.difference_type == "front_difference_":
                    q_value = (away_model_value_nex - away_model_value)
                    # - (home_model_value_nex - home_model_value)
                elif self.difference_type == "skip_difference_":
                    q_value = (away_model_value_nex - away_model_value_pre)
                    # - (home_model_value_nex - home_model_value_pre)
                else:
                    raise ValueError('unknown difference type')

                if player_value is None:
                    partial_player_value_dict.update({playerId: {"value": q_value}})
                else:
                    player_value_number = player_value.get("value") + q_value
                    partial_player_value_dict.update(
                        {playerId: {"value": player_value_number}})

    def compute_game_by_round(self, team_game_dict):
        teams = team_game_dict.keys()
        game_by_round_dict = {}
        for round_num in range(0, self.ROUND_NUMBER):
            game_by_round_dict.update({round_num + 1: []})

        for teamId in teams:
            date_dict = team_game_dict.get(teamId)

            if len(date_dict) < self.ROUND_NUMBER:
                raise Exception("ROUND_NUMBER is too large")

            sorted_date_dict = sorted(date_dict.items(), key=operator.itemgetter(0))

            for round_num in range(1, self.ROUND_NUMBER + 1):
                games_list = game_by_round_dict.get(round_num)

                for date in sorted_date_dict[:round_num]:
                    games_list.append(date[0] + '$' + date[1])
                game_by_round_dict.update({round_num: games_list})

        return game_by_round_dict

    def compute_player_season_totals(self):
        with open(self.player_summary_info_dir) as f:
            lines = f.readlines()
        player_id_info_dict = {}
        for line in lines[1:]:
            # name,playerId,team,teamId,Apps,Mins,Goals,Assists,Yel,Red,SpG,PS,AeriaisWon,MotM,Rating
            items = line.split(',')
            name = items[0]
            playerId = items[1]
            team = items[2]
            Goals = items[6]
            Assist = items[7]
            player_id_info_dict.update({playerId: [name, team, Goals, Assist]})
        return player_id_info_dict

    def compute_correlations_by_round(self, game_by_round_dict,
                                      player_id_info_dict):
        correlated_coefficient_round_by_round = {}

        playerIds = player_id_info_dict.keys()

        for round_num in range(1, self.ROUND_NUMBER + 1):
            partial_player_value_dict = {}
            game_info_lists = game_by_round_dict.get(round_num)
            game_dir_all = []
            game_ha_id_all = []
            for game_info in game_info_lists:
                game_info_items = game_info.split('$')
                h_a_id = game_info_items[1]
                game_ha_id_all.append(h_a_id)
                game_dir = game_info[2]
                game_dir_all.append(game_dir)
                partial_player_value_dict = self.aggregate_partial_impact_values(dir_game=game_dir, ha_id=h_a_id,
                                                                                 partial_player_value_dict=partial_player_value_dict)

            player_assist_list = []
            player_goal_list = []
            partial_player_GIM_list = []
            for playerId in playerIds:
                player_assist_list.append(int(player_id_info_dict.get(playerId)[3]))
                player_goal_list.append(int(player_id_info_dict.get(playerId)[2]))
                partial_player_GIM_list.append(partial_player_value_dict.get(playerId))

            goal_correlation = self.compute_correlated_coefficient(partial_player_GIM_list,
                                                                   player_goal_list)
            assistant_correlation = self.compute_correlated_coefficient(partial_player_GIM_list,
                                                                        player_assist_list)

            correlated_coefficient_round_by_round.update({round_num: {'assistant': assistant_correlation,
                                                                      'goal': goal_correlation}})
        return correlated_coefficient_round_by_round

    def compute_correlated_coefficient(self, listA, listB):
        return pearsonr(listA, listB)[0]

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
            for round_number in range(1, self.ROUND_NUMBER + 1):
                round_correlated_coefficients = correlated_coefficient_round_by_round.get(round_number)
                round_correlated_coefficients_record = {}
                for standard_statistic in standard_statistic_fields:
                    round_correlated_coefficients_record.update(
                        {standard_statistic: round_correlated_coefficients.get(standard_statistic)})
                writer.writerow(round_correlated_coefficients_record)


if __name__ == "__main__":
    raw_data_path = "/cs/oschulte/soccer-data/sequences_append_goal/"
    interested_metric = ['Goals', 'Assists', 'Auto']

    # print 'still working'
