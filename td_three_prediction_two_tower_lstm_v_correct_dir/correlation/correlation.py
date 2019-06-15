import csv
import json
import numpy as np
import td_three_prediction_two_tower_lstm_v_correct_dir.resource.plus_minus_1718 as plus_minus


class Correlation:
    def __init__(self, game_info_path, online_info_path_list):
        self.online_info_path_dict = {'summary': online_info_path_list[0],
                                      'defensive': online_info_path_list[1],
                                      'offensive': online_info_path_list[2]}
        self.game_info_path = game_info_path
        self.ranking_dir_dict = {
            'PM': ['', ''],
            'GIM': ['GIM', '../compute_impact/player_impact/ijcai_soccer_player_GIM_2019June01.json'],
            'SI': ['', '../resource/soccer_player_markov_impact-2019June13.json'],
            'GIM2t': ['GIM', '../compute_impact/player_impact/soccer_player_GIM_2019June13_back_difference_.json'],
            'EG': ['GIM', '../compute_impact/player_impact/bak-soccer_player_GIM_2019June05_expected_goal.json'],
            # 'PM': ['', ''],
            # 'ALG': ''
        }
        self.interested_standard_metric = {'summary': ['Mins', 'Goals', 'Assists', 'Yel', 'Red',
                                                       'SpG', 'PS', 'AeriaisWon', 'MotM'],
                                           'defensive': ['Mins', 'Tackles', 'Inter', 'Fouls', 'Offsides',
                                                         'Clear', 'Drb', 'Blocks', 'OwnG'],
                                           'offensive': ['Mins', 'Goals', 'SpG', 'KeyP', 'Drb', 'Fouled',
                                                         'Off', 'Disp', 'UnsTch']
                                           }
        self.game_info_file = open(self.game_info_path)
        game_reader = csv.DictReader(self.game_info_file)
        self.game_info_all = []
        for r in game_reader:
            p_name = r['playerName']
            t_name = r['teamName']
            id = r['playerId']
            self.game_info_all.append([p_name, t_name, id])

    def __del__(self):
        self.game_info_file.close()

    def get_id(self, playername, teamname):
        for info in self.game_info_all:
            p_name = info[0]
            t_name = info[1]
            # if playername == p_name:
            #     print('find name')
            if playername in p_name and teamname in t_name:
                return True, info[2]
        return False, ' '

    def get_markov_rank_value(self, metric_name):
        metric_info = self.ranking_dir_dict.get(metric_name)
        with open(metric_info[1]) as f:
            d = json.load(f)
        return d

    def get_GIM_rank_value(self, metric_name):
        metric_info = self.ranking_dir_dict.get(metric_name)
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

    def compute_correlation(self, rank_value_dict, interest_metric, category):
        mins_online_list = []
        mins_game_list = []
        with open(self.online_info_path_dict[category]) as online_info_file:
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
                if id not in rank_value_dict:
                    continue
                value = rank_value_dict[id]
                if not Flag:
                    continue
                # print(value)
                mins_online_list.append(float(standard_value))
                # print(value)
                mins_game_list.append(float(value))
                i += 1

        # print(len(mins_online_list))
        # print(len(mins_game_list))
        # print('matched number is ' + str(i))
        return np.corrcoef(mins_online_list, mins_game_list)

    def compute_all_correlations(self):
        correlation_record_all_dict = {'summary': {},
                                       'defensive': {},
                                       'offensive': {}}

        for category in ['summary', 'defensive', 'offensive']:
            interest_metric_all = self.interested_standard_metric.get(category)
            metric_string = 'model'
            for interest_metric in interest_metric_all:
                metric_string += ' & ' + interest_metric
            print metric_string
            correlation_record_rank_dict = correlation_record_all_dict.get(category)
            for rank_value_name in self.ranking_dir_dict.keys():
                if rank_value_name == 'GIM' or rank_value_name == 'GIM2t' or rank_value_name == 'EG':
                    rank_value_dict = self.get_GIM_rank_value(rank_value_name)
                elif rank_value_name == 'SI':
                    rank_value_dict = self.get_markov_rank_value(rank_value_name)
                elif rank_value_name == 'PM':
                    plus_minus_dict = plus_minus.pm
                    rank_value_dict = {}
                    for player_id in plus_minus_dict:
                        rank_value_dict.update({str(player_id): plus_minus_dict.get(player_id)})
                correlation_rank_list = []
                for interest_metric in interest_metric_all:
                    correlation = self.compute_correlation(rank_value_dict, interest_metric, category)
                    correlation_rank_list.append(correlation)
                str_line = rank_value_name + ' '
                for index in range(0, len(correlation_rank_list)):
                    str_line = str_line + ' & ' + str(round(correlation_rank_list[index][0][1], 3))
                str_line += '\\\\'
                print(str_line)
                correlation_record_rank_dict.update({rank_value_name: correlation_rank_list})

            correlation_record_all_dict.update({category: correlation_record_rank_dict})
            print('\\hline')

        return correlation_record_all_dict
