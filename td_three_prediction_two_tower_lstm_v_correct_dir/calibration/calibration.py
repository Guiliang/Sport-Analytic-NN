import os
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import read_features_within_events


class Calibration:

    def __init__(self, bins, data_path):
        self.bins = bins
        self.bins_names = bins.keys()
        self.data_path = data_path
        pass

    def construct_bin_dicts(self):
        pass

    def aggregate_calibration_values(self):

        dir_all = os.listdir(self.data_path)
        for dir in dir_all:
            features_all = []
            for bin_name in self.bins_names:
                features = self.bins.get(bin_name).get('feature_name')
                features_all.append(features)

            features_all = read_features_within_events(feature_name_list=features_all,
                                                       data_path=self.data_path,
                                                       directory=dir)
