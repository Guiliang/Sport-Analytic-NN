import csv
import json

import matplotlib.pyplot as plt
import matplotlib as mpl

label_size = 20
# mpl.style.use('seaborn')
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


# mpl.style.use('seaborn')


class DrawRoundCorrelation:
    def __init__(self, rbr_results_dir):
        self.rbr_results_dir = rbr_results_dir
        self.metric_names_all = ['EG', 'GIM2t']
        self.field_dict = {'assistant': {}, 'goal': {}}
        for field_name in self.field_dict.keys():
            field_dict = self.field_dict.get(field_name)
            for method in self.metric_names_all:
                field_dict.update({method: []})
            self.field_dict.update({field_name: field_dict})
        self.ROUND_NUMBER = 60

    def read_round_by_round_correlation(self):
        methods_record_all = {}
        with open(self.rbr_results_dir) as fp:
            record_rbr_dict = json.load(fp)
        for metric_name in record_rbr_dict.keys():
            metric_values = record_rbr_dict.get(metric_name)
            round_number_all = sorted(metric_values.keys(), reverse=True)
            for round_number in round_number_all:
                values = metric_values.get(round_number)
                for field_name in values.keys():
                    correl_value = values.get(field_name)
                    metric_dict = self.field_dict.get(field_name)
                    inner_list = metric_dict.get(metric_name)
                    inner_list.append(correl_value)
                    metric_dict.update({metric_name: inner_list})
                    self.field_dict.update({field_name: metric_dict})
        return methods_record_all

    # def read_append_round_by_round_correlation(self, methods_record_all):
    #     record_all_dict = self.read_csv(self.append_round_by_round_dir, 'sta_auto')
    #     methods_record_all.update({'sta_auto': record_all_dict})
    #     # return methods_record_all

    def draw_round_by_round_correlation(self, methods_record_all, scale=2):

        field_names = {'assistant': 'Assists', 'goal': 'Goals', 'point': 'Points', 'auto': 'Auto'}
        # methods_marker_all = {'plusminus': '^', 'EG': '*', 'SI': 'x', 'GIM': 'P'}
        # methods_color_all = {'plusminus': 'b', 'EG': 'y', 'SI': 'g', 'GIM': 'r'}
        methods_marker_all = {'GIM-T1': '^', 'EG': '*', 'SI': 'x', 'GIM': 'P'}
        methods_color_all = {'GIM-T1': 'b', 'EG': 'y', 'SI': 'g', 'GIM': 'r'}

        x_list = range(1, self.ROUND_NUMBER + 1, scale)

        for field in self.field_dict.keys():

            plt.figure(figsize=(8, 7))

            for method in self.metric_names_all:
                field_name = method if field == 'auto' else field
                correlations = methods_record_all.get(field_name).get(method)

                for pop_number in range(0, self.ROUND_NUMBER / scale):
                    correlations.pop(pop_number + 1)

                # plt.plot(x_list, correlations, label=method)
                plt.plot(x_list, correlations,
                         label=method, marker=methods_marker_all.get(method), color=methods_color_all.get(method),
                         linewidth=2.0, markersize=15, alpha=0.5)
                # plt.show()
            # if field != 'auto':
            #     correlations = methods_record_all.get('sta_auto').get(field)
            #     for pop_number in range(0, self.ROUND_NUMBER / scale):
            #         correlations.pop(pop_number + 1)
            #     plt.plot(x_list, correlations,
            #              label=field.title(), marker=methods_marker_all.get('Sta'), color=methods_color_all.get('Sta'),
            #              linewidth=3.0, markersize=15, alpha=0.5, ls='--')
            plt.legend(loc='lower right', fontsize=25)
            # plt.title("Round by Round Correlation in 2015-2016 NHL season", fontsize=14)
            plt.xlabel("Round", fontsize=25)
            if field == 'auto':
                plt.ylabel("Auto-Correlation", fontsize=25)
            else:
                plt.ylabel("Correlation with {0}".format(field_names.get(field)), fontsize=25)
            # plt.show()
            plt.savefig("./figures/{0}_round_by_round.png".format(field))


if __name__ == "__main__":
    DRC = DrawRoundCorrelation()
    record_all_dict = DRC.read_round_by_round_correlation()
    # DRC.read_append_round_by_round_correlation(record_all_dict)
    DRC.draw_round_by_round_correlation(record_all_dict)
