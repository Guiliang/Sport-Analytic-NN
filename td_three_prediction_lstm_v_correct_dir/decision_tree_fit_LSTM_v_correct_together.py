import csv
import numpy as np
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# ### Example ###
# boston = load_boston()
# regressor = DecisionTreeRegressor(random_state=0, max_depth=3, max_leaf_nodes=8)
# clf = regressor.fit(boston.data, boston.target)
# tree.export_graphviz(clf, out_file='./decision-tree/tree.dot')

PLAYER_NAME = "Taylor Hall"
feature_names = []
for trace_number in range(10, 0, -1):
    feature_names = feature_names + ['T{0}:velocity_x'.format(trace_number), 'T{0}:velocity_y'.format(trace_number),
                                     'T{0}:xAdjCoord'.format(trace_number), 'T{0}:yAdjCoord'.format(trace_number),
                                     'T{0}:time remained'.format(trace_number),
                                     'T{0}:scoreDifferential'.format(trace_number),
                                     'T{0}:Penalty'.format(trace_number), 'T{0}:duration'.format(trace_number),
                                     'T{0}:block'.format(trace_number),
                                     'T{0}:carry'.format(trace_number), 'T{0}:check'.format(trace_number),
                                     'T{0}:dumpin'.format(trace_number), 'T{0}:dumpout'.format(trace_number),
                                     'T{0}:goal'.format(trace_number), 'T{0}:lpr'.format(trace_number),
                                     'T{0}:offside'.format(trace_number), 'T{0}:pass'.format(trace_number),
                                     'T{0}:puckprotection'.format(trace_number), 'T{0}:reception'.format(trace_number),
                                     'T{0}:shot'.format(trace_number), 'T{0}:shotagainst'.format(trace_number),
                                     'T{0}:event_outcome'.format(trace_number), 'T{0}:home'.format(trace_number),
                                     'T{0}:away'.format(trace_number),
                                     'T{0}:angel2gate'.format(trace_number)]


def read_csv(csv_name):
    csv_dict_list = []
    with open(csv_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 1:
                csv_dict_list.append(row[0])
            else:
                csv_dict_list.append(np.asarray(row))
    return np.asarray(csv_dict_list)


def dt_regression():
    data = read_csv("./decision-tree/sequence-input-{0}.csv".format(PLAYER_NAME))
    target = read_csv("./decision-tree/sequence-value-{0}.csv".format(PLAYER_NAME))
    regressor = DecisionTreeRegressor(random_state=0, max_depth=3, max_leaf_nodes=8)
    clf = regressor.fit(data, target)
    tree.export_graphviz(clf, out_file='./decision-tree/sequence-{0}-tree.dot'.format(PLAYER_NAME),
                         feature_names=feature_names)


def de_standardization():
    Mean = [1.74025221e+00, - 1.75194799e-02, - 7.16269249e+00, 1.01382147e-01,
            1.79378049e+03, - 5.06994263e-02, 8.71900510e-02, 1.22843732e+00,
            6.74545530e-02, 7.60798893e-02, 2.34529789e-02, 2.58576772e-02,
            2.89613436e-02, 1.79206648e-03, 2.06730435e-01, 2.08714688e-03,
            2.73795589e-01, 3.17167086e-02, 2.09885844e-01, 4.16518708e-02,
            1.05338974e-02, 6.45696542e-01, 5.06210437e-01, 4.93789563e-01,
            4.56009158e-01]
    scale = [3.85699229e+01, 3.22348154e+01, 6.00791101e+01, 2.74292974e+01,
             1.05221308e+03, 1.44755408e+00, 3.98078716e-01, 2.38686914e+00,
             2.50807568e-01, 2.65125894e-01, 1.51337162e-01, 1.58710610e-01,
             1.67697896e-01, 4.22948575e-02, 4.04960445e-01, 4.56376018e-02,
             4.45905331e-01, 1.75244854e-01, 4.07226935e-01, 1.99792373e-01,
             1.02092774e-01, 7.63594117e-01, 4.99961429e-01, 4.99961429e-01,
             5.24450612e-01]

    features_select_dict = {'velocity_x': 0, 'velocity_y': 1, 'xAdjCoord': 2, 'yAdjCoord': 3, 'time remained': 4,
                            'scoreDifferential': 5,
                            'Penalty': 6, 'duration': 7, 'block': 8, 'carry': 9, 'check': 10, 'dumpin': 11,
                            'dumpout': 12, 'goal': 13, 'lpr': 14, 'offside': 15,
                            'pass': 16, 'puckprotection': 17, 'reception': 18,
                            'shot': 19, 'shotagainst': 20, 'event_outcome': 21, 'home': 22, 'away': 23,
                            'angel2gate': 24}

    number2compute_dict = {"time remained": [-1.4297, -1.7048, -1.1009, -1.594],
                           "scoreDifferential": [1.0713],
                           "Penalty": [1.037],
                           "goal": [11.7794]
                           }

    for key in number2compute_dict.keys():
        index = features_select_dict.get(key)
        number_mean = Mean[index]
        number_scale = scale[index]
        for number in number2compute_dict.get(key):
            origin_number = number * number_scale + number_mean
            print "{2} after standardization:{0}, before standardization:{1}".format(number, origin_number, key)


if __name__ == '__main__':
    # dt_regression()
    de_standardization()
