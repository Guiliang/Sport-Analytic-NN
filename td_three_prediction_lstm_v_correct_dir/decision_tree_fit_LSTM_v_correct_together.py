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

PLAYER_NAME = "Patrick Kane"
MAX_DEPTH = None
MAX_LEAF_NODE = None
feature_names = []
COMPUTE_IMPACT = True

if COMPUTE_IMPACT:
    trace_number_total = 10
    trace_number_total_pre = 10
else:
    trace_number_total = 10
    trace_number_total_pre = 0
for trace_number in range(trace_number_total - 1, 0 - 1, -1):
    status = "pre"
    feature_names = feature_names + ['T{0}_({1}):velocity_x'.format(trace_number, status),
                                     'T{0}_({1}):velocity_y'.format(trace_number, status),
                                     'T{0}_({1}):xAdjCoord'.format(trace_number, status),
                                     'T{0}_({1}):yAdjCoord'.format(trace_number, status),
                                     'T{0}_({1}):time remained'.format(trace_number, status),
                                     'T{0}_({1}):scoreDifferential'.format(trace_number, status),
                                     'T{0}_({1}):Penalty'.format(trace_number, status),
                                     'T{0}_({1}):duration'.format(trace_number, status),
                                     'T{0}_({1}):block'.format(trace_number, status),
                                     'T{0}_({1}):carry'.format(trace_number, status),
                                     'T{0}_({1}):check'.format(trace_number, status),
                                     'T{0}_({1}):dumpin'.format(trace_number, status),
                                     'T{0}_({1}):dumpout'.format(trace_number, status),
                                     'T{0}_({1}):goal'.format(trace_number, status),
                                     'T{0}_({1}):lpr'.format(trace_number, status),
                                     'T{0}_({1}):offside'.format(trace_number, status),
                                     'T{0}_({1}):pass'.format(trace_number, status),
                                     'T{0}_({1}):puckprotection'.format(trace_number, status),
                                     'T{0}_({1}):reception'.format(trace_number, status),
                                     'T{0}_({1}):shot'.format(trace_number, status),
                                     'T{0}_({1}):shotagainst'.format(trace_number, status),
                                     'T{0}_({1}):event_outcome'.format(trace_number, status),
                                     'T{0}_({1}):home'.format(trace_number, status),
                                     'T{0}_({1}):away'.format(trace_number, status),
                                     'T{0}_({1}):angel2gate'.format(trace_number, status)]
for trace_number in range(trace_number_total_pre - 1, 0 - 1, -1):
    status = "t"
    feature_names = feature_names + ['T{0}_({1}):velocity_x'.format(trace_number, status),
                                     'T{0}_({1}):velocity_y'.format(trace_number, status),
                                     'T{0}_({1}):xAdjCoord'.format(trace_number, status),
                                     'T{0}_({1}):yAdjCoord'.format(trace_number, status),
                                     'T{0}_({1}):time remained'.format(trace_number, status),
                                     'T{0}_({1}):scoreDifferential'.format(trace_number, status),
                                     'T{0}_({1}):Penalty'.format(trace_number, status),
                                     'T{0}_({1}):duration'.format(trace_number, status),
                                     'T{0}_({1}):block'.format(trace_number, status),
                                     'T{0}_({1}):carry'.format(trace_number, status),
                                     'T{0}_({1}):check'.format(trace_number, status),
                                     'T{0}_({1}):dumpin'.format(trace_number, status),
                                     'T{0}_({1}):dumpout'.format(trace_number, status),
                                     'T{0}_({1}):goal'.format(trace_number, status),
                                     'T{0}_({1}):lpr'.format(trace_number, status),
                                     'T{0}_({1}):offside'.format(trace_number, status),
                                     'T{0}_({1}):pass'.format(trace_number, status),
                                     'T{0}_({1}):puckprotection'.format(trace_number, status),
                                     'T{0}_({1}):reception'.format(trace_number, status),
                                     'T{0}_({1}):shot'.format(trace_number, status),
                                     'T{0}_({1}):shotagainst'.format(trace_number, status),
                                     'T{0}_({1}):event_outcome'.format(trace_number, status),
                                     'T{0}_({1}):home'.format(trace_number, status),
                                     'T{0}_({1}):away'.format(trace_number, status),
                                     'T{0}_({1}):angel2gate'.format(trace_number, status)]


def read_csv(csv_name):
    csv_dict_list = []
    with open(csv_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 1:
                csv_dict_list.append(float(row[0]))
            else:
                csv_dict_list.append(np.asarray([float(i) for i in row]))
    return np.asarray(csv_dict_list)


def dt_regression():
    if COMPUTE_IMPACT:
        data = read_csv("./decision-tree/sequence-impact-input-{0}-2018-08-29.csv".format(PLAYER_NAME))
        target = read_csv("./decision-tree/sequence-impact-value-{0}-2018-08-29.csv".format(PLAYER_NAME))
        regressor = DecisionTreeRegressor(random_state=0, max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAF_NODE)
        clf = regressor.fit(data, target)
        target_output = regressor.predict(data)
        diff_avg = sum(abs(target_output - target)) / target.size
        print "average training difference for impact is {0}".format(str(diff_avg))
        tree.export_graphviz(clf,
                             out_file='./decision-tree/imapact-sequence-{0}-tree-depth{1}-max_leaf{2}-2018-08-29.dot'.format(
                                 PLAYER_NAME, str(MAX_DEPTH), str(MAX_LEAF_NODE)),
                             feature_names=feature_names)
    else:
        data = read_csv("./decision-tree/sequence-input-{0}-2018-08-29.csv".format(PLAYER_NAME))
        target = read_csv("./decision-tree/sequence-value-{0}-2018-08-29.csv".format(PLAYER_NAME))
        regressor = DecisionTreeRegressor(random_state=0, max_depth=MAX_DEPTH, max_leaf_nodes=MAX_LEAF_NODE)
        clf = regressor.fit(data, target)
        target_output = regressor.predict(data)
        diff_avg = sum(abs(target_output - target)) / target.size
        print "average training difference is {0}".format(str(diff_avg))
        tree.export_graphviz(clf,
                             out_file='./decision-tree/sequence-{0}-tree-depth{1}-max_leaf{2}-2018-08-29.dot'.format(
                                 PLAYER_NAME, str(MAX_DEPTH), str(MAX_LEAF_NODE)),
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

    number2compute_dict = {
        # "time remained": [-1.1136, -1.5945, -1.4294, -0.8397, -0.8118],
        # "scoreDifferential": [0.3804],
        # "Penalty": [1.037],
        # "goal": [11.7794]
        # "velocity_x": [1.4044],
        # "yAdjCoord": [-0.6782],
        # "time remained": [-1.0068]
        "xAdjCoord": [0.0734],
        "angel2gate": [0.1614, -0.6543],
        "velocity_y": [-0.3018]

    }

    for key in number2compute_dict.keys():
        index = features_select_dict.get(key)
        number_mean = Mean[index]
        number_scale = scale[index]
        for number in number2compute_dict.get(key):
            origin_number = number * number_scale + number_mean
            print "{2} after standardization:{0}, before standardization:{1}".format(number, origin_number, key)


if __name__ == '__main__':
    dt_regression()
    # de_standardization()
