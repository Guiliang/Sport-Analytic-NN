import pickle
import os
import json
import operator
import numpy as np
import scipy.io as sio
from sklearn.tree import DecisionTreeRegressor
import td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools as tools
from sklearn.ensemble import GradientBoostingRegressor
from td_three_prediction_two_tower_lstm_v_correct_dir.config.soccer_feature_setting import select_feature_setting


class TreeRegression:
    def __init__(self, cart_model_name, data_name,
                 model_data_store_dir, game_data_dir,
                 difference_type, min_sample_leaf,
                 trace_length=10,
                 action_selected=None):
        self.difference_type = difference_type
        self.game_data_dir = game_data_dir
        self.model_data_store_dir = model_data_store_dir
        self.cart_model_name = cart_model_name
        self.max_depth = None
        self.max_leaf_node = None
        self.min_sample_leaf = min_sample_leaf
        self.model_store_mother_dir = '/cs/oschulte/Galen/soccer-models/dt_models/'
        self.data_name = data_name
        self.write_feature_importance_dir = './dt_record/feature_importance_{0}_tree.txt'.format(action_selected)
        self.write_print_tree_dir = './dt_record/print_{0}_tree.txt'.format(action_selected)
        self.action_selected = action_selected
        self.regressor = None
        features_train, self.features_mean_dic, self.features_scale_dic, _ = select_feature_setting(5)
        self.features_train_all = []
        self.trace_length = trace_length
        for j in range(0, self.trace_length):
            for feature in features_train[0:16]:  # ignore the actions
                self.features_train_all.append(feature + '${0}'.format(str(j)))

    def cart_validation_model(self, data_train, target_train, data_test, target_test, read_model=True, test_flag=False):

        # for train_input_feature in data_train:
        #     self.return_unscaled_input_feature(train_input_feature)

        if read_model:
            self.regressor = pickle.load(open(self.cart_model_name, 'rb'))
        else:
            self.regressor = DecisionTreeRegressor(random_state=0, max_depth=self.max_depth,
                                                   max_leaf_nodes=self.max_leaf_node,
                                                   min_samples_leaf=self.min_sample_leaf)
            self.regressor.fit(data_train, target_train)
            if not test_flag:
                with open(self.model_store_mother_dir + self.cart_model_name, 'wb') as f:
                    pickle.dump(self.regressor, f)
        feature_importances_dict = {}
        feature_importances = self.regressor.feature_importances_
        for index in range(0, len(feature_importances)):
            feature_importance = feature_importances[index]
            feature_importances_dict.update({self.features_train_all[index]: feature_importance})
        feature_importances_sorted = sorted(feature_importances_dict.items(), key=operator.itemgetter(1), reverse=True)
        print '\nfeature importance:'
        print feature_importances_sorted[:20]
        print '\n'
        with open(self.write_feature_importance_dir, 'w') as f:
            for feature_importance in feature_importances_sorted:
                f.write(str(feature_importance[0]) + '&' + str(feature_importance[1]) + '\\\\ \n')

        target_output = self.regressor.predict(data_test)
        mae = sum(abs(target_output - target_test)) / target_test.size
        var_mae = np.var(abs(target_output - target_test), axis=0)

        mse = sum((target_output - target_test) ** 2) / target_test.size
        var_mse = np.var((target_output - target_test) ** 2, axis=0)

        return mae, var_mae, mse, var_mse

    def gather_all_training_data(self):
        all_input_list = []
        all_impact_list = []
        all_model_data_dir = os.listdir(self.model_data_store_dir)
        for model_dir in all_model_data_dir:
            if model_dir.startswith('.'):
                continue
            game_input_list, game_impact_list = self.gather_game_training_data(dir_game=model_dir)
            all_input_list = all_input_list + game_input_list
            all_impact_list = all_impact_list + game_impact_list
        return all_input_list, all_impact_list

    def return_unscaled_threshold(self, threshold, feature_node_name_all):
        # value * variance + mean
        threshold_unscaled = []
        for index in range(0, len(feature_node_name_all)):
            feature_node_name = feature_node_name_all[index]
            if feature_node_name is not None:
                mean = self.features_mean_dic.get(feature_node_name.split('$')[0])
                variance = self.features_scale_dic.get(feature_node_name.split('$')[0])
                try:
                    threshold_unscaled.append(threshold[index] * variance + mean)
                except:
                    print ('error')
            else:
                threshold_unscaled.append(None)

        return threshold_unscaled

    def return_unscaled_input_feature(self, feature_input):
        unscaled_input_feature_list = []
        for index in range(0, len(self.features_train_all)):
            feature_name = self.features_train_all[index]
            mean = self.features_mean_dic.get(feature_name.split('$')[0])
            variance = self.features_scale_dic.get(feature_name.split('$')[0])
            feature_unrescaled_value = feature_input[index] * variance + mean
            unscaled_input_feature_list.append('{0}:{1}'.format(feature_name, str(feature_unrescaled_value)))
        return unscaled_input_feature_list

    def print_decision_path(self, max_print_node=20):

        estimator = self.regressor
        assert estimator is not None
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        node_values = estimator.tree_.value
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        feature_node_name_all = []
        for i in range(n_nodes):
            if feature[i] >= 0:
                feature_node_name_all.append(self.features_train_all[feature[i]])
            else:
                feature_node_name_all.append(None)
        threshold = self.return_unscaled_threshold(threshold=threshold, feature_node_name_all=feature_node_name_all)
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # n_nodes = n_nodes if n_nodes < max_print_node else max_print_node
        with open(self.write_print_tree_dir, 'w') as f:
            print("The binary tree structure has %s nodes and has "
                  "the following tree structure:"
                  % n_nodes)
            f.write("The binary tree structure has %s nodes and has "
                    "the following tree structure:\n"
                    % n_nodes)
            for i in range(n_nodes):
                if is_leaves[i]:
                    print("%snode=%s leaf node value%s." % (node_depth[i] * "\t", i, sum(node_values[i])/len(node_values[i])))
                    f.write("%snode=%s leaf node value%s.\n" % (node_depth[i] * "\t", i, sum(node_values[i])/len(node_values[i])))
                else:
                    print("%snode=%s test node value%s: go to node %s if %s <= %s else to "
                          "node %s."
                          % (node_depth[i] * "\t",
                             i,
                             sum(node_values[i])/len(node_values[i]),
                             children_left[i],
                             # feature[i],
                             self.features_train_all[feature[i]],
                             threshold[i],
                             children_right[i],
                             ))
                    f.write("%snode=%s test node value%s: go to node %s if %s <= %s else to "
                          "node %s."
                          % (node_depth[i] * "\t",
                             i,
                             sum(node_values[i]) / len(node_values[i]),
                             children_left[i],
                             # feature[i],
                             self.features_train_all[feature[i]],
                             threshold[i],
                             children_right[i],
                             ))
            print()

    def gather_game_training_data(self, dir_game):
        game_impact_list = []
        game_input_list = []
        # print self.data_name
        for file_name in os.listdir(self.model_data_store_dir + "/" + dir_game):
            # print file_name
            if file_name.startswith('.'):
                continue
            if file_name == self.data_name:
                model_data_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                with open(model_data_name) as f:
                    model_data = json.load(f)
            elif file_name.startswith("state"):
                state_input_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                state_input = sio.loadmat(state_input_name)['state']
            elif file_name.startswith("home_away"):
                home_identifier_name = self.model_data_store_dir + "/" + dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_away"][0]
            else:
                continue
        # TODO: fix the names of features

        actions = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
                                                   feature_name='action')
        # print actions
        # playerIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game + '.json',
        #                                              feature_name='playerId')
        # teamIds = tools.read_feature_within_events(data_path=self.game_data_dir, directory=dir_game+'.json',
        #                                           feature_name='teamIds')
        # print len(playerIds)
        # print len(actions)
        for event_Index in range(0, len(actions)):

            if self.action_selected is not None:
                if self.action_selected not in actions[event_Index]:
                    continue

            # playerId = playerIds[event_Index]
            # teamId = teamIds[player_Index]
            # if int(teamId_target) == int(teamId):
            # print model_data
            model_value = model_data[str(event_Index)]
            state_input_value = state_input[event_Index]
            state_input_value = state_input_value[:, 0:16]
            state_input_value_flat = tools.flat_state_input(state_input=state_input_value)
            game_input_list.append(state_input_value_flat)
            if event_Index - 1 >= 0:  # define model pre
                if actions[event_Index - 1] == "goal":
                    model_value_pre = model_data[str(event_Index)]  # the goal cancel out here, just as we cut the game
                else:
                    model_value_pre = model_data[str(event_Index - 1)]
            else:
                model_value_pre = model_data[str(event_Index)]
            if event_Index + 1 < len(actions):  # define model next
                if actions[event_Index + 1] == "goal":
                    model_value_nex = model_data[str(event_Index)]
                else:
                    model_value_nex = model_data[str(event_Index + 1)]
            else:
                model_value_nex = model_data[str(event_Index)]

            ishome = home_identifier[event_Index]

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
            game_impact_list.append(value)
        return game_input_list, game_impact_list
