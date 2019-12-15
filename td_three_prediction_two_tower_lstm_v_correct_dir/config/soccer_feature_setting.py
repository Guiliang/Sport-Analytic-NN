def select_feature_setting(feature_type):
    if feature_type == 5:
        interested_raw_features = ['angle',
                                   'distance',
                                   'duration',
                                   'gameTimeRemain',
                                   'periodId',
                                   'interrupted',
                                   'manPower',
                                   'outcome',
                                   'scoreDiff',
                                   'x',
                                   'y'
                                   ]
        interested_compute_features = ['velocity_x',
                                       'velocity_y',
                                       'distance_x',
                                       'distance_y',
                                       'distance_to_goal'
                                       ]

        actions = ['caught-offside', 'pass_from_fk', 'cross_from_fk', 'pass_from_corner',
                   'cross_from_corner', 'cross', 'throw-in', 'through-ball', 'switch-of-play',
                   'long-ball', 'simple-pass', 'take-on_drible', 'skill', 'tackle', 'interception',
                   'aerial-challenge', 'clearance', 'ball-recovery', 'offside-provoked', 'own-goal',
                   'penalty_shot', 'fk_shot', 'corner_shot', 'standard_shot', 'blocked_shot',
                   'save', 'claim', 'punch', 'pick-up', 'smother', 'keeper-sweeper', 'penalty_save',
                   'penalising_foul', 'minor_foul', 'penalty_obtained', 'dangerous_foul', 'dangerous_foul_obtained',
                   'run_with_ball', 'dispossessed', 'bad-touch', 'miss', 'error', 'goal']
        features_mean = [1.14834335e-01, 6.00962937e+01, 2.18816976e-01, 4.92630563e+01,
                         1.49678182e+00, 3.11587882e-01, 7.75192473e-03, 7.55148467e-01,
                         1.29772400e-01, 4.72680586e+01, 5.00663159e+01, 1.05478686e+01,
                         -3.31495439e-01, 5.27319414e+01, 2.55756127e+01, 6.18125980e+01,
                         2.59587114e-03, 1.60440950e-02, 1.83657830e-03, 1.03069783e-03,
                         5.38514504e-03, 1.66674716e-02, 2.89016390e-02, 1.36450459e-03,
                         4.80322711e-03, 7.51259682e-02, 4.60145995e-01, 2.35284614e-02,
                         1.45960318e-04, 2.22054155e-02, 2.51141504e-02, 2.36402290e-02,
                         3.25707352e-02, 6.66483023e-02, 2.59587114e-03, 5.19302451e-05,
                         1.75024159e-04, 7.15910786e-04, 2.39691205e-03, 1.29024220e-02,
                         3.93558598e-03, 3.85779747e-03, 1.11447007e-03, 5.19516155e-04,
                         9.08565584e-03, 2.40417801e-04, 4.96863456e-04, 1.75024159e-04,
                         2.92134342e-04, 1.46951908e-02, 1.51089232e-04, 1.48567516e-03,
                         1.41536631e-03, 8.70553927e-02, 1.35608462e-02, 3.29992131e-02,
                         6.73169844e-05, 5.21439498e-04, 1.73848783e-03]

        features_scale = [9.33897973e-02, 2.26213058e+01, 2.93384409e-01, 2.69654524e+01,
                          4.99989643e-01, 4.63142390e-01, 2.26252391e-01, 4.29999139e-01,
                          1.18500223e+00, 2.41083751e+01, 2.96170548e+01, 3.46585362e+02,
                          4.68577095e+02, 2.41083751e+01, 1.49352726e+01, 2.04511236e+01,
                          5.08835199e-02, 1.25645064e-01, 4.28159465e-02, 3.20879337e-02,
                          7.31856902e-02, 1.28022135e-01, 1.67530100e-01, 3.69139908e-02,
                          6.91386731e-02, 2.63594494e-01, 4.98409127e-01, 1.51574645e-01,
                          1.20805221e-02, 1.47351061e-01, 1.56471818e-01, 1.51925536e-01,
                          1.77510232e-01, 2.49411921e-01, 5.08835199e-02, 7.20607718e-03,
                          1.32285119e-02, 2.67469299e-02, 4.88995589e-02, 1.12853664e-01,
                          6.26106791e-02, 6.19912483e-02, 3.33650720e-02, 2.27869756e-02,
                          9.48847021e-02, 1.55035480e-02, 2.22848958e-02, 1.32285119e-02,
                          1.70894412e-02, 1.20329723e-01, 1.22909074e-02, 3.85158140e-02,
                          3.75947210e-02, 2.81916213e-01, 1.15658764e-01, 1.78634445e-01,
                          8.20441666e-03, 2.28290954e-02, 4.16589185e-02]

        complete_feature_space = interested_raw_features + interested_compute_features + actions

        features_mean_dic = {}
        for index in range(0, len(complete_feature_space)):
            features_mean_dic.update({complete_feature_space[index]: features_mean[index]})

        features_scale_dic = {}
        for index in range(0, len(complete_feature_space)):
            features_scale_dic.update({complete_feature_space[index]: features_scale[index]})

        features_train = interested_raw_features + interested_compute_features + actions

        return features_train, features_mean_dic, features_scale_dic, actions

select_feature_setting(5)