def select_feature_setting(feature_type):
    if feature_type == 5:
        interested_raw_features = ['angle', 'distance', 'duration', 'gameTimeRemain', 'periodId', 'home_away',
                                   'interrupted', 'manPower', 'outcome', 'scoreDiff', 'x', 'y']
        actions = ['caught-offside', 'pass_from_fk', 'cross_from_fk', 'pass_from_corner',
                   'cross_from_corner', 'cross', 'throw-in', 'through-ball', 'switch-of-play',
                   'long-ball', 'simple-pass', 'take-on_drible', 'skill', 'tackle', 'interception',
                   'aerial-challenge', 'clearance', 'ball-recovery', 'offside-provoked', 'own-goal',
                   'penalty_shot', 'fk_shot', 'corner_shot', 'standard_shot', 'blocked_shot',
                   'save', 'claim', 'punch', 'pick-up', 'smother', 'keeper-sweeper', 'penalty_save',
                   'penalising_foul', 'minor_foul', 'penalty_obtained', 'dangerous_foul', 'dangerous_foul_obtained',
                   'run_with_ball', 'dispossessed', 'bad-touch', 'miss', 'error']
        interested_compute_features = ['velocity_x', 'velocity_y']
        features_mean = [1.15034321e-01, 6.01738414e+01, 2.18661602e-01, 4.92712306e+01,
                         1.49666693e+00, 5.10437425e-01, 4.89562575e-01, 3.11270142e-01,
                         7.74979722e-03, 7.54722054e-01, 1.29755424e-01, 4.71762251e+01,
                         5.00664314e+01, 1.19359338e+01, 9.95259091e-01, 2.60039189e-03,
                         1.60720360e-02, 1.83977673e-03, 1.03249280e-03, 5.39452336e-03,
                         1.66964983e-02, 2.89519716e-02, 1.36688089e-03, 4.81159201e-03,
                         7.52568013e-02, 4.60947346e-01, 2.35694366e-02, 1.46214511e-04,
                         2.22440866e-02, 2.51578871e-02, 2.36813988e-02, 3.26274576e-02,
                         6.67643714e-02, 2.60039189e-03, 5.20206824e-05, 1.75328967e-04,
                         7.17157556e-04, 2.40108631e-03, 1.29248918e-02, 3.94243986e-03,
                         3.86451588e-03, 1.11641094e-03, 5.20420901e-04, 9.10147865e-03,
                         2.40836493e-04, 4.97728751e-04, 1.75328967e-04, 2.92643098e-04,
                         1.47207827e-02, 1.51352356e-04, 1.48826249e-03, 1.41783119e-03,
                         8.72070010e-02, 1.35844626e-02, 3.30566818e-02, 6.74342179e-05,
                         5.22347593e-04]

        features_scale = [9.33479400e-02, 2.25543229e+01, 2.93176260e-01, 2.69656562e+01,
                          4.99988891e-01, 4.99891048e-01, 4.99891048e-01, 4.63013003e-01,
                          2.26185237e-01, 4.30251874e-01, 1.18486366e+00, 2.40286278e+01,
                          2.96428327e+01, 1.86714518e+02, 2.50160618e+02, 5.09276924e-02,
                          1.25752637e-01, 4.28531440e-02, 3.21158335e-02, 7.32490442e-02,
                          1.28131671e-01, 1.67671569e-01, 3.69460760e-02, 6.91985591e-02,
                          2.63805260e-01, 4.98472557e-01, 1.51703389e-01, 1.20910352e-02,
                          1.47476395e-01, 1.56604495e-01, 1.52054563e-01, 1.77659524e-01,
                          2.49613481e-01, 5.09276924e-02, 7.21234887e-03, 1.32400237e-02,
                          2.67701931e-02, 4.89420177e-02, 1.12950604e-01, 6.26649586e-02,
                          6.20449949e-02, 3.33940798e-02, 2.28067986e-02, 9.49665296e-02,
                          1.55170387e-02, 2.23042825e-02, 1.32400237e-02, 1.71043111e-02,
                          1.20432891e-01, 1.23016035e-02, 3.85492874e-02, 3.76273962e-02,
                          2.82138158e-01, 1.15758045e-01, 1.78784612e-01, 8.21155713e-03,
                          2.28489550e-02]

        complete_feature_space = interested_raw_features + interested_compute_features + actions

        features_mean_dir = {}
        for index in range(0, len(complete_feature_space)):
            features_mean_dir.update({complete_feature_space[index]: features_mean[index]})

        features_scale_dir = {}
        for index in range(0, len(complete_feature_space)):
            features_mean_dir.update({complete_feature_space[index]: features_scale[index]})

        features_train = interested_raw_features + interested_compute_features + actions

        return features_train, features_mean_dir, features_scale_dir, actions
