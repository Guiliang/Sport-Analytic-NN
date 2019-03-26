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
        features_mean = {}
        features_scale = {}

        features_train = interested_raw_features + interested_compute_features + actions

        return features_train, features_mean, features_scale, actions
