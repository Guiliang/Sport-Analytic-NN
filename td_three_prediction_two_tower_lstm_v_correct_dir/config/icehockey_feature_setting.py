

def select_feature_setting(feature_type):
    if feature_type == 5:
        features_train = ['velocity_x', 'velocity_y', 'xAdjCoord', 'yAdjCoord', 'time remained', 'scoreDifferential',
                          'Penalty',
                          'duration', 'block', 'carry', 'check', 'dumpin', 'dumpout', 'goal', 'lpr', 'offside', 'pass',
                          'puckprotection', 'reception',
                          'shot', 'shotagainst', 'event_outcome', 'home', 'away', 'angel2gate']
        features_mean = {'velocity_x': 1.75319470e+00, 'velocity_y': -1.82810625e-02, 'xAdjCoord': -7.14319081e+00,
                         'yAdjCoord': 1.01754097e-01, 'time remained': 1.79384554e+03, 'scoreDifferential': -5.07447651e-02,
                         'Penalty': 8.72872190e-02,
                         'duration': 1.22884672e+00, 'block': 6.73442968e-02, 'carry': 7.61053168e-02,
                         'check': 2.34492823e-02,
                         'dumpin': 2.58663194e-02, 'dumpout': 1.03478291e-02, 'goal': 1.79266542e-03, 'lpr': 2.06799529e-01,
                         'offside': 2.08754868e-03, 'pass': 2.73887097e-01, 'puckprotection': 3.17273090e-02,
                         'reception': 2.09955992e-01,
                         'shot': 4.16657917e-02, 'shotagainst': 1.03478291e-02, 'faceoff': 0.00000000e+00,
                         'event_outcome': 6.46196878e-01, 'home': 5.06214879e-01, 'away': 4.93785121e-01,
                         'angel2gate': 4.56133916e-01}
        features_scale = {'velocity_x': 3.85190530e+01, 'velocity_y': 3.22120227e+01, 'xAdjCoord': 6.00776950e+01,
                          'yAdjCoord': 2.74326827e+01, 'time remained': 1.05219380e+03, 'scoreDifferential': 1.44754521e+00,
                          'Penalty': 3.98020884e-01, 'duration': 2.38732281e+00, 'block': 2.50617323e-01,
                          'carry': 2.65166547e-01,
                          'check': 1.51325522e-01, 'dumpin': 1.58736426e-01, 'dumpout': 1.67725081e-01,
                          'goal': 4.23019122e-02,
                          'lpr': 4.05010474e-01, 'offside': 4.56419853e-02, 'pass': 4.45951741e-01,
                          'puckprotection': 1.75273178e-01, 'reception': 4.07276900e-01,
                          'shot': 1.99824307e-01, 'shotagainst': 1.01196599e-01, 'faceoff': 1.00000000e+00,
                          'event_outcome': 7.63170750e-01, 'home': 4.99961374e-01, 'away': 4.99961374e-01,
                          'angel2gate': 5.24486930e-01}
        actions = ['block', 'carry', 'check', 'dumpin', 'dumpout', 'goal', 'lpr', 'offside', 'pass', 'puckprotection',
                   'reception', 'shot', 'shotagainst', 'faceoff']
    else:
        raise ValueError("unknown feature type")

    return features_train, features_mean, features_scale, actions
