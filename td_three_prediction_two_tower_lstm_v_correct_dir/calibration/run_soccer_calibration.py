if __name__ == '__main__':
    calibration_bins = {'period': {'feature_name': ['sec', 'min'], 'range': {[1, 2]}},
                        'score_differential': {'feature_name': ['scoreDiff'], 'range': range(-8, 8)},
                        'pitch': {'feature_name': ['x'], 'range': ['left', 'right']},
                        'manpower': {'feature_name': ['manPower'], 'range': [-1, 0, 1]}
                        }
