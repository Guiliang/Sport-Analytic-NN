import numpy as np
import os
import scipy.io as sio
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import construct_simulation_data, \
    padding_hybrid_feature_input


def start_lstm_generate_spatial_simulation(history_action_type, history_action_type_coord,
                                           action_type, data_simulation_dir, simulation_type, feature_type):
    for history_index in range(0, len(history_action_type) + 1):
        state_ycoord_list = []
        for ycoord in np.linspace(-42.5, 42.5, 171):
            state_xcoord_list = []
            for xcoord in np.linspace(-100.0, 100.0, 401):
                set_dict = {'xAdjCoord': xcoord, 'yAdjCoord': ycoord}
                state_generated = construct_simulation_data(set_dict=set_dict)
                state_generated_list = [state_generated]
                for inner_history in range(0, history_index):
                    xAdjCoord = history_action_type_coord[inner_history].get('xAdjCoord')
                    yAdjCoord = history_action_type_coord[inner_history].get('yAdjCoord')
                    action = history_action_type[inner_history]
                    if action != action_type:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1, action_type: 0}
                    else:
                        set_dict_history = {'xAdjCoord': xAdjCoord, 'yAdjCoord': yAdjCoord, action: 1}
                    state_generated_history = construct_simulation_data(set_dict=set_dict_history)
                    state_generated_list = [state_generated_history] + state_generated_list

                state_generated_padding = padding_hybrid_feature_input(state_generated_list)
                state_xcoord_list.append(state_generated_padding)
            state_ycoord_list.append(np.asarray(state_xcoord_list))

        store_data_dir = data_simulation_dir + '/' + simulation_type

        if not os.path.isdir(store_data_dir):
            os.makedirs(store_data_dir)
        # else:
        #     raise Exception
        if ISHOME:
            sio.savemat(
                store_data_dir + "/LSTM_Home_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})
        else:
            sio.savemat(
                store_data_dir + "/LSTM_Away_" + simulation_type + "-" + action_type + '-' + str(
                    history_action_type[0:history_index]) + "-feature" + str(
                    feature_type) + ".mat",
                {'simulate_data': np.asarray(state_ycoord_list)})


if __name__ == '__main__':
    history_action_type = ['reception', 'pass', 'reception']
    history_action_type_coord = [{'xAdjCoord': 50.18904442739472, 'yAdjCoord': 0.47699011276943787},
                             {'xAdjCoord': 48.06645981534736, 'yAdjCoord': 0.7993870137732708},
                             {'xAdjCoord': 38.898981773048014, 'yAdjCoord': 1.1692141494472155}]
    feature_type = 5
    if_correct_velocity = "_v_correct_"
    action_type = 'shot'
    simulation_type = 'entire_spatial_simulation'
    data_simulation_dir = '../simulated_data/'
    start_lstm_generate_spatial_simulation(history_action_type=history_action_type,
                                           history_action_type_coord=history_action_type_coord,
                                           action_type=action_type,
                                           data_simulation_dir=data_simulation_dir,
                                           simulation_type=simulation_type,
                                           feature_type=feature_type)
