import tensorflow as tf
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import image_blending
from td_three_prediction_two_tower_lstm_v_correct_dir.nn_structure import td_tt_lstm
from td_three_prediction_two_tower_lstm_v_correct_dir.config.tt_lstm_config import TTLSTMCongfig
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import \
    start_lstm_generate_spatial_simulation
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import nn_simulation

if __name__ == '__main__':
    tt_lstm_config_path = "../icehockey-config.yaml"
    tt_lstm_config = TTLSTMCongfig.load(tt_lstm_config_path)
    history_action_type = ['reception', 'pass', 'reception']
    history_action_type_coord = [{'xAdjCoord': 50.18904442739472, 'yAdjCoord': 0.47699011276943787},
                                 {'xAdjCoord': 48.06645981534736, 'yAdjCoord': 0.7993870137732708},
                                 {'xAdjCoord': 38.898981773048014, 'yAdjCoord': 1.1692141494472155}]

    # history_action_type = []
    # history_action_type_coord = []

    feature_type = tt_lstm_config.learn.feature_type
    batch_size = tt_lstm_config.learn.batch_size
    iterate_num = tt_lstm_config.learn.iterate_num
    learning_rate = tt_lstm_config.learn.learning_rate
    if learning_rate == 1e-5:
        learning_rate_write = 5
    elif learning_rate == 1e-4:
        learning_rate_write = 4
    if_correct_velocity = tt_lstm_config.learn.if_correct_velocity
    action_type = 'shot'
    simulation_type = 'entire_spatial_simulation'
    data_simulation_dir = '../simulated_data/'
    draw_target = 'Q_home'
    model_type = tt_lstm_config.learn.model_type
    is_diff = True
    if is_diff:
        diff_str = "_diff"
    else:
        diff_str = ""

    simulated_data_all = start_lstm_generate_spatial_simulation(history_action_type=history_action_type,
                                                                history_action_type_coord=history_action_type_coord,
                                                                action_type=action_type,
                                                                data_simulation_dir=data_simulation_dir,
                                                                simulation_type=simulation_type,
                                                                feature_type=feature_type,
                                                                features_num=tt_lstm_config.learn.feature_number,
                                                                max_trace_length=tt_lstm_config.learn.max_trace_length
                                                                )

    sess_nn = tf.InteractiveSession()

    model_nn = td_tt_lstm.td_prediction_tt_embed(
        feature_number=tt_lstm_config.learn.feature_number,
        home_h_size=tt_lstm_config.Arch.HomeTower.home_h_size,
        away_h_size=tt_lstm_config.Arch.AwayTower.away_h_size,
        max_trace_length=tt_lstm_config.learn.max_trace_length,
        learning_rate=tt_lstm_config.learn.learning_rate,
        embed_size=tt_lstm_config.learn.embed_size,
        output_layer_size=tt_lstm_config.learn.output_layer_size,
        home_lstm_layer_num=tt_lstm_config.Arch.HomeTower.lstm_layer_num,
        away_lstm_layer_num=tt_lstm_config.Arch.AwayTower.lstm_layer_num,
        dense_layer_num=tt_lstm_config.learn.dense_layer_num,
        apply_softmax=tt_lstm_config.learn.apply_softmax
    )
    model_nn.initialize_ph()
    model_nn.build()
    model_nn.call()
    # saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/soccer-models/hybrid_sl_saved_NN" \
    #                                                             "/Scale-tt-three-cut_together_saved_networks_feature" \
    #                      + str(tt_lstm_config.learn.feature_type) + "_batch" \
    #                      + str(tt_lstm_config.learn.batch_size) + "_iterate" \
    #                      + str(tt_lstm_config.learn.iterate_num) + "_lr" \
    #                      + str(tt_lstm_config.learn.learning_rate) + "_" \
    #                      + str(tt_lstm_config.learn.model_type) + \
    #                      tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + \
    #                      str(tt_lstm_config.learn.max_trace_length)

    saved_network_path = tt_lstm_config.learn.save_mother_dir + "/oschulte/Galen/icehockey-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature" + str(
        tt_lstm_config.learn.feature_type) + "_batch" + str(
        tt_lstm_config.learn.batch_size) + "_iterate" + str(
        tt_lstm_config.learn.iterate_num) + "_lr" + str(
        tt_lstm_config.learn.learning_rate) + "_" + str(
        tt_lstm_config.learn.model_type) + tt_lstm_config.learn.if_correct_velocity + "_MaxTL" + str(
        tt_lstm_config.learn.max_trace_length)

    for data_index in range(0, len(simulated_data_all)):
        nn_image_save_dir = "./icehockey-image/{7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png". \
            format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        nn_half_image_save_dir = "./icehockey-image/right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        blend_image_save_dir = "./icehockey-image/blend {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)
        blend_half_image_save_dir = "./icehockey-image/blend right half {7} {0}-{1} with Dynamic LSTM feature{2}_batch{3}_iterate{4}_lr{5}_{6}{8}{9}.png".format(
            action_type, str(history_action_type[:data_index]), str(feature_type), str(batch_size),
            str(iterate_num),
            str(learning_rate),
            str(model_type), draw_target, if_correct_velocity, diff_str)

        simulate_data = simulated_data_all[data_index]
        nn_simulation(simulate_data=simulate_data,
                      model_path=saved_network_path,
                      model_nn=model_nn,
                      history_action_type=history_action_type[:data_index],
                      action_type=action_type[:data_index],
                      sess_nn=sess_nn,
                      nn_save_image_dir=nn_image_save_dir,
                      nn_half_save_image_dir=nn_half_image_save_dir)
        image_blending(nn_image_save_dir, blend_image_save_dir, nn_half_image_save_dir, blend_half_image_save_dir,
                       background_image_dir="../resource/hockey-field.png")
