import scipy.io as sio


def print_mark_info(data_store_dir, game_name_dir, home_max_index, away_max_index, home_maxs, away_maxs):
    training_data_info = sio.loadmat(data_store_dir + game_name_dir + "/" + "training_data_dict_all_name.mat")
    training_data_info = training_data_info["training_data_dict_all_name"]
    home_training_data_info = training_data_info[home_max_index]
    away_training_data_info = training_data_info[away_max_index]
    print "\nhome_training_data_info"
    print zip(home_maxs.tolist(), home_max_index.tolist())
    print home_training_data_info
    print "\naway_training_data_info"
    print zip(away_maxs.tolist(), away_max_index.tolist())
    print away_training_data_info
