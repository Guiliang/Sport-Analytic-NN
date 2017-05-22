import os
import scipy.io as sio
import shutil


def test_feature4_empty_data():
    DATA_STORE = "/media/gla68/Windows/Hockey-data/Hockey-Training-All-feature4-scale-neg_reward"
    DIR_GAMES_ALL = os.listdir(DATA_STORE)

    for dir_game in DIR_GAMES_ALL:
        game_files = os.listdir(DATA_STORE + "/" + dir_game)
        for filename in game_files:
            if filename.startswith("reward"):
                reward_name = filename
            elif filename.startswith("state"):
                state_name = filename

        reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
        try:
            reward = (reward['reward'][0]).tolist()
        except:
            print "\n" + dir_game
            # shutil.rmtree(DATA_STORE + "/" + dir_game)
            continue


if __name__ == '__main__':
    test_feature4_empty_data()
