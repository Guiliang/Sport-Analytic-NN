import sys

print sys.path
sys.path.append('/Local-Scratch/PycharmProjects/Sport-Analytic-NN/')

import numpy as np
import matplotlib.pyplot as plt
import os
from td_three_prediction_two_tower_lstm_v_correct_dir.support.data_processing_tools import read_features_within_events
from td_three_prediction_two_tower_lstm_v_correct_dir.support.plot_tools import image_blending


def plot_2d_soccer_goals(soccer_data_store_dir, save_plot_dir):
    goal_shot_position = []
    dir_all = os.listdir(soccer_data_store_dir)

    top_score_number = 0
    bottom_score_number = 0

    for game_name_dir in dir_all:
        actions_position = read_features_within_events(data_path=soccer_data_store_dir,
                                                       directory=game_name_dir,
                                                       feature_name_list=['action', 'x', 'y'])

        for item_index in range(0, len(actions_position)):
            item = actions_position[item_index]
            # if 'goal' in item.get('action'):
            if 'goal' == item.get('action'):
                # if actions_position[item_index - 1].get('action') == 'own-goal':
                if 'shot' not in actions_position[item_index - 1].get('action'):
                    # print(actions_position[item_index - 1].get('action'))
                    continue
                else:
                    # print(actions_position[item_index - 1].get('action'))
                    goal_shot_position.append([actions_position[item_index - 1].get('x'),
                                               actions_position[item_index - 1].get('y')])
                    if actions_position[item_index - 1].get('y') > 50:
                        top_score_number += 1
                    elif actions_position[item_index - 1].get('y') < 50:
                        bottom_score_number += 1

    goal_shot_position = np.asarray(goal_shot_position)

    print(
        'score position summation is top: {0} and bottom: {1}'.format(str(top_score_number), str(bottom_score_number)))
    print('mean coordinate is {0}'.format(str(np.mean(goal_shot_position, axis=0))))

    plt.figure(figsize=(10, 6))
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel('X Coordinate', fontsize=20)
    plt.ylabel('Y Coordinate', fontsize=20)
    plt.scatter(goal_shot_position[:, 0], goal_shot_position[:, 1], s=8)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    # plt.show()
    plt.savefig('./tmp_goal_scatter.png')
    # image_y = [66, 539]
    # image_x = [100, 800]
    image_blending(value_Img_dir='./tmp_goal_scatter.png', save_dir=save_plot_dir,
                   value_Img_half_dir=None, half_save_dir=None,
                   background_image_dir="../resource/soccer-field.png", sport='soccer',
                   image_x=[103, 824], image_y=[65, 540])
    # value_Img_half_dir, half_save_dir


if __name__ == '__main__':
    soccer_data_store_dir = "/cs/oschulte/soccer-data/sequences_append_goal/"
    save_plot_dir = './goal_scatter.png'
    plot_2d_soccer_goals(soccer_data_store_dir, save_plot_dir)
