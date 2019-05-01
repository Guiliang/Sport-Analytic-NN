import json
import pickle


def construct_player_id_pair(info_dir):
    id_name_pair = {}
    with open(info_dir) as f:
        players_all = json.load(f)
    for player in players_all:
        stats = player['stats']
        for stat in stats:
            if stat['type'] == 'first_name':
                first_name = stat['value'].encode('utf-8')
            elif stat['type'] == 'last_name':
                last_name = stat['value'].encode('utf-8')
        id = player['playerId']
        id_name_pair.update({id: {'first_name': first_name, 'last_name': last_name}})
    return id_name_pair


if __name__ == '__main__':
    data_dir = '/home/gla68/Downloads/players.json'
    id_name_pair = construct_player_id_pair(info_dir=data_dir)
    with open('../resource/soccer_id_name_pair.json', 'w') as f:
        json.dump(id_name_pair, f)
