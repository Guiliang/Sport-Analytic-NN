import csv
import json
import numpy as np 
null = None
online_info_path = '/cs/oschulte/miyunLuo/Documents/Code/Soccer_ActSpkLodThGo_twoModel/action_value_k=10/online_info/Soccer_defensive.csv'
game_info_path = '/cs/oschulte/miyunLuo/Documents/Code/Soccer_ActSpkLodThGo_twoModel/action_value_k=10/player_info/player_team_id_name_value.csv'
GIM_dict = {}
with open('soccer_player_GIM.json') as f:
    d = json.load(f)
    for k in d.keys():
        dic = d[k]
        gim = dic['GIM']
        id = dic['id']
        if gim == null:
            continue
        value = gim['value']
        GIM_dict[str(id)] = value

# name,team,Apps, | Mins,Tackles,Inter,Fouls,Offsides,Clear,Drb,Blocks,OwnG,Rating
# playerId,playerName,teamId,teamName,value

# ----------------------  Mins ------------------------------#
def get_id(playername, teamname):
    with open(game_info_path) as game_info_file:
        game_reader = csv.DictReader(game_info_file)
        for r in game_reader:
            p_name = r['playerName']
            t_name = r['teamName']
            if playername in p_name and teamname in t_name:
                return True, r['playerId']
        return False, ' '

mins_online_list = []
mins_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        mins = r['Mins']
        if mins == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        mins_online_list.append(float(mins))
        mins_game_list.append(float(value))

print(len(mins_online_list))
print(len(mins_game_list))
print('Mins')
print(np.corrcoef(mins_online_list, mins_game_list))

# ----------------------  Tackles ------------------------------#
goals_online_list = []
goals_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        goals = r['Tackles']
        if goals == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        goals_online_list.append(float(goals))
        goals_game_list.append(float(value))

print(len(goals_online_list))
print(len(goals_game_list))
print('Tackles')
print(np.corrcoef(goals_online_list, goals_game_list)) 

# ----------------------  Inter ------------------------------#
assists_online_list = []
assists_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        assists = r['Inter']
        if assists == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        assists_online_list.append(float(assists))
        assists_game_list.append(float(value))

print(len(assists_online_list))
print(len(assists_game_list))
print('Inter')
print(np.corrcoef(assists_online_list, assists_game_list)) 

# ----------------------  Fouls ------------------------------#
yel_online_list = []
yel_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        yel = r['Fouls']
        if yel == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        yel_online_list.append(float(yel))
        yel_game_list.append(float(value))

print(len(yel_online_list))
print(len(yel_game_list))
print('Fouls')
print(np.corrcoef(yel_online_list, yel_game_list)) 

# ----------------------  Offsides ------------------------------#
red_online_list = []
red_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        red = r['Offsides']
        if red == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        red_online_list.append(float(red))
        red_game_list.append(float(value))

print(len(red_online_list))
print(len(red_game_list))
print('Offsides')
print(np.corrcoef(red_online_list, red_game_list)) 

# ----------------------  Clear ------------------------------#
spg_online_list = []
spg_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        spg = r['Clear']
        if spg == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        spg_online_list.append(float(spg))
        spg_game_list.append(float(value))

print(len(spg_online_list))
print(len(spg_game_list))
print('Clear')
print(np.corrcoef(spg_online_list, spg_game_list)) 

# ----------------------  Drb ------------------------------#
ps_online_list = []
ps_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        ps = r['Drb']
        if ps == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        ps_online_list.append(float(ps))
        ps_game_list.append(float(value))

print(len(ps_online_list))
print(len(ps_game_list))
print('Drb')
print(np.corrcoef(ps_online_list, ps_game_list)) 

# ----------------------  Blocks ------------------------------#
aer_online_list = []
aer_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        aer = r['Blocks']
        if aer == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        aer_online_list.append(float(aer))
        aer_game_list.append(float(value))

print(len(aer_online_list))
print(len(aer_game_list))
print('Blocks')
print(np.corrcoef(aer_online_list, aer_game_list)) 

# ----------------------  OwnG ------------------------------#
motm_online_list = []
motm_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        motm = r['OwnG']
        if motm == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        motm_online_list.append(float(motm))
        motm_game_list.append(float(value))

print(len(motm_online_list))
print(len(motm_game_list))
print('OwnG')
print(np.corrcoef(motm_online_list, motm_game_list))

# ----------------------  Rating ------------------------------#
rating_online_list = []
rating_game_list = []

with open(online_info_path) as online_info_file:
    online_reader = csv.DictReader(online_info_file)
    for r in online_reader:
        playername = r['name']
        teamname = r['team']
        if teamname[0] == '"':
            teamname = teamname[1:-1]
        teamname = teamname.split(',')[0]
        rating = r['Rating']
        if rating == '-':
            continue
        #print(playername, ' ', teamname)
        Flag, id = get_id(playername, teamname)
        if id not in GIM_dict:
            continue
        value = GIM_dict[id]
        if Flag == False:
            continue
        #print(value)
        rating_online_list.append(float(rating))
        rating_game_list.append(float(value))

print(len(rating_online_list))
print(len(rating_game_list))
print('Rating')
print(np.corrcoef(rating_online_list, rating_game_list)) 
