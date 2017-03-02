import pickle
import scipy

gameInfo_dir = "/home/gla68/Documents/Hockey-data"

game1 = scipy.io.loadmat(gameInfo_dir + '/gamesInfo.mat')
gamesInfo = game1['gamesInfo']


def get_game_info(gameId):
    foundFlag = False
    for ff in range(len(gamesInfo[0])):  # fixed bug
        gamesInfoTemp = gamesInfo[0, ff]
        gamesInfoId = gamesInfoTemp['id']
        if gamesInfoId == gameId:
            foundFlag = True
            gamesInfoFound = gamesInfo[0, ff]  # find the gameInfo we want
            break

    if foundFlag:
        return gamesInfoFound
    else:
        raise ValueError("can't find the game")


def judge_home(gameId, teamId):
    foundFlag = False
    for ff in range(len(gamesInfo[0])):  # fixed bug
        gamesInfoTemp = gamesInfo[0, ff]
        gamesInfoId = gamesInfoTemp['id']
        if gamesInfoId == gameId:
            foundFlag = True
            gamesInfoFound = gamesInfo[0, ff]  # find the gameInfo we want
            break

    if foundFlag:
        gamesInfoHome = (gamesInfoFound['home'])[0, 0]
        gamesInfoHomeId = gamesInfoHome['id']
        gamesInfoHomeIdStr = ((gamesInfoHomeId[0])[0])[0]  # home ID

        gamesInfoVisitor = (gamesInfoFound['visitors'])[0, 0]
        gamesInfoVisitorId = gamesInfoVisitor['id']
        gamesInfoVisitorIdStr = ((gamesInfoVisitorId[0])[0])[0]  # visitors ID
        if gamesInfoHomeIdStr == teamId:
            return True
        elif gamesInfoVisitorIdStr == teamId:
            return False
        else:
            raise ValueError("can't judge home or away!")
    else:
        raise ValueError("can't find the game")


def write_pickle(dir, data):
    with open(dir, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    return data
