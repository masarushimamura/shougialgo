
# coding: utf-8

# %load shougi_algorithm.py


# # 仕組み
# 1、評価と選定
# ３要素（駒の価値、成るまでの距離、王（玉）までの距離　）でどの駒

# # 各オブジェクト説明
# 
# kind = {0: "ou",  1: "gyoku",  2: "hisha",  3: "kaku", 4: "kin",  5: "gin",  6: "keima",  7: "kyousha",
#  8: "hu"}　#駒の種類を特定 （自分は王、相手は玉）
# 
# way = {"ou": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]],  
#        "gyoku": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]],  
#        "hisha": [[-1,  0],  [0,  1],  [1,  0],  [0,  -1]],  
#        "kaku": [[-1,  1],  [1,  1],  [1,  -1],  [-1,  -1]],  
#        "kin": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],  
#        "gin": [[-1,  1],  [0,  1],  [1,  1],  [1,  -1],  [-1,  -1]],  
#        "keima": [[-1,  2],  [1,  2]],  
#        "kyousha": [[0,  1]],  
#        "hu": [[0,  1]],  
#        "ryu": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]],  
#        "uma": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]], 
#       } #各駒の進む方向
# 
# value_k = {"ou": [8,  8],  "gyoku": [8,  8],  "hisha": [12,  16],  "kaku": [12,  16], "kin": [6,  6],
#  "gin": [5,  6],  "keima": [6,  6],
# "kyousha": [3,  6],  "hu": [1,  6]} #各駒の評価係数
# value_p = {"first": 1,  "second": 2,  "third": 3} #各列の評価係数（相手陣地に近いほど高い）
# value_d = {"first": 2,  "second": 1} #王との距離の評価係数
# end_f = 0 #対局終了フラグ
# player = [0,  2,  4] #各プレーヤの持ち駒
# computer = [3,  1,  6]
# 
# stage = {0: [1,  0],  2: [0,  1],  4: [2,  0],  3: [0,  2],  1: [1,  2],  6: [2,  2]} #初期配置
# 

import numpy as np


kind = {0:  "ou",  1:  "gyoku",  2:  "hisha",  3:  "kaku", 4:  "kin",  5:  "gin",  6: "keima",  7: "kyousha",  8: "hu",
        9: "ryu", 10: "uma", 11: "narigin", 12: "narikei", 13: "narikyou", 14: "to"
        }

way = {"ou": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]],  
       "gyoku": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]],
       "hisha": [[-i, 0] for i in range(1, 3)] +
                [[i, 0] for i in range(1, 3)] +
                [[0, i] for i in range(1, 3)] +
                [[0, -i] for i in range(1, 3)],
       # "hisha": [unlmmove1,  [0,  1],  [1,  0],  [0,  -1]],  # なんかunlmmoveでも動いてるけど
       "kaku": [[-i, i] for i in range(1, 3)] +
                [[i, i] for i in range(1, 3)] +
                [[i, -i] for i in range(1, 3)] +
                [[-i, -i] for i in range(1, 3)],
       # "kaku": [[-1,  1],  [1,  1],  [1,  -1],  [-1,  -1]],
       "kin": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],  
       "gin": [[-1,  1],  [0,  1],  [1,  1],  [1,  -1],  [-1,  -1]],  
       "keima": [[-1,  2],  [1,  2]],  
       "kyousha": [[0,  1]],  
       "hu": [[0,  1]],  
       "ryu": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]].extend(
                [[-i, 0] for i in range(1, 3)] +
                [[i, 0] for i in range(1, 3)] +
                [[0, i] for i in range(1, 3)] +
                [[0, -i] for i in range(1, 3)]
       ),
       # 被ってる移動先残ってる
       "uma": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [1,  -1],  [0,  -1],  [-1,  -1],  [-1,  0]].extend(
                [[-i, i] for i in range(1, 3)] +
                [[i, i] for i in range(1, 3)] +
                [[i, -i] for i in range(1, 3)] +
                [[-i, -i] for i in range(1, 3)]
       ),
       # 被ってる移動先残ってる。
       "narigin": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],
       "narikei": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],
       "narikyou": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],
       "to": [[-1,  1],  [0,  1],  [1,  1],  [1,  0],  [0,  -1],  [-1,  0]],
       }

value_k = {"ou": [8,  8],  "gyoku": [8,  8],  "hisha": [12,  16],  "kaku": [12,  16], "kin": [6,  6],  "gin": [5,  6],
           "keima": [6,  6],  "kyousha": [3,  6],  "hu": [1,  6], "ryu": [16], "uma": [16], "narigin": [6],
           "narikei": [6], "narikyou": [6], "to": [6],
           }
value_l = {"first": 1,  "second": 2,  "third": 3} 
value_d = {"first": 2,  "second": 1}
end_f = 0

# player = [0,  2,  4]  # 各プレーヤ所属の駒
# computer = [3,  1,  6]
# stage = {0: [1,  0],  2: [0,  1],  4: [2,  0],  3: [0,  2],  1: [1,  2],  6: [2,  2]}
# 初期盤面（自分が下）後で自分とcpの駒の区別をつけられるようにしなきゃだめ


# 1,  cp, pl共に使う共通の評価関数。


def rob(player,  nextpos,  stage):  # 相手プレーヤーの手持ちコマがplayerに入る
    enemyplaces = {}
    for i in player: 
        enemyplaces[i] = stage[i]
    for i in enemyplaces.keys(): 
        if enemyplaces[i] == nextpos: 
            return i  # なんか0の駒（王）の時pythonだと0 = Falseみたいな感じで処理されるから数値そのままだとダメくさい。
    return "false"  # 上のやつを変えるのめんどいからFalseの時文字列のfalseにした


def evaluation_r(k,  l,  d,  r):  # 点数の合計をだすよ～
    return value_k[kind[k]][0] + value_l[l] + value_d[d] + value_k[kind[r]][0]
    # まだ成ることは考えてないよ～（value_k[kind[0]][0]の０のとこ）


def evaluation_n(k,  l,  d):    # 点数の合計をだすよ～
    return value_k[kind[k]][0] + value_l[l] + value_d[d]  # まだ成ることは考えてないよ～（value_k[kind[0]][0]の０のとこ）


def lin(me):  # 列の点数を出すよ～
    if me[1] == 2: 
        return "third"
    elif me[1] == 1: 
        return "second"
    elif me[1] == 0: 
        return "first"


def dist(me,  king):  # 王までの距離の点数を出すよ～
    dis = np.sqrt((king[0] - me[0]) ** 2 + (king[1] - me[1]) ** 2)
    if dis < 2 :
        return "first"
    elif dis >= 2 and dis <= 4:
        return "second"


def movedstage(tomovestage,  choices): 
    stageset = []
    originstage = {}
    for i in range(len(choices)): 
        for k,  v in tomovestage.items(): 
            originstage[k] = v
        originstage[choices[i][1]] = choices[i][2]
        stageset.append([choices[i][0],  originstage])
        # print (originstage)
    return stageset


# 2,  コンピュータの手をシミュレーションする関数


def cp_simu(stage, player, computer):
    points = []
    for piece in computer:  # pieceを選択
        # print("*piece=",  piece)
        for w in way[kind[piece]]:  # wayを選択
            # print("way=",  w)
            reverse_w = list(map(lambda x:  x * -1,  w))
            position = np.array(stage[piece])  # 選んだ駒の現在の座標
            next_position = position + np.array(reverse_w)  # 進んだ先の座標
            l_next_position = next_position.tolist()  # numpyの配列と標準のリストの形が違うからinとかやってもダメ臭い=>ndarray.tolist()使おうや
            # print(list(map(lambda x: x >= 0 and x <= 2,  l_next_position)))
            if all(map(lambda x: x >= 0 and x <= 2, l_next_position)):
                # 進んだ先が盤面内に収まっているか
                king_position = stage[0]  # 自分は王だから王のポジションを特定する
                line = str(lin(next_position))  # 列の評価係数(function_lin)
                distance = str(dist(l_next_position, king_position))  # 距離の評価係数(function_dist)
                if l_next_position not in stage.values():  # 進んだ先に駒がなかった場合
                    # print("l_next_position=",  l_next_position)
                    # print("distance=",  distance)
                    point = evaluation_n(piece,  line,  distance)
                    # print("*point=",  point) #手の駒の能力×列×距離の点数(function_evaluation)
                    points.append([point,  piece,  l_next_position])  # 動かした駒、動かした結果の評価点数、動かした場合その駒の位置
                else:
                    # print("もうおるやんけ！")#進んだ先が相手の駒だった場合は点数加算＆駒の位置の得点も加算
                    robable_piece = rob(player,  l_next_position,  stage)  # 取れる駒の種類(function_rob)
                    if robable_piece != "false":
                        # 進んだ先の駒が敵駒だったばあい
                        # print("robable_piece=",  robable_piece)
                        # print("line=",  line)
                        # print("l_next_position=",  l_next_position)
                        # print("distance=",  distance)
                        point = evaluation_r(piece,  line,  distance,  robable_piece)
                        # print("*駒取った+point=",  point) #手の駒の能力+列+距離の点数+取った駒の能力(function_evaluation)
                        points.append([point,  piece,  l_next_position, "robable_piece=",  robable_piece])
                        # 動かした結果の評価点数、動かした駒、動かした場合その駒の位置、手に入れた駒
                    # else:
                        # print("味方やんけ")
            # else:
                # print("進めないよ")
    return points

"""
def cp_simu(stage, player, computer):
    points = []
    for piece in computer:  # pieceを選択
        # print("*piece=",  piece)
        for w in way[kind[piece]]:  # wayを選択
            # print("way=",  w)
            reverse_w = list(map(lambda x:  x * -1,  w))
            position = np.array(stage[piece])  # 選んだ駒の現在の座標
            next_position = position + np.array(reverse_w)  # 進んだ先の座標
            l_next_position = next_position.tolist()  # numpyの配列と標準のリストの形が違うからinとかやってもダメ臭い=>ndarray.tolist()使おうや
            # print(list(map(lambda x: x >= 0 and x <= 2,  l_next_position)))
            if all(map(lambda x: x >= 0 and x <= 2, l_next_position)):
                # 進んだ先が盤面内に収まっているか
                king_position = stage[0]  # 自分は王だから王のポジションを特定する
                line = str(lin(next_position))  # 列の評価係数(function_lin)
                distance = str(dist(l_next_position, king_position))  # 距離の評価係数(function_dist)
                if l_next_position not in stage.values():  # 進んだ先に駒がなかった場合
                    # print("l_next_position=",  l_next_position)
                    # print("distance=",  distance)
                    point = evaluation_n(piece,  line,  distance)
                    # print("*point=",  point) #手の駒の能力×列×距離の点数(function_evaluation)
                    points.append([point,  piece,  l_next_position])  # 動かした駒、動かした結果の評価点数、動かした場合その駒の位置
                else: 
                    # print("もうおるやんけ！")#進んだ先が相手の駒だった場合は点数加算＆駒の位置の得点も加算
                    robable_piece = rob(player,  l_next_position,  stage)  # 取れる駒の種類(function_rob)
                    if robable_piece != "false": 
                        # 進んだ先の駒が敵駒だったばあい
                        # print("robable_piece=",  robable_piece)
                        # print("line=",  line)
                        # print("l_next_position=",  l_next_position)
                        # print("distance=",  distance)
                        point = evaluation_r(piece,  line,  distance,  robable_piece)
                        # print("*駒取った+point=",  point) #手の駒の能力+列+距離の点数+取った駒の能力(function_evaluation)
                        points.append([point,  piece,  l_next_position, "robable_piece=",  robable_piece])
                        # 動かした結果の評価点数、動かした駒、動かした場合その駒の位置、手に入れた駒
                    # else:
                        # print("味方やんけ")
            # else:
                # print("進めないよ")
    return points
"""
# 3,  プレイヤーの手をシミュレートする関数


def pl_simu(prestage,  player, computer):
    points2 = []
    stage = prestage
    for piece in player:  # pieceを選択
        # print("*piece=",  piece)
        for w in way[kind[piece]]:  # wayを選択
            # print("way=",  w)
            position = np.array(stage[piece])  # 選んだ駒の現在の座標
            next_position = position + np.array(w)  # 進んだ先の座標
            l_next_position = next_position.tolist()  # numpyの配列と標準のリストの形が違うからinとかやってもダメ臭い=>ndarray.tolist()使おうや
            # print(list(map(lambda x: x >= 0 and x <= 2,  l_next_position)))
            if all(map(lambda x: x >= 0 and x <= 2,  l_next_position)):
                # 進んだ先が盤面内に収まっているか
                king_position = stage[1]  # コンピュータは玉だから玉のポジションを特定する
                line = str(lin(next_position))  # 列の評価係数(function_lin)
                # print("line=",  line)
                distance = str(dist(l_next_position, king_position))  # 距離の評価係数(function_dist)
                # 進んだ先が盤面内に収まっているか
                if l_next_position not in stage.values():  # 進んだ先に駒がなかった場合
                    king_position = stage[1]  # コンピュータは玉だから玉のポジションを特定する
                    line = str(lin(next_position))  # 列の評価係数(function_lin)
                    # print("line=",  line)
                    distance = str(dist(l_next_position,  king_position))  # 距離の評価係数(function_dist)
                    # print("l_next_position=",  l_next_position)
                    # print("distance=",  distance)
                    point = evaluation_n(piece,  line,  distance)
                    # print("*point=",  point) #手の駒の能力×列×距離の点数(function_evaluation)
                    points2.append([point,  piece,  l_next_position])  # 動かした駒、動かした結果の評価点数、動かした場合その駒の位置
                else: 
                    # print("もうおるやんけ！")#進んだ先が相手の駒だった場合は点数加算＆駒の位置の得点も加算
                    robable_piece = rob(computer,  l_next_position,  stage)  # 取れる駒の種類(function_rob)
                    if robable_piece != "false": 
                        # 進んだ先の駒が敵駒だったばあい
                        # print("robable_piece=",  robable_piece)
                        # print("line=",  line)
                        # print("l_next_position=",  l_next_position)
                        # print("distance=",  distance)
                        point = evaluation_r(piece,  line,  distance,  robable_piece)
                        # print("*駒取った+point=",  point) #手の駒の能力+列+距離の点数+取った駒の能力(function_evaluation)
                        points2.append([point,  piece,  l_next_position, "robable_piece=",  robable_piece])
                        # 動かした結果の評価点数、動かした駒、動かした場合その駒の位置、手に入れた駒
                    # else:
                        # print("味方やんけ")
            # else:
                # print("進めないよ")
    # print("points2=",  points2)
    return points2


# 4,  コンピュータの手とその次のプレーヤーの手の評価をして、コンピュータはどの手を出せばいいのかを選ぶ関数。


def selectway(stg, plr, cp):
    cpways = movedstage(stg,  cp_simu(stg, plr, cp))
    # print("cpways=",  cpways)
    finaly_points = []
    for i in cpways: 
        plways = pl_simu(i[1],  plr, cp)
        finaly_points.append(i[0] - max(plways)[0])
        # print("plways=",  plways)
        # print("finaly_points=",  finaly_points)
    return([cp_simu(stg, plr, cp)[finaly_points.index(max(finaly_points))][1],   # 動かす駒と動かす先
           cp_simu(stg, plr, cp)[finaly_points.index(max(finaly_points))][2]]) 


if __name__ == "__main__":
    print(selectway({0: [1,  0],  7: [0,  1],  4: [2,  0],  3: [0,  2],  1: [1,  2],  6: [2,  2]},
                    [0, 7, 4], [6, 3, 1]))




