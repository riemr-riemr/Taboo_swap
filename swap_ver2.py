import numpy as np
import random
import sys
import math
import time
import copy
import csv
import matplotlib.pyplot as plt

def distance(x1, x2, y1, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d

def hyojun(route): #標準偏差の計算, disに各車両の移動距離格納
    dis = []
    for i in range(len(route)):
        sum_k = 0
        if len(route[i]) != 0:
            for j in range(len(route[i]) - 1):
                sum_k += c[route[i][j]][route[i][j + 1]]
            sum_k += c[0][route[i][0]]
            sum_k += c[0][route[i][j + 1]]
        dis.append(sum_k)
    return np.std(dis)*40

def Setting(FILENAME):
    mat = []             # 距離計算用のdepo+出発地+目的地
    with open('/home/rei/ドキュメント/benchmark/' + FILENAME, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            row = []
            toks = line.split()
            for tok in toks:
                try:
                    num = float(tok)
                except ValueError:
                    continue
                row.append(num)
            mat.append(row)

    # インスタンスの複数の行（問題設定）を取り出す
    Setting_Info = mat.pop(0)  # 0:車両数、4:キャパシティ、8:一台あたりの最大移動時間(min)、9:一人あたりの最大移動時間(min)

    # デポの座標を取り出す
    depo_zahyo = np.zeros(2)  # デポ座標配列
    x = mat.pop(-1)
    depo_zahyo[0] = x[1]
    depo_zahyo[1] = x[2]


    # 各距離の計算
    c = np.zeros((len(mat), len(mat)), dtype=float, order='C')
    for i in range(len(mat)):
        for j in range(len(mat)):
            c[i][j] = distance(mat[i][1], mat[j][1], mat[i][2], mat[j][2])

    request_number = len(mat) - 1

    #乗り降りの0-1情報を格納
    noriori = np.zeros(len(mat), dtype=int, order='C')
    for i in range(len(mat)):
        noriori[i] = mat[i][4]

    return Setting_Info, request_number, depo_zahyo, c, noriori


def initial_sulution(request_node, vehicle_number):  #初期解生成
    riyoukyaku_number = np.arange(1, request_node / 2 + 1)
    route = [[] * 1 for i in range(vehicle_number)]
    i = 0
    while True:
        if riyoukyaku_number.size == 0:
            break
        if i > vehicle_number - 1:
            i = 0
        a = int(np.random.choice(riyoukyaku_number, 1))
        route[i].append(a)
        b = a + int(request_node / 2)
        route[i].append(b)
        riyoukyaku_number = np.delete(riyoukyaku_number, np.where(riyoukyaku_number == a))
        i = i + 1

    return route

def initial_solution(request_number, vehicle_number, passe_max):  # グリーディ初期化
    route = [[]*1 for i in range(vehicle_number)]
    bunpai = int((request_number/2)/vehicle_number) + 2
    c_copy = copy.deepcopy(c)
    riding_id = []
    passe = 0
    for i in range(request_number + 1):
        c_copy[i][i] = 1000
    for i in range(vehicle_number):
        status = 0
        riyoukyaku = 0
        a_route = []
        count = 0
        while True:
            next_passe_dis = 1000  # 次の乗客の場所までの距離
            next_dis_dis = 1000 # 次の乗客の目的地までの距離
            if len(riding_id) != 0:
                for j in range(len(riding_id)):  # 乗車中の顧客の最短目的地
                    if next_dis_dis >= c_copy[status][riding_id[j] + int(request_number/2)]:
                        next_dis_dis = c_copy[status][riding_id[j] + int(request_number/2)]
                        arrive_passe_id = riding_id[j] + int(request_number/2) # 次に降車する顧客ID

            for j in range(int(request_number/2 + 1)):  # 最近の客を探す
                if next_passe_dis >= c_copy[status][j] and c_copy[0][j] < 1000:
                    next_passe_dis = c_copy[status][j]
                    ride_passe_id = j  # 次に乗車する顧客ID

            if not count == 0:
                for j in range(request_number + 1):
                    c_copy[j][status] = 1000
                    c_copy[status][j] = 1000

            if (riyoukyaku == bunpai or passe == int(request_number/2)) and len(riding_id) == 0:
                break


            if next_dis_dis < next_passe_dis or len(riding_id) == passe_max or  riyoukyaku == bunpai or passe == int(request_number/2):  # 目的地へ向かう場合
                riding_id.remove(arrive_passe_id - int(request_number/2))  # 乗車中顧客番号から消す
                c_copy[status][arrive_passe_id] = 1000
                c_copy[arrive_passe_id][status] = 1000
                status = arrive_passe_id  # 降車した顧客IDから現在地のID取得
                a_route.append(status)

            else:  # 新たな顧客を乗せる場合
                riding_id.append(ride_passe_id)  # 乗車中顧客番号を追加
                c_copy[status][ride_passe_id] = 1000
                c_copy[ride_passe_id][status] = 1000
                status = ride_passe_id  # 現在地を移動
                a_route.append(status)  #route register
                riyoukyaku += 1
                passe += 1

            count += 1


        route[i] = a_route

    return route



def Route_cost(route):  #総移動距離の計算
    Route_sum = 0
    Route_sum_k = np.zeros(len(route), dtype=float, order='C')
    for i in range(len(route)):
        if len(route[i]) == 0:
            Route_sum_k[i] = 0
        else:
            for j in range(len(route[i]) - 1):
                Route_sum_k[i] = Route_sum_k[i] + c[route[i][j]][route[i][j + 1]]
            Route_sum_k[i] = Route_sum_k[i] + c[0][route[i][0]]
            Route_sum_k[i] = Route_sum_k[i] + c[0][route[i][j + 1]]
        Route_sum = Route_sum + Route_sum_k[i]

    return Route_sum


def route_k_cost_sum(route_k):  #1台ずつの移動距離計算
    route_k_sum = 0
    for i in range(len(route_k) - 1):
        route_k_sum = route_k_sum + c[route_k[i]][route_k[i + 1]]
    if len(route_k) != 0:
        route_k_sum = route_k_sum + c[0][route_k[0]]
        route_k_sum = route_k_sum + c[0][route_k[i + 1]]

    return route_k_sum


def capacity(route):  #全車両のキャパオーバー回数
    q = np.zeros(int(m), dtype=int, order='C')
    capacity_over = 0
    for i in range(len(route)):
        for j in range(len(route[i])):
            q[i] = q[i] + noriori[route[i][j]]
            if q[i] > Q_max:
                capacity_over += 1
    return capacity_over


def capacity_route_k(route_k):  #1台ずつのキャパオーバー回数
    capacity_over = 0
    q = 0
    for i in range(len(route_k)):
        q = q + noriori[route_k[i]]
        if q > Q_max:
            capacity_over += 1
    return capacity_over


def time_caluculation(Route_k, request_node):  #ノードに到着する時間、出発する時間、顧客の乗車時間
    A = np.zeros(n + 2, dtype=float, order='C')  # ノード到着時間
    D = np.zeros(n + 2, dtype=float, order='C')  # ノード出発時間
    L = np.zeros(int(request_node / 2), dtype=float, order='C')  # リクエストiの乗車時間
    if not len(Route_k) == 0:
        for i in range(len(Route_k)):
            if i == 0:
                A[Route_k[i]] = D[i] + c[i][Route_k[i]]
                D[Route_k[i]] = A[Route_k[i]] + d
            else:
                A[Route_k[i]] = D[Route_k[i - 1]] + c[Route_k[i - 1]][Route_k[i]]
                D[Route_k[i]] = A[Route_k[i]] + d
        A[-1] = D[Route_k[-1]] + c[0][Route_k[-1]]
        for i in range(len(Route_k)):
            if Route_k[i] <= request_node / 2:
                L[Route_k[i] - 1] = A[Route_k[i] + int(request_node / 2)] - D[Route_k[i]]
    return A, D, L


def ride_time_penalty(L):  # 論文でのt_s
    sum = 0
    for i in range(len(L)):
        a = L[i] - L_max
        if a > 0:
            sum = sum + a
    return sum


'''
enumerate関数を使っている
for j,row in enumerate(Route)
jが車両番号、rowは車両jの顧客リストを取得
中のforループで近傍探索で変更するランダムで選ばれた顧客のindex値（あるいはそのものの値）を取得できるまで回している
'''


def neighbourhood(route, requestnode):  #近傍探索ランダムver.
    mm = np.arange(len(route))
    i = random.randint(1, requestnode / 2)
    for j, row in enumerate(route):
        try:
            u_before = [i, j]
            i_index = row.index(i)
            break
        except ValueError:
            pass
    u_before = np.array(u_before)  # 車両変更前 U = [顧客番号、車両番号]
    k_new = int(np.random.choice(mm[mm != u_before[1]], size=1))
    u_after = np.array([u_before[0], k_new])  # 車両変更後 U = [顧客番号、新たな車両番号]
    neighbour = np.append(u_before, u_after, axis=0).reshape(2, 2)
    return neighbour


def newRoute(route, requestnode, neighbour):  #
    new_route = copy.deepcopy(route)
    for j in range(len(route)):
        try:
            new_route[j].remove(neighbour[0][0])
            new_route[j].remove(neighbour[0][0] + requestnode / 2)
            break
        except ValueError:
            pass
    new_route = insert_route(new_route, requestnode, neighbour)
    return new_route


def penalty_sum(route, requestnode): #係数ありの評価関数、係数、係数なしの評価関数を返す(全体)
    parameta = np.zeros(4)
    c_s = Route_cost(route)
    q_s = capacity(route)  #キャパオーバー
    d_s = 0                #1台あたりの移動時間違反
    t_s = 0                #1人あたりの移動時間違反
    h_s = hyojun(route)
    for i in range(len(route)):
        ROUTE_TIME_info = time_caluculation(route[i], requestnode)
        d_s_s = ROUTE_TIME_info[0][-1] - T_max
        if d_s_s < 0:
            d_s_s = 0
        d_s = d_s + d_s_s
        t_s = t_s + ride_time_penalty(ROUTE_TIME_info[2])

    penalty = keisu[0] * q_s + 50 * d_s + 20 * t_s + h_s
    no_penalty = c_s + q_s + 50*d_s + 20*t_s + h_s
    parameta[0] = q_s
    parameta[1] = d_s
    parameta[2] = t_s
    parameta[3] = h_s
    return penalty, parameta, no_penalty


def penalty_sum_route_k(route_k, requestnode):  #係数ありの評価関数(1台あたり)
    c_s = route_k_cost_sum(route_k)
    q_s = capacity_route_k(route_k)
    d_s = 0
    t_s = 0
    ROUTE_TIME_info = time_caluculation(route_k, requestnode)
    d_s_s = ROUTE_TIME_info[0][-1] - T_max
    if d_s_s < 0:
        d_s_s = 0
    d_s += d_s_s
    t_s += ride_time_penalty(ROUTE_TIME_info[2])

    penalty = c_s + keisu[0] * q_s + 50 * d_s + 20 * t_s
    return penalty


def insert_route(route, requestnode, neighbor):
    new_route_k = copy.deepcopy(route[neighbor[1][1]])
    insert_number = neighbor[0][0]
    route_k_node = len(route[neighbor[1][1]])

    new_route_k.insert(0, insert_number)
    new_route_k.insert(1, insert_number + int(requestnode / 2))
    penalty = penalty_sum_route_k(new_route_k, requestnode)
    check_route = copy.deepcopy(route[neighbor[1][1]])
    for i in range(route_k_node):
        j = i + 1
        while j <= 4:
            check_route = copy.deepcopy(route[neighbor[1][1]])
            check_route.insert(i, insert_number)
            check_route.insert(j, int(insert_number + requestnode / 2))
            check_penalty = penalty_sum_route_k(check_route, requestnode)
            if check_penalty < penalty:
                penalty = check_penalty
                new_route_k = copy.deepcopy(check_route)
            j = j + 1
        if j == route_k_node+1:
            check_route = copy.deepcopy(route[neighbor[1][1]])
            check_route.insert(i, insert_number)
            check_route.append(int(insert_number + requestnode / 2))
            check_penalty = penalty_sum_route_k(check_route, requestnode)
            if check_penalty < penalty:
                penalty = check_penalty
                new_route_k = copy.deepcopy(check_route)
    check_route = copy.deepcopy(route[neighbor[1][1]])
    check_route.append(insert_number)
    check_route.append(int(insert_number + requestnode / 2))
    check_penalty = penalty_sum_route_k(check_route, requestnode)
    if check_penalty < penalty:
        penalty = check_penalty
        new_route_k = copy.deepcopy(check_route)
    check_route = copy.deepcopy(route)
    new_route = copy.deepcopy(route)
    new_route[neighbor[1][1]] = copy.deepcopy(new_route_k)
    return new_route


#〇〇ver2・・・neighborを使わずに新たなルートを作る
#引数(現在のルート、depoを除いたノード、入れ替える顧客番号、入れ替える)
def newRoute_ver2(route,requestnode,riyoukyakunumber,vehiclenumber,new_vehiclenumber):
    new_route = copy.deepcopy(route)
    old_vehicle = vehiclenumber
    new_route[vehiclenumber].remove(riyoukyakunumber)
    new_route[vehiclenumber].remove(riyoukyakunumber+requestnode/2)
    new_route = insert_route_ver2(new_route,requestnode,riyoukyakunumber,new_vehiclenumber)
    return new_route

def insert_route_ver2(route,requestnode,riyoukyakunumber,new_vehiclenumber):  #全探索ver.近傍探索 個別を考える
    new_route_k =copy.deepcopy(route[new_vehiclenumber])
    route_k_node = len(route[new_vehiclenumber])
    riyoukyakunumber = int(riyoukyakunumber)
    new_route_k.insert(0, riyoukyakunumber)
    new_route_k.insert(1, riyoukyakunumber + int(requestnode / 2))
    penalty = penalty_sum_route_k(new_route_k, requestnode)
    check_route = copy.deepcopy(route[new_vehiclenumber])
    for i in range(route_k_node):
        j = i + 1
        while j <= route_k_node:
            check_route = copy.deepcopy(route[new_vehiclenumber])
            check_route.insert(i, riyoukyakunumber)
            check_route.insert(j, int(riyoukyakunumber + requestnode / 2))
            check_penalty = penalty_sum_route_k(check_route, requestnode)
            if check_penalty < penalty:
                penalty = check_penalty
                new_route_k = copy.deepcopy(check_route)
            j = j + 1
        if j == route_k_node + 1:
            check_route = copy.deepcopy(route[new_vehiclenumber])
            check_route.insert(i, riyoukyakunumber)
            check_route.append(int(riyoukyakunumber + requestnode / 2))
            check_penalty = penalty_sum_route_k(check_route, requestnode)
            if check_penalty < penalty:
                penalty = check_penalty
                new_route_k = copy.deepcopy(check_route)
    check_route = copy.deepcopy(route[new_vehiclenumber])
    check_route.append(riyoukyakunumber)
    check_route.append(int(riyoukyakunumber + requestnode / 2))
    check_penalty = penalty_sum_route_k(check_route, requestnode)
    if check_penalty < penalty:
        penalty = check_penalty
        new_route_k = copy.deepcopy(check_route)
    new_route = copy.deepcopy(route)
    new_route[new_vehiclenumber] = copy.deepcopy(new_route_k)
    return new_route

def keisu_update(delta, parameta):
    for i in range(len(parameta)):
        if parameta[i] > 0:
            keisu[i] = keisu[i] * (1 + delta)
        else:
            keisu[i] = keisu[i] / (1 + delta)


def tabu_update(theta, tabu_list, neighbour):
    for i in range(math.ceil(theta)):
        if tabu_list[i][2] == -1:
            tabu_list[i][0] = neighbour[0][0]
            tabu_list[i][1] = neighbour[0][1]
            tabu_list[i][2] = math.ceil(theta) + 1
            break
    for i in range(math.ceil(theta)):
        if tabu_list[i][2] >= 0:
            tabu_list[i][2] = tabu_list[i][2] - 1


def syutyu(route, requestnode):  #車両内の順番最適化
    newroute = copy.deepcopy(route)
    loop = 1
    for i in range(len(route)):
        for j in range(requestnode):
            try:
                newroute[i].remove(loop)
                newroute[i].remove(loop + requestnode / 2)
                break
            except ValueError:
                pass
            newroute = insert_route_k(newroute, i, j, requestnode)
            loop += 1
    return newroute


def insert_route_k(route, veichle, number, requestnode):
    new_route_k = copy.deepcopy(route[veichle])
    insert_number = number
    route_k_node = len(route[veichle])

    new_route_k.insert(0, insert_number)
    new_route_k.insert(1, insert_number + int(requestnode / 2))
    penalty = penalty_sum_route_k(new_route_k, requestnode)
    check_route = copy.deepcopy(route[veichle])
    for i in range(route_k_node):
        j = i + 1
        while j <= 4:
            check_route = copy.deepcopy(route[veichle])
            check_route.insert(i, insert_number)
            check_route.insert(j, int(insert_number + requestnode / 2))
            check_penalty = penalty_sum_route_k(check_route, requestnode)
            if check_penalty < penalty:
                penalty = check_penalty
                new_route_k = copy.deepcopy(check_route)
            j = j + 1
    new_route = copy.deepcopy(route)
    new_route[veichle] = copy.deepcopy(new_route_k)
    return new_route

def syaryo_tokutei(route, riyoukyakunumber):  #ある顧客がどの車両に乗っているか
    for j, row in enumerate(route):
        try:
            u_before = [riyoukyakunumber, j]
            i_index = row.index(riyoukyakunumber)
            break
        except ValueError:
            pass
    u_before = np.array(u_before)  # 車両変更前 U = [顧客番号、車両番号]
    return int(u_before[1])

def tabu_check(riyoukyakunumber,vehiclenumber,tabu_list):
    check = 0
    for i in range(len(tabu_list)):  # ループ回数をtabu_listのサイズにあわせなければならない
        if tabu_list[i][0] == riyoukyakunumber and tabu_list[i][1] == vehiclenumber and tabu_list[i][
            2] >= 0:  # たぶん間違い
            check += 1
    return  check

def tabu_update_ver2(kinsi,tabu_list,neighbour):    #kinsiはその近傍を何回禁止にするか
    for i in range(len(tabu_list)):
        if tabu_list[i][2] == -1:
            tabu_list[i][0] = neighbour[0]
            tabu_list[i][1] = neighbour[1]
            tabu_list[i][2] = kinsi + 1
            break
    for i in range(len(tabu_list)):
        if tabu_list[i][2] >= 0:
            tabu_list[i][2] = tabu_list[i][2] - 1

def swap(route, requestnode):
    max = 0
    min = 1000
    for i in range(len(route)):  # 初期解
        if max <= penalty_sum_route_k(route[i], requestnode):
            max = penalty_sum_route_k(route[i], requestnode)
            max_no = i
        if min >= penalty_sum_route_k(route[i], requestnode):
            min = penalty_sum_route_k(route[i], requestnode)
            min_no = i

    for i in range(len(route[max_no])):
        if route[max_no][i] <= int(requestnode / 2):
            NEWroute = copy.deepcopy(newRoute_ver2(route, requestnode, route[max_no][i], max_no, min_no))
            if i == 0:
                change = copy.deepcopy(NEWroute)
            elif penalty_sum(NEWroute, requestnode)[2] < penalty_sum(change, requestnode)[2]:
                change = copy.deepcopy(NEWroute)
        for j in range(len(route[min_no])):
            if route[min_no][j] <= int(requestnode/2):
                NEWroute = copy.deepcopy(newRoute_ver2(change, requestnode, route[min_no][j], min_no, max_no))
                if j == 0:
                    change_saiteki = copy.deepcopy(NEWroute)
                elif penalty_sum(NEWroute, requestnode)[2] < penalty_sum(change_saiteki, requestnode)[2]:
                        change_saiteki = copy.deepcopy(NEWroute)

    return change_saiteki


def main(LOOP):
    data = np.zeros(LOOP)
    initial_Route = initial_solution(n, m, Q_max)  # 初期解生成
    syoki = copy.deepcopy(initial_Route)
    opt = penalty_sum(initial_Route, n)[2]
    test = penalty_sum(initial_Route, n)[2]
    loop = 0  # メインのループ回数
    parameta_loop = 0  # パラメーター調整と集中化のループ回数(ループ回数は10回)
    delta = 0.5
    theta = int(n/2*(m-1))  # タブーリストの大きさ
    kinsi = theta
    tabu_list = np.zeros((theta, 3)) - 1
    kinbo_cost = float('inf')
    syutyu_loop = 0
    while True: #近傍探索　
        best_neighbour = np.zeros(2)
        riyoukyaku_list = np.arange(1,n/2+1)
        for i in riyoukyaku_list:
            syaryo_loop = np.arange(m)
            syaryo_loop = np.delete(syaryo_loop,int(syaryo_tokutei(initial_Route,i)))
            for j in syaryo_loop:
                skip = 0
                check = tabu_check(i,j,tabu_list)
                if not check > 0:
                    old_vehiclenumber = syaryo_tokutei(initial_Route,i)
                    NewRoute = copy.deepcopy(newRoute_ver2(initial_Route,n,i,old_vehiclenumber,j))
                else:
                    continue
                if penalty_sum(NewRoute, n)[2] < kinbo_cost:
                    best_neighbour[0] = i
                    best_neighbour[1] = old_vehiclenumber
                    NextRoute = copy.deepcopy(NewRoute)
                    kinbo_cost = penalty_sum(NextRoute, n)[2]
                    skip = 1
                    break
            if skip == 1:
                break

        if kinbo_cost <= opt:
            opt = kinbo_cost
            saiteki_route = copy.deepcopy(NextRoute)
            saiteki = penalty_sum(saiteki_route, n)[2]

        tabu_update_ver2(kinsi, tabu_list, best_neighbour)
        kinbo_cost = float('inf')

        initial_Route = copy.deepcopy(NextRoute)

        parameta_loop += 1
        if parameta_loop == 100:
            delta = np.random.uniform(0, 0.5)
            parameta_loop = 0

        data[loop] = opt
        loop += 1
        swap_route = swap(saiteki_route, n)
        if penalty_sum(swap_route, n)[2] <= penalty_sum(saiteki_route, n)[2]:
            saiteki_route = copy.deepcopy(swap_route)
            saiteki = penalty_sum(swap_route, n)[2]
        if loop == LOOP:
            break


    print(syoki)
    print(saiteki_route)
    print(test, opt)
    print(saiteki)
    print(saiteki - penalty_sum(saiteki_route, n)[0])
    print(penalty_sum(saiteki_route, n)[1])
    print(keisu)
    print(FILENAME)
    np.savetxt('/home/rei/ドキュメント/data_darp02/swap_ver2.ods', data, delimiter=",")


if __name__ == '__main__':
    FILENAME = 'darp02.txt'
    Setting_Info = Setting(FILENAME)[0]

    n = int(Setting(FILENAME)[1])  # depoを除いたノード数
    m = int(Setting_Info[0])  # 車両数
    d = 5  # 乗り降りの時間
    Q_max = Setting_Info[4]  # 車両の最大容量 global変数 capacity関数で使用
    T_max = Setting_Info[8]  # 一台当たりの最大移動時間
    L_max = Setting_Info[9]  # 一人あたりの最大移動時間

    noriori = np.zeros(n + 1, dtype=int, order='C')
    noriori = Setting(FILENAME)[4]  # global変数  capacity関数で使用

    depo_zahyo = Setting(FILENAME)[2]  # デポの座標

    c = np.zeros((n + 1, n + 1), dtype=float, order='C')
    c = Setting(FILENAME)[3]  # 各ノード間のコスト

    keisu = np.ones(4)
    t1 = time.time()
    main(500)
    t2 = time.time()
    print(f"time:{t2 - t1}")
