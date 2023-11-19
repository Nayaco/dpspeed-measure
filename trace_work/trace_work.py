"""
GPT-2 Model info, including calculation time and Tensor size.
"""

import readline
import numpy as np
import random
import math
import tqdm,numpy



Hot_Ratio = 0.001 # 热数据的比例
dataset = "/dataset/rec_data/NVRec_Dataset/Avazu/Avazu_dataset_reid.txt"


Avazu_item_num = 40428967 # Avazu数据行数 len(file.readlines())
every_day_num = Avazu_item_num / 10
total_embedding_num = 9449206 # Embedding个数
Popular_ID = int(9449206 * Hot_Ratio) # Popular ID


def get_frist_day_popular():
    SparseVisitarray = [0 for i in range(total_embedding_num)] # 创建数组用来记录每个Embedding的访问次数
    ID_access_map = dict() # 用dict记录每个ID下Embedding访问次数
    file = open(dataset,"r")
    file = file.readlines()
    cur_id = 0
    for train_set in (file):
        if(cur_id>every_day_num): # 只留下一天的量
            break
        content = train_set.split(' ')[2:23]
        for j in content:
            SparseVisitarray[int(j)] += 1
        cur_id+=1
    for i in range(total_embedding_num):
        ID_access_map[i] = SparseVisitarray[i]
    ID_access_map = sorted(ID_access_map.items(), key=lambda d:d[1], reverse = True)   # 根据访问次数排序

    data_array_flag = [0 for i in range(total_embedding_num)] # 标志位，辅助判断该id的数是否是热的
    for popular_cur in range(Popular_ID):
        popular_id = ID_access_map[popular_cur][0]
        data_array_flag[popular_id] = 1
    return data_array_flag



data_array_flag = get_frist_day_popular()
def get_hot_emb(begin, end, day):
    file = open(dataset,"r")
    file = file.readlines()
    All_access = 0
    Popular_access = 0
    cur_id = 0
    for train_set in file:
        if(cur_id>=begin and cur_id<end):
            content = train_set.split(' ')[2:23]
            for j in content:
                All_access+=1
                if data_array_flag[int(j)] == 1:
                    Popular_access+=1
        cur_id+=1
    print(day," day, The Ratio of Popular item is:", float(Popular_access) / float(All_access))

for i in range(10):
    get_hot_emb(every_day_num*i,every_day_num*(i+1), i+1)