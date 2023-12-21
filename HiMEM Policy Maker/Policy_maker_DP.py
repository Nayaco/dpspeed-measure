"""
HiMEM Policy Maker.
"""
import array
import copy
import math
from bayes_opt import BayesianOptimization
class Computation_Operator:
    def __init__(self, _id, _output_size, _compute_time):
        self.id = _id
        self.output_size = _output_size
        self.compute_time = _compute_time
        self.livetime = 0
        self.first_use_time = 0
        self.last_use_time = 0
        self.policy = -1 # 0存在GPU内；1转移；2转移+压缩；3重计算；4压缩；
        self.type = 0 # 0为计算算子
        self.transformer_block_id = 0
        
class Communication_Operator:
    def __init__(self, _id, _communication_time):
        self.id = _id
        self.compute_time = _communication_time
        self.type = 1 # 1为通信算子
        self.transformer_block_id = -1
        self.first_use_time = 0
        self.last_use_time = -1
        self.livetime = -1

Transformer_Block = 2 # 

op_array = [] # 所有操作的数组
transformer_op_array = [] # Transformer块的数组
transformer_op_array_forward = [] # Transformer块前传的数组
transformer_op_array_backward = [] # Transformer块反传的数组

op0 = Computation_Operator(0, 0, 4) # init
transformer_op_array.append(op0)
## ------------------------------------------ Forward
op1 = Computation_Operator(1, 28, 0.067) # FusedLayerNormAffineFunction
transformer_op_array_forward.append(op1)
op2 = Computation_Operator(2, 42, 0.49) # QKV_Linear
transformer_op_array_forward.append(op2)
op3 = Computation_Operator(3, 128, 0.377) # 合并Self-Attention-Q*K + Softmax
transformer_op_array_forward.append(op3)
op4 = Computation_Operator(4, 192, 0.286) # Dropout
transformer_op_array_forward.append(op4)
op5 = Computation_Operator(5, 14, 0.191) # Self-Attention-Q*K*V
transformer_op_array_forward.append(op5)
op6 = Computation_Operator(6, 0, 0.146) # Attention_Linear
transformer_op_array_forward.append(op6)
op7 = Communication_Operator(7, 2.2) # Attention_forward_AllReduce
transformer_op_array_forward.append(op7)
op8 = Computation_Operator(8, 42, 0.14) # Attention_Dropout
transformer_op_array_forward.append(op8)
op9 = Computation_Operator(9, 28, 0.067) # FusedLayerNormAffineFunction
transformer_op_array_forward.append(op9)
op10 = Computation_Operator(10, 56, 0.512) # MLP_Linear_1
transformer_op_array_forward.append(op10)
op11 = Computation_Operator(11, 56, 0.428) # GeLU_1
transformer_op_array_forward.append(op11)
op12 = Computation_Operator(12, 0, 0.483) # MLP_Linear_2
transformer_op_array_forward.append(op12)
op13 = Communication_Operator(13, 2.2) # Attention_forward_AllReduce
transformer_op_array_forward.append(op13)
op14 = Computation_Operator(14, 42, 0.075) # Dropout
transformer_op_array_forward.append(op14)

## ------------------------------------------ Backward
op15 = Computation_Operator(15, 0, 0.053) # Dropout_backward
transformer_op_array_backward.append(op15)
op16 = Computation_Operator(16, 0, 0.1) # MLP_Linear_2_back
transformer_op_array_backward.append(op16)
op17 = Computation_Operator(17, 0, 0.21) # GeLU
transformer_op_array_backward.append(op17)
op18 = Computation_Operator(18, 0, 0.442) # MLP_Linear_1_back
transformer_op_array_backward.append(op18)
op19 = Communication_Operator(19, 2.231) # MLP_backward_AllReduce
transformer_op_array_backward.append(op19)
op20 = Computation_Operator(20, 0, 0.222) # FusedLayerNormAffineFunctionBackward
transformer_op_array_backward.append(op20)
op21 = Computation_Operator(21, 0, 0.055) # Dropout_backward
transformer_op_array_backward.append(op21)
op22 = Computation_Operator(22, 0, 0.25) # Attention_Linear
transformer_op_array_backward.append(op22)
op23 = Computation_Operator(23, 0, 0.36) # Self-Attention-Q*K*V
transformer_op_array_backward.append(op23)
op24 = Computation_Operator(24, 0, 0.248) # Dropout_backward
transformer_op_array_backward.append(op24)
op25 = Computation_Operator(25, 0, 0.743) # 合并Softmax_backward + Self-Attention-Q*K
transformer_op_array_backward.append(op25)
op26 = Computation_Operator(26, 0, 0.34) # Linear_back
transformer_op_array_backward.append(op26)
op27 = Communication_Operator(27, 2.275) # Attention_backward_AllReduce
transformer_op_array_backward.append(op27)
op28 = Computation_Operator(28, 0, 0.236) # FusedLayerNormAffineFunctionBackward
transformer_op_array_backward.append(op28)



# Step1：组织训练数组，N个Transformer Block

begin_id = 1

# Forward
for i in range(Transformer_Block):
    tmp_transformer_op_array_forward = copy.deepcopy(transformer_op_array_forward)
    for op_id in range(len(tmp_transformer_op_array_forward)): 
        tmp_transformer_op_array_forward[op_id].transformer_block_id = i # 所属Transformer的block号
        tmp_transformer_op_array_forward[op_id].id = begin_id
        begin_id+=1
    transformer_op_array += tmp_transformer_op_array_forward

end_forward_id = begin_id
# Backward

for i in range(Transformer_Block):
    tmp_transformer_op_array_backward = copy.deepcopy(transformer_op_array_backward)
    for op_id in range(len(tmp_transformer_op_array_backward)): 
        tmp_transformer_op_array_backward[op_id].transformer_block_id = Transformer_Block-i-1
        tmp_transformer_op_array_backward[op_id].id = begin_id
        begin_id+=1
    transformer_op_array += tmp_transformer_op_array_backward


# Step2：建立开始时间

for op_id in range(len(transformer_op_array)):
    op = transformer_op_array[op_id]
    begin_time = 0
    for front_op in range(0, op_id):
        begin_time+=transformer_op_array[front_op].compute_time
        transformer_op_array[op_id].first_use_time = begin_time

# Step3：建立last use时间与生命周期
for op_id in range(1, end_forward_id):
    op_base_id = op_id - 14 * transformer_op_array[op_id].transformer_block_id
    # 寻找反传对应的id
    op_last_use_id = 14 * Transformer_Block + (Transformer_Block - 1 - transformer_op_array[op_id].transformer_block_id) * 14 + (15 - op_base_id)
    
    if transformer_op_array[op_id].type == 0:
        transformer_op_array[op_id].last_use_time = transformer_op_array[op_last_use_id].first_use_time
        live_time = transformer_op_array[op_id].last_use_time - transformer_op_array[op_id].first_use_time
        transformer_op_array[op_id].livetime = live_time # 赋予生命周期


# Step4：计算重计算+通信并行
# 使用01背包，引入新的评价指标决策出需要重计算的Tensor

recompute_candidate = [op2, op3, op4, op5, op6, op9, op10, op11, op12] # 重计算哪些算子，对应会生成哪些tensor

def dp_value(i,j, id, policy):
    if id-1 in policy[i][j]: # 前置数据已经在策略中，
        return transformer_op_array_forward[id-1].compute_time / 0.01 # 不需要保存ckpt数据，直接算即可
    else:#前置策略不在策略中
        # id对应的数据在id-1的位置
        return transformer_op_array_forward[id-1].compute_time / transformer_op_array_forward[id-2].output_size # ms/mb
        # 节省的计算时间 / 需要保存的常驻的数据开销。value越大越好。

def dp_search(pipeline_time): # 将时间均*10，便于搜索
    n = len(recompute_candidate) # n个算子
    bag_volume = int(pipeline_time*100) + 1 # 调整可并行的时间bag大小，均*100便于计算
    policy = [[[] for x in range(bag_volume)] for x in range(n + 1)]
    # print(policy, len(policy[0]))
    value = [[0 for x in range(bag_volume)] for x in range(n + 1)]
    for j in range(int(recompute_candidate[0].compute_time * 100), bag_volume):# 初始化
        value[0][j] = recompute_candidate[0].compute_time / transformer_op_array_forward[recompute_candidate[0].id-1].output_size

    for i in range(0, n):
        for w in range(0, bag_volume):
            cur_compute_time = int(recompute_candidate[i].compute_time * 100) #对当前算子的计算时间进行处理
            if w < cur_compute_time: #当前可并行部分的容量都没有当前算子i的计算时间大的时候，是不放算子i的
                value[i][w] = value[i-1][w] # 前i-1个算子能放下的最大价值就是当前情况的最大价值
                policy[i][w] = policy[i-1][w]
            else:
                put_in_value = value[i-1][w - cur_compute_time] + dp_value(i-1, w - cur_compute_time, recompute_candidate[i].id, policy)
                if value[i-1][w]>=put_in_value:
                    value[i][w] = value[i-1][w]
                    policy[i][w] = policy[i-1][w]
                else:
                    value[i][w] = put_in_value
                    policy[i][w] = policy[i-1][w - cur_compute_time]+ [recompute_candidate[i].id]
    return policy, value


def time_comput(policy):
    time = 0
    real_time = 0
    space_overhead = 0
    for i in policy:
        time+=int(transformer_op_array_forward[i-1].compute_time *100)
        real_time+=transformer_op_array_forward[i-1].compute_time *100
        space_overhead+=transformer_op_array_forward[i-2].output_size
    return time, real_time, space_overhead


package = 2
policy, value = dp_search(package)
# print(policy)

for i in range(int(package*100)+1):
    print("可并行的潜力：",i, "; 数据重量与空间代价：",time_comput(policy[len(recompute_candidate)-1][i]),"; 哪些数据重计算：" ,policy[len(recompute_candidate)-1][i],"; Value: ", value[len(recompute_candidate)-1][i])





# Step5：网格搜索策略
import itertools

# 优化目标为吞吐量最大，即计算时间最短

def grid_value_best_throughput(policy):
    recompute_time = 0
    recompute_array = [1,2,3,4,5,6,8,9,10,11,12,14]
    for i in recompute_array:
        recompute_time +=transformer_op_array_forward[i-1].compute_time
    


def grid_value(policy):
    compute_benifit = 0
    space_overhead = 0
    for i in policy:
        compute_benifit += transformer_op_array_forward[i-1].compute_time 
        if i-1 in policy:
            space_overhead += 0
        else:
            space_overhead += transformer_op_array_forward[i-2].output_size # ms/mb
    
    return compute_benifit/space_overhead, compute_benifit, space_overhead
    # for i in candidate:
    #     print(i)

# 计算重计算策略是否超过最大可并行时间
def grid_time(policy):
    totle_time = 0
    for i in policy:
        totle_time += transformer_op_array_forward[i-1].compute_time
    return totle_time


def grid_search(grid_recompute_candidate):
    best_value = -1
    best_candidate = []
    best_compute_benifit=0
    best_space_overhead=0
    for i in range(1,len(grid_recompute_candidate)+1):
        comb = list(itertools.combinations(grid_recompute_candidate,i))
        for policy in comb:
            if grid_time(policy) > package:
                continue
            value, compute_benifit, space_overhead = grid_value(policy)
            if value > best_value:
                best_value = value
                best_candidate = policy
                best_compute_benifit = compute_benifit
                best_space_overhead = space_overhead
    print("网格计算，重计算候策略：",best_candidate,"；可并行时间：", best_compute_benifit,"；额外的空间开销：", best_space_overhead)
    return best_candidate, best_compute_benifit, best_space_overhead





grid_recompute_candidate = [2,3,4,5,6,9,10,11,12]
best_candidate_1, compute_benifit_1, space_overhead_1  = grid_search(grid_recompute_candidate)

for remove_id in best_candidate_1:
    grid_recompute_candidate.remove(remove_id)
best_candidate_2, compute_benifit_2, space_overhead_2  = grid_search(grid_recompute_candidate)

recompute_time = 0
recompute_array = [1,2,3,4,5,6,8,9,10,11,12,14]
for i in recompute_array:
    recompute_time +=transformer_op_array_forward[i-1].compute_time
print("重计算总时间为：", recompute_time, "可并行时间为：", compute_benifit_1+compute_benifit_2, ",减少重计算开销:", (compute_benifit_1+compute_benifit_2)/recompute_time)