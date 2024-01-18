"""
HiMEM Policy Maker.
"""
import array
import copy
import math
import search_algorithm
class Computation_Operator:
    def __init__(self, _id, _output_size, _compute_time):
        self.id = _id
        self.backward_id = -1
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

Transformer_Block = 32 # 
communication_time = 1 # 反传通信时间，ms
Memory_saving = 19200 #目标节省的显存值 - MB

op_array = [] # 所有操作的数组
transformer_op_array = [] # Transformer块的数组
transformer_op_array_forward = [] # Transformer块前传的数组
transformer_op_array_backward = [] # Transformer块反传的数组

op0 = Computation_Operator(0, 42, 1) # init
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



# Step1：初始化算子数组，N个Transformer Block

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

op_end = Computation_Operator(begin_id, 0, 1) # init
transformer_op_array.append(op_end)

# Step2：更新算子属性
import tools
transformer_op_array = tools.time_build(transformer_op_array, end_forward_id, Transformer_Block)

# Step3：grid搜索所有可并行的可能性
import itertools
grid_recompute_candidate = [1,2,3,4,5,9,10,11] # 可并行重计算的算子




# 生成所有并行重计算候选方案，网格搜索
def grid_search(grid_recompute_candidate):
    pipeline_policy = []
    for i in range(1,len(grid_recompute_candidate)+1):
        comb = list(itertools.combinations(grid_recompute_candidate,i))
        for policy in comb:
            policy_time, policy_space = tools.get_policy_time_space(policy,transformer_op_array_forward)
            if  policy_time > communication_time: # 剪枝
                continue
            pipeline_policy.append(policy)
    return pipeline_policy

pipeline_policy = grid_search(grid_recompute_candidate)

tensor_container_id_base = [1,2,3,4,5,6,8,9,10,11,12,14] 
tensor_container_id = [0] #所有产生Activation的算子id


for i in range(Transformer_Block): # 生成所有产生Activation的算子id
    for j in tensor_container_id_base:
        tensor_container_id.append(j + i*14)



policy_time, policy_space = tools.get_policy_time_space(tensor_container_id_base,transformer_op_array_forward)
print("总Activation空间大小 (MB)：",policy_space * Transformer_Block)
training_time = tools.get_policy_time(transformer_op_array)

min_policy_training_time = 1000000000
best_policy_count = -1
best_pipeline_tensor_id = []
best_swap_candidate_id = []
best_compression_tensor_id = []
best_stable_tensor_id = []

print("共有策略（个）：",len(pipeline_policy))
for count in range(len(pipeline_policy)):
    each_policy = pipeline_policy[count]
    policy_time, policy_space_ = tools.get_policy_time_space(each_policy,transformer_op_array_forward)
    print("Pipeline Policy time: ",policy_time)
    remaining_tensor_id = copy.deepcopy(tensor_container_id)#重计算后剩余的tensor
    pipeline_tensor_id = [] #使用可并行重计算的tensor
    stable_tensor_id = [] #固定在显存中的tensor
    for i in range(Transformer_Block):
        remaining_tensor_id.remove(6 + i*14) #剔除输出为0的tensor
        remaining_tensor_id.remove(12 + i*14)
        for j in each_policy:
            pipeline_id = j + i*14
            pipeline_tensor_id.append(pipeline_id)
            remaining_tensor_id.remove(pipeline_id) # 剩余非可并行策略的Tensor
            if (j-1) not in each_policy:
                stable_tensor_id.append((j-1) + i*14)
    print("========================== 搜索策略:",count,"==================================")
    training_time_after_policy,pipeline_tensor_id, swap_candidate_id, compression_tensor_id,compute_tensor,compression_tensor = search_algorithm.policy_generate(Memory_saving, transformer_op_array, pipeline_tensor_id, remaining_tensor_id, stable_tensor_id, end_forward_id, Transformer_Block, training_time)
    print("===========================================================================\n\n")

    if training_time_after_policy!=0:
        if training_time_after_policy<min_policy_training_time:
            min_policy_training_time = training_time_after_policy
            best_policy_count = count
            best_pipeline_tensor_id = pipeline_tensor_id
            best_swap_candidate_id = swap_candidate_id
            best_compression_tensor_id = compression_tensor_id
            best_stable_tensor_id = stable_tensor_id
    # exit()



print("模型训练时间 (ms)：",training_time)
print("Deepspeed重计算模型训练时间 (ms)：",training_time + Transformer_Block * tools.get_policy_time(transformer_op_array_forward))
print("总Activation空间大小 (MB)：",policy_space * Transformer_Block)
print("目标优化空间大小 (MB)：",Memory_saving,"；节约空间百分比 (%)", Memory_saving/(policy_space * Transformer_Block) * 100)
print("Best case: ",best_policy_count,"; training time: ",min_policy_training_time, "; Training time overhead: (%)", (min_policy_training_time-training_time)/training_time *100)
print("Pipeline Recomputation: ",best_pipeline_tensor_id)
print("Swap: ",best_swap_candidate_id)
print("Compression+Swap: ",best_compression_tensor_id)
print("Recomputation: ",compute_tensor)
print("Compression: ", compression_tensor)
for i in best_swap_candidate_id:
    if i in stable_tensor_id:
        stable_tensor_id.remove(i)
print("Remaining: ", stable_tensor_id)

