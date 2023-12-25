# 生成启发式策略

"""
策略生成思路, (1)对于剩余的算子,对swap无任何Overhead的算子(tensor)进行转移操作; (2)空间还不够就进行压缩+转移; (3)再对剩余
的算子进行转移/压缩/压缩+转移/重计算开销的判断,选定开销最小的; 依次重复以上三个步骤,直到满足Memory_saving的条件;
"""


"""
计算算子数组对应的Tensor空间大小;
"""
def memory_saving_calculation(tensor_id_container, transformer_op_array):
    total_space = 0
    for i in tensor_id_container:
        total_space += transformer_op_array[i].output_size #从全局算子array中找
    return total_space


"""
制定无Overhead的swap策略;
"""
def swap_policy(remaining_tensor_id, transformer_op_array, end_forward_id):
    forward_time =transformer_op_array[end_forward_id].first_use_time
    PCIe_Bandwidth = 24 #24MB/ms 
    forward_cur_swap = 0 #当前正在转移的tensor结束转出的时间
    backward_cur_swap = 10000000 #当前正在转入的tensor最早发起转入的时间
    swap_candidate_id = []
    for tensor_candidate in remaining_tensor_id:
        swap_time = transformer_op_array[tensor_candidate].output_size / PCIe_Bandwidth
        tensor_swap_forward_begin = transformer_op_array[tensor_candidate].first_use_time + transformer_op_array[tensor_candidate].compute_time #数据可以释放的时刻
        tensor_swap_backward_end = transformer_op_array[transformer_op_array[tensor_candidate].backward_id].first_use_time #数据需要使用的时刻

        #判断是否可以完全并行
        real_swap_out_begin_time = max(forward_cur_swap, tensor_swap_forward_begin) #真正转出发起的时间，考虑PCIe同时只能串行转一份数据
        real_swap_out_end_time =  real_swap_out_begin_time + swap_time #真正转出结束的时间
        real_swap_in_end_time = min(tensor_swap_backward_end, backward_cur_swap) #真正转入结束的时间
        real_swap_in_begin_time = real_swap_in_end_time - swap_time #真正转入开始的时间

        if real_swap_out_end_time < forward_time and real_swap_in_begin_time > forward_time:
            forward_cur_swap = real_swap_out_end_time
            backward_cur_swap = real_swap_in_begin_time
            swap_candidate_id.append(tensor_candidate)
    return swap_candidate_id

"""
制定用于压缩的swap策略;
remaining_tensor_id:待判定为转移的candidate;
forward_cur_swap:当前压缩+转移的tensor结束转出的时间
backward_cur_swap:当前正在转入的tensor最早发起转入的时间

"""
def swap_policy_based_on_compression(remaining_tensor_id, transformer_op_array, end_forward_id, forward_cur_swap, backward_cur_swap):
    forward_time =transformer_op_array[end_forward_id].first_use_time
    PCIe_Bandwidth = 24 #24MB/ms 
    swap_candidate_id = []
    for tensor_candidate in remaining_tensor_id:
        swap_time = transformer_op_array[tensor_candidate].output_size / PCIe_Bandwidth
        tensor_swap_forward_begin = transformer_op_array[tensor_candidate].first_use_time + transformer_op_array[tensor_candidate].compute_time #数据可以释放的时刻
        tensor_swap_backward_end = transformer_op_array[transformer_op_array[tensor_candidate].backward_id].first_use_time #数据需要使用的时刻

        #判断是否可以完全并行
        real_swap_out_begin_time = max(forward_cur_swap, tensor_swap_forward_begin) #真正转出发起的时间，考虑PCIe同时只能串行转一份数据
        real_swap_out_end_time =  real_swap_out_begin_time + swap_time #真正转出结束的时间
        real_swap_in_end_time = min(tensor_swap_backward_end, backward_cur_swap) #真正转入结束的时间
        real_swap_in_begin_time = real_swap_in_end_time - swap_time #真正转入开始的时间
        if real_swap_out_end_time < forward_time and real_swap_in_begin_time > forward_time:
            forward_cur_swap = real_swap_out_end_time
            backward_cur_swap = real_swap_in_begin_time
            swap_candidate_id.append(tensor_candidate)
        return swap_candidate_id, forward_cur_swap, backward_cur_swap # 返回新增的swap tensor以及现在转出，转入的时间截止点

"""
制定压缩+转移策略;
"""

# 重写=>将转移的数据逐个压缩，看是否能新加入可并行swap的tensor，如果不够持续上述过程；


def compression_swap_policy_new(Memory_saving,stable_tensor_id, remaining_tensor_id, swap_candidate_id, transformer_op_array, end_forward_id, Transformer_Block):
    import tools,copy
    Compression_Bandwidth = 500 #MB/ms
    print("剩余目标：",Memory_saving,swap_candidate_id)
    while Memory_saving>0:
        for compression_tensor in swap_candidate_id: #将swap的tensor逐个赋予压缩
            if compression_tensor in stable_tensor_id: #跳过锚点
                continue
            
            compression_time = transformer_op_array_compression_swap[compression_tensor].output_size / Compression_Bandwidth # 压缩、解压时间，ms
            compressed_size = transformer_op_array_compression_swap[compression_tensor].output_size / 2 #假设FP16->FP8
            transformer_op_array_compression_swap[compression_tensor].compute_time += compression_time #将压缩时间加入到计算时间上
            transformer_op_array_compression_swap[transformer_op_array_compression_swap[compression_tensor].backward_id].compute_time += compression_time #将解压缩时间加入到计算时间上
            transformer_op_array_compression_swap[compression_tensor].output_size = compressed_size #更新输出数据大小
            transformer_op_array_compression_swap = tools.time_build(transformer_op_array_compression_swap, end_forward_id, Transformer_Block) #更新算子信息


        #提前占住压缩+转移的数据，更新算子数组，需要更新remaining_tensor_id，保留压缩转移之后的数据 > max_id


    return 0

def compression_swap_policy(Memory_saving,stable_tensor_id, remaining_tensor_id, transformer_op_array, end_forward_id, Transformer_Block):
    import tools,copy
    transformer_op_array_compression_swap = copy.deepcopy(transformer_op_array)
    Compression_Bandwidth = 500 #MB/ms
    #算子压缩，降低算子的大小，将压缩、解压缩时间判断增加到算子的执行时间上
    compression_and_swap_tensor_id = [] #转移/压缩转移的tensor
    only_swap_tensor_id = [] #只转移的tensor
    only_compression_tensor_id = [] #只压缩的tensor
    save_memory_compression_swap = 0
    cur = 0
    for cur in range(len(remaining_tensor_id)):
        compression_candidate = remaining_tensor_id[cur]
        if compression_candidate not in stable_tensor_id: # stable数据不进行压缩，但考虑转移；其余尝试数据压缩+转移

            compression_time = transformer_op_array_compression_swap[compression_candidate].output_size / Compression_Bandwidth # 压缩、解压时间，ms
            compressed_size = transformer_op_array_compression_swap[compression_candidate].output_size / 2 #假设FP16->FP8
            transformer_op_array_compression_swap[compression_candidate].compute_time += compression_time #将压缩时间加入到计算时间上
            transformer_op_array_compression_swap[transformer_op_array_compression_swap[compression_candidate].backward_id].compute_time += compression_time #将解压缩时间加入到计算时间上
            transformer_op_array_compression_swap[compression_candidate].output_size = compressed_size #更新输出数据大小
            transformer_op_array_compression_swap = tools.time_build(transformer_op_array_compression_swap, end_forward_id, Transformer_Block) #更新算子信息
        
        
        tmp_remaining_tensor_id = copy.deepcopy(compression_and_swap_tensor_id) #将已经决策为压缩/压缩+转移的数据放到策略列表中

        for cur_ in range(cur, len(remaining_tensor_id)): #将cur之后的数据添加进候选者
            tmp_remaining_tensor_id.append(remaining_tensor_id[cur_])

        # 制定转移策略，其中compression_tensor_id中为压缩+转移
        compression_swap_candidate_id = swap_policy(tmp_remaining_tensor_id, transformer_op_array_compression_swap, end_forward_id) # 尝试对compression_candidate进行压缩，并生成转移策略
        if compression_candidate not in compression_swap_candidate_id: #判断该数据压缩后还是无法转移，所以没有必要压缩，还原数据
            if compression_candidate in stable_tensor_id: 
                continue
            else: #如果数据不是锚点，需要把压缩后的属性调整回来
                transformer_op_array_compression_swap[compression_candidate].compute_time -= compression_time
                transformer_op_array_compression_swap[transformer_op_array_compression_swap[compression_candidate].backward_id].compute_time -= compression_time
                transformer_op_array_compression_swap[compression_candidate].output_size = compressed_size * 2
                continue
        
        compression_and_swap_tensor_id.append(compression_candidate)
        if compression_candidate in stable_tensor_id: #判断是不是锚点
            only_swap_tensor_id.append(compression_candidate) #锚点只转移，前面没有对锚点数据修改压缩属性
        else:
            only_compression_tensor_id.append(compression_candidate)

        save_memory_compression_swap = memory_saving_calculation(compression_swap_candidate_id, transformer_op_array) #求解转移释放的空间开销
        remaining_memory = Memory_saving - save_memory_compression_swap # 还需优化的空间
        if remaining_memory<0:
            training_time_after_policy = tools.get_policy_time(transformer_op_array_compression_swap)
            return 1,only_compression_tensor_id, only_swap_tensor_id, save_memory_compression_swap,training_time_after_policy
    # print("COmpression: ",compression_tensor_id, compression_swap_candidate_id)
    return 0,only_compression_tensor_id, only_swap_tensor_id,save_memory_compression_swap,0


def best_policy_for_tensor(Memory_saving,stable_tensor_id, remaining_tensor_id, transformer_op_array, end_forward_id, Transformer_Block):
    return 0


"""
Memory_saving: 目标优化的显存值；
transformer_op_array: 所有算子信息;
pipeline_tensor_id: 用于并行的算子id;
remaining_tensor_id: 保留下来的算子id;
stable_tensor_id: 由于提前做检查点, 需要长期保留到GPU中的算子id;
end_forward_id: 反传开始的算子id;
Transformer_Block: Transformer的块数;
training_time: 模型训练时间;
"""
def policy_generate(Memory_saving, transformer_op_array, pipeline_tensor_id, remaining_tensor_id, stable_tensor_id, end_forward_id, Transformer_Block, training_time):
    

    ####
    #### 并行重计算
    save_memory_pipeline = memory_saving_calculation(pipeline_tensor_id, transformer_op_array) #确定并行方案能释放多少显存
    Memory_saving = Memory_saving - save_memory_pipeline #更新目标Memory
    
    if Memory_saving<0:
        print("找到解，选择Pipeline重计算策略:",pipeline_tensor_id)
        print()
        return training_time,pipeline_tensor_id,[],[]
    else:
        print("(Step - 1 没找到解) 并行重计算释放的显存：",save_memory_pipeline,"；剩余存储优化目标：",Memory_saving,"；并行重计算ID：" ,pipeline_tensor_id,"；剩余Tensor的ID：", remaining_tensor_id,"；需要常驻显存的Tensor ID：" ,stable_tensor_id)
        print()

    ####
    #### 无开销转移
    swap_candidate_id = swap_policy(remaining_tensor_id, transformer_op_array, end_forward_id)
    save_memory_swap = memory_saving_calculation(swap_candidate_id, transformer_op_array) #求解转移释放的空间开销

    if Memory_saving - save_memory_swap<0:
        print("找到解，选择Pipeline重计算+转移的策略:",pipeline_tensor_id, swap_candidate_id)
        print()
        return training_time,pipeline_tensor_id, swap_candidate_id,[]
    else:
        print("(Step - 2 没找到解) 转移释放的显存：",save_memory_swap,"；剩余存储优化目标：",Memory_saving - save_memory_swap,"；转移Tensor的ID：" ,swap_candidate_id)
        print()
    
    ####
    #### 压缩+转移
    flag, compression_tensor_id, swap_candidate_id, save_memory_compression_swap,training_time_after_policy = compression_swap_policy(Memory_saving, stable_tensor_id, remaining_tensor_id, transformer_op_array, end_forward_id, Transformer_Block)
    # compression_swap_policy_new(Memory_saving, stable_tensor_id, remaining_tensor_id, swap_candidate_id, transformer_op_array, end_forward_id, Transformer_Block)
    if flag == 0:
        for i in compression_tensor_id:
            remaining_tensor_id.remove(i)
        for i in swap_candidate_id:
            remaining_tensor_id.remove(i)
        print("(Step - 3 没找到解) 压缩+转移释放的最大显存量为：",save_memory_compression_swap,"；剩余存储优化目标：",Memory_saving - save_memory_compression_swap, "；压缩+转移的数据：",compression_tensor_id, "；只转移的数据：",swap_candidate_id, "；剩余Tensor：",remaining_tensor_id)
        print()
    else:
        print("找到解，选择Pipeline重计算+压缩转移+转移的策略:",pipeline_tensor_id, compression_tensor_id, swap_candidate_id, "；训练时间为 (ms)：",training_time_after_policy)
        print()
        return training_time_after_policy, pipeline_tensor_id, swap_candidate_id, compression_tensor_id


    ####
    #### 对剩余Tensor选择最优策略
    # Memory_saving = Memory_saving - save_memory_compression_swap
    
    # best_policy_for_tensor(Memory_saving, stable_tensor_id, remaining_tensor_id, "transformer_op_array", end_forward_id, Transformer_Block)
    # print(Memory_saving)

    return 0,[],[],[]


