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
            # print(tensor_candidate,swap_time,tensor_swap_forward_begin,forward_cur_swap,real_swap_out_end_time)
            # print(tensor_candidate,swap_time,backward_cur_swap,tensor_swap_backward_end,real_swap_in_begin_time)
            # print()
            forward_cur_swap = real_swap_out_end_time
            backward_cur_swap = real_swap_in_begin_time
            swap_candidate_id.append(tensor_candidate)
    return swap_candidate_id


"""
制定压缩+转移策略;
"""

# 重写=>将转移的数据逐个压缩，看是否能新加入可并行swap的tensor，如果不够持续上述过程；

def compression_swap_policy(Memory_saving,stable_tensor_id, remaining_tensor_id, transformer_op_array, end_forward_id, Transformer_Block):
    import tools,copy
    transformer_op_array_compression_swap = copy.deepcopy(transformer_op_array)
    Compression_Bandwidth = 500 #MB/ms
    #算子压缩，降低算子的大小，将压缩、解压缩时间判断增加到算子的执行时间上
    compression_tensor_id = []
    save_memory_compression_swap = 0
    for compression_candidate in remaining_tensor_id:
        if compression_candidate in stable_tensor_id: # 跳过作为检查点的stable数据: 需要保障检查点数据需要时立刻在，不能引入压缩、解压缩
            continue

        compression_time = transformer_op_array_compression_swap[compression_candidate].output_size / Compression_Bandwidth # 压缩、解压时间，ms
        compressed_size = transformer_op_array_compression_swap[compression_candidate].output_size / 2 #假设FP16->FP8
        transformer_op_array_compression_swap[compression_candidate].compute_time += compression_time #将压缩时间加入到计算时间上
        transformer_op_array_compression_swap[transformer_op_array_compression_swap[compression_candidate].backward_id].compute_time += compression_time #将解压缩时间加入到计算时间上
        transformer_op_array_compression_swap[compression_candidate].output_size = compressed_size #更新输出数据大小
        transformer_op_array_compression_swap = tools.time_build(transformer_op_array_compression_swap, end_forward_id, Transformer_Block) #更新算子信息
        
        
        # 有前后影响，加入33后，有可能22就可以能转移了，本来是不可以转移的



        compression_swap_candidate_id = swap_policy(remaining_tensor_id, transformer_op_array_compression_swap, end_forward_id) # 尝试对compression_candidate进行压缩，并生成转移策略
        # print(compression_candidate, compression_swap_candidate_id)
        print("Compre:",compression_candidate,compression_swap_candidate_id,compression_tensor_id)
        if compression_candidate not in compression_swap_candidate_id: #判断该数据压缩后还是无法转移，所以没有必要压缩，还原数据
            transformer_op_array_compression_swap[compression_candidate].compute_time -= compression_time
            transformer_op_array_compression_swap[transformer_op_array_compression_swap[compression_candidate].backward_id].compute_time -= compression_time
            transformer_op_array_compression_swap[compression_candidate].output_size = compressed_size * 2
            continue
        


        compression_tensor_id.append(compression_candidate)
        save_memory_compression_swap = memory_saving_calculation(compression_swap_candidate_id, transformer_op_array) #求解转移释放的空间开销
        remaining_memory = Memory_saving - save_memory_compression_swap # 还需优化的空间
        if remaining_memory<0:
            training_time_after_policy = tools.get_policy_time(transformer_op_array_compression_swap)
            for i in compression_tensor_id:
                compression_swap_candidate_id.remove(i)
            # print("压缩+转移的Tensor：",compression_tensor_id, "; 只转移的Tensor：",compression_swap_candidate_id)
            return 1,compression_tensor_id, compression_swap_candidate_id, save_memory_compression_swap,training_time_after_policy
    # print("COmpression: ",compression_tensor_id, compression_swap_candidate_id)
    return 0,compression_tensor_id, compression_swap_candidate_id,save_memory_compression_swap,0

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
        return training_time
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
        return training_time
    else:
        print("(Step - 2 没找到解) 转移释放的显存：",save_memory_swap,"；剩余存储优化目标：",Memory_saving - save_memory_swap,"；转移Tensor的ID：" ,swap_candidate_id)
        print()
    
    ####
    #### 压缩+转移
    flag, compression_tensor_id, swap_candidate_id, save_memory_compression_swap,training_time_after_policy = compression_swap_policy(Memory_saving, stable_tensor_id, remaining_tensor_id, transformer_op_array, end_forward_id, Transformer_Block)
    if flag == 0:
        for i in swap_candidate_id:
            remaining_tensor_id.remove(i)

        for i in compression_tensor_id:
            swap_candidate_id.remove(i)
        print("(Step - 3 没找到解) 压缩+转移释放的最大显存量为：",save_memory_compression_swap,"；剩余存储优化目标：",Memory_saving - save_memory_compression_swap, "；压缩+转移的数据：",compression_tensor_id, "；只转移的数据：",swap_candidate_id, "；剩余Tensor：",remaining_tensor_id)
        print()
    else:
        print("找到解，选择Pipeline重计算+压缩转移+转移的策略:",pipeline_tensor_id, compression_tensor_id, swap_candidate_id, "；训练时间为 (ms)：",training_time_after_policy)
        print()
        return training_time_after_policy


    ####
    #### 对剩余Tensor选择最优策略
    Memory_saving = Memory_saving - save_memory_compression_swap
    
    best_policy_for_tensor(Memory_saving, stable_tensor_id, remaining_tensor_id, "transformer_op_array", end_forward_id, Transformer_Block)
    print(Memory_saving)

    return 0


