
"""
将算子的属性更新，包括执行时间与生命周期；
"""
def time_build(transformer_op_array, end_forward_id, Transformer_Block):
# Step1：建立开始时间
    for op_id in range(len(transformer_op_array)):
        op = transformer_op_array[op_id]
        begin_time = 0
        for front_op in range(0, op_id):
            begin_time+=transformer_op_array[front_op].compute_time
            transformer_op_array[op_id].first_use_time = begin_time

    # Step2：建立last use时间与生命周期
    for op_id in range(0, end_forward_id):
        op_base_id = op_id - 14 * transformer_op_array[op_id].transformer_block_id
        # 寻找反传对应的id
        op_last_use_id = 14 * Transformer_Block + (Transformer_Block - 1 - transformer_op_array[op_id].transformer_block_id) * 14 + (15 - op_base_id)
        
        if transformer_op_array[op_id].type == 0:
            transformer_op_array[op_id].backward_id = op_last_use_id
            transformer_op_array[op_id].last_use_time = transformer_op_array[op_last_use_id].first_use_time
            live_time = transformer_op_array[op_id].last_use_time - transformer_op_array[op_id].first_use_time
            transformer_op_array[op_id].livetime = live_time # 赋予生命周期
    return transformer_op_array

"""
计算重计算算子的总时间与空间
"""
def get_policy_time_space(policy, transformer_op_array_forward):
    total_time = 0
    total_space = 0
    for i in policy:
        total_time += transformer_op_array_forward[i-1].compute_time
        total_space += transformer_op_array_forward[i-1].output_size
    return total_time, total_space

"""
计算训练总时间
"""
def get_policy_time(transformer_op_array):
    total_time = 0
    for i in range(len(transformer_op_array)):
        total_time += transformer_op_array[i-1].compute_time
    return total_time