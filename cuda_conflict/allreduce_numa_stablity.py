#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import datetime
import time

def run_single(rank, size):
    """ Simple collective communication. """
    single_event_0_0 = torch.cuda.Event(enable_timing=True) 
    single_event_0_1 = torch.cuda.Event(enable_timing=True) 
    
    single_event_1_0 = torch.cuda.Event(enable_timing=True) 
    single_event_1_1 = torch.cuda.Event(enable_timing=True) 
    
    # stream_comm = torch.get_stream()
    # stream_matmul = torch.get_stream()
    group = dist.new_group([0, 1])
    device = torch.device('cuda:%d'%rank)
    comm_size = 1024 * 1024 * 512 # 1GB
    matmul_scale = (8192, 8192) # 4GB
    tensor_0 = torch.rand(comm_size, dtype=torch.float16, device=device)
    tensor_1 = torch.rand(*matmul_scale, dtype=torch.float16, device=device)
    tensor_2 = torch.rand(*matmul_scale, dtype=torch.float16, device=device)
    # warm up
    for i in range(40):
        _ = torch.matmul(tensor_1, tensor_2)
    
    torch.cuda.synchronize(device=device)
    dist.barrier()
    start = time.time()
    # single_event_0_0.record()
    dist.all_reduce(tensor_0, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize(device=device)
    end = time.time()
    # single_event_0_1.record()
    torch.cuda.synchronize(device=device)
    print('Rank ', rank, 'single allreduce in ', (end-start)*1000.0) 
    # print('Rank ', rank, 'single allreduce in ', single_event_0_0.elapsed_time(single_event_0_1), 'ms')
    torch.cuda.synchronize(device=device)
    tensor_0 = tensor_0 + tensor_0
    # print('Rank ', rank, ' has data ', tensor_0[:10])

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlap Matmul and Allreduce')
    parser.add_argument('--backend', type=str, default='nccl')
    # parser.add_argument('--overlap', action='store_true')
    args = parser.parse_args()
    # overlap = args.overlap
    # backend = args.backend
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_single))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()