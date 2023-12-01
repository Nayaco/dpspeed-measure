import torch
import torch.nn as nn
import threading
import math
import time
def gpu_pcie_latency_test():
    with torch.no_grad():
        # cpu_2_gpu_event_0 = torch.cuda.Event(enable_timing=True) 
        # cpu_2_gpu_event_1 = torch.cuda.Event(enable_timing=True) 
        # gpu_2_cpu_event_0 = torch.cuda.Event(enable_timing=True) 
        # gpu_2_cpu_event_1 = torch.cuda.Event(enable_timing=True) 
        gpu_side_chunck_size = 128 * 1024 * 1024 # 512MB
        cpu_side_chunck_size = gpu_side_chunck_size
        gpu_side_chunck_0 = torch.rand(gpu_side_chunck_size, device='cuda:0', dtype=torch.float32)
        cpu_side_chunck_0 = torch.rand(cpu_side_chunck_size, device='cpu', pin_memory=True, dtype=torch.float32)
        gpu_side_chunck_1 = torch.rand(gpu_side_chunck_size, device='cuda:0', dtype=torch.float32)
        cpu_side_chunck_1 = torch.rand(cpu_side_chunck_size, device='cpu', pin_memory=True, dtype=torch.float32)
        torch.cuda.synchronize()
        # mixsure_event_0.record()
        with torch.cuda.stream(torch.cuda.Stream()):
            gpu_side_chunck_0.copy_(cpu_side_chunck_0, non_blocking=True)
            gpu_side_chunck_1.copy_(cpu_side_chunck_1, non_blocking=True)
        # mixsure_event_1.record()
        torch.cuda.synchronize()

        # print(cpu_2_gpu_event_0.elapsed_time(cpu_2_gpu_event_1))
        # print(gpu_2_cpu_event_0.elapsed_time(gpu_2_cpu_event_1))

def gpu_pcie_latency_test_dual():
    with torch.no_grad():
        mixsure_event_0 = torch.cuda.Event(enable_timing=True) 
        mixsure_event_1 = torch.cuda.Event(enable_timing=True) 
        gpu_side_chunck_size = 1024 * 1024 * 1024 #  4 * 1024 * 1024 * 1024 # 4 GB
        cpu_side_chunck_size = gpu_side_chunck_size
        gpu_side_chunck_0 = torch.rand(gpu_side_chunck_size, device='cuda:0', dtype=torch.float32)
        cpu_side_chunck_0 = torch.rand(cpu_side_chunck_size, device='cpu', pin_memory=True, dtype=torch.float32)
        gpu_side_chunck_1 = torch.rand(gpu_side_chunck_size, device='cuda:0', dtype=torch.float32)
        cpu_side_chunck_1 = torch.rand(cpu_side_chunck_size, device='cpu', pin_memory=True, dtype=torch.float32)
        
        torch.cuda.synchronize()
        mixsure_event_0.record()
        with torch.cuda.stream(torch.cuda.Stream()):
            gpu_side_chunck_0.copy_(cpu_side_chunck_0, non_blocking=True)
        with torch.cuda.stream(torch.cuda.Stream()):
            cpu_side_chunck_1.copy_(gpu_side_chunck_1, non_blocking=True)
        torch.cuda.synchronize()
        mixsure_event_1.record()
        
        print(mixsure_event_0.elapsed_time(mixsure_event_1))

def ssd_pcie_bandwidth_test():

    ssd_chunck_size = '4G'
    buffer_size = 4096
    size_byte = convert_to_bytes(ssd_chunck_size)
    n = math.ceil(size_byte/buffer_size)
    with open('ssd_write_test.bin', 'wb') as f:
        # barrier.wait()
        start = time.time()
        s = b'0'*buffer_size
        for i in range(n):
            f.write(s)
        end = time.time()
    time_used = end - start
    print("SSD write speed: {} GB/s".format(size_byte/1024/1024/1024/time_used))

def convert_to_bytes(size_str):
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4, 'P': 1024**5}
    size_str = size_str.upper()
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            size = float(size_str[:-1]) * multiplier
            return int(size)
    return int(size_str)

if __name__ == "__main__":
    # global barrier 
    # barrier = threading.Barrier(2)
    # t1 = threading.Thread(target=gpu_pcie_bandwidth_test)
    # t2 = threading.Thread(target=ssd_pcie_bandwidth_test)
    
    # start threads
    # t1.start()
    # t2.start()
  
    # wait until threads finish their job
    # t1.join()
    # t2.join()
    gpu_pcie_latency_test()
    # gpu_pcie_latency_test_dual()
    # ssd_pcie_bandwidth_test()
    print("GPU-PCIe latency test done!")