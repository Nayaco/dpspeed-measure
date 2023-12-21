"""
显存计算工具：给定任意配置，用于计算无显存优化策略下的GPT模型对应的显存使用量；
Author： Ping
Date：12.10
"""
import numpy

# Model Config
attentionHeads = 16 
microBatchSize = 1
hiddenDimension = 1792
transformerLayers = 350
vocabularySize = 50304
sequenceLength = 1024
positionEmbeddings = 1024
optimizer = "Adam" #暂不支持修改

# Training Config

GPUSize = 16
ZeRO = 2 # 0 means DP
tensorParallelSize = 8
sequenceParallelSize = 0
pipelineParallelSize = 1 #暂不支持修改

# Model B

weights_Mem_all = (vocabularySize * hiddenDimension + positionEmbeddings * hiddenDimension + (12 * hiddenDimension 
        * hiddenDimension + 13 * hiddenDimension) * transformerLayers + 2 * hiddenDimension)


# Weight + bias
weights_Mem_per_GPU = vocabularySize * hiddenDimension / tensorParallelSize + positionEmbeddings * hiddenDimension + (12 * hiddenDimension 
    * hiddenDimension / tensorParallelSize + (6+7/tensorParallelSize) * hiddenDimension) * transformerLayers + 2 * hiddenDimension

# Weight Memory, MB
weights_Mem_FP16 = weights_Mem_per_GPU * 2 / 1024 /1024 / 1024
weights_Mem_FP32 = weights_Mem_per_GPU * 4 / 1024 /1024 / 1024

# Gradient Memory, MB
gradient_Mem_FP16 = weights_Mem_FP16

# Optimizer Memory, MB
momentum_Mem_FP32 = weights_Mem_FP32
variance_Mem_FP32 = weights_Mem_FP32

def zero_func(FP_Byte):
    return (vocabularySize * hiddenDimension + positionEmbeddings * hiddenDimension + (12 * hiddenDimension 
        * hiddenDimension + 13 * hiddenDimension) * transformerLayers + 2 * hiddenDimension) / GPUSize * FP_Byte / 1024 /1024 / 1024

if ZeRO == 1:
    momentum_Mem_FP32 = zero_func(4)
    variance_Mem_FP32 = zero_func(4)
    weights_Mem_FP32 = zero_func(4)

if ZeRO == 2:
    momentum_Mem_FP32 = zero_func(4)
    variance_Mem_FP32 = zero_func(4)
    gradient_Mem_FP16 = zero_func(2)
    weights_Mem_FP32 = zero_func(4)

if ZeRO == 3:
    momentum_Mem_FP32 = zero_func(4)
    variance_Mem_FP32 = zero_func(4)
    weights_Mem_FP32 = zero_func(4)
    gradient_Mem_FP16 = zero_func(2)
    weights_Mem_FP16 = zero_func(2)
    

# GB
Model_Mem = (weights_Mem_FP16 + gradient_Mem_FP16 + weights_Mem_FP32 + momentum_Mem_FP32 + variance_Mem_FP32) 

# Activation Memory GB
TransformerActivationMem = 0
if sequenceParallelSize==0:
    TransformerActivationMem = (10 * sequenceLength * microBatchSize * hiddenDimension + (24 * sequenceLength * microBatchSize * 
                    hiddenDimension + 5 * microBatchSize * attentionHeads * sequenceLength * sequenceLength) / tensorParallelSize) * transformerLayers
else:
    TransformerActivationMem = ((34 * sequenceLength * microBatchSize * hiddenDimension + 5 * microBatchSize * attentionHeads * sequenceLength * sequenceLength) / tensorParallelSize) * transformerLayers

activation_Mem = (TransformerActivationMem + 2 * sequenceLength * microBatchSize * hiddenDimension) / 1024 /1024 / 1024

# All Memory Occupy
All_Mem = Model_Mem + activation_Mem

print("##############################")
print("## 模型大小 (B): ",weights_Mem_all/1000000000)
print("## 模型大小(GB): ",Model_Mem)
print("##### -- 模型权重+偏置大小: ",weights_Mem_FP16)
print("##### -- 模型优化器大小: ",weights_Mem_FP32+momentum_Mem_FP32+variance_Mem_FP32)
print("##### -- 模型梯度大小: ",gradient_Mem_FP16)
print("## 模型Activation大小(GB): ",activation_Mem)
print("## 显存总大小(GB): ",All_Mem)
print("##############################")
