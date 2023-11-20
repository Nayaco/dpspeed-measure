## 模型配置

```
GPT_ARGS="
    --num-layers 32 \
    --hidden-size 1792 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --ffn-hidden-size 7168 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 16 \
    --hidden-dropout 0.1 \
    --lr 0.0001 \
    --train-iters 500 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --min-lr 1.0e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --adam-eps 1e-06 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --no-pipeline-parallel \
    --zero-stage 0 \
"
```

TP = 2, DP, GPU = 2

## 模型结构

```
ParallelTransformerLayer(
  (input_layernorm): MixedFusedLayerNorm()
  (self_attention): ParallelAttention(
    (query_key_value): ColumnParallelLinear()
    (core_attention): CoreAttention(
      (scale_mask_softmax): FusedScaleMaskSoftmax()
      (attention_dropout): Dropout(p=0.1, inplace=False)
    )
    (dense): RowParallelLinear()
  )
  (post_attention_layernorm): MixedFusedLayerNorm()
  (mlp): ParallelMLP(
    (dense_h_to_4h): ColumnParallelLinear()
    (dense_4h_to_h): RowParallelLinear()
  )
)
```

## 模型各层的大小：

T为TP的并行度，会影响数据的大小；

```
Layer name: language_model.embedding.word_embeddings.weight, Layer size: torch.Size([25216, 1792])
Layer name: language_model.embedding.position_embeddings.weight, Layer size: torch.Size([1024, 1792])
```

- (1) 词典大小： [50304, hidden-size] = [25216, 1792] = V/T×h
- (2) 位置Embedding： [max-position-embeddings, hidden-size] = [1024, 1792] = P×h

下面为一个Transform Layer的结构，具体Size要乘N；


```
Layer name: language_model.encoder.layers.0.input_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.input_layernorm.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.self_attention.query_key_value.weight, Layer size: torch.Size([2688, 1792])
Layer name: language_model.encoder.layers.0.self_attention.query_key_value.bias, Layer size: torch.Size([2688])
Layer name: language_model.encoder.layers.0.self_attention.dense.weight, Layer size: torch.Size([1792, 896])
Layer name: language_model.encoder.layers.0.self_attention.dense.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.post_attention_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.post_attention_layernorm.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.mlp.dense_h_to_4h.weight, Layer size: torch.Size([3584, 1792])
Layer name: language_model.encoder.layers.0.mlp.dense_h_to_4h.bias, Layer size: torch.Size([3584])
Layer name: language_model.encoder.layers.0.mlp.dense_4h_to_h.weight, Layer size: torch.Size([1792, 3584])
Layer name: language_model.encoder.layers.0.mlp.dense_4h_to_h.bias, Layer size: torch.Size([1792])
```

- (3) Attention LayerNorm的权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2×h
- (4) Attention QKV线性层权重：[hidden-size×3 / T, hidden-size] = [2688, 1792] = 3×h×h / T 
- (5) Attention QKV线性层偏置：[hidden-size×3 / T] = [2688] = 3×h / T
- (6) Attention 线性层输出结果权重：[hidden-size, hidden-size / T] = [1792, 896] = h×h / T
- (7) Attention 线性层输出结果偏置：[hidden-size] = [1792] = h


- (8) MLP LayerNorm的权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2×h
- (9) MLP层h_to_4h线性层权重：[hidden-size×4, hidden-size] = [3584, 1792] = 4×h×h / T
- (10) MLP层h_to_4h线性层偏置：[hidden-size] = [3584] = 4×h / T
- (11) MLP层4h_to_h线性层权重：[hidden-size×4, hidden-size] = [1792, 3584] = 4×h×h / T
- (12) MLP层4h_to_h线性层偏置：[hidden-size] = [1792] = h

One Transformer Layer Memory Size = 12×h×h/T +  6×h + 7/T×h = 12×h×h/T + (6+7/T)×h


```
Layer name: language_model.encoder.final_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.final_layernorm.bias, Layer size: torch.Size([1792])
```
- (13) 最后一层Layernorm权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2×h


总结模型大小为：V×h + P×h + (12×h×h/T + (6+7/T)×h)×l + 2×h，其中V为字典维度，P为位置向量维度，h为Embedding的维度，l为Transformer层数，T为TP的并行度。

在当前测试模型下，大小为 25216 * 1792 + 1024 * 1792 + (12 * 1792 * 1792 / 2 + (6 + 7/2) * 1792) * 32 + 2 * 1792


**在每一轮迭代前，存储空间占用包括FP16的权重 + FP32的权重 + FP32的优化器M和V +FP16的梯度（工程实现上没有free，理论上前传不需要） = 9.9GB **
