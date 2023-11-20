## 模型配置

```
GPT_ARGS="
    --num-layers 32 \
    --hidden-size 1792 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --ffn-hidden-size 7168 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --no-pipeline-parallel \
    --zero-stage 0 \
"
```

TP = 1, GPU = 1

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

可参考：https://zhuanlan.zhihu.com/p/624740065

```
Layer name: language_model.embedding.word_embeddings.weight, Layer size: torch.Size([50304, 1792])
Layer name: language_model.embedding.position_embeddings.weight, Layer size: torch.Size([1024, 1792])
```

- (1) 词典大小： [50304, hidden-size] = [50304, 1792] = V * h
- (2) 位置Embedding： [max-position-embeddings, hidden-size] = [1024, 1792] = P * h

下面为一个Transform Layer的结构，具体Size要乘N；


```
Layer name: language_model.encoder.layers.0.input_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.input_layernorm.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.self_attention.query_key_value.weight, Layer size: torch.Size([5376, 1792])
Layer name: language_model.encoder.layers.0.self_attention.query_key_value.bias, Layer size: torch.Size([5376])
Layer name: language_model.encoder.layers.0.self_attention.dense.weight, Layer size: torch.Size([1792, 1792])
Layer name: language_model.encoder.layers.0.self_attention.dense.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.post_attention_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.post_attention_layernorm.bias, Layer size: torch.Size([1792])
Layer name: language_model.encoder.layers.0.mlp.dense_h_to_4h.weight, Layer size: torch.Size([7168, 1792])
Layer name: language_model.encoder.layers.0.mlp.dense_h_to_4h.bias, Layer size: torch.Size([7168])
Layer name: language_model.encoder.layers.0.mlp.dense_4h_to_h.weight, Layer size: torch.Size([1792, 7168])
Layer name: language_model.encoder.layers.0.mlp.dense_4h_to_h.bias, Layer size: torch.Size([1792])
```
- (3) Attention LayerNorm的权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2 * h
- (4) Attention QKV线性层权重：[hidden-size * 3, hidden-size] = [5376, 1792] = 3 * h * h
- (5) Attention QKV线性层偏置：[hidden-size * 3] = [5376] = 3 * h
- (6) Attention QKV计算后的输出结果权重：[hidden-size, hidden-size] = [1792, 1792] = h * h
- (7) Attention QKV计算后的输出结果偏置：[hidden-size] = [1792] = h


- (8) MLP LayerNorm的权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2 * h
- (9) MLP层h_to_4h线性层权重：[hidden-size * 4, hidden-size] = [7168, 1792] = 4 * h * h
- (10) MLP层h_to_4h线性层偏置：[hidden-size] = [7168] = 4 * h
- (11) MLP层4h_to_h线性层权重：[hidden-size * 4, hidden-size] = [1792,7168] = 4 * h * h
- (12) MLP层4h_to_h线性层偏置：[hidden-size] = [1792] = h

One Transformer Layer Memory Size = 12 * h * h + 13 * h

```
Layer name: language_model.encoder.final_layernorm.weight, Layer size: torch.Size([1792])
Layer name: language_model.encoder.final_layernorm.bias, Layer size: torch.Size([1792])
```

- (13) 最后一层Layernorm权重+偏置： [hidden-size] + [hidden-size] = [1792] + [1792] = 2 * h

总结模型大小为：V×h + P×h + (12×h×h + 13×h) × l + 2×h，其中V为字典维度，P为位置向量维度，h为Embedding的维度，l为Transformer层数。
