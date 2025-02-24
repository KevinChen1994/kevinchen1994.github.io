---
{
  "title": "Mixtral 8x7B MoE模型笔记",
  "date": "2024-01-05",
  "tags": [
    "MoE"
  ],
  "categories": [
    "LLM"
  ],
  "summary": "随着 Mixtral 8x7B 的推出，一种称为混合专家模型 (Mixed Expert Models，简称 MoEs) 的 Transformer 模型在开源人工智能社区引起了广泛关注。",
  "authors": [],
  "draft": false,
  "generated_time": "2025-02-24T09:43:15.436906"
}
---

## 前言

大模型时代模型的参数量越来越大，GPT3的模型参数达到了175B，各大厂商也在不断突破模型参数量的天花板。模型参数量增大显而易见的好处就是模型的能力越来越强，并且模型的参数量达到一定的规模后，模型就会出现涌现能力（Emergent Abilities），而带来的坏处也是很明显的，那就是训练和推理的硬件成本不断增加。

2023年12月，Mistral AI在开源社区扔了一条磁力链接，引爆了社交网络。Mistral AI基于混合专家模型Mixture of Experts（MoE），证明通过8个7B的模型就能超越LLaMA 2 70B模型的效果，甚至部分超越了GPT 3.5的水平。之前就有人分析过GPT4就是使用8个专家模型组成的专家系统，这给我们带来了很多启发，是否未来大模型未来会朝着这个方向发展呢？

![](/images/notion_17348d53-6124-4fc9-8432-55260709f9a874b12913-f31e-4dca-acca-ccbd24cf586e.png)

## Mixtral 8x7B模型架构

Mixtral 8x7B与LLaMA模型的区别就是在attention计算中将MLP Layer替换成了一个门控层和8个专家模型，通过门控层会给出每个专家层的权重，每个token会选择top2的专家进行计算， 这使得模型训练和推理的速度相比于LLaMA 2 70B会显著提高。

不过这里有一个误区，那就是模型虽然叫8x7B，但是模型的参数并不是56B，因为在每个层中只有专家层是独立存在的，其他部分如attention是权重共享的，所以模型的参数量在47B左右。

![](/images/notion_4fb45021-7953-4490-b3c2-52479c26700c0775893f-4189-40b0-a47f-cf4a45d07bff.png)

## 专家模块细节

因为Mixtral 8x7B模型架构与LLaMA相比只有FFN块不同，所以我们只关注这块的细节。

我之前看过一个代码解读，他说MoE层是先计算所有专家的输出，然后在选择每个token对应的专家，其实这样的说法是错误的。因为如果每次都计算所有专家的输出，那就不能体现出MoE模型的优势了，8个模型都计算的话，你的计算量是很大的，那么你的耗时也会增加，起不到提速的效果。所以正确的理解是通过门控层来选择专家，然后只有对应的2个专家会进行前向计算，这样就起到了减少计算量和提速的效果。

首先输入经过attention计算后，经过残差连接、Norm层会输入进行专家层，专家层由门控层和8个专家构成。门控层其实就是一个全连接层，其输出结果再经过softmax函数得到各专家的权重，我们会选择权重排名的top2作为当前token要使用的专家。然后对这两个专家的权重重新进行归一化，得到这两个专家的权重，在前向计算的过程中，top2的专家的输出结果会与其对应的权重进行加权求和，最终我们就能得到整个输入使用不同的专家的结果。

整个流程是比较简单的，可能容易弄混的地方在于模型的输入是**token粒度**的，所以我们在计算权重和选择专家的过程中，都是以每个token为视角的。也就是每一个token都会计算专家权重，并选择top2的专家计算前向结果，最终得到的是整个输入的结果。这个过程结合代码看会更加清晰。

## 专家模块代码注释

下边的代码是专家模块的实现方法，我对重要部分写上了注释，可以结合注释进行理解。

```python
class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        # 由attention计算后输出的hidden_states作为输入
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将hidden_states构建成一个二维的形状，用于处理每一个token
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # 通过门控来生成路由，用来决定每一个token由哪些专家处理
        router_logits = self.gate(hidden_states)

        # 通过softmax计算每一个专家对于每个token的处理权重
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # 选取每个token的前top_k个专家和其对应的权重  selected_experts: (batch * sequence_length, top_k)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # 对每一个token对应的专家的权重值进行归一化，使其权重之和为1
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # final_hidden_states用来存储每个token对应的专家结果，初始值为0
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # 使用one hot编码来代表每个token使用哪些专家
        # one hot: (batch * sequence_length, top_k, num_experts) => expert_mask: (num_experts, top_k, batch * sequence_length)
        # 这样做的好处就是，用专家的视角，每次遍历只需要遍历每个专家所需要处理的token即可，否则需要遍历每个token使用了哪个专家，前向的次数随着文本的长度线性增加。
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # idx代表当前专家作为top1需要负责的token索引、作为top2需要负责的token的索引
            # top_x代表当前专家负责的token的索引位置。
            idx, top_x = torch.where(expert_mask[expert_idx])

            # 如果top_x中没有1，则代表当前专家不负责任何token，就跳过这个专家
            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # 根据索引从输入的隐向量中取得对应的向量，传入到专家模型中进行前向计算
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # 将当前专家模型的输出写入到预先定义好的final_hidden_states中
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
```

> 🚀 如果上边的代码看一遍不好理解的话，可以看一下下边的简化版本，结合debug的输出看更容易理解。

下图中的代码代表了选择专家的简单逻辑，假设当前有10个token和4个专家，每个token选择2个专家。expert_mask输出的结果就是每个专家需要负责的token，我们拆开来看。

在第一次遍历的过程中，我们遍历的是第一个专家，可以看到他是一个(2, 10)的矩阵，第一行代表了当前专家作为top1负责的token，第二行代表了当前专家作为top2负责的token。

我们通过`torch.where(expert_mask)` 来进行解析这个结果，得到的`idx` 中的0代表了当前专家作为top1需要负责的token，1代表了当前专家作为top2需要负责的token，对应的`top_x` 则代表了当前专家负责的token的索引位置，将`idx`和`top_x` 组合就得到了当前专家作为top1、top2负责的token的索引，例如(0,2)、(0,8)、(0,9)、(1,0)、(1,1)、(1,5)，对应的意思就是当前专家作为top1负责的token索引为2、8、9，当前专家作为top2负责的token索引为0、1、5。

![](/images/notion_2e47442c-a3f0-46c7-b02f-5a100f13581e7a072245-b5a2-42a6-92be-0d87ff5bf77c.png)

## 更多阅读资料

[Mistral AI官网对Mixtral 8x7B介绍](https://mistral.ai/news/mixtral-of-experts/)

[huggingface 混合专家模型 (MoE) 详解](https://huggingface.co/blog/zh/moe)

[这篇文章提出的关于Mixtral 8x7B的几个问题很有意思，比如训练的时候几个专家同时训练、8个专家的贡献度怎么样](https://zhuanlan.zhihu.com/p/674751021)



