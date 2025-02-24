---
{
  "title": "PEFT",
  "date": "2023-08-25",
  "tags": [
    "大模型微调"
  ],
  "categories": [
    "LLM"
  ],
  "summary": "对于大模型微调起来不仅需要很多计算资源（GPU），还需要大量的训练时间，PEFT通过高效微调模型，使得我们可以使用单卡去训练比较大的LLM。",
  "authors": [],
  "draft": false,
  "generated_time": "2025-02-24T12:15:47.617017"
}
---

## Fine-Tuning

Pre-training后使用有标注的数据对模型进行微调，需要大量的有监督数据，预训练与微调之间存在gap，效果提升不够明显。代表模型EMLO、BERT、GPT1

## Prompt-Tuning系列

通过添加模版的方式，避免引入额外的参数，从而让语言模型能够在Zero-Shot，或者Few-Shot的场景下达到理想的效果。代表模型GPT3

### Discrete Prompt（离散Prompt）

离散Prompt是指将离散字符与在原始文本进拼接，且在训练中保持不变，这里保持不变是指这些离散字符的词向量（Word Embedding）在训练过程中保持不变。通常情况下，离散法不需要引入任何参数。

### Continuous Prompt（连续Prompt）

连续Prompt是指让模型在训练过程中，根据具体的上下文语义和任务目标对模板参数进行连续可调。因为在离散训练中，模板无法参与模型的训练环境，容易陷入局部最优，而如果将模版变为可以训练的参数，那么不同的样本可以在连续的向量空间中寻找合适的伪标记，同时也可以增强模型的泛化能力。因此，连续法需要引入少量的参数让模型在训练时进行参数更新。

### **Prefix Tuning （2021.01）**

_论文题目：Prefix-Tuning: Optimizing Continuous Prompts for Generation_

_论文地址：_[_https://arxiv.org/pdf/2101.00190.pdf _](https://arxiv.org/pdf/2101.00190.pdf)

_论文源码：_[_https://github.com/XiangLi1999/PrefixTuning_](https://github.com/XiangLi1999/PrefixTuning)

背景：1. 人工设计的离散Prompt模板成本高，并且模型对模板特别敏感，多一个字少一个字都可能造成较大的变化，并且效果可能不是最优的；2. 传统微调需要针对每一个下游任务单独保存一份模型的权重，训练成本太高。

基于此，Prefix Tuning提出了固定LM参数，为LM提供可训练的、特定任务的前缀，这样就可以针对不同的任务使用不同的前缀，并且也可以复用LM的参数了。其次，使用连续的Prompt，相比离散的Prompt效果更好。在实际使用中挑选任务相关的prefix与transformer进行组装，实现热插拔。



![](/images/notion_f6c47447-b851-4d8d-a50a-a256fb77c5f075e2cf07-b2cd-4739-9ef9-c77d8bbf4c32.png)

prefix tuning可以应用在decoder-only的模型上，也可以应用在encoder-decoder模型上，但主要应用的任务是NLG任务。

![](/images/notion_e5b7c228-a4e2-4f8c-8984-83cadcd704055703ecb9-a68b-44a3-84a0-745713812d06.png)

### **P-tuning （2021.03）**

_论文题目：GPT Understands, Too_

_论文源码：_[_https://github.com/THUDM/P-tuning_](https://github.com/THUDM/P-tuning)

_论文地址：_[_https://arxiv.org/pdf/2103.10385.pdf_](https://arxiv.org/pdf/2103.10385.pdf)

背景：人工构建prompt效率低，效果差，想通过自动化的构建模板而不调整模型参数。

构建连续可微的虚拟token（与prefix-tuning类似），该方法将prompt转换为可以学习的embedding，但**仅限于输入层**，并没有像prefix-tuning一样在每一层Transformer都添加。

另外还通过使用MLP+LSTM的方法对prompt embedding进行处理，加速训练。

![](/images/notion_cde7fc01-738f-44df-80b9-8560f970abc6ec199841-2c12-4b48-bfb2-e8a31b4720b3.png)

### **Prompt Tuning (2021.09)**

_论文地址：https://arxiv.org/pdf/2104.08691.pdf_

_论文题目：The Power of Scale for Parameter-Efficient Prompt Tuning_

_论文源码：https://github.com/google-research/prompt-tuning_

背景：有人提出了自动化在离散的空间中自动搜索prompt的技术，这种方法虽然优于人工设定的prompt，但是跟在连续空间搜索prompt仍有差距。

固定整个模型参数，对于不同的任务，设定不同的前缀，这些前缀token是可以更新参数的，将不同的任务数据同时输入到模型中进行训练，可以理解prompt tuning是prefix tuning的简化版本。

实验表明，随着模型参数的增加，prompt tuning的效果越来越好，但在小模型上效果不明显。

![](/images/notion_e7135fb0-4718-4876-a214-c122347950ce2d0c434e-f749-493d-815d-c59644b92411.png)

### **P-tuning-v2 (2022.03)**

_论文题目：P-Tuning v2: Prompt Tuning Can Be Comparable to Finetuning Universally Across Scales and Tasks_

_论文源码：_[_https://github.com/THUDM/P-tuning-v2_](https://github.com/THUDM/P-tuning-v2)

_论文地址：_[_https://arxiv.org/pdf/2110.07602.pdf_](https://arxiv.org/pdf/2110.07602.pdf)

背景：为了解决P-tuning和prompt tuning在小模型、跨类任务上效果不佳的问题，作者提出了P-tuning-v2

相较于P-tuning v1，P-tuning v2将连续提示应用于预训练的每一层，而不仅仅是输入层。P-tuning v2与prefix tuning类似，不同的是prefix tuning应用于NLG任务，而P-tuning v2应用于NLU任务。

通过增加prompt可调参数量（from 0.01% to 1%~3%），P-tuning v2提高了训练的性能。

![](/images/notion_52a2a745-80fd-4e89-919e-242245102564d66eea32-ca00-40f8-a68f-44855e3ce705.png)

## LoRA系列

### LoRA（2021.11）

论文题目：LoRA: Low-Rank Adaptation of Large Language Models

论文源码：[https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

论文地址：[https://arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2106.09685.pdf)

背景：当前PEFT方法中，有增加模型深度导致增加了模型推理时间的，例如Adapter，有训练Prompt，同时减少了模型可用输入的，同时Prompt训练起来也比较难，例如Prompt tuning、Prefix tuning、P-tuning，这些方法的效果都差于full-finetuning。有研究者对语言模型的参数进行研究发现，语言模型虽然参数众多，但是起到作用的还是其中低秩的本质维度（Low instrisic dimension）。

![](/images/notion_ebc37671-1c8c-4e18-abdf-77f8ce863f1d06c54517-a664-4e66-8d14-817354da433f.png)

Lora核心思想就是通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练，在涉及到矩阵相乘的模块，在原始的PLM旁边增加一个新的通路，通过前后两个矩阵A,B相乘，第一个矩阵A负责降维，第二个矩阵B负责升维，中间层维度为r，从而来模拟所谓的本征秩（intrinsic rank）。

h=W_0x+ \triangle W_x=W_0x+BAx



在训练的时候，LoRA一般只对每层的self-attention进行微调，即对W_q、W_k、W_v、W_o四个映射层进行微调，实验表明同时调整这四个映射层效果是最好的。在推理时，只需要将训练完成的矩阵BA乘积加到原始矩阵W即可，即h=Wx+BAx=(W+BA)x，不会增加额外的计算资源和推理时间。

![](/images/notion_1efe017a-6c5d-4c58-a591-bc32dd073f4a81337d4f-7c6a-4b78-abed-f607a526554e.png)

对于LoRA的秩取多大，论文中进行了实验，从实验结果来看，在秩极低（r=1）的情况下，对W_q、W_v微调就能获得与高秩相当的性能。

![](/images/notion_9a396a2f-b5d0-4773-b830-00106d97628d83f05cbb-bdd8-464f-a675-7282a4c1eb68.png)

### AdaLoRA（2023.03）

论文题目：Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning

论文源码：[https://github.com/QingruZhang/AdaLoRA](https://github.com/QingruZhang/AdaLoRA)

论文地址：[https://arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2303.10512.pdf)

背景：LoRA需要预先指定每个增量矩阵的本征秩 r 相同，在微调预训练模型时，LoRA均匀地分配增量更新的预算到所有预训练权重矩阵上，并忽视了不同权重参数的重要性差异。

所以AdaLoRA提出了动态调整增量矩阵，对于重要的增量矩阵分配比较大的r，对于不重要的增量矩阵分配比较小的r，防止过拟合，并且节省计算资源。

**以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量**。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。

![](/images/notion_cc1325df-ebe1-4507-ac8e-d14ef91046b7b9367bdb-92b5-4683-8bbb-8f94b166a3a9.png)

实验结果证明AdaLoRA效果好于LoRA，但是没好太多，个人认为方法是好方法，提升有限。

### QLoRA（2023.05）

论文题目：QLoRA: Efficient Finetuning of Quantized LLMs

论文源码：[https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)

论文地址：[https://arxiv.org/pdf/2305.14314.pdf](https://arxiv.org/pdf/2305.14314.pdf)

背景：量化方法可以显著减少LLM的内存占用，然而只限于在推理阶段，基于此，QLoRA提出了在不降低性能的前提下，微调量化为4bit的LLM。

具体操作为将预训练模型量化到4bit，并且添加可以学习的低秩适配器权重，这些权重通过量化的反向传播梯度进行优化。QLoRA 有一种低精度存储数据类型（4 bit），还有一种计算数据类型（BFloat16）。实际上，这意味着无论何时使用 QLoRA 权重张量，我们都会将张量反量化为 BFloat16，然后执行 16 位矩阵乘法。QLoRA提出的一些新技术来实现了上述操作，具体为：

- **4bit NormalFloat(NF4)**：对于正态分布权重而言，一种信息理论上最优的新数据类型，该数据类型对正态分布数据产生比 4 bit整数和 4bit 浮点数更好的实证结果。
- **双量化**：对第一次量化后的那些常量再进行一次量化，减少存储空间。
- **分页优化器**：使用NVIDIA统一内存特性，该特性可以在在GPU偶尔OOM的情况下，进行CPU和GPU之间自动分页到分页的传输，以实现无错误的 GPU 处理。该功能的工作方式类似于 CPU 内存和磁盘之间的常规内存分页。使用此功能为优化器状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新步骤需要时将其加载回 GPU 内存。
![](/images/notion_5b7a788a-7e52-4200-a27b-a0223749f092062a3d3e-c3c4-4994-b310-c0c6537a4844.png)

## 总结

> 📌 P-tuning v1与Prompt tuning是比较类似的方法

P-tuning v1和Prompt tuning都是在**输入层**添加连续的虚拟token；

P-tuning v1插入虚拟token的位置可以是前缀，也可以插入到中间；

Prompt tuning是针对不同的任务设定不同的prompt，将prompt与特定任务的数据进行拼接作为输入；

P-tuning v1将prompt使用MLP+LSTM的方式对prompt进行embedding处理，这里起到的作用的加速训练，Prompt tuning不需要使用MLP进行向量化；

> 📌 P-tuning v2与Prefix tuning是比较类似的方法

Prefix tuning和P-tuning v2都是在**transformer的每一层**添加虚拟token；

Prefix tuning应用于NLG任务，而P-tuning v2应用于NLU任务;

Prefix tuning可以应用在decoder-only模型，也可以应用在encoder-decoder模型，P-tuning v2只能应用在decoder-only模型；

> 📌 LoRA系列

LoRA通过引入低秩的旁路网络（增量矩阵）可获得与全量微调相当的性能，且极大降低训练显存依赖（将GPT-3可调参数减少10000倍，GPU内存需求减少3倍）；LoRA需要预先设置相同增量矩阵的秩，忽略了不同权重参数的重要性差异，AdaLora对这一问题进行改进，通过引入SVD技术，达到对参数矩阵自适应分配秩的效果，从而获得了相较Lora更优的性能；与AdaLoRA思路不同，QLoRA在LoRA基础上，对大模型基座进行4-bit量化，同时引入双量化和分页优化，极大减少训练显存依赖，且效果与16-bit全量微调相当，QLoRA将65B Llama模型的显存需求从大于780G降低到小于48G，AdaLoRA和QLoRA可看作是对LoRA不同的两种改进方式。



## 参考文献

[https://mp.weixin.qq.com/s/E_0-skD3__w5jLGEJlDpoA](https://mp.weixin.qq.com/s/E_0-skD3__w5jLGEJlDpoA)

[https://mp.weixin.qq.com/s/webUB5j8nNQsthTFQNiqpA](https://mp.weixin.qq.com/s/webUB5j8nNQsthTFQNiqpA)

[https://mp.weixin.qq.com/s/8A7aLiknSDCBfMuUKg4eiw](https://mp.weixin.qq.com/s/8A7aLiknSDCBfMuUKg4eiw)

[https://zhuanlan.zhihu.com/p/636215898](https://zhuanlan.zhihu.com/p/636215898)



