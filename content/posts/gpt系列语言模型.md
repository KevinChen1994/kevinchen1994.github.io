---
{
  "title": "GPT系列语言模型",
  "date": "2023-08-16",
  "tags": [
    "GPT"
  ],
  "categories": [
    "LLM"
  ],
  "summary": "科普，无技术技术分享，像讲故事一样根据时间线来介绍GPT1到GPT4。",
  "authors": [],
  "draft": false,
  "generated_time": "2025-02-24T12:16:00.705630"
}
---

## 前言

最近在部门做了一次GPT系列语言模型的分享，这次分享受众很大一部分没有算法背景，所以这篇文章想用讲故事的方式给大家介绍GPT系列语言模型，可以称为无技术技术分享。

想必大家最近都被ChatGPT刷屏了吧，那为什么ChatGPT这么智能呢？其实最主要的还是因为他的底座模型很强大，所谓RLHF都是为了让底座模型学习到人类的交互方式，让它更好的相应人类的指令。那接下来我会沿着GPT发布的时间顺序分别给大家介绍一下这几个模型。

![图1 语言模型发布时间线](/images/notion_23a6129b-3b3e-41f9-b674-a9330fbf1b41a8f7f1b0-2509-45a9-82a8-da0f4d743256.png)

## Transformer 梦开始的地方

首先我们先简单介绍一下Transformer，因为GPT中的T就是指Transformer，这是大模型梦开始的地方。

2017年，Google发表了一篇论文《Attention is all you need》，这篇文章主要讲他们设计了一个新的网络结构，以往我们使用的网络结构基本就是卷积神经网络（CNN）、循环神经网络（RNN），而Transformer只使用了self-attention机制，基于seq2seq结构设计了一种新的网络。

所谓seq2seq结构就是网络具有编码器和解码器，编码器就是将我们的输入进行编码，提取特征，而解码器就是对编码进行解析，输出结果，如果是做中英翻译任务的话，输入就是中文，然后编码器对中文进行编码，解码器会对编码进行解码，输出就是对应的英文。这是整个Transformer的结构图，左边是编码器，右边是解码器。

![图2 Transformer结构图](/images/notion_a8f3187a-d233-40c9-984e-303b3e075a4d3c5ae160-94ed-4a2f-858d-1a4857533b9f.png)

什么是 self-attention 呢，翻译过来就是自注意力，自己跟自己算 attention，具体的做法就是将输入转换成三个向量，分别是K、Q、V，然后K与Q进行点乘、缩放，再经过 softmax 以后得到一个权重矩阵，在与V相乘，V拿到的就是加权后的向量，也就是他该关注哪里，不该关注哪里都计算好了。这样的好处就是可以忽略文本之间的距离问题，直接计算全局的特征。举一个简单的例子，一段文本中说：小明是一个好学生，他上课认真听讲，下课好好做作业，然后巴拉巴拉说了一堆，最后说他真值得我们学习。这句话在计算self-attention 的时候，小明这个名字就与最后的他这个字关联性很高，即使他们之间的距离很远，Transformer也关注到了他就是小明，如果使用的是RNN，这么远的距离可能会遗忘掉小明这个名字。

所谓多头自注意力，就是我们有多对KQV，他们会分别进行attention，之间互不干扰，这样我们就能拿到多种特征，从而提高了模型的特征抽取能力，并且多头自注意力之间还是并行计算的，也就是能够更好的利用 GPU 资源，提高我们的计算效率。

![图3 自注意力机制和多头自注意力机制](/images/notion_983d97c9-bc65-4d48-8c7e-a0dca899251093f9310e-784a-4ed2-b810-da5c2f86f6e4.png)

具体细节大家不需要过多关注，只需要了解Transformer相比于之前的CNN和RNN，不仅特征抽取能力强、还并行度高，还能更好的关注全局信息，减少学到后边忘了前边的这种错误，解决了长距离依赖的问题。

基于这些能力，使我们训练像GPT和BERT这种大模型成为了可能。

## BERT与GPT1 既生瑜何生亮

在2018年openAI发布了GPT1，采用的方法是使用Transformer-decoder的架构，也就是只使用了Transformer的解码器，可以通过左边的图看出来，GPT1的训练过程是单向的，因为GPT1的训练方法是单字接龙，也就是根据上文预测下文，从这个例子大家可以看到，通过输入的招字，GPT1预测聘字，然后通过招聘预测饿字，以此类推直到预测结束。通过这种训练方式，GPT1能够学习到文本之间的语义知识。一般我们称这种模型为自回归模型，他更适用于生成类的任务。

紧接着Google发布了BERT，大家对BERT可能或多或少都听说过，我们很多工作都是基于BERT去做的，BERT是基于Transformer-encoder架构设计的，也就是只使用了Transformer的编码器，从图中可以看出来，BERT的训练过程是双向的。因为encoder可以看到输入文本上下文的全部信息，所以需要在训练时mask掉一些字，因此BERT的训练方法为完形填空，让模型根据上下文进行预测mask掉的字。一般我们称这种模型为自编码模型，他在自然语言理解任务上会有不错的效果。

![图4 BERT与GPT1模型结构对比](/images/notion_bf5fec66-e26f-4eab-a8f2-c77986d584d78c392985-ebf8-4359-b9eb-46054973a14d.png)

![图5 BERT与GPT1训练方式对比](/images/notion_daa987f4-28e9-43c8-812d-1ebc130cafba17efa5e9-49b8-47de-896c-322f87b689c7.png)

通过对比我们可以看到两个模型设计是很像的，只不过互为对立面，一个使用的是encoder，一个使用的是decoder。我们再来看一下两个模型的参数，由于GPT1是先发布的，所以BERT完全按照GPT1的参数量进行了设计，都是12层transformer，768维的隐藏层和12个自注意力头。那这两个模型的参数量几乎一致的，他们俩当时都刷新了各大NLP任务的榜单，所以我给这一节的标题起名为既生瑜何生亮。

![图6 BERT与GPT1参数对比](/images/notion_a13681ab-6f6c-41af-aedf-41c05ff8a3c6539fecc7-8111-4846-aa76-3b4409669d7d.png)

GPT1与BERT的使用方法都分为两个阶段，第一阶段都是在大规模无标注语料上进行预训练，也就是pre-training，就是我们刚才提到的单字接龙和完形填空，这阶段的目的就是通过预训练获取无标注语料中的语义知识。这一步Google和openAI都替我们训练好了，我们可以直接下载他们开源的模型，第二阶段是根据不同的下游任务，使用我们自己的有标注的数据进行有监督训练，也就是fine-tuning。这样的做的目的就是利用在预训练阶段模型学习到的语义知识，通过fine-tuning把这部分知识迁移到下游任务中。

下图为BERT的使用方法，根据不同的任务，在BERT后添加不同的网络层就可以了，在输入部分通过添加特殊标识符[CLS]表示句子的向量，如果是双句任务，通过[SEP]来分割两个句子。例如做分类任务，直接拿到BERT的输出，添加一个分类层就可以；序列标注任务中，我们可以拿到每一个字的输出，在通过分类器进行判断标签即可。

![图6 BERT使用方法](/images/notion_185affea-4215-4d69-b39b-58049b72068376396371-b8f3-4381-861f-321af06f54bb.png)

下图为GPT1的使用方法，使用方法也很类似，通过特殊标识符标识整个句子，双句任务按照特殊标识符进行分割，输入给模型，在拿到模型的输出后，拼接一层线性层就可以拿到预测结果。

![图7 GPT1使用方法](/images/notion_8436a5b9-1d26-4351-9060-586686cd84b77be7405a-aa13-45b8-8f29-9d98dfc60967.png)

我们来看一下实验结果。

BERT-base为与GPT1参数规模相同的模型，可以看到在多项NLP任务中，他的成绩都优于GPT1。

GPT1的效果比之前的方法都要好，但是由于BERT的训练方法为完形填空，所以学习到的语义知识更多，因此在下游任务中的表现都是优于GPT1的。所以在此以后，BERT大火了起来，成为了我们最强的baseline，无论什么任务，我们优先都会去尝试BERT的效果怎么样，我们一度以为BERT才是未来的方向，当然后面的故事大家都清楚了。

![图8 BERT与GPT1实验结果](/images/notion_fc9a482c-9df0-477b-8ae7-d3acd8dd38681ceda2a3-63d7-4567-8036-d3951273f29f.png)

## GPT2 无监督学习者

在BERT发布以后，openAI痛定思痛，决定继续深挖自回归模型的潜力，于是在2019年GPT2发布了，论文名称为Language Models are Unsupervised Multitask Learners，语言模型是无监督的多任务学习者。主打的是Zero-Shot learning，所谓Zero-Shot learning就是指在预训练完成后，不需要任何标注的数据，直接对下游任务进行预测。openAI这次想体现的是模型泛化能力强。

想要做的Zero-Shot，那前提就是模型一定要强，所以openAI加大了参数量和训练数据，参数量提升到了1.5B，也就是15亿的参数，相比于GPT1和BERT提升了13倍多，训练数据也由之前的4GB提升到了40GB。通过实验对比，在Zero-Shot的设定下，GPT2在多个任务上能够吊打绝大部分模型，当然也有一些任务表现没有那么好。

![图9 GPT2在zero-shot任务上的表现](/images/notion_fbaca336-df89-4588-a623-6dd8b18b5e0c9d434d51-ec7e-4afa-8805-4d722c579e5a.png)

使用生成式的模型如何进行预测呢？在GPT1中因为使用了微调的方法，所以在输入文本中添加了一些特殊符号，这样模型可以记住这些符号，而在GPT2中，我们直接进行预测，并没有这些特殊符号，所以使用方式为直接写出需要模型做的任务，模型自动识别并输出结果，翻译任务：(translate to french, english text, french text)；阅读理解任务：(answer the question, document, question, answer)。这种方式也称为prompt，也就是提示模型他需要做什么，prompt的目的其实就是为了激发模型的补全能力，我们的prompt作为模型的输入，让模型输出我们想要的东西。后来这种方式成为了大模型主要的使用方法。

## GPT3 大模型的顿悟时刻

实际使用场景下，大家其实都能提供一些标注数据，人类在学习的时候有一些样本学习的也会很好，于是openAI继续深挖，发布了GPT3，论文名称为Language Models are Few-Shot Learners。所谓Few-Shot就是指训练完成后，给我少量的标注数据，但是不进行微调，只是让我看几个例子，我就能帮你做下游任务。

这次GPT3的参数量到了175B，也就是1750亿参数，相较于GPT2提升了100多倍，训练数据到了570GB。在这也样数据量和参数量下，GPT3展示了什么叫大力出奇迹。除了在很多下游任务上有不错的表现以外，还解锁了其他的能力。

### In-Context Learning

什么是In-Context Learning呢，简单来说，就是模型在不更新自身参数的情况下，通过在模型输入中带入新任务的描述与少量的样本，就能让模型”学习”到新任务的特征，并且对新任务中的样本产生不错的预测效果。这种能力可以当做是一种小样本学习能力。可以参考下图的例子来理解：在预测阶段，在输入中输入几个任务样例，这样模型就能在不调整模型参数的情况下学习到该任务需要解决什么问题，你再问类似的问题，模型就会输出正确的答案。

![图10 In-Context Learning](/images/notion_17be8f2a-6ff4-4308-aee4-dbee8083e90b30eea9a3-e501-40ce-a953-ef07fdfe5b6f.png)

我们可以看一下实验结果，In-Context Learning在大模型上随着标注样本的增多，最多也就几十条，准确率能够稳定提升，这里大模型指千亿参数以上，在小模型上虽然也有提升，但是效果没有大模型效果明显，所以这个能力是大模型独有的，所以这里我们称其为大模型的顿悟时刻。

![图11 不同大小模型的In-Context-Learning的能力](/images/notion_e7dfccec-1b63-4424-9e1c-344b0a147e151f5c5dbb-5839-40b4-b874-c0f3bf61eeab.png)

### chain of thought

我们再来讲一下大模型解锁的另外一个能力，chain of thought，也就是思维链。在一般的prompt中，对于一个复杂或者需要多步计算的问题，如果你直接问模型，他大概率会直接给你一个错误答案，例如左下角这个例子。如果你给出样例，但是没有提供推导过程，模型也很难给出正确答案，例如左上角这个例子，绿色是给了一个样例，然而在继续问一个新的问题的时候，模型直接输出了结果，但是结果错误了。而右上角这个图中展示了，我给了你一个样例，还给了你推导过程，再问你一个新问题的时候，模型不仅能输出解题过程，还能计算正确。这就是思维链的能力，让模型具有了像人类处理复杂问题一样的思维能力。

甚至还有人提出了一种方法，在问问题的时候，不告诉模型推导过程，只是跟模型说一句，Let’s think step by step，模型就能给出推导过程和正确答案，这妥妥的模型鼓励师了，鼓励模型不要着急，一步一步想，模型就能给出正确的答案。

![图12 chain of thought](/images/notion_95e47b20-8866-4943-8049-4ed1744dda8419264f3d-9727-4779-95bc-526b8cbe3fb4.png)

## 能力涌现

In-Context Learning 和 chain of thought 的能力是怎么来的呢？目前学术界也没有确定的答案，现在大家称这种现象为大模型的Emergent Abilities，也就是能力涌现。一般模型的大小和模型的能力是有固定比例的，一般是称线性关系，即模型越大能力越强，但是当模型的规模到了一定的程度以后，像是GPT3、chatGPT这种千亿级别的大模型，模型的能力会有一个质的变化，也就是能力涌现。

可以通过实验结果图来看一下，横轴为模型的计算量，我们可以理解为模型参数量越大，模型的计算量也就越大，纵轴为模型在不同任务上的表现。随着模型规模变化，模型的能力开始是一点点提升的，有的任务甚至是一条水平线，但是当模型的规模到了一定的程度，模型的能力突然就提升了很多。这也许就是大力出奇迹吧。

![图13 Emergent Abilities](/images/notion_a4cd6319-df46-408e-ae00-46364ab6c3b80d9bc725-4591-4150-813c-9638b7ec7f30.png)

## GPT3.5 守得云开见月明

GPT3已经能帮我们做很多事情了，但远没有到chatGPT那么强，我们知道chatGPT是基于GPT3.5进行训练得来的，chatGPT之所以这么强，是因为他有一个强有力的底座模型，也就是GPT的守得云开见月明的时刻。GPT3.5并不单单指一个模型，而是一系列的模型，通过openAI官网我们可以看到，这一系列模型分别擅长不同的任务，有的善于理解代码，有的善于执行人类命令，有的善于聊天。

![图14 GPT 3.5系列模型](/images/notion_2a227b8d-172f-43c2-aa18-fe0c8866d6f047d4bc8e-1e87-4b62-bc20-9c37fa343b65.png)

目前openAI没有公开GPT3.5具体是多大规模和怎么训练的，不过通过模型擅长的任务，GPT3.5可能不仅仅增大了模型参数量和训练数据量，还加入了代码的训练数据，并且加入了instruction tuning。

加入代码训练数据我们很好理解，让模型能够读懂代码，能够帮助我们生成代码和修改代码bug，还有一些研究表明在加入了代码训练以后，模型的逻辑推理能力能够有效的增强，原因可能是因为代码都是逻辑比较清晰的，模型通过学习代码，从而增强了自己的逻辑推理能力。

那什么是instruction tuning呢？我们先来回顾一下BERT、GPT1的使用方法，也就是fine-tuning，针对不同的下游任务，使用不同的标注数据对模型进行微调，此时模型的参数是进行调整的。

在GPT2和GPT3中，因为模型参数量太大了，一般人也没有那么多的计算资源去优化，也很有可能越训练模型效果越差，所以模型的参数是不允许被修改的，大家是通过prompt也就是提示去命令模型帮助我们解决问题，prompt的目的其实就是为了激发模型的补全能力，我们的prompt作为模型的输入，让模型输出我们想要的东西。

instruction tuning直接翻译就是指令学习，是指使用将有标注的数据集使用自然语言描述的方式对模型参数进行微调，可以使用多个不同任务的数据集对模型进行指令学习，这样做的目的不是为了让模型学习到标注数据中的知识，因为在预训练阶段模型已经学习到了充足的知识，这样做是为了让模型能够更好的响应人类的指令，从而能做出正确的反馈。

![图15 instruction tuning](/images/notion_4aa702b8-0e47-4d64-8fbf-aeb42be6d06ccd3275a5-4d7e-4d53-948e-36b518e13eff.png)

当然GPT3.5还有使用到instructGPT的训练方法和强化学习的训练方法，这里我们不做介绍。

## GPT4 更高更快更强

就在3月15号，openAI发布了GPT4，这里我们需要弄清楚GPT4与之前的模型都不一样，因为GPT4不单单指一个模型了，他对标的是chatGPT，是一套系统，相较于chatGPT他更加强大和安全了。

![图16 GPT4](/images/notion_a8a62905-c781-4656-9aa7-9d3c008182dae1d1b71e-0df3-43f5-b5f9-3a7f4098dbbd.png)

由于openAI没有公布GPT4具体的参数和训练方法，所以这里我们简单介绍一下GPT4强大在哪里。

首先是多模态，GPT4支持文本和图片的输入，输出只支持文本，你可以通过输入图片让他解释这张图里都有什么，也可以给他一张搞笑图片让他给你解释为什么好笑，这里体现了GPT4强大的常识知识，也就是他会有类似人类的认知，知道一张搞笑图片搞笑在哪里。

再就是更强大的编程能力，通过发布会的视频，我们看到了一个例子，演示者通过手绘了一张网页的草图输入给GPT4，GPT4很快就生成了这个网页的源代码，拿过来可以直接进行部署。他还支持你描述一个游戏的玩法，他去给你生成游戏的源代码。

GPT4支持更长的输入，chatGPT最长是4000个token，当你们之间的对话长度超过这个值，他就会丢失一些上文的信息，从而输出的内容可能会不符合你的要求，GPT4直接升级到了3.2万个token，大约25000个字，所以能更好的支持对轮对话的内容。

GPT4有了更强大的处理复杂问题的能力，在普通的对话中可能体现不出来GPT4与chatGPT之间的区别，但是在复杂问题上，比如数学问题，或者需要多步推理的问题，GPT4的能力是显著优于chatGPT的。

openAI还用GPT4参加了人类的考试，在美国的高考、律师从业资格证都能取得前10%的成绩，而chatGPT参加这些考试的成绩在后10%-30%。

GPT4的多语言能力也得到了提升，除了英语，在一些小语种中都是优于chatGPT的。这里说一下我个人的之前的一个错误观点，因为在GPT3的训练数据中中文的占比仅为0.1%，我猜测chatGPT中中文的训练数据也不会太多，所以我一直以为在处理非英语问答时，chatGPT内部是不是翻译成了英文再去处理的，后来听了微软亚洲研究院周明博士的一个观点我觉得可能是正确的，他说世界的语言之间其实都是可以对齐的，模型在接受多语言训练数据时，他内部能够对齐不同语言之间的关系，也就是他即使不用翻译也能理解各种语言是什么意思，这可能就是GPT4为什么能很好处理多语言的原因吧。

最后就是openAI一直强调的，GPT4更加安全可控了，GPT4在去年8月份就训练好了，在今年3月份才发布，用一个比喻来说就是openAI这半年一直在驯服这头能力强大的野兽。他们用了基于人类反馈的强化学习让模型输出更符合人类要求的内容，尽量避免有毒的、性别歧视、种族歧视、黄暴等内容的生成。

## 总结 openAI不忘初心

我们总结一下吧，通过下面的表格，我们可以更清晰的看到这些模型之间的区别和他们这几年的发展路线。我们可以看到openAI一直沿着他们最初的想法在做，不断把模型做深做大，用到的数据越来越多，训练资源越来越大，当然他们的效果也越来越好，不断在打破我们的认知，希望我们国内也能早点做出能与之抗行的模型。

最后请把不忘初心打在公屏上。

![图17 总结](/images/notion_a4878454-c242-441e-8be9-c8392c4b1f262730b1da-43a5-453f-a367-b98acf3d9590.png)

## 参考文献

[1] [Vaswani A , Shazeer N , Parmar N , et al. Attention Is All You Need[J]. 2017.](http://link.zhihu.com/?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.03762.pdf)

[2] [Devlin J , Chang M W , Lee K , et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[J]. 2018.](http://link.zhihu.com/?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1810.04805.pdf)

[3] [Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.](http://link.zhihu.com/?target=https%3A%2F%2Fwww.cs.ubc.ca%2F~amuham01%2FLING530%2Fpapers%2Fradford2018improving.pdf)

[4] [Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf)

[5] [Brown T, Mann B, Ryder N, et al. Language models are few-shot learners[J]. Advances in neural information processing systems, 2020, 33: 1877-1901.](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)

[6] [Kojima T, Gu S S, Reid M, et al. Large language models are zero-shot reasoners[J]. arXiv preprint arXiv:2205.11916, 2022.](https://arxiv.org/abs/2205.11916)

[7] [Wei J, Tay Y, Bommasani R, et al. Emergent abilities of large language models[J]. arXiv preprint arXiv:2206.07682, 2022.](https://arxiv.org/abs/2206.07682)

[8] [Wei J, Bosma M, Zhao V Y, et al. Finetuned language models are zero-shot learners[J]. arXiv preprint arXiv:2109.01652, 2021.](https://arxiv.org/abs/2109.01652)

[9] [openAI GPT4](https://openai.com/product/gpt-4)

[10] [Fu, Yao; Peng, Hao and Khot, Tushar. 拆解追溯 GPT-3.5 各项能力的起源. 2022](/360081d91ec245f29029d37b54573756)



