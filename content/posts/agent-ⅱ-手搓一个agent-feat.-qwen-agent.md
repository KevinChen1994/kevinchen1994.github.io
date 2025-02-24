---
{
  "title": "Agent Ⅱ 手搓一个Agent feat. Qwen-Agent",
  "date": "2024-06-21",
  "tags": [
    "Agent"
  ],
  "categories": [
    "LLM"
  ],
  "summary": "",
  "authors": [],
  "draft": false,
  "generated_time": "2025-02-24T12:15:02.296493"
}
---

## 前言

上一篇文章介绍了关于Agent的一些相关内容，这篇文章来动手实现一个Agent。目前比较流行的Agent开发框架有很多，比如可视化的字节的扣子，还有代码实现的babyagi、AutoGPT，因为我用qwen用的比较多，正好我最近看到qwen也开源了一套框架，qwen-agents，封装了工具、助手和一些规划的方法，使用起来比较方便，那就用这套框架实现一个简单的Agent。

## Agent设计开发

既然是Agent，那就让他做一个LLM不能直接实现的任务，我这边有一个北京的房价表格，我们可以让Agent帮我们分析一下房价，最后生成一个图标来展示每平米的价格。

因为这个任务需要多步骤才能实现，所以我们可以使用上篇文章中将的ReAct的思路，让模型自己规划任务，根据每一次的执行结果来调整任务，并且可以利用代码助手来帮助他实现目标。

先配置LLM，这里可以使用阿里的api，也可以使用第三方的api，只要与openAI库兼容就可以。我这里尝试了付费的qwen-max和免费的Qwen2-7B-Instruct。

```python
llm_cfg = {
        # 使用 DashScope 提供的模型服务：
        # 'model': 'qwen-max',
        # 'model_server': 'dashscope',
        # 'api_key': 'EMPTY',
        # 如果这里没有设置 'api_key'，它将读取 `DASHSCOPE_API_KEY` 环境变量。

        # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
        'model': 'alibaba/Qwen2-7B-Instruct',
        'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
        'api_key': 'EMPTY',

        # （可选） LLM 的超参数：
        'generate_cfg': {
            'top_p': 0.8
        }
    }
```

然后去定义我们的Agent，因为框架封装的很方便了，所以几行代码就可以实现。

可以看到我们只需要配置Agent能够使用的工具，以及Agent的名称，他能做什么就可以了。工具方面我这边只使用了代码解释器，框架还封装了很多其他的工具，比如联网搜索、图片生成（需要api支持）、RAG等。框架也支持自定义工具，可以根据自己的需要实现特定功能的工具。

```python
def init_agent_service():
    llm_cfg = {
    }
    tools = ['code_interpreter']
    bot = agents.ReActChat(llm=llm_cfg,
                           name='house price analyzer',
                           description='You are a house price analyzer, you can run code to analyze house price.',
                           function_list=tools)
    return bot
```

然后就可以配置测试的方法了，可以通过命令行进行测试，也可以通过GUI进行测试，框架基于gradio开发了很方便的GUI方法，我就选择GUI来进行测试。

这里可以提前写好建议的prompt和需要上传的文件，也可以在页面自己手动填写和上传。

```python
def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [{
            'text': '先使用pd.head()查看数据，然后帮我分析昌平区两居室的每平米均价，最后帮我画一个关于昌平区两居室房子面积与价格的折线图',
            'files': [os.path.join(ROOT_RESOURCE, 'lianjia.xlsx')]
        }]
    }
    gui.WebUI(bot, chatbot_config=chatbot_config).run()
```

最后运行程序就可以了。

## Agent实际效果

这里可以看到他根据我的指令，进行了思考，他需要先加载文件，然后通过pandas来查看数据，然后根据我的要求进行筛序，最后进行计算和画图。

接着他实现了代码，代码逻辑看上去没有啥问题，有一处小问题，那就是两居室可能不能使用等于2室来表示，而是包含2室，我们看看运行结果怎么样。

![](/images/notion_3e3bdccb-b610-4f5b-9a99-cc4386368e470a5d54ec-a589-4509-b198-1000ca4d87cb.png)

![](/images/notion_368dd949-5d20-432a-aa81-375deb2bc5dbac8ab89b-633f-4919-bfa6-2d86d28ae82b.png)

代码运行出错了，因为数据中并没有区域这一列，实际是Region，也没有居室这样的列名，接着他通过pandas查看了列名来确认具体的列名是什么。

![](/images/notion_761f4c9f-f8d5-433f-bd1a-34888710afb43b718e77-1946-4ac3-a089-866d3a754068.png)

这次的代码没有问题了，字段使用都是正确的，两居室也是使用contains来表示。

![](/images/notion_2432196c-2d0e-4bb5-9eb4-383a38aedeffc090d249-4a2e-42b7-8bb1-b469afd6f8c5.png)

每平米房价跟房屋面积的图画出来了，看上去不错。

![](/images/notion_12a98c02-01df-4399-8cfe-b820d322dbb0356e3fa9-a7a7-4c14-97cc-a1053e50368c.png)

![](/images/notion_91dfcf96-4f71-4fa8-9172-4f71997c481b1cd562c0-28a7-486f-97c3-9f4878b101e2.png)

尽管代码报错了，但是最终得到了答案，最后还给出了一些分析，很好的完成了我的任务。

![](/images/notion_4daa91bf-d768-4153-b0b9-60b767a8056e95839a71-f818-4faa-83e2-c4d89d13fb1c.png)

## 总结

房价分析Agent很好的完成了我的任务，尽管过程中有出现一些错误，但是经过ReAct的这种观测、思考最终还是顺利的完成了。

其实实际运行中，Agent并不是一次性就完成了我的任务，中间运行了好多次都是错误的结果，要么是代码写的有问题，要么就是观测的结果与思考有问题，所以想要Agent完全代替人的工作为时尚早。

Agent相比于LLM来说，确实更加智能，但是其中需要人根据具体的任务去设计Agent，如果是复杂的任务，可能需要多个Agent来协作，这就涉及到了上篇文章讲到的挑战了，复杂的任务通常需要很多步骤才能完成，那LLM的上下文长度就会成为瓶颈。根据这次的经验，LLM通过自然语言作为接口来进行交互难免出错，所以还是需要提升LLM的能力，让其成为一个合格的大脑。

