# Notion作为CRM，自动获取数据生成博客
本项目可以将Notion中的内容自动生成博客，然后发布到GitHub Pages。
## 配置
### 基于当前仓库创建一个新仓库
![复制该仓库](/resources/github_template.png)
### 创建Notion integration
在Notion的integration中创建一个新的integration，需要登录Notion账号。
![创建Notion integration](/resources/notion_integrations.png)
勾选权限，需要勾选读取内容的权限，因为需要通过该integration来获取Notion中的内容。
![Notion integration config](/resources/notion_integration_config.png)
保存该integration，会生成一个secret token，需要记录下来。
![Notion integration config](/resources/notion_integration_secret.png)
本地测试的时候可以通过config.py来配置该secret token。
在GitHub中需要再Settings->Secrets and variables->Actions中添加该secret token。注意命名要统一，按照下图的方式添加。
![GitHub Secret](/resources/github_secret.png)
复制这个[Notion CRM数据库模板](https://www.notion.so/kevinchen1994/1a5f9fa1519f80b881b4ddbbd326320c?v=1a5f9fa1519f818e82ea000c2836b12f&pvs=4)到你的Notion，然后打开你Notion中的这个页面，点击右上角的省略号，添加connections，选择刚才创建的integration。
![Notion connection](/resources/notion_connection.png)
点击Share，然后点击Copy link，复制该链接，链接长这样：https://www.notion.so/kevinchen1994/1a5f9fa1519f80b881b4ddbbd326320c?v=1a5f9fa1519f818e82ea000c2836b12f&pvs=4，其中1a5f9fa1519f80b881b4ddbbd326320c是databaseID，复制这个值（请使用你自己的ID，不要使用这个ID），使用同样的方法配置GitHub Secret中。
![Github databaseID](/resources/databaseID.png)
配置完成后，就可以进行测试是否可以运行。可以在本地运行，执行.github/workflows中的脚本，或者在GitHub Action中手动执行。
在GitHub Action中我配置的是每天零点自动运行一次，你也可以自己修改，或者选择手动执行。
```bash
  schedule:
    - cron: '0 0 * * *'  # 每天0点运行
  workflow_dispatch:     # 支持手动触发
```
## 选择Hugo主题
去Hugo的主题库中选择一个主题，然后将该主题的代码复制到该仓库的themes目录下。
可以使用 git submodule 来管理主题代码。
## 本地测试
通过```hugo server```来启动本地服务器，然后在浏览器中访问URL_ADDRESS:PORT，即可看到本地的博客。
## 感谢
本项目灵感来自于[Notion-Hugo](https://github.com/HEIGE-PCloud/Notion-Hugo)，我自己用了挺长时间，决定使用Python复现该项目，本项目实现的功能略微比Notion-Hugo少，比如少一些格式的支持、自定义页面等，不过对于个人来说已经够用了。