---
{
  "title": "VS Code使用记录",
  "date": "2023-08-21",
  "tags": [
    "VS Code"
  ],
  "categories": [
    "Tools"
  ],
  "summary": "由PyCharm转到VS Code，记录一些使用方法以及在使用过程中遇到的问题，持续更新。",
  "authors": [],
  "draft": false,
  "generated_time": "2026-02-12T09:56:16.619885"
}
---

开发Python一直是在用PyCharm，对VS Code了解不太多，不过在一些视频里见过不少大神是在用VS Code的。PyCharm很强，能满足我所有的需求，有两点我不太满意的点，一是需要续费，或者破解（不支持这种做法），二是有点笨重，不够轻量化。于是我开始尝试免费、开源、轻量化的VS Code，上手还好，基本看一些介绍基本功能都了解的差不多了，可能有些快捷键需要适应一下。VS Code的插件很丰富，使用起来还是很方便的，最重要的是真的轻量级，使用起来就很干净的感觉，同时我把配置云端同步，保证家里的开发环境跟公司的保持一致，这点很棒。

这篇文章主要记录我使用过程中遇到的一些问题，使用过程中遇到新的问题会及时同步到这里。

## 常用快捷键

移动行：option+↑↓

复制当前行：shift+option+↑↓

删除当前行：shift+command+k

代码格式化：shift+option+f（推荐使用ruff插件）

对import进行排序：shift+optin+o（前提是安装isort插件）

新建一个窗口（不是新建文件，是新开一个VS Code）：shift+command+n

## 导入上一级目录的包

```python
project/
    common
        common_script.py
    main_script.py
    utils/
        helper_module.py
```

这种方法会报错，找不到common这个包

```python
# 如果我想在helper_module.py使用common下的common_script.py
from common import common_script
```

所以需要将上级目录配置到sys.path中

```python
# 如果我想在helper_module.py使用common下的common_script.py
import sys
sys.path.append('..')
from common import common_script
```

这种方法在任何地方都适用，但是在VS Code中仍会报错，这是因为需要找到根目录的包，VS Code并没有进行配置，需要在settings.json中进行配置，添加。

```json
} 
  "terminal.integrated.env.osx": {
    "PYTHONPATH": "${workspaceFolder}",
  },
  "terminal.integrated.env.linux": {
      "PYTHONPATH": "${workspaceFolder}",
  },
  "terminal.integrated.env.windows": {
      "PYTHONPATH": "${workspaceFolder}",
  },
  "code-runner.runInTerminal": true,
  "python.terminal.executeInFileDir": true,
}
```

如果是debug模型，文件运行默认是根目录，还是有可能找不到相对路径，那就需要在launch.json中进行配置，将cwd配置当前路径。

```json
{
  "name": "Python: 当前文件",
  "type": "python",
  "request": "launch",
  "program": "${file}",
  "console": "integratedTerminal",
  "stopOnEntry": true,
  "cwd": "${fileDirname}"
}
```

## 使用参数进行debug

需要在运行-配置中进行配置，配置文件为launch.json

如果想调试的文件为根目录下的test3.py

name为你调试配置的名称，理论上来说不需要按照下边这么写，取个名区分的名字就行；

program是要调试的文件，如果只有一个文件的话，默认${file}即可，就可以运行当前文件；

args是我们的配置，数据结构为一个list，在里边填上参数名称和参数值；

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "${workspace}/test3.py",
            "type": "python",
            "request": "launch",
            // "program": "${file}",
            "program": "test3.py",
            "console": "integratedTerminal",
            "args": [
                "--device", "cpu",
                "--output_path", "../test3"
            ]
        }
    ]
}
```

如果要调试的为多个文件，也需要在同一个launch.json中进行配置

这时候只需要把name定义好，program分别写文件的相对路径即可。想运行哪个文件行，在当前文件窗口界面点击debug选择对应的name就行。

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "${workspace}/test3.py",
            "type": "python",
            "request": "launch",
            // "program": "${file}",
            "program": "test3.py",
            "console": "integratedTerminal",
            "args": [
                "--device", "cpu",
                "--output_path", "../test3"
            ]
        },
        {
            "name": "${workspace}/test2.py",
            "type": "python",
            "request": "launch",
            // "program": "${file}",
            "program": "test2.py",
            "console": "integratedTerminal",
            "args": [
                "--device", "cpu",
                "--output_path", "../test2"
            ]
        }
    ]
}
```

## 创建新文件后生成文件头

通过首选项中的用户代码片段实现，选择python文件后，在配置文件中添加想要的代码头。

```json
{
	// Place your snippets for python here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	// "Print to console": {
	// 	"prefix": "log",
	// 	"body": [
	// 		"console.log('$1');",
	// 		"$2"
	// 	],
	// 	"description": "Log output to console"
	// }
	"HEADER":{"prefix": "header",
        "body": [
        "# -*- coding: utf-8 -*-",
        "'''",
        "@File    :   $TM_FILENAME",
        "@Time    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND",
        "@Author  :   chenmeng",
        "@Desc    :   None",
        "'''",
        ""
    ],

  }
}
```

配置好以后，创建新文件后，输入header，会自动生成配置好的文件头。

## 自动生成python main方法

通过首选项中的用户代码片段实现，选择python文件后，添加对应配置即可。也可以通过插件实现，不过我配置了代码生成插件，所以不想安装其他的代码提示插件，就选择这种方法了。

```json
{
	// Place your snippets for python here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	// "Print to console": {
	// 	"prefix": "log",
	// 	"body": [
	// 		"console.log('$1');",
	// 		"$2"
	// 	],
	// 	"description": "Log output to console"
	// }
	"HEADER":{"prefix": "header",
        "body": [
        "# -*- coding: utf-8 -*-",
        "'''",
        "@File    :   $TM_FILENAME",
        "@Time    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND",
        "@Author  :   chenmeng",
        "@Desc    :   None",
        "'''"
    ],
  },
  "PYMAIN":{
	"prefix": "pymain",
	"body": [
	"if __name__ == '__main__':",
	"",
	],
	"description": "python–main"
	}
}
```





