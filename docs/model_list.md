## 👏 模型介绍

在**模型用途**方面, 目前公开的MindChat系列模型分为**心理抚慰**、**心理评测**两个方面. 当然, 我们也规划了针对(泛)心理领域其他方向的模型开发, 并且这些内容正在进行中. 这里需要特殊注明的是, 一般情况下, 公开的MindChat系列模型并没有混合通用数据, 因此在通用能力上, 将MindChat和其他通用大语言模型进行对比是不合适的.

在**模型参数**方面, 目前MindChat系列模型涵盖小参数(如1.8B甚至更小)、中参数(如7B左右)、大参数模型(如14B甚至更大), 能够满足不同社区用户或合作伙伴在不同场景下的使用(如端侧、云端、端云结合等). 

在**模型使用**方面, 目前我们对部分模型进行了权限分类, 分为**完全开源**、**申请下载**、**商务合作**等. 其中, 分级标准并非完全依照模型性能, 而也可能是因为训练使用的数据权限问题或其他特殊问题等等. 当然, 模型权限也可能随时间的变化而改变. 

* 针对**完全开源类**模型, 您可以在以下模型托管平台中进行下载; 
* 针对**申请下载类**模型, 您可以在🤗HuggingFace或者<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope上申请相关模型并且发送申请邮件至mindchat0606@163.com说明**使用用途**并注明相关平台的**用户ID**, 工作人员一般会在24小时内进行审核; 
* 针对**商务合作类**模型, 您可以发送邮件至mindchat0606@163.com说明使用用途和初步合作事项, 工作人员会在第一时间联系您!  

模型权限的详细信息您可以在下表中查看.

## 🔥 模型列表

### 开源模型

| 模型名称 | 模型参数 | 用途分类 | HuggingFace 下载 | ModelScope 下载 | wisemodel 下载 | 权限分类 | 生成2048个token的最小显存占用 | 公开日期 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| MindChat-Qwen-1_8B🆕 | 1.8B | 心理抚慰 | 🤗[HuggingFace](https://huggingface.co/X-D-Lab/MindChat-Qwen-1_8B) | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-1_8B/summary) | [wisemodel](https://wisemodel.cn/models/X-D-Lab/MindChat-Qwen-1_8B) | 完全开源 | 2.9GB | 2024.01.01 |
| *MindChat-Evaluator-EN-1_8B🆕 | 1.8B | 心理评测 | 🤗HuggingFace | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-1_8B/summary) | / | 申请下载 | 2.9GB | 2024.01.08 |
| MindChat-Qwen-7B | 7B | 心理抚慰 | 🤗[HuggingFace](https://huggingface.co/X-D-Lab/MindChat-Qwen-7B) | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-7B/summary) | / | 完全开源 | 8.2GB | 2023.08.05 |
| MindChat-Qwen-7B-v2 | 7B | 心理抚慰 | 🤗[HuggingFace](https://huggingface.co/X-D-Lab/MindChat-Qwen-7B-v2) | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-7B-v2/summary) | [wisemodel](https://wisemodel.cn/models/X-D-Lab/MindChat) | 完全开源 | 8.2GB | 2023.09.04 |
| *MindChat-Qwen-7B-v3🆕 | 7B | 心理抚慰 | 🤗[HuggingFace](https://huggingface.co/X-D-Lab/MindChat-Qwen-7B-v3) | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-7B-v3/summary) | / | 申请下载 | 8.2GB | 2024.01.05 |
| *MindChat-Qwen-14B🆕 | 14B | 心理抚慰 | 🤗[HuggingFace](https://huggingface.co/X-D-Lab/MindChat-Qwen-14B) | [<img src="../assets/image/modelscope_logo.png" width="20px" />ModelScope](https://modelscope.cn/models/X-D-Lab/MindChat-Qwen-14B/summary) | / | 申请下载 | 13.0GB | 2024.01.13 |

### 闭源模型

| 模型名称 | 模型参数 | 用途分类 | 权限分类 | 更新日期 |
| :----: | :----: | :----: | :----: | :----: |
| MindChat-Tiny | / | 心理抚慰 | 商务合作 | 2024.02.01 |
| MindChat-Small | / | 心理抚慰 | 商务合作 | 2024.02.03 |
| MindChat-Medium | / | 心理抚慰 | 商务合作 | 2024.02.03 |
| MindChat-Base | / | 心理抚慰 | 商务合作 | 2024.01.20 |
| MindChat-Large | / | 心理抚慰 | 商务合作 | 2024.01.20 |

*注: 部分信息摘录自[Qwen Repo](https://github.com/QwenLM/Qwen).
