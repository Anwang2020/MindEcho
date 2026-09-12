# MindEcho

MindEcho 是一个基于大语言模型的对话与文档问答应用，包含上下文管理、多格式文档解析、检索增强生成（RAG）、Agent 调度、独立的模型微调模块，以及可选的 PySide6 本地桌面客户端。

PDF 解析借鉴了 RAGFlow 的 DeepDoc 源码，并在项目中组织了原生文字与 OCR 结合、版面及表格处理，以及面向 CPU 的并行和批量处理流程。

> 当前版本为开发原型，不包含前端和模型权重。`requirements.txt` 已根据项目 Conda 环境提取并固定直接依赖；首次使用请同时阅读[当前限制](#当前限制)。

## 功能概览

| 模块 | 内容 |
| --- | --- |
| 上下文管理 | 保存问答记录，组合会话历史、语义相关消息和摘要，补全依赖前文的问题 |
| 文档解析 | PDF、Word、Excel 与文本文件解析；PDF 包含 OCR、版面识别、表格结构和文本合并流程 |
| 知识库 | 文本分块、本地 BGE 向量化、LanceDB 存储、知识库描述与检索工具生成 |
| Agent | 监督节点路由至文档问答、Text2SQL 或网络搜索；文档问答包含相关性判断、问题重写和答案校验 |
| Text2SQL | 读取库表结构、模型生成 SQL、单语句只读校验、受限执行与 Markdown 结果返回 |
| 微调 | 指令样本处理，以及 LoRA、Prompt Tuning、P-Tuning、Prefix Tuning、IA³、BitFit 配置逻辑 |
| 桌面客户端 | 本地启动后端、WebSocket 流式对话和多文件上传入口 |

## 目录结构

```text
MindEcho/
├── apps/
│   ├── main.py                  # FastAPI 入口
│   ├── api.py                   # 路由汇总，统一前缀 /apps
│   ├── config.py                # 大模型连接配置
│   ├── common/                  # 模型、嵌入器与提示词加载
│   ├── features/
│   │   ├── conversation/        # 对话 Agent、记忆、SQLite 和历史向量库
│   │   └── rag/                 # 解析、分块、知识库、RAG Agent 和网络搜索
│   │   └── text2sql/            # 数据库结构读取、只读 SQL 执行与 API
│   │       └── parse/
│   │           ├── process_pdf/
│   │           ├── process_word/
│   │           └── process_excel/
│   ├── template/                # 角色、摘要、改写和校验提示词
│   ├── docs/parse.md            # 文档解析笔记
│   └── logs/                   # 日志模块及运行日志
└── train/fine_tune/
    ├── base.py                  # ModelTrainer 与训练示例
    └── train_mode.py            # 微调方法封装
├── interface/desktop.py         # 可选 PySide6 桌面客户端
└── desktop.py                   # 桌面客户端启动器
```

## 环境准备

### 1. 创建 Python 环境

源码使用 Python 3.10 及以上语法。以下以 Python 3.11、Windows PowerShell 为例。所有命令均在仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS 的环境激活命令为：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

项目根目录的 `requirements.txt` 来自 `mindecho` Conda 环境，记录了服务、解析、向量检索与微调所需的直接依赖。

```bash
python -m pip install -r requirements.txt
```

注意：

- 源码使用 `langchain.tools`、`langchain_core.prompts.load_prompt` 和 `langgraph.prebuilt.create_react_agent` 等接口。请优先使用 `requirements.txt` 中的版本组合，不要单独升级 LangChain/LangGraph。
- 本地 BGE 模型采用按需加载；启动服务不要求先加载模型，但 RAG 入库、检索和会话记忆嵌入前必须完成模型准备。
- CPU 文档解析可使用 `onnxruntime`；GPU 环境需要自行匹配 PyTorch、CUDA 和 ONNX Runtime。代码根据 PyTorch 是否检测到 CUDA 选择执行方式，不能仅安装 CPU 版 ONNX Runtime 后就假定一定走 CPU。
- 微调的设备与精度配置独立于 PDF 解析，不能将“PDF 支持 CPU”理解为“大模型微调已在 CPU 上验证”。

### 3. 配置对话模型

在仓库根目录创建 `.env`：

```dotenv
MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
```

| 配置项 | 说明 |
| --- | --- |
| `MODEL` | 服务端实际提供的模型名称 |
| `LLM_API_KEY` | 对话模型服务的访问密钥 |
| `LLM_BASE_URL` | 兼容 OpenAI 协议的接口地址，通常包含 `/v1` |

项目通过 `ChatOpenAI` 调用模型，温度设置为 `0`。模型服务需要支持工具调用及结构化输出，RAG 路由和二元校验依赖这些能力。上传文档时也会调用模型生成知识库用途描述，所以入库并非完全离线。

不要提交 `.env` 或真实密钥。默认嵌入器使用下面的本地 BGE 模型；如直接使用备用远程嵌入器，还需要设置 `EMBEDDING_API_KEY`，并可选设置 `EMBEDDING_BASE_URL`。

### 4. 准备本地模型

模型权重需要另外准备，路径如下：

```text
apps/common/models_dir/bge-base-zh-v1.5/
    ...完整 SentenceTransformer 模型文件...

model/rag/res/deepdoc/
    det.onnx
    rec.onnx
    ocr.res
    layout.onnx
    tsr.onnx
```

- BGE：源码使用 `bge-base-zh-v1.5`，目录定位在 `apps/common/models.py`，首次需要嵌入时加载。请准备完整模型与 tokenizer 文件，而非单个权重文件。
- DeepDoc：用于 OCR 检测、文字识别、版面和表格结构识别。文件名由当前源码确定，模型输入输出需要与解析器兼容。
- OCR 和表格识别器包含从 `InfiniFlow/deepdoc` 下载的回退逻辑，OCR 回退下载地址配置了镜像；版面识别器不会自行下载。建议提前准备所有文件。
- `model/rag/res/deepdoc` 相对于**进程工作目录**定位，因此应从仓库根目录启动。

### 5. 检查知识库注册表

应用会在导入时读取 `apps/features/rag/table_registry.json`。如果是全新环境，且没有配套的 LanceDB 数据，注册表应使用空结构：

```json
[{}]
```

如果已有知识库，请保留注册表及对应的 `embedding_db`，不要直接清空注册表。旧注册表中的表标识不会自动生成同名向量表。

## 启动服务

从仓库根目录以包入口启动即可。

Windows PowerShell：

```powershell
python -m uvicorn apps.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS：

```bash
python -m uvicorn apps.main:app --host 127.0.0.1 --port 8000
```

首次使用保持单进程运行。不要直接运行某个路由文件中的 `__main__` 示例作为服务入口。

启动后访问：

- Swagger UI：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- OpenAPI：[http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

HTTP 接口可以在 Swagger UI 中测试，WebSocket 对话需要单独使用客户端。当前对话保存逻辑会回调 `http://127.0.0.1:8000/apps/chat/save_message`；更改端口或拆分部署时需要同步修改该地址。

## 可选：启动桌面客户端

恢复版本中包含的 PySide6 客户端已按当前后端路由重新接入。它会在本机启动 FastAPI 服务，并提供流式对话与知识库文件上传入口：

```powershell
python desktop.py
```

客户端默认使用 `127.0.0.1:8000`、`/apps/chat/ws/chat` 和 `/apps/rag/upload`，因此须先按上文准备 `.env` 与本地模型。桌面端依赖已包含在 `requirements.txt` 的 `PySide6` 与 `websockets` 条目中；仅运行服务端时也可以不使用该入口。

## Text2SQL：连接结构化数据库

Text2SQL 默认不连接任何数据库。要启用它，请在 `.env` 中配置 SQLAlchemy 数据库连接串，例如 SQLite：

```dotenv
TEXT2SQL_DATABASE_URL=sqlite:///D:/data/business.db
TEXT2SQL_MAX_ROWS=100
```

服务会先读取表和字段结构，再让模型生成 SQL。执行层只接受**一条**以 `SELECT` 或 `WITH` 开头的查询，拒绝写入、DDL、管理命令和多语句；结果最多返回 `TEXT2SQL_MAX_ROWS` 行。仍建议使用数据库账户本身的只读权限，而不要把高权限生产库凭据交给应用。

可通过以下接口使用：

- `GET /apps/text2sql/schema`：查看应用可读取的表结构。
- `POST /apps/text2sql/query`：提交 `{"question":"按地区统计本月销售额"}`，返回 SQL、行数据和 Markdown 格式答案。

在对话中，监督 Agent 也可以将明确依赖数据库的数据问题转交给 Text2SQL。Text2SQL 需要支持结构化输出的模型服务；若未配置数据源，接口与对话会返回明确的配置错误，而不会尝试访问本地其他数据库。

## 上传文件并建立知识库

接口：`POST /apps/rag/upload`，使用 `multipart/form-data`，字段名为 `files`，支持重复字段上传多文件。

Windows 示例：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/apps/rag/upload" -F "files=@./example.docx"
```

Linux/macOS 将 `curl.exe` 替换为 `curl`。多文件示例：

```bash
curl -X POST "http://127.0.0.1:8000/apps/rag/upload" -F "files=@./example.docx" -F "files=@./notes.md"
```

处理返回：

```json
{"message": "upload success"}
```

或：

```json
{"message": "upload failed"}
```

调用流程为：文件解析 → 内容分块 → 知识库描述生成 → 文本向量化 → LanceDB 写入。一次上传请求中的多个文件合并为一个知识库表。临时上传文件在请求处理结束时删除，原文件请自行留存。

当前接口会把部分失败封装在响应正文中，**HTTP 200 不代表入库成功**。请检查 `message`、知识库列表和日志。首次测试可使用有完整正文的 DOCX；极短文本可能受分块最小长度条件影响。

### 文件格式与行为

| 后缀 | 处理方式 | 当前注意事项 |
| --- | --- | --- |
| `.pdf` | 原生文字、OCR、版面和表格处理 | 原生文字和 OCR 路径均输出统一内容块结构 |
| `.docx` | 标题层级、段落和表格 | 表格转换为 Markdown 文本 |
| `.doc` | Spire.Doc 文本抽取 | 需要对应依赖及其运行环境 |
| `.xlsx`、`.xls` | 逐工作表读取并转为 Markdown | 表格按统一内容块进入后续分块流程 |
| `.txt`、`.md`、`.html`、`.py`、`.log` | UTF-8 读取，失败回退 GBK | 按文本处理，不执行代码，也不专门解析 HTML 结构 |

后缀判断区分大小写，请使用小写扩展名。

## 知识库管理

| 方法 | 路径 | 请求体 / 用途 |
| --- | --- | --- |
| `GET` | `/apps/rag/validate` | 查看知识库标识与描述 |
| `POST` | `/apps/rag/update` | JSON 对象：`{"知识库ID": "新的用途描述"}` |
| `POST` | `/apps/rag/delete` | JSON 数组：`["知识库ID"]`，删除对应向量表和注册信息 |

查看列表：

```powershell
curl.exe "http://127.0.0.1:8000/apps/rag/validate"
```

可在 Swagger UI 中提交更新或删除请求。删除操作会移除整个知识库表；如果一次上传了多个文件，这些文件的入库内容会一起删除。当前管理接口部分状态字段使用字符串，成功值拼写为 `"ture"`，并非标准布尔值。

## WebSocket 对话

地址：`ws://127.0.0.1:8000/apps/chat/ws/chat`

发送一个文本消息，内容为以下对象的 JSON 字符串：

```json
{
  "content": "请概括已上传资料中的主要方法",
  "type": "chat",
  "session_id": "demo-session-001"
}
```

- `content`：用户问题。
- `type`：对话类型；常规调用使用 `chat`。
- `session_id`：会话标识；连续提问复用同一值，新会话换用新的值。

浏览器开发者控制台示例：

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/apps/chat/ws/chat');
let answer = '';

ws.onopen = () => ws.send(JSON.stringify({
  content: '请概括已上传资料中的主要方法',
  type: 'chat',
  session_id: 'demo-session-001'
}));
ws.onmessage = event => {
  answer += event.data;
  console.log(event.data);
};
ws.onerror = event => console.error('WebSocket 连接或处理异常', event);
ws.onclose = event => console.log('连接关闭', event.code, answer);
```

当前协议一条连接处理一个问题，然后关闭连接；下一次提问需要重新连接。客户端应拼接收到的文本片段。后端先执行完整 Agent 流程，再按每段最多 80 个字符发送最终回答，**不是模型生成阶段的逐 token 流式输出**。

消息通常由对话流程自动保存。`POST /apps/chat/save_message` 接受以下格式，返回 `{"status":"ok"}` 表示后台任务已提交，不表示存储已经完成：

```json
{
  "session_id": "demo-session-001",
  "chat_type": "chat",
  "message": {"user": "你好", "system": "你好，有什么可以帮助你？"}
}
```

正常使用 WebSocket 时，不需要额外调用保存接口，避免重复写入。

## PDF 与 RAG 参数位置

| 位置 | 当前参数 / 行为 |
| --- | --- |
| `apps/features/rag/parse/process_pdf/service.py` | 页面渲染比例为 3；长文档字符提取按 32 页分组；页面 OCR 使用 4 个工作线程 |
| `apps/features/rag/parse/process_pdf/utils.py` | 文字识别每批最多 16 个区域，按宽高比组织批次 |
| `apps/features/rag/parse/process_pdf/util.py` | ONNX 执行方式、模型缓存与会话选项 |
| `apps/features/rag/service.py` | 默认分块参数为 500～2000 token，超长内容使用语义分块；这些值不是严格保证的最终块长度 |
| `apps/features/rag/tools.py` | Agent 检索工具默认返回最多 3 条内容 |
| `apps/features/conversation/memory.py` | 历史消息读取上限为 10，历史语义检索上限为 5 |
| `apps/template/CharacterSetting.yaml` | 对话角色及监督路由提示 |
| `apps/template/SummarySession.yaml` | 会话摘要提示 |

这些解析参数目前主要通过修改源码调整，不是统一的环境变量配置项。

## 模型微调

微调模块独立于在线服务，入口为 `train/fine_tune/base.py`，不会自动将训练结果部署到对话接口。

### 训练数据

准备 JSON 文件，样本包含 `instruction`、`input`、`output` 三个字符串字段；没有额外输入时，`input` 使用空字符串。

```json
[
  {
    "instruction": "解释什么是检索增强生成",
    "input": "请使用简洁的中文回答",
    "output": "检索增强生成先获取与问题相关的资料，再将资料作为上下文交给模型生成回答。"
  }
]
```

预处理将指令和输入拼接为 Human 提示，将目标回答作为 Assistant 输出，并对提示部分使用 `-100` 标签屏蔽损失。序列超过 `max_len` 时直接截断，因此需要检查长输入是否挤占了目标回答。

### 配置并运行

修改 `base.py` 底部的示例，至少设置以下参数为本机实际值：

| 参数 | 说明 |
| --- | --- |
| `model_path` | 本地因果语言模型完整目录 |
| `data_path` | 指令数据 JSON 路径 |
| `output_dir` | Trainer 输出目录 |
| `max_len` | 最大训练序列长度 |
| `train_type` | 微调方式，默认 `lora` |
| `use_gpu` | 是否允许将模型移动到可用 CUDA 设备 |
| `lora_target_modules` | 与基础模型结构匹配的目标层名称 |

当前示例使用 Qwen2.5-7B-Instruct，LoRA 参数为 `r=16`、`alpha=32`、`dropout=0.1`，目标层为 `q_proj` 和 `v_proj`。代码中的本地绝对路径需要自行替换。

从仓库根目录执行：

```bash
python -m train.fine_tune.base
```

运行前先按实际模型、数据集及输出目录修改 `base.py` 底部示例配置。`train_type` 的源码取值为 `lora`、`prompt_tuning`、`p_tuning`、`prefix_tuning`、`IA3`、`fitbit`；最后一个是 BitFit 分支使用的字符串。

## 数据保存位置

| 数据 | 位置 |
| --- | --- |
| 会话 SQLite 数据库 | `apps/features/conversation/db/app_data.db` |
| 历史消息向量库 | `apps/features/conversation/embedding/embedding_db/` |
| 历史向量库注册表 | `apps/features/conversation/embedding/table_registry.json` |
| 文档向量库 | `apps/features/rag/embedding_db/` |
| 知识库注册表 | `apps/features/rag/table_registry.json` |
| 临时上传目录 | `apps/features/rag/uploads/` |
| 日志 | `apps/logs/app.log` |

备份知识库时，应同时保留向量库目录与对应注册表。上传原文件不在持久化备份范围内。

## 当前限制

以下是当前源码中会影响实际使用的事项，不应将本仓库直接当作公网或多租户服务部署。

### 文件与检索

- 过短内容可能不满足分块器的最小长度条件，因此不会生成知识库；服务会明确返回失败，上传后仍应检查日志与实际检索结果。
- PDF 图像渲染默认切片上限为 299 页，且解析器实例保存可变状态。长文档和多个并发上传请求需要额外验证；本地使用时先串行上传。
- ONNX 的 CPU 会话固定为顺序执行、每类线程 2 个，并通过 `sess_options` 传入运行时；不同机器仍应以实际压测结果确定最佳线程数。
- 网络搜索抓取公开搜索页与网页正文，可能受网络、反爬和页面结构变化影响。

### 对话与隔离

- WebSocket 仅接受 JSON 对象并校验必要字符串字段；接口仍没有身份认证，当前只适合受信任的本地使用。
- 历史语义检索没有按用户或会话过滤。不同会话的历史消息可能被召回，不具备多用户隔离能力。
- 回答重生成计数未在所有分支中正确写回状态，复杂问题可能触发状态图递归限制；遇到该错误需检查流程状态更新，不能只提高递归上限。

### 微调

- 微调入口会在无 CUDA 时使用 `float32` 并保持在 CPU；但大模型 CPU 训练的耗时和内存开销很高，建议先以小模型和小数据集验证。
- `inference(model_path, adapter_path, input_text)` 为本地 PEFT 适配器提供基础推理封装，实际部署仍应按模型的聊天模板、上下文窗口和服务框架完善。

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| `No module named apps/api/features` | 是否位于仓库根目录，并使用 `python -m uvicorn apps.main:app` 启动 |
| LangChain 导入失败 | 包版本是否属于兼容的一组，源码中的旧接口是否仍存在 |
| 启动时找不到 BGE | 本地模型目录是否完整，路径是否为 `apps/common/models_dir/bge-base-zh-v1.5/` |
| PDF 报模型不存在 | 工作目录和 `det/rec/layout/tsr.onnx`、`ocr.res` 是否正确 |
| 上传返回 `upload failed` | 查看 `apps/logs/app.log`，检查模型连接、返回格式与空分块 |
| 检索提示表不存在 | 注册表是否来自另一套环境，是否缺失对应向量库 |
| 对话首段返回较慢 | 后端等待完整 Agent 执行完成后才发送文本片段 |
| 对话结束后保存失败 | 内部保存地址是否仍能通过 `127.0.0.1:8000` 访问 |

## 源码参考

PDF 解析借鉴 RAGFlow 的 DeepDoc 实现，解析笔记位于 `apps/docs/parse.md`。发布和分发时，应保留所借鉴源码的来源说明，并核对其许可要求及模型使用条款。本仓库当前未提供独立的项目许可证文件。
