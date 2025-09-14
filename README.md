# VideoRAG Batch Pipeline 简要说明

本 README 旨在快速了解当前 `test.py` 运行整条视频多模态问答（VideoRAG + 迭代细化）流水线所依赖的核心文件与步骤，力求精简。

## 1. 入口脚本
| 文件 | 作用 |
|------|------|
| `test/test.py` | 主入口：批量遍历输入 JSON / 目录，加载 ASR 与 LLM，调度下载、切片、帧提取、转写、迭代细化、提示构建、答案输出与失败日志。|

## 2. 直接被入口调用的辅助模块
| 文件 | 作用 |
|------|------|
| `test/question_processing.py` | 封装单个问题 end‑to‑end 处理：下载片段 → 校验/修复 → 30s 子段时间推导 → 帧提取 + ASR → (可选) 迭代细化 → 构造多模态 Prompt → 调 LLM → 解析最终 JSON 答案。|
| `test/segment_caption.py` | 原 `question_processing.py` 中的段级多帧 Caption 生成逻辑（逐帧客观描述 + 汇总段落）。|
| `test/llm_prompt_builder.py` | 原 `question_processing.py` 中 Step3~5：多模态文本块拼装、最终 Prompt 模板与 LLM 调用、答案解析。|
| `test/processing_utils.py` | 帧提取与增量补帧、帧缓存目录命名、音频 30s 片段抽取与转写（调用 faster-whisper）。|
| `test/test_media_utils.py` | 通用视频/媒体工具：下载、分辨率/时长探测、损坏检测与 faststart 修复、压缩比计算。|
| `test/test_env_utils.py` | 环境与模型检查、CUDA 库路径清理、LLM 配置选择、问题/答案规范化、图片转 base64、离线兜底逻辑。|
| `video_urls.py` | 定义 `video_urls_multi_segment`：视频 ID 到分片文件(URL) 映射；用于生成 (3min → 30s) 逻辑及 Top5 解析。|

## 3. `videorag` 包内关键子模块
| 文件/目录 | 作用（摘要） |
|-----------|--------------|
| `videorag/_llm.py` | 定义多种 LLM/Embedding 配置（本地 Hugging Face 模型短名/路径 / OpenAI / Azure / DeepSeek / InternVL 等），暴露 `cheap_model_func` 接口供调用。|
| `videorag/_llm_azure.py` | (新增) 将原 `videorag/_llm.py` 中 Azure/OpenAI on Azure 相关的 helper 与配置拆分到此文件，便于维护。|
| `videorag/_config.py` | 常量配置（如 `MINICPM_MODEL_PATH` 等）。|
| `videorag/iterative_refinement.py` | 提供 `refine_context` 与高层 refine 调度逻辑。|
| `videorag/iterative_refiner.py` | 迭代规划器实现（生成细化 plan、确定哪些片段需要 OCR/DET / 追加帧）。|
| `videorag/prompt.py` | 与提示构建相关的模板/辅助（若后续扩展）。|
| `videorag/base.py`, `_op.py`, `_splitter.py`, `_utils.py` | 底层通用操作、切分、工具函数（被迭代细化与 LLM 流程间接使用）。|
| `videorag/_storage/` | 图数据库 / 向量库 / KV 存储抽象实现：`gdb_neo4j.py`, `gdb_networkx.py`, `vdb_hnswlib.py`, `vdb_nanovectordb.py`, `kv_json.py`。当前主流程以最小依赖方式运行，可按需扩展。|
| `videorag/_videoutil/` | 语音识别/字幕/特征/切分等视频底层工具（迭代细化或后续扩展可用）。|
| `videorag/evaluate/evaluate.py` | 评估主脚本：遍历结果 JSON，匹配 Ground Truth，调用评估 LLM，计算 ROUGE、BERTScore、启发式覆盖等。|
| `videorag/evaluate/utils.py` | 从 `evaluate.py` 中抽离的小型工具函数集合（例如 `format_test_input_for_prompt`），便于复用和单测。|
| `videorag/evaluate/keypoint_match.py` | 从 `evaluate.py` 抽离的 Keypoint 启发式匹配与文本归一化逻辑，纯函数化，便于单测与复用。|
| `videorag/evaluate/config.py` | 从 `evaluate.py` 抽离配置与环境常量、缓存与模型路径设定、评估相关默认值。|
| `videorag/evaluate/prompts.py` | 从 `evaluate.py` 原样迁移评估 Prompt 模板（EVAL_PROMPT_TEMPLATE、SIMPLE_EVAL_PROMPT_TEMPLATE，逐字不改）。|

## 4. 运行所需外部资源/目录
| 资源 | 说明 |
|------|------|
| `faster-distil-whisper-large-v3/` | ASR 模型目录（WhisperModel 加载）。建议放到共享模型目录，如 `/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/00_opensource_models/huggingface.co/deepdml/faster-distil-whisper-large-v3.5`。|
| `MINICPM_MODEL_PATH` 指向的目录 |（可选）MiniCPM / 相关多模态本地模型。|
| 输入视频/JSON 根目录 | 代码默认 `/root/autodl-tmp/903test`（可修改，或使用环境变量 `INPUT_BASE_DIR` 指定）。包含：1) 子目录(每视频)内多个 QA JSON；或 2) 顶层 *top5* 结构 JSON。|
| 输出目录 | 默认写入到部署用户的 Result 目录，例如 `/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/lx/Result/<model_tag>/...`。可通过环境变量 `OUTPUT_BASE_DIR` 覆盖。|

## 5. 主要执行流程（逻辑阶段）
1. 环境准备：`sanitize_cuda_libs()` 清理冲突 CUDA 库路径；`check_dependencies()`、`check_models()`。  
2. 加载 ASR：智能判定 CUDA → `WhisperModel` (int8)。  
3. 选择 LLM：`choose_llm_config()`（优先本地短名或外部 LLM 客户端，可通过环境变量切换）。  
4. 解析输入：两种模式  
   - 目录模式：遍历每个视频子目录内问题 JSON。  
   - 顶层 Top5 JSON 模式：从 `query` + `top5_segment_names` 重建 30s 片段 URL 映射。  
5. 对每个问题调用 `process_question()`：  
   - 片段下载 + 校验/修复（坏 MP4 重下或 faststart 重封装）。  
  # VideoRAG 流水线（test.py） — 概要说明

  这是项目中用于批量运行 VideoRAG + 迭代细化流水线的说明文档（简洁版），放在 `test/` 目录下。

  要点摘要
  - 入口脚本：`test/test.py`（批量处理问题 JSON，调用帧提取、ASR、可选迭代细化、Prompt 构建与 LLM 推理）。
  - 细化与检测：`videorag/iterative_refinement.py` 与 `videorag/_videoutil/refinement_utils.py` 负责补帧、OCR（EasyOCR）和 DET（YOLO-World）。

  快速开始
  1. 安装依赖（推荐在虚拟环境或 conda 环境中）：

    在 `test/` 目录下运行：

    ```cmd
    pip install -r requirements.txt
    ```

  2. 准备模型与资源：
  - ASR 模型目录（例如 `faster-distil-whisper-large-v3/`）。建议放到 `/home/hadoop-aipnlp/.../00_opensource_models/...` 并通过 `FASTER_WHISPER_DIR` 或 `ASR_MODEL_PATH` 环境变量指定。
  - 如使用本地 MiniCPM，请设置 `MINICPM_MODEL_PATH` 环境变量指向模型路径（示例：`/home/.../00_opensource_models/huggingface.co/openbmb/MiniCPM-V-4_5`）。
  - 如使用 YOLO-World，请把模型文件放到可访问路径（示例：`/home/.../00_opensource_models/yolov8m-worldv2.pt`），并可通过 `YOLOWORLD_MODEL_PATH` 环境变量覆盖。 

  3. 运行（示例）：

  ```cmd
  # 跳过迭代细化并强制重跑
  python test/test.py --base-mode --force

  # 默认模式
  python test/test.py --force

  # 若需要指定环境变量（示例，Linux/bash）
  export FASTER_WHISPER_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/deepdml/faster-distil-whisper-large-v3.5
  export MINICPM_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/openbmb/MiniCPM-V-4_5
  export YOLOWORLD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/yolov8m-worldv2.pt
  export OUTPUT_BASE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/lx/Result
  python test/test.py --force
  ```

  主要目录与文件（简要）
  - `test/test.py`：主入口脚本，负责调度整个任务流。  
  - `test/question_processing.py`：单题端到端逻辑（下载、切分、帧提取、ASR、构建 context、调用 refine）。  
  - `test/processing_utils.py`：帧/音频处理工具与缓存管理。  
  - `videorag/iterative_refinement.py`：高层 refine 流程，调用 OCR/DET 并合并结果。  
  - `videorag/_videoutil/refinement_utils.py`：OCR 与 YOLO DET 的实现（本次已修复 set_classes 的 device 问题）。

  配置与环境变量（常用）
  - `--base-mode`：跳过迭代细化（仅基础帧+字幕）。
  - `--force`：忽略已有结果，强制重跑。 
  - `OLLAMA_CHAT_MODEL`：指定模型短名或本地 Hugging Face 模型路径（例如短名 `llama`, `qwen`, `gemma`, `internvl`, `minicpm`，或直接写完整路径）。
  - `MINICPM_MODEL_PATH`：MiniCPM 本地模型路径（如使用）。

  调试建议
  - 在运行前，先调用 `test/test_env_utils.py` 中的依赖检查函数以确认模型与 CUDA 环境是否就绪。  
  - 若使用 GPU，请确认 `torch.cuda.is_available()` 为 True，并根据需要调整模型加载目标设备（多卡时注意 device id）。

  更多细节与评估工具放在 `videorag/` 下，README 中的长版说明保留在项目文档历史记录中。

  ---

  ## requirements
  项目运行所需的 Python 包列在同目录 `requirements.txt`（已提供，见 test/requirements.txt）。

  ## 运行示例（补充）

  快速运行示例（Windows / cmd.exe 环境）

   1) 单进程本地运行（使用默认环境变量或配置的本地/外部 LLM 客户端或 OpenAI 后端）：
  python test.py

  1) 以指定多卡并发运行（例如使用 GPU 0 和 1，每个子进程绑定一个 GPU）：
  set WORKER_GPUS=0,1
  python test.py --gpus 0,1 --workers 2

  1) 为不同 GPU 指定不同的模型（父进程会将对应模型设置为子进程的 `OLLAMA_CHAT_MODEL`）：
  set WORKER_GPUS=0,1
  set MODEL_GPU_OVERRIDES=0=llama,1=internvl
  python test.py --gpus 0,1 --workers 2

  说明：VLM 加速（视觉-语言模块加速）在脚本中默认启用（llm_cfg.vlm_accel=True），子进程会收到 `VLM_ACCEL=1` 环境变量，后端实现需读取该标记并执行具体优化。

  python test.py --base-mode
# 或包含细化：
python test.py
# 指定单 JSON：
python test.py /root/autodl-tmp/903test/videorag_top5_segments_20250909_2222221.json --force

python test.py --force