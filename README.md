#rag-eval-project

 RAG知识库问答系统自动化测试与评测项目


##项目简介

本项目针对RAG知识库问答系统构建自动化测试与评测流程，从检索质量、回答质量、安全性和性能四个维度对系统进行综合评估。
项目通过构建结构化测试数据集，自动执行多类型测试用例，并计算Recall@K、MRR、Correctness、Faithfulness、Hallucination Rate、拒答率以及响应时间等指标。
同时结合LLM‑as‑a‑Judge对生成答案进行自动化质量评测，并根据预设阈值自动识别潜在缺陷，最终生成HTML测试报告。


##技术栈
- Python

- Pytest

- ChromaDB

- LLM API

- Pandas

- NumPy

- Matplotlib

- python‑dotenv


##项目结构

rag‑eval‑project

├── chroma_experiments   # 不同分块、重叠参数下的向量库实验结果

├── chroma_store         # Chroma向量数据库存储目录

├── knowledge_base       # 原始知识库文档

├── results              # 输出结果：指标图、csv统计、html评测报告

├── src                  # 评测核心源码

│   ├── evaluator.py         # 答案评测打分模块

│   ├── experiment.py        # 对照实验执行逻辑

│   ├── rag_engine.py        # RAG检索引擎封装

│   ├── report_generator.py  # 评测报告生成

│   └── visualize_results.py # 指标可视化绘图

├── test_dataset         # QA测试数据集 qa_set.csv

├── .env                 # 密钥、环境配置

├── .gitignore

├── README.md

└── requirements.txt


## 评测指标可视化
![chunk-mrr](./doc/chunk_mrr.png)
![chunk-recall](./doc/chunk_recall.png)
![topk-latency](./doc/topk_latency.png)
![topk-recall](./doc/topk_recall.png)

## 评测报告预览
完整评测报告共8个模块，下图为关键页面预览，拉取项目本地运行可生成完整评测报告。

![测试概览](./doc/report_1.png)
![检索与回答质量指标](./doc/report_2.png)
![安全质量与性能指标](./doc/report_3.png)
![缺陷分析](./doc/report_4.png)
![详细测试结果](./doc/report_5.png)
![测试结论](./doc/report_6.png)


## 运行方式

1. 安装依赖
pip install -r requirements.txt

2. 配置API Key
OPENAI_API_KEY=你的密钥

3.执行测试
python evaluator.py

4.查看测试报告
运行完打开results/rag_test_report.html即可查看完整测试报告

## 项目成果
通过自动化测试流程，对 RAG 系统的检索、生成、安全和性能进行量化评估，实现从测试数据构建、自动执行、指标计算、缺陷识别到报告生成的完整测试闭环。
