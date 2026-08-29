###### \#rag-eval-project



&#x20;RAG知识库问答系统自动化测试与评测项目





\##项目简介



本项目针对RAG知识库问答系统构建自动化测试与评测流程，从检索质量、回答质量、安全性和性能四个维度对系统进行综合评估。

项目通过构建结构化测试数据集，自动执行多类型测试用例，并计算Recall@K、MRR、Correctness、Faithfulness、Hallucination Rate、拒答率以及响应时间等指标。

同时结合LLM‑as‑a‑Judge对生成答案进行自动化质量评测，并根据预设阈值自动识别潜在缺陷，最终生成HTML测试报告。





\##技术栈

\- Python



\- Pytest



\- ChromaDB



\- LLM API



\- Pandas



\- NumPy



\- Matplotlib



\- python‑dotenv





\##项目结构



rag‑eval‑project



├── chroma\_experiments   # 不同分块、重叠参数下的向量库实验结果



├── chroma\_store         # Chroma向量数据库存储目录



├── knowledge\_base       # 原始知识库文档



├── results              # 输出结果：指标图、csv统计、html评测报告



├── src                  # 评测核心源码



│   ├── evaluator.py         # 答案评测打分模块



│   ├── experiment.py        # 对照实验执行逻辑



│   ├── rag\_engine.py        # RAG检索引擎封装



│   ├── report\_generator.py  # 评测报告生成



│   └── visualize\_results.py # 指标可视化绘图



├── test\_dataset         # QA测试数据集 qa\_set.csv



├── .env                 # 密钥、环境配置



├── .gitignore



├── README.md



└── requirements.txt





\## 评测指标可视化

!\[chunk‑mrr](./doc/chunk\_mrr.png)

!\[chunk‑recall](./doc/chunk\_recall.png)

!\[topk‑latency](./doc/topk\_latency.png)

!\[topk‑recall](./doc/topk\_recall.png)



\## 评测报告预览

完整评测报告共8个模块，下图为关键页面预览，拉取项目本地运行可生成完整HTML格式测试报告。



!\[测试概览](./doc/report\_1.png)

!\[检索与回答质量指标](./doc/report\_2.png)

!\[安全质量与性能指标](./doc/report\_3.png)

!\[缺陷分析](./doc/report\_4.png)

!\[测试结论](./doc/report\_6.png)





\## 运行方式



1\. 安装依赖

pip install -r requirements.txt



2. 配置API Key

OPENAI\_API\_KEY=你的密钥



3.执行测试

python evaluator.py



4.查看测试报告

运行完打开results/rag\_test\_report.html即可查看完整测试报告



5.项目成果

通过自动化测试流程，对 RAG 系统的检索、生成、安全和性能进行量化评估，实现从测试数据构建、自动执行、指标计算、缺陷识别到报告生成的完整测试闭环。

