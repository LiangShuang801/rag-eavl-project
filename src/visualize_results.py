import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 文件路径
# ==========================================

CHUNK_RESULT_PATH = "./output/experiment_result.csv"
TOPK_RESULT_PATH = "./output/topk_experiment_result.csv"
OUTPUT_DIR = "./output"

# ==========================================
# 2. 创建输出目录
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# 3. 读取实验结果
# ==========================================

chunk_df = pd.read_csv(
    CHUNK_RESULT_PATH
)
topk_df = pd.read_csv(
    TOPK_RESULT_PATH
)
print("实验结果读取完成！")
print()
print("第一轮实验结果：")
print(chunk_df.to_string(index=False))
print()
print("第二轮实验结果：")
print(topk_df.to_string(index=False))

# ==========================================
# 4. 第一轮：Chunk Size 对 Recall@1
# ==========================================

# 固定每个 Chunk Size 下的最佳 Recall@1
chunk_recall = (
    chunk_df
    .groupby("chunk_size")["avg_recall_at_1"]
    .max()
)
plt.figure()
plt.plot(
    chunk_recall.index,
    chunk_recall.values,
    marker="o"
)
plt.xlabel("Chunk Size")
plt.ylabel("Recall@1")
plt.title(
    "Chunk Size vs Recall@1"
)
plt.grid(True)
plt.savefig(
    "./output/chunk_recall.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 5. 第一轮：Chunk Size 对 MRR
# ==========================================

chunk_mrr = (
    chunk_df
    .groupby("chunk_size")["avg_mrr"]
    .max()
)
plt.figure()
plt.plot(
    chunk_mrr.index,
    chunk_mrr.values,
    marker="o"
)
plt.xlabel("Chunk Size")
plt.ylabel("MRR")
plt.title(
    "Chunk Size vs MRR"
)
plt.grid(True)
plt.savefig(
    "./output/chunk_mrr.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 6. 第二轮：Top-K 对 Recall
# ==========================================

plt.figure()
plt.plot(
    topk_df["top_k"],
    topk_df["avg_recall_at_1"],
    marker="o",
    label="Recall@1"
)
plt.plot(
    topk_df["top_k"],
    topk_df["avg_recall_at_3"],
    marker="o",
    label="Recall@3"
)
plt.plot(
    topk_df["top_k"],
    topk_df["avg_recall_at_5"],
    marker="o",
    label="Recall@5"
)
plt.xlabel("Top-K")
plt.ylabel("Recall")

plt.title(
    "Top-K vs Recall"
)
plt.legend()
plt.grid(True)
plt.savefig(
    "./output/topk_recall.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 7. 第二轮：Top-K 对检索耗时
# ==========================================

plt.figure()
plt.plot(
    topk_df["top_k"],
    topk_df["avg_latency"],
    marker="o"
)
plt.xlabel("Top-K")
plt.ylabel(
    "Average Retrieval Latency (s)"
)
plt.title(
    "Top-K vs Retrieval Latency"
)
plt.grid(True)
plt.savefig(
    "./output/topk_latency.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 8. 完成
# ==========================================

print()
print("========================================")
print("实验结果可视化完成")
print("========================================")
print()
print("生成的图表：")
print(
    "./output/chunk_recall.png"
)
print(
    "./output/chunk_mrr.png"
)
print(
    "./output/topk_recall.png"
)
print(
    "./output/topk_latency.png"
)