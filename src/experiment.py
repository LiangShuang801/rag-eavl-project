import os
import time
import pandas as pd

from rag_engine import (
    build_vector_store,
    retrieve_documents
)

# ==========================================
# 1. 测试数据
# ==========================================

DATASET_PATH = "./test_dataset/qa_set.csv"

# 实验专用数据库根目录
EXPERIMENT_DB_ROOT = "./chroma_experiments"

# ==========================================
# 2. 第二轮实验参数
# ==========================================

# 第一轮实验得到的最佳 Chunk 参数
BEST_CHUNK_SIZE = 200
BEST_CHUNK_OVERLAP = 30

# 第二轮重点比较 Top-K
TOP_K_VALUES = [1, 3, 5]

# ==========================================
# 3. 判断检索结果是否命中目标知识
# ==========================================

def is_relevant(doc, gold_keywords):
    text = doc.page_content.lower()
    for keyword in gold_keywords:
        if keyword.lower() in text:
            return True
    return False

# ==========================================
# 4. 计算 Recall@K
# ==========================================

def calculate_recall_at_k(
    retrieved_docs,
    gold_keywords,
    k
):
    if not gold_keywords:
        return None
    top_docs = retrieved_docs[:k]
    for doc in top_docs:
        if is_relevant(
            doc,
            gold_keywords
        ):
            return 1.0
    return 0.0

# ==========================================
# 5. 计算 MRR
# ==========================================

def calculate_mrr(
    retrieved_docs,
    gold_keywords
):
    if not gold_keywords:
        return None
    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):
        if is_relevant(
            doc,
            gold_keywords
        ):
            return 1 / rank
    return 0.0

# ==========================================
# 6. 解析关键词
# ==========================================

def get_gold_keywords(row):
    keyword_text = str(
        row["gold_keywords"]
    )
    if (
        keyword_text == "nan"
        or not keyword_text
    ):
        return []
    return [
        keyword.strip()
        for keyword in keyword_text.split("|")
        if keyword.strip()
    ]

# ==========================================
# 7. 获取实验数据库路径
# ==========================================

def get_experiment_db_path():
    path = os.path.join(
        EXPERIMENT_DB_ROOT,
        f"chunk_{BEST_CHUNK_SIZE}"
        f"_overlap_{BEST_CHUNK_OVERLAP}"
    )
    os.makedirs(
        path,
        exist_ok=True
    )
    return path

# ==========================================
# 8. 单组 Top-K 实验
# ==========================================

def run_single_experiment(
    top_k,
    df,
    db_path
):
    print()
    print("========================================")
    print(
        f"实验参数："
        f"chunk_size={BEST_CHUNK_SIZE}, "
        f"chunk_overlap={BEST_CHUNK_OVERLAP}, "
        f"top_k={top_k}"
    )
    print("========================================")

    recall_1_list = []
    recall_3_list = []
    recall_5_list = []
    mrr_list = []
    latency_list = []

    # --------------------------------------
    # 遍历测试数据
    # --------------------------------------

    for _, row in df.iterrows():
        question = row["question"]
        gold_keywords = get_gold_keywords(
            row
        )

        # ----------------------------------
        # 开始计时
        # ----------------------------------

        start_time = time.perf_counter()

        # ----------------------------------
        # 检索
        # ----------------------------------

        retrieved_docs = retrieve_documents(
            question,
            top_k=5,
            db_path=db_path
        )

        end_time = time.perf_counter()

        latency = (
            end_time -
            start_time
        )
        latency_list.append(
            latency
        )

        # ==================================
        # 根据当前 Top-K 计算指标
        # ==================================

        recall_1 = None
        recall_3 = None
        recall_5 = None

        if top_k >= 1:
            recall_1 = calculate_recall_at_k(
                retrieved_docs,
                gold_keywords,
                1
            )

        if top_k >= 3:
            recall_3 = calculate_recall_at_k(
                retrieved_docs,
                gold_keywords,
                3
            )

        if top_k >= 5:
            recall_5 = calculate_recall_at_k(
                retrieved_docs,
                gold_keywords,
                5
            )

        mrr = calculate_mrr(
            retrieved_docs[:top_k],
            gold_keywords
        )


        # ----------------------------------
        # 保存结果
        # ----------------------------------

        if recall_1 is not None:
            recall_1_list.append(
                recall_1
            )

        if recall_3 is not None:
            recall_3_list.append(
                recall_3
            )

        if recall_5 is not None:
            recall_5_list.append(
                recall_5
            )

        if mrr is not None:
            mrr_list.append(
                mrr
            )

    # ======================================
    # 计算平均指标
    # ======================================

    avg_recall_1 = (
        sum(recall_1_list) / len(recall_1_list)
        if recall_1_list
        else None
    )

    avg_recall_3 = (
        sum(recall_3_list) / len(recall_3_list)
        if recall_3_list
        else None
    )

    avg_recall_5 = (
        sum(recall_5_list) / len(recall_5_list)
        if recall_5_list
        else None
    )

    avg_mrr = (
        sum(mrr_list) /
        len(mrr_list)
        if mrr_list
        else 0
    )

    avg_latency = (
        sum(latency_list) /
        len(latency_list)
        if latency_list
        else 0
    )

    # ======================================
    # 输出结果
    # ======================================

    print()
    print(
        f"平均 Recall@1："
        f"{avg_recall_1:.3f}"
        if avg_recall_1 is not None
        else "平均 Recall@1：N/A"
    )

    print(
        f"平均 Recall@3："
        f"{avg_recall_3:.3f}"
        if avg_recall_3 is not None
        else "平均 Recall@3：N/A"
    )

    print(
        f"平均 Recall@5："
        f"{avg_recall_5:.3f}"
        if avg_recall_5 is not None
        else "平均 Recall@5：N/A"
    )
    print(
        f"平均 MRR："
        f"{avg_mrr:.3f}"
    )
    print(
        f"平均检索耗时："
        f"{avg_latency:.3f} 秒"
    )

    # ======================================
    # 返回结果
    # ======================================

    return {
        "chunk_size":
            BEST_CHUNK_SIZE,
        "chunk_overlap":
            BEST_CHUNK_OVERLAP,
        "top_k":
            top_k,
        "avg_recall_at_1":
            round(
                avg_recall_1,
                3
            ),
        "avg_recall_at_3":
            round(
                avg_recall_3,
                3
            ),
        "avg_recall_at_5":
            round(
                avg_recall_5,
                3
            ),
        "avg_mrr":
            round(
                avg_mrr,
                3
            ),
        "avg_latency":
            round(
                avg_latency,
                3
            )
    }

# ==========================================
# 9. 执行第二轮实验
# ==========================================

def run_all_experiments():
    print()
    print("========================================")
    print("RAG第二轮 Top-K 对照实验开始")
    print("========================================")
    print(
        f"固定 Chunk 参数："
        f"chunk_size={BEST_CHUNK_SIZE}, "
        f"chunk_overlap={BEST_CHUNK_OVERLAP}"
    )
    print(
        f"测试 Top-K：{TOP_K_VALUES}"
    )

    # --------------------------------------
    # 创建实验数据库目录
    # --------------------------------------

    os.makedirs(
        EXPERIMENT_DB_ROOT,
        exist_ok=True
    )

    # --------------------------------------
    # 读取测试数据
    # --------------------------------------

    df = pd.read_csv(
        DATASET_PATH,
        encoding="utf-8"
    )

    print(
        f"测试数据数量：{len(df)}"
    )

    # ======================================
    # 构建一次最佳参数对应的向量数据库
    # ======================================

    db_path = get_experiment_db_path()
    print()
    print(
        f"实验数据库：{db_path}"
    )
    print(
        "正在构建最佳 Chunk 参数对应的向量数据库..."
    )

    build_vector_store(
        chunk_size=
            BEST_CHUNK_SIZE,
        chunk_overlap=
            BEST_CHUNK_OVERLAP,
        db_path=
            db_path
    )

    # ======================================
    # 开始 Top-K 实验
    # ======================================

    results = []

    total_experiments = len(
        TOP_K_VALUES
    )

    current = 0

    for top_k in TOP_K_VALUES:
        current += 1

        print()
        print(
            f"正在执行实验 "
            f"{current}/"
            f"{total_experiments}"
        )

        result = run_single_experiment(
            top_k=
                top_k,
            df=
                df,
            db_path=
                db_path
        )

        results.append(
            result
        )

    # ======================================
    # 保存实验结果
    # ======================================

    result_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "./output",
        exist_ok=True
    )

    result_df.to_csv(
        "./output/topk_experiment_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # 输出最终结果
    # ======================================

    print()
    print("========================================")
    print("第二轮 Top-K 实验完成")
    print("========================================")

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print(
        "实验结果已保存："
        "./output/topk_experiment_result.csv"
    )

# ==========================================
# 10. 程序入口
# ==========================================

if __name__ == "__main__":

    run_all_experiments()