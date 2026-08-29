import os
import gc

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import dashscope


# ============================================================
# 1. 加载环境变量
# ============================================================

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError(
        "没有找到 DASHSCOPE_API_KEY，请检查 .env 文件"
    )

dashscope.api_key = API_KEY


# ============================================================
# 2. 项目路径
# ============================================================

DOC_PATH = "./knowledge_base/test_doc.txt"

DB_PATH = "./chroma_store"


# ============================================================
# 3. 创建 Embedding 模型
# ============================================================

def get_embeddings():

    return DashScopeEmbeddings(
        model="text-embedding-v3"
    )


# ============================================================
# 4. 构建向量数据库
# ============================================================

def build_vector_store(
    chunk_size=200,
    chunk_overlap=30,
    db_path=DB_PATH
):
    """
    根据指定的 chunk_size 和 chunk_overlap
    重新构建 Chroma 向量数据库。
    """

    print("正在加载知识库...")

    loader = TextLoader(
        DOC_PATH,
        encoding="utf-8"
    )

    documents = loader.load()

    print(
        f"原始文档数量：{len(documents)}"
    )

    # --------------------------------------------------------
    # 文档切分
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(
        documents
    )

    # --------------------------------------------------------
    # 添加 chunk_id
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    print(
        f"切分后 chunk 数量：{len(chunks)}"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    print(
        "正在调用阿里云百炼 Embedding..."
    )

    embeddings = get_embeddings()

    # --------------------------------------------------------
    # 创建 Chroma
    # --------------------------------------------------------

    print(
        "正在创建 Chroma 向量数据库..."
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    print(
        "向量数据库创建完成！"
    )

    return vector_db, chunks


# ============================================================
# 5. 获取已有向量数据库
# ============================================================

def get_vector_store(
    db_path=DB_PATH
):

    embeddings = get_embeddings()

    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    return vector_db


# ============================================================
# 6. 提取问题关键词
# ============================================================

def extract_keywords(question):
    """
    对测试问题进行轻量级关键词提取。

    不依赖额外 NLP 模型。
    """

    stop_words = {
        "什么是",
        "是什么",
        "怎么",
        "如何",
        "哪些",
        "可以",
        "吗",
        "呢",
        "的",
        "是",
        "有",
        "什么",
        "为什么",
        "请",
        "一下",
        "介绍",
        "说一下",
        "告诉我",
        "能否",
        "是否",
        "怎么做",
        "可以怎么",
        "可以如何"
    }

    text = question.strip()

    # --------------------------------------------------------
    # 去除停用词
    # --------------------------------------------------------

    for word in stop_words:

        text = text.replace(
            word,
            " "
        )

    # --------------------------------------------------------
    # 常见技术关键词
    # --------------------------------------------------------

    tech_keywords = [
        "Playwright",
        "Selenium",
        "Pytest",
        "RAG",
        "Chroma",
        "Locust",
        "Appium",
        "PO",
        "自动化测试",
        "接口测试",
        "性能测试",
        "功能测试",
        "安全测试",
        "向量数据库",
        "向量检索",
        "幻觉",
        "浏览器",
        "测试代码",
        "测试框架",
        "Embedding",
        "Embedding模型"
    ]

    keywords = []

    for keyword in tech_keywords:

        if keyword.lower() in text.lower():

            keywords.append(
                keyword.lower()
            )

    return keywords


# ============================================================
# 7. Chunk 重排序
# ============================================================

def rerank_documents(
    question,
    documents
):
    """
    对 Embedding 召回结果进行轻量级重排序。

    评分规则：

    核心关键词命中：+2
    问题词命中：+0.5
    原始 Embedding 排名：额外保留一定权重
    """

    keywords = extract_keywords(
        question
    )

    scored_documents = []

    # --------------------------------------------------------
    # 提取问题词
    # --------------------------------------------------------

    question_text = (
        question
        .replace("？", " ")
        .replace("?", " ")
    )

    question_words = [
        word.strip()
        for word in question_text.split()
        if word.strip()
    ]

    # --------------------------------------------------------
    # 对每个 Chunk 进行评分
    # --------------------------------------------------------

    for index, doc in enumerate(
        documents
    ):

        text = doc.page_content.lower()

        score = 0

        # ----------------------------------------------------
        # 核心关键词匹配
        # ----------------------------------------------------

        for keyword in keywords:

            if keyword in text:

                score += 2

        # ----------------------------------------------------
        # 问题词匹配
        # ----------------------------------------------------

        for word in question_words:

            if len(word) >= 2:

                if word.lower() in text:

                    score += 0.5

        # ----------------------------------------------------
        # 保留 Embedding 原始排序
        # ----------------------------------------------------

        score += max(
            0,
            1 - index * 0.05
        )

        scored_documents.append(
            (
                score,
                doc
            )
        )

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc
        in scored_documents
    ]


# ============================================================
# 8. 检索文档
# ============================================================

def retrieve_documents(
    question,
    top_k=3,
    db_path=DB_PATH
):
    """
    执行两阶段检索：

    第一阶段：
        Embedding 召回候选 Chunk

    第二阶段：
        关键词 + 原始排名进行轻量级重排序

    最终返回 Top-K。
    """

    vector_db = None

    try:

        # ----------------------------------------------------
        # 获取向量数据库
        # ----------------------------------------------------

        vector_db = get_vector_store(
            db_path
        )

        # ----------------------------------------------------
        # Embedding 召回
        #
        # 多召回一些候选，方便后续重排序
        # ----------------------------------------------------

        candidate_k = max(
            top_k * 3,
            5
        )

        candidates = vector_db.similarity_search(
            question,
            k=candidate_k
        )

        # ----------------------------------------------------
        # 去除重复 Chunk
        # ----------------------------------------------------

        documents = []

        seen_ids = set()

        for doc in candidates:

            chunk_id = doc.metadata.get(
                "chunk_id"
            )

            if chunk_id in seen_ids:

                continue

            seen_ids.add(
                chunk_id
            )

            documents.append(
                doc
            )

        # ----------------------------------------------------
        # 轻量级重排序
        # ----------------------------------------------------

        documents = rerank_documents(
            question,
            documents
        )

        # ----------------------------------------------------
        # 返回 Top-K
        # ----------------------------------------------------

        return documents[:top_k]

    finally:

        # ----------------------------------------------------
        # 非常重要：
        # 主动释放 Chroma 对象
        #
        # Windows 下可以减少文件锁问题
        # ----------------------------------------------------

        vector_db = None

        gc.collect()


# ============================================================
# 9. 调用 Qwen 生成答案
# ============================================================

def generate_answer(
    question,
    documents
):
    """
    根据检索到的知识片段生成最终答案。
    """

    context = "\n\n".join(
        [
            (
                f"[知识片段 "
                f"{doc.metadata.get('chunk_id')}]"
                f"\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]
    )

    prompt = f"""
你是一个知识库问答助手。

请严格根据下面提供的知识库内容回答问题。

如果知识库中没有足够的信息，请明确回答：

“知识库中没有相关信息，我无法回答。”

不要使用知识库之外的知识进行补充或猜测。

【知识库内容】

{context}

【用户问题】

{question}

【回答】
"""

    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        result_format="message"
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"通义千问调用失败：{response}"
        )

    answer = (
        response
        .output
        .choices[0]
        .message
        .content
    )

    return answer


# ============================================================
# 10. 完整 RAG
# ============================================================

def ask_rag(
    question,
    top_k=3,
    db_path=DB_PATH
):
    """
    完整 RAG 流程：

    Question
        ↓
    Retriever
        ↓
    Rerank
        ↓
    Context
        ↓
    Qwen
        ↓
    Answer
    """

    documents = retrieve_documents(
        question,
        top_k=top_k,
        db_path=db_path
    )

    answer = generate_answer(
        question,
        documents
    )

    return {
        "question": question,
        "answer": answer,
        "source_documents": documents
    }


# ============================================================
# 11. 本地测试
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 构建测试向量数据库
    # --------------------------------------------------------

    build_vector_store(
        chunk_size=200,
        chunk_overlap=30
    )

    # --------------------------------------------------------
    # 测试问题
    # --------------------------------------------------------

    question = (
        "Playwright可以怎么编写测试代码?"
    )

    print(
        "\n=============================="
    )

    print("问题：")

    print(
        question
    )

    # --------------------------------------------------------
    # 执行 RAG
    # --------------------------------------------------------

    result = ask_rag(
        question,
        top_k=3
    )

    # --------------------------------------------------------
    # 输出回答
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print("RAG回答：")

    print(
        result["answer"]
    )

    # --------------------------------------------------------
    # 输出检索结果
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print("检索到的知识片段：")

    for doc in result[
        "source_documents"
    ]:

        print(
            f"\n[chunk_id="
            f"{doc.metadata.get('chunk_id')}]"
        )

        print(
            doc.page_content
        )