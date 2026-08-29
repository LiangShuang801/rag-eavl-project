import os
import dashscope
from dotenv import load_dotenv

# =========================
# 读取环境变量
# =========================

load_dotenv()
api_key = os.getenv(
    "DASHSCOPE_API_KEY"
)
dashscope.api_key = api_key

# =========================
# LLM-as-a-Judge
# =========================

def llm_judge(
    question,
    standard_answer,
    model_answer,
    context
):
    prompt = f"""
你是一名专业的RAG系统质量评测工程师。

请根据“参考知识”和“标准答案”，
对RAG系统生成的答案进行评测。

【用户问题】
{question}

【参考知识】
{context}

【标准答案】
{standard_answer}

【模型回答】
{model_answer}

请从以下三个维度进行评分：

1. correctness：答案正确性
2. faithfulness：答案是否忠实于参考知识
3. relevance：答案是否与用户问题相关

每项满分5分：

5分：非常好
4分：基本正确，仅有轻微问题
3分：部分正确
2分：存在明显问题
1分：完全错误

另外判断是否存在幻觉：

hallucination：
0 = 没有明显幻觉
1 = 存在明显幻觉

请严格按照下面JSON格式输出，
不要输出任何额外文字：

{{
    "correctness": 5,
    "faithfulness": 5,
    "relevance": 5,
    "hallucination": 0,
    "reason": "简要说明评分理由"
}}
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
    content = (
        response
        .output
        .choices[0]
        .message
        .content
    )
    return content

# =========================
# 测试
# =========================

if __name__ == "__main__":
    result = llm_judge(
        question="Playwright是谁开发的？",
        standard_answer="Playwright由微软开发。",
        model_answer="Playwright由微软开发，是微软推出的Web自动化测试工具。",
        context="""
Playwright由微软开发，支持Chrome、Edge、Firefox等浏览器，
可以录制脚本，也可以手写PO模式自动化代码。
"""
    )
    print(
        "\n=============================="
    )
    print("LLM Judge结果：")
    print(
        result
    )
    print(
        "=============================="
    )