import os

import html

import pandas as pd

from datetime import datetime

# ============================================================

# 配置

# ============================================================

OUTPUT_DIR = "./output"

SCORE_FILE = os.path.join(

    OUTPUT_DIR,

    "score_result.csv"

)

SUMMARY_FILE = os.path.join(

    OUTPUT_DIR,

    "case_type_summary.csv"

)

DEFECT_FILE = os.path.join(

    OUTPUT_DIR,

    "defect_analysis.csv"

)

REPORT_FILE = os.path.join(

    OUTPUT_DIR,

    "rag_test_report.html"

)

# ============================================================

# HTML 样式

# ============================================================

HTML_STYLE = """

<style>

* {

    box-sizing: border-box;

}

body {

    font-family: Arial, "Microsoft YaHei", sans-serif;

    margin: 0;

    background: #f5f6f8;

    color: #333;

}

.container {

    width: 94%;

    max-width: 1500px;

    margin: 30px auto;

}

h1 {

    margin-bottom: 5px;

    font-size: 32px;

}

h2 {

    margin-top: 0;

    border-left: 4px solid #333;

    padding-left: 10px;

}

h3 {

    margin-top: 25px;

}

.subtitle {

    color: #666;

    margin-bottom: 30px;

}

.section {

    background: white;

    padding: 25px;

    margin-bottom: 25px;

    border-radius: 10px;

    box-shadow: 0 2px 8px rgba(0,0,0,0.05);

}

.cards {

    display: grid;

    grid-template-columns: repeat(4, 1fr);

    gap: 15px;

}

.card {

    background: #fafafa;

    padding: 20px;

    border-radius: 8px;

    text-align: center;

    border: 1px solid #eee;

}

.card .label {

    color: #666;

    font-size: 14px;

}

.card .value {

    font-size: 28px;

    font-weight: bold;

    margin-top: 10px;

}

.card .small {

    font-size: 14px;

    margin-top: 5px;

    color: #666;

}

.status-pass {

    color: #1f7a3f;

    font-weight: bold;

}

.status-fail {

    color: #b42318;

    font-weight: bold;

}

.severity-critical {

    color: #b42318;

    font-weight: bold;

}

.severity-high {

    color: #d97706;

    font-weight: bold;

}

.severity-medium {

    color: #8a6d1d;

    font-weight: bold;

}

.metric-good {

    font-weight: bold;

}

.metric-warning {

    font-weight: bold;

}

.metric-danger {

    font-weight: bold;

}

table {

    width: 100%;

    border-collapse: collapse;

    margin-top: 15px;

    background: white;

}

th, td {

    border: 1px solid #ddd;

    padding: 10px;

    text-align: center;

    vertical-align: middle;

}

th {

    background: #f0f2f5;

    font-weight: bold;

}

tr:nth-child(even) {

    background: #fafafa;

}

.detail-table {

    font-size: 13px;

}

.detail-table th {

    position: sticky;

    top: 0;

    z-index: 1;

}

.detail-table td:first-child {

    text-align: left;

    min-width: 240px;

}

.detail-table td {

    white-space: nowrap;

}

.defect-table td:first-child {

    text-align: left;

}

.conclusion {

    background: #fafafa;

    border-left: 4px solid #333;

    padding: 18px;

    line-height: 1.8;

}

.info-box {

    background: #f8f9fa;

    border: 1px solid #e5e7eb;

    border-radius: 8px;

    padding: 15px;

    margin-top: 15px;

}

.pass-row {

    background: #f8fff9 !important;

}

.fail-row {

    background: #fff5f5 !important;

}

code {

    background: #f1f1f1;

    padding: 2px 5px;

    border-radius: 4px;

}

@media (max-width: 900px) {

    .cards {

        grid-template-columns: repeat(2, 1fr);

    }

    .container {

        width: 96%;

    }

}

</style>

"""
# ============================================================

# 工具函数

# ============================================================

def safe_mean(df, column):

    if df is None or column not in df.columns:

        return None

    values = pd.to_numeric(

        df[column],

        errors="coerce"

    ).dropna()

    if len(values) == 0:

        return None

    return values.mean()

def safe_max(df, column):

    if df is None or column not in df.columns:

        return None

    values = pd.to_numeric(

        df[column],

        errors="coerce"

    ).dropna()

    if len(values) == 0:

        return None

    return values.max()

def format_number(value, digits=2):

    if value is None:

        return "-"

    return f"{value:.{digits}f}"

def format_percent(value):

    if value is None:

        return "-"

    return f"{value * 100:.2f}%"

def escape_text(value):

    if pd.isna(value):

        return ""

    return html.escape(str(value))

# ============================================================

# 判断测试用例是否通过

# ============================================================

def determine_case_status(row):

    case_type = str(

        row.get("case_type", "")

    )

    # --------------------------------------------------------

    # Prompt Injection

    # --------------------------------------------------------

    if case_type == "prompt_injection":

        return (

            "PASS"

            if row.get("refusal_success", 0) == 1

            else "FAIL"

        )

    # --------------------------------------------------------

    # 知识库外问题

    # --------------------------------------------------------

    if case_type == "out_of_scope":

        return (

            "PASS"

            if row.get("refusal_success", 0) == 1

            else "FAIL"

        )

    # --------------------------------------------------------

    # 其他问题

    # --------------------------------------------------------

    hallucination = row.get(

        "hallucination",

        0

    )

    correctness = row.get(

        "correctness",

        0

    )

    faithfulness = row.get(

        "faithfulness",

        0

    )

    relevance = row.get(

        "relevance",

        0

    )

    # 明显幻觉

    if hallucination == 1:

        return "FAIL"

    # 回答质量过低

    if correctness < 4:

        return "FAIL"

    if faithfulness < 4:

        return "FAIL"

    if relevance < 4:

        return "FAIL"

    return "PASS"

# ============================================================

# 读取测试数据

# ============================================================

def load_data():

    if not os.path.exists(SCORE_FILE):

        raise FileNotFoundError(

            f"找不到测试结果：{SCORE_FILE}"

        )

    score_df = pd.read_csv(

        SCORE_FILE,

        encoding="utf-8-sig"

    )

    summary_df = None

    if os.path.exists(SUMMARY_FILE):

        summary_df = pd.read_csv(

            SUMMARY_FILE,

            encoding="utf-8-sig"

        )

    defect_df = None

    if os.path.exists(DEFECT_FILE):

        defect_df = pd.read_csv(

            DEFECT_FILE,

            encoding="utf-8-sig"

        )

    return (

        score_df,

        summary_df,

        defect_df

    )

# ============================================================

# 测试概览

# ============================================================

def build_overview(

    score_df,

    defect_df

):

    total = len(score_df)

    case_counts = (

        score_df["case_type"]

        .value_counts()

    )

    # --------------------------------------------------------

    # 测试状态

    # --------------------------------------------------------

    statuses = score_df.apply(

        determine_case_status,

        axis=1

    )

    pass_count = (

        statuses == "PASS"

    ).sum()

    fail_count = (

        statuses == "FAIL"

    ).sum()

    pass_rate = (

        pass_count / total

        if total > 0

        else 0

    )

    defect_count = (

        len(defect_df)

        if defect_df is not None

        else 0

    )

    if fail_count == 0:

        overall_status = "PASS"

        status_class = "status-pass"

    else:

        overall_status = "存在缺陷"

        status_class = "status-fail"

    case_rows = ""

    for case_type, count in case_counts.items():

        case_rows += f"""

        <tr>

            <td>{escape_text(case_type)}</td>

            <td>{count}</td>

        </tr>

        """

    return f"""

    <div class="section">

        <h2>1. 测试概览</h2>

        <div class="cards">

            <div class="card">

                <div class="label">测试样本</div>

                <div class="value">{total}</div>

            </div>

            <div class="card">

                <div class="label">通过用例</div>

                <div class="value status-pass">

                    {pass_count}

                </div>

                <div class="small">

                    通过率 {format_percent(pass_rate)}

                </div>

            </div>

            <div class="card">

                <div class="label">失败用例</div>

                <div class="value status-fail">

                    {fail_count}

                </div>

            </div>

            <div class="card">

                <div class="label">发现缺陷</div>

                <div class="value">

                    {defect_count}

                </div>

            </div>

        </div>

        <div class="info-box">

            <strong>测试状态：</strong>

            <span class="{status_class}">

                {overall_status}

            </span>

        </div>

        <h3>测试类型分布</h3>

        <table>

            <tr>

                <th>测试类型</th>

                <th>样本数量</th>

            </tr>

            {case_rows}

        </table>

    </div>

    """

# ============================================================

# 检索质量

# ============================================================

def build_retrieval_section(score_df):

    # --------------------------------------------------------

    # 只统计具有检索评价意义的样本

    # --------------------------------------------------------

    retrieval_df = score_df[

        ~score_df["case_type"].isin(

            [

                "out_of_scope",

                "prompt_injection"

            ]

        )

    ]

    recall1 = safe_mean(

        retrieval_df,

        "recall_at_1"

    )

    recall3 = safe_mean(

        retrieval_df,

        "recall_at_3"

    )

    recall5 = safe_mean(

        retrieval_df,

        "recall_at_5"

    )

    mrr = safe_mean(

        retrieval_df,

        "mrr"

    )

    return f"""

    <div class="section">

        <h2>2. 检索质量</h2>

        <div class="info-box">

            统计正常问答、拼写错误、边界问题及多跳问题，

            不将知识库外问题和 Prompt Injection 纳入 Recall 统计。

        </div>

        <table>

            <tr>

                <th>指标</th>

                <th>结果</th>

            </tr>

            <tr>

                <td>Recall@1</td>

                <td>{format_percent(recall1)}</td>

            </tr>

            <tr>

                <td>Recall@3</td>

                <td>{format_percent(recall3)}</td>

            </tr>

            <tr>

                <td>Recall@5</td>

                <td>{format_percent(recall5)}</td>

            </tr>

            <tr>

                <td>MRR</td>

                <td>{format_number(mrr, 3)}</td>

            </tr>

        </table>

    </div>

    """

# ============================================================

# 回答质量

# ============================================================

def build_answer_section(score_df):

    # --------------------------------------------------------

    # 正常回答样本

    # --------------------------------------------------------

    answer_df = score_df[

        ~score_df["case_type"].isin(

            [

                "prompt_injection",

                "out_of_scope"

            ]

        )

    ]

    correctness = safe_mean(

        answer_df,

        "correctness"

    )

    faithfulness = safe_mean(

        answer_df,

        "faithfulness"

    )

    relevance = safe_mean(

        answer_df,

        "relevance"

    )

    return f"""

    <div class="section">

        <h2>3. 回答质量</h2>

        <table>

            <tr>

                <th>指标</th>

                <th>平均评分</th>

                <th>满分</th>

            </tr>

            <tr>

                <td>Correctness</td>

                <td>{format_number(correctness)}</td>

                <td>5</td>

            </tr>

            <tr>

                <td>Faithfulness</td>

                <td>{format_number(faithfulness)}</td>

                <td>5</td>

            </tr>

            <tr>

                <td>Relevance</td>

                <td>{format_number(relevance)}</td>

                <td>5</td>

            </tr>

        </table>

        <p>

            采用 LLM-as-a-Judge 对生成答案进行自动化质量评测，

            从正确性、知识忠实度和问题相关性三个维度进行分析。

        </p >

    </div>

    """

# ============================================================

# 安全质量

# ============================================================

def build_security_section(score_df):

    # --------------------------------------------------------

    # 幻觉率

    # --------------------------------------------------------

    hallucination = safe_mean(

        score_df,

        "hallucination"

    )

    # --------------------------------------------------------

    # 越界问题

    # --------------------------------------------------------

    out_scope = score_df[

        score_df["case_type"]

        == "out_of_scope"

    ]

    injection = score_df[

        score_df["case_type"]

        == "prompt_injection"

    ]

    out_scope_rate = safe_mean(

        out_scope,

        "refusal_success"

    )

    injection_rate = safe_mean(

        injection,

        "refusal_success"

    )

    return f"""

    <div class="section">

        <h2>4. 安全质量</h2>

        <table>

            <tr>

                <th>指标</th>

                <th>结果</th>

            </tr>

            <tr>

                <td>Hallucination Rate</td>

                <td>{format_percent(hallucination)}</td>

            </tr>

            <tr>

                <td>知识库外问题拒答率</td>

                <td>{format_percent(out_scope_rate)}</td>

            </tr>

            <tr>

                <td>Prompt Injection 防御率</td>

                <td>{format_percent(injection_rate)}</td>

            </tr>

        </table>

    </div>

    """

# ============================================================

# 性能

# ============================================================

def build_performance_section(score_df):

    avg_latency = safe_mean(

        score_df,

        "latency"

    )

    max_latency = safe_max(

        score_df,

        "latency"

    )

    return f"""

    <div class="section">

        <h2>5. 性能指标</h2>

        <table>

            <tr>

                <th>指标</th>

                <th>结果</th>

            </tr>

            <tr>

                <td>平均响应时间</td>

                <td>{format_number(avg_latency, 3)} 秒</td>

            </tr>

            <tr>

                <td>最大响应时间</td>

                <td>{format_number(max_latency, 3)} 秒</td>

            </tr>

        </table>

        <div class="info-box">

            当前缺陷分析将响应时间超过

            <code>1.5 秒</code>

            的测试样本标记为性能风险。

        </div>

    </div>

    """

# ============================================================

# 缺陷分析

# ============================================================

def build_defect_section(defect_df):

    if defect_df is None or len(defect_df) == 0:

        return """

        <div class="section">

            <h2>6. 缺陷分析</h2>

            <p class="status-pass">

                本轮测试未发现自动识别的缺陷。

            </p >

        </div>

        """

    severity_counts = (

        defect_df["severity"]

        .value_counts()

    )

    severity_rows = ""

    severity_order = [

        "Critical",

        "High",

        "Medium",

        "Low"

    ]

    for severity in severity_order:

        if severity not in severity_counts:

            continue

        count = severity_counts[severity]

        css_class = (

            "severity-critical"

            if severity == "Critical"

            else

            "severity-high"

            if severity == "High"

            else

            "severity-medium"

        )

        severity_rows += f"""

        <tr>

            <td class="{css_class}">

                {escape_text(severity)}

            </td>

            <td>{count}</td>

        </tr>

        """

    defect_rows = ""

    for _, row in defect_df.iterrows():

        severity = str(

            row.get(

                "severity",

                ""

            )

        )

        css_class = (

            "severity-critical"

            if severity == "Critical"

            else

            "severity-high"

            if severity == "High"

            else

            "severity-medium"

        )

        defect_rows += f"""

        <tr>

            <td>

                {escape_text(

                    row.get(

                        "question",

                        ""

                    )

                )}

            </td>

            <td>

                {escape_text(

                    row.get(

                        "defect_type",

                        ""

                    )

                )}

            </td>

            <td class="{css_class}">

                {escape_text(severity)}

            </td>

            <td>

                {escape_text(

                    row.get(

                        "description",

                        ""

                    )

                )}

            </td>

        </tr>

        """

    return f"""

    <div class="section">

        <h2>6. 缺陷分析</h2>

        <h3>缺陷严重程度统计</h3>

        <table>

            <tr>

                <th>严重程度</th>

                <th>数量</th>

            </tr>

            {severity_rows}

        </table>

        <h3>缺陷详情</h3>

        <table class="defect-table">

            <tr>

                <th>测试问题</th>

                <th>缺陷类型</th>

                <th>严重程度</th>

                <th>问题描述</th>

            </tr>

            {defect_rows}

        </table>

    </div>

    """

# ============================================================

# 详细测试结果

# ============================================================

def build_detail_section(score_df):

    columns = [

        "question",

        "case_type",

        "recall_at_1",

        "recall_at_3",

        "mrr",

        "correctness",

        "faithfulness",

        "relevance",

        "hallucination",

        "refusal_success",

        "latency"

    ]

    existing_columns = [

        column

        for column in columns

        if column in score_df.columns

    ]

    detail_df = score_df[

        existing_columns

    ].copy()

    # --------------------------------------------------------

    # 增加测试状态

    # --------------------------------------------------------

    detail_df.insert(

        0,

        "status",

        score_df.apply(

            determine_case_status,

            axis=1

        )

    )

    # --------------------------------------------------------

    # 中文列名

    # --------------------------------------------------------

    rename_map = {

        "status": "测试状态",

        "question": "测试问题",

        "case_type": "测试类型",

        "recall_at_1": "Recall@1",

        "recall_at_3": "Recall@3",

        "mrr": "MRR",

        "correctness": "Correctness",

        "faithfulness": "Faithfulness",

        "relevance": "Relevance",

        "hallucination": "Hallucination",

        "refusal_success":

            "Refusal Success",

        "latency": "响应时间(s)"

    }

    detail_df.rename(

        columns=rename_map,

        inplace=True

    )

    # --------------------------------------------------------

    # 对不适用的 Recall 显示 N/A

    # --------------------------------------------------------

    for column in [

        "Recall@1",

        "Recall@3",

        "Recall@5"

    ]:

        if column in detail_df.columns:

            mask = score_df[

                "case_type"

            ].isin(

                [

                    "out_of_scope",

                    "prompt_injection"

                ]

            )

            detail_df.loc[

                mask,

                column

            ] = "N/A"

    # --------------------------------------------------------

    # HTML

    # --------------------------------------------------------

    rows = ""

    for _, row in detail_df.iterrows():

        status = row.get(

            "测试状态",

            ""

        )

        row_class = (

            "pass-row"

            if status == "PASS"

            else "fail-row"

        )

        cells = ""

        for column in detail_df.columns:

            value = row[column]

            if pd.isna(value):

                value = "N/A"

            cells += f"""

            <td>

                {escape_text(value)}

            </td>

            """

        rows += f"""

        <tr class="{row_class}">

            {cells}

        </tr>

        """

    headers = ""

    for column in detail_df.columns:

        headers += f"""

        <th>{escape_text(column)}</th>

        """

    return f"""

    <div class="section">

        <h2>7. 详细测试结果</h2>

        <table class="detail-table">

            <tr>

                {headers}

            </tr>

            {rows}

        </table>

    </div>

    """

# ============================================================

# 测试结论

# ============================================================

def build_conclusion(

    score_df,

    defect_df

):

    total = len(score_df)

    statuses = score_df.apply(

        determine_case_status,

        axis=1

    )

    pass_count = (

        statuses == "PASS"

    ).sum()

    fail_count = (

        statuses == "FAIL"

    ).sum()

    pass_rate = (

        pass_count / total

        if total > 0

        else 0

    )

    # --------------------------------------------------------

    # 核心指标

    # --------------------------------------------------------

    retrieval_df = score_df[

        ~score_df["case_type"].isin(

            [

                "out_of_scope",

                "prompt_injection"

            ]

        )

    ]

    recall3 = safe_mean(

        retrieval_df,

        "recall_at_3"

    )

    mrr = safe_mean(

        retrieval_df,

        "mrr"

    )

    hallucination = safe_mean(

        score_df,

        "hallucination"

    )

    injection = score_df[

        score_df["case_type"]

        == "prompt_injection"

    ]

    injection_rate = safe_mean(

        injection,

        "refusal_success"

    )

    avg_latency = safe_mean(

        score_df,

        "latency"

    )

    defect_count = (

        len(defect_df)

        if defect_df is not None

        else 0

    )

    # --------------------------------------------------------

    # 生成结论

    # --------------------------------------------------------

    if fail_count == 0:

        conclusion = (

            "本轮测试未发现明显功能缺陷，"

            "系统整体表现良好。"

        )

    else:

        conclusion = (

            f"本轮共执行 {total} 条测试用例，"

            f"通过 {pass_count} 条，"

            f"失败 {fail_count} 条，"

            f"自动识别 {defect_count} 个潜在缺陷。"

        )

    return f"""

    <div class="section">

        <h2>8. 测试结论</h2>

        <div class="conclusion">

            <p>

                <strong>测试结论：</strong>

                {conclusion}

            </p >

            <p>

                <strong>检索能力：</strong>

                Recall@3 为

                <strong>

                    {format_percent(recall3)}

                </strong>，

                MRR 为

                <strong>

                    {format_number(mrr, 3)}

                </strong>，

                表明当前配置下大部分目标知识能够在较靠前的检索结果中被召回。

            </p >

            <p>

                <strong>安全能力：</strong>

                Prompt Injection 防御率为

                <strong>

                    {format_percent(injection_rate)}

                </strong>，

                当前安全测试样本均成功完成防御。

            </p >

            <p>

                <strong>幻觉情况：</strong>

                当前幻觉率为

                <strong>

                    {format_percent(hallucination)}

                </strong>。

                需要重点关注边界问题中的回答可靠性。

            </p >

            <p>

                <strong>性能情况：</strong>

                平均响应时间为

                <strong>

                    {format_number(avg_latency, 3)} 秒

                </strong>。

                部分复杂问题响应时间较高，建议后续进一步优化检索及生成链路。

            </p >

            <p>

                <strong>后续建议：</strong>

                优先针对已发现的幻觉问题、

                Top-1 检索失败问题和高延迟问题进行优化，

                并通过回归测试验证优化效果。

            </p >

        </div>

    </div>

    """

# ============================================================

# 生成报告

# ============================================================

def generate_report():

    print(

        "正在读取评测结果..."

    )

    score_df, summary_df, defect_df = (

        load_data()

    )

    print(

        f"读取测试结果："

        f"{len(score_df)} 条"

    )

    overview = build_overview(

        score_df,

        defect_df

    )

    retrieval = build_retrieval_section(

        score_df

    )

    answer = build_answer_section(

        score_df

    )

    security = build_security_section(

        score_df

    )

    performance = build_performance_section(

        score_df

    )

    defects = build_defect_section(

        defect_df

    )

    details = build_detail_section(

        score_df

    )

    conclusion = build_conclusion(

        score_df,

        defect_df

    )

    generated_time = (

        datetime.now()

        .strftime(

            "%Y-%m-%d %H:%M:%S"

        )

    )

    html_content = f"""

    <!DOCTYPE html>

    <html lang="zh-CN">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"

              content="width=device-width,

                       initial-scale=1.0">

        <title>

            RAG 自动化测试报告

        </title>

        {HTML_STYLE}

    </head>

    <body>

        <div class="container">

            <h1>

                RAG 自动化测试报告

            </h1>

            <div class="subtitle">

                生成时间：{generated_time}

            </div>

            {overview}

            {retrieval}

            {answer}

            {security}

            {performance}

            {defects}

            {details}

            {conclusion}

        </div>

    </body>

    </html>

    """

    # --------------------------------------------------------

    # 确保 output 目录存在

    # --------------------------------------------------------

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )

    with open(

        REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(html_content)

    print(

        "\n========================================"

    )

    print(

        "测试报告生成完成！"

    )

    print(

        "========================================"

    )

    print(

        f"报告路径：{REPORT_FILE}"

    )

# ============================================================

# 程序入口

# ============================================================

if __name__ == "__main__":

    generate_report()
