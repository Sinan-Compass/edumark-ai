"""
师评智伴 EduMark AI — Streamlit 版
教师作业批改助手：支持语文、数学、英语、计算机、生物五学科
用户自行提供 API Key，选择智谱/DeepSeek/千问等 OpenAI 兼容接口
"""
import json
import re
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="师评智伴 · 教师作业批改助手",
    page_icon="评",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 常量 & 数据
# =============================================================================
SUBJECT_NAMES = ["语文", "数学", "英语", "计算机", "生物"]
MAX_CACHE_PER_SUBJECT = 30
MAX_ANALYSIS_SAMPLES = 20

PROVIDERS = {
    "智谱 (GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "models": [
            "glm-4-plus",           # 旗舰大模型，128K上下文
            "glm-4.7",              # 最新旗舰，200K上下文，高智能
            "glm-4.6",              # 超强编码/推理，200K上下文
            "glm-4-flash-250414",   # 免费模型，128K上下文
            "glm-4.7-flash",        # 最新免费模型，200K上下文
            "glm-4-long",           # 超长文本，1M上下文
            "glm-4-air-250414",     # 高性价比轻量模型
        ],
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": [
            "deepseek-v4-pro",      # V4 旗舰，685B 参数（1.6T MoE），1M 上下文
            "deepseek-v4-flash",    # V4 轻量版，快且便宜，1M 上下文
        ],
    },
    "千问 (Qwen)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
        "models": [
            "qwen-max",             # Qwen2.5 旗舰，最强性能
            "qwen-plus",            # Qwen2.5 增强版，性价比高
            "qwen-turbo",           # Qwen2.5 速度优先
            "qwen-flash",           # Qwen2.5 轻量快速
            "qwen3-max",            # Qwen3 最新旗舰，支持思考模式
            "qwen3-plus",           # Qwen3 增强版，128K-1M上下文
            "qwen3-turbo",          # Qwen3 速度优化版
            "qwen3-flash",          # Qwen3 低延迟版
            "qwq-plus",             # 商业推理模型
        ],
    },
}

OUTPUT_SCHEMA = """{
  "report_type": "grading",
  "subject": "学科名称",
  "student_id": "学生编号",
  "title": "作业题目",
  "total_score": 0,
  "grade": "等级",
  "summary": "总体评语",
  "dimensions": [
    {"name": "评分维度", "max_score": 0, "score": 0, "evidence": "...", "comment": "..."}
  ],
  "strengths": [{"title": "...", "evidence": "..."}],
  "issues": [{"title": "...", "evidence": "...", "impact": "..."}],
  "suggestions": [{"action": "...", "method": "..."}],
  "special_checks": [{"name": "...", "triggered": false, "explanation": "..."}],
  "subject_details": {}
}"""

ANALYSIS_SCHEMA = """{
  "report_type": "analysis",
  "subject": "学科名称",
  "sample_size": 0,
  "average_score": 0,
  "overview": "总体表现概述",
  "score_distribution": [
    {"level": "优秀", "range": "90-100", "count": 0, "percentage": 0}
  ],
  "dimension_stats": [
    {"name": "维度", "max_score": 0, "average_score": 0, "rate": 0, "judgement": "表现判断"}
  ],
  "common_strengths": [{"title": "共性优点", "evidence": "..."}],
  "common_issues": [{"title": "共性问题", "diagnosis": "...", "evidence": "..."}],
  "student_profiles": [{"student_id": "...", "type": "表现突出/重点关注", "reason": "..."}],
  "teaching_suggestions": [{"action": "...", "method": "...", "goal": "..."}],
  "limitations": ["数据局限说明"]
}"""

# ── 学科配置 ─────────────────────────────────────────────────────────────
SUBJECT_CONFIGS = {
    "语文": {
        "label": "语文作文",
        "reference_visible": False,
        "reference_title": "写作要求或参考材料",
        "analysis_focus": "重点统计立意是否明确、文本细节使用、生活联系、结构表达，以及离题、复述和套话等共性现象。",
        "rubric": """你是一名经验丰富的中学语文教师。请对作文、读后感或议论文进行证据充分、尺度稳定的批改。

【100分量规】
1. 主旨理解与内容立意（30分）：中心明确、切合题意、思想健康，有真实且深入的认识。
2. 材料与细节分析（25分）：材料具体，细节能够支撑观点；读后感不能停留在情节复述。
3. 结构与逻辑（20分）：结构完整，层次清晰，过渡自然，详略得当。
4. 语言与表达（20分）：语言通顺准确，有表现力，符合文体要求。
5. 规范与创新（5分）：标点、错别字、格式规范，具有个性化思考。

【封顶规则】
- 严重离题：总分不高于40分。
- 主要复述、缺少分析与感悟：总分不高于65分。
- 少于100字：总分不高于50分；少于50字：不高于30分。
- 分项得分之和必须等于总分。

subject_details 必须为：
{"writing_type":"文体判断","cap_rule":{"triggered":false,"rule":"未触发或具体规则","reason":"依据"},"revision_focus":"最优先修改方向"}""",
    },
    "数学": {
        "label": "数学解答题",
        "reference_visible": True,
        "reference_title": "题目与参考答案",
        "analysis_focus": "重点统计逻辑、计算、步骤和规范四项能力，归纳首个错误类型，区分概念、方法、计算与表达问题。",
        "rubric": """你是一名严谨耐心的中学数学教师。请检查思路、计算、步骤和数学表达，定位第一个关键错误。

【100分量规】
1. 逻辑严密性（40分）：思路清晰，推理有据，无逻辑跳跃或定理误用。
2. 计算准确性（30分）：数值、代数运算和符号变形准确。
3. 步骤完整性（20分）：关键步骤齐全，不只写最终结果。
4. 表达规范性（10分）：符号、格式和说明规范清晰。

【封顶规则】
- 空白或只抄题：不高于10分。
- 只有最终答案、没有过程：不高于30分。
- 解题方法完全错误：不高于40分。
- 无参考答案时必须提示教师复核。
- 分项得分之和必须等于总分。

subject_details 必须为：
{"is_correct":false,"first_error_step":"首个错误步骤或无","error_type":"错误类型","error_reason":"错误原因","correction_steps":["纠正步骤1","纠正步骤2"],"teacher_review_note":"复核提示"}""",
    },
    "英语": {
        "label": "英语作文",
        "reference_visible": False,
        "reference_title": "写作要求",
        "analysis_focus": "重点统计内容、结构、语法、词汇、拼写五项能力，归纳高频语言错误及可迁移的写作教学重点。",
        "rubric": """You are an experienced English writing teacher for Chinese middle-school students. Give specific, constructive feedback in Chinese while preserving quoted English.

【100-point rubric】
1. Content & Ideas (25): clear topic, sufficient content and concrete support.
2. Organization & Structure (25): effective beginning, body, ending and transitions.
3. Grammar Accuracy (20): tense, voice, agreement and sentence structure.
4. Vocabulary & Expression (20): accurate and natural wording without Chinglish.
5. Spelling & Punctuation (10): spelling, punctuation and capitalization.

【Rules】
- Blank or wholly irrelevant writing is capped at 10.
- Fewer than 30 English words is capped at 30.
- All-Chinese writing is not a valid English composition.
- Category scores must equal the total.

subject_details 必须为：
{"word_count":0,"language_errors":[{"type":"错误类型","original":"原文","explanation":"中文解释","correction":"修改"}],"improved_version":"保留学生原意的完整英文修改版"}""",
    },
    "计算机": {
        "label": "计算机作业",
        "reference_visible": True,
        "reference_title": "任务要求、测试用例与参考实现",
        "analysis_focus": "重点统计功能正确性、算法逻辑、代码质量、测试健壮性和知识解释，归纳高频代码缺陷与测试盲区。",
        "rubric": """你是一名中学信息科技与计算机编程教师。请从功能、算法、代码质量和解释能力评价学生的程序设计、伪代码或信息科技作业。不要执行代码，不要声称已经实际运行；只能依据文本静态分析。

【100分量规】
1. 功能正确性（35分）：是否满足题目要求，输入、处理与输出是否正确，边界情况是否覆盖。
2. 算法与逻辑（25分）：算法合理，流程清楚，无明显逻辑漏洞或不必要复杂度。
3. 代码质量（20分）：命名、结构、可读性、重复、注释和模块化。
4. 测试与健壮性（10分）：能否处理异常输入、边界值，是否给出测试说明。
5. 知识理解与表达（10分）：能够解释关键语句、概念和设计选择。

【规则】
- 完全无法运行或核心功能缺失：功能正确性不高于12分。
- 只给代码无任何解释时，知识理解与表达不高于5分。
- 不得编造运行结果；缺少测试用例时应标记"需实际运行复核"。
- 分项得分之和必须等于总分。

subject_details 必须为：
{"task_type":"程序设计/概念题/其他","static_assessment":"静态分析结论","code_issues":[{"location":"位置或代码片段","type":"问题类型","original":"原代码","explanation":"原因","fix":"修改建议"}],"test_cases":[{"input":"建议输入","expected":"预期结果","purpose":"测试目的"}],"improved_code":"可选的改进代码；不适用时为空字符串","runtime_note":"必须说明未实际运行"}""",
    },
    "生物": {
        "label": "生物作业",
        "reference_visible": True,
        "reference_title": "题目、参考答案与评分要点",
        "analysis_focus": "重点统计概念事实、科学推理、实验探究、证据使用和科学表达，归纳概念混淆、变量控制及结论过度等问题。",
        "rubric": """你是一名严谨的中学生物教师。请依据生物学事实、概念准确性、证据推理和科学表达批改简答题、实验题或探究报告。

【100分量规】
1. 核心概念与事实（35分）：生物学概念、结构、功能、过程和术语准确。
2. 科学推理与因果解释（25分）：能用证据建立合理因果关系，不混淆相关与因果。
3. 实验与探究能力（20分）：变量、对照、步骤、数据、结论和误差分析合理。
4. 证据使用与完整性（10分）：回答紧扣材料、图表或实验现象，关键点完整。
5. 科学表达规范（10分）：术语规范，表达清楚，避免拟人化和绝对化。

【规则】
- 核心概念完全错误导致结论相反时，总分不高于45分。
- 实验题未设置对照或混淆自变量、因变量，应在实验维度明显扣分。
- 不得编造题目未提供的实验数据。
- 分项得分之和必须等于总分。

subject_details 必须为：
{"question_type":"简答题/实验题/探究报告/其他","concept_errors":[{"concept":"概念","student_statement":"学生表述","correction":"正确表述","explanation":"解释"}],"reasoning_assessment":"科学推理评价","experiment_review":{"variables":"变量判断","control":"对照设置","procedure":"步骤评价","data_conclusion":"数据与结论评价"},"model_answer":"参考作答或改进版答案"}""",
    },
}

ANALYSIS_PROMPT_BASE = """你是一名教学数据分析师。请只根据提供的多份结构化批改记录生成学情报告。

要求：
1. 所有统计数字必须来自输入，不得编造。
2. 计算总体平均分、等级分布、各维度平均分与得分率。
3. 共性优点和问题必须给出记录中的证据。
4. 区分知识能力问题、方法问题和学习习惯问题。
5. 给出3至5条可执行教学建议，写清动作、实施方法和目标。
6. 样本少于10份时，在 limitations 中明确"小样本，结论仅供参考"。
7. 只输出一个合法 JSON 对象，禁止 Markdown、代码围栏、解释性前言和结尾。

输出结构必须严格遵守：
""" + ANALYSIS_SCHEMA

TONE_RULES = {
    "均衡": "语气均衡严谨，既肯定具体优点，也明确指出问题。",
    "鼓励": "语气温和鼓励，先肯定再纠正，但不回避实质问题。",
    "严格": "严格按量规扣分，逐项给出证据，避免虚高分数。",
    "简洁": "表达精炼，保留关键证据和可执行建议，避免重复。",
}

EXAMPLES = [
    {
        "id": "cn-reading", "subject": "语文",
        "name": "《背影》读后感（较好）",
        "description": "有文本细节和生活联系，适合观察语文量规评分。",
        "student": "八年级1班-示例01", "title": "《背影》读后感", "reference": "",
        "content": "读完朱自清的《背影》，我最难忘的是父亲翻过月台去买橘子的背影。他穿着臃肿的棉袍，走路很不方便，却仍坚持为儿子做这件小事。过去我总觉得父母的关心是理所当然的，有时还会嫌妈妈反复提醒我带伞、早点回家。现在我明白，真正的爱往往不是响亮的话，而是藏在一次次不起眼的行动里。以后我不仅要接受父母的爱，也要学会理解他们，在生活的小事中表达感谢。"
    },
    {
        "id": "cn-weak", "subject": "语文",
        "name": "校园生活作文（待改进）",
        "description": "内容较空泛，可测试具体证据与修改建议。",
        "student": "七年级2班-示例02", "title": "难忘的一天", "reference": "",
        "content": "今天是难忘的一天。早上我来到学校，看见同学们都很高兴。我们上了语文课、数学课和英语课。中午我吃了饭，下午又上课。放学以后我回家写作业。我觉得这一天非常有意义，让我懂得了要珍惜时间。这真是难忘的一天。"
    },
    {
        "id": "math-correct", "subject": "数学",
        "name": "一元二次方程（正确解答）",
        "description": "步骤完整的正确答案，适合验证高分报告。",
        "student": "八年级2班-示例03", "title": "一元二次方程应用题",
        "reference": "题目：长方形花园的长比宽多3米，面积为40平方米，求长和宽。\n参考：设宽为x米，x(x+3)=40，解得x=5或-8，舍负值，宽5米、长8米。",
        "content": "解：设宽为x米，长为x+3米。\nx(x+3)=40\nx²+3x-40=0\n(x+8)(x-5)=0\n所以x=-8或x=5。\n因为宽不能为负数，所以宽为5米，长为8米。"
    },
    {
        "id": "math-error", "subject": "数学",
        "name": "行程问题（计算错误）",
        "description": "包含除法颠倒错误，可测试首个错误定位。",
        "student": "七年级4班-示例04", "title": "一次函数行程问题",
        "reference": "甲乙两地相距120千米。前2小时每小时40千米，之后每小时50千米。剩余时间应为(120-80)÷50=0.8小时。",
        "content": "前2小时行驶40×2=80千米，剩下120-80=40千米。汽车还要行驶50÷40=1.25小时，所以还需1.25小时到达。"
    },
    {
        "id": "en-weekend", "subject": "英语",
        "name": "My Weekend（语法错误）",
        "description": "包含时态、搭配和副词错误，可展示 Error List。",
        "student": "七年级3班-示例05", "title": "My Weekend", "reference": "",
        "content": "Last weekend, I had a busy but meaningful time. On Saturday morning, I finished my homework and helped my mother cleaned the room. In the afternoon, I went to the library with my best friend. We read some interesting books and discuss our favorite stories together. Although I lose the chess game on Sunday, I learned a lot. I was tired, but I felt very happily."
    },
    {
        "id": "en-health", "subject": "英语",
        "name": "How to Keep Healthy（多处错误）",
        "description": "适合测试主谓一致、词形和中式英语诊断。",
        "student": "八年级3班-示例06", "title": "How to Keep Healthy", "reference": "",
        "content": "If we wants keep healthy, we should eat more vegetable and fruit. Doing exercise are also important. I play basketball two time a week, it make me strong. We must go to bed early and don't play phone for a long time. Many students sleeps late, so they can't listen teacher carefully."
    },
    {
        "id": "cs-python", "subject": "计算机",
        "name": "Python 求平均值（逻辑问题）",
        "description": "包含除数错误和边界问题，可展示代码诊断及测试用例。",
        "student": "八年级信息科技-示例07", "title": "计算若干成绩的平均值",
        "reference": "要求：输入用空格分隔的若干成绩，输出平均值，保留两位小数；输入为空时给出提示。\n示例：80 90 100 → 90.00。",
        "content": "scores = input().split()\ntotal = 0\nfor score in scores:\n    total += int(score)\naverage = total / (len(scores) - 1)\nprint('平均分：' + average)"
    },
    {
        "id": "cs-concept", "subject": "计算机",
        "name": "网络安全简答题（概念混淆）",
        "description": "非编程作业，用于测试计算机概念题批改。",
        "student": "七年级信息科技-示例08", "title": "如何设置安全密码",
        "reference": "要点：长度足够；混合字符；避免个人信息和常见词；不同账号不复用；开启多因素认证；不向他人泄露。",
        "content": "安全密码应该用自己的生日和名字，这样自己不会忘记。所有网站用同一个密码比较方便。如果怕别人猜到，可以每个月把密码最后一个数字加一。验证码也可以告诉熟悉的朋友。"
    },
    {
        "id": "bio-experiment", "subject": "生物",
        "name": "探究光对植物生长影响",
        "description": "实验变量和对照设置不完整，适合实验能力诊断。",
        "student": "七年级生物-示例09", "title": "探究光照对幼苗生长的影响",
        "reference": "要点：选择长势相近幼苗，除光照外其他条件相同；设置有光与无光（或不同光强）对照；重复实验；定期测量；依据数据得出结论。",
        "content": "我把一盆幼苗放在窗边，另一盆放在柜子里。窗边的每天浇水，柜子里的有时忘记浇水。一周后窗边的长得更高，所以证明阳光越多植物一定长得越高，而且所有植物都必须一直放在太阳下。"
    },
    {
        "id": "bio-concept", "subject": "生物",
        "name": "呼吸作用简答题（概念错误）",
        "description": "混淆光合作用和呼吸作用，可测试概念纠正。",
        "student": "八年级生物-示例10", "title": "植物的呼吸作用",
        "reference": "植物呼吸作用在活细胞中持续进行，吸收氧气、分解有机物、释放能量并产生二氧化碳和水；白天和夜间都进行。",
        "content": "植物只有晚上才呼吸，因为白天进行光合作用就不需要呼吸。植物呼吸时吸收二氧化碳，放出氧气，所以夜晚把很多植物放在卧室里可以增加氧气。"
    },
]

# =============================================================================
# 工具函数
# =============================================================================

def grade_from_score(score: int) -> str:
    if score >= 90: return "优秀"
    if score >= 80: return "良好"
    if score >= 70: return "中等"
    if score >= 60: return "及格"
    return "待改进"


def repair_json(text: str) -> str:
    """修复 LLM 常见的 JSON 语法错误."""
    # 1. 去除 markdown 代码围栏
    fixed = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    fixed = re.sub(r"\s*```$", "", fixed)

    # 2. 提取最外层 {} 块
    start = fixed.find("{")
    end = fixed.rfind("}")
    if start >= 0 and end > start:
        fixed = fixed[start:end + 1]
    else:
        return text  # 无法提取，原样返回

    # 3. 去除尾随逗号（} 或 ] 前的逗号）
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # 4. 替换 NaN / Infinity / -Infinity 为 null
    fixed = re.sub(r"\bNaN\b", "null", fixed)
    fixed = re.sub(r"\bInfinity\b", "null", fixed)
    fixed = re.sub(r"\b-Infinity\b", "null", fixed)

    # 5. 修复字段名使用单引号的问题
    #    匹配行首或逗号后的 'fieldName': 模式，将单引号换为双引号
    fixed = re.sub(r"""(?<=[\s,\[{])'([^']+)'(?=\s*:)""", r'"\1"', fixed)

    # 6. 尝试修复字符串值中未转义的双引号（有点风险，只在出错时用）
    #    不在这里做，在最终兜底中处理

    return fixed


def parse_json_response(text: str) -> dict:
    """从模型响应中提取 JSON，多层容错修复."""
    if not text or not text.strip():
        raise ValueError("模型返回了空内容。请重试或切换模型。")

    attempts = []

    # 第 1 次：直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        attempts.append(f"直接解析失败：{e}")

    # 第 2 次：repair_json 后解析
    repaired = repair_json(text)
    last_error_msg = ""
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e2:
        last_error_msg = str(e2)
        attempts.append(f"修复后解析失败：{e2}")

    # 第 3 次：尝试逐行修复（移除可疑行）
    try:
        lines = repaired.split("\n")
        lines = [l for l in lines if l.strip() and not l.strip().startswith("//")]
        repaired2 = "\n".join(lines)
        return json.loads(repaired2)
    except json.JSONDecodeError as e3:
        attempts.append(f"二次修复失败：{e3}")

    # 第 4 次兜底：使用 json_repair 库（专为 LLM 输出设计）
    try:
        from json_repair import repair_json as json_repair_func
        repaired3 = json_repair_func(text)
        return json.loads(repaired3)
    except ImportError:
        attempts.append("json_repair 库未安装，跳过")
    except Exception as e3:
        attempts.append(f"json_repair 修复失败：{e3}")

    # 收集错误上下文
    error_detail = "\n".join(attempts)
    snippet = text.strip()[-600:] if len(text) > 600 else text.strip()
    raise ValueError(
        f"模型返回的 JSON 格式有误，自动修复失败。\n\n"
        f"**调试信息：**\n{error_detail}\n\n"
        f"**原始响应末尾（最后 600 字符）：**\n```\n{snippet}\n```\n\n"
        f"**常见原因：** 模型生成的 JSON 中包含未转义的特殊字符（如作文中的引号）、"
        f"嵌套结构错误、或超出了 max_tokens 限制导致截断。可尝试重新批改或切换模型。"
    )


def validate_grading_report(report: dict, data: dict) -> dict:
    """校验并修复批改报告."""
    if not report or not isinstance(report.get("dimensions"), list):
        raise ValueError("返回 JSON 缺少 dimensions 评分数据。")
    report["report_type"] = "grading"
    report.setdefault("subject", data["subject"])
    report.setdefault("student_id", data["studentId"])
    report.setdefault("title", data["title"])
    report["strengths"] = report.get("strengths") if isinstance(report.get("strengths"), list) else []
    report["issues"] = report.get("issues") if isinstance(report.get("issues"), list) else []
    report["suggestions"] = report.get("suggestions") if isinstance(report.get("suggestions"), list) else []
    report["special_checks"] = report.get("special_checks") if isinstance(report.get("special_checks"), list) else []
    report["subject_details"] = report.get("subject_details") if isinstance(report.get("subject_details"), dict) else {}

    dimensions = []
    for item in report["dimensions"]:
        dimensions.append({
            "name": str(item.get("name", "未命名维度")),
            "max_score": int(item.get("max_score") or 0),
            "score": int(item.get("score") or 0),
            "evidence": str(item.get("evidence", "")),
            "comment": str(item.get("comment", "")),
        })
    report["dimensions"] = dimensions
    calculated_total = sum(d["score"] for d in dimensions)
    report["total_score"] = calculated_total
    report.setdefault("grade", grade_from_score(calculated_total))
    report.setdefault("summary", "模型未提供总体评语，请教师结合分项评价复核。")
    return report


def build_grading_prompt(data: dict) -> str:
    config = SUBJECT_CONFIGS[data["subject"]]
    return f"""{config['rubric']}

【反馈风格】
{TONE_RULES[data['tone']]}

【学生作业信息】
学科：{data['subject']}
学生编号：{data['studentId']}
作业题目：{data['title']}
题目要求、参考答案或评分要点：
{data['reference'] or '未提供'}

学生作业原文：
{data['content']}

【输出协议】
1. 只能输出一个合法 JSON 对象，不得输出 Markdown、代码围栏、前言、后记或 JSON 之外的任何字符。
2. 所有字符串必须使用双引号；不得包含注释、尾随逗号、NaN 或 Infinity。
3. dimensions 的维度名称、满分和顺序必须与本学科量规完全一致。
4. dimensions 中 score 之和必须严格等于 total_score，且不得超过各项 max_score。
5. evidence 必须来自学生作业中的真实内容；不能编造原文。
6. strengths 至少2项，issues 至少2项，suggestions 至少3项。
7. subject_details 必须严格使用本学科提示词指定的结构。
8. 无法确定的信息使用空字符串、空数组或明确的"需教师复核"，不要猜测。

通用 JSON 结构：
{OUTPUT_SCHEMA}"""


def build_analysis_prompt(subject: str, records: list) -> str:
    return f"""{ANALYSIS_PROMPT_BASE}

【当前学科专项关注】
{SUBJECT_CONFIGS[subject]['analysis_focus']}

【输入记录】
{json.dumps([r['report'] for r in records], ensure_ascii=False, indent=2)}"""


def extract_response_text(resp) -> str:
    """从 OpenAI 响应中提取文本."""
    content = resp.choices[0].message.content
    return content or ""


def call_model(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """调用 OpenAI 兼容接口，返回完整响应文本."""
    # 智谱 glm-4-plus/flash 系列 max_tokens 上限为 4095
    is_zhipu_legacy = any(m in model for m in ["glm-4-plus", "glm-4-flash", "glm-4-air"])
    is_deepseek_flash = "flash" in model
    if is_zhipu_legacy:
        max_tokens_val = 4000
    elif is_deepseek_flash:
        max_tokens_val = 2048
    else:
        max_tokens_val = 8192

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=180.0,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens_val,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower() or "invalid" in msg.lower():
            raise ValueError(f"API Key 无效或无权限：{msg}")
        elif "429" in msg or "rate" in msg.lower():
            raise ValueError(f"请求过于频繁，请稍后重试：{msg}")
        elif "404" in msg or "not found" in msg.lower():
            raise ValueError(f"模型 {model} 不存在或 Base URL 不正确。请检查提供商和模型名称是否匹配。\n{msg}")
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            raise ValueError(f"请求超时，模型响应时间过长。可尝试切换更快的模型（如 flash 系列）。\n{msg}")
        elif "connection" in msg.lower() or "refused" in msg.lower():
            raise ValueError(f"无法连接 API 服务器，请检查 Base URL 和网络连接。\n{msg}")
        else:
            raise ValueError(f"API 调用失败：{msg}")

    return extract_response_text(response)


def call_model_stream(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """流式调用 API，使用 st.write_stream 增量更新，不会触发全页刷新."""
    is_zhipu_legacy = any(m in model for m in ["glm-4-plus", "glm-4-flash", "glm-4-air"])
    is_deepseek_flash = "flash" in model
    if is_zhipu_legacy:
        max_tokens_val = 4000
    elif is_deepseek_flash:
        max_tokens_val = 2048
    else:
        max_tokens_val = 8192

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=180.0,
    )

    accumulated = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens_val,
            stream=True,
        )

        def chunk_generator():
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    yield delta.content

        # write_stream 通过 WebSocket 增量更新，不引发全页重渲染
        st.info("⏳ 模型正在思考，约 10-30 秒后开始输出……全部生成完毕后将自动整理为格式化报告，请耐心等待。")
        with st.container(border=True):
            st.caption("📝 实时生成中……")
            st.write_stream(chunk_generator)

        return "".join(accumulated)

    except Exception as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower() or "invalid" in msg.lower():
            raise ValueError(f"API Key 无效或无权限：{msg}")
        elif "429" in msg or "rate" in msg.lower():
            raise ValueError(f"请求过于频繁，请稍后重试：{msg}")
        elif "404" in msg or "not found" in msg.lower():
            raise ValueError(f"模型 {model} 不存在或 Base URL 不正确。请检查提供商和模型名称是否匹配。\n{msg}")
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            raise ValueError(f"请求超时。可尝试切换更快的模型（如 flash 系列）。\n{msg}")
        elif "connection" in msg.lower() or "refused" in msg.lower():
            raise ValueError(f"无法连接 API 服务器，请检查 Base URL 和网络连接。\n{msg}")
        else:
            raise ValueError(f"API 调用失败：{msg}")


def parse_file(uploaded_file) -> str:
    """解析上传文件，返回文本内容."""
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    content_bytes = uploaded_file.read()

    if filename.endswith((".txt", ".md")):
        return content_bytes.decode("utf-8", errors="replace")

    if filename.endswith(".docx"):
        from docx import Document
        doc = Document(BytesIO(content_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if filename.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(content_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    raise ValueError(f"暂不支持的文件类型：{filename.split('.')[-1]}")


def report_to_markdown(report: dict) -> str:
    """将报告转为 Markdown."""
    if not report:
        return ""
    if report.get("report_type") == "analysis":
        lines = [
            f"# {report.get('subject','')}学情分析报告",
            "",
            f"样本数：{report.get('sample_size','')}  ",
            f"平均分：{report.get('average_score','')}",
            "",
            f"## 总体概述",
            report.get("overview", ""),
            "",
            "## 共性优点",
        ]
        for i, item in enumerate(report.get("common_strengths", []) or []):
            lines.append(f"{i+1}. **{item.get('title','')}**：{item.get('evidence','')}")
        lines.append("")
        lines.append("## 共性问题")
        for i, item in enumerate(report.get("common_issues", []) or []):
            lines.append(f"{i+1}. **{item.get('title','')}**：{item.get('diagnosis','')}；{item.get('evidence','')}")
        lines.append("")
        lines.append("## 教学建议")
        for i, item in enumerate(report.get("teaching_suggestions", []) or []):
            lines.append(f"{i+1}. **{item.get('action','')}**：{item.get('method','')}（目标：{item.get('goal','')}）")
        return "\n".join(lines)

    lines = [
        f"# {report.get('subject','')}作业批改报告",
        "",
        f"学生：{report.get('student_id','')}  ",
        f"题目：{report.get('title','')}  ",
        f"总分：{report.get('total_score','')}  ",
        f"等级：{report.get('grade','')}",
        "",
        f"## 总体评语",
        report.get("summary", ""),
        "",
        "## 分项评分",
        "| 维度 | 满分 | 得分 | 评价 | 证据 |",
        "|---|---:|---:|---|---|",
    ]
    for d in report.get("dimensions", []):
        lines.append(f"| {d['name']} | {d['max_score']} | {d['score']} | {d['comment']} | {d['evidence']} |")
    lines.append("")
    lines.append("## 主要优点")
    for i, item in enumerate(report.get("strengths", [])):
        lines.append(f"{i+1}. **{item.get('title','')}**：{item.get('evidence','')}")
    lines.append("")
    lines.append("## 主要问题")
    for i, item in enumerate(report.get("issues", [])):
        lines.append(f"{i+1}. **{item.get('title','')}**：{item.get('evidence','')}；影响：{item.get('impact','')}")
    lines.append("")
    lines.append("## 修改建议")
    for i, item in enumerate(report.get("suggestions", [])):
        lines.append(f"{i+1}. **{item.get('action','')}**：{item.get('method','')}")
    lines.append("")
    lines.append("## 学科专项数据")
    lines.append("```json")
    lines.append(json.dumps(report.get("subject_details", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("> AI 辅助生成，请教师复核。")
    return "\n".join(lines)

# =============================================================================
# Session State 初始化
# =============================================================================

DEFAULTS = {
    "cache": {s: [] for s in SUBJECT_NAMES},  # 批改缓存 {学科: [records]}
    "current_report": None,     # 当前显示的报告
    "raw_result": "",           # 最后一次原始响应
    "last_request_prompt": "",  # 最后一次发送的完整提示词
    "last_analysis_prompt": "", # 最后一次学情分析提示词
    "active_history_subject": "语文",
    "confirm_delete_idx": None,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
      <span style="background:#1D6B56;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:18px">评</span>
      <div><strong style="font-size:16px">师评智伴</strong><br><small>EduMark AI</small></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 API 配置")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="输入你的 API Key",
        help="Key 仅保存在当前会话中，不会上传到任何服务器。",
    )

    provider = st.selectbox("模型提供商", list(PROVIDERS.keys()))
    provider_config = PROVIDERS[provider]

    base_url = st.text_input(
        "Base URL",
        value=provider_config["base_url"],
        help="OpenAI 兼容接口地址，选择提供商后自动填充，也可手动修改。",
    )

    model = st.selectbox("模型名称", provider_config["models"])

    st.caption(f"Base URL: `{base_url}`")
    st.divider()

    # 提示词预览
    subject_for_trace = st.session_state.get("active_history_subject", "语文")
    with st.expander("📖 提示词中心"):
        trace_tab = st.radio("查看", ["学科批改提示词", "学情报告提示词", "本次完整请求", "原始返回结果"],
                            label_visibility="collapsed", key="trace_tab")
        if trace_tab == "学科批改提示词":
            st.code(SUBJECT_CONFIGS[subject_for_trace]["rubric"], language="markdown")
        elif trace_tab == "学情报告提示词":
            st.code(f"{ANALYSIS_PROMPT_BASE}\n\n【{subject_for_trace}专项关注】\n{SUBJECT_CONFIGS[subject_for_trace]['analysis_focus']}", language="markdown")
        elif trace_tab == "本次完整请求":
            st.code(st.session_state.last_request_prompt or "尚未发起批改请求。", language="markdown")
        else:
            st.code(st.session_state.raw_result or "尚未收到模型响应。", language="json")

    # 使用帮助
    with st.expander("❓ 使用帮助"):
        st.markdown("""
        **透明完成一次 AI 批改：**
        1. 在左侧配置 API Key 和模型
        2. 选择学科或内置样例
        3. 查看提示词后点击"开始智能批改"
        4. 复核美化后的报告
        5. 在"提示词中心"追溯完整请求和原始响应

        **建议：** 批量使用时采用匿名编号，避免上传姓名、电话等敏感信息。
        """)

# =============================================================================
# 主区域
# =============================================================================

# 顶部 Hero
st.markdown("""
<p style="color:#62746E;font-size:13px;letter-spacing:2px;margin:0">AI TEACHING COPILOT</p>
<h1 style="margin-top:0">把重复批改交给 AI，<span style="color:#1D6B56">把教学判断留给老师。</span></h1>
<p style="color:#62746E">覆盖语文、数学、英语、计算机与生物。提示词可查看、交互可追溯、结果按 JSON 协议解析，让每一次 AI 批改更透明。</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 提交批改", "📋 批改历史", "📊 学情分析"])

# ═══════════════════════════════════════════════════════════════════════════
# Tab 1：提交批改
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    # ── 顶部：学科 & 反馈风格 ──
    top_cols = st.columns([1, 1, 2])
    with top_cols[0]:
        subject = st.selectbox(
            "课程学科 *",
            SUBJECT_NAMES,
            format_func=lambda s: f"{s} - {SUBJECT_CONFIGS[s]['label']}",
            key="tab1_subject",
        )
    with top_cols[1]:
        tone = st.selectbox(
            "反馈风格 *",
            list(TONE_RULES.keys()),
            format_func=lambda t: f"{t}（{TONE_RULES[t][:6]}…）",
            key="tab1_tone",
        )
    with top_cols[2]:
        # 样例选择
        example_options = [e for e in EXAMPLES if e["subject"] == subject]
        example_id = st.selectbox(
            "快速选择作业样例（可选）",
            [""] + [e["id"] for e in example_options],
            format_func=lambda x: "" if not x else next((e["name"] for e in EXAMPLES if e["id"] == x), x),
            key="example_select",
        )

    # 样例应用 & 描述
    if example_id:
        ex = next(e for e in EXAMPLES if e["id"] == example_id)
        st.caption(f"📌 {ex.get('description','')}")
        if st.button("📥 应用样例到表单", key="apply_example_btn"):
            st.session_state["form_student_id"] = ex["student"]
            st.session_state["form_title"] = ex["title"]
            if SUBJECT_CONFIGS[subject]["reference_visible"]:
                st.session_state["form_reference"] = ex["reference"]
            st.session_state["form_content"] = ex["content"]
            st.session_state["active_history_subject"] = subject
            st.toast(f"已应用：{ex['name']}")
            st.rerun()

    st.divider()

    # ── 表单 & 报告（上下布局）──
    config = SUBJECT_CONFIGS[subject]

    form_col, report_col = st.columns([5, 5], gap="large")

    with form_col:
        st.caption("STEP 01")
        st.markdown("### 📝 提交学生作业")

        with st.form("grading_form", clear_on_submit=False):
            student_id = st.text_input(
                "学生姓名/编号 *",
                max_chars=50,
                placeholder="例如：八年级1班-李明",
                key="form_student_id",
            )
            homework_title = st.text_input(
                "作业题目 *",
                max_chars=150,
                placeholder="例如：《背影》读后感",
                key="form_title",
            )

            if config["reference_visible"]:
                reference = st.text_area(
                    f"题目与参考答案（{config['reference_title']}）",
                    placeholder=f"请填写{config['reference_title']}。没有时也可继续，但结果需要重点复核。",
                    key="form_reference",
                    height=120,
                )
            else:
                reference = ""
                st.caption(f"💡 {config['reference_title']}：本学科不需要参考答案字段。")

            homework_content = st.text_area(
                "学生作业 *",
                height=250,
                placeholder="在此粘贴学生作业，或使用下方的文件上传……",
                key="form_content",
            )

            # 字符计数
            content_val = st.session_state.get("form_content", "")
            if content_val:
                if subject == "英语":
                    words = len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", content_val))
                    st.caption(f"📊 {words} 词 · {len(content_val)} 字符")
                elif subject == "计算机":
                    lines = content_val.count("\n") + 1 if content_val else 0
                    st.caption(f"📊 {lines} 行 · {len(content_val)} 字符")
                else:
                    st.caption(f"📊 {len(content_val)} 字符")

            privacy_ok = st.checkbox(
                "我已隐去不必要的敏感个人信息，并同意将以上内容发送给所选 AI 服务进行处理。"
            )

            submitted = st.form_submit_button("🚀 开始智能批改", type="primary", use_container_width=True)

        # ── 文件上传（表单外）──
        uploaded_file = st.file_uploader(
            "📎 上传作业文档（TXT, MD, DOCX, PDF，最大 10 MB）",
            type=["txt", "md", "docx", "pdf"],
            key="file_uploader",
        )
        if uploaded_file and st.button("📄 解析文件并填入", key="parse_btn"):
            try:
                text = parse_file(uploaded_file)
                if text.strip():
                    st.session_state["form_content"] = text
                    st.toast("文档内容已提取到作业框")
                    st.rerun()
                else:
                    st.error("未提取到文字内容，扫描版 PDF 请先 OCR。")
            except Exception as e:
                st.error(str(e))

    # ── 报告区 ──
    with report_col:
        st.caption("STEP 02")
        st.markdown("### 📊 批改报告")

        if submitted:
            if not api_key:
                st.error("请先在侧边栏输入 API Key。")
            elif not homework_content or not homework_content.strip():
                st.error("请粘贴、上传或选择学生作业。")
            elif not privacy_ok:
                st.error("请先确认隐私与数据处理提示。")
            else:
                data = {
                    "subject": subject,
                    "studentId": (student_id or "").strip(),
                    "title": (homework_title or "").strip(),
                    "content": homework_content.strip(),
                    "reference": (reference or "").strip() if config["reference_visible"] else "",
                    "tone": tone,
                    "model": model,
                }

                prompt = build_grading_prompt(data)
                st.session_state.last_request_prompt = prompt

                with st.spinner("正在连接模型……"):
                    try:
                        raw = call_model_stream(prompt, api_key, base_url, model)
                        st.session_state.raw_result = raw
                        report = parse_json_response(raw)
                        report = validate_grading_report(report, data)
                        st.session_state.current_report = report

                        # 写入缓存
                        cache = st.session_state.cache
                        record = {
                            "id": str(uuid.uuid4()),
                            "subject": subject,
                            "studentId": data["studentId"],
                            "title": data["title"],
                            "content": data["content"],
                            "reference": data["reference"],
                            "tone": tone,
                            "model": model,
                            "report": report,
                            "prompt": prompt,
                            "rawResponse": raw,
                            "createdAt": datetime.now().isoformat(),
                        }
                        cache[subject].insert(0, record)
                        if len(cache[subject]) > MAX_CACHE_PER_SUBJECT:
                            cache[subject] = cache[subject][:MAX_CACHE_PER_SUBJECT]
                        st.session_state.cache = cache
                        st.session_state.active_history_subject = subject

                        st.toast("✅ JSON 批改完成，请教师复核")
                        st.rerun()

                    except Exception as e:
                        st.error(f"批改失败：{e}")
                        if not st.session_state.raw_result:
                            st.info("请在侧边栏「提示词中心」查看发送给模型的完整请求。")

        # 显示报告
        report = st.session_state.current_report
        if report is None:
            st.info("批改报告将在这里生成。\n\n提交左侧作业后，系统会依据所选学科量规给出评分、问题诊断与修改建议。")
        else:
            display_subject = report.get("subject", subject)
            max_total = sum(d["max_score"] for d in report.get("dimensions", [])) or 100
            pct = max(0, min(100, report["total_score"] / max_total * 100)) if max_total > 0 else 0

            # 总分卡片
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:16px;padding:14px;border-radius:12px;
                        background:linear-gradient(135deg,#1D6B56,#17352D);color:white;margin-bottom:12px">
              <div style="flex:1">
                <small style="opacity:0.8">{display_subject} · {report.get('student_id','')}</small>
                <h3 style="margin:2px 0;color:white;font-size:16px">{report.get('title','')}</h3>
                <p style="margin:0;opacity:0.85;font-size:12px">
                  {report.get('summary','')[:100]}{'…' if len(report.get('summary','') or '') > 100 else ''}
                </p>
              </div>
              <div style="text-align:center;min-width:80px">
                <span style="font-size:42px;font-weight:bold">{report['total_score']}</span>
                <span style="opacity:0.7;font-size:14px">/ {max_total}</span>
                <br><span style="background:#ffffff33;padding:1px 8px;border-radius:8px;font-size:12px">
                  {report.get('grade','')}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # 分项评分
            with st.expander("📊 分项评分详情", expanded=True):
                for dim in report.get("dimensions", []):
                    dim_pct = dim["score"] / dim["max_score"] * 100 if dim["max_score"] > 0 else 0
                    color = "#1D6B56" if dim_pct >= 60 else "#E47B5D"
                    st.markdown(f"""
                    <div style="margin-bottom:10px">
                      <div style="display:flex;justify-content:space-between;align-items:baseline">
                        <strong>{dim['name']}</strong>
                        <span style="font-size:16px;font-weight:bold;color:{color}">
                          {dim['score']}<small style="font-size:12px;color:#888">/{dim['max_score']}</small>
                        </span>
                      </div>
                      <div style="background:#E8ECE9;border-radius:4px;height:6px;margin:4px 0">
                        <div style="background:{color};width:{dim_pct}%;height:6px;border-radius:4px"></div>
                      </div>
                      <p style="font-size:13px;color:#555;margin:2px 0">{dim['comment']}</p>
                      {f'<blockquote style="font-size:12px;color:#999;border-left:2px solid {color};padding-left:8px;margin:2px 0">{dim["evidence"]}</blockquote>' if dim.get('evidence') else ''}
                    </div>
                    """, unsafe_allow_html=True)

            # 优点 & 问题
            c1, c2 = st.columns(2)
            with c1:
                st.caption("✅ 主要优点")
                for i, item in enumerate(report.get("strengths", [])):
                    with st.container(border=True):
                        st.caption(f"#{i+1} · {item.get('title','')}")
                        if item.get("evidence"):
                            st.caption(f"📌 {item['evidence']}")

            with c2:
                st.caption("🔍 主要问题")
                for i, item in enumerate(report.get("issues", [])):
                    with st.container(border=True):
                        st.caption(f"#{i+1} · {item.get('title','')}")
                        if item.get("evidence"):
                            st.caption(f"📌 {item['evidence']}")
                        if item.get("impact"):
                            st.caption(f"⚡ 影响：{item['impact']}")

            # 修改建议
            with st.expander("💡 修改建议", expanded=True):
                for i, item in enumerate(report.get("suggestions", [])):
                    st.markdown(f"{i+1}. **{item.get('action','')}**：{item.get('method','')}")

            # ── 学科专项 ──
            details = report.get("subject_details", {})
            if display_subject == "英语":
                with st.expander("🔤 语言错误清单 & Improved Version", expanded=True):
                    st.caption(f"词数：{details.get('word_count', 0)}")
                    errors = details.get("language_errors", [])
                    if errors:
                        st.dataframe(
                            [{"类型": e.get("type",""), "原文": e.get("original",""),
                              "问题": e.get("explanation",""), "修改": e.get("correction","")}
                             for e in errors],
                            use_container_width=True, hide_index=True,
                        )
                    st.markdown("**✏️ Improved Version**")
                    st.info(details.get("improved_version", "未提供"))

            elif display_subject == "数学":
                with st.expander("🔢 正误判断与纠错", expanded=True):
                    st.markdown(f"**结论**：{'✅ 完全正确' if details.get('is_correct') else '⚠️ 需要修正'}")
                    mc1, mc2 = st.columns(2)
                    with mc1: st.metric("首个关键错误", details.get("first_error_step", "无"))
                    with mc2: st.metric("错误类型", details.get("error_type", "无"))
                    st.caption(details.get("error_reason", ""))
                    for s in details.get("correction_steps", []):
                        st.markdown(f"- {s}")
                    if details.get("teacher_review_note"):
                        st.warning(details["teacher_review_note"])

            elif display_subject == "计算机":
                with st.expander("💻 代码诊断", expanded=True):
                    st.caption(f"任务类型：{details.get('task_type','计算机作业')}")
                    st.caption(details.get("static_assessment", ""))
                    for ci in details.get("code_issues", []):
                        with st.container(border=True):
                            st.markdown(f"**{ci.get('type','')}** · `{ci.get('location','')}`")
                            if ci.get("original"): st.code(ci["original"], language=None)
                            st.caption(f"{ci.get('explanation','')} → 修改：{ci.get('fix','')}")
                    tests = details.get("test_cases", [])
                    if tests:
                        st.markdown("**建议测试用例**")
                        st.dataframe(tests, use_container_width=True, hide_index=True)
                    if details.get("improved_code"):
                        st.markdown("**改进代码**")
                        st.code(details["improved_code"], language=None)
                    st.caption(details.get("runtime_note", ""))

            elif display_subject == "生物":
                with st.expander("🧬 概念与实验复核", expanded=True):
                    st.caption(f"题型：{details.get('question_type','生物作业')}")
                    for ce in details.get("concept_errors", []):
                        with st.container(border=True):
                            st.markdown(f"**{ce.get('concept','')}**")
                            st.caption(f"学生：{ce.get('student_statement','')}")
                            st.caption(f"正确：{ce.get('correction','')}")
                    st.caption(details.get("reasoning_assessment", ""))
                    exp = details.get("experiment_review", {})
                    if exp:
                        ec1, ec2, ec3, ec4 = st.columns(4)
                        with ec1: st.metric("变量", exp.get("variables", "N/A"))
                        with ec2: st.metric("对照", exp.get("control", "N/A"))
                        with ec3: st.metric("步骤", exp.get("procedure", "N/A"))
                        with ec4: st.metric("数据结论", exp.get("data_conclusion", "N/A"))
                    if details.get("model_answer"):
                        st.info(details["model_answer"])

            elif display_subject == "语文":
                with st.expander("📝 写作专项检查", expanded=True):
                    st.caption(f"文体：{details.get('writing_type','作文')}")
                    st.caption(details.get("revision_focus", ""))
                    cap = details.get("cap_rule", {})
                    if cap.get("triggered"):
                        st.error(f"⚠️ 已触发封顶规则：{cap.get('rule','')} — {cap.get('reason','')}")
                    else:
                        st.success(f"✅ {cap.get('rule','')} — {cap.get('reason','')}")

            # 规则检查
            checks = report.get("special_checks", [])
            if checks:
                with st.expander("🔎 规则检查", expanded=False):
                    for ck in checks:
                        icon = "⚠️" if ck.get("triggered") else "✅"
                        st.caption(f"{icon} {ck.get('name')} {'(已触发)' if ck.get('triggered') else '(未触发)'} — {ck.get('explanation','')}")

            st.divider()
            # 导出
            md = report_to_markdown(report)
            dl_cols = st.columns(3)
            with dl_cols[0]:
                st.download_button("📥 下载 Markdown", md,
                                   file_name=f"{report.get('student_id','report')}_{report.get('title','report')}_AI报告.md",
                                   use_container_width=True)
            with dl_cols[1]:
                if st.button("📋 查看源码（可复制）", use_container_width=True):
                    st.code(md, language="markdown")
            with dl_cols[2]:
                st.caption("🖨️ 浏览器打印：Ctrl+P")

            st.caption("> AI 辅助生成 · 请教师结合教学目标和学生实际复核")

# ═══════════════════════════════════════════════════════════════════════════
# Tab 2：批改历史
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    cache = st.session_state.cache

    # 学科筛选按钮
    filter_cols = st.columns(len(SUBJECT_NAMES))
    for i, s in enumerate(SUBJECT_NAMES):
        with filter_cols[i]:
            count = len(cache.get(s, []))
            if st.button(f"{s}\n({count})", use_container_width=True,
                         type="primary" if s == st.session_state.active_history_subject else "secondary",
                         key=f"hist_filter_{s}"):
                st.session_state.active_history_subject = s
                st.rerun()

    active_subject = st.session_state.active_history_subject
    records = cache.get(active_subject, [])

    if not records:
        st.info(f"暂无 {active_subject} 批改缓存。完成批改后会自动保存到该学科。")
    else:
        cols_per_row = 3
        for i in range(0, len(records), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx >= len(records):
                    break
                r = records[idx]
                with col:
                    with st.container(border=True):
                        st.caption(f"{r['subject']} · {r.get('grade', grade_from_score(r['report'].get('total_score',0)))}")
                        score = r["report"].get("total_score", "--")
                        st.markdown(f"### {score}")
                        st.markdown(f"**{r['title']}**")
                        st.caption(f"{r['studentId']} · {r['createdAt'][:10]}")
                        c_act1, c_act2 = st.columns(2)
                        with c_act1:
                            if st.button("📖 查看", key=f"view_{r['id']}", use_container_width=True):
                                st.session_state.current_report = r["report"]
                                st.session_state.last_request_prompt = r.get("prompt", "")
                                st.session_state.raw_result = r.get("rawResponse", "")
                                st.toast("已恢复批改结果")
                                st.rerun()
                        with c_act2:
                            if st.button("🗑️ 删除", key=f"del_{r['id']}", use_container_width=True):
                                cache[active_subject] = [x for x in cache[active_subject] if x["id"] != r["id"]]
                                st.session_state.cache = cache
                                st.rerun()

    # 清空全部
    all_count = sum(len(v) for v in cache.values())
    if all_count > 0:
        st.divider()
        if st.button(f"🗑️ 清空全部批改记录（共 {all_count} 条，不可撤销）", type="secondary"):
            st.session_state.cache = {s: [] for s in SUBJECT_NAMES}
            st.toast("所有记录已清空")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# Tab 3：学情分析
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 选择用于学情分析的作业")
    st.caption("批改结果已按学科缓存。选择同一学科中的至少2份作业，只有勾选的记录会发送给模型。")

    analysis_subject = st.selectbox(
        "分析学科",
        SUBJECT_NAMES,
        format_func=lambda s: f"{s}（{len(cache.get(s, []))}份缓存）",
        key="analysis_subject",
    )

    analysis_records = cache.get(analysis_subject, [])

    if not analysis_records:
        st.info(f"暂无 {analysis_subject} 批改缓存，请先完成该学科作业批改。")
    else:
        # 勾选列表
        selected_ids = []
        for r in analysis_records:
            key = f"sel_{r['id']}"
            if st.checkbox(
                f"**{r['title']}** — {r['studentId']} — **{r['report'].get('total_score','--')}分** — {r['report'].get('grade','')} — {r['createdAt'][:10]}",
                key=key,
            ):
                selected_ids.append(r["id"])

        selected = [r for r in analysis_records if r["id"] in selected_ids]
        valid = 2 <= len(selected) <= MAX_ANALYSIS_SAMPLES

        # 选择统计
        if selected:
            avg = sum(r["report"].get("total_score", 0) for r in selected) / len(selected)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("已选作业", len(selected))
            with c2:
                st.metric("均分预览", f"{avg:.1f}")

        if not valid:
            st.warning(f"请选择 2 至 {MAX_ANALYSIS_SAMPLES} 份 {analysis_subject} 作业。")

        if st.button("📊 用所选作业生成学情分析", type="primary", disabled=not valid):
            if not api_key:
                st.error("请先在侧边栏输入 API Key。")
            else:
                prompt = build_analysis_prompt(analysis_subject, selected)
                st.session_state.last_analysis_prompt = prompt
                st.session_state.last_request_prompt = prompt

                with st.spinner("正在连接模型……"):
                    try:
                        raw = call_model_stream(prompt, api_key, base_url, model)
                        st.session_state.raw_result = raw
                        analysis_report = parse_json_response(raw)
                        analysis_report["report_type"] = "analysis"
                        analysis_report.setdefault("subject", analysis_subject)
                        analysis_report["sample_size"] = int(analysis_report.get("sample_size") or len(selected))
                        st.session_state.current_report = analysis_report
                        st.toast("学情分析已生成，请复核统计")
                        st.rerun()
                    except Exception as e:
                        st.error(f"分析失败：{e}")

        # 如果当前报告是分析报告，就展示
        report = st.session_state.current_report
        if report and report.get("report_type") == "analysis":
            st.divider()
            st.markdown("### 📊 学情分析报告")

            # 顶部概览
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{report.get('subject','')} · 班级学情**")
                st.markdown(report.get("overview", ""))
            with c2:
                st.metric("平均分", report.get("average_score", "--"))
                st.caption(f"{report.get('sample_size','')} 份样本")

            # 成绩分布
            dist = report.get("score_distribution", [])
            if dist:
                st.markdown("**成绩分布**")
                dist_cols = st.columns(len(dist))
                for ii, d in enumerate(dist):
                    with dist_cols[ii]:
                        with st.container(border=True):
                            st.metric(d.get("level", ""), f"{d.get('count',0)} 人")
                            st.caption(f"{d.get('range','')} · {d.get('percentage',0)}%")

            # 分项能力
            dims = report.get("dimension_stats", [])
            if dims:
                st.markdown("**分项能力分析**")
                for dim in dims:
                    rate = dim.get("rate", 0)
                    st.markdown(f"**{dim.get('name','')}**：{dim.get('average_score',0)}/{dim.get('max_score',0)} — {dim.get('judgement','')}")
                    st.progress(int(rate) if rate else 0)

            # 共性
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ 共性优点**")
                for item in report.get("common_strengths", []):
                    st.markdown(f"- **{item.get('title','')}**：{item.get('evidence','')}")
            with c2:
                st.markdown("**🔍 共性问题**")
                for item in report.get("common_issues", []):
                    st.markdown(f"- **{item.get('title','')}**：{item.get('diagnosis','')}")

            # 典型学生
            profiles = report.get("student_profiles", [])
            if profiles:
                st.markdown("**👤 典型学生表现**")
                for p in profiles:
                    st.markdown(f"- **{p.get('student_id','')}** ({p.get('type','')})：{p.get('reason','')}")

            # 教学建议
            suggestions = report.get("teaching_suggestions", [])
            if suggestions:
                st.markdown("**📋 后续教学建议**")
                for i, s in enumerate(suggestions):
                    st.markdown(f"{i+1}. **{s.get('action','')}** — {s.get('method','')}（目标：{s.get('goal','')}）")

            # 局限
            limits = report.get("limitations", [])
            if limits:
                st.caption("**数据说明：** " + "；".join(limits))

            st.caption("> AI 辅助分析 · 统计与教学判断请由教师复核")

st.divider()
st.caption("师评智伴 EduMark AI · 内容仅用于本次批改 · AI 结果仅作教学辅助，分数和事实判断请由教师最终复核。")
