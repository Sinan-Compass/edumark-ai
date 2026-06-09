---
title: 师评智伴 EduMark AI
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
fullWidth: true
header: mini
pinned: false
license: mit
short_description: 五学科、JSON结构化输出、用户自带API Key的教师作业批改 AI Agent
tags:
  - education
  - ai-agent
  - homework
  - chinese
  - streamlit
---

# 师评智伴 EduMark AI

面向中学教师的透明化作业批改 AI Agent，支持语文、数学、英语、计算机和生物。**Streamlit 版**：用户自行提供 API Key，选择智谱/DeepSeek/千问等 OpenAI 兼容接口。

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
streamlit run app.py
```

访问 http://localhost:8501

## 使用流程

1. **配置 API**：在侧边栏输入你的 API Key，选择模型提供商（智谱/DeepSeek/千问），Base URL 自动填充
2. **提交作业**：选择学科 → 填写学生信息 → 粘贴或上传作业（支持 TXT/MD/DOCX/PDF）
3. **查看报告**：AI 返回严格 JSON 格式的批改报告，前端解析为可读的分项评分、优缺点分析和修改建议
4. **追溯交互**：侧边栏"提示词中心"可查看完整请求和原始响应
5. **学情分析**：在"学情分析"标签页勾选多份已批改作业，生成班级统计报告

## 支持的模型提供商

| 提供商 | Base URL | 可选模型 |
|--------|----------|---------|
| 智谱 (GLM) | `https://open.bigmodel.cn/api/paas/v4/` | glm-4-plus, glm-4-flash, glm-4-32b, glm-4-air, glm-4-long |
| DeepSeek | `https://api.deepseek.com` | deepseek-chat, deepseek-reasoner |
| 千问 (Qwen) | `https://dashscope.aliyuncs.com/compatible-mode/v1/` | qwen-turbo, qwen-plus, qwen-max, qwen3-30b-a3b-instruct |

Base URL 可手动修改，支持任意 OpenAI 兼容接口。

## 部署到 Streamlit Cloud

1. 将本项目推送到 GitHub 仓库
2. 登录 [Streamlit Cloud](https://share.streamlit.io/)
3. 点击 "New app"，选择仓库和分支
4. Main file path 填写 `app.py`
5. 点击 Deploy

## 部署到 Hugging Face Spaces

使用 Docker SDK：

1. 在 HF 创建 Space，SDK 选择 **Docker**
2. 上传本目录所有文件（包括 `packages.txt` 和 `requirements.txt`）
3. 如需静态版，仍可使用原有的 `index.html` + `app.js` 部署为 Static Space

## 核心功能

- 粘贴文字或上传 TXT / MD / DOCX / PDF
- 选择语文、数学、英语、计算机、生物课程
- 内置 10 个作业样例，每个学科 2 个
- 用户自行提供 API Key，支持智谱、DeepSeek、千问及任意 OpenAI 兼容接口
- 五科差异化量规与专属诊断模块
- 查看学科批改提示词和学情分析提示词
- 查看发送给 LLM 的完整请求和未经处理的原始响应
- 约束模型输出 JSON，由前端校验、提取并美化展示
- 批改结果按五个学科分别缓存（会话级），每科最多 30 条
- 学情分析前可勾选 2 至 20 份同学科缓存
- 导出 Markdown、复制报告

## 隐私说明

- API Key 仅保存在当前会话中，关闭浏览器后消失
- 建议使用匿名学生编号
- 上传前移除电话、住址、证件号等敏感数据
- 批改记录保存在当前会话中
- AI 输出仅作教学辅助，须由教师最终复核
