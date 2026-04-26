import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

## 从 .env 文件加载环境变量以确保安全
load_dotenv()

## 1. 为清晰起见，明确定义语言模型
# 使用字符串形式的模型配置，而不是ChatOpenAI实例
llm_config = {
    "openai_api_key": "ollama",
    "openai_api_base": "http://localhost:11434/v1",
    "model_name": "qwen3.5:9b",
    "temperature": 0.0,
    "timeout": 30000
}

## 2. 定义一个清晰且专注的智能体
planner_writer_agent = Agent(
    role='文章规划者和撰写者',
    goal='规划然后撰写关于指定主题的简洁、引人入胜的摘要。',
    backstory=(
        '你是一位专业的技术作家和内容策略师。'
        '你的优势在于在写作之前创建清晰、可操作的计划，'
        '确保最终摘要既信息丰富又易于理解。'
    ),
    verbose=True,
    allow_delegation=False,
    llm="qwen3.5:9b", # 使用模型名称
    llm_config=llm_config # 传递模型配置
)

## 3. 定义具有更结构化和具体的预期输出的任务
topic = "强化学习在 AI 中的重要性"
high_level_task = Task(
    description=(
        f"1. 为主题'{topic}'的摘要创建要点计划。\n"
        f"2. 根据您的计划撰写摘要，保持在 200 字左右。"
    ),
    expected_output=(
        "包含两个不同部分的最终报告：\n\n"
        "### 计划\n"
        "- 概述摘要要点的项目符号列表。\n\n"
        "### 摘要\n"
        "- 主题的简洁且结构良好的摘要。"
    ),
    agent=planner_writer_agent,
)

## 使用清晰的流程创建团队
crew = Crew(
    agents=[planner_writer_agent],
    tasks=[high_level_task],
    process=Process.sequential,
)

## 执行任务
print("## 运行规划和写作任务 ##")
result = crew.kickoff()
print("\n\n---\n## 任务结果 ##\n---")
print(result)