import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 1. 定义语言模型
# llm = ChatOpenAI(model="gpt-4-turbo")
# 使用字符串形式的模型配置，而不是ChatOpenAI实例
llm_config = {
    "openai_api_key": "ollama",
    "openai_api_base": "http://localhost:11434/v1",
    "model_name": "qwen3.5:9b",
    "temperature": 0.0,
    "timeout": 30000
}
# llm = ChatOpenAI(**llm_config)

# 2. 创建一个具有强大规划能力的Manager Agent
manager_agent = Agent(
    role='项目总规划师',
    goal='分析复杂项目需求，自主分解任务，协调团队成员，并确保项目成功完成',
    backstory=(
        '你是一位经验丰富的项目管理专家，擅长将复杂项目分解为可管理的子任务，'
        '并能智能地分配给最合适的团队成员。你具有敏锐的判断力，知道何时需要'
        '调整计划或重新分配任务以确保项目成功。'
    ),
    verbose=True,
    allow_delegation=True,  # 允许委派任务
    llm="qwen3.5:9b", # 使用模型名称
    llm_config=llm_config
)

# 3. 创建专业化的执行Agent
research_agent = Agent(
    role='深度研究员',
    goal='深入研究特定主题，收集准确、全面的信息',
    backstory=(
        '你是一位学术背景深厚的研究员，擅长从多个角度分析复杂主题，'
        '能够识别关键信息源并验证信息的真实性。'
    ),
    verbose=True,
    allow_delegation=False,
    llm="qwen3.5:9b", # 使用模型名称
    llm_config=llm_config
)

analysis_agent = Agent(
    role='数据分析师',
    goal='分析收集到的信息，识别模式、趋势和关键见解',
    backstory=(
        '你是一位数据分析师，擅长将原始信息转化为有价值的见解。'
        '你能够识别数据中的模式，进行对比分析，并提炼出核心结论。'
    ),
    verbose=True,
    allow_delegation=False,
    llm="qwen3.5:9b", # 使用模型名称
    llm_config=llm_config
)

synthesis_agent = Agent(
    role='内容整合专家',
    goal='将分析结果整合为结构清晰、逻辑连贯的最终输出',
    backstory=(
        '你是一位内容架构师，擅长将复杂的分析结果整合为易于理解的报告。'
        '你注重逻辑结构，确保内容既全面又简洁。'
    ),
    verbose=True,
    allow_delegation=False,
    llm="qwen3.5:9b", # 使用模型名称
    llm_config=llm_config
)

# 4. 定义一个真正需要自主规划的复杂任务
# 注意：这里只给出高层目标，让Manager Agent自行分解任务
complex_project_task = Task(
    description=(
        "分析'人工智能在教育领域的革命性影响'这一主题，并生成一份全面的分析报告。"
    ),
    expected_output=(
        "一份结构完整的分析报告，包含：\n"
        "- 执行摘要\n"
        "- 当前AI教育应用概览\n"
        "- 具体的变革性案例\n"
        "- 面临的挑战与解决方案\n"
        "- 未来发展趋势预测\n"
        "- 实用建议"
    ),
    agent=manager_agent,  # 由Manager Agent负责整体规划
)

# 5. 创建Hierarchical流程的Crew
# 关键：使用hierarchical流程，让Manager Agent真正发挥规划作用
crew = Crew(
    agents=[manager_agent, research_agent, analysis_agent, synthesis_agent],
    tasks=[complex_project_task],
    process=Process.hierarchical,  # 使用层级流程
    manager_llm="qwen3.5:9b",  # 为Manager指定LLM模型名称
    verbose=True,
    memory=True,  # 启用记忆功能，让Agent记住之前的决策
    cache=True,   # 启用缓存
    max_rpm=100   # 设置最大请求速率
)

# 6. 执行任务 - 观察Manager Agent如何自主规划
print("## 启动AI教育影响分析项目 ##")
print("Manager Agent将自主分解任务并协调团队...")
print("-" * 50)

result = crew.kickoff()

print("\n\n" + "="*60)
print("## 项目最终成果 ##")
print("="*60)
print(result)