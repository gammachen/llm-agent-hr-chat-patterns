from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama

# 1. 定义共享工具（结果会自动存入团队上下文）
from crewai_tools import SerpApiGoogleSearchTool
## 需要用到api key
# os.environ["SERPAPI_API_KEY"] = "YOUR_API_KEY"

# 参考：https://docs.crewai.org.cn/en/tools/search-research/serpapi-googlesearchtool
# 但是安装这个tool却失败了（日）
'''shell

(langgraph)  ✘  🐍 langgraph  shhaofu@shhaofudeMacBook-Pro  ~/Code/Codes/autonomous-hr-chatbot/crewai_context_sharing   main ±  python crewai_context_sharing.py
You are missing the 'serpapi' package. Would you like to install it? [y/N]: n
Traceback (most recent call last):
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/crewai_context_sharing/crewai_context_sharing.py", line 6, in <module>
    search_tool = SerpApiGoogleSearchTool(
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/shhaofu/.local/lib/python3.11/site-packages/crewai_tools/tools/serpapi_tool/serpapi_base_tool.py", line 41, in __init__
    raise ImportError(
ImportError: `serpapi` package not found, please install with `uv add serpapi`
(langgraph)  ✘  🐍 langgraph  shhaofu@shhaofudeMacBook-Pro  ~/Code/Codes/autonomous-hr-chatbot/crewai_context_sharing   main ±  uv add serpapi
error: No `pyproject.toml` found in current directory or any parent directory
'''
search_tool = SerpApiGoogleSearchTool(
    search_type="google",
    search_results=3,
    search_depth="medium",
)

# search_tool2 = DuckDuckGoSearchTool()

print("📡 初始化Ollama LLM (qwen3.5:9b)...")
llm = ChatOllama(
    model="qwen3.5:9b",  # 或者使用 "ollama/qwen3:1.7b"
    base_url="http://localhost:11434",
    temperature=0.3,
    # num_predict=512,  # 限制生成长度
)

# 2. 创建Agent（无需显式声明记忆，团队级上下文自动共享）
researcher = Agent(
    role="市场研究员",
    goal="通过搜索获取相关内容的关键信息",
    backstory="专注科技市场10年",
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role="内容撰写专家",
    goal="基于研究数据生成报告",
    backstory="资深科技专栏作家",
    verbose=True
)

# 3. 定义任务（关键：通过output传递上下文）
research_task = Task(
    description="搜索2026年AI行业关键趋势",
    expected_output="JSON格式的3个核心趋势，含数据来源",  # 强制结构化输出
    agent=researcher,
    llm=llm  # 为研究员指定Ollama模型
)

research_task2 = Task(
    description="搜索2026年编程领域的Vibe Coding的核心趋势",
    expected_output='''JSON格式的3个核心趋势，含数据来源，严格按以下格式输出：
{
  "trends": [
    {
      "trend_name": "趋势名称（如：多模态大模型）",
      "description": "趋势的详细描述（200字左右）",
      "key_players": ["主要公司/产品1", "主要公司/产品2"],
      "data_source": "数据来源URL或机构名称",
      "impact_level": "high/medium/low"
    }
  ],
  "summary": "三大趋势的整体总结（100字左右）"
}''',
    agent=researcher,
    llm=llm
)

write_task = Task(
    description="用研究员的发现撰写简明报告",
    expected_output="800字行业分析，含数据引用",
    agent=writer,
    llm=llm,
    context=[research_task]  # ⭐ 自动注入前序任务输出作为上下文
)

write_task2 = Task(
    description='''基于研究员提供的JSON数据撰写详细行业报告。

输入数据格式说明：
- trends: 包含3个趋势对象的数组
  - trend_name: 趋势名称
  - description: 趋势描述
  - key_players: 主要参与者数组
  - data_source: 数据来源
  - impact_level: 影响等级(high/medium/low)
- summary: 整体总结

写作要求：
1. 为每个趋势撰写一个完整章节（约400字/章），包含：
   - 趋势背景与现状（基于description扩展）
   - 关键参与者分析（基于key_players详细阐述）
   - 行业影响评估（基于impact_level展开）
   - 数据引用标注（标注data_source）
2. 撰写执行摘要（约300字，基于summary扩展）
3. 总字数约1500字
''',
    expected_output="1500字结构化行业分析报告，包含执行摘要、3个趋势详细章节（含背景、参与者、影响评估、数据来源标注）",
    agent=writer,
    llm=llm,
    context=[research_task2]
)

# 4. 创建团队（自动管理上下文传递）
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task, research_task2, write_task2],
    process=Process.sequential  # 顺序执行，自动传递上下文
)

# 执行任务（research_task的输出会自动作为write_task的输入）
result = crew.kickoff()
print(result)

"""
上下文传递路径：
1. researcher 执行 research_task → 输出结构化JSON到团队记忆
2. CrewAI 自动将JSON注入 writer 的提示词：
   "以下是从研究员处获取的数据：{research_task.output}"
3. writer 直接引用该数据生成报告，无需手动传递变量
"""