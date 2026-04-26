import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# 1. 定义语言模型
# llm = ChatOpenAI(model="gpt-4-turbo")
llm_config = {
    "openai_api_key": "ollama",
    "openai_api_base": "http://localhost:11434/v1",
    "model_name": "qwen3.5:9b",
    "temperature": 0.0,
    "timeout": 30000
}

llm_model = "qwen3.5:9b"

from pydantic import Field

# 2. 定义工具类
class CurrentTimeTool(BaseTool):
    """获取当前时间的工具"""
    name: str = Field(default="current_time_tool")
    description: str = Field(default="获取当前的日期和时间，用于规划行程时间")
    
    def _run(self):
        """执行工具，返回当前时间"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class TripEstimationTool(BaseTool):
    """行程预估工具"""
    name: str = Field(default="trip_estimation_tool")
    description: str = Field(default="根据起始地、目的地和交通方式估算行程时间和距离")
    
    def _run(self, start_location: str, end_location: str, transport_type: str):
        """执行工具，估算行程
        
        Args:
            start_location: 起始地点
            end_location: 目标地点
            transport_type: 交通方式（如：driving, public_transport, flight）
            
        Returns:
            包含行程信息的字典
        """
        # 模拟行程估算
        # 在实际应用中，这里可以调用地图API获取真实数据
        estimations = {
            "driving": {
                "duration": "2小时30分钟",
                "distance": "150公里",
                "cost": "约100元"
            },
            "public_transport": {
                "duration": "3小时15分钟",
                "distance": "150公里",
                "cost": "约50元"
            },
            "flight": {
                "duration": "45分钟",
                "distance": "150公里",
                "cost": "约300元"
            }
        }
        
        if transport_type in estimations:
            return {
                "start_location": start_location,
                "end_location": end_location,
                "transport_type": transport_type,
                **estimations[transport_type]
            }
        else:
            return {
                "error": f"不支持的交通方式: {transport_type}",
                "supported_types": list(estimations.keys())
            }

class ReminderTool(BaseTool):
    """提醒工具"""
    name: str = Field(default="reminder_tool")
    description: str = Field(default="创建行程提醒，将计划保存为文本文件")
    
    def _run(self, trip_plan: str, filename: str = "trip_reminder.txt"):
        """执行工具，创建提醒文件
        
        Args:
            trip_plan: 行程计划内容
            filename: 保存文件名
            
        Returns:
            操作结果
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"行程提醒\n")
                f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                f.write(trip_plan)
            return f"提醒已创建并保存到 {filename}"
        except Exception as e:
            return f"创建提醒失败: {str(e)}"

# 3. 创建工具实例
current_time_tool = CurrentTimeTool()
trip_estimation_tool = TripEstimationTool()
reminder_tool = ReminderTool()

# 4. 创建工具列表
tools = [current_time_tool, trip_estimation_tool, reminder_tool]

# 3. 创建具有不同专业能力的智能体
trip_planner_agent = Agent(
    role='高级旅行规划师',
    goal='制定最优的跨城出行方案，确保准时到达目的地并应对潜在风险',
    backstory=(
        '你是一位经验丰富的旅行规划专家，精通各种交通方式的优缺点，'
        '擅长分析时间约束、预算限制和风险因素，能够制定详细可靠的出行计划。'
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm_model,
    llm_config=llm_config,
    tools=tools
)

logistics_coordinator = Agent(
    role='物流协调员',
    goal='协调具体的交通预订、住宿安排和后勤保障',
    backstory=(
        '你是一位精细的后勤专家，擅长处理预订、时间安排和资源协调，'
        '确保每个环节都无缝衔接。'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
    llm_config=llm_config,
    tools=tools
)

risk_assessment_agent = Agent(
    role='风险评估专家',
    goal='识别潜在风险并制定应急预案',
    backstory=(
        '你是一位风险管理专家，擅长预见各种可能出现的问题，'
        '并制定有效的应对策略。'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
    llm_config=llm_config,
    tools=[tools[0]]  # 只需要获取当前时间的工具
)

detail_executor = Agent(
    role='执行专员',
    goal='制定具体的执行清单和提醒事项',
    backstory=(
        '你是一位注重细节的执行专家，擅长将计划转化为可操作的清单，'
        '确保每个细节都得到妥善处理。'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
    llm_config=llm_config,
    tools=[tools[0], tools[2]]  # 需要获取当前时间和创建提醒的工具
)

# 3. 定义高层任务 - 让Trip Planner进行自主规划
main_trip_task = Task(
    description=(
        "智能体当前状态：在家中的书房里，现在是2026年4月26日晚上8点。"
        "需要在明天（2026年4月27日）早上9点到达另一个城市的客户公司参加重要会议。"
        "目标：明天上午8:50前出现在客户公司的会议室中，准备就绪。"
        "\n\n"
        "请制定一个完整的出行计划，包括："
        "1. 分析最佳交通方式"
        "2. 制定详细的行程安排"
        "3. 考虑风险因素和应急预案"
        "4. 提供执行清单"
    ),
    expected_output=(
        "一份完整的出行规划报告，包含：\n"
        "1. 交通方案比较与推荐\n"
        "2. 详细的时间安排表\n" 
        "3. 风险评估与应急预案\n"
        "4. 出行准备清单\n"
        "5. 具体的执行步骤"
    ),
    agent=trip_planner_agent
)

# 4. 让Manager Agent自主分解任务
# 在hierarchical模式下，Manager会自动调用其他Agent来完成子任务
crew = Crew(
    agents=[trip_planner_agent, logistics_coordinator, risk_assessment_agent, detail_executor],
    tasks=[main_trip_task],
    process=Process.hierarchical,  # 关键：使用层级流程，让Manager自主规划
    manager_llm=llm_model,  # 为Manager指定LLM模型名称
    manager_llm_config=llm_config,  # 为Manager指定LLM配置参数
    verbose=True,
    planning=True,  # 启用内置规划功能
    memory=True
)

print("## 智能体开始规划跨城出行方案 ##")
print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("智能体需要规划明天的重要商务出行...")
print("-" * 60)

result = crew.kickoff()

print("\n\n" + "="*80)
print("## 完整的出行规划方案 ##")
print("="*80)
print(result)