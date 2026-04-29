"""
旅行规划Agent - 使用Ollama Qwen模型
目标：根据用户需求和目标，生成并优化旅行计划
"""

import os
import random
import re
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv, find_dotenv

## 🔐 加载环境变量
_ = load_dotenv(find_dotenv())

## ✅ 初始化Ollama模型
print("📡 初始化Ollama LLM (qwen3.5:9b)...")
llm = ChatOllama(
    model="qwen3.5:9b",  # 或者使用 "ollama/qwen3:1.7b"
    base_url="http://localhost:11434",
    temperature=0.3,
    # num_predict=512,  # 限制生成长度
)

## --- 实用函数 ---

def generate_travel_prompt(
    destination: str, 
    duration: int, 
    budget: int, 
    goals: list[str], 
    previous_plan: str = "", 
    feedback: str = ""
) -> str:
    """生成旅行计划的提示词"""
    print("📝 构建旅行计划提示词...")
    
    base_prompt = f"""
你是一个专业的旅行规划师AI。你的工作是为用户创建详细的旅行计划。

旅行信息：
- 目的地：{destination}
- 行程天数：{duration}天
- 预算：¥{budget}
- 核心目标：{', '.join(goals)}

请创建一个详细的旅行计划，包括：
1. 每日行程安排（具体到上午、下午、晚上）
2. 住宿和餐饮建议
3. 交通方式
4. 预算分配（各项费用预估）
5. 特别注意事项
"""
    
    if previous_plan:
        print("🔄 将之前的计划添加到提示词中以进行完善。")
        base_prompt += f"\n之前生成的计划：\n{previous_plan}"
    
    if feedback:
        print("📋 包含反馈以进行修订。")
        base_prompt += f"\n对之前版本的反馈：\n{feedback}\n"
    
    base_prompt += "\n请仅返回修订后的旅行计划。不要在计划之外包含额外的解释或评论。使用中文输出。"
    return base_prompt

def get_plan_feedback(plan: str, goals: list[str]) -> str:
    """获取对旅行计划的反馈"""
    print("🔍 根据目标评估旅行计划...")
    
    feedback_prompt = f"""
你是一个旅行规划审查专家。下面显示了一个旅行计划。

基于以下目标：
{chr(10).join(f"- {g.strip()}" for g in goals)}

请对此旅行计划进行严格审查，评估是否满足所有目标。重点关注：
- 预算是否合理分配且不超出总预算
- 每日行程是否充实但不过于紧张
- 是否包含所有要求的目标元素
- 交通和住宿安排是否合理
- 是否考虑了实用性和可行性

如果计划有不足，请具体指出需要改进的地方。

旅行计划：
{plan}
"""
    
    response = llm.invoke(feedback_prompt)
    return response.content.strip()

def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """
    使用LLM根据反馈文本评估目标是否已达成。
    返回True或False。
    """
    review_prompt = f"""
你是一个严格的旅行规划审查员。这些是原始目标：
{chr(10).join(f"- {g.strip()}" for g in goals)}

这是关于旅行计划的反馈：
\"\"\"
{feedback_text}
\"\"\"

根据上述反馈，所有目标是否都已完全达成？请仅用一个词回答：True 或 False。
"""
    
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true" or response == "是"

def clean_plan_format(plan: str) -> str:
    """清理计划格式，移除多余的Markdown标记"""
    # 移除代码块标记
    plan = re.sub(r'^```[\w]*\n', '', plan, flags=re.MULTILINE)
    plan = re.sub(r'\n```$', '', plan, flags=re.MULTILINE)
    
    # 移除多余的空行
    plan = re.sub(r'\n{3,}', '\n\n', plan)
    
    return plan.strip()

def add_plan_header(plan: str, destination: str, duration: int, budget: int, goals: list[str]) -> str:
    """为计划添加标题和基本信息"""
    header = f"""
# 旅行计划
**目的地**: {destination}
**行程天数**: {duration}天
**总预算**: ¥{budget}
**核心目标**: {', '.join(goals)}

{'=' * 50}
"""
    return header + "\n" + plan

def to_snake_case(text: str) -> str:
    """将文本转换为蛇形命名格式"""
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return re.sub(r"\s+", "_", text.strip().lower())

def save_plan_to_file(plan: str, destination: str) -> str:
    """将最终计划保存到文件"""
    print("💾 保存最终旅行计划到文件...")
    
    # 生成文件名
    short_name = to_snake_case(destination)[:10]
    random_suffix = str(random.randint(1000, 9999))
    filename = f"travel_plan_{short_name}_{random_suffix}.txt"
    filepath = Path.cwd() / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(plan)
    
    print(f"✅ 旅行计划保存到：{filepath}")
    return str(filepath)

## --- 主智能体 ---

def run_travel_agent(
    destination: str, 
    duration: int, 
    budget: int, 
    goals_input: str, 
    max_iterations: int = 3
) -> str:
    """主执行函数"""
    goals = [g.strip() for g in goals_input.split(",")]
    print(f"\n🎯 旅行目的地：{destination}")
    print(f"📅 行程天数：{duration}天")
    print(f"💰 预算：¥{budget}")
    print("🎯 旅行目标：")
    for g in goals:
        print(f"  - {g}")
    
    previous_plan = ""
    feedback = ""
    
    for i in range(max_iterations):
        print(f"\n=== 🔁 迭代 {i + 1} / {max_iterations} ===")
        
        # 生成提示词
        prompt = generate_travel_prompt(
            destination, duration, budget, goals, 
            previous_plan, feedback
        )
        
        print("✈️ 生成旅行计划...")
        plan_response = llm.invoke(prompt)
        print(plan_response.content)
        
        raw_plan = plan_response.content.strip()
        clean_plan = clean_plan_format(raw_plan)
        
        print("\n🧾 生成的旅行计划：\n" + "-" * 50)
        print(clean_plan[:500] + "..." if len(clean_plan) > 500 else clean_plan)  # 只显示部分
        print("-" * 50)
        
        print("\n📤 提交计划进行反馈审查...")
        feedback = get_plan_feedback(clean_plan, goals)
        
        print("\n📥 收到反馈：\n" + "-" * 50)
        print(feedback[:300] + "..." if len(feedback) > 300 else feedback)  # 只显示部分
        print("-" * 50)
        
        if goals_met(feedback, goals):
            print("✅ LLM确认所有目标已达成。停止迭代。")
            final_plan = clean_plan
            break
        
        print("🔄 目标尚未完全达成。准备下一次迭代...")
        previous_plan = clean_plan
    else:
        print("⚠️ 达到最大迭代次数，使用当前最佳计划")
        final_plan = clean_plan
    
    # 添加标题和保存
    final_plan_with_header = add_plan_header(final_plan, destination, duration, budget, goals)
    return save_plan_to_file(final_plan_with_header, destination)

## --- CLI 测试运行 ---

if __name__ == "__main__":
    print("\n🌍 欢迎使用 AI 旅行规划 Agent")
    print("=" * 50)
    
    # 示例1：京都文化之旅
    print("\n" + "="*20 + " 示例1：京都文化之旅 " + "="*20)
    destination = "日本京都"
    duration = 5
    budget = 15000
    goals_input = "深度文化体验,预算合理分配,包含著名寺庙参观,体验传统日式料理,交通便利"
    
    run_travel_agent(destination, duration, budget, goals_input, max_iterations=7)
    
    # 示例2：云南自然风光之旅
    print("\n" + "="*20 + " 示例2：云南自然风光之旅 " + "="*20)
    destination = "中国云南"
    duration = 7
    budget = 10000
    goals_input = "自然风光为主,徒步体验,摄影机会多,当地特色美食,经济实惠"
    
    run_travel_agent(destination, duration, budget, goals_input, max_iterations=7)
    
    # 示例3：欧洲城市探索
    print("\n" + "="*20 + " 示例3：欧洲城市探索 " + "="*20)
    destination = "法国巴黎"
    duration = 4
    budget = 20000
    goals_input = "浪漫体验,艺术博物馆,美食购物,高效行程,中等预算"
    
    run_travel_agent(destination, duration, budget, goals_input, max_iterations=7)
    
    print("\n✨ 旅行规划完成！所有计划已保存到文件。")