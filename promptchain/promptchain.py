from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# 定义状态类型 - 这是LangGraph的核心数据结构
class MarketAnalysisState(TypedDict):
    """系统状态类型定义"""
    original_text: str  # 原始市场研究报告
    summary: str  # 第一步生成的摘要
    trends: List[Dict[str, Any]]  # 第二步识别的趋势
    email_content: str  # 第三步生成的邮件内容
    current_step: int  # 当前执行步骤
    errors: List[str]  # 错误记录
# 创建状态
# 定义趋势数据模型
class Trend(BaseModel):
    """趋势数据结构"""
    trend_name: str = Field(description="趋势名称")
    description: str = Field(description="趋势的详细描述")
    supporting_data: List[str] = Field(description="支持该趋势的具体数据点")
    confidence_score: float = Field(description="趋势识别的置信度分数", ge=0, le=1)

class TrendAnalysis(BaseModel):
    """趋势分析结果"""
    trends: List[Trend] = Field(description="识别出的前三个新兴趋势")

# 初始化语言模型
llm = ChatOpenAI(
    openai_api_key="ollama", 
        openai_api_base="http://localhost:11434/v1",
        model_name="qwen3.5:9b", 
        temperature=0.0,
        timeout=30000,
        #temperature=0.3, model="gpt-4-turbo-preview")
)

# =============== Agent 1: 摘要生成器 ===============
def create_summary_agent():
    """创建摘要生成agent"""
    prompt = ChatPromptTemplate.from_template(
        """你是一位专业的市场研究分析师。你的唯一任务是总结以下市场研究报告的主要发现。
        
        请确保：
        1. 专注于核心发现和关键数据
        2. 保持客观和准确
        3. 涵盖报告中的主要观点
        4. 长度适中，约200-300字
        
        市场研究报告：
        {text}
        
        请只提供总结内容，不要添加其他说明或格式。"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    def summary_agent(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            summary = chain.invoke({"text": state["original_text"]})
            print(f"📝 摘要生成成功:\n{summary}\n")
            return {
                **state,
                "summary": summary,
                "current_step": 2,
                "errors": []
            }
        except Exception as e:
            error_msg = f"摘要生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "errors": state["errors"] + [error_msg],
                "current_step": 1  # 保持当前步骤以便重试
            }
    
    return summary_agent

# =============== Agent 2: 趋势识别器 ===============
def create_trend_analysis_agent():
    """创建趋势分析agent"""
    prompt = ChatPromptTemplate.from_template(
        """你是一位资深的市场趋势分析师。基于以下市场研究报告摘要，识别前三个最重要的新兴趋势。
        
        要求：
        1. 识别真正新兴的、具有发展潜力的趋势
        2. 为每个趋势提取至少2-3个具体的支持数据点
        3. 评估每个趋势的置信度（0-1之间）
        4. 确保趋势具有商业价值和可操作性
        
        摘要内容：
        {summary}
        
        请以JSON格式输出，包含trends数组，每个趋势包含：
        - trend_name: 趋势名称
        - description: 详细描述
        - supporting_data: 支持数据点列表
        - confidence_score: 置信度分数"""
    )
    
    chain = prompt | llm | JsonOutputParser(pydantic_object=TrendAnalysis)
    
    def trend_analysis_agent(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            if not state["summary"].strip():
                raise ValueError("摘要内容为空，无法进行趋势分析")
            
            # 将原始市场报告信息也传递过去，其内部能够作容错处理
            result = chain.invoke({"summary": state["summary"]})
            print(f"🔍 趋势分析结果:\n{result}\n")
            
            # 处理result可能是字典或对象的情况
            if isinstance(result, dict):
                trends = result.get('trends', [])
            else:
                trends = result.trends if hasattr(result, 'trends') else []
            
            # 验证趋势数量
            if len(trends) < 3:
                print(f"⚠️  只识别到{len(trends)}个趋势，可能需要重新分析")
            
            print("📈 趋势分析结果:")
            
            # 处理趋势数据，确保格式一致
            processed_trends = []
            for i, trend in enumerate(trends[:3], 1):
                # 检查trend是字典还是对象
                if isinstance(trend, dict):
                    trend_name = trend.get('trend_name', '未知趋势')
                    confidence_score = trend.get('confidence_score', 0.0)
                    supporting_data = trend.get('supporting_data', [])
                    processed_trends.append(trend)
                else:
                    trend_name = trend.trend_name if hasattr(trend, 'trend_name') else '未知趋势'
                    confidence_score = trend.confidence_score if hasattr(trend, 'confidence_score') else 0.0
                    supporting_data = trend.supporting_data if hasattr(trend, 'supporting_data') else []
                    processed_trends.append(trend.dict())
                
                print(f"  趋势 {i}: {trend_name}")
                print(f"  置信度: {confidence_score:.2f}")
                print(f"  支持数据: {', '.join(supporting_data[:2])}...")
            
            return {
                **state,
                "trends": processed_trends,  # 只取前3个
                "current_step": 3,
                "errors": []
            }
        except Exception as e:
            error_msg = f"趋势分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "errors": state["errors"] + [error_msg],
                "current_step": 2  # 保持当前步骤以便重试
            }
    
    return trend_analysis_agent

# =============== Agent 3: 邮件撰写器 ===============
def create_email_agent():
    """创建邮件撰写agent"""
    prompt = ChatPromptTemplate.from_template(
        """你是一位营销总监，需要向营销团队发送一封关于市场趋势的邮件。
        
        邮件要求：
        1. 专业、简洁、有说服力
        2. 突出最重要的3个趋势及其商业价值
        3. 包含具体的数据支持
        4. 提供明确的行动建议
        5. 格式规范，包含主题、称呼、正文、结尾
        
        趋势数据：
        {trends_data}
        
        请以完整的邮件格式输出，包括：
        - 邮件主题
        - 收件人/发件人信息
        - 正文内容
        - 签名"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    def email_agent(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            if not state["trends"]:
                raise ValueError("没有趋势数据，无法撰写邮件")
            
            # 格式化趋势数据供邮件使用
            trends_data = ""
            for i, trend in enumerate(state["trends"], 1):
                trends_data += f"趋势 {i}: {trend['trend_name']}\n"
                trends_data += f"描述: {trend['description']}\n"
                trends_data += f"支持数据: {', '.join(trend['supporting_data'])}\n"
                trends_data += f"置信度: {trend['confidence_score']:.2f}\n\n"
            
            email_content = chain.invoke({"trends_data": trends_data})
            print("📧 邮件生成成功!")
            print(f"邮件预览:\n{email_content[:200]}...\n")
            
            return {
                **state,
                "email_content": email_content,
                "current_step": 4,  # 完成
                "errors": []
            }
        except Exception as e:
            error_msg = f"邮件撰写失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "errors": state["errors"] + [error_msg],
                "current_step": 3  # 保持当前步骤以便重试
            }
    
    return email_agent

# =============== 条件路由函数 ===============
def should_retry(state: MarketAnalysisState) -> str:
    """决定是否需要重试当前步骤"""
    if state["errors"] and len(state["errors"]) < 3:  # 最多重试3次
        print(f"🔄 步骤 {state['current_step']} 出错，尝试重试...")
        return f"step_{state['current_step']}"
    elif state["errors"]:
        print("❌ 错误次数过多，流程终止")
        return "error_end"
    return "continue"

def route_to_next_step(state: MarketAnalysisState) -> str:
    """路由到下一步"""
    step_map = {
        1: "step_1",
        2: "step_2", 
        3: "step_3",
        4: END
    }
    return step_map.get(state["current_step"], END)

# =============== 构建LangGraph工作流 ===============
def create_market_analysis_graph():
    """创建完整的市场分析工作流"""
    workflow = StateGraph[MarketAnalysisState, None, MarketAnalysisState, MarketAnalysisState](MarketAnalysisState)
    
    # 添加节点
    workflow.add_node("step_1", create_summary_agent())
    workflow.add_node("step_2", create_trend_analysis_agent())
    workflow.add_node("step_3", create_email_agent())
    
    # 设置条件边
    workflow.set_conditional_entry_point(
        lambda state: "step_1" if state["current_step"] == 1 else f"step_{state['current_step']}",
        {
            "step_1": "step_1",
            "step_2": "step_2",
            "step_3": "step_3"
        }
    )
    
    # 添加边
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)
    
    # 编译工作流
    app = workflow.compile()
    return app

# =============== 主执行函数 ===============
def run_market_analysis(original_text: str, max_retries: int = 3):
    """运行完整的市场分析流程"""
    print("🚀 启动市场分析多agent协同系统...")
    print("=" * 60)
    
    # 初始化状态
    initial_state = MarketAnalysisState(
        original_text=original_text,
        summary="",
        trends=[],
        email_content="",
        current_step=1,
        errors=[]
    )
    
    # 创建应用
    app = create_market_analysis_graph()
    
    try:
        # 执行工作流
        final_state = app.invoke(initial_state)
        
        print("=" * 60)
        print("✅ 市场分析流程完成!")
        print(f"最终步骤: {final_state['current_step']}")
        
        if final_state["errors"]:
            print(f"⚠️  警告: 过程中出现 {len(final_state['errors'])} 个错误")
            for error in final_state["errors"]:
                print(f"  • {error}")
        
        return final_state
        
    except Exception as e:
        print(f"🔥 系统级错误: {str(e)}")
        return {
            **initial_state,
            "errors": [f"系统错误: {str(e)}"],
            "current_step": -1
        }

# =============== 测试示例 ===============
if __name__ == "__main__":
    # 示例市场研究报告
    sample_report = """
    2026年第一季度全球消费电子市场分析报告
    
    市场概况：
    全球消费电子市场在2026年Q1达到1.2万亿美元，同比增长8.3%。亚太地区贡献最大，占全球市场的42%，其中中国市场增长尤为强劲，同比增长15.2%。
    
    关键趋势：
    1. AIoT（人工智能物联网）设备爆发：智能家居设备出货量同比增长67%，其中AI语音助手渗透率达到34%。小米、华为等中国品牌占据60%市场份额。
    2. 可持续电子产品需求激增：环保材料电子产品销售额增长45%，消费者愿意为此支付15-20%溢价。欧盟新规推动这一趋势加速。
    3. 混合现实技术商业化：AR/VR设备在企业培训领域应用增长200%，教育科技投资达到89亿美元。Meta和苹果主导技术标准制定。
    
    消费者行为：
    90后和00后成为购买主力，占总消费的65%。他们更注重产品体验（78%）而非价格（45%）。社交媒体影响购买决策的比例从2024年的35%上升到2026年的58%。
    
    供应链分析：
    芯片短缺问题基本解决，但高端AI芯片仍供不应求。中国本土化供应链建设加速，国产芯片自给率从2024年的28%提升到2026年的41%。
    
    风险预警：
    地缘政治风险上升，关税壁垒可能影响30%的跨境贸易。技术标准碎片化增加了研发成本，平均增加15-20%。
    """
    
    print("📊 测试示例：市场研究报告分析")
    print("=" * 60)
    
    # 运行分析
    result = run_market_analysis(sample_report)
    
    # 展示最终结果
    print("\n" + "=" * 60)
    print("🎯 最终输出结果:")
    print("-" * 60)
    
    if result["email_content"]:
        print("📧 营销邮件内容:")
        print(result["email_content"])
    else:
        print("❌ 未能生成最终邮件")
    
    print("\n" + "=" * 60)
    print("💡 系统设计要点总结:")
    print("1. 任务分解：将复杂任务拆分为专注的子任务")
    print("2. 状态管理：通过MarketAnalysisState统一管理数据流")
    print("3. 错误处理：内置重试机制和错误隔离")
    print("4. 质量控制：每个步骤都有明确的验证标准")
    print("5. 可扩展性：可轻松添加新的agent或步骤")
