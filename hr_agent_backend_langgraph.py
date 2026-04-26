# load core modules
import os
from typing import List, Dict, Any, Optional, Annotated, TypedDict, Sequence, Tuple, Union
import json
import operator
from pathlib import Path

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Vector store and embeddings
from langchain_community.vectorstores import FAISS
import requests
import pandas as pd

# LLM
from langchain_community.chat_models import ChatOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(
  disable_batch=True, 
  api_key="tl_995eca70959f4e15b059b10215f0cba8"
)

@workflow(name="ollama_embed_query")
def ollama_embed_query(text: str):
    """
    使用Ollama API生成文本嵌入
    """
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]

class OllamaEmbeddings:
    """
    Ollama嵌入包装类，提供LangChain期望的接口
    """
    def __init__(self, model: str = "nomic-embed-text:latest"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档嵌入向量
        """
        embeddings = []
        for text in texts:
            embedding = ollama_embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成查询嵌入向量
        """
        return ollama_embed_query(text)
        
    # 添加__call__方法以使对象可调用
    def __call__(self, text: str) -> List[float]:
        """
        使对象可调用，返回查询嵌入向量
        """
        return self.embed_query(text)

@workflow(name="load_hr_policy_vectorstore")
def load_hr_policy_vectorstore():
    """
    加载预构建的HR政策FAISS向量数据库
    """
    faiss_path = "hr_policy_faiss"
    
    if Path(faiss_path).exists():
        print(f"正在加载预构建的HR政策向量数据库: {faiss_path}")
        try:
            embeddings = OllamaEmbeddings()
            vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            print("HR政策向量数据库加载成功")
            return vectorstore
        except Exception as e:
            print(f"加载预构建向量数据库失败: {e}")
            print("将使用示例文档作为后备方案")
            return create_fallback_vectorstore()
    else:
        print("未找到预构建的HR政策向量数据库")
        print("请先运行 init_hr_policy_faiss.py 来初始化向量数据库")
        print("将使用示例文档作为后备方案")
        return create_fallback_vectorstore()

@workflow(name="create_fallback_vectorstore")
def create_fallback_vectorstore():
    """
    创建后备的示例文档向量数据库
    """
    from langchain.schema import Document
    
    print("创建后备示例文档向量数据库...")
    
    # Create some sample documents for timekeeping policies
    sample_docs = [
        Document(page_content="Employees must clock in and out daily. Late arrivals after 9:00 AM require manager approval.", metadata={"source": "timekeeping_policy"}),
        Document(page_content="Vacation leave must be requested at least 2 weeks in advance. Unused vacation days carry over to next year.", metadata={"source": "vacation_policy"}),
        Document(page_content="Sick leave requires doctor's note for absences longer than 3 days. Emergency sick leave can be reported within 24 hours.", metadata={"source": "sick_leave_policy"}),
        Document(page_content="Overtime must be pre-approved by supervisor. Overtime rate is 1.5x regular hourly rate.", metadata={"source": "overtime_policy"}),
        Document(page_content="Break time: 15 minutes in morning, 30 minutes for lunch, 15 minutes in afternoon.", metadata={"source": "break_policy"}),
    ]

    # Create FAISS vectorstore with sample documents
    texts = [doc.page_content for doc in sample_docs]
    metadatas = [doc.metadata for doc in sample_docs]

    # Create FAISS vectorstore
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=sample_docs,
        embedding=embeddings
    )
    
    print("后备示例文档向量数据库创建完成")
    return vectorstore

@workflow(name="load_employee_data")
def load_employee_data():
    """
    加载员工数据
    """
    try:
        return pd.read_csv("employee_data.csv")
    except Exception as e:
        print(f"加载员工数据失败: {e}")
        # 创建一个简单的示例数据框
        return pd.DataFrame({
            "name": ["陈皮皮", "张三", "李四"],
            "position": ["工程师", "经理", "设计师"],
            "organizational_unit": ["技术部", "管理部", "设计部"],
            "rank": ["P5", "M3", "P4"],
            "supervisor": ["张三", "王五", "张三"],
            "hire_date": ["2020-01-15", "2018-05-20", "2021-03-10"],
            "sick_leave": [8, 5, 10],
            "vacation_leave": [15, 20, 12],
            "overtime_leave": [3, 0, 2]
        })

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user: str
    df: pd.DataFrame
    vectorstore: Any
    route_result: str

# 初始化LLM
def init_llm():
    """
    初始化LLM
    """
    return ChatOpenAI(
        openai_api_key="ollama", 
        openai_api_base="http://localhost:11434/v1",
        model_name="qwen2:latest", 
        temperature=0.0
    )

# 定义工具函数
@workflow(name="search_hr_policies")
def search_hr_policies(state: AgentState, query: str) -> AgentState:
    """
    搜索HR政策
    """
    vectorstore = state["vectorstore"]
    docs = vectorstore.similarity_search(query, k=2)
    results = []
    
    for i, doc in enumerate(docs, 1):
        results.append(f"结果 {i}: {doc.page_content}")
        results.append(f"来源: {doc.metadata}\n")
    
    response = "\n".join(results)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)]
    }

@workflow(name="query_employee_data")
def query_employee_data(state: AgentState, query: str) -> AgentState:
    """
    查询员工数据
    """
    df = state["df"]
    user = state["user"]
    
    try:
        # 安全地执行查询，限制只能查询当前用户的数据
        # 这里使用eval是为了演示，实际生产环境应该使用更安全的方法
        modified_query = query.replace("df", "df[df['name'] == user]")
        result = eval(modified_query)
        response = f"查询结果: {result}"
    except Exception as e:
        response = f"查询失败: {str(e)}"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)]
    }

def calculate(state: AgentState, expression: str) -> AgentState:
    """
    计算表达式
    """
    try:
        # 安全地执行计算，限制只能使用基本运算
        # 这里使用eval是为了演示，实际生产环境应该使用更安全的方法
        result = eval(expression, {"__builtins__": {}}, {"math": __import__("math")})
        response = f"计算结果: {result}"
    except Exception as e:
        response = f"计算失败: {str(e)}"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)]
    }

# 定义系统提示
@workflow(name="get_system_prompt")
def get_system_prompt(user: str) -> str:
    return f"""你是友好的HR助手。你的任务是协助当前用户: {user} 解答与HR相关的问题。
    
你可以使用以下工具:
1. HR政策搜索 - 用于查询公司的HR政策，包括考勤、休假、病假、出勤等员工政策
2. 员工数据查询 - 用于查询员工数据，如休假余额、职位、部门等
3. 计算器 - 用于执行数学运算

请根据用户的问题，决定使用哪个工具来回答。如果不需要使用工具，可以直接回答。

回答时要保持友好、专业的态度，并提供准确的信息。
"""

# 定义路由函数
@workflow(name="route_tool")
def route_tool(state: AgentState) -> AgentState:
    """
    根据用户问题路由到合适的工具
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个路由助手，负责确定用户问题应该使用哪个工具来回答。
        
可用的工具有:
- hr_policies: 用于查询HR政策相关问题
- employee_data: 用于查询员工个人数据
- calculator: 用于执行数学计算
- none: 不需要使用工具，可以直接回答

请根据用户问题，返回应该使用的工具名称（仅返回工具名称，不要有其他文本）。"""),
        ("human", "{input}")
    ])
    
    # 使用LLM确定路由
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": last_message})
    
    # 清理结果
    result = result.strip().lower()
    
    route_result = "direct_response"  # 默认路由
    
    if "hr_policies" in result or "政策" in result:
        route_result = "hr_policies"
    elif "employee_data" in result or "员工" in result or "数据" in result:
        route_result = "employee_data"
    elif "calculator" in result or "计算" in result:
        route_result = "calculator"
    
    # 将路由结果添加到状态中并返回更新后的状态
    return {
        **state,
        "route_result": route_result
    }

# 定义直接回答函数
@workflow(name="direct_response")
def direct_response(state: AgentState) -> AgentState:
    """
    直接回答用户问题
    """
    messages = state["messages"]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(state["user"])),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 使用LLM生成回答
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": messages[:-1],
        "input": messages[-1].content
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)]
    }

# 定义HR政策工具节点
@workflow(name="hr_policies")
def hr_policies_node(state: AgentState) -> AgentState:
    """
    HR政策工具节点
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是HR政策专家。请根据用户的问题，生成一个查询来搜索相关的HR政策。
        
只返回搜索查询，不要有其他文本。例如：
- 如果用户问"休假政策是什么"，返回：休假政策
- 如果用户问"如何申请病假"，返回：病假申请流程"""),
        ("human", "{input}")
    ])
    
    # 使用LLM生成查询
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    query = chain.invoke({"input": last_message})
    
    # 搜索HR政策
    result_messages = search_hr_policies(state, query)
    
    # 只返回更新后的messages
    return {"messages": result_messages["messages"]}

# 定义员工数据工具节点
@workflow(name="employee_data")
def employee_data_node(state: AgentState) -> AgentState:
    """
    员工数据工具节点
    """
    messages = state["messages"]
    last_message = messages[-1].content
    user = state["user"]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是员工数据专家。请根据用户的问题，生成一个pandas查询表达式来获取所需信息。
        
可用的DataFrame是df，包含以下列: name, position, organizational_unit, rank, supervisor, hire_date, sick_leave, vacation_leave, overtime_leave

只返回pandas查询表达式，不要有其他文本。例如：
- 如果用户问"我有多少天病假"，返回：df[df['name'] == '{user}']['sick_leave']
- 如果用户问"我的职位是什么"，返回：df[df['name'] == '{user}']['position']"""),
        ("human", "{input}")
    ])
    
    # 使用LLM生成查询表达式
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    query = chain.invoke({"input": last_message, "user": user})
    
    # 执行查询
    result_messages = query_employee_data(state, query)
    
    # 只返回更新后的messages
    return {"messages": result_messages["messages"]}

# 定义计算器工具节点
@workflow(name="calculator")
def calculator_node(state: AgentState) -> AgentState:
    """
    计算器工具节点
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是计算专家。请根据用户的问题，生成一个Python表达式来执行计算。
        
只返回Python表达式，不要有其他文本。例如：
- 如果用户问"1+1等于多少"，返回：1+1
- 如果用户问"8小时加班费是多少，假设时薪100元"，返回：8*100*1.5"""),
        ("human", "{input}")
    ])
    
    # 使用LLM生成计算表达式
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    expression = chain.invoke({"input": last_message})
    
    # 执行计算
    result_messages = calculate(state, expression)
    
    # 只返回更新后的messages
    return {"messages": result_messages["messages"]}

# 定义最终回答函数
@workflow(name="final_response")
def final_response(state: AgentState) -> AgentState:
    """
    生成最终回答
    """
    messages = state["messages"]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(state["user"])),
        MessagesPlaceholder(variable_name="history"),
        ("human", "根据以上信息，请给出最终回答。")
    ])
    
    # 使用LLM生成最终回答
    llm = init_llm()
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": messages
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)]
    }

# 构建LangGraph
@workflow(name="build_graph")
def build_graph():
    """
    构建LangGraph
    """
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("route", route_tool)
    workflow.add_node("hr_policies", hr_policies_node)
    workflow.add_node("employee_data", employee_data_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("direct_response", direct_response)
    workflow.add_node("final_response", final_response)
    
    # 添加边
    # 添加入口点
    workflow.set_entry_point("route")
    
    workflow.add_edge("route", "hr_policies")
    workflow.add_edge("route", "employee_data")
    workflow.add_edge("route", "calculator")
    workflow.add_edge("route", "direct_response")
    
    workflow.add_edge("hr_policies", "final_response")
    workflow.add_edge("employee_data", "final_response")
    workflow.add_edge("calculator", "final_response")
    workflow.add_edge("direct_response", "final_response")
    
    workflow.add_edge("final_response", END)
    
    # 设置条件边
    workflow.add_conditional_edges(
        "route",
        lambda state: state["route_result"],
        {
            "hr_policies": "hr_policies",
            "employee_data": "employee_data",
            "calculator": "calculator",
            "direct_response": "direct_response"
        }
    )
    
    # 编译图
    return workflow.compile()

# 初始化图
graph = build_graph()

# 加载资源
vectorstore = load_hr_policy_vectorstore()
df = load_employee_data()

# 定义响应函数
@workflow(name="get_response")
def get_response(user_input: str, user: str = "陈皮皮"):
    """
    处理用户输入并返回响应
    """
    # 初始化状态
    state = {
        "messages": [HumanMessage(content=user_input)],
        "user": user,
        "df": df,
        "vectorstore": vectorstore,
        "route_result": "direct_response"  # 默认路由
    }
    
    # 运行图
    result = graph.invoke(state)
    
    # 提取最终回答
    final_message = result["messages"][-1].content
    return final_message

# 测试函数
def test():
    """
    测试HR聊天机器人
    """
    test_questions = [
        "我有多少天病假？",
        "年假政策是什么？",
        "加班怎么计算？",
        "我的职位是什么？",
        "What is the vacation leave policy?",
        "How many days of sick leave am I entitled to?"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        response = get_response(question)
        print(f"回答: {response}")

# 如果直接运行此脚本，执行测试
if __name__ == "__main__":
    test()