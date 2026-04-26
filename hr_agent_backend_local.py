# load core modules

from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# load agents and tools modules
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMMathChain

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], 
#                 base_url=os.environ["OPENAI_BASE_URL"]) #"https://api.openai-proxy.com/v1")

Traceloop.init(
  disable_batch=True, 
  api_key="tl_995eca70959f4e15b059b10215f0cba8"
)


# Replace Pinecone with FAISS for simplicity
# from pinecone import Pinecone

# Initialize with API key only
# pinecone = Pinecone(api_key="pcsk_267CTD_PrE2LN7ZUGYpg27r6td6ZuH8zbBEDmvUWRompavRrNJTMmNA7f54VWEJjEJHoGe")

# List indexes to verify connection
print("Using FAISS instead of Pinecone for simplicity")
# print(pinecone.list_indexes())

# initialize pinecone client and connect to pinecone index
# pinecone.init(
#         # api_key="pcsk_3robfD_PJdSHCqGVszkBG6sSRRCxsrYKGXzevwZzXgvJwftxTLYZmjiW8SZJAYcDHmJ6z1",  
#         api_key="pcsk_267CTD_PrE2LN7ZUGYpg27r6td6ZuH8zbBEDmvUWRompavRrNJTMmNA7f54VWEJjEJHoGe",
#         environment="aped-4627-b74a"  
#         # environment="gcp-starter"
# ) 

# index_name = 'tk-policy'
# index = pinecone.Index(index_name) # connect to pinecone index

# initialize embeddings object; for use with user query/input
# embed = OpenAIEmbeddings(
#                 model = 'text-embedding-ada-002',
#                 openai_api_key="<your openai api key from from platform.openai.com>",
#             )

# Replace OpenAIEmbeddings with a direct call to Ollama embeddings API
import requests

@workflow(name="ollama_embed_query")
def ollama_embed_query(text: str):
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
        
    # 添加__call__方法以使对象可调用，解决TypeError: 'OllamaEmbeddings' object is not callable
    def __call__(self, text: str) -> List[float]:
        """
        使对象可调用，返回查询嵌入向量
        """
        return self.embed_query(text)

# Create a simple in-memory vectorstore using FAISS
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pathlib import Path

# Load the pre-built HR policy FAISS vector database
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

# Load the HR policy vector database
vectorstore = load_hr_policy_vectorstore()

# llm = ChatOpenAI(    
#     openai_api_key="<your openai api key from from platform.openai.com>", 
#     model_name="gpt-3.5-turbo", 
#     temperature=0.0
#     )

llm = ChatOpenAI(    
    openai_api_key="ollama", 
    openai_api_base="http://localhost:11434/v1",
    model_name="qwen2:latest", 
    temperature=0.0
    )

# initialize vectorstore retriever object
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

df = pd.read_csv("employee_data.csv") # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the tools

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = '陈皮皮' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "HR Policies",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about HR policies including timekeeping, vacation, sick leave, attendance, and other employee policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the HR policies to answer this question.
        <assistant>: Action: HR Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """
    ),
    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        当需要回答关于存储在pandas数据框'df'中的员工数据问题时，这个模板很有用。
        可以使用python pandas操作在'df'上运行来获取正确答案。
        'df'包含以下列: {df_columns}
        
        example a:
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        
        example b:
        <user>: How many days do I have left in my vacation leave?
        <assistant>: df[df['name'] == '{user}']['vacation_leave']
        <assistant>: You have n days of vacation leave left.
        
        example c:
        <user>: How many days do I have left in my overtime leave?
        <assistant>: df[df['name'] == '{user}']['overtime_leave']
        <assistant>: You have n days of overtime leave left.
        
        example d:
        <user>: What is my hire date?
        <assistant>: df[df['name'] == '{user}']['hire_date']
        <assistant>: You were hired on hire_date.
        
        example e:
        <user>: What is my position?
        <assistant>: df[df['name'] == '{user}']['position']
        <assistant>: You are a position.
        
        example f:
        <user>: What is my organizational unit?
        <assistant>: df[df['name'] == '{user}']['organizational_unit']
        <assistant>: You are in organizational_unit.
        
        example g:
        <user>: What is my rank?
        <assistant>: df[df['name'] == '{user}']['rank']
        <assistant>: You are a rank.
        
        example h:
        <user>: What is my supervisor?
        <assistant>: df[df['name'] == '{user}']['supervisor']
        <assistant>: Your supervisor is supervisor.
        
        示例A:
        <用户>: 我还剩多少天病假？
        <助手>: df[df['name'] == '{user}']['sick_leave']
        <助手>: 您还剩下n天病假。

        示例B:
        <用户>: 我还剩多少天年假？
        <助手>: df[df['name'] == '{user}']['vacation_leave']
        <助手>: 您还剩下n天年假。

        示例C:
        <用户>: 我还剩多少天调休假？
        <助手>: df[df['name'] == '{user}']['overtime_leave']
        <助手>: 您还剩下n天调休假。

        示例D:
        <用户>: 我的入职日期是什么时候？
        <助手>: df[df['name'] == '{user}']['hire_date']
        <助手>: 您的入职日期是hire_date。

        示例E:
        <用户>: 我的职位是什么？
        <助手>: df[df['name'] == '{user}']['position']
        <助手>: 您的职位是position。

        示例F:
        <用户>: 我属于哪个部门？
        <助手>: df[df['name'] == '{user}']['organizational_unit']
        <助手>: 您属于organizational_unit部门。

        示例G:
        <用户>: 我的职级是什么？
        <助手>: df[df['name'] == '{user}']['rank']
        <助手>: 您的职级是rank。

        示例H:
        <用户>: 我的主管是谁？
        <助手>: df[df['name'] == '{user}']['supervisor']
        <助手>: 您的主管是supervisor。
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )
# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response
