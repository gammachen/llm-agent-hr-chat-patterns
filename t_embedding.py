import ollama
from pyarrow import scalar
import requests  # 新增: 用于HTTP请求

from traceloop.sdk import Traceloop



import os
from openai import OpenAI

print(os.getenv("OPENAI_BASE_URL"))
print(os.getenv("OPENAI_API_KEY"))

# 修改OpenAI客户端初始化逻辑，使用os.getenv设置默认值
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # 默认值设为空字符串
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1") #"https://api.openai-proxy.com/v1")  # 使用原注释中的默认URL
)
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], 
#                 base_url=os.environ["OPENAI_BASE_URL"]) #"https://api.openai-proxy.com/v1")

Traceloop.init(
  disable_batch=True, 
  api_key="tl_995eca70959f4e15b059b10215f0cba8"
)

@workflow(name="joke_creation")
def create_joke():
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="qwen2:latest",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content

joke = create_joke()
print(joke)

@workflow(name="give_me_a_script")
def give_me_a_script():
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="qwen2:latest",
        messages=[{"role": "user", "content": "请给出姚明在NBA的发展历程，输出格式为html代码"}],
    )

    return completion.choices[0].message.content

scalar = give_me_a_script()
print(scalar)
scalar = give_me_a_script()
print(scalar)
scalar = give_me_a_script()
print(scalar)

# 初始化客户端
client = ollama.Client()

# 测试文本
text = "你好，世界"

# 调用嵌入模型
response = client.embeddings(model='quentinz/bge-small-zh-v1.5:latest', prompt=text)

# 输出结果
print(f"输入文本: {text}")
print(f"嵌入向量维度: {len(response['embedding'])}")
print(f"前5维向量: {response['embedding'][:5]}")

# 调用嵌入模型
response = client.embeddings(model='nomic-embed-text:latest', prompt=text)

# 输出结果
print(f"输入文本: {text}")
print(f"嵌入向量维度: {len(response['embedding'])}")
print(f"前5维向量: {response['embedding'][:5]}")

# 新增: 使用HTTP请求调用嵌入模型
print("=== 通过HTTP请求获取嵌入 ===")
url = "http://localhost:11434/api/embeddings"  # Ollama默认API地址
headers = {"Content-Type": "application/json"}

data = {
    "model": "quentinz/bge-small-zh-v1.5:latest",
    "prompt": text
}

response = requests.post(url, json=data, headers=headers)  # 发送HTTP请求
if response.status_code == 200:
    embedding = response.json()["embedding"]
    print(f"输入文本: {text}")
    print(f"嵌入向量维度: {len(embedding)}")
    print(f"前5维向量: {embedding[:5]}")
else:
    print(f"HTTP请求失败，状态码: {response.status_code}")

'''
(base) shhaofu@shhaofudeMacBook-Pro autonomous-hr-chatbot % python t_embedding.py      
Traceloop exporting traces to https://api.traceloop.com authenticating with bearer token

DependencyConflict: requested: "ollama >= 0.4.0, < 1" but found: "ollama 0.3.1"
ERROR:root:Error initializing SQLAlchemy instrumentor: No module named 'sqlalchemy'
ERROR:root:Error initializing Transformers instrumentor: No module named 'regex'
ERROR:root:Error initializing Gemini instrumentor: No module named 'google.auth'
ERROR:root:Error initializing Bedrock instrumentor: No module named 'botocore.response'
ERROR:root:Error initializing SageMaker instrumentor: No module named 'botocore.response'
ERROR:root:Error initializing LangChain instrumentor: No module named 'regex'
Traceback (most recent call last):
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/t_embedding.py", line 37, in <module>
    joke = create_joke()
           ^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/traceloop/sdk/decorators/base.py", line 267, in sync_wrap
    res = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/t_embedding.py", line 29, in create_joke
    completion = client.chat.completions.create(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/opentelemetry/instrumentation/openai/utils.py", line 88, in wrapper
    return func(
           ^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/opentelemetry/instrumentation/openai/shared/chat_wrappers.py", line 94, in chat_wrapper
    response = wrapped(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/openai/_utils/_utils.py", line 287, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py", line 1150, in create
    return self._post(
           ^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/openai/_base_client.py", line 1259, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.11/site-packages/openai/_base_client.py", line 1047, in request
    raise self._make_status_error_from_response(err.response) from None
openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: ollama. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
'''