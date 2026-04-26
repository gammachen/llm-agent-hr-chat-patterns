import streamlit as st
import random
from streamlit_chat import message
from hr_agent_backend_langgraph import get_response


def process_input(user_input):
    """
    处理用户输入并获取响应
    """
    # 从会话状态中获取当前用户
    user = st.session_state.get("current_user", "陈皮皮")
    response = get_response(user_input, user)
    return response

# 设置页面标题和说明
st.set_page_config(page_title="HR 聊天机器人 (LangGraph版)", layout="wide")
st.header("HR 聊天机器人 (LangGraph版)")
st.markdown("在这里提问您的HR相关问题。使用LangGraph实现的智能HR助手。")

# 初始化会话状态
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "input_message_key" not in st.session_state:
    st.session_state["input_message_key"] = str(random.random())
if "current_user" not in st.session_state:
    st.session_state["current_user"] = "陈皮皮"

# 创建侧边栏用于用户选择
with st.sidebar:
    st.header("用户设置")
    
    # 用户选择下拉框
    user_options = ["陈皮皮", "张三", "李四"]
    selected_user = st.selectbox(
        "选择用户",
        options=user_options,
        index=user_options.index(st.session_state["current_user"]),
    )
    
    # 更新当前用户
    if selected_user != st.session_state["current_user"]:
        st.session_state["current_user"] = selected_user
        st.session_state["past"] = []
        st.session_state["generated"] = []
        st.rerun()
    
    # 添加清除对话按钮
    if st.button("清除对话历史"):
        st.session_state["past"] = []
        st.session_state["generated"] = []
        st.rerun()
    
    # 显示示例问题
    st.header("示例问题")
    example_questions = [
        "我有多少天病假？",
        "年假政策是什么？",
        "加班怎么计算？",
        "我的职位是什么？",
        "What is the vacation leave policy?",
        "How many days of sick leave am I entitled to?"
    ]
    
    for q in example_questions:
        if st.button(q):
            st.session_state["past"].append(q)
            response = process_input(q)
            st.session_state["generated"].append(response)
            st.rerun()

# 创建主聊天界面
chat_container = st.container()

# 创建用户输入区域
user_input = st.text_input(
    "输入您的问题并按回车发送",
    key=st.session_state["input_message_key"]
)

# 发送按钮
if st.button("发送"):
    if user_input.strip() != "":
        response = process_input(user_input)
        
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(response)
        
        # 重置输入框
        st.session_state["input_message_key"] = str(random.random())
        
        st.rerun()

# 显示聊天历史
if st.session_state["generated"]:
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

# 显示当前用户信息
st.caption(f"当前用户: {st.session_state['current_user']}")