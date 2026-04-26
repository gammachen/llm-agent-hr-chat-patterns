import streamlit as st
import random
from streamlit_chat import message
#from hr_agent_backend_azure import get_response
from hr_agent_backend_local import get_response


def process_input(user_input):
    response = get_response(user_input)
    return response

st.header("人力资源_HR Chatbot")
st.markdown("这里可以问与人力资源相关的问题。")
st.markdown("例如：公司的政策、员工的福利、工作环境等。")
st.markdown("注意：本聊天机器人仅用于回答与人力资源相关的问题，其他问题请咨询人工客服。")

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "input_message_key" not in st.session_state:
    st.session_state["input_message_key"] = str(random.random())

chat_container = st.container()

user_input = st.text_input("Type your message and press Enter to send.", key=st.session_state["input_message_key"])

if st.button("Send"):
    response = process_input(user_input)

    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(response)

    st.session_state["input_message_key"] = str(random.random())

    st.rerun()

if st.session_state["generated"]:
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
