import streamlit as st
from test_vllm import init_model, generate_text, answer_question

# Streamlit Web界面
def streamlit_app():
    """使用Streamlit创建Web界面"""
    st.set_page_config(page_title="vLLM文本生成与问答", page_icon="🤖")
    st.title("vLLM文本生成与问答")
    
    # 侧边栏配置
    st.sidebar.title("模型配置")
    max_tokens = st.sidebar.slider("最大生成长度", 16, 512, 128)
    temperature = st.sidebar.slider("温度", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9)
    
    # 选择模式
    mode = st.sidebar.radio("选择模式", ["文本生成", "问答"])
    
    # 初始化模型
    if "model" not in st.session_state:
        with st.spinner("正在加载模型..."):
            st.session_state.model = init_model()
    
    model = st.session_state.model
    
    # 文本生成模式
    if mode == "文本生成":
        st.subheader("文本生成")
        prompt = st.text_area("输入提示", height=150)
        
        if st.button("生成"):
            if prompt:
                with st.spinner("正在生成文本..."):
                    generated_text = generate_text(model, prompt, max_tokens, temperature, top_p)
                st.markdown("### 生成结果")
                st.write(generated_text)
            else:
                st.error("请输入提示文本")
    
    # 问答模式
    else:
        st.subheader("问答")
        question = st.text_input("输入问题")
        
        if st.button("回答"):
            if question:
                with st.spinner("正在生成回答..."):
                    answer = answer_question(model, question, max_tokens, temperature, top_p)
                st.markdown("### 回答")
                st.write(answer)
            else:
                st.error("请输入问题")

if __name__ == "__main__":
    streamlit_app()