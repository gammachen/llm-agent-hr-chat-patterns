#!/usr/bin/env python3
"""
vLLM的替代方案 - 使用Hugging Face Transformers
当vLLM出现兼容性问题时使用
"""
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class TransformersModel:
    def __init__(self, model_name="facebook/opt-125m"):
        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        try:
            print(f"正在加载模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
            self.model.to(self.device)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise e
    
    def generate(self, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
        """生成文本"""
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 只返回新生成的部分
            new_text = generated_text[len(prompt):]
            return new_text.strip()
            
        except Exception as e:
            return f"生成错误: {str(e)}"

def streamlit_app():
    """使用Transformers的Streamlit应用"""
    st.set_page_config(page_title="Transformers文本生成与问答", page_icon="🤖")
    st.title("Transformers文本生成与问答")
    
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
            try:
                st.session_state.model = TransformersModel()
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                return
    
    model = st.session_state.model
    
    # 文本生成模式
    if mode == "文本生成":
        st.subheader("文本生成")
        prompt = st.text_area("输入提示", height=150)
        
        if st.button("生成"):
            if prompt:
                with st.spinner("正在生成文本..."):
                    generated_text = model.generate(prompt, max_tokens, temperature, top_p)
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
                    prompt = f"问题: {question}\n回答:"
                    answer = model.generate(prompt, max_tokens, temperature, top_p)
                st.markdown("### 回答")
                st.write(answer)
            else:
                st.error("请输入问题")

def main():
    """主函数"""
    print("启动Transformers替代方案...")
    print("使用命令: streamlit run test_transformers_fallback.py")
    streamlit_app()

if __name__ == "__main__":
    main()