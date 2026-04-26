import gradio as gr
from test_vllm import init_model, generate_text, answer_question

# Gradio API界面
def gradio_app():
    """使用Gradio创建API界面"""
    # 初始化模型
    model = init_model()
    
    # 文本生成接口
    def generate_interface(prompt, max_tokens=128, temperature=0.7, top_p=0.9):
        return generate_text(model, prompt, int(max_tokens), float(temperature), float(top_p))
    
    # 问答接口
    def qa_interface(question, max_tokens=128, temperature=0.7, top_p=0.9):
        return answer_question(model, question, int(max_tokens), float(temperature), float(top_p))
    
    # 创建文本生成界面
    generate_interface_gradio = gr.Interface(
        fn=generate_interface,
        inputs=[
            gr.Textbox(lines=5, label="提示"),
            gr.Slider(16, 512, 128, label="最大生成长度"),
            gr.Slider(0.0, 1.0, 0.7, label="温度"),
            gr.Slider(0.0, 1.0, 0.9, label="Top-p"),
        ],
        outputs=gr.Textbox(label="生成结果"),
        title="vLLM文本生成",
        description="使用量化的facebook/opt-125m模型进行文本生成",
    )
    
    # 创建问答界面
    qa_interface_gradio = gr.Interface(
        fn=qa_interface,
        inputs=[
            gr.Textbox(lines=2, label="问题"),
            gr.Slider(16, 512, 128, label="最大生成长度"),
            gr.Slider(0.0, 1.0, 0.7, label="温度"),
            gr.Slider(0.0, 1.0, 0.9, label="Top-p"),
        ],
        outputs=gr.Textbox(label="回答"),
        title="vLLM问答",
        description="使用量化的facebook/opt-125m模型进行问答",
    )
    
    # 创建Gradio应用
    demo = gr.TabbedInterface(
        [generate_interface_gradio, qa_interface_gradio],
        ["文本生成", "问答"]
    )
    
    return demo

if __name__ == "__main__":
    app = gradio_app()
    app.launch()