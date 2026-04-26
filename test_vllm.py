import torch
import streamlit as st
import gradio as gr
from vllm import LLM, SamplingParams
import os

# 初始化vLLM模型
def init_model():
    """初始化量化的facebook/opt-125m模型"""
    try:
        # 使用量化配置初始化模型 - 针对CPU优化
        import os
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"  # 设置CPU缓存空间
        
        model = LLM(
            model="facebook/opt-125m",
            dtype="float16",        # 使用float16进行量化
            tensor_parallel_size=1, # 如果有多个GPU，可以增加这个值
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True,    # 强制使用eager模式，避免CPU上的编译问题
            disable_custom_all_reduce=True,  # 禁用自定义all-reduce，CPU不需要
        )
        return model
    except Exception as e:
        st.error(f"模型初始化失败: {str(e)}")
        st.error("请确保已正确安装vLLM CPU版本")
        return None

# 文本生成函数
def generate_text(model, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    """使用模型生成文本"""
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # 生成文本
    outputs = model.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text

# 问答函数
def answer_question(model, question, max_tokens=128, temperature=0.7, top_p=0.9):
    """使用模型回答问题"""
    # 构建提示
    prompt = f"问题: {question}\n回答:"
    
    # 生成回答
    answer = generate_text(model, prompt, max_tokens, temperature, top_p)
    
    return answer

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
    
    if model is None:
        st.error("无法加载模型，请检查控制台错误信息")
        return
    
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

# 主函数
def main():
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="vLLM文本生成与问答")
    parser.add_argument("--app", type=str, default="streamlit", choices=["streamlit", "gradio"], help="选择应用类型")
    args = parser.parse_args()
    
    print("注意：请使用以下命令运行此应用：")
    print("  Streamlit: streamlit run test_vllm.py")
    print("  Gradio: python test_vllm.py --app gradio")
    
    # 根据参数选择应用
    if args.app == "streamlit":
        streamlit_app()
    else:
        app = gradio_app()
        app.launch()

if __name__ == "__main__":
    main()
    
'''
(vllm) shhaofu@shhaofudeMacBook-Pro autonomous-hr-chatbot % python test_vllm.py
INFO 08-13 09:54:53 [__init__.py:235] Automatically detected platform cpu.
WARNING 08-13 09:54:54 [_custom_ops.py:20] Failed to import from vllm._C with ImportError("dlopen(/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/_C.abi3.so, 0x0002): symbol not found in flat namespace '__Z14int8_scaled_mmRN2at6TensorERKS0_S3_S3_S3_RKNSt3__18optionalIS0_EE'")
2025-08-13 09:54:54.623 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.624 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.629 
Warning: the config option 'server.enableCORS=false' is not compatible with
'server.enableXsrfProtection=true'.
As a result, 'server.enableCORS' is being overridden to 'true'.

More information:
In order to protect against CSRF attacks, we send a cookie with each request.
To do so, we must specify allowable origins, which places a restriction on
cross-origin resource sharing.

If cross origin resource sharing is required, please disable server.enableXsrfProtection.
            
2025-08-13 09:54:54.631 
  Warning: to view this Streamlit app on a browser, run it with the following
  command:

    streamlit run test_vllm.py [ARGUMENTS]
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.631 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Session state does not function when running a script without `streamlit run`
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:54.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:55.137 Thread 'Thread-1': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:55.137 Thread 'Thread-1': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:54:55.137 Thread 'Thread-1': missing ScriptRunContext! This warning can be ignored when running in bare mode.
config.json: 651B [00:00, 706kB/s]
INFO 08-13 09:55:01 [config.py:1604] Using max model len 2048
WARNING 08-13 09:55:01 [cpu.py:113] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
INFO 08-13 09:55:01 [arg_utils.py:1030] Chunked prefill is not supported for ARM and POWER CPUs; disabling it for V1 backend.
tokenizer_config.json: 685B [00:00, 3.07MB/s]
vocab.json: 899kB [00:00, 2.69MB/s]
merges.txt: 456kB [00:00, 566kB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████| 441/441 [00:00<00:00, 3.00MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████| 137/137 [00:00<00:00, 1.00MB/s]
INFO 08-13 09:55:12 [__init__.py:235] Automatically detected platform cpu.
WARNING 08-13 09:55:12 [_custom_ops.py:20] Failed to import from vllm._C with ImportError("dlopen(/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/_C.abi3.so, 0x0002): symbol not found in flat namespace '__Z14int8_scaled_mmRN2at6TensorERKS0_S3_S3_S3_RKNSt3__18optionalIS0_EE'")
INFO 08-13 09:55:12 [core.py:572] Waiting for init message from front-end.
INFO 08-13 09:55:12 [core.py:71] Initializing a V1 LLM engine (v0.10.0) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cpu, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=facebook/opt-125m, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=False, pooler_config=None, compilation_config={"level":2,"debug_dump_path":"","cache_dir":"","backend":"inductor","custom_ops":["none"],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"dce":true,"size_asserts":false,"nan_asserts":false,"epilogue_fusion":true},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
INFO 08-13 09:55:13 [importing.py:63] Triton not installed or not compatible; certain GPU-related functions will not be available.
ERROR 08-13 09:55:15 [core.py:632] EngineCore failed to start.
ERROR 08-13 09:55:15 [core.py:632] Traceback (most recent call last):
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 623, in run_engine_core
ERROR 08-13 09:55:15 [core.py:632]     engine_core = EngineCoreProc(*args, **kwargs)
ERROR 08-13 09:55:15 [core.py:632]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 441, in __init__
ERROR 08-13 09:55:15 [core.py:632]     super().__init__(vllm_config, executor_class, log_stats,
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 77, in __init__
ERROR 08-13 09:55:15 [core.py:632]     self.model_executor = executor_class(vllm_config)
ERROR 08-13 09:55:15 [core.py:632]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/executor_base.py", line 53, in __init__
ERROR 08-13 09:55:15 [core.py:632]     self._init_executor()
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
ERROR 08-13 09:55:15 [core.py:632]     self.collective_rpc("init_device")
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
ERROR 08-13 09:55:15 [core.py:632]     answer = run_method(self.driver_worker, method, args, kwargs)
ERROR 08-13 09:55:15 [core.py:632]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/utils/__init__.py", line 2985, in run_method
ERROR 08-13 09:55:15 [core.py:632]     return func(*args, **kwargs)
ERROR 08-13 09:55:15 [core.py:632]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/worker/worker_base.py", line 603, in init_device
ERROR 08-13 09:55:15 [core.py:632]     self.worker.init_device()  # type: ignore
ERROR 08-13 09:55:15 [core.py:632]     ^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/worker/cpu_worker.py", line 60, in init_device
ERROR 08-13 09:55:15 [core.py:632]     ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
ERROR 08-13 09:55:15 [core.py:632]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 08-13 09:55:15 [core.py:632]   File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/_ops.py", line 1267, in __getattr__
ERROR 08-13 09:55:15 [core.py:632]     raise AttributeError(
ERROR 08-13 09:55:15 [core.py:632] AttributeError: '_OpNamespace' '_C_utils' object has no attribute 'init_cpu_threads_env'
Process EngineCore_0:
Traceback (most recent call last):
  File "/opt/anaconda3/envs/vllm/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/opt/anaconda3/envs/vllm/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 636, in run_engine_core
    raise e
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 623, in run_engine_core
    engine_core = EngineCoreProc(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 441, in __init__
    super().__init__(vllm_config, executor_class, log_stats,
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 77, in __init__
    self.model_executor = executor_class(vllm_config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/executor_base.py", line 53, in __init__
    self._init_executor()
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
    self.collective_rpc("init_device")
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
    answer = run_method(self.driver_worker, method, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/utils/__init__.py", line 2985, in run_method
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/worker/worker_base.py", line 603, in init_device
    self.worker.init_device()  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/worker/cpu_worker.py", line 60, in init_device
    ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/torch/_ops.py", line 1267, in __getattr__
    raise AttributeError(
AttributeError: '_OpNamespace' '_C_utils' object has no attribute 'init_cpu_threads_env'
2025-08-13 09:55:15.957 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:55:15.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2025-08-13 09:55:15.958 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
Traceback (most recent call last):
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/test_vllm.py", line 163, in <module>
    main()
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/test_vllm.py", line 157, in main
    streamlit_app()
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/test_vllm.py", line 65, in streamlit_app
    st.session_state.model = init_model()
                             ^^^^^^^^^^^^
  File "/Users/shhaofu/Code/Codes/autonomous-hr-chatbot/test_vllm.py", line 11, in init_model
    model = LLM(
            ^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/entrypoints/llm.py", line 273, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 497, in from_engine_args
    return engine_cls.from_vllm_config(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 126, in from_vllm_config
    return cls(vllm_config=vllm_config,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 103, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 77, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 514, in __init__
    super().__init__(
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 408, in __init__
    with launch_core_engines(vllm_config, executor_class,
  File "/opt/anaconda3/envs/vllm/lib/python3.11/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 697, in launch_core_engines
    wait_for_engine_startup(
  File "/opt/anaconda3/envs/vllm/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 750, in wait_for_engine_startup
    raise RuntimeError("Engine core initialization failed. "
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_0': 1}
'''

'''
(vllm) shhaofu@shhaofudeMacBook-Pro cursor-projects % python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

PyTorch: 2.7.0, CUDA: None
(vllm) shhaofu@shhaofudeMacBook-Pro cursor-projects % python -c "from vllm import __version__; print(f'vLLM: {__version__}')"

vLLM: 0.10.0
'''


