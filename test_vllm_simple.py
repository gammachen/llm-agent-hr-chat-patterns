#!/usr/bin/env python3
"""
简单测试vLLM是否能正常工作
"""
import os
import warnings
warnings.filterwarnings("ignore")

def test_vllm_basic():
    """测试基本的vLLM初始化"""
    try:
        from vllm import LLM, SamplingParams
        import os
        print("✓ vLLM库导入成功")
        
        # 设置环境变量
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"
        
        # 尝试初始化一个简单的模型
        print("正在初始化模型...")
        model = LLM(
            model="facebook/opt-125m",
            dtype="float16",
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        print("✓ 模型初始化成功")
        
        # 测试生成
        print("测试文本生成...")
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        outputs = model.generate(["Hello, how are you?"], sampling_params)
        print("✓ 文本生成成功")
        print(f"生成结果: {outputs[0].outputs[0].text}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请检查vLLM是否正确安装")
        return False
    except Exception as e:
        print(f"✗ 初始化错误: {e}")
        print("这可能是由于CPU架构兼容性问题")
        return False

def check_installation():
    """检查vLLM安装状态"""
    try:
        import vllm
        print(f"vLLM版本: {vllm.__version__}")
        
        # 检查环境变量
        if "VLLM_CPU_KVCACHE_SPACE" not in os.environ:
            print("⚠ 警告: VLLM_CPU_KVCACHE_SPACE未设置，使用默认值4GB")
            print("  可以通过设置环境变量来增加缓存空间:")
            print("  export VLLM_CPU_KVCACHE_SPACE=8  # 设置为8GB")
        
        return True
    except ImportError:
        print("✗ vLLM未安装")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("vLLM安装和兼容性测试")
    print("=" * 50)
    
    if check_installation():
        test_vllm_basic()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)