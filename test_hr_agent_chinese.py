#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HR聊天机器人对中文问题的回答能力
"""

from hr_agent_backend_local import get_response

# 中文测试问题
test_questions = [
    "年假政策是什么？",
    "我有多少天病假？",
    "病假需要什么证明？",
    "加班怎么计算？",
    "试用期员工有什么限制？",
    "如何申请休假？"
]

# 测试每个问题
for i, question in enumerate(test_questions):
    print(f"\n===== 测试问题 {i+1} =====")
    print(f"问题: {question}")
    try:
        response = get_response(question)
        print(f"回答: {response}")
    except Exception as e:
        print(f"错误: {e}")

print("\n测试完成！")