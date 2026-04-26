#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HR聊天机器人对英文问题的回答能力
"""

from hr_agent_backend_local import get_response

# 英文测试问题
test_questions = [
    "What is the vacation leave policy?",
    "How many days of sick leave am I entitled to?",
    "What documentation is required for sick leave?",
    "How is overtime calculated?",
    "What restrictions apply to employees on probation?",
    "How do I apply for leave?"
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