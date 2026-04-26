#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR政策初始化脚本
将hr_policy_zh.txt的内容智能分割并加载到FAISS向量数据库中
"""

import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import requests
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ollama_embed_query(text: str) -> List[float]:
    """
    使用Ollama API生成文本嵌入向量
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text:latest", "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception as e:
        print(f"嵌入生成失败: {e}")
        # 返回一个默认的零向量（384维，根据nomic-embed-text模型）
        return [0.0] * 384

class OllamaEmbeddings:
    """
    Ollama嵌入包装类，提供LangChain期望的接口
    """
    def __init__(self, model: str = "nomic-embed-text:latest"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档嵌入向量
        """
        embeddings = []
        for text in texts:
            embedding = ollama_embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成查询嵌入向量
        """
        return ollama_embed_query(text)

def smart_split_policy_text(text: str, is_english: bool = False) -> List[Dict[str, Any]]:
    """
    智能分割HR政策文本，保持逻辑结构
    """
    # 按主要章节分割
    sections = []
    
    if is_english:
        # 处理英文政策文本
        # 分割休假政策部分
        vacation_policy_match = re.search(r'HR Policy Manual - Leave Policy(.*?)(?=HR Policy Manual - Attendance Policy|$)', text, re.DOTALL)
        if not vacation_policy_match:  # 如果没有找到标题，假设整个文档是休假政策
            vacation_text = text
            sections.extend(split_english_vacation_policy(vacation_text))
        else:
            vacation_text = vacation_policy_match.group(1).strip()
            sections.extend(split_english_vacation_policy(vacation_text))
        
        # 分割考勤政策部分
        attendance_policy_match = re.search(r'HR Policy Manual - Attendance Policy(.*?)$', text, re.DOTALL)
        if attendance_policy_match:
            attendance_text = attendance_policy_match.group(1).strip()
            sections.extend(split_english_attendance_policy(attendance_text))
    else:
        # 处理中文政策文本
        # 分割休假政策部分
        vacation_policy_match = re.search(r'HR政策手册 - 休假政策(.*?)(?=HR政策手册 - 考勤政策|$)', text, re.DOTALL)
        if vacation_policy_match:
            vacation_text = vacation_policy_match.group(1).strip()
            sections.extend(split_vacation_policy(vacation_text))
        
        # 分割考勤政策部分
        attendance_policy_match = re.search(r'HR政策手册 - 考勤政策(.*?)$', text, re.DOTALL)
        if attendance_policy_match:
            attendance_text = attendance_policy_match.group(1).strip()
            sections.extend(split_attendance_policy(attendance_text))
    
    return sections

def split_vacation_policy(text: str) -> List[Dict[str, Any]]:
    """
    分割中文休假政策文本
    """
    sections = []
    
    # 添加总体政策说明
    sections.append({
        "content": "休假政策概述：本政策涵盖年假、病假、服务奖励假、陪产假、产假、反暴力侵害妇女假、单亲父母假、丧假、陪审/公民义务假、学习/考试假、慰问假、学术休假、婚假、领养假、兵役假、志愿服务假、家庭照顾假、个人事假、宗教节日假、医疗预约假、选举假、无薪假等各类休假规定。",
        "metadata": {"source": "vacation_policy", "type": "overview", "category": "休假政策"}
    })
    
    # 按字母分割各个休假类型
    letter_pattern = r'([A-Z]\.\s+[^A-Z]+?)(?=[A-Z]\.\s+|$)'
    letter_matches = re.findall(letter_pattern, text, re.DOTALL)
    
    for match in letter_matches:
        if match.strip():
            # 提取休假类型名称
            type_match = re.search(r'([A-Z]\.\s+)(.+?)(?=\n|$)', match)
            if type_match:
                type_name = type_match.group(2).strip()
                sections.append({
                    "content": f"{type_name}：{match.strip()}",
                    "metadata": {"source": "vacation_policy", "type": "leave_type", "category": type_name}
                })
    
    # 添加通用规定
    general_patterns = [
        (r'IV\.\s+公共假期(.*?)(?=V\.|$)', "公共假期规定"),
        (r'V\.\s+试用期休假(.*?)(?=VI\.|$)', "试用期休假规定"),
        (r'VI\.\s+休假折现(.*?)(?=VII\.|$)', "休假折现规定"),
        (r'VII\.\s+休假申请流程(.*?)(?=VIII\.|$)', "休假申请流程"),
        (r'VIII\.\s+违规处理(.*?)(?=IX\.|$)', "违规处理规定"),
        (r'IX\.\s+政策修订(.*?)(?=X\.|$)', "政策修订规定"),
        (r'X\.\s+疑问咨询(.*?)(?=XI\.|$)', "疑问咨询方式"),
        (r'XI\.\s+确认条款(.*?)(?=$)', "确认条款")
    ]
    
    for pattern, title in general_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections.append({
                "content": f"{title}：{match.group(1).strip()}",
                "metadata": {"source": "vacation_policy", "type": "general_rule", "category": title}
            })
    
    return sections

def split_english_vacation_policy(text: str) -> List[Dict[str, Any]]:
    """
    分割英文休假政策文本
    """
    sections = []
    
    # 添加总体政策说明
    sections.append({
        "content": "Leave Policy Overview: This policy covers various types of leave including Vacation, Sick, Service Incentive, Paternity, Maternity, Violence Against Women, Single Parent, Bereavement, Jury Duty/Civic Duty, Study/Examination, Compassionate, Sabbatical, Marriage, Adoption, Military Service, Volunteer Service, Family Care, Personal, Religious Observance, Medical Appointment, Election, and Unpaid Leave.",
        "metadata": {"source": "vacation_policy", "type": "overview", "category": "Leave Policy"}
    })
    
    # 按字母分割各个休假类型
    letter_pattern = r'([A-Z]\.\s+[^A-Z]+?)(?=[A-Z]\.\s+|$)'
    letter_matches = re.findall(letter_pattern, text, re.DOTALL)
    
    for match in letter_matches:
        if match.strip():
            # 提取休假类型名称
            type_match = re.search(r'([A-Z]\.\s+)(.+?)(?=\n|$)', match)
            if type_match:
                type_name = type_match.group(2).strip()
                sections.append({
                    "content": f"{type_name}: {match.strip()}",
                    "metadata": {"source": "vacation_policy", "type": "leave_type", "category": type_name}
                })
    
    # 添加通用规定
    general_patterns = [
        (r'IV\.\s+PUBLIC HOLIDAYS(.*?)(?=V\.|$)', "Public Holidays"),
        (r'V\.\s+LEAVE DURING PROBATION(.*?)(?=VI\.|$)', "Leave During Probation"),
        (r'VI\.\s+LEAVE ENCASHMENT(.*?)(?=VII\.|$)', "Leave Encashment"),
        (r'VII\.\s+LEAVE APPLICATION PROCESS(.*?)(?=VIII\.|$)', "Leave Application Process"),
        (r'VIII\.\s+POLICY VIOLATION(.*?)(?=IX\.|$)', "Policy Violation"),
        (r'IX\.\s+POLICY REVIEW(.*?)(?=X\.|$)', "Policy Review"),
        (r'X\.\s+QUESTIONS(.*?)(?=XI\.|$)', "Questions"),
        (r'XI\.\s+ACKNOWLEDGEMENT(.*?)(?=$)', "Acknowledgement")
    ]
    
    for pattern, title in general_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections.append({
                "content": f"{title}: {match.group(1).strip()}",
                "metadata": {"source": "vacation_policy", "type": "general_rule", "category": title}
            })
    
    return sections

def split_attendance_policy(text: str) -> List[Dict[str, Any]]:
    """
    分割中文考勤政策文本
    """
    sections = []
    
    # 添加总体政策说明
    sections.append({
        "content": "考勤政策概述：本政策明确全体员工考勤要求，规范公务外出和加班管理，包括弹性工作时间、轮班津贴、待命值班、差旅时间、紧急任务、周末工作、节假日工作、夜班工作、备勤值班、延长工作、外勤工作等规定。",
        "metadata": {"source": "attendance_policy", "type": "overview", "category": "考勤政策"}
    })
    
    # 按字母分割各个考勤类型
    letter_pattern = r'([A-Z]\.\s+[^A-Z]+?)(?=[A-Z]\.\s+|$)'
    letter_matches = re.findall(letter_pattern, text, re.DOTALL)
    
    for match in letter_matches:
        if match.strip():
            # 提取考勤类型名称
            type_match = re.search(r'([A-Z]\.\s+)(.+?)(?=\n|$)', match)
            if type_match:
                type_name = type_match.group(2).strip()
                sections.append({
                    "content": f"{type_name}：{match.strip()}",
                    "metadata": {"source": "attendance_policy", "type": "attendance_type", "category": type_name}
                })
    
    # 添加通用规定
    general_patterns = [
        (r'IV\.\s+公共假期(.*?)(?=V\.|$)', "公共假期规定"),
        (r'V\.\s+试用期考勤(.*?)(?=VI\.|$)', "试用期考勤规定"),
        (r'VI\.\s+考勤申请流程(.*?)(?=VII\.|$)', "考勤申请流程"),
        (r'VII\.\s+违规处理(.*?)(?=VIII\.|$)', "违规处理规定"),
        (r'VIII\.\s+政策修订(.*?)(?=IX\.|$)', "政策修订规定"),
        (r'IX\.\s+疑问咨询(.*?)(?=X\.|$)', "疑问咨询方式"),
        (r'X\.\s+确认条款(.*?)(?=$)', "确认条款")
    ]
    
    for pattern, title in general_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections.append({
                "content": f"{title}：{match.group(1).strip()}",
                "metadata": {"source": "attendance_policy", "type": "general_rule", "category": title}
            })
    
    return sections

def split_english_attendance_policy(text: str) -> List[Dict[str, Any]]:
    """
    分割英文考勤政策文本
    """
    sections = []
    
    # 添加总体政策说明
    sections.append({
        "content": "Attendance Policy Overview: This policy outlines attendance requirements for all employees, regulates official business and overtime management, including flexible working hours, shift differential, on-call duty, travel time, emergency duty, weekend duty, holiday duty, night shift duty, standby duty, extended duty, field work duty, and other regulations.",
        "metadata": {"source": "attendance_policy", "type": "overview", "category": "Attendance Policy"}
    })
    
    # 按字母分割各个考勤类型
    letter_pattern = r'([A-Z]\.\s+[^A-Z]+?)(?=[A-Z]\.\s+|$)'
    letter_matches = re.findall(letter_pattern, text, re.DOTALL)
    
    for match in letter_matches:
        if match.strip():
            # 提取考勤类型名称
            type_match = re.search(r'([A-Z]\.\s+)(.+?)(?=\n|$)', match)
            if type_match:
                type_name = type_match.group(2).strip()
                sections.append({
                    "content": f"{type_name}: {match.strip()}",
                    "metadata": {"source": "attendance_policy", "type": "attendance_type", "category": type_name}
                })
    
    # 添加通用规定
    general_patterns = [
        (r'IV\.\s+PUBLIC HOLIDAYS(.*?)(?=V\.|$)', "Public Holidays"),
        (r'V\.\s+ATTENDANCE DURING PROBATION(.*?)(?=VI\.|$)', "Attendance During Probation"),
        (r'VI\.\s+ATTENDANCE APPLICATION PROCESS(.*?)(?=VII\.|$)', "Attendance Application Process"),
        (r'VII\.\s+POLICY VIOLATION(.*?)(?=VIII\.|$)', "Policy Violation"),
        (r'VIII\.\s+POLICY REVIEW(.*?)(?=IX\.|$)', "Policy Review"),
        (r'IX\.\s+QUESTIONS(.*?)(?=X\.|$)', "Questions"),
        (r'X\.\s+ACKNOWLEDGEMENT(.*?)(?=$)', "Acknowledgement")
    ]
    
    for pattern, title in general_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections.append({
                "content": f"{title}: {match.group(1).strip()}",
                "metadata": {"source": "attendance_policy", "type": "general_rule", "category": title}
            })
    
    return sections

def create_faiss_vectorstore(sections: List[Dict[str, Any]], save_path: str = "hr_policy_faiss") -> FAISS:
    """
    创建FAISS向量存储
    """
    print(f"正在处理 {len(sections)} 个政策片段...")
    
    # 创建文档对象
    documents = []
    for i, section in enumerate(sections):
        doc = Document(
            page_content=section["content"],
            metadata=section["metadata"]
        )
        documents.append(doc)
        print(f"  片段 {i+1}: {section['metadata']['category']} ({len(section['content'])} 字符)")
    
    # 使用文本分割器进一步优化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    # 分割文档
    split_docs = text_splitter.split_documents(documents)
    print(f"分割后共有 {len(split_docs)} 个文档片段")
    
    # 创建FAISS向量存储
    print("正在生成向量嵌入...")
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    
    # 保存向量存储
    vectorstore.save_local(save_path)
    print(f"FAISS向量存储已保存到: {save_path}")
    
    return vectorstore

def save_sections_info(sections: List[Dict[str, Any]], save_path: str = "hr_policy_sections.json"):
    """
    保存分割后的政策片段信息
    """
    # 转换为可序列化的格式
    serializable_sections = []
    for section in sections:
        serializable_sections.append({
            "content": section["content"],
            "metadata": section["metadata"]
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_sections, f, ensure_ascii=False, indent=2)
    
    print(f"政策片段信息已保存到: {save_path}")

def test_vectorstore(vectorstore: FAISS):
    """
    测试向量存储的检索功能
    """
    print("\n=== 测试向量存储检索功能 ===")
    
    # 中文测试查询
    zh_test_queries = [
        "年假政策是什么？",
        "病假需要什么证明？",
        "加班怎么计算？",
        "试用期员工有什么限制？",
        "如何申请休假？"
    ]
    
    # 英文测试查询
    en_test_queries = [
        "What is the vacation leave policy?",
        "What documentation is required for sick leave?",
        "How is overtime calculated?",
        "What restrictions apply to employees on probation?",
        "How do I apply for leave?"
    ]
    
    # 合并测试查询
    test_queries = zh_test_queries + en_test_queries
    
    for query in test_queries:
        print(f"\n查询/Query: {query}")
        try:
            # 生成查询的嵌入向量
            embeddings = OllamaEmbeddings()
            query_embedding = embeddings.embed_query(query)
            
            # 检索相似文档
            docs = vectorstore.similarity_search_by_vector(query_embedding, k=2)
            
            for i, doc in enumerate(docs):
                print(f"  结果/Result {i+1}: {doc.page_content[:100]}...")
                print(f"  来源/Source: {doc.metadata}")
        except Exception as e:
            print(f"  检索失败/Retrieval failed: {e}")

def main():
    """
    主函数
    """
    print("=== HR政策FAISS向量数据库初始化脚本 ===")
    
    all_sections = []
    
    # 处理中文政策文件
    zh_policy_file = Path("hr_policy_zh.txt")
    if zh_policy_file.exists():
        # 读取中文政策文件
        print("正在读取中文HR政策文件...")
        with open(zh_policy_file, 'r', encoding='utf-8') as f:
            zh_policy_text = f.read()
        
        print(f"中文文件大小: {len(zh_policy_text)} 字符")
        
        # 智能分割中文政策文本
        print("正在智能分割中文政策文本...")
        zh_sections = smart_split_policy_text(zh_policy_text, is_english=False)
        print(f"中文分割完成，共 {len(zh_sections)} 个政策片段")
        
        all_sections.extend(zh_sections)
    else:
        print("未找到中文HR政策文件 hr_policy_zh.txt")
    
    # 处理英文政策文件
    en_policy_file = Path("hr_policy.txt")
    if en_policy_file.exists():
        # 读取英文政策文件
        print("正在读取英文HR政策文件...")
        with open(en_policy_file, 'r', encoding='utf-8') as f:
            en_policy_text = f.read()
        
        print(f"英文文件大小: {len(en_policy_text)} 字符")
        
        # 智能分割英文政策文本
        print("正在智能分割英文政策文本...")
        en_sections = smart_split_policy_text(en_policy_text, is_english=True)
        print(f"英文分割完成，共 {len(en_sections)} 个政策片段")
        
        all_sections.extend(en_sections)
    else:
        print("未找到英文HR政策文件 hr_policy.txt")
    
    if not all_sections:
        print("错误: 未找到任何HR政策文件")
        return
    
    # 保存分割信息
    print(f"总共处理了 {len(all_sections)} 个政策片段")
    save_sections_info(all_sections, "hr_policy_sections.json")
    
    # 创建FAISS向量存储
    print("正在创建FAISS向量存储...")
    vectorstore = create_faiss_vectorstore(all_sections)
    
    # 测试向量存储
    test_vectorstore(vectorstore)
    
    print("\n=== 初始化完成！ ===")
    print("现在可以在 hr_agent_backend_local.py 中使用这个向量存储了")

if __name__ == "__main__":
    main()