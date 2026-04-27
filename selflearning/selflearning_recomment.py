"""
推荐智能体架构设计
- 用户偏好学习模块：负责从用户行为中学习偏好
- 上下文感知模块：考虑时间、地点、设备等上下文信息
- LLM推理模块：利用大语言模型进行高级推理
- 记忆存储模块：存储用户历史和偏好
- 反思与迭代模块：持续优化推荐策略
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_openai import ChatOpenAI
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    timestamp: datetime
    location: str
    device: str
    session_id: str

@dataclass
class UserPreference:
    """用户偏好信息"""
    categories: List[str]
    keywords: List[str]
    rating_weights: Dict[str, float]  # 不同类型的权重
    temporal_preferences: Dict[str, float]  # 时间偏好
    interaction_history: List[Dict]

@dataclass
class Recommendation:
    """推荐结果"""
    item_id: str
    score: float
    reason: str
    context_relevance: float
    explanation: str

class MemoryBuffer:
    """记忆缓冲区，用于存储短期记忆"""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, item: Any):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)
    
    def get_recent(self, n: int = 10) -> List[Any]:
        return self.buffer[-n:]

class LongTermMemory:
    """长期记忆存储"""
    def __init__(self):
        self.user_profiles = {}
        self.item_features = {}
        self.interaction_history = {}
    
    def update_user_profile(self, user_id: str, profile: UserPreference):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
        self.user_profiles[user_id].append(profile)
    
    def get_user_profile(self, user_id: str) -> Optional[UserPreference]:
        if user_id in self.user_profiles:
            return self.user_profiles[user_id][-1]  # 返回最新的profile
        return None
    
    def update_interaction(self, user_id: str, item_id: str, interaction_data: Dict):
        key = f"{user_id}_{item_id}"
        self.interaction_history[key] = {
            "timestamp": datetime.now(),
            "data": interaction_data
        }

class UserPreferenceLearner:
    """用户偏好学习器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.preference_threshold = 0.3
    
    def learn_from_interactions(self, interactions: List[Dict]) -> UserPreference:
        """从用户交互中学习偏好"""
        if not interactions:
            return UserPreference(
                categories=[],
                keywords=[],
                rating_weights={},
                temporal_preferences={},
                interaction_history=[]
            )
        
        # 分析交互类型和频率
        category_counts = {}
        keyword_texts = []
        ratings = []
        
        for interaction in interactions:
            # 收集分类信息
            if 'category' in interaction:
                cat = interaction['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # 收集文本信息用于关键词提取
            if 'title' in interaction or 'description' in interaction:
                text = f"{interaction.get('title', '')} {interaction.get('description', '')}"
                keyword_texts.append(text)
            
            # 收集评分信息
            if 'rating' in interaction:
                ratings.append(interaction['rating'])
        
        # 提取关键词
        keywords = []
        if keyword_texts:
            tfidf_matrix = self.vectorizer.fit_transform(keyword_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-20:]  # 取前20个关键词
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > self.preference_threshold]
        
        # 计算评分权重
        rating_weights = {}
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            rating_weights = {'avg_rating': avg_rating, 'count': len(ratings)}
        
        # 分析时间偏好
        temporal_prefs = self._analyze_temporal_preferences(interactions)
        
        return UserPreference(
            categories=list(category_counts.keys()),
            keywords=keywords,
            rating_weights=rating_weights,
            temporal_preferences=temporal_prefs,
            interaction_history=interactions
        )
    
    def _analyze_temporal_preferences(self, interactions: List[Dict]) -> Dict[str, float]:
        """分析时间偏好"""
        hour_counts = {}
        day_counts = {}
        
        for interaction in interactions:
            if 'timestamp' in interaction:
                dt = datetime.fromisoformat(interaction['timestamp']) if isinstance(interaction['timestamp'], str) else interaction['timestamp']
                hour = dt.hour
                day = dt.weekday()  # 0=Monday, 6=Sunday
                
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                day_counts[day] = day_counts.get(day, 0) + 1
        
        # 归一化
        total_hours = sum(hour_counts.values()) if hour_counts else 1
        total_days = sum(day_counts.values()) if day_counts else 1
        
        hour_prefs = {k: v/total_hours for k, v in hour_counts.items()}
        day_prefs = {k: v/total_days for k, v in day_counts.items()}
        
        return {
            'hourly_preferences': hour_prefs,
            'daily_preferences': day_prefs
        }
    
    def update_preference_with_feedback(self, current_pref: UserPreference, new_feedback: Dict) -> UserPreference:
        """根据新反馈更新偏好"""
        # 将新反馈添加到历史记录
        updated_history = current_pref.interaction_history + [new_feedback]
        
        # 重新学习整个偏好
        return self.learn_from_interactions(updated_history)

class ContextAnalyzer:
    """上下文分析器"""
    
    def analyze_context_relevance(self, item: Dict, context: UserContext, user_pref: UserPreference) -> float:
        """分析项目与上下文的相关性"""
        relevance_score = 0.0
        max_score = 4.0  # 最大可能分数
        
        # 1. 时间相关性
        if hasattr(context, 'timestamp') and 'temporal_preferences' in user_pref.temporal_preferences:
            hour = context.timestamp.hour
            hourly_prefs = user_pref.temporal_preferences.get('hourly_preferences', {})
            if hour in hourly_prefs:
                relevance_score += hourly_prefs[hour] * 1.0
        
        # 2. 地理位置相关性 (简化处理)
        if hasattr(context, 'location'):
            # 这里可以根据地理位置做更复杂的匹配
            relevance_score += 0.5  # 基础分
        
        # 3. 设备相关性
        if hasattr(context, 'device'):
            # 根据设备类型调整权重
            if context.device.lower() in ['mobile', 'phone']:
                if item.get('format') == 'short':
                    relevance_score += 0.8
            elif context.device.lower() in ['desktop', 'tablet']:
                if item.get('format') in ['long', 'detailed']:
                    relevance_score += 0.8
        
        # 4. 类别匹配
        if 'category' in item and item['category'] in user_pref.categories:
            relevance_score += 1.0
        
        # 5. 关键词匹配
        item_text = f"{item.get('title', '')} {item.get('description', '')} {item.get('tags', '')}"
        matching_keywords = [kw for kw in user_pref.keywords if kw.lower() in item_text.lower()]
        relevance_score += min(len(matching_keywords) * 0.2, 0.7)  # 最多加0.7分
        
        return min(relevance_score / max_score, 1.0)  # 归一化到0-1

class LLMReasoner:
    """LLM推理器"""
    
    def __init__(self, api_key: str = None):
        # 初始化本地大模型
        self.llm = ChatOpenAI(
            openai_api_key="ollama", 
            openai_api_base="http://localhost:11434/v1",
            model_name="qwen3.5:9b", 
            temperature=0.0,
            timeout=30000
        )
        self.system_prompt = """
        你是一个专业的推荐系统助手，负责为用户提供个性化推荐并解释推荐原因。
        请根据用户的历史偏好、当前上下文和待推荐项目，给出推荐理由和解释。
        回答格式必须是JSON格式：
        {
            "reason": "推荐理由",
            "explanation": "详细解释为什么推荐这个项目",
            "confidence": 0.0-1.0的置信度分数
        }
        """
    
    async def generate_explanation(self, user_pref: UserPreference, item: Dict, context: UserContext) -> Dict:
        """生成推荐解释"""
        user_history_str = ", ".join([f"{item.get('title', 'unknown')}" for item in user_pref.interaction_history[-5:]])
        categories_str = ", ".join(user_pref.categories[:5])
        keywords_str = ", ".join(user_pref.keywords[:10])
        
        prompt = f"""
        {self.system_prompt}
        
        用户历史偏好: {user_history_str}
        用户主要兴趣类别: {categories_str}
        用户关键词偏好: {keywords_str}
        当前时间: {context.timestamp}
        当前地点: {context.location}
        当前设备: {context.device}
        
        待推荐项目: 
        - 标题: {item.get('title', '')}
        - 描述: {item.get('description', '')}
        - 类别: {item.get('category', '')}
        - 格式: {item.get('format', '')}
        
        请按照要求的JSON格式回复:
        """
        
        try:
            # 使用本地大模型进行推理
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            logger.info(f"LLM响应: {response_text}")
            
            # 解析JSON响应
            import json
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {response_text}")
            # 尝试提取JSON部分
            try:
                # 查找JSON的开始和结束位置
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except Exception as e2:
                logger.error(f"提取JSON失败: {e2}")
            # 回退到默认响应
            return {
                "reason": "基于您的历史偏好",
                "explanation": "该项目与您过去喜欢的项目类似",
                "confidence": 0.7
            }
        except Exception as e:
            logger.error(f"LLM推理失败: {e}")
            return {
                "reason": "基于您的历史偏好",
                "explanation": "该项目与您过去喜欢的项目类似",
                "confidence": 0.7
            }

class ItemSimilarityCalculator:
    """项目相似度计算器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def calculate_similarity(self, items: List[Dict], target_item: Dict) -> List[float]:
        """计算项目与目标项目的相似度"""
        if not items:
            return []
        
        # 准备文本数据
        all_texts = []
        target_idx = -1
        
        for i, item in enumerate(items):
            text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')} {' '.join(item.get('tags', []))}"
            all_texts.append(text)
            
            if item['item_id'] == target_item['item_id']:
                target_idx = i
        
        # 如果目标项目不在列表中，单独处理
        target_text = f"{target_item.get('title', '')} {target_item.get('description', '')} {target_item.get('category', '')} {' '.join(target_item.get('tags', []))}"
        
        if target_idx != -1:
            # 目标项目在列表中
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            target_vector = tfidf_matrix[target_idx:target_idx+1]
            similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
            return similarities.tolist()
        else:
            # 目标项目不在列表中，需要合并计算
            all_texts.append(target_text)
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            target_vector = tfidf_matrix[-1:]
            similarities = cosine_similarity(target_vector, tfidf_matrix[:-1]).flatten()
            return similarities.tolist()

class RecommendationAgent:
    """推荐智能体主类"""
    
    def __init__(self, llm_api_key: str = None):
        self.preference_learner = UserPreferenceLearner()
        self.context_analyzer = ContextAnalyzer()
        self.llm_reasoner = LLMReasoner(api_key=llm_api_key)
        self.similarity_calculator = ItemSimilarityCalculator()
        self.memory = LongTermMemory()
        self.feedback_buffer = MemoryBuffer(capacity=50)
        
    async def get_recommendations(self, user_id: str, context: UserContext, 
                                 candidate_items: List[Dict], n_recommendations: int = 10) -> List[Recommendation]:
        """获取推荐结果"""
        logger.info(f"为用户 {user_id} 生成推荐")
        
        # 1. 获取用户历史偏好
        user_pref = self.memory.get_user_profile(user_id)
        if not user_pref:
            # 如果没有历史偏好，创建默认偏好
            user_pref = UserPreference(
                categories=[],
                keywords=[],
                rating_weights={},
                temporal_preferences={},
                interaction_history=[]
            )
        
        # 2. 为每个候选项目计算推荐分数
        recommendations = []
        
        for item in candidate_items:
            # 计算上下文相关性
            context_relevance = self.context_analyzer.analyze_context_relevance(item, context, user_pref)
            
            # 使用LLM生成解释
            llm_result = await self.llm_reasoner.generate_explanation(user_pref, item, context)
            
            # 综合评分（这里可以调整不同因素的权重）
            score = (
                context_relevance * 0.4 +  # 上下文相关性占40%
                llm_result['confidence'] * 0.4 +  # LLM置信度占40%
                np.random.random() * 0.2  # 随机因素防止过度拟合，占20%
            )
            
            recommendation = Recommendation(
                item_id=item['item_id'],
                score=score,
                reason=llm_result['reason'],
                context_relevance=context_relevance,
                explanation=llm_result['explanation']
            )
            
            recommendations.append(recommendation)
        
        # 3. 按分数排序并返回top-n
        sorted_recommendations = sorted(recommendations, key=lambda x: x.score, reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def update_user_preference(self, user_id: str, interaction_data: Dict):
        """更新用户偏好"""
        # 获取当前用户历史
        current_pref = self.memory.get_user_profile(user_id)
        if current_pref:
            # 更新现有偏好
            updated_pref = self.preference_learner.update_preference_with_feedback(current_pref, interaction_data)
        else:
            # 创建新偏好
            updated_pref = self.preference_learner.learn_from_interactions([interaction_data])
        
        # 存储到长期记忆
        self.memory.update_user_profile(user_id, updated_pref)
        
        # 存储交互历史
        self.memory.update_interaction(user_id, interaction_data.get('item_id'), interaction_data)
        
        logger.info(f"更新用户 {user_id} 的偏好")
    
    async def process_feedback(self, user_id: str, item_id: str, feedback: Dict):
        """处理用户反馈"""
        # 将反馈添加到缓冲区
        feedback_record = {
            'user_id': user_id,
            'item_id': item_id,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_buffer.add(feedback_record)
        
        # 根据反馈更新用户偏好
        interaction_data = {
            'item_id': item_id,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat(),
            **feedback  # 展开反馈中的其他字段
        }
        
        self.update_user_preference(user_id, interaction_data)
        
        logger.info(f"处理用户 {user_id} 对项目 {item_id} 的反馈")

# 示例使用代码
async def main():
    """示例主函数"""
    # 初始化推荐智能体
    agent = RecommendationAgent(llm_api_key="your-openai-api-key")
    
    # 模拟用户上下文
    context = UserContext(
        user_id="user_123",
        timestamp=datetime.now(),
        location="Beijing",
        device="mobile",
        session_id="session_abc"
    )
    
    # 模拟候选项目
    candidate_items = [
        {
            "item_id": "item_001",
            "title": "Python机器学习实战",
            "description": "深入学习Python机器学习的实用指南",
            "category": "technology",
            "tags": ["python", "machine learning", "programming"],
            "format": "book"
        },
        {
            "item_id": "item_002", 
            "title": "数据科学入门",
            "description": "从零开始学习数据科学",
            "category": "data science",
            "tags": ["data science", "statistics", "analysis"],
            "format": "course"
        },
        {
            "item_id": "item_003",
            "title": "人工智能发展趋势",
            "description": "探索AI技术的未来发展",
            "category": "technology",
            "tags": ["ai", "future", "innovation"],
            "format": "article"
        },
        {
            "item_id": "item_004",
            "title": "深度学习入门",
            "description": "从基础到应用的深度学习教程",
            "category": "technology",
            "tags": ["deep learning", "neural networks", "ai"],
            "format": "course"
        },
        {
            "item_id": "item_005",
            "title": "数据分析实战",
            "description": "使用Python进行数据分析的实战指南",
            "category": "data science",
            "tags": ["data analysis", "python", "pandas"],
            "format": "book"
        }
    ]
    
    # 1. 初始化用户偏好 - 添加历史交互数据
    print("=== 初始化用户偏好 ===")
    # 添加一些历史交互数据
    historical_interactions = [
        {
            "item_id": "hist_001",
            "title": "Python编程基础",
            "description": "Python语言入门教程",
            "category": "technology",
            "rating": 5,
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat()
        },
        {
            "item_id": "hist_002",
            "title": "机器学习入门",
            "description": "机器学习基础概念和算法",
            "category": "technology",
            "rating": 4,
            "timestamp": (datetime.now() - timedelta(days=7)).isoformat()
        },
        {
            "item_id": "hist_003",
            "title": "数据分析工具",
            "description": "数据科学常用工具介绍",
            "category": "data science",
            "rating": 4,
            "timestamp": (datetime.now() - timedelta(days=3)).isoformat()
        }
    ]
    
    # 为用户添加历史交互
    for interaction in historical_interactions:
        agent.update_user_preference("user_123", interaction)
    print("用户偏好初始化完成，添加了3条历史交互记录")
    print()
    
    # 2. 首次推荐
    print("=== 首次推荐结果 ===")
    recommendations = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()
    
    # 3. 模拟用户反馈
    print("=== 处理用户反馈 ===")
    await agent.process_feedback(
        user_id="user_123",
        item_id="item_001", 
        feedback={
            "rating": 5,
            "liked": True,
            "category": "technology",
            "title": "Python机器学习实战",
            "description": "深入学习Python机器学习的实用指南",
            "tags": ["python", "machine learning", "programming"]
        }
    )
    print("用户反馈处理完成")
    print()
    
    # 4. 第二次推荐（学习后）
    print("=== 学习后的推荐结果 ===")
    recommendations_after_learning = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations_after_learning:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()
    
    # 5. 再次模拟用户反馈 - 技能升级
    print("=== 处理第二次用户反馈（技能升级）===")
    await agent.process_feedback(
        user_id="user_123",
        item_id="item_004", 
        feedback={
            "rating": 5,
            "liked": True,
            "category": "technology",
            "title": "深度学习入门",
            "description": "从基础到应用的深度学习教程",
            "tags": ["deep learning", "neural networks", "ai"]
        }
    )
    print("用户反馈处理完成")
    print()
    
    # 6. 第三次推荐（进一步学习后）
    print("=== 进一步学习后的推荐结果 ===")
    recommendations_after_more_learning = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations_after_more_learning:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()
    
    # 7. 模拟用户兴趣变更 - 新增候选项目
    print("=== 模拟用户兴趣变更 ===")
    # 添加新的候选项目，涉及不同类别
    new_candidate_items = [
        {
            "item_id": "item_001",
            "title": "Python机器学习实战",
            "description": "深入学习Python机器学习的实用指南",
            "category": "technology",
            "tags": ["python", "machine learning", "programming"],
            "format": "book"
        },
        {
            "item_id": "item_004",
            "title": "深度学习入门",
            "description": "从基础到应用的深度学习教程",
            "category": "technology",
            "tags": ["deep learning", "neural networks", "ai"],
            "format": "course"
        },
        {
            "item_id": "item_006",
            "title": "区块链技术原理",
            "description": "区块链技术的基础原理和应用场景",
            "category": "blockchain",
            "tags": ["blockchain", "cryptocurrency", "distributed systems"],
            "format": "article"
        },
        {
            "item_id": "item_007",
            "title": "Web3.0开发实战",
            "description": "基于区块链的Web3.0应用开发",
            "category": "blockchain",
            "tags": ["web3", "blockchain", "dapp"],
            "format": "course"
        },
        {
            "item_id": "item_008",
            "title": "量子计算入门",
            "description": "量子计算的基本概念和应用",
            "category": "quantum",
            "tags": ["quantum computing", "physics", "algorithms"],
            "format": "book"
        }
    ]
    
    # 8. 第四次推荐（兴趣变更前）
    print("=== 兴趣变更前的推荐结果 ===")
    recommendations_before_interest_change = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=new_candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations_before_interest_change:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()
    
    # 9. 模拟用户兴趣变更反馈
    print("=== 处理兴趣变更反馈 ===")
    await agent.process_feedback(
        user_id="user_123",
        item_id="item_006", 
        feedback={
            "rating": 5,
            "liked": True,
            "category": "blockchain",
            "title": "区块链技术原理",
            "description": "区块链技术的基础原理和应用场景",
            "tags": ["blockchain", "cryptocurrency", "distributed systems"]
        }
    )
    print("用户反馈处理完成")
    print()
    
    # 10. 第五次推荐（兴趣变更后）
    print("=== 兴趣变更后的推荐结果 ===")
    recommendations_after_interest_change = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=new_candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations_after_interest_change:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()
    
    # 11. 模拟用户对新兴趣的进一步反馈
    print("=== 处理新兴趣的进一步反馈 ===")
    await agent.process_feedback(
        user_id="user_123",
        item_id="item_007", 
        feedback={
            "rating": 5,
            "liked": True,
            "category": "blockchain",
            "title": "Web3.0开发实战",
            "description": "基于区块链的Web3.0应用开发",
            "tags": ["web3", "blockchain", "dapp"]
        }
    )
    print("用户反馈处理完成")
    print()
    
    # 12. 第六次推荐（新兴趣巩固后）
    print("=== 新兴趣巩固后的推荐结果 ===")
    recommendations_after_interest_consolidation = await agent.get_recommendations(
        user_id="user_123",
        context=context, 
        candidate_items=new_candidate_items,
        n_recommendations=3
    )
    
    print("推荐结果:")
    for rec in recommendations_after_interest_consolidation:
        print(f"- 项目: {rec.item_id} - {rec.reason}")
        print(f"  分数: {rec.score:.3f}")
        print(f"  解释: {rec.explanation}")
        print(f"  上下文相关性: {rec.context_relevance:.3f}")
        print()

if __name__ == "__main__":
    asyncio.run(main())