"""
CrewAI自适应智能体系统
基于CrewAI框架实现具有自我进化能力的智能助手
使用本地Ollama部署的qwen3.5:9b模型
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import random
import logging
import re
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: str
    name: str = ""
    life_stage: str = "unknown"
    interests: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    last_updated: str = ""
    evolution_tracking: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InteractionEvent:
    """交互事件数据结构"""
    event_id: str
    user_id: str
    event_type: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False

class MemoryStorage:
    """统一记忆存储层 - 支持长期记忆、实体记忆、事件记忆"""

    def __init__(self, storage_path: str = "./data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "memory.db"
        self.long_term_memory_path = self.storage_path / "long_term_memory.json"
        self.entity_memory_path = self.storage_path / "entity_memory.json"
        self._init_storage()

    def _init_storage(self):
        """初始化存储"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                processed INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        if not self.long_term_memory_path.exists():
            with open(self.long_term_memory_path, 'w') as f:
                json.dump({}, f)
        if not self.entity_memory_path.exists():
            with open(self.entity_memory_path, 'w') as f:
                json.dump({}, f)

    def store_event(self, event: InteractionEvent):
        """存储交互事件"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO interaction_events
            (event_id, user_id, event_type, content, timestamp, metadata, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.user_id,
            event.event_type,
            event.content,
            event.timestamp,
            json.dumps(event.metadata),
            1 if event.processed else 0
        ))
        conn.commit()
        conn.close()

    def get_unprocessed_events(self, user_id: str = None) -> List[InteractionEvent]:
        """获取未处理的事件"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        if user_id:
            cursor.execute("""
                SELECT event_id, user_id, event_type, content, timestamp, metadata, processed
                FROM interaction_events
                WHERE processed = 0 AND user_id = ?
                ORDER BY timestamp ASC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT event_id, user_id, event_type, content, timestamp, metadata, processed
                FROM interaction_events
                WHERE processed = 0
                ORDER BY timestamp ASC
            """)
        rows = cursor.fetchall()
        conn.close()
        return [
            InteractionEvent(
                event_id=row[0],
                user_id=row[1],
                event_type=row[2],
                content=row[3],
                timestamp=row[4],
                metadata=json.loads(row[5]) if row[5] else {},
                processed=bool(row[6])
            ) for row in rows
        ]

    def mark_events_processed(self, event_ids: List[str]):
        """标记事件为已处理"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(event_ids))
        cursor.execute(f"""
            UPDATE interaction_events
            SET processed = 1
            WHERE event_id IN ({placeholders})
        """, event_ids)
        conn.commit()
        conn.close()

    def store_user_profile(self, profile: UserProfile):
        """存储用户画像"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles (user_id, profile_data, last_updated)
            VALUES (?, ?, ?)
        """, (profile.user_id, json.dumps(asdict(profile)), datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT profile_data FROM user_profiles WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            return UserProfile(**data)
        return None

    def update_long_term_memory(self, user_id: str, memory_type: str, content: Dict):
        """更新长期记忆"""
        with open(self.long_term_memory_path, 'r') as f:
            memories = json.load(f)
        if user_id not in memories:
            memories[user_id] = {}
        if memory_type not in memories[user_id]:
            memories[user_id][memory_type] = []
        memories[user_id][memory_type].append({
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        with open(self.long_term_memory_path, 'w') as f:
            json.dump(memories, f, indent=2)

    def get_long_term_memory(self, user_id: str, memory_type: str = None) -> Dict:
        """获取长期记忆"""
        with open(self.long_term_memory_path, 'r') as f:
            memories = json.load(f)
        if user_id not in memories:
            return {}
        if memory_type:
            return memories[user_id].get(memory_type, [])
        return memories[user_id]

    def update_entity_memory(self, user_id: str, entity_type: str, entity_data: Dict):
        """更新实体记忆"""
        with open(self.entity_memory_path, 'r') as f:
            entities = json.load(f)
        if user_id not in entities:
            entities[user_id] = {}
        entities[user_id][entity_type] = entity_data
        with open(self.entity_memory_path, 'w') as f:
            json.dump(entities, f, indent=2)

    def get_entity_memory(self, user_id: str, entity_type: str = None) -> Dict:
        """获取实体记忆"""
        with open(self.entity_memory_path, 'r') as f:
            entities = json.load(f)
        if user_id not in entities:
            return {}
        if entity_type:
            return entities[user_id].get(entity_type, {})
        return entities[user_id]

class ObserverTool(BaseTool):
    """观察者工具 - 记录用户交互事件"""
    name: str = "observer_tool"
    description: str = "用于记录用户的所有交互事件，包括对话、行为、反馈等"

    def _run(self, user_id: str, event_type: str, content: str, metadata: Dict = None) -> str:
        """记录用户事件"""
        storage = MemoryStorage()
        event = InteractionEvent(
            event_id=f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            user_id=user_id,
            event_type=event_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        storage.store_event(event)
        return f"已记录事件: {event_type} - {content[:50]}..."

class UserProfileTool(BaseTool):
    """用户画像工具 - 获取和更新用户信息"""
    name: str = "user_profile_tool"
    description: str = (
        "用于获取和更新用户画像信息。"
        "动作 'get': 不需要 profile_data 参数，返回当前用户画像 JSON。"
        "动作 'update': 必须提供 profile_data 字典，包含需要更新的字段 (如 interests, preferences, life_stage)。"
        "注意: update 操作是合并更新，不是全量替换，会保留未提供的字段。"
    )

    def _run(self, user_id: str, action: str = "get", profile_data: Dict = None) -> str:
        """获取或更新用户画像"""
        logger.info(f"[UserProfileTool] ========== 开始执行 ==========")
        logger.info(f"[UserProfileTool] 输入参数 - User: {user_id}, Action: {action}")
        
        storage = MemoryStorage()
        
        try:
            if action == "get":
                logger.debug(f"[UserProfileTool] 正在查询用户 {user_id} 的画像")
                profile = storage.get_user_profile(user_id)
                
                if profile:
                    result = json.dumps(asdict(profile), ensure_ascii=False, indent=2)
                    logger.info(f"[UserProfileTool] ✅ 成功获取用户 {user_id} 画像")
                    logger.debug(f"[UserProfileTool] 画像内容预览: {result[:200]}...")
                    return result
                else:
                    logger.warning(f"[UserProfileTool] ⚠️ 未找到用户 {user_id} 的画像")
                    return f"未找到用户 {user_id} 的画像，可能需要先初始化。"
            
            elif action == "update":
                if not profile_data:
                    logger.error(f"[UserProfileTool] ❌ 更新操作缺少 profile_data 参数")
                    return "错误: 更新操作必须提供 profile_data 参数"
                
                logger.info(f"[UserProfileTool] 正在更新用户 {user_id} 画像")
                logger.info(f"[UserProfileTool] 待更新字段: {list(profile_data.keys())}")
                logger.debug(f"[UserProfileTool] 更新数据: {json.dumps(profile_data, ensure_ascii=False)}")
                
                # 先获取现有画像以进行合并，避免覆盖未提供的字段
                existing_profile = storage.get_user_profile(user_id)
                if not existing_profile:
                    logger.info(f"[UserProfileTool] 用户 {user_id} 不存在，创建新画像")
                    existing_profile = UserProfile(user_id=user_id)
                else:
                    logger.debug(f"[UserProfileTool] 现有画像 - 兴趣: {existing_profile.interests}, 阶段: {existing_profile.life_stage}")
                
                # 记录更新前的状态
                old_interests = existing_profile.interests.copy()
                old_preferences = existing_profile.preferences.copy()
                old_life_stage = existing_profile.life_stage
                
                # 合并数据 - 智能更新策略
                if "interests" in profile_data:
                    current_interests = set(existing_profile.interests)
                    new_interests = set(profile_data["interests"])
                    merged_interests = list(current_interests | new_interests)
                    existing_profile.interests = merged_interests
                    logger.info(f"[UserProfileTool] 兴趣更新: {old_interests} -> {merged_interests}")
                
                if "preferences" in profile_data:
                    existing_profile.preferences.update(profile_data["preferences"])
                    logger.info(f"[UserProfileTool] 偏好更新: 新增/修改 {len(profile_data['preferences'])} 个偏好项")
                    logger.debug(f"[UserProfileTool] 偏好详情: {profile_data['preferences']}")
                
                if "life_stage" in profile_data and profile_data["life_stage"]:
                    old_stage = existing_profile.life_stage
                    existing_profile.life_stage = profile_data["life_stage"]
                    logger.info(f"[UserProfileTool] 生命周期阶段更新: {old_stage} -> {profile_data['life_stage']}")
                
                if "name" in profile_data and profile_data["name"]:
                    existing_profile.name = profile_data["name"]
                    logger.info(f"[UserProfileTool] 用户名更新: {profile_data['name']}")

                existing_profile.last_updated = datetime.now().isoformat()
                
                # 保存更新后的画像
                storage.store_user_profile(existing_profile)
                
                # 构建详细的返回信息
                result_msg = (
                    f"✅ 成功更新用户 {user_id} 画像\n"
                    f"  - 当前兴趣: {existing_profile.interests}\n"
                    f"  - 生命周期阶段: {existing_profile.life_stage}\n"
                    f"  - 偏好设置数量: {len(existing_profile.preferences)}\n"
                    f"  - 最后更新: {existing_profile.last_updated}"
                )
                logger.info(f"[UserProfileTool] {result_msg}")
                logger.info(f"[UserProfileTool] ========== 更新完成 ==========")
                return result_msg
            
            else:
                logger.error(f"[UserProfileTool] ❌ 未知的动作: {action}")
                return f"错误: 未知的动作 '{action}'，支持 'get' 或 'update'"

        except Exception as e:
            logger.error(f"[UserProfileTool] ❌ 执行出错: {str(e)}", exc_info=True)
            return f"工具执行错误: {str(e)}"

class MemoryRecallTool(BaseTool):
    """记忆召回工具 - 检索用户历史记忆"""
    name: str = "memory_recall_tool"
    description: str = "用于检索用户的长期记忆和实体记忆"

    def _run(self, user_id: str, memory_type: str = "all") -> str:
        """召回用户记忆"""
        storage = MemoryStorage()
        if memory_type == "long_term":
            memories = storage.get_long_term_memory(user_id)
        elif memory_type == "entity":
            memories = storage.get_entity_memory(user_id)
        else:
            memories = {
                "long_term": storage.get_long_term_memory(user_id),
                "entity": storage.get_entity_memory(user_id)
            }
        return json.dumps(memories, ensure_ascii=False, indent=2)

class AnalysisTool(BaseTool):
    """分析工具 - 分析用户行为模式"""
    name: str = "analysis_tool"
    description: str = (
        "用于分析用户未处理的交互事件，生成行为洞察。"
        "输入: user_id。"
        "输出: JSON 格式的事件摘要，包括事件类型统计、最近兴趣话题、情绪模式等。"
        "注意: 调用此工具后，相关事件会被标记为已处理。"
    )

    def _run(self, user_id: str) -> str:
        """分析用户行为"""
        logger.info(f"[AnalysisTool] ========== 开始分析 ==========")
        logger.info(f"[AnalysisTool] 分析目标用户: {user_id}")
        
        storage = MemoryStorage()
        
        try:
            events = storage.get_unprocessed_events(user_id)
            logger.info(f"[AnalysisTool] 获取到 {len(events)} 个未处理事件")
            
            if not events:
                logger.info(f"[AnalysisTool] ⚠️ 用户 {user_id} 没有新的交互事件需要分析")
                return "没有新的交互事件需要分析"

            # 详细记录事件信息
            logger.debug(f"[AnalysisTool] 事件详情:")
            for i, event in enumerate(events[:5]):  # 只记录前5个事件的详细信息
                logger.debug(f"  [{i+1}] 类型: {event.event_type}, 内容: {event.content[:50]}, 时间: {event.timestamp}")
            if len(events) > 5:
                logger.debug(f"  ... 还有 {len(events) - 5} 个事件")

            event_summary = {
                "total_events": len(events),
                "event_types": {},
                "recent_topics": [],
                "sentiment_trends": [],
                "behavior_patterns": [],
                "raw_content_samples": []  # 添加少量原始内容供LLM参考
            }

            for event in events:
                # 统计事件类型
                event_summary["event_types"][event.event_type] = \
                    event_summary["event_types"].get(event.event_type, 0) + 1

                # 提取元数据中的关键信息
                if event.metadata:
                    # 提取话题
                    if "topics" in event.metadata:
                        topics = event.metadata["topics"]
                        if isinstance(topics, list):
                            event_summary["recent_topics"].extend(topics)
                            logger.debug(f"[AnalysisTool] 提取话题: {topics}")
                        elif isinstance(topics, str):
                            event_summary["recent_topics"].append(topics)
                            logger.debug(f"[AnalysisTool] 提取话题: {topics}")
                    
                    # 提取情绪
                    if "sentiment" in event.metadata:
                        sentiment_info = {
                            "type": event.event_type,
                            "sentiment": event.metadata["sentiment"],
                            "content_preview": event.content[:30]
                        }
                        event_summary["sentiment_trends"].append(sentiment_info)
                        logger.debug(f"[AnalysisTool] 情绪记录: {event.metadata['sentiment']}")
                    
                    # 提取其他行为模式
                    if "duration" in event.metadata:
                        event_summary["behavior_patterns"].append(
                            f"浏览时长: {event.metadata['duration']}"
                        )
                    if "rating" in event.metadata:
                        event_summary["behavior_patterns"].append(
                            f"评分: {event.metadata['rating']}"
                        )

                # 仅保留前3个事件的简短内容作为上下文参考，避免Token溢出
                if len(event_summary["raw_content_samples"]) < 3:
                    event_summary["raw_content_samples"].append({
                        "type": event.event_type,
                        "content_preview": event.content[:80],
                        "timestamp": event.timestamp
                    })

            # 去重话题
            event_summary["recent_topics"] = list(set(event_summary["recent_topics"]))
            
            logger.info(f"[AnalysisTool] 分析结果汇总:")
            logger.info(f"  - 事件总数: {event_summary['total_events']}")
            logger.info(f"  - 事件类型分布: {event_summary['event_types']}")
            logger.info(f"  - 发现话题数: {len(event_summary['recent_topics'])}")
            logger.info(f"  - 情绪记录数: {len(event_summary['sentiment_trends'])}")
            logger.info(f"  - 行为模式数: {len(event_summary['behavior_patterns'])}")

            # 标记这些事件为已处理
            event_ids = [e.event_id for e in events]
            storage.mark_events_processed(event_ids)
            logger.info(f"[AnalysisTool] ✅ 已标记 {len(event_ids)} 个事件为已处理")

            result_json = json.dumps(event_summary, ensure_ascii=False, indent=2)
            logger.info(f"[AnalysisTool] 生成分析摘要，长度: {len(result_json)} 字符")
            logger.info(f"[AnalysisTool] ========== 分析完成 ==========")
            return result_json

        except Exception as e:
            logger.error(f"[AnalysisTool] ❌ 分析过程出错: {str(e)}", exc_info=True)
            return f"分析工具执行错误: {str(e)}"

class EvolutionTool(BaseTool):
    """进化工具 - 更新用户画像和记忆"""
    name: str = "evolution_tool"
    description: str = (
        "用于根据分析结果更新用户画像和长期记忆。"
        "输入参数:"
        "- user_id: 用户ID"
        "- analysis_result: 分析结果的文本描述"
        "- recommended_updates: 字典格式，包含 interests(列表), preferences(字典), life_stage(字符串)"
        "注意: preferences 必须是字典格式，不能是字符串。"
    )

    def _run(self, user_id: str, analysis_result: str, recommended_updates: Dict) -> str:
        """执行用户画像更新"""
        logger.info(f"[EvolutionTool] ========== 开始进化更新 ==========")
        logger.info(f"[EvolutionTool] 目标用户: {user_id}")
        logger.info(f"[EvolutionTool] 推荐更新参数类型检查:")
        logger.info(f"  - recommended_updates 类型: {type(recommended_updates)}")
        
        # 参数验证和类型转换
        if isinstance(recommended_updates, str):
            logger.warning(f"[EvolutionTool] ⚠️ recommended_updates 是字符串，尝试解析为JSON")
            try:
                recommended_updates = json.loads(recommended_updates)
                logger.info(f"[EvolutionTool] ✅ 成功解析字符串为字典")
            except json.JSONDecodeError as e:
                logger.error(f"[EvolutionTool] ❌ 无法解析 recommended_updates: {e}")
                return f"错误: recommended_updates 格式无效，必须是字典或JSON字符串。错误: {str(e)}"
        
        if not isinstance(recommended_updates, dict):
            logger.error(f"[EvolutionTool] ❌ recommended_updates 不是字典类型: {type(recommended_updates)}")
            return f"错误: recommended_updates 必须是字典类型，当前类型: {type(recommended_updates)}"
        
        logger.info(f"[EvolutionTool] 推荐更新字段: {list(recommended_updates.keys())}")
        logger.debug(f"[EvolutionTool] 推荐更新内容: {json.dumps(recommended_updates, ensure_ascii=False, indent=2)}")
        
        storage = MemoryStorage()

        try:
            # 获取现有用户画像
            profile = storage.get_user_profile(user_id)
            if not profile:
                logger.info(f"[EvolutionTool] 用户 {user_id} 不存在，创建新画像")
                profile = UserProfile(user_id=user_id)
            else:
                logger.info(f"[EvolutionTool] 找到现有用户画像")
                logger.debug(f"[EvolutionTool] 当前兴趣: {profile.interests}")
                logger.debug(f"[EvolutionTool] 当前偏好: {profile.preferences}")
                logger.debug(f"[EvolutionTool] 当前阶段: {profile.life_stage}")

            # 更新兴趣列表
            if "interests" in recommended_updates:
                new_interests = recommended_updates["interests"]
                
                # 确保 interests 是列表
                if isinstance(new_interests, str):
                    logger.warning(f"[EvolutionTool] ⚠️ interests 是字符串，尝试解析")
                    try:
                        new_interests = json.loads(new_interests)
                    except:
                        new_interests = [new_interests]
                
                if isinstance(new_interests, list):
                    current_interests = set(profile.interests)
                    new_interests_set = set([str(i) for i in new_interests if i])  # 过滤空值并转为字符串
                    merged_interests = list(current_interests | new_interests_set)
                    old_count = len(profile.interests)
                    profile.interests = merged_interests
                    logger.info(f"[EvolutionTool] ✅ 兴趣更新: {old_count}个 -> {len(merged_interests)}个")
                    logger.debug(f"[EvolutionTool] 新增兴趣: {new_interests_set - current_interests}")
                else:
                    logger.warning(f"[EvolutionTool] ⚠️ interests 格式不正确，跳过更新")

            # 更新偏好设置
            if "preferences" in recommended_updates:
                new_preferences = recommended_updates["preferences"]
                
                # 关键修复：确保 preferences 是字典
                if isinstance(new_preferences, str):
                    logger.warning(f"[EvolutionTool] ⚠️ preferences 是字符串，尝试解析为JSON")
                    try:
                        new_preferences = json.loads(new_preferences)
                        logger.info(f"[EvolutionTool] ✅ 成功解析 preferences 字符串为字典")
                    except json.JSONDecodeError as e:
                        logger.error(f"[EvolutionTool] ❌ 无法解析 preferences: {e}")
                        logger.warning(f"[EvolutionTool] 跳过 preferences 更新")
                        new_preferences = {}
                
                if isinstance(new_preferences, dict):
                    old_pref_count = len(profile.preferences)
                    profile.preferences.update(new_preferences)
                    new_pref_count = len(profile.preferences)
                    logger.info(f"[EvolutionTool] ✅ 偏好更新: {old_pref_count}个 -> {new_pref_count}个")
                    logger.debug(f"[EvolutionTool] 新增/修改的偏好: {new_preferences}")
                else:
                    logger.error(f"[EvolutionTool] ❌ preferences 不是字典类型: {type(new_preferences)}")
                    logger.warning(f"[EvolutionTool] 跳过 preferences 更新")

            # 更新生命周期阶段
            if "life_stage" in recommended_updates:
                new_life_stage = recommended_updates["life_stage"]
                if new_life_stage and isinstance(new_life_stage, str) and new_life_stage.strip():
                    old_stage = profile.life_stage
                    profile.life_stage = new_life_stage.strip()
                    logger.info(f"[EvolutionTool] ✅ 生命周期阶段更新: '{old_stage}' -> '{profile.life_stage}'")
                else:
                    logger.debug(f"[EvolutionTool] life_stage 为空或无效，跳过更新")

            # 更新交互计数和时间戳
            profile.interaction_count += 1
            profile.last_updated = datetime.now().isoformat()
            logger.info(f"[EvolutionTool] 交互次数: {profile.interaction_count}")

            # 保存更新后的画像
            storage.store_user_profile(profile)
            logger.info(f"[EvolutionTool] ✅ 用户画像已保存到数据库")

            # 更新长期记忆
            memory_update = {
                "analysis_summary": analysis_result[:500] if analysis_result else "",  # 限制长度
                "updates_applied": {
                    "interests": profile.interests,
                    "preferences_keys": list(profile.preferences.keys()),
                    "life_stage": profile.life_stage
                },
                "timestamp": datetime.now().isoformat()
            }
            storage.update_long_term_memory(user_id, "profile_updates", memory_update)
            logger.info(f"[EvolutionTool] ✅ 长期记忆已更新")

            result_msg = (
                f"✅ 用户 {user_id} 画像已成功更新\n"
                f"  - 交互次数: {profile.interaction_count}\n"
                f"  - 兴趣数量: {len(profile.interests)}\n"
                f"  - 偏好设置数量: {len(profile.preferences)}\n"
                f"  - 生命周期阶段: {profile.life_stage}\n"
                f"  - 最后更新: {profile.last_updated}"
            )
            logger.info(f"[EvolutionTool] {result_msg}")
            logger.info(f"[EvolutionTool] ========== 进化更新完成 ==========")
            return result_msg

        except Exception as e:
            logger.error(f"[EvolutionTool] ❌ 进化更新过程出错: {str(e)}", exc_info=True)
            return f"进化工具执行错误: {str(e)}"

class AdaptiveAgentSystem:
    """自适应智能体系统主类"""

    def __init__(self):
        self.storage = MemoryStorage()
        self.llm = LLM(
            model="ollama/qwen3.5:9b",
            # model="ollama/qwen3:1.7b",
            base_url="http://localhost:11434"
        )
        self._setup_agents()

    def _setup_agents(self):
        """设置Agent团队"""
        self.observer_agent = Agent(
            role="观察者Agent",
            goal="准确记录用户的每一个交互事件，不遗漏任何细节",
            backstory="我是用户的专属观察者，负责细心观察和记录用户的所有行为",
            verbose=True,
            tools=[ObserverTool()]
        )

        self.user_profile_agent = Agent(
            role="用户画像Agent",
            goal="维护准确、最新的用户画像信息",
            backstory="我是用户档案管理员，负责维护用户的完整画像和历史记录",
            verbose=True,
            tools=[UserProfileTool(), MemoryRecallTool()]
        )

        self.reflective_analyst_agent = Agent(
            role="反思分析师Agent",
            goal="深入分析用户行为，生成有价值的洞察和建议",
            backstory="我是专业的分析师，通过分析用户行为模式来提供个性化建议",
            verbose=True,
            tools=[AnalysisTool(), UserProfileTool()]
        )

        self.evolution_agent = Agent(
            role="进化Agent",
            goal="根据分析结果更新用户画像，实现智能体的自我进化",
            backstory="我是进化专家，负责根据用户反馈更新系统，实现自我完善",
            verbose=True,
            tools=[EvolutionTool(), MemoryRecallTool()]
        )

        self.concierge_agent = Agent(
            role="管家Agent",
            goal="响应用户指令，提供个性化服务",
            backstory="我是您的专属管家，了解您的喜好，为您提供贴心服务",
            verbose=True,
            llm=self.llm
        )

        self.personalizer_agent = Agent(
            role="个性化Agent",
            goal="生成符合用户画像的个性化响应",
            backstory="我是个性化专家，根据用户特点生成专属内容",
            verbose=True,
            llm=self.llm
        )

    def process_user_interaction(self, user_id: str, interaction_type: str,
                                content: str, metadata: Dict = None) -> Dict:
        """处理用户交互的主流程"""
        logger.info(f"处理用户 {user_id} 的 {interaction_type} 交互")

        try:
            observe_task = Task(
                description=f"记录用户 {user_id} 的 {interaction_type} 交互: {content}",
                agent=self.observer_agent,
                expected_output="事件记录确认"
            )

            crew = Crew(
                agents=[self.observer_agent],
                tasks=[observe_task],
                verbose=False
            )
            crew.kickoff()
        except Exception as e:
            logger.error(f"观察者Agent执行失败: {e}")

        try:
            profile_task = Task(
                description=f"获取用户 {user_id} 的画像",
                agent=self.user_profile_agent,
                expected_output="用户画像JSON"
            )

            crew = Crew(
                agents=[self.user_profile_agent],
                tasks=[profile_task],
                verbose=False
            )
            profile_result = crew.kickoff()
        except Exception as e:
            logger.error(f"用户画像Agent执行失败: {e}")
            profile_result = "{}"

        try:
            analyze_task = Task(
                description=f"分析用户 {user_id} 的最新交互",
                agent=self.reflective_analyst_agent,
                expected_output="行为分析报告"
            )

            crew = Crew(
                agents=[self.reflective_analyst_agent],
                tasks=[analyze_task],
                verbose=False
            )
            analysis_result = crew.kickoff()
        except Exception as e:
            logger.error(f"分析Agent执行失败: {e}")
            analysis_result = "{}"

        try:
            evolution_task = Task(
                description=f"根据分析结果更新用户 {user_id} 的画像",
                agent=self.evolution_agent,
                expected_output="更新确认"
            )

            crew = Crew(
                agents=[self.evolution_agent],
                tasks=[evolution_task],
                verbose=False
            )
            evolution_result = crew.kickoff()
        except Exception as e:
            logger.error(f"进化Agent执行失败: {e}")
            evolution_result = "{}"

        return {
            "profile": profile_result,
            "analysis": analysis_result,
            "evolution": evolution_result
        }

    def get_personalized_response(self, user_id: str, query: str) -> str:
        """获取个性化响应"""
        try:
            profile = self.storage.get_user_profile(user_id)

            if not profile:
                profile = UserProfile(user_id=user_id, name="新用户")
                self.storage.store_user_profile(profile)

            memory = self.storage.get_long_term_memory(user_id)
            entity = self.storage.get_entity_memory(user_id)

            context = f"""
            用户ID: {user_id}
            用户名: {profile.name}
            生命周期阶段: {profile.life_stage}
            兴趣爱好: {', '.join(profile.interests)}
            偏好设置: {json.dumps(profile.preferences, ensure_ascii=False)}
            交互次数: {profile.interaction_count}
            历史记忆: {json.dumps(memory, ensure_ascii=False)[:500]}
            实体记忆: {json.dumps(entity, ensure_ascii=False)[:500]}
            """

            prompt = f"""
            {context}

            用户查询: {query}

            请根据用户画像和历史记忆，给出个性化的回复。
            """

            personalizer_task = Task(
                description=prompt,
                agent=self.personalizer_agent,
                expected_output="个性化回复文本"
            )

            crew = Crew(
                agents=[self.personalizer_agent],
                tasks=[personalizer_task],
                verbose=True
            )
            result = crew.kickoff()

            self.process_user_interaction(
                user_id=user_id,
                interaction_type="conversation",
                content=query,
                metadata={"topics": self._extract_topics(query)}
            )

            return result
        except Exception as e:
            logger.error(f"个性化响应生成失败: {e}")
            return f"抱歉，我暂时无法提供个性化回复。错误信息: {str(e)[:100]}"

    def _extract_topics(self, text: str) -> List[str]:
        """提取文本中的话题"""
        keywords = ["工作", "生活", "学习", "技术", "旅行", "音乐", "电影", "阅读", "运动", "美食"]
        return [kw for kw in keywords if kw in text]

    def run_evolution_cycle(self, user_id: str):
        """运行自我进化闭环"""
        logger.info(f"启动用户 {user_id} 的自我进化闭环")

        try:
            storage = MemoryStorage()
            events = storage.get_unprocessed_events(user_id)

            if not events:
                logger.info("没有新的交互事件")
                return

            events_text = "\n".join([
                f"- 类型: {e.event_type}, 内容: {e.content[:100]}, 时间: {e.timestamp}"
                for e in events[:10]
            ])

            profile = storage.get_user_profile(user_id)
            current_profile = json.dumps(asdict(profile), ensure_ascii=False, indent=2) if profile else "无"

            analysis_prompt = f"""
            当前用户画像:
            {current_profile}

            最近交互事件:
            {events_text}

            请分析这些事件，识别用户兴趣变化、行为模式，并给出推荐更新。
            返回JSON格式:
            {{
                "new_interests": ["新发现的兴趣"],
                "updated_preferences": {{"偏好键": "偏好值"}},
                "life_stage_change": "如果检测到阶段变化则填写，否则为空",
                "insights": ["关键洞察1", "关键洞察2"]
            }}
            """

            analyst_task = Task(
                description=analysis_prompt,
                agent=self.reflective_analyst_agent,
                expected_output="JSON格式的分析和建议"
            )

            crew = Crew(
                agents=[self.reflective_analyst_agent],
                tasks=[analyst_task],
                verbose=False
            )
            analysis = crew.kickoff()

            try:
                if isinstance(analysis, str):
                    analysis_json = self._extract_json(analysis)
                else:
                    analysis_json = analysis

                if analysis_json and "new_interests" in analysis_json:
                    evolution_tool = EvolutionTool()
                    evolution_tool._run(
                        user_id=user_id,
                        analysis_result=str(analysis),
                        recommended_updates={
                            "interests": analysis_json.get("new_interests", []),
                            "preferences": analysis_json.get("updated_preferences", {}),
                            "life_stage": analysis_json.get("life_stage_change", "")
                        }
                    )
                    logger.info(f"进化完成: {analysis_json}")
            except Exception as e:
                logger.error(f"进化处理错误: {e}")
        except Exception as e:
            logger.error(f"自我进化闭环执行失败: {e}")

    def _extract_json(self, text: str) -> Dict:
        """从文本中提取JSON"""
        try:
            if not text or not isinstance(text, str):
                logger.warning(f"无效的输入文本: {type(text)}")
                return {}
            
            text = text.strip()
            
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}")
            
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f"直接解析JSON失败: {e}")
            
            logger.warning(f"无法从文本中提取有效JSON: {text[:100]}...")
            return {}
        except Exception as e:
            logger.error(f"提取JSON时发生意外错误: {e}")
            return {}

class UserBehaviorSimulator:
    """用户行为模拟器 - 用于演示和测试"""

    def __init__(self, agent_system: AdaptiveAgentSystem):
        self.agent = agent_system
        self.user_id = "demo_user"
    
    def _get_default_interactions(self):
        """获取默认的模拟交互数据"""
        return [
            {"type": "conversation", "content": "想开始学Python，有什么入门建议吗", "metadata": {"topics": ["Python", "学习"], "sentiment": "求知"}},
            {"type": "browse", "content": "搜索Python基础教程", "metadata": {"topics": ["Python", "教程"], "duration": "15分钟"}},
            {"type": "conversation", "content": "学完了Python基础，接下来该学什么", "metadata": {"topics": ["Python进阶", "学习规划"], "sentiment": "期待"}}
        ]

    def simulate_user_journey(self):
        """模拟用户旅程，展示自我进化效果"""
        print("\n" + "="*80)
        print("🚀 启动用户行为模拟 - 展示自我进化能力")
        print("="*80 + "\n")

        initial_profile = self.agent.storage.get_user_profile(self.user_id)
        if not initial_profile:
            initial_profile = UserProfile(
                user_id=self.user_id,
                name="张三",
                life_stage="探索期",
                interests=["基础编程"],
                preferences={"response_style": "简洁"},
                interaction_count=0
            )
            self.agent.storage.store_user_profile(initial_profile)

        interactions = [
            # ========== 阶段1：Python 基础 ==========
            {
                "type": "conversation",
                "content": "想开始学AI，听说要先学Python，不知道从哪入手",
                "metadata": {"topics": ["Python", "入门", "学习规划"], "sentiment": "迷茫"}
            },
            {
                "type": "browse",
                "content": "搜索了「Python基础教程 推荐」",
                "metadata": {"topics": ["Python", "教程"], "duration": "10分钟"}
            },
            {
                "type": "feedback",
                "content": "廖雪峰的Python教程很适合新手，例子很多",
                "metadata": {"rating": 5, "category": "教程评价"}
            },
            {
                "type": "conversation",
                "content": "刚学完循环和函数，自己写了一个猜数字小游戏，很有成就感",
                "metadata": {"topics": ["Python", "实践", "学习成果"], "sentiment": "积极"}
            },
            {
                "type": "browse",
                "content": "在GitHub上看了几个Python小游戏项目",
                "metadata": {"topics": ["Python", "开源项目"], "duration": "20分钟"}
            },
            # 生活场景：健身
            {
                "type": "conversation",
                "content": "晚上去健身房练了背，做了引体向上和划船",
                "metadata": {"topics": ["健身", "力量训练"], "sentiment": "中性"}
            },

            # ========== 阶段2：Python 进阶 + 数据分析 ==========
            {
                "type": "conversation",
                "content": "开始学装饰器、生成器，有点难但理解了闭包概念",
                "metadata": {"topics": ["Python进阶", "闭包"], "sentiment": "挑战"}
            },
            {
                "type": "browse",
                "content": "阅读了《Fluent Python》部分章节",
                "metadata": {"topics": ["Python进阶", "书籍"], "duration": "45分钟"}
            },
            {
                "type": "feedback",
                "content": "想找一些用pandas做数据清洗的实战案例",
                "metadata": {"rating": 4, "suggestion": "更多数据实战"}
            },
            {
                "type": "conversation",
                "content": "用pandas处理了一份电商销售数据，做了透视表和可视化",
                "metadata": {"topics": ["pandas", "数据分析", "可视化"], "sentiment": "兴奋"}
            },
            # 生活场景：购物
            {
                "type": "browse",
                "content": "在京东买了《利用Python进行数据分析》",
                "metadata": {"topics": ["购物", "书籍"], "duration": "5分钟"}
            },

            # ========== 阶段3：机器学习基础 ==========
            {
                "type": "conversation",
                "content": "我对机器学习很感兴趣，最近在自学Python和sklearn",
                "metadata": {"topics": ["技术", "学习", "机器学习"], "sentiment": "积极"}
            },
            {
                "type": "browse",
                "content": "观看了吴恩达《Machine Learning》第一周视频",
                "metadata": {"topics": ["机器学习", "课程"], "duration": "60分钟"}
            },
            {
                "type": "feedback",
                "content": "线性回归的梯度推导终于搞懂了，作业也做完了",
                "metadata": {"rating": 5, "category": "学习反馈"}
            },
            {
                "type": "conversation",
                "content": "用KNN做了一个鸢尾花分类，准确率96%，太有意思了",
                "metadata": {"topics": ["KNN", "分类", "实践"], "sentiment": "自豪"}
            },
            {
                "type": "browse",
                "content": "研究决策树和随机森林的区别",
                "metadata": {"topics": ["决策树", "随机森林"], "duration": "30分钟"}
            },
            # 生活场景：亲子活动
            {
                "type": "conversation",
                "content": "周末带孩子去了科技馆，体验了AI画画展区",
                "metadata": {"topics": ["亲子", "科技馆", "AI绘画"], "sentiment": "愉快"}
            },

            # ========== 阶段4：深度学习 (CNN/RNN) ==========
            {
                "type": "conversation",
                "content": "想了解深度学习在图像识别中的应用",
                "metadata": {"topics": ["深度学习", "图像识别", "技术"], "sentiment": "好奇"}
            },
            {
                "type": "browse",
                "content": "学习PyTorch官方教程，搭建了一个简单的CNN用于MNIST",
                "metadata": {"topics": ["PyTorch", "CNN", "MNIST"], "duration": "90分钟"}
            },
            {
                "type": "feedback",
                "content": "AI助手推荐的课程很有用，尤其是李沐的《动手学深度学习》",
                "metadata": {"rating": 5, "category": "课程推荐"}
            },
            {
                "type": "conversation",
                "content": "调了一晚上CNN的dropout和batch norm，终于把验证集准确率提到了99%",
                "metadata": {"topics": ["CNN", "调参", "优化"], "sentiment": "激动"}
            },
            {
                "type": "browse",
                "content": "读了一篇关于ResNet的论文，跳过了数学细节，主要看架构",
                "metadata": {"topics": ["ResNet", "论文"], "duration": "40分钟"}
            },
            # 生活场景：理发
            {
                "type": "conversation",
                "content": "去理发店剪了个短发，清爽过夏天",
                "metadata": {"topics": ["理发", "生活"], "sentiment": "轻松"}
            },

            # ========== 阶段5：自然语言处理 (NLP) ==========
            {
                "type": "conversation",
                "content": "开始学NLP，刚看完词向量和RNN，对LSTM的遗忘门有点晕",
                "metadata": {"topics": ["NLP", "RNN", "LSTM"], "sentiment": "困惑"}
            },
            {
                "type": "browse",
                "content": "在Kaggle上看了IMDB情感分析baseline",
                "metadata": {"topics": ["情感分析", "Kaggle"], "duration": "35分钟"}
            },
            {
                "type": "feedback",
                "content": "用GRU替换LSTM后训练快了很多，效果差不多，学到一招",
                "metadata": {"rating": 4, "category": "实践心得"}
            },
            {
                "type": "conversation",
                "content": "用Seq2Seq+Attention做了一个简单的英中翻译demo，虽然BLEU很低但能跑通",
                "metadata": {"topics": ["Seq2Seq", "Attention", "翻译"], "sentiment": "成就感"}
            },
            # 生活场景：同学聚会
            {
                "type": "conversation",
                "content": "周末参加大学同学聚会，好几个同学也在做AI，交流了转行经验",
                "metadata": {"topics": ["聚会", "社交", "职业"], "sentiment": "开心"}
            },

            # ========== 阶段6：大语言模型 (LLM) ==========
            {
                "type": "conversation",
                "content": "最近对强化学习产生了兴趣，有什么入门建议吗",
                "metadata": {"topics": ["强化学习", "技术", "学习建议"], "sentiment": "求知"}
            },
            {
                "type": "browse",
                "content": "看了HuggingFace的Transformer教程，理解了BERT的预训练任务",
                "metadata": {"topics": ["Transformer", "BERT", "HuggingFace"], "duration": "70分钟"}
            },
            {
                "type": "feedback",
                "content": "微调了一个BERT做新闻分类，F1达到0.92，比传统模型好太多",
                "metadata": {"rating": 5, "category": "项目成果"}
            },
            {
                "type": "conversation",
                "content": "尝试用GPT-2生成故事，发现prompt设计很重要，学到了temperature和top_p",
                "metadata": {"topics": ["GPT", "文本生成", "Prompt"], "sentiment": "有趣"}
            },
            {
                "type": "browse",
                "content": "阅读了《Attention Is All You Need》原文，理解了多头注意力的细节",
                "metadata": {"topics": ["论文", "Transformer"], "duration": "50分钟"}
            },
            # 生活场景：旅行
            {
                "type": "conversation",
                "content": "端午去了大理，在洱海边骑行，放松了一周",
                "metadata": {"topics": ["旅行", "骑行", "放松"], "sentiment": "愉悦"}
            },

            # ========== 阶段7：检索增强生成 (RAG) ==========
            {
                "type": "conversation",
                "content": "现在想学RAG，需要选哪个向量数据库？",
                "metadata": {"topics": ["RAG", "向量数据库"], "sentiment": "理性"}
            },
            {
                "type": "browse",
                "content": "对比了Chroma、FAISS、Pinecone，决定先用FAISS本地试试",
                "metadata": {"topics": ["向量数据库", "FAISS"], "duration": "40分钟"}
            },
            {
                "type": "feedback",
                "content": "搭建了一个简单RAG pipeline：PDF加载->分块->embedding->检索->LLM回答，效果不错",
                "metadata": {"rating": 5, "category": "项目实践"}
            },
            {
                "type": "conversation",
                "content": "遇到一个问题：检索出来的片段和问题不相关，需要调chunk_size和重排序",
                "metadata": {"topics": ["RAG", "调优", "Retrieval"], "sentiment": "困惑"}
            },
            {
                "type": "browse",
                "content": "研究了HyDE和Query Rewriting技术，决定试试Multi-Query",
                "metadata": {"topics": ["RAG优化", "HyDE"], "duration": "35分钟"}
            },
            # 生活场景：开项目例会
            {
                "type": "conversation",
                "content": "公司项目例会，我提议用RAG做内部知识库问答，技术负责人很感兴趣",
                "metadata": {"topics": ["工作", "会议", "项目"], "sentiment": "积极"}
            },

            # ========== 阶段8：AI Agent ==========
            {
                "type": "conversation",
                "content": "最近在学习openclaw，有什么入门建议吗",
                "metadata": {"topics": ["openclaw", "技术", "学习建议"], "sentiment": "求知"}
            },
            {
                "type": "browse",
                "content": "浏览了openclaw相关文档，发现它是一个多agent协作框架",
                "metadata": {"topics": ["openclaw", "Agent"], "duration": "30分钟"}
            },
            {
                "type": "feedback",
                "content": "希望推荐更多openclaw相关项目，比如基于openclaw的自动化测试项目",
                "metadata": {"rating": 4, "suggestion": "需要更多实践"}
            },
            {
                "type": "conversation",
                "content": "刚用LangChain写了一个ReAct Agent，能调用搜索引擎和计算器",
                "metadata": {"topics": ["LangChain", "ReAct", "Agent"], "sentiment": "兴奋"}
            },
            {
                "type": "browse",
                "content": "研究了AutoGPT和BabyAGI的设计思路，计划把记忆模块加进自己的Agent",
                "metadata": {"topics": ["AutoGPT", "Agent记忆"], "duration": "60分钟"}
            },
            {
                "type": "feedback",
                "content": "跑通了Multi-Agent对话（用户-助手-工具），下一步加路由和规划",
                "metadata": {"rating": 5, "category": "里程碑"}
            },
            # 生活场景：心理咨询
            {
                "type": "conversation",
                "content": "因为连续加班压力大，约了一次线上心理咨询，聊了情绪管理",
                "metadata": {"topics": ["心理健康", "压力"], "sentiment": "舒缓"}
            },

            # ========== 阶段9：高级/前沿（Claude Code Skill等） ==========
            {
                "type": "conversation",
                "content": "最近在学习claudecode skill，有什么入门建议吗",
                "metadata": {"topics": ["claudecode skill", "技术", "学习建议"], "sentiment": "求知"}
            },
            {
                "type": "browse",
                "content": "浏览了claudecode skill相关书籍和官方例子",
                "metadata": {"topics": ["claudecode skill", "书籍"], "duration": "30分钟"}
            },
            {
                "type": "feedback",
                "content": "希望推荐更多claudecode无线续杯、Cursor无限续杯解决方案",
                "metadata": {"rating": 4, "suggestion": "需要更多实践"}
            },
            {
                "type": "conversation",
                "content": "用Claude Code Skill做了一个自动生成代码注释的插件，省了不少时间",
                "metadata": {"topics": ["Claude", "自动化", "效率"], "sentiment": "满足"}
            },
            {
                "type": "browse",
                "content": "跟踪最新Agent论文：《WebArena》、《SWE-Agent》",
                "metadata": {"topics": ["论文", "Agent"], "duration": "45分钟"}
            },
            {
                "type": "feedback",
                "content": "把整个学习路径整理了博客，希望帮助同样转行AI的人",
                "metadata": {"rating": 5, "category": "知识输出"}
            },
            # 生活场景：参加培训
            {
                "type": "conversation",
                "content": "报名了一个线上AI产品经理训练营，学习如何从技术视角设计AI功能",
                "metadata": {"topics": ["培训", "AI产品"], "sentiment": "期待"}
            },
        ]
        
        ## 从文件中读取交互数据
        try:
            interactions = json.load(open("data/life_data.json"))
            logger.info(f"从 life_data.json 加载了 {len(interactions)} 条记录")
        except Exception as e:
            logger.error(f"加载 life_data.json 失败: {e}")
            interactions = []
        
        if not interactions:
            try:
                interactions = json.load(open("data/purchase_data.json"))
                logger.info(f"从 purchase_data.json 加载了 {len(interactions)} 条记录")
            except Exception as e:
                logger.error(f"加载 purchase_data.json 失败: {e}")
                interactions = []
        
        if not interactions:
            logger.warning("未加载到任何交互数据，使用默认模拟数据")
            interactions = self._get_default_interactions()

        for i, interaction in enumerate(interactions, 1):
            try:
                print(f"\n{'='*80}")
                print(f"📌 第 {i} 次交互: {interaction['type']}")
                print(f"内容: {interaction['content']}")
                print(f"元数据: {json.dumps(interaction['metadata'], ensure_ascii=False)}")
                print("="*80)

                self.agent.process_user_interaction(
                    user_id=self.user_id,
                    interaction_type=interaction["type"],
                    content=interaction["content"],
                    metadata=interaction["metadata"]
                )

                ## 每次都进化好了
                '''
                if i % 2 == 0:
                    print(f"\n🔄 执行第 {i//2} 次自我进化闭环...")
                    self.agent.run_evolution_cycle(self.user_id)
                '''
                
                print(f"\n🔄 执行第 {i} 次自我进化闭环...")
                self.agent.run_evolution_cycle(self.user_id)
                
                current_profile = self.agent.storage.get_user_profile(self.user_id)
                if current_profile:
                    print(f"\n📊 当前用户画像:")
                    print(f"  - 用户名: {current_profile.name}")
                    print(f"  - 生命周期阶段: {current_profile.life_stage}")
                    print(f"  - 兴趣爱好: {', '.join(current_profile.interests)}")
                    print(f"  - 偏好设置: {json.dumps(current_profile.preferences, ensure_ascii=False)}")
                    print(f"  - 交互次数: {current_profile.interaction_count}")
                    
                    query = "分析一下用户现在的状况，给一些生活或者学习的指导建议"
                    print(f"\n用户查询: {query}")
                    response = self.agent.get_personalized_response(self.user_id, query)
                    print(f"\n系统回复:\n{response}")
                else:
                    print("\n⚠️  用户画像获取失败")
            except Exception as e:
                logger.error(f"第 {i} 次交互处理失败: {e}")
                print(f"❌ 第 {i} 次交互处理失败，继续下一次...")

        # 进化之后直接进行推荐，这样才能够凸显agent的能力
        try:
            print(f"\n{'='*80}")
            print("📈 最终用户画像进化结果:")
            print("="*80)

            final_profile = self.agent.storage.get_user_profile(self.user_id)
            if final_profile:
                print(f"\n用户ID: {final_profile.user_id}")
                print(f"用户名: {final_profile.name}")
                print(f"生命周期阶段: {final_profile.life_stage}")
                print(f"兴趣爱好: {', '.join(final_profile.interests)}")
                print(f"偏好设置: {json.dumps(final_profile.preferences, ensure_ascii=False)}")
                print(f"交互次数: {final_profile.interaction_count}")
                print(f"最后更新: {final_profile.last_updated}")
            else:
                print("\n⚠️  无法获取最终用户画像")

            print(f"\n{'='*80}")
            print("💡 个性化推荐演示:")
            print("="*80)

            query = "推荐一些学习课程或者资料，给出一下实施建议或者一些实施方案应对策略"
            print(f"\n用户查询: {query}")
            response = self.agent.get_personalized_response(self.user_id, query)
            print(f"\n系统回复:\n{response}")

            print(f"\n{'='*80}")
            print("✅ 自我进化演示完成")
            print("="*80 + "\n")
        except Exception as e:
            logger.error(f"演示结束阶段出错: {e}")
            print(f"\n❌ 演示结束阶段出错: {e}")

async def main():
    """主函数"""
    try:
        print("\n" + "="*80)
        print("🎯 CrewAI自适应智能体系统初始化")
        print("="*80 + "\n")

        agent_system = AdaptiveAgentSystem()

        simulator = UserBehaviorSimulator(agent_system)
        simulator.simulate_user_journey()
    except Exception as e:
        logger.error(f"主程序执行失败: {e}", exc_info=True)
        print(f"\n❌ 系统运行失败: {e}")
        print("请检查Ollama服务是否正常运行，以及模型是否已正确下载")

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序启动失败: {e}")