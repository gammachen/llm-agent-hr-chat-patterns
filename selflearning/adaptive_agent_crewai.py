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
    description: str = "用于获取和更新用户画像信息"

    def _run(self, user_id: str, action: str = "get", profile_data: Dict = None) -> str:
        """获取或更新用户画像"""
        storage = MemoryStorage()
        if action == "get":
            profile = storage.get_user_profile(user_id)
            if profile:
                return json.dumps(asdict(profile), ensure_ascii=False, indent=2)
            return f"未找到用户 {user_id} 的画像"
        elif action == "update" and profile_data:
            profile = UserProfile(
                user_id=user_id,
                last_updated=datetime.now().isoformat(),
                **profile_data
            )
            storage.store_user_profile(profile)
            return f"已更新用户 {user_id} 的画像"

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
    description: str = "用于分析用户交互事件，生成行为洞察"

    def _run(self, user_id: str) -> str:
        """分析用户行为"""
        storage = MemoryStorage()
        events = storage.get_unprocessed_events(user_id)

        if not events:
            return "没有新的交互事件需要分析"

        event_summary = {
            "total_events": len(events),
            "event_types": {},
            "recent_interests": [],
            "behavior_patterns": []
        }

        for event in events:
            event_summary["event_types"][event.event_type] = \
                event_summary["event_types"].get(event.event_type, 0) + 1

            if event.event_type == "conversation" and event.metadata:
                if "topics" in event.metadata:
                    event_summary["recent_interests"].extend(event.metadata["topics"])
                if "sentiment" in event.metadata:
                    event_summary["behavior_patterns"].append(f"情绪:{event.metadata['sentiment']}")

        storage.mark_events_processed([e.event_id for e in events])

        return json.dumps(event_summary, ensure_ascii=False, indent=2)

class EvolutionTool(BaseTool):
    """进化工具 - 更新用户画像和记忆"""
    name: str = "evolution_tool"
    description: str = "用于根据分析结果更新用户画像和长期记忆"

    def _run(self, user_id: str, analysis_result: str, recommended_updates: Dict) -> str:
        """执行用户画像更新"""
        storage = MemoryStorage()

        profile = storage.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)

        if "interests" in recommended_updates:
            current_interests = set(profile.interests)
            new_interests = set(recommended_updates["interests"])
            profile.interests = list(current_interests | new_interests)

        if "preferences" in recommended_updates:
            profile.preferences.update(recommended_updates["preferences"])

        if "life_stage" in recommended_updates:
            profile.life_stage = recommended_updates["life_stage"]

        profile.interaction_count += 1
        profile.last_updated = datetime.now().isoformat()

        storage.store_user_profile(profile)

        storage.update_long_term_memory(user_id, "profile_updates", {
            "analysis": analysis_result,
            "updates": recommended_updates,
            "timestamp": datetime.now().isoformat()
        })

        return f"用户 {user_id} 画像已更新: 交互次数={profile.interaction_count}"

class AdaptiveAgentSystem:
    """自适应智能体系统主类"""

    def __init__(self):
        self.storage = MemoryStorage()
        self.llm = LLM(
            model="ollama/qwen3.5:9b",
            base_url="http://localhost:11434"
        )
        self._setup_agents()

    def _setup_agents(self):
        """设置Agent团队"""
        self.observer_agent = Agent(
            role="观察者Agent",
            goal="准确记录用户的每一个交互事件，不遗漏任何细节",
            backstory="我是用户的专属观察者，负责细心观察和记录用户的所有行为",
            verbose=False,
            tools=[ObserverTool()]
        )

        self.user_profile_agent = Agent(
            role="用户画像Agent",
            goal="维护准确、最新的用户画像信息",
            backstory="我是用户档案管理员，负责维护用户的完整画像和历史记录",
            verbose=False,
            tools=[UserProfileTool(), MemoryRecallTool()]
        )

        self.reflective_analyst_agent = Agent(
            role="反思分析师Agent",
            goal="深入分析用户行为，生成有价值的洞察和建议",
            backstory="我是专业的分析师，通过分析用户行为模式来提供个性化建议",
            verbose=False,
            tools=[AnalysisTool(), UserProfileTool()]
        )

        self.evolution_agent = Agent(
            role="进化Agent",
            goal="根据分析结果更新用户画像，实现智能体的自我进化",
            backstory="我是进化专家，负责根据用户反馈更新系统，实现自我完善",
            verbose=False,
            tools=[EvolutionTool(), MemoryRecallTool()]
        )

        self.concierge_agent = Agent(
            role="管家Agent",
            goal="响应用户指令，提供个性化服务",
            backstory="我是您的专属管家，了解您的喜好，为您提供贴心服务",
            verbose=False,
            llm=self.llm
        )

        self.personalizer_agent = Agent(
            role="个性化Agent",
            goal="生成符合用户画像的个性化响应",
            backstory="我是个性化专家，根据用户特点生成专属内容",
            verbose=False,
            llm=self.llm
        )

    def process_user_interaction(self, user_id: str, interaction_type: str,
                                content: str, metadata: Dict = None) -> Dict:
        """处理用户交互的主流程"""
        logger.info(f"处理用户 {user_id} 的 {interaction_type} 交互")

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

        return {
            "profile": profile_result,
            "analysis": analysis_result,
            "evolution": evolution_result
        }

    def get_personalized_response(self, user_id: str, query: str) -> str:
        """获取个性化响应"""
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
            verbose=False
        )
        result = crew.kickoff()

        self.process_user_interaction(
            user_id=user_id,
            interaction_type="conversation",
            content=query,
            metadata={"topics": self._extract_topics(query)}
        )

        return result

    def _extract_topics(self, text: str) -> List[str]:
        """提取文本中的话题"""
        keywords = ["工作", "生活", "学习", "技术", "旅行", "音乐", "电影", "阅读", "运动", "美食"]
        return [kw for kw in keywords if kw in text]

    def run_evolution_cycle(self, user_id: str):
        """运行自我进化闭环"""
        logger.info(f"启动用户 {user_id} 的自我进化闭环")

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

    def _extract_json(self, text: str) -> Dict:
        """从文本中提取JSON"""
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {}

class UserBehaviorSimulator:
    """用户行为模拟器 - 用于演示和测试"""

    def __init__(self, agent_system: AdaptiveAgentSystem):
        self.agent = agent_system
        self.user_id = "demo_user"

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
            {
                "type": "conversation",
                "content": "我对机器学习很感兴趣，最近在自学Python",
                "metadata": {"topics": ["技术", "学习", "机器学习"], "sentiment": "积极"}
            },
            {
                "type": "conversation",
                "content": "想了解深度学习在图像识别中的应用",
                "metadata": {"topics": ["深度学习", "图像识别", "技术"], "sentiment": "好奇"}
            },
            {
                "type": "feedback",
                "content": "AI助手推荐的课程很有用",
                "metadata": {"rating": 5, "category": "课程推荐"}
            },
            {
                "type": "conversation",
                "content": "最近对强化学习产生了兴趣，有什么入门建议吗",
                "metadata": {"topics": ["强化学习", "技术", "学习建议"], "sentiment": "求知"}
            },
            {
                "type": "browse",
                "content": "浏览了强化学习相关书籍",
                "metadata": {"topics": ["强化学习", "书籍"], "duration": "30分钟"}
            },
            {
                "type": "feedback",
                "content": "希望推荐更多实战项目",
                "metadata": {"rating": 4, "suggestion": "需要更多实践"}
            },
        ]

        for i, interaction in enumerate(interactions, 1):
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

            if i % 2 == 0:
                print(f"\n🔄 执行第 {i//2} 次自我进化闭环...")
                self.agent.run_evolution_cycle(self.user_id)

            current_profile = self.agent.storage.get_user_profile(self.user_id)
            print(f"\n📊 当前用户画像:")
            print(f"  - 用户名: {current_profile.name}")
            print(f"  - 生命周期阶段: {current_profile.life_stage}")
            print(f"  - 兴趣爱好: {', '.join(current_profile.interests)}")
            print(f"  - 偏好设置: {json.dumps(current_profile.preferences, ensure_ascii=False)}")
            print(f"  - 交互次数: {current_profile.interaction_count}")

        print(f"\n{'='*80}")
        print("📈 最终用户画像进化结果:")
        print("="*80)

        final_profile = self.agent.storage.get_user_profile(self.user_id)
        print(f"\n用户ID: {final_profile.user_id}")
        print(f"用户名: {final_profile.name}")
        print(f"生命周期阶段: {final_profile.life_stage}")
        print(f"兴趣爱好: {', '.join(final_profile.interests)}")
        print(f"偏好设置: {json.dumps(final_profile.preferences, ensure_ascii=False)}")
        print(f"交互次数: {final_profile.interaction_count}")
        print(f"最后更新: {final_profile.last_updated}")

        print(f"\n{'='*80}")
        print("💡 个性化推荐演示:")
        print("="*80)

        query = "我想深入学习AI，应该从哪里开始？"
        print(f"\n用户查询: {query}")
        response = self.agent.get_personalized_response(self.user_id, query)
        print(f"\n系统回复:\n{response}")

        print(f"\n{'='*80}")
        print("✅ 自我进化演示完成")
        print("="*80 + "\n")

async def main():
    """主函数"""
    print("\n" + "="*80)
    print("🎯 CrewAI自适应智能体系统初始化")
    print("="*80 + "\n")

    agent_system = AdaptiveAgentSystem()

    simulator = UserBehaviorSimulator(agent_system)
    simulator.simulate_user_journey()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())