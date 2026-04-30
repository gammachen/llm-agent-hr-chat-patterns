from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama

# 1. 定义共享工具（结果会自动存入团队上下文）
from crewai_tools import SerpApiGoogleSearchTool
## 需要用到api key
# os.environ["SERPAPI_API_KEY"] = "YOUR_API_KEY"

search_tool = SerpApiGoogleSearchTool(
    search_type="google",
    search_results=10,
    search_depth="deep",
)

print("📡 初始化Ollama LLM (qwen3.5:9b)...")
llm = ChatOllama(
    model="qwen3.5:9b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# 2. 创建Agent
researcher = Agent(
    role="Redis技术研究员",
    goal="搜索Redis最新关键特性、核心功能和技术实现细节",
    backstory="专注于NoSQL数据库技术研究10年，深入了解Redis架构和实现原理",
    tools=[search_tool],
    verbose=True
)

outline_writer = Agent(
    role="技术大纲架构师",
    goal="将研究成果整理成结构化的技术文档大纲",
    backstory="资深技术文档工程师，擅长设计清晰、层次分明的技术文档结构",
    verbose=True
)

detail_writer = Agent(
    role="深度技术撰稿人",
    goal="基于大纲撰写深入、专业的技术内容",
    backstory="资深后端开发工程师，具有丰富的Redis实战经验，擅长撰写技术深度文章",
    verbose=True
)

# 3. 定义任务流程
# 阶段1：搜索Redis关键特性
research_task = Task(
    description="搜索Redis的核心技术特性，包括但不限于：数据结构、持久化机制、集群架构、缓存策略、性能优化、最新版本新特性",
    expected_output='''JSON格式的Redis核心技术特性列表，包含以下字段：
{
  "features": [
    {
      "feature_name": "特性名称（如：Redis Cluster集群架构）",
      "description": "特性概述（100字左右）",
      "importance": "high/medium/low",
      "latest_info": "最新相关信息或版本更新",
      "key_topics": ["子主题1", "子主题2", "子主题3"]
    }
  ],
  "total_count": 6
}''',
    agent=researcher,
    llm=llm
)

# 阶段2：生成技术文档大纲
outline_task = Task(
    description='''基于研究员提供的Redis特性数据，设计一份完整的技术文档大纲。

输入数据包含多个特性，每个特性有：
- feature_name: 特性名称
- description: 特性概述
- importance: 重要性等级
- latest_info: 最新信息
- key_topics: 子主题列表

要求：
1. 将特性按逻辑分组（如：核心数据结构、高可用架构、性能优化等）
2. 为每个分组设计章节结构
3. 每个章节至少包含3个小节
4. 标记各章节的建议字数
5. 确保大纲层次清晰，符合技术文档规范
''',
    expected_output='''JSON格式的技术文档大纲：
{
  "title": "Redis核心技术深度解析",
  "version": "1.0",
  "chapters": [
    {
      "chapter_num": "第1章",
      "chapter_title": "章节标题",
      "description": "章节概述",
      "suggested_words": 2000,
      "sections": [
        {
          "section_num": "1.1",
          "section_title": "小节标题",
          "description": "小节内容概述",
          "suggested_words": 800,
          "topics": ["需要覆盖的要点1", "需要覆盖的要点2"]
        }
      ]
    }
  ],
  "appendix": ["附录内容列表"]
}''',
    agent=outline_writer,
    llm=llm,
    context=[research_task]
)

# 阶段3：详细撰写各章节内容
# 第1章：核心数据结构详解
write_chapter1 = Task(
    description='''基于大纲撰写第1章内容：Redis核心数据结构深度解析

要求：
1. 深入讲解String、List、Hash、Set、Sorted Set五种基本数据结构
2. 分析每种数据结构的底层实现原理（如：SDS、双向链表、哈希表、跳表等）
3. 对比各数据结构的适用场景和性能特点
4. 提供实际使用示例和最佳实践
5. 引用官方文档和权威资料

目标字数：2000字
''',
    expected_output="详细的技术章节内容，包含代码示例、原理分析、性能对比表格",
    agent=detail_writer,
    llm=llm,
    context=[outline_task],
    output_file="redis_chapter1.md"
)

# 第2章：持久化机制剖析
write_chapter2 = Task(
    description='''基于大纲撰写第2章内容：Redis持久化机制深度剖析

要求：
1. 详细分析RDB快照机制（触发条件、执行流程、优缺点）
2. 深入讲解AOF日志机制（三种同步策略、重写机制）
3. 对比RDB和AOF的适用场景和性能影响
4. 分析混合持久化模式（RDB+AOF）的工作原理
5. 提供持久化配置优化建议

目标字数：2500字
''',
    expected_output="详细的技术章节内容，包含流程图、配置示例、性能测试数据",
    agent=detail_writer,
    llm=llm,
    context=[outline_task, write_chapter1],
    output_file="redis_chapter2.md"
)

# 第3章：集群架构设计
write_chapter3 = Task(
    description='''基于大纲撰写第3章内容：Redis Cluster集群架构设计

要求：
1. 分析Redis Cluster的分布式架构原理
2. 深入讲解槽位（Slot）分配和迁移机制
3. 分析主从复制和故障转移流程
4. 讲解Gossip协议在集群中的应用
5. 提供集群部署、扩容和运维最佳实践

目标字数：3000字
''',
    expected_output="详细的技术章节内容，包含架构图、故障场景分析、运维指南",
    agent=detail_writer,
    llm=llm,
    context=[outline_task, write_chapter1, write_chapter2],
    output_file="redis_chapter3.md"
)

# 第4章：性能优化实战
write_chapter4 = Task(
    description='''基于大纲撰写第4章内容：Redis性能优化实战

要求：
1. 分析Redis性能瓶颈常见原因
2. 深入讲解内存优化策略（数据结构优化、过期策略、内存碎片整理）
3. 分析网络IO优化（连接池、Pipeline、集群分片）
4. 讲解CPU优化（命令复杂度、Lua脚本、多线程IO）
5. 提供性能监控和调优方法论

目标字数：2500字
''',
    expected_output="详细的技术章节内容，包含性能测试方法、优化前后对比、监控指标说明",
    agent=detail_writer,
    llm=llm,
    context=[outline_task, write_chapter1, write_chapter2, write_chapter3],
    output_file="redis_chapter4.md"
)

# 4. 创建团队
crew = Crew(
    agents=[researcher, outline_writer, detail_writer],
    tasks=[
        research_task,
        outline_task,
        write_chapter1,
        write_chapter2,
        write_chapter3,
        write_chapter4
    ],
    process=Process.sequential,
    verbose=True
)

# 执行任务
result = crew.kickoff()
print("\n" + "="*80)
print("📄 最终输出摘要:")
print("="*80)
print(result)
