# Keep 项目 AI/LLM 机制深度分析与实现文档

## 目录

1. [概述](#1-概述)
2. [LLM 大模型机制分析](#2-llm-大模型机制分析)
   - 2.1 [OpenAI 集成架构](#21-openai-集成架构)
   - 2.2 [多 LLM 提供者支持](#22-多-llm-提供者支持)
   - 2.3 [结构化输出与 JSON Schema](#23-结构化输出与-json-schema)
3. [AI 助手机制分析](#3-ai-助手机制分析)
   - 3.1 [AI 工作流构建助手](#31-ai-工作流构建助手)
   - 3.2 [AI 事件助手](#32-ai-事件助手)
   - 3.3 [AI 半自动关联](#33-ai-半自动关联)
   - 3.4 [AI 事件报告生成](#34-ai-事件报告生成)
4. [AI 关联引擎机制分析](#4-ai-关联引擎机制分析)
   - 4.1 [告警处理流水线](#41-告警处理流水线)
   - 4.2 [基于规则的关联引擎](#42-基于规则的关联引擎)
   - 4.3 [基于 Transformer 的外部 AI 关联](#43-基于-transformer-的外部-ai-关联)
   - 4.4 [历史数据训练机制](#44-历史数据训练机制)
5. [核心代码解析](#5-核心代码解析)
   - 5.1 [AI 建议业务逻辑层](#51-ai-建议业务逻辑层)
   - 5.2 [告警处理流水线代码](#52-告警处理流水线代码)
   - 5.3 [规则引擎代码](#53-规则引擎代码)
6. [如何在其他项目中复用](#6-如何在其他项目中复用)
   - 6.1 [LLM 集成通用方案](#61-llm-集成通用方案)
   - 6.2 [AI 助手集成方案](#62-ai-助手集成方案)
   - 6.3 [AI 关联引擎集成方案](#63-ai-关联引擎集成方案)
   - 6.4 [完整实现路线图](#64-完整实现路线图)

---

## 1. 概述

Keep 是一个开源的告警管理与事件响应平台，深度集成了 AI/LLM 能力，主要体现在三个层面：

| 层面 | 功能 | 技术实现 |
|------|------|----------|
| **LLM 大模型集成** | 多提供者 LLM 支持 | OpenAI、Anthropic、DeepSeek、Gemini、Ollama |
| **AI 助手** | 工作流构建、事件分析、报告生成 | CopilotKit、OpenAI API 结构化输出 |
| **AI 关联引擎** | 告警聚类、事件创建、智能关联 | CEL 规则引擎 + Transformer 模型 |

**架构设计理念**：
- **混合 AI 策略**：规则引擎（确定性）+ LLM（智能推理）+ Transformer 模型（自学习）
- **人机协同**：AI 提供建议，人类做最终决策
- **多租户隔离**：每个租户独立的 AI 模型和数据
- **反馈闭环**：收集用户反馈持续优化 AI 输出

---

## 2. LLM 大模型机制分析

### 2.1 OpenAI 集成架构

Keep 的 LLM 核心集成基于 OpenAI API，主要用于以下场景：

#### 代码位置与关键文件

| 文件路径 | 功能 |
|----------|------|
| `keep/providers/openai_provider/openai_provider.py` | OpenAI Provider 封装 |
| `keep/api/bl/ai_suggestion_bl.py` | AI 建议业务逻辑 |
| `keep/api/bl/incident_reports.py` | 事件报告 AI 生成 |
| `keep-ui/app/api/copilotkit/route.ts` | CopilotKit 路由 |

#### OpenAI Provider 核心实现

```python
# 文件：keep/providers/openai_provider/openai_provider.py
class OpenaiProvider(BaseProvider):
    def __init__(self, context_manager, provider_id, config):
        self.client = OpenAI(
            api_key=config.authentication.get("api_key"),
            organization=config.authentication.get("organization_id")
        )

    def _query(self, prompt, model="gpt-3.5-turbo", max_tokens=1024,
               structured_output_format=None):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if structured_output_format:
            kwargs["response_format"] = structured_output_format

        response = self.client.chat.completions.create(**kwargs)
        result = response.choices[0].message.content

        # 尝试解析为 JSON，失败则返回原始文本
        try:
            return {"response": json.loads(result)}
        except json.JSONDecodeError:
            return {"response": result}
```

**设计要点**：
- **统一抽象**：继承 `BaseProvider`，所有 LLM 提供者使用相同接口
- **结构化输出**：支持 `response_format` 参数强制 JSON Schema 输出
- **容错处理**：JSON 解析失败时优雅降级为原始文本

#### AI 建议中的 LLM 调用

```python
# 文件：keep/api/bl/ai_suggestion_bl.py
class AISuggestionBl:
    def suggest_incidents(self, alerts_dto, topology_data, user_id):
        # 1. 构建系统提示词
        system_prompt = """你是一位专业的事件管理系统，负责分析告警并将其聚类为事件。
        分析维度包括：
        - 告警描述和内容
        - 时间接近性
        - 受影响的系统/服务
        - IT 问题类型（性能、故障、资源）
        - 根因分析
        - 服务拓扑依赖关系"""

        # 2. 构建用户提示词（包含告警和拓扑数据）
        alerts_text = "\n".join([
            f"Alert {idx+1}: {json.dumps(alert.dict(), default=str)}"
            for idx, alert in enumerate(alerts_dto[:50])  # 最多50条告警，避免超出 Token 限制
        ])
        topology_text = "\n".join([
            f"Topology {idx+1}: {json.dumps(t.dict(), default=str)}"
            for idx, t in enumerate(topology_data)
        ])

        # 3. 调用 OpenAI API（结构化 JSON 输出）
        response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"),
            temperature=0.2,  # 低随机性，确保输出一致性
            response_format={"type": "json_schema", "json_schema": {...}},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
```

**关键参数说明**：
- `temperature=0.2`：低温度保证输出确定性，适合需要稳定结果的场景
- `response_format`：使用 JSON Schema 约束输出格式，确保结构化数据
- 最多处理 50 条告警：避免超出 Token 限制

### 2.2 多 LLM 提供者支持

Keep 通过 Provider 模式支持多种 LLM 后端：

| 提供者 | 文件路径 | 特点 |
|--------|----------|------|
| OpenAI | `keep/providers/openai_provider/` | 主力提供者，支持结构化输出 |
| Anthropic | `keep/providers/anthropic_provider/` | Claude 模型，支持 JSON 约束 |
| DeepSeek | `keep/providers/deepseek_provider/` | 兼容 OpenAI API 格式 |
| Gemini | `keep/providers/gemini_provider/` | Google AI SDK |
| Ollama | `keep/providers/ollama_provider/` | 本地部署模型 |

**Provider 统一接口**：

```python
# 所有 LLM Provider 都实现统一的 _query 方法
class BaseLLMProvider(BaseProvider):
    def _query(self, prompt: str, model: str = None,
               max_tokens: int = 1024,
               structured_output_format: dict = None) -> dict:
        """
        统一查询接口
        :param prompt: 用户提示词
        :param model: 模型名称
        :param max_tokens: 最大输出 Token 数
        :param structured_output_format: JSON Schema 输出格式
        :return: {"response": 解析后的响应}
        """
        pass
```

### 2.3 结构化输出与 JSON Schema

Keep 在 AI 建议中大量使用 JSON Schema 来约束 LLM 输出：

```python
# AI 事件聚类的输出 Schema
incident_suggestion_schema = {
    "type": "object",
    "properties": {
        "incidents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "incident_name": {"type": "string"},
                    "alerts": {
                        "type": "array",
                        "items": {"type": "integer"}  # 告警索引
                    },
                    "reasoning": {"type": "string"},      # 聚类理由
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "warning", "info", "low"]
                    },
                    "recommended_actions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence_score": {"type": "number"},  # 置信度 0-1
                    "confidence_explanation": {"type": "string"}
                }
            }
        }
    }
}
```

**设计优势**：
- 输出格式完全可控，无需后处理
- 包含置信度分数，支持质量评估
- 包含推理过程，提高可解释性

---

## 3. AI 助手机制分析

### 3.1 AI 工作流构建助手

#### 架构概览

Keep 使用 [CopilotKit](https://www.copilotkit.ai/) 框架实现 AI 工作流构建助手，采用"人在回路中"（Human-in-the-Loop）范式。

#### 后端路由

```typescript
// 文件：keep-ui/app/api/copilotkit/route.ts
import { CopilotRuntime, OpenAIAdapter } from "@copilotkit/runtime";
import OpenAI from "openai";

export const POST = async (req: Request) => {
    const openai = new OpenAI({
        apiKey: process.env.OPEN_AI_API_KEY,
        organization: process.env.OPEN_AI_ORGANIZATION_ID,
    });
    const copilotKit = new CopilotRuntime();
    return copilotKit.response(req, new OpenAIAdapter({
        openai,
        model: process.env.OPENAI_MODEL_NAME,
    }));
};
```

#### 前端交互组件

```typescript
// 文件：keep-ui/features/workflows/ai-assistant/ui/WorkflowBuilderChat.tsx
import { CopilotChat, useCopilotAction, useCopilotReadable } from "@copilotkit/react-core";

function WorkflowBuilderChat() {
    // 1. 向 AI 暴露当前工作流上下文
    useCopilotReadable({
        description: "当前工作流配置",
        value: workflowConfig,
    });

    // 2. 注册 AI 可执行的操作
    useCopilotAction({
        name: "addWorkflowStep",
        description: "向工作流添加新步骤",
        handler: (params) => { /* 添加步骤逻辑 */ },
    });

    // 3. 渲染对话界面
    return <CopilotChat instructions="..." />;
}
```

**工作流程**：
1. 用户通过自然语言描述需求（如"当 CPU 使用率超过 90% 时发送 Slack 通知"）
2. AI 理解意图并提出工作流变更建议
3. 用户审核并确认变更
4. 系统执行变更，更新工作流配置

### 3.2 AI 事件助手

AI 事件助手嵌入在事件详情页中，提供上下文感知的智能分析：

**功能特点**：
- **全上下文注入**：将事件的告警列表、描述、拓扑关系注入 AI 上下文
- **交互式分析**：用户可以与 AI 对话，询问事件原因、影响范围等
- **操作执行**：AI 可通过 Provider 执行操作（如查询日志、执行命令）

### 3.3 AI 半自动关联

这是 Keep 中 LLM 最核心的应用场景——智能告警聚类：

#### 工作流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  用户选择     │    │  系统构建     │    │  OpenAI      │    │  用户审核     │
│  5-50条告警   │───►│  提示词+拓扑  │───►│  智能聚类    │───►│  确认/拒绝    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                           ┌───────┴───────┐
                                                           │  创建事件      │
                                                           │  存储反馈      │
                                                           └───────────────┘
```

#### API 端点

```python
# 文件：keep/api/routes/incidents.py
@router.post("/ai/suggest")
def create_with_ai(alerts_fingerprints: list[str]):
    # 1. 获取告警详情
    alerts_dto = convert_db_alerts_to_dto_alerts(alerts)

    # 2. 获取服务拓扑数据
    topology_data = TopologiesService.get_all_topology_data(tenant_id, session)

    # 3. 调用 AI 建议引擎
    suggestion = suggestion_bl.suggest_incidents(
        alerts_dto=alerts_dto,
        topology_data=topology_data,
        user_id=user.email
    )
    return suggestion

@router.post("/ai/{suggestion_id}/commit")
def commit_ai_suggestion(suggestion_id: str, accepted_incidents: list):
    # 用户确认后创建事件
    suggestion_bl.commit_incidents(suggestion_id, accepted_incidents)
```

#### 缓存机制

```python
# 文件：keep/api/bl/ai_suggestion_bl.py
def add_suggestion(self, inputs, output, model_name, suggestion_type):
    # 使用输入内容的哈希值作为缓存键
    input_hash = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()

    # 检查是否已有相同输入的建议
    existing = db.query(AISuggestion).filter(
        AISuggestion.input_hash == input_hash
    ).first()

    if existing:
        return existing  # 返回缓存结果，避免重复调用 LLM

    # 保存新建议
    suggestion = AISuggestion(
        input=inputs,
        input_hash=input_hash,
        output=output,
        model_name=model_name,
        suggestion_type=suggestion_type
    )
    db.add(suggestion)
```

### 3.4 AI 事件报告生成

```python
# 文件：keep/api/bl/incident_reports.py
class IncidentReportsBl:
    def generate_report(self):
        # 1. 获取近期事件数据（最多40条）
        incidents = get_last_incidents(limit=40)

        # 2. 构建精简的事件摘要
        incidents_json = [
            {
                "incident_id": str(inc.id),
                "name": inc.user_generated_name or inc.ai_generated_name,
                "summary": inc.user_summary or inc.generated_summary,
                "severity": inc.severity,
                "services": inc.services
            }
            for inc in incidents
        ]

        # 3. 调用 OpenAI 分析根因
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            seed=1239,  # 固定种子保证可复现性（任意固定值均可）
            response_format={"type": "json_schema", "json_schema": {
                "properties": {
                    "most_frequent_reasons": {
                        "type": "object"  # 原因 -> 事件ID列表
                    }
                }
            }},
            messages=[
                {"role": "system", "content": "分析事件数据，提取前6个最常见根因"},
                {"role": "user", "content": json.dumps(incidents_json)}
            ]
        )

        # 4. 补充统计指标
        report = {
            "root_causes": response,
            "mttd": calculate_mttd(incidents),  # 平均检测时间
            "mttr": calculate_mttr(incidents),  # 平均解决时间
            "severity_breakdown": calculate_severity(incidents),
        }
        return report
```

---

## 4. AI 关联引擎机制分析

### 4.1 告警处理流水线

Keep 的告警处理遵循严格的流水线架构：

```
┌────────────┐
│ 告警源接入  │  Provider（Prometheus, Grafana, CloudWatch 等）
└─────┬──────┘
      ▼
┌────────────┐
│ 预处理格式化│  统一告警格式
└─────┬──────┘
      ▼
┌────────────┐
│ 维护窗口检查│  过滤维护期间告警
└─────┬──────┘
      ▼
┌────────────┐
│ 告警去重    │  AlertDeduplicator 基于指纹去重
└─────┬──────┘
      ▼
┌────────────┐
│ 数据库持久化│  保存告警和审计日志
└─────┬──────┘
      ▼
┌────────────┐
│ 字段索引    │  更新可用字段列表（用于未来去重和检索）
└─────┬──────┘
      ▼
┌────────────┐
│ ES 索引     │  推送至 Elasticsearch
└─────┬──────┘
      ▼
┌────────────┐
│ 工作流触发  │  根据告警触发自动化工作流
└─────┬──────┘
      ▼
┌────────────┐
│ 规则引擎    │  CEL 规则评估 → 告警分组 → 事件创建/更新
│ 关联处理    │  （KEEP_CORRELATION_ENABLED=true 时启用）
└─────┬──────┘
      ▼
┌────────────┐
│ 客户端通知  │  通过 Pusher 推送 UI 更新
└────────────┘
```

#### 核心代码

```python
# 文件：keep/api/tasks/process_event_task.py
async def process_event(
    ctx, tenant_id, provider_id, provider_type, fingerprint, api_key_name, trace_id
):
    with tracer.start_as_current_span("process_event"):
        # 步骤 1-6：预处理、去重、持久化...

        # 步骤 7：规则引擎关联
        if os.environ.get("KEEP_CORRELATION_ENABLED", "true") == "true":
            rules_engine = RulesEngine(tenant_id)
            rules_engine.run_rules(events)  # 评估所有关联规则
```

### 4.2 基于规则的关联引擎

规则引擎是 Keep 关联体系的确定性基础层：

#### CEL 规则评估

```python
# 文件：keep/rulesengine/rulesengine.py
class RulesEngine:
    def run_rules(self, events: list[AlertDto]):
        rules = get_rules_db(tenant_id=self.tenant_id)

        for rule in rules:
            for event in events:
                # 使用 CEL 表达式评估规则
                # 例如: source == "prometheus" && severity == "critical"
                if self._evaluate_cel_expression(rule.definition, event):
                    self._handle_rule_match(rule, event)

    def _handle_rule_match(self, rule, event):
        # 1. 计算规则指纹（用于分组）
        fingerprint = self._calculate_rule_fingerprint(rule, event)

        # 2. 获取或创建事件
        incident = self._get_or_create_incident(rule, fingerprint)

        # 3. 关联告警到事件
        self._assign_alert_to_incident(event, incident)

        # 4. 检查阈值（是否需要显示事件）
        if incident.alerts_count >= rule.threshold:
            incident.is_visible = True
```

#### 分组策略

```python
# 分组条件支持灵活配置
grouping_criteria = ["labels.host", "labels.cluster", "service"]

# 根据分组条件计算指纹
def _calculate_rule_fingerprint(self, rule, event):
    fingerprint_parts = [str(rule.id)]
    for criteria in rule.grouping_criteria:
        value = get_nested_attribute(event, criteria)
        fingerprint_parts.append(str(value))
    return hashlib.sha256(":".join(fingerprint_parts).encode()).hexdigest()
```

#### 事件创建模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `create_on: "any"` | 任一子规则匹配即触发 | 单一条件触发 |
| `create_on: "all"` | 所有子规则都匹配才触发 | 多条件组合 |
| `require_approve: true` | 需人工审批 | 高敏感度事件 |
| `threshold: N` | 累积 N 条告警后显示 | 批量告警聚合 |

#### 事件命名模板

```python
# 支持变量替换的事件命名
# 模板示例: "{{severity}} alert on {{labels.host}}"
incident_name = rule.name_template.format(**event.dict())
```

### 4.3 基于 Transformer 的外部 AI 关联

这是 Keep 的高级 AI 关联机制，使用 Transformer 模型进行自学习关联：

#### 架构设计

```
┌────────────────────────────────────────────────┐
│                  Keep 平台                      │
│                                                 │
│  ┌───────────────┐    ┌──────────────────────┐ │
│  │ 告警处理流水线 │    │ AI 配置管理          │ │
│  │               │    │ - 算法注册            │ │
│  │ 告警 ──► 规则 │    │ - 参数配置            │ │
│  │         引擎  │    │ - 反馈收集            │ │
│  └───────┬───────┘    └──────────┬───────────┘ │
│          │                        │              │
│          │  历史数据               │  配置/API    │
│          ▼                        ▼              │
│  ┌─────────────────────────────────────────────┐│
│  │          External AI Service API             ││
│  │  POST /remind_about_the_client               ││
│  └────────────────────┬────────────────────────┘│
└───────────────────────┼─────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│           外部 Transformer 服务                    │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐ │
│  │ 数据提取  │  │ 模型训练  │  │ 推理/关联       │ │
│  │ (通过API) │  │ (Epochs)  │  │ (阈值过滤)      │ │
│  └──────────┘  └──────────┘  └─────────────────┘ │
└───────────────────────────────────────────────────┘
```

#### 外部 AI 算法注册

```python
# 文件：keep/api/models/db/ai_external.py
external_ai_transformers = ExternalAI(
    id="external_ai_transformers",
    algorithm_name="Transformers Correlation",
    description="基于 Transformer 模型的告警聚类算法，自动将新告警关联到已有事件或创建新事件",
    api_url=os.environ.get("KEEP_EXTERNAL_AI_TRANSFORMERS_URL"),
    api_key=os.environ.get("KEEP_EXTERNAL_AI_TRANSFORMERS_API_KEY"),
    # 可配置参数
    configuration_default={
        "settings": {
            "model_accuracy_threshold": 0.9,    # 模型准确度阈值
            "correlation_threshold": 0.9,        # 关联阈值
            "train_epochs": 1,                   # 训练轮数（防止过拟合）
            "create_new_incidents": True,         # 是否自动创建新事件
            "enabled": True                       # 是否启用
        }
    }
)
```

#### 无状态微服务通信

```python
# 文件：keep/api/models/ai_external.py
class ExternalAIDto:
    def remind_about_the_client(self, tenant_id, api_key):
        """
        定期提醒外部 AI 服务当前租户的存在
        外部 AI 服务设计为无状态，需要 Keep 主动推送租户信息
        """
        if self._should_remind():  # 30秒节流
            try:
                requests.post(
                    f"{self.api_url}/remind_about_the_client",
                    json={
                        "tenant_id": tenant_id,
                        "api_key": api_key,
                        "back_api_key": self.api_key,
                        "keep_api_url": os.environ.get("KEEP_API_URL")
                    },
                    timeout=0.5  # 非阻塞调用
                )
            except Exception:
                pass  # 静默失败，不影响主流程
```

### 4.4 历史数据训练机制

#### 训练数据来源

Transformer 关联模型的训练数据来自租户的历史告警-事件关联记录：

```
训练数据结构:
┌─────────────────────────────────────┐
│ 历史告警 A ──► 事件 1               │
│ 历史告警 B ──► 事件 1               │
│ 历史告警 C ──► 事件 2               │
│ 历史告警 D ──► 事件 2               │
│ ...                                  │
│ 30% 数据用于验证集                   │
│ 70% 数据用于训练集                   │
└─────────────────────────────────────┘
```

#### 训练流程

1. **数据收集**：外部 AI 服务通过 Keep API 拉取租户的历史告警和事件关联数据
2. **数据分割**：30% 作为验证集，70% 作为训练集
3. **模型训练**：使用 Transformer 架构进行训练，可配置 epoch 数（1-20）
4. **准确度验证**：使用验证集评估模型准确度，与阈值比较
5. **在线推理**：模型训练完成后，对新告警进行分类
6. **关联决策**：
   - 相似度 > `correlation_threshold` → 关联到已有事件
   - 相似度 < `correlation_threshold` 且 `create_new_incidents=true` → 创建新事件

#### 可配置参数详解

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `model_accuracy_threshold` | 0.3-0.99 | 0.9 | 模型准确度达到此阈值后才启用 |
| `correlation_threshold` | 0.3-0.99 | 0.9 | 告警与事件的相似度阈值 |
| `train_epochs` | 1-20 | 1 | 训练轮数，默认1轮避免过拟合 |
| `create_new_incidents` | bool | true | 是否为无法关联的告警创建新事件 |
| `enabled` | bool | true | 是否启用此算法 |

#### 反馈闭环

```python
# 文件：keep/api/models/db/ai_suggestion.py
class AIFeedback(SQLModel):
    """存储用户对 AI 建议的反馈"""
    suggestion_id: str       # 关联的 AI 建议
    user_id: str             # 反馈用户
    rating: int              # 评分
    comment: str             # 评论
    is_accepted: bool        # 是否接受建议
    changes_made: dict       # 用户做了哪些修改

# 反馈数据可用于：
# 1. 评估 AI 建议质量
# 2. 微调模型参数
# 3. 改进提示词工程
# 4. 生成训练数据
```

---

## 5. 核心代码解析

### 5.1 AI 建议业务逻辑层

```
文件：keep/api/bl/ai_suggestion_bl.py

核心类：AISuggestionBl

┌─────────────────────────────────────┐
│          AISuggestionBl             │
├─────────────────────────────────────┤
│ + suggest_incidents()               │ ── 调用 OpenAI 生成事件聚类建议
│ + add_suggestion()                  │ ── 缓存建议（基于输入哈希）
│ + add_feedback()                    │ ── 存储用户反馈
│ + get_feedback()                    │ ── 获取反馈历史
│ + commit_incidents()                │ ── 根据用户确认创建事件
└─────────────────────────────────────┘
```

**关键流程**：
1. 接收告警列表和拓扑数据
2. 计算输入哈希，检查是否有缓存的建议
3. 构建系统提示词（分析维度指导）和用户提示词（告警+拓扑数据）
4. 调用 OpenAI API，获取结构化 JSON 响应
5. 缓存结果，返回给前端展示
6. 用户确认后，创建事件并存储反馈

### 5.2 告警处理流水线代码

```
文件：keep/api/tasks/process_event_task.py

核心函数：process_event()

该函数使用 OpenTelemetry 进行全链路追踪，每个步骤都有独立的 Span：
- pre_format_event
- provider_format_event
- check_maintenance_windows
- deduplicate_alert
- save_to_db
- upsert_alert_fields
- index_to_elasticsearch
- run_workflows
- run_rules_engine（关联引擎入口）
- notify_client
```

### 5.3 规则引擎代码

```
文件：keep/rulesengine/rulesengine.py

核心类：RulesEngine

关键方法：
- run_rules(events)          ── 主入口，遍历规则和事件
- _evaluate_cel(rule, event) ── CEL 表达式求值
- _calculate_fingerprint()   ── 计算分组指纹
- _get_or_create_incident()  ── 获取或创建事件（带重试）
- _assign_alert()            ── 关联告警到事件
- _check_threshold()         ── 检查阈值条件
- _resolve_incidents()       ── 自动解决恢复的事件
```

---

## 6. 如何在其他项目中复用

### 6.1 LLM 集成通用方案

#### 方案一：直接 OpenAI API 集成

适用场景：快速原型、简单的 AI 功能

```python
# 步骤 1：安装依赖
# pip install openai

# 步骤 2：创建 LLM 服务层
import openai
import json
import os

class LLMService:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

    def analyze_with_schema(self, system_prompt: str, user_data: str,
                           output_schema: dict) -> dict:
        """
        使用 JSON Schema 约束的 LLM 分析
        复用 Keep 的核心模式：低温度 + 结构化输出
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": output_schema
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_data}
            ]
        )
        return json.loads(response.choices[0].message.content)
```

#### 方案二：多 Provider 抽象层

适用场景：需要支持多种 LLM 后端的项目

```python
# 步骤 1：定义抽象接口
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def query(self, prompt: str, model: str = None,
              structured_output: dict = None) -> dict:
        pass

# 步骤 2：实现具体 Provider
class OpenAIProvider(BaseLLMProvider):
    def query(self, prompt, model=None, structured_output=None):
        # OpenAI 实现
        pass

class OllamaProvider(BaseLLMProvider):
    def query(self, prompt, model=None, structured_output=None):
        # 本地 Ollama 实现
        pass

# 步骤 3：Provider 工厂
class LLMProviderFactory:
    _providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def create(cls, provider_type: str, config: dict) -> BaseLLMProvider:
        return cls._providers[provider_type](config)
```

### 6.2 AI 助手集成方案

#### 方案一：CopilotKit 集成（推荐用于 React 项目）

```typescript
// 步骤 1：安装依赖
// npm install @copilotkit/react-core @copilotkit/runtime openai

// 步骤 2：创建后端路由 (Next.js API Route)
// app/api/copilotkit/route.ts
import { CopilotRuntime, OpenAIAdapter } from "@copilotkit/runtime";
import OpenAI from "openai";

export const POST = async (req: Request) => {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const runtime = new CopilotRuntime();
    return runtime.response(req, new OpenAIAdapter({ openai }));
};

// 步骤 3：前端组件
// components/AIAssistant.tsx
import { CopilotChat, useCopilotReadable, useCopilotAction } from "@copilotkit/react-core";

function AIAssistant({ context }) {
    // 注入业务上下文
    useCopilotReadable({ description: "业务数据", value: context });

    // 注册可执行操作
    useCopilotAction({
        name: "executeAction",
        description: "执行业务操作",
        parameters: [{ name: "action", type: "string" }],
        handler: async ({ action }) => { /* 执行逻辑 */ },
    });

    return <CopilotChat instructions="你是一个专业的业务助手..." />;
}
```

#### 方案二：自建 AI 助手（无框架依赖）

```python
# 基于 Keep 的模式，构建简单的 AI 助手后端
class AIAssistant:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.actions = {}  # 注册可执行操作

    def register_action(self, name: str, handler, description: str):
        """注册 AI 可调用的操作"""
        self.actions[name] = {
            "handler": handler,
            "description": description
        }

    def chat(self, user_message: str, context: dict) -> str:
        """
        上下文感知的 AI 对话
        复用 Keep 的模式：业务上下文 + 可执行操作
        """
        system_prompt = f"""
        你是一个业务助手。

        当前上下文：
        {json.dumps(context, ensure_ascii=False)}

        可用操作：
        {json.dumps({k: v["description"] for k, v in self.actions.items()},
                    ensure_ascii=False)}
        """
        response = self.llm.analyze_with_schema(
            system_prompt=system_prompt,
            user_data=user_message,
            output_schema={...}
        )
        return response
```

### 6.3 AI 关联引擎集成方案

#### 方案一：基于规则的关联引擎

```python
# 复用 Keep 的 CEL 规则引擎模式

# 步骤 1：安装 CEL 库
# pip install cel-python

# 步骤 2：实现规则引擎
import cel
import hashlib

class CorrelationRule:
    """关联规则定义"""
    def __init__(self, name, cel_expression, grouping_criteria, threshold=1):
        self.name = name
        self.cel_expression = cel_expression
        self.grouping_criteria = grouping_criteria
        self.threshold = threshold

class RuleBasedCorrelationEngine:
    """
    基于 Keep 的规则引擎模式
    使用 CEL 表达式评估规则，使用分组条件聚合事件
    """
    def __init__(self):
        self.rules = []
        self.incidents = {}  # fingerprint -> incident

    def add_rule(self, rule: CorrelationRule):
        self.rules.append(rule)

    def process_event(self, event: dict):
        for rule in self.rules:
            if self._evaluate_cel(rule.cel_expression, event):
                fingerprint = self._calculate_fingerprint(rule, event)
                incident = self._get_or_create_incident(rule, fingerprint)
                incident["events"].append(event)

                if len(incident["events"]) >= rule.threshold:
                    incident["visible"] = True
                    self._notify_incident_change(incident)

    def _evaluate_cel(self, expression: str, event: dict) -> bool:
        env = cel.Environment()
        ast = env.compile(expression)
        prog = env.program(ast)
        return prog.evaluate(event)

    def _calculate_fingerprint(self, rule, event):
        parts = [str(rule.name)]
        for criteria in rule.grouping_criteria:
            parts.append(str(event.get(criteria, "")))
        return hashlib.sha256(":".join(parts).encode()).hexdigest()

    def _get_or_create_incident(self, rule, fingerprint):
        if fingerprint not in self.incidents:
            self.incidents[fingerprint] = {
                "name": rule.name,
                "fingerprint": fingerprint,
                "events": [],
                "visible": False,
                "created_at": datetime.utcnow()
            }
        return self.incidents[fingerprint]
```

#### 方案二：LLM 驱动的智能关联

```python
# 复用 Keep 的 AI 建议模式

class AICorrelationEngine:
    """
    基于 Keep 的 AI 关联引擎模式
    使用 LLM 智能聚类事件
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.suggestion_cache = {}  # 输入哈希 -> 建议缓存

    def suggest_correlations(self, events: list[dict],
                            context_data: dict = None) -> dict:
        """
        智能事件关联建议
        复用 Keep 的核心模式：
        1. 构建上下文丰富的提示词
        2. 使用 JSON Schema 约束输出
        3. 缓存避免重复调用
        4. 人工审核确认
        """
        # 1. 缓存检查
        cache_key = hashlib.sha256(
            json.dumps(events, default=str).encode()
        ).hexdigest()
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]

        # 2. 构建提示词
        system_prompt = """
        你是一个智能事件关联分析系统。
        分析以下事件数据，将相关事件聚类为事件组。
        考虑因素：
        - 事件描述和内容的相似性
        - 时间接近性
        - 受影响的系统/服务
        - 可能的根因关联
        """

        user_prompt = f"""
        事件数据：
        {json.dumps(events, ensure_ascii=False, default=str)}

        上下文数据：
        {json.dumps(context_data or {}, ensure_ascii=False, default=str)}
        """

        # 3. 调用 LLM
        result = self.llm.analyze_with_schema(
            system_prompt=system_prompt,
            user_data=user_prompt,
            output_schema={
                "name": "correlation_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "group_name": {"type": "string"},
                                    "event_indices": {
                                        "type": "array",
                                        "items": {"type": "integer"}
                                    },
                                    "reasoning": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "confidence": {"type": "number"}
                                }
                            }
                        }
                    }
                }
            }
        )

        # 4. 缓存结果
        self.suggestion_cache[cache_key] = result
        return result
```

#### 方案三：Transformer 模型自学习关联

```python
# 复用 Keep 的外部 AI 服务模式

class TransformerCorrelationService:
    """
    基于 Keep 的外部 Transformer 关联模式
    使用历史数据训练模型，自动关联新事件
    """
    def __init__(self, config):
        self.accuracy_threshold = config.get("model_accuracy_threshold", 0.9)
        self.correlation_threshold = config.get("correlation_threshold", 0.9)
        self.train_epochs = config.get("train_epochs", 1)
        self.model = None

    def train(self, historical_data: list[dict]):
        """
        使用历史数据训练关联模型

        historical_data 格式:
        [
            {"event": {...}, "incident_id": "inc_001"},
            {"event": {...}, "incident_id": "inc_001"},
            {"event": {...}, "incident_id": "inc_002"},
        ]
        """
        # 1. 数据分割（70% 训练, 30% 验证）
        split_idx = int(len(historical_data) * 0.7)
        train_data = historical_data[:split_idx]
        val_data = historical_data[split_idx:]

        # 2. 特征提取和模型训练
        # 使用 sentence-transformers 等库进行文本嵌入
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. 训练（具体实现取决于你的模型架构）
        for epoch in range(self.train_epochs):
            self._train_epoch(train_data)

        # 4. 验证准确度
        accuracy = self._validate(val_data)
        if accuracy < self.accuracy_threshold:
            raise ValueError(f"模型准确度 {accuracy} 低于阈值 {self.accuracy_threshold}")

    def correlate(self, new_event: dict, existing_incidents: list[dict]) -> dict:
        """
        将新事件关联到已有事件组

        返回:
        - incident_id: 关联到的事件ID（如果匹配）
        - confidence: 关联置信度
        - create_new: 是否需要创建新事件
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        # 计算新事件与所有已有事件的相似度
        new_embedding = self.model.encode(json.dumps(new_event))

        best_match = None
        best_score = 0

        for incident in existing_incidents:
            incident_embedding = self.model.encode(
                json.dumps(incident["summary"])
            )
            score = cosine_similarity(new_embedding, incident_embedding)

            if score > best_score:
                best_score = score
                best_match = incident

        if best_score >= self.correlation_threshold:
            return {
                "incident_id": best_match["id"],
                "confidence": best_score,
                "create_new": False
            }
        else:
            return {
                "incident_id": None,
                "confidence": best_score,
                "create_new": True
            }
```

### 6.4 完整实现路线图

以下是在其他项目中复用 Keep AI 机制的推荐路线图：

#### 阶段一：基础集成（1-2 周）

```
□ 搭建多 Provider LLM 抽象层
  ├── 实现 OpenAI Provider
  ├── 实现 Ollama Provider（本地部署）
  └── 实现 Provider 工厂模式

□ 实现基础 AI 分析功能
  ├── 结构化输出（JSON Schema）
  ├── 输入哈希缓存
  └── 低温度确定性输出
```

#### 阶段二：规则引擎（2-3 周）

```
□ 实现 CEL 规则引擎
  ├── 规则定义和存储
  ├── CEL 表达式评估
  ├── 分组条件和指纹计算
  └── 阈值触发机制

□ 实现事件处理流水线
  ├── 事件预处理和去重
  ├── 规则匹配和事件创建
  ├── 事件命名模板
  └── 事件生命周期管理
```

#### 阶段三：AI 关联引擎（3-4 周）

```
□ 实现 LLM 驱动的智能关联
  ├── 系统提示词工程
  ├── 上下文数据注入（拓扑、依赖关系）
  ├── 置信度评分和解释
  └── 人工审核流程

□ 实现反馈闭环
  ├── AI 建议存储
  ├── 用户反馈收集
  ├── 反馈数据分析
  └── 模型参数优化
```

#### 阶段四：自学习模型（4-6 周）

```
□ 实现 Transformer 关联模型
  ├── 历史数据收集和预处理
  ├── 训练/验证数据分割
  ├── 模型训练流水线
  ├── 准确度验证和阈值控制
  └── 在线推理服务

□ 实现外部 AI 服务架构
  ├── 无状态微服务设计
  ├── 租户隔离和数据安全
  ├── 定期提醒机制
  └── 配置热更新
```

#### 阶段五：AI 助手（2-3 周）

```
□ 集成 AI 助手
  ├── CopilotKit 集成（React 项目）
  ├── 上下文注入机制
  ├── 可执行操作注册
  └── 对话界面实现

□ 实现 AI 报告生成
  ├── 事件摘要生成
  ├── 根因分析
  ├── 统计指标计算（MTTD/MTTR）
  └── 定期报告自动化
```

---

## 附录

### A. 环境变量配置参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_MODEL_NAME` | 使用的模型名称 | gpt-4o |
| `OPEN_AI_ORGANIZATION_ID` | OpenAI 组织 ID | - |
| `KEEP_CORRELATION_ENABLED` | 启用关联引擎 | true |
| `KEEP_EXTERNAL_AI_TRANSFORMERS_URL` | 外部 Transformer 服务 URL | - |
| `KEEP_EXTERNAL_AI_TRANSFORMERS_API_KEY` | 外部 Transformer 服务密钥 | - |
| `EE_ENABLED` | 启用企业版功能 | false |

### B. 数据库模型参考

```
AISuggestion（AI 建议表）
├── id: UUID
├── tenant_id: str
├── input: JSON           # 输入数据
├── input_hash: str       # 输入哈希（用于缓存）
├── output: JSON          # AI 输出结果
├── model_name: str       # 使用的模型
├── suggestion_type: enum # 建议类型
└── created_at: datetime

AIFeedback（AI 反馈表）
├── id: UUID
├── suggestion_id: FK     # 关联建议
├── user_id: str
├── rating: int           # 评分
├── comment: str          # 评论
├── is_accepted: bool     # 是否接受
├── changes_made: JSON    # 修改内容
└── created_at: datetime

ExternalAI（外部 AI 算法表）
├── id: str
├── algorithm_name: str
├── description: str
├── api_url: str
├── api_key: str
└── configuration_default: JSON

ExternalAIConfigAndMetadata（租户 AI 配置表）
├── algorithm_id: FK
├── tenant_id: str
├── settings: JSON        # 当前配置
├── feedback_logs: JSON   # 反馈日志
└── enabled: bool
```

### C. 关键设计模式总结

| 设计模式 | Keep 中的应用 | 复用建议 |
|----------|--------------|----------|
| Provider 模式 | 多 LLM 后端抽象 | 所有需要多后端支持的场景 |
| 输入哈希缓存 | 避免重复 LLM 调用 | 所有 LLM 调用场景 |
| 结构化输出 | JSON Schema 约束 | 需要可靠结构化数据的场景 |
| 人在回路 | AI 建议 + 人工确认 | 高敏感度决策场景 |
| 反馈闭环 | 用户反馈收集和利用 | 持续优化 AI 输出 |
| 无状态微服务 | 外部 AI 服务通信 | 分布式 AI 推理服务 |
| CEL 规则引擎 | 确定性告警关联 | 灵活的事件匹配 |
| 分组指纹 | 事件聚合分组 | 事件去重和聚类 |
