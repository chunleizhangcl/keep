# Keep 告警去重算法 - 独立可复用版本

这个目录包含了从 Keep 项目中剥离的告警去重算法的独立实现。

## 目录结构

```
deduplication/
├── README.md              # 本文件
├── alert_deduplicator.py  # 去重算法核心实现
└── requirements.txt       # 依赖（可选，用于 Redis 后端）
```

## 快速开始

### 基本使用

```python
from alert_deduplicator import AlertDeduplicator, Alert

# 创建去重器
deduplicator = AlertDeduplicator(
    fingerprint_fields=["service", "error_code"],  # 用于识别同类告警的字段
    ignore_fields=["timestamp", "cpu_percent"]      # 比较时忽略的字段
)

# 处理第一个告警
alert1 = Alert(
    id="1",
    name="HighCPU",
    data={
        "service": "api-gateway",
        "error_code": "E001",
        "cpu_percent": 85,
        "timestamp": "2024-01-01T10:00:00Z"
    }
)
result1 = deduplicator.process_alert(alert1)
print(f"告警1: 新告警")  # 第一个告警是新的

# 处理第二个告警（相同的 service 和 error_code）
alert2 = Alert(
    id="2",
    name="HighCPU",
    data={
        "service": "api-gateway",
        "error_code": "E001",
        "cpu_percent": 90,  # CPU 值变了，但这是忽略字段
        "timestamp": "2024-01-01T10:01:00Z"
    }
)
result2 = deduplicator.process_alert(alert2)

if result2.is_full_duplicate:
    print("告警2: 完全重复，应该丢弃")
elif result2.is_partial_duplicate:
    print("告警2: 部分重复，应该更新")
```

### 使用预定义的 Provider 配置

```python
from alert_deduplicator import create_deduplicator_for_provider

# 使用 Datadog 的默认配置
datadog_dedup = create_deduplicator_for_provider("datadog")

# 使用 Prometheus 的默认配置
prometheus_dedup = create_deduplicator_for_provider("prometheus")
```

### 分布式场景（Redis 后端）

```python
from alert_deduplicator import AlertDeduplicator, RedisHashStore

# 创建 Redis 存储后端
redis_store = RedisHashStore(
    redis_url="redis://localhost:6379",
    ttl=86400,  # 24小时过期
    key_prefix="my_app:alert:hash:"
)

# 使用 Redis 后端创建去重器
deduplicator = AlertDeduplicator(
    fingerprint_fields=["service", "alert_name"],
    hash_store=redis_store
)
```

## 算法说明

### 两阶段去重

1. **Fingerprint（指纹）计算**
   - 使用指定的 `fingerprint_fields` 字段计算 SHA256 哈希
   - 相同指纹的告警被视为同一类告警
   - 如果没有指定字段，使用告警名称作为指纹

2. **Hash（哈希）比较**
   - 计算告警完整内容的 SHA256 哈希（排除 `ignore_fields`）
   - 相同指纹 + 相同哈希 = 完全重复（丢弃）
   - 相同指纹 + 不同哈希 = 部分重复（更新）
   - 不同指纹 = 新告警（保存）

### 去重结果

| 条件 | is_full_duplicate | is_partial_duplicate | 建议操作 |
|------|-------------------|---------------------|----------|
| 相同 fingerprint + 相同 hash | True | False | 丢弃 |
| 相同 fingerprint + 不同 hash | False | True | 更新 |
| 不同 fingerprint | False | False | 保存 |

## API 参考

### Alert 类

```python
@dataclass
class Alert:
    id: str                          # 告警ID
    name: str                        # 告警名称
    fingerprint: Optional[str]       # 指纹（自动计算）
    alert_hash: Optional[str]        # 哈希（自动计算）
    is_full_duplicate: bool          # 是否完全重复
    is_partial_duplicate: bool       # 是否部分重复
    data: Dict[str, Any]             # 告警数据
```

### AlertDeduplicator 类

```python
class AlertDeduplicator:
    def __init__(
        self,
        fingerprint_fields: List[str] = None,  # 指纹计算字段
        ignore_fields: List[str] = None,       # 哈希忽略字段
        hash_store: HashStoreBackend = None    # 存储后端
    ): ...
    
    def calculate_fingerprint(self, alert: Alert) -> str: ...
    def calculate_hash(self, alert: Alert) -> str: ...
    def process_alert(self, alert: Alert) -> Alert: ...
    def process_alerts(self, alerts: List[Alert]) -> List[Alert]: ...
    def reset(self) -> None: ...
```

## 依赖

- Python 3.7+
- （可选）redis: 用于 Redis 后端

```bash
pip install redis  # 如果需要 Redis 后端
```

## 更多信息

详细的算法说明请参考：
- [Keep 文档 - Fingerprints](../../docs/overview/fingerprints.mdx)
- [Keep 文档 - Deduplication](../../docs/overview/deduplication.mdx)
- [Keep 文档 - Fingerprint与去重算法详解（中文）](../../docs/overview/fingerprint-and-deduplication-guide-zh.mdx)
