"""
Keep Alert Deduplication Algorithm - 独立可复用版本
Keep Alert Deduplication Algorithm - Standalone Reusable Version

该模块提供了一个通用的告警去重算法，可以在任何 Python 项目中使用。
This module provides a generic alert deduplication algorithm that can be used in any Python project.

使用方法 / Usage:
    from alert_deduplicator import AlertDeduplicator, Alert
    
    # 创建去重器 / Create deduplicator
    deduplicator = AlertDeduplicator(
        fingerprint_fields=["service", "error_code"],
        ignore_fields=["timestamp", "cpu_percent"]
    )
    
    # 处理告警 / Process alerts
    alert = Alert(id="1", name="HighCPU", data={"service": "api", "cpu": 95})
    result = deduplicator.process_alert(alert)
    
    if result.is_full_duplicate:
        print("完全重复，应该丢弃 / Full duplicate, should be discarded")
    elif result.is_partial_duplicate:
        print("部分重复，应该更新 / Partial duplicate, should be updated")
    else:
        print("新告警 / New alert")
"""

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class Alert:
    """
    告警数据模型 / Alert Data Model
    
    Attributes:
        id: 告警ID / Alert ID
        name: 告警名称 / Alert name
        fingerprint: 告警指纹（可自动计算）/ Alert fingerprint (auto-calculated)
        alert_hash: 告警哈希（可自动计算）/ Alert hash (auto-calculated)
        is_full_duplicate: 是否完全重复 / Whether it's a full duplicate
        is_partial_duplicate: 是否部分重复 / Whether it's a partial duplicate
        data: 告警数据 / Alert data
    """
    id: str
    name: str
    fingerprint: Optional[str] = None
    alert_hash: Optional[str] = None
    is_full_duplicate: bool = False
    is_partial_duplicate: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 / Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "alert_hash": self.alert_hash,
            "is_full_duplicate": self.is_full_duplicate,
            "is_partial_duplicate": self.is_partial_duplicate,
            "data": self.data
        }


@dataclass
class DeduplicationRule:
    """
    去重规则 / Deduplication Rule
    
    Attributes:
        name: 规则名称 / Rule name
        fingerprint_fields: 用于计算 fingerprint 的字段 / Fields for fingerprint calculation
        ignore_fields: 计算 hash 时忽略的字段 / Fields to ignore when calculating hash
        full_deduplication: 是否启用完全去重 / Whether to enable full deduplication
    """
    name: str
    fingerprint_fields: List[str]
    ignore_fields: List[str]
    full_deduplication: bool = True


class HashStoreBackend(ABC):
    """
    哈希存储后端抽象类 / Abstract Hash Store Backend
    
    可以实现不同的后端：内存、Redis、数据库等
    Can implement different backends: memory, Redis, database, etc.
    """
    
    @abstractmethod
    def get(self, fingerprint: str) -> Optional[str]:
        """获取指纹对应的哈希 / Get hash for fingerprint"""
        pass
    
    @abstractmethod
    def set(self, fingerprint: str, hash_value: str) -> None:
        """设置指纹对应的哈希 / Set hash for fingerprint"""
        pass
    
    @abstractmethod
    def delete(self, fingerprint: str) -> None:
        """删除指纹对应的哈希 / Delete hash for fingerprint"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有哈希 / Clear all hashes"""
        pass


class MemoryHashStore(HashStoreBackend):
    """
    内存哈希存储 / In-Memory Hash Store
    
    适用于单进程场景
    Suitable for single-process scenarios
    """
    
    def __init__(self):
        self._store: Dict[str, str] = {}
    
    def get(self, fingerprint: str) -> Optional[str]:
        return self._store.get(fingerprint)
    
    def set(self, fingerprint: str, hash_value: str) -> None:
        self._store[fingerprint] = hash_value
    
    def delete(self, fingerprint: str) -> None:
        self._store.pop(fingerprint, None)
    
    def clear(self) -> None:
        self._store.clear()
    
    def __len__(self) -> int:
        return len(self._store)


class AlertDeduplicator:
    """
    告警去重器 / Alert Deduplicator
    
    实现了 Keep 的两阶段去重算法：
    1. Fingerprint 阶段：计算告警的唯一标识
    2. Hash 阶段：计算告警的完整哈希进行精确比较
    
    Implements Keep's two-stage deduplication algorithm:
    1. Fingerprint stage: Calculate unique identifier for alert
    2. Hash stage: Calculate full hash for exact comparison
    
    Example:
        >>> deduplicator = AlertDeduplicator(fingerprint_fields=["service", "error_code"])
        >>> alert1 = Alert(id="1", name="HighCPU", data={"service": "api", "error_code": "E001"})
        >>> result1 = deduplicator.process_alert(alert1)
        >>> alert2 = Alert(id="2", name="HighCPU", data={"service": "api", "error_code": "E001"})
        >>> result2 = deduplicator.process_alert(alert2)
        >>> assert result2.is_full_duplicate == True
    """
    
    def __init__(
        self,
        fingerprint_fields: Optional[List[str]] = None,
        ignore_fields: Optional[List[str]] = None,
        hash_store: Optional[HashStoreBackend] = None,
    ):
        """
        初始化去重器 / Initialize deduplicator
        
        Args:
            fingerprint_fields: 用于计算 fingerprint 的字段列表
                               Fields for fingerprint calculation
            ignore_fields: 计算 hash 时忽略的字段列表（默认忽略 timestamp, lastReceived）
                          Fields to ignore when calculating hash
            hash_store: 哈希存储后端（默认使用内存存储）
                       Hash store backend (defaults to memory store)
        """
        self.fingerprint_fields = fingerprint_fields or []
        self.ignore_fields = ignore_fields or ["timestamp", "lastReceived"]
        self._hash_store = hash_store or MemoryHashStore()
    
    def calculate_fingerprint(self, alert: Alert) -> str:
        """
        计算告警的指纹 / Calculate alert fingerprint
        
        算法说明 / Algorithm:
        1. 如果没有指定字段，使用告警名称的 SHA256 哈希
        2. 如果指定了字段，将这些字段的值拼接后计算 SHA256 哈希
        3. 支持嵌套字段，如 "labels.host"
        
        Args:
            alert: 告警对象 / Alert object
            
        Returns:
            指纹的十六进制字符串（64位）/ Fingerprint hex string (64 chars)
        """
        if not self.fingerprint_fields:
            # 没有指定字段时，使用告警名称
            # Use alert name when no fields specified
            return hashlib.sha256(alert.name.encode()).hexdigest()
        
        fingerprint = hashlib.sha256()
        
        for field_path in self.fingerprint_fields:
            value = self._get_nested_value(alert.data, field_path)
            if value is not None:
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, sort_keys=True)
                fingerprint.update(str(value).encode())
        
        return fingerprint.hexdigest()
    
    def calculate_hash(self, alert: Alert) -> str:
        """
        计算告警的完整哈希 / Calculate full alert hash
        
        算法说明 / Algorithm:
        1. 深拷贝告警数据
        2. 移除忽略字段
        3. 将剩余数据序列化为 JSON（key 排序）
        4. 计算 SHA256 哈希
        
        Args:
            alert: 告警对象 / Alert object
            
        Returns:
            哈希的十六进制字符串（64位）/ Hash hex string (64 chars)
        """
        # 复制数据并移除忽略字段
        # Copy data and remove ignored fields
        data_copy = copy.deepcopy(alert.data)
        for field_path in self.ignore_fields:
            self._remove_nested_value(data_copy, field_path)
        
        # 构建哈希数据（包含告警名称）
        # Build hash data (includes alert name)
        hash_data = {
            "name": alert.name,
            "data": data_copy
        }
        
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True, default=str).encode()
        ).hexdigest()
    
    def process_alert(self, alert: Alert) -> Alert:
        """
        处理告警并进行去重判断 / Process alert and determine deduplication
        
        处理流程 / Process flow:
        1. 计算 fingerprint（如果未提供）
        2. 计算完整 hash
        3. 与历史哈希比较
        4. 设置去重标志
        5. 更新哈希存储
        
        Args:
            alert: 待处理的告警 / Alert to process
            
        Returns:
            处理后的告警（带有去重标志）/ Processed alert with dedup flags
        """
        # 计算 fingerprint
        if alert.fingerprint is None:
            alert.fingerprint = self.calculate_fingerprint(alert)
        
        # 计算 hash
        alert.alert_hash = self.calculate_hash(alert)
        
        # 获取相同 fingerprint 的历史 hash
        # Get historical hash for same fingerprint
        last_hash = self._hash_store.get(alert.fingerprint)
        
        if last_hash is not None and last_hash == alert.alert_hash:
            # 完全重复 / Full duplicate
            alert.is_full_duplicate = True
        elif last_hash is not None:
            # 部分重复（相同 fingerprint，不同 hash）
            # Partial duplicate (same fingerprint, different hash)
            alert.is_partial_duplicate = True
        # 否则为新告警，不做标记
        # Otherwise it's a new alert, no flags set
        
        # 更新 hash 存储
        # Update hash store
        self._hash_store.set(alert.fingerprint, alert.alert_hash)
        
        return alert
    
    def process_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        批量处理告警 / Batch process alerts
        
        Args:
            alerts: 告警列表 / List of alerts
            
        Returns:
            处理后的告警列表 / List of processed alerts
        """
        return [self.process_alert(alert) for alert in alerts]
    
    def reset(self) -> None:
        """
        重置去重器状态 / Reset deduplicator state
        
        清空所有历史哈希记录
        Clears all historical hash records
        """
        self._hash_store.clear()
    
    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """
        获取嵌套字段的值 / Get nested field value
        
        支持点分隔的路径，如 "labels.host"
        Supports dot-separated paths like "labels.host"
        """
        keys = field_path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    
    def _remove_nested_value(self, data: Dict, field_path: str) -> None:
        """
        移除嵌套字段 / Remove nested field
        
        支持点分隔的路径，如 "labels.host"
        Supports dot-separated paths like "labels.host"
        """
        keys = field_path.split(".")
        if len(keys) == 1:
            data.pop(field_path, None)
        else:
            parent = data
            for key in keys[:-1]:
                if isinstance(parent, dict):
                    parent = parent.get(key, {})
                else:
                    return
            if isinstance(parent, dict):
                parent.pop(keys[-1], None)


# ============== Redis 后端实现 / Redis Backend Implementation ==============

class RedisHashStore(HashStoreBackend):
    """
    Redis 哈希存储 / Redis Hash Store
    
    适用于分布式场景
    Suitable for distributed scenarios
    
    需要安装 redis 包：pip install redis
    Requires redis package: pip install redis
    
    Example:
        >>> store = RedisHashStore(redis_url="redis://localhost:6379")
        >>> deduplicator = AlertDeduplicator(hash_store=store)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 86400,
        key_prefix: str = "alert:hash:"
    ):
        """
        初始化 Redis 存储 / Initialize Redis store
        
        Args:
            redis_url: Redis 连接 URL / Redis connection URL
            ttl: 哈希过期时间（秒）/ Hash TTL in seconds
            key_prefix: 键前缀 / Key prefix
        """
        try:
            import redis
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package. Install with: pip install redis")
        
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl
        self.key_prefix = key_prefix
    
    def get(self, fingerprint: str) -> Optional[str]:
        value = self.redis.get(f"{self.key_prefix}{fingerprint}")
        return value.decode() if value else None
    
    def set(self, fingerprint: str, hash_value: str) -> None:
        self.redis.setex(f"{self.key_prefix}{fingerprint}", self.ttl, hash_value)
    
    def delete(self, fingerprint: str) -> None:
        self.redis.delete(f"{self.key_prefix}{fingerprint}")
    
    def clear(self) -> None:
        keys = self.redis.keys(f"{self.key_prefix}*")
        if keys:
            self.redis.delete(*keys)


# ============== 工具函数 / Utility Functions ==============

def create_deduplicator_for_provider(provider_type: str) -> AlertDeduplicator:
    """
    为特定 Provider 创建去重器 / Create deduplicator for specific provider
    
    使用 Keep 中各 Provider 的默认配置
    Uses default configurations from Keep providers
    
    Args:
        provider_type: Provider 类型 / Provider type
        
    Returns:
        配置好的去重器 / Configured deduplicator
    """
    # Keep 中各 Provider 的默认 fingerprint 字段
    # Default fingerprint fields from Keep providers
    provider_configs = {
        "datadog": {
            "fingerprint_fields": ["groups", "monitor_id"],
            "ignore_fields": ["lastReceived"]
        },
        "pagerduty": {
            "fingerprint_fields": ["alert_key"],
            "ignore_fields": ["lastReceived"]
        },
        "prometheus": {
            "fingerprint_fields": ["fingerprint"],
            "ignore_fields": ["lastReceived"]
        },
        "grafana": {
            "fingerprint_fields": ["fingerprint"],
            "ignore_fields": ["lastReceived"]
        },
        "sentry": {
            "fingerprint_fields": ["issue_id"],
            "ignore_fields": ["lastReceived"]
        },
        "default": {
            "fingerprint_fields": [],
            "ignore_fields": ["lastReceived"]
        }
    }
    
    config = provider_configs.get(provider_type.lower(), provider_configs["default"])
    return AlertDeduplicator(
        fingerprint_fields=config["fingerprint_fields"],
        ignore_fields=config["ignore_fields"]
    )


# ============== 使用示例 / Usage Examples ==============

def example_basic_usage():
    """基本使用示例 / Basic usage example"""
    print("=" * 60)
    print("基本使用示例 / Basic Usage Example")
    print("=" * 60)
    
    # 创建去重器
    deduplicator = AlertDeduplicator(
        fingerprint_fields=["service", "error_code"],
        ignore_fields=["timestamp", "cpu_percent"]
    )
    
    # 模拟告警流
    alerts = [
        Alert(id="1", name="HighCPU", data={
            "service": "api-gateway",
            "error_code": "E001",
            "cpu_percent": 85,
            "timestamp": "2024-01-01T10:00:00Z"
        }),
        Alert(id="2", name="HighCPU", data={
            "service": "api-gateway",
            "error_code": "E001",
            "cpu_percent": 90,
            "timestamp": "2024-01-01T10:01:00Z"
        }),
        Alert(id="3", name="HighCPU", data={
            "service": "api-gateway",
            "error_code": "E002",
            "cpu_percent": 95,
            "timestamp": "2024-01-01T10:02:00Z"
        }),
    ]
    
    for alert in alerts:
        processed = deduplicator.process_alert(alert)
        status = "新告警 / New"
        if processed.is_full_duplicate:
            status = "完全重复 / Full Duplicate"
        elif processed.is_partial_duplicate:
            status = "部分重复 / Partial Duplicate"
        
        print(f"\n告警 {alert.id}:")
        print(f"  名称: {alert.name}")
        print(f"  数据: {alert.data}")
        print(f"  状态: {status}")


def example_provider_specific():
    """Provider 特定配置示例 / Provider-specific configuration example"""
    print("\n" + "=" * 60)
    print("Provider 特定配置示例 / Provider-specific Example")
    print("=" * 60)
    
    # 使用 Datadog 配置
    datadog_deduplicator = create_deduplicator_for_provider("datadog")
    print(f"\nDatadog 配置:")
    print(f"  Fingerprint 字段: {datadog_deduplicator.fingerprint_fields}")
    print(f"  忽略字段: {datadog_deduplicator.ignore_fields}")
    
    # 使用 Prometheus 配置
    prometheus_deduplicator = create_deduplicator_for_provider("prometheus")
    print(f"\nPrometheus 配置:")
    print(f"  Fingerprint 字段: {prometheus_deduplicator.fingerprint_fields}")
    print(f"  忽略字段: {prometheus_deduplicator.ignore_fields}")


def example_batch_processing():
    """批量处理示例 / Batch processing example"""
    print("\n" + "=" * 60)
    print("批量处理示例 / Batch Processing Example")
    print("=" * 60)
    
    deduplicator = AlertDeduplicator(fingerprint_fields=["host", "alert_type"])
    
    alerts = [
        Alert(id=str(i), name="Alert", data={
            "host": f"server-{i % 3}",
            "alert_type": "cpu_high",
            "value": 80 + i
        })
        for i in range(10)
    ]
    
    results = deduplicator.process_alerts(alerts)
    
    full_duplicates = sum(1 for a in results if a.is_full_duplicate)
    partial_duplicates = sum(1 for a in results if a.is_partial_duplicate)
    new_alerts = sum(1 for a in results if not a.is_full_duplicate and not a.is_partial_duplicate)
    
    print(f"\n处理了 {len(results)} 条告警:")
    print(f"  新告警: {new_alerts}")
    print(f"  完全重复: {full_duplicates}")
    print(f"  部分重复: {partial_duplicates}")


if __name__ == "__main__":
    example_basic_usage()
    example_provider_specific()
    example_batch_processing()
