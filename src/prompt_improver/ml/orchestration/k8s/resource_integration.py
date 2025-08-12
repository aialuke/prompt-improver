"""
Kubernetes Resource Integration for ML Orchestrator.

Provides integration with Kubernetes resource management including:
- Resource quotas and limits
- Horizontal Pod Autoscaler (HPA) integration
- Vertical Pod Autoscaler (VPA) support
- Node resource discovery
- Pod resource requests/limits awareness
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
try:
    from kubernetes import client, config
    from kubernetes.client.rest import api_exception
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    client = None
    config = None
    api_exception = Exception
logger = logging.getLogger(__name__)

@dataclass
class NodeResourceInfo:
    """Information about node resources."""
    node_name: str
    allocatable_cpu: str
    allocatable_memory: str
    allocatable_gpu: str
    capacity_cpu: str
    capacity_memory: str
    capacity_gpu: str
    labels: dict[str, str]
    taints: list[dict[str, str]]

@dataclass
class PodResourceInfo:
    """Information about pod resource requests/limits."""
    pod_name: str
    namespace: str
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str
    gpu_request: str

@dataclass
class ResourceQuotaInfo:
    """Information about namespace resource quotas."""
    namespace: str
    quota_name: str
    cpu_hard: str
    memory_hard: str
    gpu_hard: str
    cpu_used: str
    memory_used: str
    gpu_used: str

# KubernetesResourceManager class removed - functionality consolidated into UnifiedOrchestrationManager
