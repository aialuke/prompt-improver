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
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Kubernetes client with graceful fallback
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
    labels: Dict[str, str]
    taints: List[Dict[str, str]]

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

class KubernetesResourceManager:
    """Kubernetes resource management integration."""
    
    def __init__(self, namespace: str = "default"):
        """Initialize Kubernetes resource manager."""
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)
        
        # Kubernetes API clients
        self.v1 = None
        self.apps_v1 = None
        self.autoscaling_v2 = None
        self.metrics_v1beta1 = None
        
        # Initialize if Kubernetes is available
        if KUBERNETES_AVAILABLE:
            self._initialize_k8s_clients()
        else:
            self.logger.warning("Kubernetes client not available - running in standalone mode")
    
    def _initialize_k8s_clients(self) -> None:
        """Initialize Kubernetes API clients."""
        try:
            # Try to load in-cluster config first, then local config
            try:
                config.load_incluster_config()
                self.logger.info("Loaded in-cluster Kubernetes configuration")
            except config.ConfigException:
                config.load_kube_config()
                self.logger.info("Loaded local Kubernetes configuration")
            
            # Initialize API clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.autoscaling_v2 = client.AutoscalingV2Api()
            
            # Metrics API (optional)
            try:
                self.metrics_v1beta1 = client.CustomObjectsApi()
            except Exception as e:
                self.logger.warning(f"Metrics API not available: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes clients: {e}")
            self.v1 = None
    
    async def get_node_resources(self) -> List[NodeResourceInfo]:
        """Get available node resources."""
        if not self.v1:
            return []
        
        try:
            nodes = self.v1.list_node()
            node_resources = []
            
            for node in nodes.items:
                node_name = node.metadata.name
                allocatable = node.status.allocatable or {}
                capacity = node.status.capacity or {}
                labels = node.metadata.labels or {}
                taints = []
                
                if node.spec.taints:
                    taints = [
                        {
                            "key": taint.key,
                            "value": taint.value or "",
                            "effect": taint.effect
                        }
                        for taint in node.spec.taints
                    ]
                
                node_info = NodeResourceInfo(
                    node_name=node_name,
                    allocatable_cpu=allocatable.get('cpu', '0'),
                    allocatable_memory=allocatable.get('memory', '0'),
                    allocatable_gpu=allocatable.get('nvidia.com/gpu', '0'),
                    capacity_cpu=capacity.get('cpu', '0'),
                    capacity_memory=capacity.get('memory', '0'),
                    capacity_gpu=capacity.get('nvidia.com/gpu', '0'),
                    labels=labels,
                    taints=taints
                )
                node_resources.append(node_info)
            
            return node_resources
            
        except api_exception as e:
            self.logger.error(f"Error getting node resources: {e}")
            return []
    
    async def get_namespace_resource_quotas(self, namespace: str = None) -> List[ResourceQuotaInfo]:
        """Get resource quotas for namespace."""
        if not self.v1:
            return []
        
        target_namespace = namespace or self.namespace
        
        try:
            quotas = self.v1.list_namespaced_resource_quota(namespace=target_namespace)
            quota_info = []
            
            for quota in quotas.items:
                quota_name = quota.metadata.name
                hard = quota.status.hard or {}
                used = quota.status.used or {}
                
                quota_data = ResourceQuotaInfo(
                    namespace=target_namespace,
                    quota_name=quota_name,
                    cpu_hard=hard.get('requests.cpu', '0'),
                    memory_hard=hard.get('requests.memory', '0'),
                    gpu_hard=hard.get('requests.nvidia.com/gpu', '0'),
                    cpu_used=used.get('requests.cpu', '0'),
                    memory_used=used.get('requests.memory', '0'),
                    gpu_used=used.get('requests.nvidia.com/gpu', '0')
                )
                quota_info.append(quota_data)
            
            return quota_info
            
        except api_exception as e:
            self.logger.error(f"Error getting resource quotas: {e}")
            return []
    
    async def get_pod_resources(self, namespace: str = None) -> List[PodResourceInfo]:
        """Get pod resource requests and limits."""
        if not self.v1:
            return []
        
        target_namespace = namespace or self.namespace
        
        try:
            pods = self.v1.list_namespaced_pod(namespace=target_namespace)
            pod_resources = []
            
            for pod in pods.items:
                if not pod.spec.containers:
                    continue
                
                # Aggregate resources across all containers
                total_cpu_request = 0
                total_memory_request = 0
                total_cpu_limit = 0
                total_memory_limit = 0
                total_gpu_request = 0
                
                for container in pod.spec.containers:
                    resources = container.resources
                    if resources:
                        requests = resources.requests or {}
                        limits = resources.limits or {}
                        
                        # Parse CPU (convert to millicores)
                        cpu_req = self._parse_cpu_resource(requests.get('cpu', '0'))
                        cpu_lim = self._parse_cpu_resource(limits.get('cpu', '0'))
                        total_cpu_request += cpu_req
                        total_cpu_limit += cpu_lim
                        
                        # Parse memory (convert to bytes)
                        mem_req = self._parse_memory_resource(requests.get('memory', '0'))
                        mem_lim = self._parse_memory_resource(limits.get('memory', '0'))
                        total_memory_request += mem_req
                        total_memory_limit += mem_lim
                        
                        # Parse GPU
                        gpu_req = int(requests.get('nvidia.com/gpu', '0'))
                        total_gpu_request += gpu_req
                
                pod_info = PodResourceInfo(
                    pod_name=pod.metadata.name,
                    namespace=target_namespace,
                    cpu_request=f"{total_cpu_request}m",
                    memory_request=f"{total_memory_request}",
                    cpu_limit=f"{total_cpu_limit}m",
                    memory_limit=f"{total_memory_limit}",
                    gpu_request=str(total_gpu_request)
                )
                pod_resources.append(pod_info)
            
            return pod_resources
            
        except api_exception as e:
            self.logger.error(f"Error getting pod resources: {e}")
            return []
    
    async def create_hpa(self, deployment_name: str, min_replicas: int, max_replicas: int,
                        target_cpu_percent: int = 70, namespace: str = None) -> bool:
        """Create Horizontal Pod Autoscaler."""
        if not self.autoscaling_v2:
            self.logger.warning("HPA creation not available - Kubernetes client not initialized")
            return False
        
        target_namespace = namespace or self.namespace
        
        try:
            hpa = client.V2HorizontalPodAutoscaler(
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa",
                    namespace=target_namespace
                ),
                spec=client.V2HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V2CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    metrics=[
                        client.V2MetricSpec(
                            type="Resource",
                            resource=client.V2ResourceMetricSource(
                                name="cpu",
                                target=client.V2MetricTarget(
                                    type="Utilization",
                                    average_utilization=target_cpu_percent
                                )
                            )
                        )
                    ]
                )
            )
            
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=target_namespace, body=hpa
            )
            
            self.logger.info(f"Created HPA for deployment {deployment_name}")
            return True
            
        except api_exception as e:
            self.logger.error(f"Error creating HPA: {e}")
            return False
    
    def _parse_cpu_resource(self, cpu_str: str) -> int:
        """Parse CPU resource string to millicores."""
        if not cpu_str or cpu_str == '0':
            return 0
        
        if cpu_str.endswith('m'):
            return int(cpu_str[:-1])
        else:
            return int(float(cpu_str) * 1000)
    
    def _parse_memory_resource(self, memory_str: str) -> int:
        """Parse memory resource string to bytes."""
        if not memory_str or memory_str == '0':
            return 0
        
        # Handle different memory units
        units = {
            'Ki': 1024,
            'Mi': 1024 ** 2,
            'Gi': 1024 ** 3,
            'Ti': 1024 ** 4,
            'K': 1000,
            'M': 1000 ** 2,
            'G': 1000 ** 3,
            'T': 1000 ** 4
        }
        
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return int(float(memory_str[:-len(unit)]) * multiplier)
        
        # Assume bytes if no unit
        return int(memory_str)
    
    async def get_cluster_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive cluster resource summary."""
        if not KUBERNETES_AVAILABLE:
            return {"error": "Kubernetes not available"}
        
        try:
            nodes = await self.get_node_resources()
            quotas = await self.get_namespace_resource_quotas()
            pods = await self.get_pod_resources()
            
            # Calculate totals
            total_allocatable_cpu = sum(self._parse_cpu_resource(node.allocatable_cpu) for node in nodes)
            total_allocatable_memory = sum(self._parse_memory_resource(node.allocatable_memory) for node in nodes)
            total_allocatable_gpu = sum(int(node.allocatable_gpu) for node in nodes)
            
            total_requested_cpu = sum(self._parse_cpu_resource(pod.cpu_request) for pod in pods)
            total_requested_memory = sum(self._parse_memory_resource(pod.memory_request) for pod in pods)
            total_requested_gpu = sum(int(pod.gpu_request) for pod in pods)
            
            return {
                "cluster_summary": {
                    "nodes": len(nodes),
                    "pods": len(pods),
                    "namespaces_with_quotas": len(set(quota.namespace for quota in quotas))
                },
                "resource_totals": {
                    "allocatable": {
                        "cpu_millicores": total_allocatable_cpu,
                        "memory_bytes": total_allocatable_memory,
                        "gpu_count": total_allocatable_gpu
                    },
                    "requested": {
                        "cpu_millicores": total_requested_cpu,
                        "memory_bytes": total_requested_memory,
                        "gpu_count": total_requested_gpu
                    },
                    "utilization": {
                        "cpu_percent": (total_requested_cpu / total_allocatable_cpu * 100) if total_allocatable_cpu > 0 else 0,
                        "memory_percent": (total_requested_memory / total_allocatable_memory * 100) if total_allocatable_memory > 0 else 0,
                        "gpu_percent": (total_requested_gpu / total_allocatable_gpu * 100) if total_allocatable_gpu > 0 else 0
                    }
                },
                "nodes": [
                    {
                        "name": node.node_name,
                        "allocatable_cpu": node.allocatable_cpu,
                        "allocatable_memory": node.allocatable_memory,
                        "allocatable_gpu": node.allocatable_gpu,
                        "gpu_enabled": int(node.allocatable_gpu) > 0
                    }
                    for node in nodes
                ],
                "resource_quotas": [
                    {
                        "namespace": quota.namespace,
                        "quota_name": quota.quota_name,
                        "cpu_utilization": f"{quota.cpu_used}/{quota.cpu_hard}",
                        "memory_utilization": f"{quota.memory_used}/{quota.memory_hard}",
                        "gpu_utilization": f"{quota.gpu_used}/{quota.gpu_hard}"
                    }
                    for quota in quotas
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cluster resource summary: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if Kubernetes integration is available."""
        return KUBERNETES_AVAILABLE and self.v1 is not None
