"""Direct component integration for ML Pipeline Orchestrator."""
from .component_invoker import ComponentInvoker
from .direct_component_loader import DirectComponentLoader
__all__ = ['DirectComponentLoader', 'ComponentInvoker']