#!/usr/bin/env python3
"""
Comprehensive ML Pipeline Orchestrator Integration Audit

This script performs a systematic audit to identify components that are not yet
fully integrated with the ML Pipeline Orchestrator, following the same approach
that achieved 100% success for Bayesian components integration.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass
from enum import Enum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Integration status levels"""
    FULLY_INTEGRATED = "fully_integrated"
    PARTIALLY_INTEGRATED = "partially_integrated"
    NOT_INTEGRATED = "not_integrated"
    SIMULATION_ONLY = "simulation_only"

class Priority(Enum):
    """Priority levels for integration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ComponentAuditResult:
    """Result of component integration audit"""
    name: str
    file_path: str
    class_name: str
    integration_status: IntegrationStatus
    priority: Priority
    tier_assignment: str
    issues: List[str]
    required_actions: List[str]
    dependencies: List[str]
    estimated_effort: str

class MLOrchestratorIntegrationAuditor:
    """Comprehensive auditor for ML Pipeline Orchestrator integration gaps"""
    
    def __init__(self):
        self.audit_results: List[ComponentAuditResult] = []
        self.registered_components: Set[str] = set()
        self.existing_components: Dict[str, Dict[str, Any]] = {}
        
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive integration audit"""
        
        print("🔍 ML Pipeline Orchestrator Integration Audit")
        print("=" * 60)
        
        # Phase 1: Discover registered components
        await self._discover_registered_components()
        
        # Phase 2: Scan codebase for existing components
        await self._scan_existing_components()
        
        # Phase 3: Analyze integration gaps
        await self._analyze_integration_gaps()
        
        # Phase 4: Generate prioritized roadmap
        roadmap = self._generate_integration_roadmap()
        
        # Phase 5: Create audit report
        self._generate_audit_report(roadmap)
        
        return roadmap
    
    async def _discover_registered_components(self):
        """Discover components currently registered with orchestrator"""
        
        print("\n📋 Phase 1: Discovering Registered Components...")
        
        try:
            from prompt_improver.ml.orchestration.core.component_registry import ComponentRegistry
            from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
            from prompt_improver.ml.orchestration.config.component_definitions import ComponentDefinitions
            
            # Get registered components
            config = OrchestratorConfig()
            registry = ComponentRegistry(config)
            components = await registry.discover_components()
            
            for comp in components:
                self.registered_components.add(comp.name)
                print(f"  ✅ Registered: {comp.name} ({comp.tier.value})")
            
            # Get all defined components (even if not discoverable)
            component_defs = ComponentDefinitions()
            all_definitions = component_defs.get_all_component_definitions()
            
            print(f"\n  📊 Summary:")
            print(f"    Discoverable components: {len(components)}")
            print(f"    Defined components: {len(all_definitions)}")
            print(f"    Definition gap: {len(all_definitions) - len(components)}")
            
        except Exception as e:
            print(f"  ❌ Failed to discover registered components: {e}")
    
    async def _scan_existing_components(self):
        """Scan codebase for existing ML components"""
        
        print("\n🔍 Phase 2: Scanning Existing Components...")
        
        # Define component directories to scan
        component_dirs = [
            "src/prompt_improver/ml/learning/algorithms",
            "src/prompt_improver/ml/learning/quality", 
            "src/prompt_improver/ml/optimization/algorithms",
            "src/prompt_improver/ml/optimization/validation",
            "src/prompt_improver/ml/evaluation",
            "src/prompt_improver/ml/preprocessing",
            "src/prompt_improver/performance/monitoring",
            "src/prompt_improver/performance/testing",
            "src/prompt_improver/performance/analytics",
            "src/prompt_improver/performance/optimization",
            "src/prompt_improver/services",
            "src/prompt_improver/ml/automl"
        ]
        
        for dir_path in component_dirs:
            if os.path.exists(dir_path):
                await self._scan_directory(dir_path)
        
        print(f"  📊 Found {len(self.existing_components)} potential components")
    
    async def _scan_directory(self, dir_path: str):
        """Scan directory for Python files with ML components"""
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    await self._analyze_python_file(file_path)
    
    async def _analyze_python_file(self, file_path: str):
        """Analyze Python file for ML component classes"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for class definitions that might be ML components
            import re
            class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
            classes = re.findall(class_pattern, content)
            
            # Filter for likely ML component classes (expanded patterns)
            ml_component_patterns = [
                r'.*Analyzer.*', r'.*Optimizer.*', r'.*Engine.*', r'.*Scorer.*',
                r'.*Validator.*', r'.*Monitor.*', r'.*Orchestrator.*', r'.*Manager.*',
                r'.*Processor.*', r'.*Extractor.*', r'.*Transformer.*', r'.*Framework.*',
                r'.*Learner.*', r'.*Service.*', r'.*Handler.*', r'.*Controller.*',
                r'.*Detector.*', r'.*Classifier.*', r'.*Predictor.*', r'.*Evaluator.*',
                r'.*Generator.*', r'.*Builder.*', r'.*Factory.*', r'.*Provider.*',
                r'.*Reducer.*', r'.*Clusterer.*', r'.*Sampler.*', r'.*Selector.*'
            ]
            
            for class_name in classes:
                if any(re.match(pattern, class_name) for pattern in ml_component_patterns):
                    # Convert file path to module path
                    module_path = file_path.replace('src/', '').replace('/', '.').replace('.py', '')
                    
                    component_key = class_name.lower().replace('analyzer', '').replace('engine', '').replace('framework', '')
                    component_key = re.sub(r'[^a-z_]', '', component_key)
                    
                    self.existing_components[component_key] = {
                        'class_name': class_name,
                        'file_path': file_path,
                        'module_path': module_path
                    }
                    
        except Exception as e:
            # Skip files that can't be read
            pass
    
    async def _analyze_integration_gaps(self):
        """Analyze integration gaps between existing and registered components"""

        print("\n🔍 Phase 3: Analyzing Integration Gaps...")

        # First, analyze ALL existing components found in codebase scan
        print(f"  📊 Analyzing {len(self.existing_components)} discovered components...")

        for comp_key, comp_info in self.existing_components.items():
            integration_status = IntegrationStatus.NOT_INTEGRATED
            issues = []
            required_actions = []
            priority = self._determine_component_priority(comp_key, comp_info)
            tier = self._determine_component_tier(comp_key, comp_info)

            if comp_key in self.registered_components:
                integration_status = IntegrationStatus.FULLY_INTEGRATED
            else:
                issues.append("Not registered in component registry")
                issues.append("Not discoverable through orchestrator")
                required_actions.append("Add to component_definitions.py")
                required_actions.append("Implement orchestrator integration methods")
                required_actions.append("Add to appropriate tier connector")

            # Check if component exists in codebase
            if not os.path.exists(comp_info['file_path']):
                issues.append("Component file does not exist")
                required_actions.append("Verify component implementation exists")

            audit_result = ComponentAuditResult(
                name=comp_key,
                file_path=comp_info['file_path'],
                class_name=comp_info['class_name'],
                integration_status=integration_status,
                priority=priority,
                tier_assignment=tier,
                issues=issues,
                required_actions=required_actions,
                dependencies=[],
                estimated_effort="2-4 hours" if integration_status == IntegrationStatus.NOT_INTEGRATED else "0 hours"
            )

            self.audit_results.append(audit_result)

            status_icon = "✅" if integration_status == IntegrationStatus.FULLY_INTEGRATED else "❌"
            print(f"    {status_icon} {comp_key}: {integration_status.value} ({comp_info['class_name']})")

    def _determine_component_priority(self, comp_key: str, comp_info: Dict[str, Any]) -> Priority:
        """Determine priority level for component integration"""

        # High priority patterns
        high_priority_patterns = [
            'scorer', 'analyzer', 'validator', 'engine', 'learner', 'optimizer',
            'orchestrator', 'monitor', 'processor', 'extractor'
        ]

        # Medium priority patterns
        medium_priority_patterns = [
            'transformer', 'manager', 'framework', 'service', 'handler'
        ]

        # Check class name and component key for priority indicators
        comp_lower = comp_key.lower()
        class_lower = comp_info['class_name'].lower()

        if any(pattern in comp_lower or pattern in class_lower for pattern in high_priority_patterns):
            return Priority.HIGH
        elif any(pattern in comp_lower or pattern in class_lower for pattern in medium_priority_patterns):
            return Priority.MEDIUM
        else:
            return Priority.LOW

    def _determine_component_tier(self, comp_key: str, comp_info: Dict[str, Any]) -> str:
        """Determine appropriate tier for component"""

        module_path = comp_info['module_path']

        # Tier 1: Core ML components
        if 'ml.core' in module_path or 'ml.integration' in module_path:
            return 'tier1_core'

        # Tier 2: Optimization & Learning
        elif ('ml.optimization' in module_path or 'ml.learning' in module_path or
              'optimization' in comp_key or 'learner' in comp_key or 'optimizer' in comp_key):
            return 'tier2_optimization'

        # Tier 3: Evaluation & Analysis
        elif ('ml.evaluation' in module_path or 'evaluation' in module_path or
              'analyzer' in comp_key or 'validator' in comp_key or 'statistical' in comp_key):
            return 'tier3_evaluation'

        # Tier 4: Performance & Monitoring
        elif ('performance' in module_path or 'monitoring' in module_path or
              'monitor' in comp_key or 'analytics' in comp_key):
            return 'tier4_performance'

        # Default to Tier 2 for ML components
        else:
            return 'tier2_optimization'
    
    def _generate_integration_roadmap(self) -> Dict[str, Any]:
        """Generate prioritized integration roadmap"""
        
        print("\n🗺️ Phase 4: Generating Integration Roadmap...")
        
        # Group by priority
        roadmap = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "summary": {
                "total_components": len(self.audit_results),
                "not_integrated": len([r for r in self.audit_results if r.integration_status == IntegrationStatus.NOT_INTEGRATED]),
                "partially_integrated": len([r for r in self.audit_results if r.integration_status == IntegrationStatus.PARTIALLY_INTEGRATED]),
                "fully_integrated": len([r for r in self.audit_results if r.integration_status == IntegrationStatus.FULLY_INTEGRATED])
            }
        }
        
        for result in self.audit_results:
            if result.integration_status != IntegrationStatus.FULLY_INTEGRATED:
                roadmap[result.priority.value].append({
                    "name": result.name,
                    "class_name": result.class_name,
                    "file_path": result.file_path,
                    "tier": result.tier_assignment,
                    "status": result.integration_status.value,
                    "issues": result.issues,
                    "actions": result.required_actions,
                    "effort": result.estimated_effort
                })
        
        return roadmap
    
    def _generate_audit_report(self, roadmap: Dict[str, Any]):
        """Generate comprehensive audit report"""
        
        print("\n📊 Phase 5: Integration Audit Report")
        print("=" * 60)
        
        summary = roadmap["summary"]
        print(f"Total Components Analyzed: {summary['total_components']}")
        print(f"Fully Integrated: {summary['fully_integrated']} ✅")
        print(f"Partially Integrated: {summary['partially_integrated']} ⚠️")
        print(f"Not Integrated: {summary['not_integrated']} ❌")
        
        integration_percentage = (summary['fully_integrated'] / summary['total_components']) * 100
        print(f"Integration Completion: {integration_percentage:.1f}%")
        
        print(f"\n🎯 Priority Breakdown:")
        for priority in ["critical", "high", "medium", "low"]:
            count = len(roadmap[priority])
            if count > 0:
                print(f"  {priority.upper()}: {count} components")
                for comp in roadmap[priority][:3]:  # Show first 3
                    print(f"    • {comp['name']} ({comp['status']})")
                if count > 3:
                    print(f"    ... and {count - 3} more")
        
        print(f"\n🚀 Next Steps:")
        print("1. Implement HIGH priority components first")
        print("2. Follow Bayesian integration pattern (100% success rate)")
        print("3. Add components to component_definitions.py")
        print("4. Implement orchestrator integration methods")
        print("5. Add comprehensive testing")


async def main():
    """Main audit execution function"""
    
    auditor = MLOrchestratorIntegrationAuditor()
    roadmap = await auditor.run_comprehensive_audit()
    
    # Save roadmap to file
    import json
    with open('ml_orchestrator_integration_roadmap.json', 'w') as f:
        json.dump(roadmap, f, indent=2)
    
    print(f"\n💾 Detailed roadmap saved to: ml_orchestrator_integration_roadmap.json")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
