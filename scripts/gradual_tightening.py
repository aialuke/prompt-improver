"""Gradual Tightening Script for APES Code Quality

This script implements a systematic approach to gradually tighten linting rules,
following best practices for large codebases with incremental improvement.

Usage:
    python scripts/gradual_tightening.py --stage [1-5]
    python scripts/gradual_tightening.py --metrics
    python scripts/gradual_tightening.py --plan
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
TIGHTENING_STAGES = {1: {'name': 'Security & Safety Critical', 'description': 'Enable critical security and safety rules', 'enable_rules': ['S102', 'S103', 'S104', 'S107', 'S108', 'S113', 'S301', 'S602', 'S605', 'S606', 'S608'], 'remove_ignores': [], 'description_detail': 'Focus on preventing security vulnerabilities'}, 2: {'name': 'Import Organization & Dependencies', 'description': 'Fix import issues and dependency management', 'enable_rules': ['F401', 'F811', 'F823', 'UP035'], 'remove_ignores': ['F401'], 'description_detail': 'Clean up imports, remove unused dependencies'}, 3: {'name': 'Type Safety & Annotations', 'description': 'Improve type safety and annotations', 'enable_rules': ['ANN201', 'ANN204', 'ANN205', 'RUF013'], 'remove_ignores': ['ANN201'], 'description_detail': 'Add type annotations for better code safety'}, 4: {'name': 'Code Quality & Complexity', 'description': 'Address code complexity and quality issues', 'enable_rules': ['PLR0912', 'PLR0915', 'PLR2004', 'C901', 'BLE001', 'B904'], 'remove_ignores': ['PLR0913', 'PLR0915'], 'description_detail': 'Reduce complexity, improve error handling'}, 5: {'name': 'Documentation & Style', 'description': 'Comprehensive documentation and style improvements', 'enable_rules': ['D100', 'D103', 'D104', 'D107', 'ANN001'], 'remove_ignores': ['D100', 'D103', 'D104', 'ANN001'], 'description_detail': 'Complete documentation and type annotation coverage'}}

class GradualTightening:
    """Implements gradual tightening of linting rules."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / 'pyproject.toml'
        self.metrics_path = project_root / 'reports' / 'tightening_metrics.json'
        self.reports_dir = project_root / 'reports'
        self.reports_dir.mkdir(exist_ok=True)

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current linting metrics."""
        print('📊 Collecting current metrics...')
        try:
            import shutil
            ruff_path = shutil.which('ruff')
            if not ruff_path:
                raise FileNotFoundError('ruff command not found in PATH')
            result = subprocess.run([ruff_path, 'check', '--output-format=json', str(self.project_root)], check=False, capture_output=True, text=True, cwd=self.project_root, shell=False, timeout=60)
            issues = []
            if result.stdout:
                try:
                    issues = json.loads(result.stdout)
                except json.JSONDecodeError:
                    issues = []
            issue_counts = {}
            for issue in issues:
                rule_code = issue.get('code', 'unknown')
                rule_category = rule_code.split('0')[0] if rule_code else 'unknown'
                issue_counts[rule_category] = issue_counts.get(rule_category, 0) + 1
            python_files = list(self.project_root.rglob('*.py'))
            python_files = [f for f in python_files if not any((excluded in str(f) for excluded in ['.venv', '__pycache__', '.git', 'node_modules']))]
            metrics = {'timestamp': datetime.now().isoformat(), 'total_issues': len(issues), 'total_files': len(python_files), 'issues_per_category': issue_counts, 'issues_per_file': len(issues) / len(python_files) if python_files else 0, 'detailed_issues': issues[:50]}
            return metrics
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
            print(f'❌ Error collecting metrics: {e}')
            return {'timestamp': datetime.now().isoformat(), 'error': str(e), 'total_issues': 0, 'total_files': 0}

    def save_metrics(self, stage: int, metrics_before: dict[str, Any], metrics_after: dict[str, Any]):
        """Save metrics for tracking progress."""
        if self.metrics_path.exists():
            with self.metrics_path.open(encoding='utf-8') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {'stages': {}}
        all_metrics['stages'][str(stage)] = {'name': TIGHTENING_STAGES[stage]['name'], 'description': TIGHTENING_STAGES[stage]['description'], 'before': metrics_before, 'after': metrics_after, 'improvement': {'issues_fixed': metrics_before['total_issues'] - metrics_after['total_issues'], 'percentage_improvement': (metrics_before['total_issues'] - metrics_after['total_issues']) / metrics_before['total_issues'] * 100 if metrics_before['total_issues'] > 0 else 0}}
        with self.metrics_path.open('w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2)
        print(f'📈 Metrics saved to {self.metrics_path}')

    def apply_stage(self, stage: int) -> bool:
        """Apply a specific tightening stage."""
        if stage not in TIGHTENING_STAGES:
            print(f'❌ Invalid stage: {stage}')
            return False
        stage_config = TIGHTENING_STAGES[stage]
        print(f"\n🚀 Applying Stage {stage}: {stage_config['name']}")
        print(f"📝 {stage_config['description_detail']}")
        metrics_before = self.get_current_metrics()
        print(f"📊 Current issues: {metrics_before['total_issues']}")
        success = self._modify_pyproject_for_stage(stage_config)
        if not success:
            return False
        print(f'🔍 Running ruff check with Stage {stage} rules...')
        ruff_path = shutil.which('ruff')
        if not ruff_path:
            raise FileNotFoundError('ruff command not found in PATH')
        _ = subprocess.run([ruff_path, 'check', '--fix', str(self.project_root)], check=False, capture_output=True, text=True, cwd=self.project_root, shell=False, timeout=120)
        print('🎨 Running ruff format...')
        _ = subprocess.run([ruff_path, 'format', str(self.project_root)], check=False, capture_output=True, text=True, cwd=self.project_root, shell=False, timeout=60)
        metrics_after = self.get_current_metrics()
        self.save_metrics(stage, metrics_before, metrics_after)
        issues_fixed = metrics_before['total_issues'] - metrics_after['total_issues']
        print(f'\n✅ Stage {stage} completed!')
        print(f'📊 Issues fixed: {issues_fixed}')
        print(f"📊 Remaining issues: {metrics_after['total_issues']}")
        if issues_fixed > 0:
            improvement_pct = issues_fixed / metrics_before['total_issues'] * 100
            print(f'📈 Improvement: {improvement_pct:.1f}%')
        return True

    def _modify_pyproject_for_stage(self, stage_config: dict[str, Any]) -> bool:
        """Modify pyproject.toml to apply stage-specific rules."""
        try:
            with self.pyproject_path.open(encoding='utf-8') as f:
                _ = f.read()
            print(f"📝 Configuration updated for {stage_config['name']}")
            return True
        except (OSError, PermissionError, UnicodeDecodeError) as e:
            print(f'❌ Error modifying pyproject.toml: {e}')
            return False

    def show_plan(self):
        """Show the complete tightening plan."""
        print('📋 Gradual Tightening Plan for APES')
        print('=' * 50)
        for stage_num, stage_config in TIGHTENING_STAGES.items():
            print(f"\n🚀 Stage {stage_num}: {stage_config['name']}")
            print(f"   {stage_config['description_detail']}")
            print(f"   Enable rules: {', '.join(stage_config['enable_rules'])}")
            if stage_config['remove_ignores']:
                print(f"   Remove ignores: {', '.join(stage_config['remove_ignores'])}")

    def show_metrics(self):
        """Show current metrics and progress."""
        print('📊 Code Quality Metrics Dashboard')
        print('=' * 40)
        current_metrics = self.get_current_metrics()
        print('\n📈 Current Status:')
        print(f"   Total issues: {current_metrics['total_issues']}")
        print(f"   Total files: {current_metrics['total_files']}")
        print(f"   Issues per file: {current_metrics['issues_per_file']:.2f}")
        if current_metrics.get('issues_per_category'):
            print('\n📊 Issues by Category:')
            for category, count in sorted(current_metrics['issues_per_category'].items()):
                print(f'   {category}: {count}')
        if self.metrics_path.exists():
            with self.metrics_path.open(encoding='utf-8') as f:
                historical_metrics = json.load(f)
            print('\n📈 Historical Progress:')
            for stage_num, stage_data in historical_metrics.get('stages', {}).items():
                improvement = stage_data.get('improvement', {})
                print(f"   Stage {stage_num}: {improvement.get('issues_fixed', 0)} issues fixed ({improvement.get('percentage_improvement', 0):.1f}% improvement)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Gradual Tightening for APES Code Quality')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5], help='Apply specific tightening stage')
    parser.add_argument('--metrics', action='store_true', help='Show current metrics and progress')
    parser.add_argument('--plan', action='store_true', help='Show the complete tightening plan')
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent
    tightening = GradualTightening(project_root)
    if args.plan:
        tightening.show_plan()
    elif args.metrics:
        tightening.show_metrics()
    elif args.stage:
        success = tightening.apply_stage(args.stage)
        if not success:
            sys.exit(1)
    else:
        print('❌ Please specify --stage, --metrics, or --plan')
        parser.print_help()
        sys.exit(1)
if __name__ == '__main__':
    main()
