#!/usr/bin/env python3
"""Repository Pattern Validation Script

Validates that database queries have been successfully extracted to repository layer.
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepositoryPatternValidator:
    """Validates repository pattern implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src" / "prompt_improver"
        
        # SQL query patterns to detect
        self.sql_patterns = [
            r"SELECT\s+",
            r"INSERT\s+INTO",
            r"UPDATE\s+",
            r"DELETE\s+FROM",
            r"text\(\s*[\"']",
            r"execute\(\s*[\"']",
            r"execute_optimized_query",
            r"pd\.read_sql_query",
            r"conn\.execute",
            r"session\.execute\(\s*text\(",
        ]
        
        # Compiled patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_patterns]
        
        # Directories that SHOULD contain SQL (repositories)
        self.allowed_sql_dirs = {
            "repositories/impl",
            "repositories/protocols", 
            "database/services",
            "database",  # Core database infrastructure
        }
        
        # Files that are allowed to have SQL
        self.allowed_sql_files = {
            "database/models.py",
            "database/connection.py", 
            "database/query_optimizer.py",
            "database/analytics_query_interface.py",  # Legacy - should be migrated
            # Add repository files
        }
    
    def scan_file_for_sql(self, file_path: Path) -> List[Dict]:
        """Scan a Python file for SQL queries."""
        sql_found = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern in self.compiled_patterns:
                    if pattern.search(line):
                        sql_found.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'content': line.strip(),
                            'pattern': pattern.pattern
                        })
                        
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
            
        return sql_found
    
    def is_file_allowed_sql(self, file_path: Path) -> bool:
        """Check if file is allowed to contain SQL queries."""
        rel_path = str(file_path.relative_to(self.src_path))
        
        # Check allowed files
        if rel_path in self.allowed_sql_files:
            return True
            
        # Check allowed directories
        for allowed_dir in self.allowed_sql_dirs:
            if rel_path.startswith(allowed_dir):
                return True
                
        return False
    
    def validate_repository_pattern(self) -> Dict[str, List]:
        """Validate repository pattern implementation."""
        results = {
            'violations': [],
            'allowed_sql': [],
            'repository_files': [],
            'migrated_files': []
        }
        
        # Find all Python files
        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
                
            sql_queries = self.scan_file_for_sql(py_file)
            
            if sql_queries:
                if self.is_file_allowed_sql(py_file):
                    results['allowed_sql'].extend(sql_queries)
                    if "repositories" in str(py_file):
                        results['repository_files'].extend(sql_queries)
                else:
                    results['violations'].extend(sql_queries)
        
        return results
    
    def check_repository_implementation(self) -> Dict[str, bool]:
        """Check that repository implementations exist."""
        checks = {
            'analytics_repository_exists': False,
            'ml_repository_exists': False, 
            'apriori_repository_exists': False,
            'intelligence_mixin_exists': False,
            'factory_exists': False,
        }
        
        repo_impl_path = self.src_path / "repositories" / "impl"
        
        # Check repository implementations
        if (repo_impl_path / "analytics_repository.py").exists():
            checks['analytics_repository_exists'] = True
            
        if (repo_impl_path / "ml_repository.py").exists():
            checks['ml_repository_exists'] = True
            
        if (repo_impl_path / "apriori_repository.py").exists():
            checks['apriori_repository_exists'] = True
            
        if (repo_impl_path / "ml_repository_intelligence.py").exists():
            checks['intelligence_mixin_exists'] = True
            
        if (self.src_path / "repositories" / "factory.py").exists():
            checks['factory_exists'] = True
            
        return checks
    
    def check_service_layer_migration(self) -> List[str]:
        """Check which service files still have database queries."""
        problem_files = []
        
        # Service directories that should NOT have SQL
        service_dirs = [
            "api",
            "core/services",
            "ml/background", 
            "ml/learning/patterns",
            "cli",
        ]
        
        for service_dir in service_dirs:
            service_path = self.src_path / service_dir
            if service_path.exists():
                for py_file in service_path.rglob("*.py"):
                    sql_queries = self.scan_file_for_sql(py_file)
                    if sql_queries:
                        problem_files.append(str(py_file.relative_to(self.project_root)))
                        
        return problem_files
    
    def generate_report(self) -> str:
        """Generate validation report."""
        repo_results = self.validate_repository_pattern()
        repo_checks = self.check_repository_implementation()
        service_problems = self.check_service_layer_migration()
        
        report = []
        report.append("# Repository Pattern Validation Report")
        report.append("")
        
        # Repository infrastructure
        report.append("## Repository Infrastructure")
        for check, passed in repo_checks.items():
            status = "✓" if passed else "✗"
            report.append(f"{status} {check}: {passed}")
        report.append("")
        
        # Repository files with SQL (good)
        report.append(f"## Repository Files with SQL Queries ({len(repo_results['repository_files'])} found)")
        for item in repo_results['repository_files'][:10]:  # Show first 10
            report.append(f"✓ {item['file']}:{item['line']} - {item['pattern']}")
        if len(repo_results['repository_files']) > 10:
            report.append(f"... and {len(repo_results['repository_files']) - 10} more")
        report.append("")
        
        # Service layer violations (bad)
        report.append(f"## Service Layer Violations ({len(repo_results['violations'])} found)")
        if repo_results['violations']:
            for item in repo_results['violations'][:15]:  # Show first 15
                report.append(f"✗ {item['file']}:{item['line']} - {item['content'][:80]}...")
            if len(repo_results['violations']) > 15:
                report.append(f"... and {len(repo_results['violations']) - 15} more violations")
        else:
            report.append("✓ No violations found - all SQL queries properly contained in repositories!")
        report.append("")
        
        # Problem files summary
        report.append(f"## Service Files Still Containing SQL ({len(service_problems)})")
        for problem_file in service_problems:
            report.append(f"✗ {problem_file}")
        report.append("")
        
        # Summary
        total_violations = len(repo_results['violations'])
        total_repo_queries = len(repo_results['repository_files'])
        
        report.append("## Summary")
        report.append(f"- Repository queries: {total_repo_queries} ✓")
        report.append(f"- Service layer violations: {total_violations}")
        report.append(f"- Files needing migration: {len(service_problems)}")
        
        if total_violations == 0 and len(service_problems) == 0:
            report.append("🎉 **Repository pattern successfully implemented!**")
        else:
            report.append("⚠️  **Repository pattern migration incomplete**")
            
        return "\n".join(report)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    validator = RepositoryPatternValidator(project_root)
    
    print("Validating repository pattern implementation...")
    report = validator.generate_report()
    print(report)
    
    # Write report to file
    report_file = project_root / "repository_validation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report written to: {report_file}")