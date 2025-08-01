#!/usr/bin/env python3
"""
Technical Debt Monitoring Script - 2025 Best Practices
Automated monitoring and reporting for technical debt metrics.
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalDebtMonitor:
    """Comprehensive technical debt monitoring and reporting system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.report_dir = self.project_root / "reports" / "debt_monitoring"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive technical debt analysis."""
        logger.info("Starting comprehensive technical debt analysis...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "metrics": {}
        }
        
        # Run all analysis components
        analysis_tasks = [
            ("code_quality", self.analyze_code_quality()),
            ("security", self.analyze_security()),
            ("dependencies", self.analyze_dependencies()),
            ("complexity", self.analyze_complexity()),
            ("test_coverage", self.analyze_test_coverage()),
            ("performance", self.analyze_performance_patterns())
        ]
        
        for name, task in analysis_tasks:
            try:
                logger.info(f"Running {name} analysis...")
                results["metrics"][name] = await task
                logger.info(f"✅ {name} analysis completed")
            except Exception as e:
                logger.error(f"❌ {name} analysis failed: {e}")
                results["metrics"][name] = {"error": str(e)}
        
        # Calculate overall debt score
        results["debt_score"] = self.calculate_debt_score(results["metrics"])
        
        # Save results
        await self.save_results(results)
        
        return results
    
    async def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality using ruff."""
        try:
            # Run ruff with JSON output
            result = subprocess.run(
                ["ruff", "check", ".", "--output-format=json", "--statistics"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                violations = json.loads(result.stdout)
                
                # Count violations by severity
                critical_count = sum(1 for v in violations if v.get("code") in [
                    "F401", "F841", "E722", "S105", "S106", "S608"
                ])
                
                return {
                    "total_violations": len(violations),
                    "critical_violations": critical_count,
                    "fixable_violations": sum(1 for v in violations if v.get("fixable", False)),
                    "top_violations": self.get_top_violations(violations),
                    "status": "success"
                }
            else:
                return {"total_violations": 0, "status": "clean"}
                
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def analyze_security(self) -> Dict[str, Any]:
        """Analyze security vulnerabilities."""
        try:
            # Run pip-audit for dependency vulnerabilities
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                vulnerabilities = []
                
                for dep in audit_data.get("dependencies", []):
                    if dep.get("vulns"):
                        vulnerabilities.extend(dep["vulns"])
                
                return {
                    "total_vulnerabilities": len(vulnerabilities),
                    "critical_vulnerabilities": sum(1 for v in vulnerabilities 
                                                  if "CRITICAL" in v.get("description", "")),
                    "vulnerabilities": vulnerabilities[:10],  # Top 10
                    "status": "success"
                }
            else:
                return {"error": result.stderr, "status": "failed"}
                
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency health."""
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                
                return {
                    "total_outdated": len(outdated),
                    "major_updates": sum(1 for pkg in outdated 
                                       if self.is_major_update(pkg)),
                    "outdated_packages": outdated[:10],  # Top 10
                    "status": "success"
                }
            else:
                return {"error": result.stderr, "status": "failed"}
                
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            src_files = list(self.project_root.glob("src/**/*.py"))
            
            # File size analysis
            large_files = []
            total_lines = 0
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        
                        if lines > 1000:
                            large_files.append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "lines": lines
                            })
                except Exception:
                    continue
            
            # Function count analysis
            function_count = 0
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        function_count += content.count("def ")
                except Exception:
                    continue
            
            return {
                "total_source_lines": total_lines,
                "total_functions": function_count,
                "large_files_count": len(large_files),
                "large_files": large_files[:10],  # Top 10
                "avg_lines_per_file": total_lines // len(src_files) if src_files else 0,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage metrics."""
        try:
            test_files = list(self.project_root.glob("tests/**/*.py"))
            src_files = list(self.project_root.glob("src/**/*.py"))
            
            # Count lines in test vs source
            test_lines = 0
            for file_path in test_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        test_lines += len(f.readlines())
                except Exception:
                    continue
            
            src_lines = 0
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        src_lines += len(f.readlines())
                except Exception:
                    continue
            
            coverage_ratio = (test_lines / src_lines * 100) if src_lines > 0 else 0
            
            return {
                "test_files_count": len(test_files),
                "source_files_count": len(src_files),
                "test_lines": test_lines,
                "source_lines": src_lines,
                "coverage_ratio": round(coverage_ratio, 2),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    async def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance anti-patterns."""
        try:
            src_files = list(self.project_root.glob("src/**/*.py"))
            
            anti_patterns = {
                "blocking_io_in_async": 0,
                "subprocess_without_check": 0,
                "bare_except": 0,
                "string_concatenation_in_loop": 0
            }
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for anti-patterns
                        if "async def" in content and "open(" in content:
                            anti_patterns["blocking_io_in_async"] += content.count("open(")
                        
                        anti_patterns["subprocess_without_check"] += content.count("subprocess.run(")
                        anti_patterns["bare_except"] += content.count("except:")
                        
                except Exception:
                    continue
            
            return {
                "anti_patterns": anti_patterns,
                "total_anti_patterns": sum(anti_patterns.values()),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def calculate_debt_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall technical debt score."""
        try:
            # Scoring weights (total = 100%)
            weights = {
                "code_quality": 0.25,
                "security": 0.20,
                "dependencies": 0.15,
                "complexity": 0.15,
                "test_coverage": 0.15,
                "performance": 0.10
            }
            
            scores = {}
            total_score = 0
            
            for category, weight in weights.items():
                if category in metrics and metrics[category].get("status") == "success":
                    score = self.calculate_category_score(category, metrics[category])
                    scores[category] = score
                    total_score += score * weight
                else:
                    scores[category] = 0  # Failed analysis gets 0 score
            
            # Convert to 1-10 scale
            debt_score = max(1, min(10, total_score))
            
            return {
                "overall_score": round(debt_score, 2),
                "category_scores": scores,
                "interpretation": self.interpret_debt_score(debt_score)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_category_score(self, category: str, data: Dict[str, Any]) -> float:
        """Calculate score for individual category (0-10 scale)."""
        if category == "code_quality":
            violations = data.get("total_violations", 0)
            # Score decreases with more violations
            return max(0, 10 - (violations / 1000))
        
        elif category == "security":
            vulns = data.get("total_vulnerabilities", 0)
            # Any vulnerability significantly impacts score
            return 10 if vulns == 0 else max(2, 8 - vulns)
        
        elif category == "dependencies":
            outdated = data.get("total_outdated", 0)
            return max(0, 10 - (outdated / 5))
        
        elif category == "complexity":
            large_files = data.get("large_files_count", 0)
            return max(0, 10 - large_files)
        
        elif category == "test_coverage":
            ratio = data.get("coverage_ratio", 0)
            return min(10, ratio / 10)  # 100% coverage = 10 points
        
        elif category == "performance":
            anti_patterns = data.get("total_anti_patterns", 0)
            return max(0, 10 - (anti_patterns / 10))
        
        return 5  # Default neutral score
    
    def interpret_debt_score(self, score: float) -> str:
        """Provide interpretation of debt score."""
        if score >= 8:
            return "LOW - Excellent code quality with minimal technical debt"
        elif score >= 6:
            return "MEDIUM - Good code quality with manageable technical debt"
        elif score >= 4:
            return "HIGH - Significant technical debt requiring attention"
        else:
            return "CRITICAL - Severe technical debt requiring immediate action"
    
    def get_top_violations(self, violations: List[Dict]) -> List[Dict]:
        """Get top 10 most frequent violations."""
        violation_counts = {}
        for violation in violations:
            code = violation.get("code", "unknown")
            violation_counts[code] = violation_counts.get(code, 0) + 1
        
        return sorted(
            [{"code": code, "count": count} for code, count in violation_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
    
    def is_major_update(self, package: Dict[str, str]) -> bool:
        """Check if package update is a major version change."""
        try:
            current = package.get("version", "0.0.0").split(".")
            latest = package.get("latest_version", "0.0.0").split(".")
            return int(current[0]) < int(latest[0])
        except (ValueError, IndexError):
            return False
    
    async def save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debt_analysis_{timestamp}.json"
        filepath = self.report_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save as latest.json for easy access
        latest_path = self.report_dir / "latest.json"
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2)

async def main():
    """Main entry point for technical debt monitoring."""
    monitor = TechnicalDebtMonitor()
    
    try:
        results = await monitor.run_comprehensive_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("TECHNICAL DEBT ANALYSIS SUMMARY")
        print("="*60)
        
        debt_score = results.get("debt_score", {})
        print(f"Overall Debt Score: {debt_score.get('overall_score', 'N/A')}/10")
        print(f"Interpretation: {debt_score.get('interpretation', 'Unknown')}")
        
        print("\nCategory Scores:")
        for category, score in debt_score.get("category_scores", {}).items():
            print(f"  {category.replace('_', ' ').title()}: {score:.1f}/10")
        
        print(f"\nDetailed report saved to: {monitor.report_dir}")
        print("="*60)
        
        return 0 if debt_score.get("overall_score", 0) >= 6 else 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
