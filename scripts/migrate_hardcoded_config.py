#!/usr/bin/env python3
"""
Migration script to help transition from hardcoded configuration values.
Scans codebase for hardcoded values and suggests environment variable replacements.
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HardcodedValue:
    """Represents a hardcoded configuration value found in code."""
    file_path: str
    line_number: int
    line_content: str
    value: str
    context: str
    suggested_env_var: str
    confidence: float  # 0.0 to 1.0


@dataclass
class MigrationReport:
    """Report of hardcoded values found and migration suggestions."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_files_scanned: int = 0
    hardcoded_values: List[HardcodedValue] = field(default_factory=list)
    high_confidence_values: List[HardcodedValue] = field(default_factory=list)
    suggested_env_vars: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_files_scanned": self.total_files_scanned,
            "total_hardcoded_values": len(self.hardcoded_values),
            "high_confidence_count": len(self.high_confidence_values),
            "suggested_env_vars": sorted(list(self.suggested_env_vars)),
            "hardcoded_values": [
                {
                    "file_path": hv.file_path,
                    "line_number": hv.line_number,
                    "line_content": hv.line_content.strip(),
                    "value": hv.value,
                    "context": hv.context,
                    "suggested_env_var": hv.suggested_env_var,
                    "confidence": hv.confidence
                }
                for hv in self.hardcoded_values
            ]
        }


class HardcodedConfigMigrator:
    """Scans codebase for hardcoded configuration values and suggests migrations."""
    
    # Patterns that suggest configuration values
    CONFIG_PATTERNS = [
        # Database related
        (r'postgresql://[^"\']+', "DATABASE_URL", 0.9),
        (r'localhost:5432', "POSTGRES_HOST,POSTGRES_PORT", 0.8),
        (r'redis://[^"\']+', "REDIS_URL", 0.9),
        (r'localhost:6379', "REDIS_HOST,REDIS_PORT", 0.8),
        
        # API endpoints and URLs
        (r'https?://[^"\']+/api', "API_BASE_URL", 0.7),
        (r'https?://[^"\']+\.com', "EXTERNAL_SERVICE_URL", 0.6),
        
        # File paths that might be configurable
        (r'/opt/[^"\']+', "ML_MODEL_PATH", 0.6),
        (r'./models?/', "ML_MODEL_PATH", 0.7),
        (r'/tmp/[^"\']+', "TEMP_DIR", 0.5),
        
        # Common configuration values
        (r'pool_size\s*[=:]\s*(\d+)', "DB_POOL_SIZE", 0.8),
        (r'timeout\s*[=:]\s*(\d+)', "REQUEST_TIMEOUT", 0.7),
        (r'max_connections?\s*[=:]\s*(\d+)', "MAX_CONNECTIONS", 0.8),
        (r'batch_size\s*[=:]\s*(\d+)', "BATCH_SIZE", 0.7),
        
        # Security related
        (r'secret[_-]?key["\']?\s*[=:]\s*["\'][^"\']{8,}', "SECRET_KEY", 0.9),
        (r'api[_-]?key["\']?\s*[=:]\s*["\'][^"\']{8,}', "API_KEY", 0.9),
        (r'password["\']?\s*[=:]\s*["\'][^"\']+', "PASSWORD", 0.8),
    ]
    
    # File extensions to scan
    SCAN_EXTENSIONS = {'.py', '.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'}
    
    # Directories to skip
    SKIP_DIRS = {
        '__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules',
        '.pytest_cache', 'htmlcov', 'mlruns', 'archive', 'migrations'
    }
    
    def __init__(self, root_path: str = "."):
        """Initialize migrator with root path to scan."""
        self.root_path = Path(root_path)
        
    def scan_codebase(self) -> MigrationReport:
        """Scan entire codebase for hardcoded configuration values."""
        report = MigrationReport()
        
        # Find all files to scan
        files_to_scan = self._find_files_to_scan()
        report.total_files_scanned = len(files_to_scan)
        
        print(f"🔍 Scanning {len(files_to_scan)} files for hardcoded configuration values...")
        
        for file_path in files_to_scan:
            try:
                hardcoded_values = self._scan_file(file_path)
                report.hardcoded_values.extend(hardcoded_values)
                
                # Track high confidence values
                high_conf = [hv for hv in hardcoded_values if hv.confidence >= 0.8]
                report.high_confidence_values.extend(high_conf)
                
                # Track suggested environment variables
                for hv in hardcoded_values:
                    report.suggested_env_vars.add(hv.suggested_env_var)
                    
            except Exception as e:
                print(f"⚠️  Error scanning {file_path}: {e}")
        
        return report
    
    def _find_files_to_scan(self) -> List[Path]:
        """Find all files that should be scanned for hardcoded values."""
        files = []
        
        for file_path in self.root_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue
                
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in self.SKIP_DIRS):
                continue
                
            # Only scan files with relevant extensions
            if file_path.suffix in self.SCAN_EXTENSIONS:
                files.append(file_path)
        
        return files
    
    def _scan_file(self, file_path: Path) -> List[HardcodedValue]:
        """Scan a single file for hardcoded configuration values."""
        hardcoded_values = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            
            for line_num, line in enumerate(lines, 1):
                # Skip comments and empty lines
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                
                # Check each pattern
                for pattern, env_var_suggestion, confidence in self.CONFIG_PATTERNS:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract the actual value
                        value = match if isinstance(match, str) else str(match)
                        
                        # Skip if it's already using environment variables
                        if 'os.getenv' in line or 'os.environ' in line or '${' in line:
                            continue
                            
                        # Create context (surrounding lines)
                        context_lines = []
                        for i in range(max(0, line_num - 3), min(len(lines), line_num + 2)):
                            context_lines.append(f"{i+1:4d}: {lines[i]}")
                        context = "\n".join(context_lines)
                        
                        hardcoded_value = HardcodedValue(
                            file_path=str(file_path.relative_to(self.root_path)),
                            line_number=line_num,
                            line_content=line,
                            value=value,
                            context=context,
                            suggested_env_var=env_var_suggestion,
                            confidence=confidence
                        )
                        
                        hardcoded_values.append(hardcoded_value)
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return hardcoded_values
    
    def generate_migration_script(self, report: MigrationReport, output_file: str = "migration_script.py") -> None:
        """Generate a Python script to help with migration."""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated migration script for hardcoded configuration values.
Generated on: {datetime.now().isoformat()}

This script provides suggestions for migrating hardcoded values to environment variables.
"""

import os
import re
from pathlib import Path

# High confidence migrations (confidence >= 0.8)
HIGH_CONFIDENCE_MIGRATIONS = [
'''
        
        # Add high confidence migrations
        for hv in report.high_confidence_values:
            script_content += f'''    {{
        "file": "{hv.file_path}",
        "line": {hv.line_number},
        "old_value": {repr(hv.value)},
        "env_var": "{hv.suggested_env_var}",
        "confidence": {hv.confidence}
    }},
'''
        
        script_content += ''']

def apply_migrations():
    """Apply high confidence migrations."""
    print("🚀 Applying high confidence migrations...")
    
    for migration in HIGH_CONFIDENCE_MIGRATIONS:
        file_path = Path(migration["file"])
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"📝 Processing {file_path} (line {migration['line']})...")
        
        # Read file content
        content = file_path.read_text()
        lines = content.splitlines()
        
        if migration["line"] <= len(lines):
            old_line = lines[migration["line"] - 1]
            print(f"   Old: {old_line.strip()}")
            
            # Suggest replacement
            env_var = migration["env_var"]
            replacement_suggestion = f'os.getenv("{env_var}", {repr(migration["old_value"])})'
            print(f"   Suggested: Replace with {replacement_suggestion}")
            print()

def create_env_template():
    """Create environment variable template."""
    env_vars = set()
    for migration in HIGH_CONFIDENCE_MIGRATIONS:
        env_vars.add(migration["env_var"])
    
    template = "# Environment variables for migrated configuration\\n"
    template += f"# Generated on: {datetime.now().isoformat()}\\n\\n"
    
    for env_var in sorted(env_vars):
        template += f"{env_var}=CHANGE_ME\\n"
    
    Path("migration_env_template.env").write_text(template)
    print("📄 Created migration_env_template.env")

if __name__ == "__main__":
    print("🔧 Configuration Migration Helper")
    print("=" * 50)
    
    create_env_template()
    apply_migrations()
    
    print("✅ Migration suggestions complete!")
    print("Review the suggestions above and apply them manually.")
'''
        
        Path(output_file).write_text(script_content)
        print(f"📄 Generated migration script: {output_file}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate hardcoded configuration values")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--output", default="hardcoded_config_report.json", 
                       help="Output file for report")
    parser.add_argument("--generate-script", action="store_true",
                       help="Generate migration script")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum confidence threshold for reporting")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = HardcodedConfigMigrator(args.root)
    
    # Scan codebase
    print("🔍 Scanning codebase for hardcoded configuration values...")
    report = migrator.scan_codebase()
    
    # Filter by confidence
    filtered_values = [hv for hv in report.hardcoded_values if hv.confidence >= args.min_confidence]
    
    # Print summary
    print(f"\\n📊 Scan Results:")
    print(f"   Files scanned: {report.total_files_scanned}")
    print(f"   Total hardcoded values: {len(report.hardcoded_values)}")
    print(f"   High confidence (>= {args.min_confidence}): {len(filtered_values)}")
    print(f"   Suggested environment variables: {len(report.suggested_env_vars)}")
    
    # Show top findings
    if filtered_values:
        print(f"\\n🔥 Top findings (confidence >= {args.min_confidence}):")
        for i, hv in enumerate(sorted(filtered_values, key=lambda x: x.confidence, reverse=True)[:10]):
            print(f"   {i+1:2d}. {hv.file_path}:{hv.line_number}")
            print(f"       Value: {hv.value}")
            print(f"       Suggested: {hv.suggested_env_var}")
            print(f"       Confidence: {hv.confidence:.1%}")
            print()
    
    # Save report
    report_data = report.to_dict()
    Path(args.output).write_text(json.dumps(report_data, indent=2))
    print(f"📄 Detailed report saved to: {args.output}")
    
    # Generate migration script if requested
    if args.generate_script:
        migrator.generate_migration_script(report)
    
    # Show suggested environment variables
    if report.suggested_env_vars:
        print(f"\\n📋 Suggested environment variables:")
        for env_var in sorted(report.suggested_env_vars):
            print(f"   {env_var}")


if __name__ == "__main__":
    main()