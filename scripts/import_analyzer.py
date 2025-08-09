"""
Import Analyzer - Agent G
Comprehensive AST-based import verification for safe removal
"""
import ast
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    name: str
    alias: Optional[str]
    line: int
    is_from_import: bool
    level: int = 0

@dataclass
class UsageInfo:
    """Information about symbol usage"""
    name: str
    line: int
    context: str
    usage_type: str

@dataclass
class ImportAnalysis:
    """Complete analysis of imports and usage"""
    file_path: str
    imports: List[ImportInfo]
    usages: List[UsageInfo]
    unused_imports: List[ImportInfo]
    risky_imports: List[ImportInfo]
    safe_removals: List[ImportInfo]

class ImportUsageAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import usage patterns"""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.imports: List[ImportInfo] = []
        self.usages: List[UsageInfo] = []
        self.string_references: Set[str] = set()
        self.type_annotations: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        """Handle 'import module' statements"""
        for alias in node.names:
            import_info = ImportInfo(module=alias.name, name=alias.name.split('.')[-1] if not alias.asname else alias.asname, alias=alias.asname, line=node.lineno, is_from_import=False)
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle 'from module import name' statements"""
        module = node.module or ''
        for alias in node.names:
            if alias.name == '*':
                continue
            import_info = ImportInfo(module=module, name=alias.asname if alias.asname else alias.name, alias=alias.asname, line=node.lineno, is_from_import=True, level=node.level)
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Track name usage"""
        if isinstance(node.ctx, ast.Load):
            context = self._get_line_context(node.lineno)
            usage_info = UsageInfo(name=node.id, line=node.lineno, context=context, usage_type='name')
            self.usages.append(usage_info)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Track attribute access"""
        if hasattr(node.value, 'id'):
            context = self._get_line_context(node.lineno)
            usage_info = UsageInfo(name=node.value.id, line=node.lineno, context=context, usage_type='attribute')
            self.usages.append(usage_info)
        self.generic_visit(node)

    def visit_Str(self, node: ast.Str):
        """Track string literals for dynamic imports"""
        if isinstance(node.s, str):
            self.string_references.add(node.s)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Track constants (Python 3.8+)"""
        if isinstance(node.value, str):
            self.string_references.add(node.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function annotations for type usage"""
        if node.returns:
            self._extract_type_annotation(node.returns)
        for arg in node.args.args:
            if arg.annotation:
                self._extract_type_annotation(arg.annotation)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Analyze variable annotations"""
        if node.annotation:
            self._extract_type_annotation(node.annotation)
        self.generic_visit(node)

    def _extract_type_annotation(self, annotation):
        """Extract type names from annotations"""
        if isinstance(annotation, ast.Name):
            self.type_annotations.add(annotation.id)
        elif isinstance(annotation, ast.Attribute):
            if hasattr(annotation.value, 'id'):
                self.type_annotations.add(annotation.value.id)
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                self.type_annotations.add(annotation.value.id)

    def _get_line_context(self, line_no: int) -> str:
        """Get the line context for debugging"""
        if 1 <= line_no <= len(self.lines):
            return self.lines[line_no - 1].strip()
        return ''

class ImportAnalyzer:
    """Main import analyzer class"""
    FRAMEWORK_IMPORTS = {'fastapi': ['Depends', 'HTTPException', 'status', 'Request', 'Response'], 'sqlalchemy': ['AsyncSession', 'Engine', 'select', 'and_', 'or_'], 'prometheus_client': ['Counter', 'Histogram', 'Gauge', 'Summary'], 'sqlmodel': ['SQLModel', 'Field', 'field_validator'], 'typer': ['Typer', 'Argument', 'Option', 'Context']}
    TYPING_IMPORTS = {'typing': ['Dict', 'List', 'Optional', 'Union', 'Tuple', 'Set', 'Callable', 'Any', 'Type', 'TypeVar', 'Generic'], 'typing_extensions': ['Annotated', 'Literal', 'Protocol', 'TypedDict']}

    def __init__(self, src_dir: str='src'):
        self.src_dir = Path(src_dir)
        self.analyses: Dict[str, ImportAnalysis] = {}
        self.global_usage_map: Dict[str, Set[str]] = defaultdict(set)

    def analyze_file(self, file_path: Path) -> ImportAnalysis:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            return ImportAnalysis(str(file_path), [], [], [], [], [])
        try:
            tree = ast.parse(source_code)
            analyzer = ImportUsageAnalyzer(source_code)
            analyzer.visit(tree)
            unused_imports = self._find_unused_imports(analyzer)
            risky_imports = self._identify_risky_imports(analyzer.imports)
            safe_removals = self._identify_safe_removals(unused_imports, analyzer)
            analysis = ImportAnalysis(file_path=str(file_path), imports=analyzer.imports, usages=analyzer.usages, unused_imports=unused_imports, risky_imports=risky_imports, safe_removals=safe_removals)
            return analysis
        except SyntaxError as e:
            print(f'Syntax error in {file_path}: {e}')
            return ImportAnalysis(str(file_path), [], [], [], [], [])

    def _find_unused_imports(self, analyzer: ImportUsageAnalyzer) -> List[ImportInfo]:
        """Find imports that appear to be unused"""
        unused = []
        used_names = {usage.name for usage in analyzer.usages}
        used_names.update(analyzer.type_annotations)
        string_usage = set()
        for string_ref in analyzer.string_references:
            for import_info in analyzer.imports:
                if import_info.name in string_ref or import_info.module in string_ref:
                    string_usage.add(import_info.name)
        used_names.update(string_usage)
        for import_info in analyzer.imports:
            if import_info.name not in used_names:
                if not self._has_special_usage(import_info, analyzer):
                    unused.append(import_info)
        return unused

    def _has_special_usage(self, import_info: ImportInfo, analyzer: ImportUsageAnalyzer) -> bool:
        """Check for special usage patterns that might not be caught by AST"""
        source_lines = analyzer.source_code.splitlines()
        for line in source_lines:
            if line.strip().startswith('@') and import_info.name in line:
                return True
        for line in source_lines:
            if '__all__' in line and import_info.name in line:
                return True
        dynamic_patterns = [f'getattr.*{import_info.name}', f'hasattr.*{import_info.name}', f'setattr.*{import_info.name}', f"'{import_info.name}'", f'"{import_info.name}"']
        source_text = analyzer.source_code
        for pattern in dynamic_patterns:
            if re.search(pattern, source_text):
                return True
        return False

    def _identify_risky_imports(self, imports: List[ImportInfo]) -> List[ImportInfo]:
        """Identify imports that are high-risk for removal"""
        risky = []
        for import_info in imports:
            for framework, names in self.FRAMEWORK_IMPORTS.items():
                if import_info.module == framework or framework in import_info.module or import_info.name in names:
                    risky.append(import_info)
                    break
        return risky

    def _identify_safe_removals(self, unused_imports: List[ImportInfo], analyzer: ImportUsageAnalyzer) -> List[ImportInfo]:
        """Identify imports that are safe to remove"""
        safe = []
        for import_info in unused_imports:
            if self._is_standard_library_import(import_info):
                safe.append(import_info)
            elif self._is_typing_import(import_info) and import_info.name not in analyzer.type_annotations:
                safe.append(import_info)
            elif not self._is_framework_import(import_info):
                safe.append(import_info)
        return safe

    def _is_standard_library_import(self, import_info: ImportInfo) -> bool:
        """Check if import is from standard library"""
        stdlib_modules = {'os', 'sys', 'pathlib', 'datetime', 'time', 'json', 'csv', 'asyncio', 'threading', 'multiprocessing', 'subprocess', 'hashlib', 'base64', 'uuid', 'random', 'math', 'statistics', 'collections', 'itertools', 'functools', 'operator', 'logging', 'warnings', 'traceback', 'inspect', 're', 'string', 'textwrap', 'unicodedata'}
        return import_info.module.split('.')[0] in stdlib_modules

    def _is_typing_import(self, import_info: ImportInfo) -> bool:
        """Check if import is from typing module"""
        for module, names in self.TYPING_IMPORTS.items():
            if import_info.module == module or import_info.name in names:
                return True
        return False

    def _is_framework_import(self, import_info: ImportInfo) -> bool:
        """Check if import is from a framework"""
        for framework in self.FRAMEWORK_IMPORTS:
            if framework in import_info.module:
                return True
        return False

    def analyze_project(self) -> Dict[str, ImportAnalysis]:
        """Analyze all Python files in the project"""
        python_files = list(self.src_dir.rglob('*.py'))
        print(f'Analyzing {len(python_files)} Python files...')
        for file_path in python_files:
            analysis = self.analyze_file(file_path)
            self.analyses[str(file_path)] = analysis
            for usage in analysis.usages:
                self.global_usage_map[usage.name].add(str(file_path))
        return self.analyses

    def generate_removal_plan(self) -> Dict[str, Any]:
        """Generate a comprehensive removal plan"""
        total_unused = 0
        total_safe = 0
        total_risky = 0
        removal_plan = {'summary': {}, 'by_category': {'standard_library': [], 'typing_imports': [], 'framework_imports': [], 'internal_imports': []}, 'files': {}}
        for file_path, analysis in self.analyses.items():
            total_unused += len(analysis.unused_imports)
            total_safe += len(analysis.safe_removals)
            total_risky += len(analysis.risky_imports)
            if analysis.unused_imports:
                removal_plan['files'][file_path] = {'unused_count': len(analysis.unused_imports), 'safe_removals': len(analysis.safe_removals), 'risky_imports': len(analysis.risky_imports), 'imports': []}
                for import_info in analysis.unused_imports:
                    import_data = {'module': import_info.module, 'name': import_info.name, 'line': import_info.line, 'is_safe': import_info in analysis.safe_removals, 'is_risky': import_info in analysis.risky_imports, 'category': self._categorize_import(import_info)}
                    removal_plan['files'][file_path]['imports'].append(import_data)
                    removal_plan['by_category'][import_data['category']].append(import_data)
        removal_plan['summary'] = {'total_files_analyzed': len(self.analyses), 'total_unused_imports': total_unused, 'total_safe_removals': total_safe, 'total_risky_imports': total_risky, 'files_with_unused_imports': len([f for f in removal_plan['files'] if removal_plan['files'][f]['unused_count'] > 0])}
        return removal_plan

    def _categorize_import(self, import_info: ImportInfo) -> str:
        """Categorize an import for removal planning"""
        if self._is_standard_library_import(import_info):
            return 'standard_library'
        elif self._is_typing_import(import_info):
            return 'typing_imports'
        elif self._is_framework_import(import_info):
            return 'framework_imports'
        else:
            return 'internal_imports'

def main():
    """Main analysis function"""
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = 'src'
    analyzer = ImportAnalyzer(src_dir)
    analyses = analyzer.analyze_project()
    plan = analyzer.generate_removal_plan()
    print('\n' + '=' * 60)
    print('IMPORT VERIFICATION ANALYSIS COMPLETE')
    print('=' * 60)
    print(f"Files analyzed: {plan['summary']['total_files_analyzed']}")
    print(f"Total unused imports: {plan['summary']['total_unused_imports']}")
    print(f"Safe removals: {plan['summary']['total_safe_removals']}")
    print(f"Risky imports: {plan['summary']['total_risky_imports']}")
    print(f"Files with unused imports: {plan['summary']['files_with_unused_imports']}")
    print('\nBY CATEGORY:')
    for category, imports in plan['by_category'].items():
        safe_count = sum((1 for imp in imports if imp['is_safe']))
        risky_count = sum((1 for imp in imports if imp['is_risky']))
        print(f"  {category.replace('_', ' ').title()}: {len(imports)} total ({safe_count} safe, {risky_count} risky)")
    print('\nTOP FILES WITH UNUSED IMPORTS:')
    file_scores = [(path, data['unused_count']) for path, data in plan['files'].items()]
    file_scores.sort(key=lambda x: x[1], reverse=True)
    for file_path, count in file_scores[:10]:
        rel_path = file_path.replace(src_dir + '/', '')
        safe_count = plan['files'][file_path]['safe_removals']
        risky_count = plan['files'][file_path]['risky_imports']
        print(f'  {rel_path}: {count} unused ({safe_count} safe, {risky_count} risky)')
    return plan
if __name__ == '__main__':
    main()
