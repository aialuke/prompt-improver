"""Automated Documentation Generation Script for APES
Following 2025 best practices for comprehensive documentation.
"""

import ast
import json
import subprocess
import sys
import time
from pathlib import Path


class DocumentationGenerator:
    """Automated documentation generator following 2025 standards."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.docs_dir = project_root / "docs"
        self.output_dir = self.docs_dir / "_build"
        self.docs_dir.mkdir(exist_ok=True)
        (self.docs_dir / "_static").mkdir(exist_ok=True)
        (self.docs_dir / "_templates").mkdir(exist_ok=True)

    def analyze_codebase(self) -> dict[str, any]:
        """Analyze the codebase for documentation metrics."""
        print("📊 Analyzing codebase for documentation coverage...")
        stats = {
            "total_modules": 0,
            "documented_modules": 0,
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "missing_docs": [],
            "type_coverage": 0.0,
        }
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
                stats["total_modules"] += 1
                if ast.get_docstring(tree):
                    stats["documented_modules"] += 1
                else:
                    stats["missing_docs"].append(
                        f"Module: {py_file.relative_to(self.src_dir)}"
                    )
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats["total_functions"] += 1
                        if ast.get_docstring(node):
                            stats["documented_functions"] += 1
                        else:
                            stats["missing_docs"].append(
                                f"Function: {node.name} in {py_file.relative_to(self.src_dir)}"
                            )
                    elif isinstance(node, ast.ClassDef):
                        stats["total_classes"] += 1
                        if ast.get_docstring(node):
                            stats["documented_classes"] += 1
                        else:
                            stats["missing_docs"].append(
                                f"Class: {node.name} in {py_file.relative_to(self.src_dir)}"
                            )
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
        return stats

    def generate_api_docs(self) -> None:
        """Generate API documentation using sphinx-apidoc."""
        print("📚 Generating API documentation...")
        api_dir = self.docs_dir / "api"
        api_dir.mkdir(exist_ok=True)
        modules = [
            "prompt_improver.core",
            "prompt_improver.ml",
            "prompt_improver.performance",
            "prompt_improver.database",
            "prompt_improver.security",
            "prompt_improver.utils",
        ]
        for module in modules:
            module_path = self.src_dir / module.replace(".", "/")
            if module_path.exists():
                output_file = api_dir / f"{module.split('.')[-1]}.rst"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"{module.split('.')[-1].title()} Module\n")
                    f.write("=" * (len(module.split(".")[-1]) + 7) + "\n\n")
                    f.write(f".. automodule:: {module}\n")
                    f.write("   :members:\n")
                    f.write("   :undoc-members:\n")
                    f.write("   :show-inheritance:\n\n")

    def generate_user_guides(self) -> None:
        """Generate user guide documentation."""
        print("📖 Generating user guides...")
        user_guide_dir = self.docs_dir / "user_guide"
        user_guide_dir.mkdir(exist_ok=True)
        guides = {
            "installation.rst": "Installation Guide\n==================\n\nSystem Requirements\n------------------\n\n* Python 3.11 or higher\n* PostgreSQL 13+ (for database operations)\n* Redis 6+ (for caching)\n* 4GB RAM minimum, 8GB recommended\n\nQuick Installation\n-----------------\n\n.. code-block:: bash\n\n   # Clone the repository\n   git clone https://github.com/your-org/apes.git\n   cd apes\n   \n   # Install dependencies\n   pip install -e .\n   \n   # Set up environment\n   cp .env.example .env\n   # Edit .env with your configuration\n   \n   # Initialize database\n   apes db init\n   \n   # Start services\n   apes start\n\nDocker Installation\n------------------\n\n.. code-block:: bash\n\n   # Using Docker Compose\n   docker-compose up -d\n   \n   # Verify installation\n   docker-compose exec apes apes health\n",
            "quickstart.rst": 'Quick Start Guide\n=================\n\nGetting Started\n--------------\n\n1. **Start the MCP Server**\n\n   .. code-block:: bash\n   \n      apes start --mcp-port 3000\n\n2. **Run Health Checks**\n\n   .. code-block:: bash\n   \n      apes health\n\n3. **Process Your First Prompt**\n\n   .. code-block:: python\n   \n      from prompt_improver import PromptImprovementService\n      \n      service = PromptImprovementService()\n      result = await service.improve_prompt("Your prompt here")\n      print(result)\n\nBasic Configuration\n------------------\n\nEdit your `.env` file:\n\n.. code-block:: bash\n\n   POSTGRES_PASSWORD=your_secure_password\n   REDIS_URL=redis://redis:6379\n   LOG_LEVEL=INFO\n',
        }
        for filename, content in guides.items():
            with open(user_guide_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)

    def run_sphinx_build(self) -> bool:
        """Run Sphinx to build the documentation."""
        print("🏗️  Building documentation with Sphinx...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "sphinx",
                    "furo",
                    "sphinx-autodoc-typehints",
                    "myst-parser",
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "html",
                    str(self.docs_dir),
                    str(self.output_dir / "html"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✅ Documentation built successfully!")
                print(f"📁 Output: {self.output_dir / 'html' / 'index.html'}")
                return True
            print(f"❌ Sphinx build failed: {result.stderr}")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running Sphinx: {e}")
            return False

    def generate_coverage_report(self, stats: dict[str, any]) -> None:
        """Generate documentation coverage report."""
        print("📊 Generating documentation coverage report...")
        module_coverage = (
            stats["documented_modules"] / max(stats["total_modules"], 1) * 100
        )
        function_coverage = (
            stats["documented_functions"] / max(stats["total_functions"], 1) * 100
        )
        class_coverage = (
            stats["documented_classes"] / max(stats["total_classes"], 1) * 100
        )
        overall_coverage = (module_coverage + function_coverage + class_coverage) / 3
        report = {
            "timestamp": time.time(),
            "coverage": {
                "overall": round(overall_coverage, 2),
                "modules": round(module_coverage, 2),
                "functions": round(function_coverage, 2),
                "classes": round(class_coverage, 2),
            },
            "stats": stats,
            "target_coverage": 90.0,
            "meets_target": overall_coverage >= 90.0,
        }
        report_file = self.docs_dir / "coverage_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("📈 Documentation Coverage Report:")
        print(f"   Overall: {overall_coverage:.1f}%")
        print(f"   Modules: {module_coverage:.1f}%")
        print(f"   Functions: {function_coverage:.1f}%")
        print(f"   Classes: {class_coverage:.1f}%")
        print(f"   Target: 90.0% {('✅' if overall_coverage >= 90.0 else '❌')}")

    def generate_all(self) -> bool:
        """Generate complete documentation suite."""
        print("🚀 Starting automated documentation generation...")
        stats = self.analyze_codebase()
        self.generate_api_docs()
        self.generate_user_guides()
        self.generate_coverage_report(stats)
        success = self.run_sphinx_build()
        if success:
            print("🎉 Documentation generation completed successfully!")
            print(f"📖 View docs: file://{self.output_dir / 'html' / 'index.html'}")
        return success


def main():
    """Main function for documentation generation."""
    project_root = Path(__file__).parent.parent
    generator = DocumentationGenerator(project_root)
    success = generator.generate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
