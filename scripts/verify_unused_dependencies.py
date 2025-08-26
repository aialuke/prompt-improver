"""Verification script to detect unused dependencies in the codebase."""

import ast
import sys
from pathlib import Path


def extract_imports_from_file(file_path: Path) -> set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    except (UnicodeDecodeError, SyntaxError):
        pass
    return imports


def find_all_imports(root_dir: Path) -> set[str]:
    """Find all imports used in the codebase."""
    all_imports = set()
    for py_file in root_dir.rglob("*.py"):
        if "venv" in str(py_file) or ".venv" in str(py_file):
            continue
        all_imports.update(extract_imports_from_file(py_file))
    return all_imports


def parse_requirements(req_file: Path) -> list[str]:
    """Parse requirements file and extract package names."""
    packages = []
    if not req_file.exists():
        return packages
    with open(req_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and (not line.startswith("#")) and (not line.startswith("-e")):
                pkg_name = (
                    line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0]
                )
                packages.append(pkg_name)
    return packages


def normalize_package_name(pkg_name: str) -> str:
    """Normalize package name for comparison."""
    return pkg_name.replace("-", "_").lower()


def main():
    """Main verification function."""
    root_dir = Path(__file__).parent.parent
    print("🔍 Analyzing dependencies usage...")
    used_imports = find_all_imports(root_dir)
    used_normalized = {normalize_package_name(imp) for imp in used_imports}
    import tomllib

    pyproject_file = root_dir / "pyproject.toml"
    all_packages = set()
    if pyproject_file.exists():
        with open(pyproject_file, "rb") as f:
            pyproject_data = tomllib.load(f)
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        for dep in dependencies:
            pkg_name = (
                dep.split(">=")[0]
                .split("==")[0]
                .split("<=")[0]
                .split("~=")[0]
                .split("[")[0]
            )
            all_packages.add(pkg_name.strip('"').strip("'"))
        optional_deps = pyproject_data.get("project", {}).get(
            "optional-dependencies", {}
        )
        for group_deps in optional_deps.values():
            for dep in group_deps:
                pkg_name = (
                    dep.split(">=")[0]
                    .split("==")[0]
                    .split("<=")[0]
                    .split("~=")[0]
                    .split("[")[0]
                )
                all_packages.add(pkg_name.strip('"').strip("'"))
    known_unused = {
        "adal",
        "azure-common",
        "azure-core",
        "azure-graphrbac",
        "azure-mgmt-authorization",
        "azure-mgmt-containerregistry",
        "azure-mgmt-core",
        "azure-mgmt-keyvault",
        "azure-mgmt-network",
        "azure-mgmt-resource",
        "azure-mgmt-storage",
        "azureml-core",
        "boto3",
        "botocore",
        "google-api-core",
        "google-auth",
        "google-cloud-core",
        "google-cloud-storage",
        "google-crc32c",
        "google-resumable-media",
        "googleapis-common-protos",
        "pip-audit",
        "pipdeptree",
        "pipreqs",
        "radon",
        "knack",
        "mando",
        "docopt",
        "yarg",
    }
    unused_packages = []
    potentially_unused = []
    for pkg in all_packages:
        pkg_normalized = normalize_package_name(pkg)
        if pkg in known_unused:
            unused_packages.append(pkg)
        elif pkg_normalized not in used_normalized:
            if pkg_normalized in {
                "setuptools",
                "wheel",
                "pip",
                "packaging",
                "certifi",
                "urllib3",
                "charset_normalizer",
                "idna",
                "requests",
                "typing_extensions",
                "zipp",
                "importlib_metadata",
            }:
                continue
            potentially_unused.append(pkg)
    print(f"\n✅ Found {len(used_imports)} unique imports in codebase")
    print(f"📦 Found {len(all_packages)} packages in requirements files")
    if unused_packages:
        print(f"\n❌ Confirmed unused packages ({len(unused_packages)}):")
        for pkg in sorted(unused_packages):
            print(f"  - {pkg}")
    if potentially_unused:
        print(f"\n⚠️  Potentially unused packages ({len(potentially_unused)}):")
        for pkg in sorted(potentially_unused):
            print(f"  - {pkg}")
    if not unused_packages and (not potentially_unused):
        print("\n🎉 No unused dependencies detected!")
    return len(unused_packages) + len(potentially_unused)


if __name__ == "__main__":
    sys.exit(main())
