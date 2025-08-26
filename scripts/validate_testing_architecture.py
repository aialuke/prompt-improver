#!/usr/bin/env python3
"""Testing Architecture Validation Script.

Validates the testing architecture implementation, performance requirements,
and proper test categorization boundaries.
"""

import subprocess
import sys
import time
from pathlib import Path


class TestingArchitectureValidator:
    """Validates testing architecture implementation."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent.parent
        self.results = {}

    def validate_directory_structure(self) -> bool:
        """Validate test directory structure exists."""
        print("📁 Validating directory structure...")

        required_dirs = [
            "tests/unit/services",
            "tests/unit/repositories",
            "tests/unit/utils",
            "tests/integration/api",
            "tests/integration/services",
            "tests/integration/repositories",
            "tests/contract/rest",
            "tests/contract/mcp",
            "tests/e2e/workflows",
            "tests/e2e/scenarios",
            "tests/utils"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)

        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False

        print("✅ All required directories exist")
        return True

    def validate_configuration_files(self) -> bool:
        """Validate configuration files exist and are properly configured."""
        print("⚙️  Validating configuration files...")

        required_files = [
            "tests/unit/conftest.py",
            "tests/integration/conftest.py",
            "tests/contract/conftest.py",
            "tests/e2e/conftest.py",
            "tests/utils/test_categories.py",
            "pytest.ini",
            "docker-compose.test.yml",
            ".github/workflows/testing-pipeline.yml"
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            print(f"❌ Missing configuration files: {missing_files}")
            return False

        print("✅ All configuration files exist")
        return True

    def validate_test_markers(self) -> bool:
        """Validate pytest markers are properly configured."""
        print("🏷️  Validating pytest markers...")

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "--markers"],
                check=False, cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"❌ Failed to get pytest markers: {result.stderr}")
                return False

            markers_output = result.stdout
            required_markers = [
                "unit:", "integration:", "contract:", "e2e:",
                "api:", "mcp:", "performance:", "workflow:"
            ]

            missing_markers = [marker for marker in required_markers if marker not in markers_output]

            if missing_markers:
                print(f"❌ Missing pytest markers: {missing_markers}")
                return False

            print("✅ All pytest markers configured")
            return True

        except subprocess.TimeoutExpired:
            print("❌ Timeout validating pytest markers")
            return False
        except Exception as e:
            print(f"❌ Error validating pytest markers: {e}")
            return False

    def run_unit_test_performance_validation(self) -> tuple[bool, dict]:
        """Run unit tests and validate performance requirements."""
        print("🏃 Running unit test performance validation...")

        try:
            start_time = time.time()
            result = subprocess.run([
                "python3", "-m", "pytest",
                "tests/unit/",
                "-m", "unit",
                "-v",
                "--tb=short",
                "--timeout=5",  # 5 second timeout for entire unit test suite
                "-x"  # Stop on first failure
            ],
            check=False, cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=10  # Overall timeout
            )

            duration = time.time() - start_time

            performance_data = {
                "duration": duration,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            if result.returncode == 0:
                if duration <= 5.0:  # Unit test suite should complete in <5 seconds
                    print(f"✅ Unit tests passed in {duration:.2f}s")
                    return True, performance_data
                print(f"⚠️  Unit tests passed but took {duration:.2f}s (should be <5s)")
                return False, performance_data
            print(f"❌ Unit tests failed: {result.stderr}")
            return False, performance_data

        except subprocess.TimeoutExpired:
            print("❌ Unit test timeout exceeded")
            return False, {"duration": 10, "success": False, "error": "timeout"}
        except Exception as e:
            print(f"❌ Error running unit tests: {e}")
            return False, {"duration": 0, "success": False, "error": str(e)}

    def validate_test_collection(self) -> tuple[bool, dict]:
        """Validate test collection and categorization."""
        print("📊 Validating test collection...")

        try:
            result = subprocess.run([
                "python3", "-m", "pytest",
                "tests/",
                "--collect-only",
                "--tb=no",
                "-q"
            ],
            check=False, cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30
            )

            if result.returncode != 0:
                print(f"❌ Test collection failed: {result.stderr}")
                return False, {"error": result.stderr}

            # Parse collection output
            output_lines = result.stdout.split('\n')
            test_count_line = [line for line in output_lines if "collected" in line]

            collection_data = {
                "success": True,
                "output": result.stdout,
                "test_count": test_count_line[0] if test_count_line else "unknown"
            }

            print(f"✅ Test collection successful: {collection_data['test_count']}")
            return True, collection_data

        except subprocess.TimeoutExpired:
            print("❌ Test collection timeout")
            return False, {"error": "timeout"}
        except Exception as e:
            print(f"❌ Error validating test collection: {e}")
            return False, {"error": str(e)}

    def validate_test_categorization(self) -> bool:
        """Validate that tests are properly categorized."""
        print("🎯 Validating test categorization...")

        categories = ["unit", "integration", "contract", "e2e"]

        for category in categories:
            try:
                result = subprocess.run([
                    "python", "-m", "pytest",
                    f"tests/{category}/",
                    "-m", category,
                    "--collect-only",
                    "--tb=no",
                    "-q"
                ],
                check=False, cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=15
                )

                if result.returncode != 0:
                    print(f"❌ {category} test categorization failed: {result.stderr}")
                    return False

                # Check if tests were collected
                if "collected 0 items" in result.stdout:
                    print(f"⚠️  No {category} tests found (expected for new architecture)")
                else:
                    print(f"✅ {category} tests properly categorized")

            except Exception as e:
                print(f"❌ Error validating {category} categorization: {e}")
                return False

        return True

    def validate_docker_compose(self) -> bool:
        """Validate Docker Compose test configuration."""
        print("🐳 Validating Docker Compose test configuration...")

        try:
            result = subprocess.run([
                "docker-compose",
                "-f", "docker-compose.test.yml",
                "config"
            ],
            check=False, cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30
            )

            if result.returncode == 0:
                print("✅ Docker Compose test configuration valid")
                return True
            print(f"❌ Docker Compose test configuration invalid: {result.stderr}")
            return False

        except FileNotFoundError:
            print("⚠️  Docker Compose not available, skipping validation")
            return True  # Not a failure, just not available
        except Exception as e:
            print(f"❌ Error validating Docker Compose: {e}")
            return False

    def run_validation(self) -> dict:
        """Run complete testing architecture validation."""
        print("🔍 Starting Testing Architecture Validation\n")

        validations = [
            ("Directory Structure", self.validate_directory_structure),
            ("Configuration Files", self.validate_configuration_files),
            ("Pytest Markers", self.validate_test_markers),
            ("Test Collection", lambda: self.validate_test_collection()[0]),
            ("Test Categorization", self.validate_test_categorization),
            ("Docker Compose", self.validate_docker_compose)
        ]

        results = {}
        all_passed = True

        for name, validator in validations:
            try:
                passed = validator()
                results[name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"❌ {name} validation failed with exception: {e}")
                results[name] = False
                all_passed = False

        # Run performance validation separately
        print("\n🏃 Performance Validation:")
        unit_perf_passed, unit_perf_data = self.run_unit_test_performance_validation()
        results["Unit Test Performance"] = unit_perf_passed
        if not unit_perf_passed:
            all_passed = False

        # Summary
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)

        for name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:<25} {status}")

        overall_status = "✅ ALL VALIDATIONS PASSED" if all_passed else "❌ SOME VALIDATIONS FAILED"
        print(f"\nOverall Status: {overall_status}")

        if all_passed:
            print("\n🎉 Testing Architecture successfully implemented!")
            print("   Ready for test execution with proper boundaries and performance.")
        else:
            print("\n🔧 Please address failed validations before proceeding.")

        return {
            "overall_success": all_passed,
            "individual_results": results,
            "performance_data": unit_perf_data if 'unit_perf_data' in locals() else None
        }


def main():
    """Main validation entry point."""
    validator = TestingArchitectureValidator()
    results = validator.run_validation()

    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)


if __name__ == "__main__":
    main()
