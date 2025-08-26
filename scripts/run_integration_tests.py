#!/usr/bin/env python3
"""Run comprehensive integration tests for DatabaseServices architecture.
Ensures real PostgreSQL and Redis containers are running.
"""

import subprocess
import sys
import time


def check_docker():
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=False, capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_test_containers():
    """Ensure test containers are running."""
    containers_needed = []

    # Check PostgreSQL container
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=postgres-test", "--format", "{{.Names}}"],
        check=False, capture_output=True,
        text=True
    )
    if "postgres-test" not in result.stdout:
        containers_needed.append("postgres")

    # Check Redis container
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=redis-test", "--format", "{{.Names}}"],
        check=False, capture_output=True,
        text=True
    )
    if "redis-test" not in result.stdout:
        containers_needed.append("redis")

    if containers_needed:
        print(f"Starting test containers: {', '.join(containers_needed)}")

        if "postgres" in containers_needed:
            subprocess.run([
                "docker", "run", "-d",
                "--name", "postgres-test",
                "-e", "POSTGRES_PASSWORD=test",
                "-e", "POSTGRES_USER=test",
                "-e", "POSTGRES_DB=test",
                "-p", "5432:5432",
                "postgres:15"
            ], check=False)

        if "redis" in containers_needed:
            subprocess.run([
                "docker", "run", "-d",
                "--name", "redis-test",
                "-p", "6379:6379",
                "redis:7-alpine"
            ], check=False)

        # Wait for containers to be ready
        print("Waiting for containers to be ready...")
        time.sleep(5)


def run_integration_tests():
    """Run the integration test suite."""
    test_files = [
        "tests/integration/database/test_database_services_integration.py",
        "tests/integration/database/test_service_components.py",
    ]

    print("\n" + "=" * 60)
    print("DATABASESERVICES INTEGRATION TEST SUITE")
    print("=" * 60)

    all_passed = True

    for test_file in test_files:
        print(f"\n📋 Running: {test_file}")
        print("-" * 40)

        result = subprocess.run(
            [
                "python", "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--timeout=60",
                "-x"  # Stop on first failure
            ],
            check=False, capture_output=False,
            text=True
        )

        if result.returncode != 0:
            all_passed = False
            print(f"❌ Tests failed in {test_file}")
            break
        print(f"✅ All tests passed in {test_file}")

    return all_passed


def cleanup_containers(force=False):
    """Optionally cleanup test containers."""
    if force or input("\nCleanup test containers? (y/n): ").lower() == 'y':
        print("Cleaning up test containers...")
        subprocess.run(["docker", "stop", "postgres-test"], check=False, capture_output=True)
        subprocess.run(["docker", "rm", "postgres-test"], check=False, capture_output=True)
        subprocess.run(["docker", "stop", "redis-test"], check=False, capture_output=True)
        subprocess.run(["docker", "rm", "redis-test"], check=False, capture_output=True)
        print("✅ Containers cleaned up")


def main():
    """Main test runner."""
    print("🚀 DatabaseServices Integration Test Runner")

    # Check Docker
    if not check_docker():
        print("❌ Docker is not running. Please start Docker first.")
        sys.exit(1)

    # Ensure containers
    ensure_test_containers()

    # Run tests
    try:
        all_passed = run_integration_tests()

        if all_passed:
            print("\n" + "=" * 60)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED")
            print("=" * 60)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Optional cleanup
        cleanup_containers()


if __name__ == "__main__":
    main()
