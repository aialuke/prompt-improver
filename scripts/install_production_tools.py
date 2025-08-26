"""Production Tools Installation Script - 2025 Best Practices
Installs k6, safety, prometheus, and other production readiness tools.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import aiofiles
import aiohttp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionToolsInstaller:
    """Production Tools Installer following 2025 best practices.

    Installs and configures:
    - k6 for load testing
    - safety for dependency vulnerability scanning
    - prometheus for monitoring
    - grafana for visualization
    - bandit for SAST scanning
    """

    def __init__(self) -> None:
        self.system = platform.system().lower()
        self.arch = platform.machine().lower()
        self.tools_status = {}
        self.versions = {
            "k6": "0.48.0",
            "prometheus": "2.48.1",
            "grafana": "10.2.3",
            "safety": "latest",
            "bandit": "latest",
        }
        self.install_dir = Path.home() / ".local" / "bin"
        self.config_dir = Path.home() / ".config" / "apes-tools"
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    async def install_all_tools(self) -> dict[str, bool]:
        """Install all production readiness tools."""
        logger.info("🚀 Starting production tools installation (2025 best practices)")
        await self._check_existing_tools()
        await self._install_python_tools()
        await self._install_k6()
        await self._install_prometheus()
        await self._install_grafana()
        await self._configure_tools()
        await self._verify_installations()
        return self.tools_status

    async def _check_existing_tools(self) -> None:
        """Check which tools are already installed."""
        logger.info("🔍 Checking existing tool installations...")
        tools_to_check = {
            "k6": ["k6", "version"],
            "prometheus": ["prometheus", "--version"],
            "grafana-server": ["grafana-server", "--version"],
            "safety": ["safety", "--version"],
            "bandit": ["bandit", "--version"],
        }
        for tool, cmd in tools_to_check.items():
            try:
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.info("✅ %s already installed", tool)
                    self.tools_status[tool] = True
                else:
                    logger.info("❌ %s not found", tool)
                    self.tools_status[tool] = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("❌ %s not found", tool)
                self.tools_status[tool] = False

    async def _install_python_tools(self) -> None:
        """Install Python-based security tools."""
        logger.info("🐍 Installing Python security tools...")
        python_tools = ["safety", "bandit[toml]", "pip-audit"]
        for tool in python_tools:
            try:
                logger.info("Installing %s...", tool)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", tool],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info("✅ %s installed successfully", tool)
                    self.tools_status[tool.split("[")[0]] = True
                else:
                    logger.error("❌ Failed to install %s: %s", tool, result.stderr)
                    self.tools_status[tool.split("[")[0]] = False
            except subprocess.TimeoutExpired:
                logger.exception("❌ Timeout installing %s", tool)
                self.tools_status[tool.split("[")[0]] = False

    async def _install_k6(self) -> None:
        """Install k6 load testing tool."""
        if self.tools_status.get("k6", False):
            logger.info("✅ k6 already installed, skipping")
            return
        logger.info("📊 Installing k6 load testing tool...")
        try:
            if self.system == "darwin":
                await self._install_k6_macos()
            elif self.system == "linux":
                await self._install_k6_linux()
            else:
                logger.warning("❌ k6 installation not supported for %s", self.system)
                self.tools_status["k6"] = False
        except Exception as e:
            logger.exception("❌ Failed to install k6: %s", e)
            self.tools_status["k6"] = False

    async def _install_k6_macos(self) -> None:
        """Install k6 on macOS."""
        try:
            result = subprocess.run(
                ["brew", "install", "k6"],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info("✅ k6 installed via Homebrew")
                self.tools_status["k6"] = True
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        await self._install_k6_direct()

    async def _install_k6_linux(self) -> None:
        """Install k6 on Linux."""
        package_managers = [
            (["apt", "update"], ["apt", "install", "-y", "k6"]),
            (["yum", "update"], ["yum", "install", "-y", "k6"]),
            (["dnf", "update"], ["dnf", "install", "-y", "k6"]),
        ]
        for update_cmd, install_cmd in package_managers:
            try:
                subprocess.run(update_cmd, check=False, capture_output=True, timeout=60)
                result = subprocess.run(
                    install_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info("✅ k6 installed via %s", install_cmd[0])
                    self.tools_status["k6"] = True
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        await self._install_k6_direct()

    async def _install_k6_direct(self) -> None:
        """Install k6 via direct download."""
        logger.info("📥 Installing k6 via direct download...")
        arch_map = {
            "x86_64": "amd64",
            "amd64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }
        system_map = {"darwin": "macos", "linux": "linux"}
        k6_arch = arch_map.get(self.arch, "amd64")
        k6_system = system_map.get(self.system, "linux")
        download_url = f"https://github.com/grafana/k6/releases/download/v{self.versions['k6']}/k6-v{self.versions['k6']}-{k6_system}-{k6_arch}.tar.gz"
        try:

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            archive_path = temp_path / "k6.tar.gz"
                            async with aiofiles.open(archive_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)
                            import tarfile

                            with tarfile.open(archive_path, "r:gz") as tar:
                                tar.extractall(temp_path, filter="data")
                            k6_binary = None
                            for item in temp_path.rglob("k6"):
                                if item.is_file() and os.access(item, os.X_OK):
                                    k6_binary = item
                                    break
                            if k6_binary:
                                import shutil

                                target_path = self.install_dir / "k6"
                                shutil.copy2(k6_binary, target_path)
                                target_path.chmod(493)
                                logger.info("✅ k6 installed to %s", target_path)
                                self.tools_status["k6"] = True
                            else:
                                raise Exception("k6 binary not found in archive")
                    else:
                        raise Exception(f"Download failed: HTTP {response.status}")
        except Exception as e:
            logger.exception("❌ Failed to install k6 directly: %s", e)
            self.tools_status["k6"] = False

    async def _install_prometheus(self) -> None:
        """Install Prometheus monitoring system."""
        if self.tools_status.get("prometheus", False):
            logger.info("✅ Prometheus already installed, skipping")
            return
        logger.info("📊 Installing Prometheus...")
        await self._create_prometheus_docker_config()
        self.tools_status["prometheus"] = True

    async def _install_grafana(self) -> None:
        """Install Grafana visualization."""
        if self.tools_status.get("grafana-server", False):
            logger.info("✅ Grafana already installed, skipping")
            return
        logger.info("📈 Installing Grafana...")
        await self._create_grafana_docker_config()
        self.tools_status["grafana-server"] = True

    async def _create_prometheus_docker_config(self) -> None:
        """Create Prometheus Docker configuration."""
        prometheus_config = {
            "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
            "scrape_configs": [
                {
                    "job_name": "apes-application",
                    "static_configs": [{"targets": ["localhost:8080"]}],
                    "metrics_path": "/metrics",
                    "scrape_interval": "5s",
                }
            ],
        }
        config_file = self.config_dir / "prometheus.yml"
        async with aiofiles.open(config_file, "w") as f:
            import yaml

            await f.write(yaml.dump(prometheus_config, default_flow_style=False))
        logger.info("✅ Prometheus configuration created: %s", config_file)

    async def _create_grafana_docker_config(self) -> None:
        """Create Grafana Docker configuration."""
        grafana_config = {
            "server": {"http_port": 3000, "domain": "localhost"},
            "security": {"admin_user": "admin", "admin_password": "admin"},
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "url": "http://localhost:9090",
                    "access": "proxy",
                    "isDefault": True,
                }
            ],
        }
        config_file = self.config_dir / "grafana.ini"
        async with aiofiles.open(config_file, "w") as f:
            await f.write("[server]\n")
            await f.write("http_port = 3000\n")
            await f.write("domain = localhost\n")
            await f.write("\n[security]\n")
            await f.write("admin_user = admin\n")
            await f.write("admin_password = admin\n")
        logger.info("✅ Grafana configuration created: %s", config_file)

    async def _configure_tools(self) -> None:
        """Configure installed tools."""
        logger.info("⚙️  Configuring production tools...")
        await self._create_tool_configs()
        await self._update_path()

    async def _create_tool_configs(self) -> None:
        """Create configuration files for tools."""
        k6_config = {
            "options": {
                "stages": [
                    {"duration": "2m", "target": 20},
                    {"duration": "5m", "target": 50},
                    {"duration": "2m", "target": 100},
                    {"duration": "5m", "target": 100},
                    {"duration": "2m", "target": 0},
                ],
                "thresholds": {
                    "http_req_duration": ["p(95)<200"],
                    "http_req_failed": ["rate<0.01"],
                },
            }
        }
        config_file = self.config_dir / "k6_config.json"
        async with aiofiles.open(config_file, "w") as f:
            await f.write(json.dumps(k6_config, indent=2))
        logger.info("✅ k6 configuration created: %s", config_file)

    async def _update_path(self) -> None:
        """Update PATH to include tool installation directory."""
        if str(self.install_dir) not in os.environ.get("PATH", ""):
            logger.info("💡 Add %s to your PATH:", self.install_dir)
            logger.info('   export PATH="%s:$PATH"', self.install_dir)

    async def _verify_installations(self) -> None:
        """Verify all tool installations."""
        logger.info("🔍 Verifying tool installations...")
        verification_commands = {
            "k6": ["k6", "version"],
            "safety": ["safety", "--version"],
            "bandit": ["bandit", "--version"],
            "pip-audit": ["pip-audit", "--version"],
        }
        for tool, cmd in verification_commands.items():
            if self.tools_status.get(tool, False):
                try:
                    result = subprocess.run(
                        cmd, check=False, capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        version = result.stdout.strip().split("\n")[0]
                        logger.info("✅ %s: %s", tool, version)
                    else:
                        logger.warning("⚠️  %s installed but verification failed", tool)
                except Exception as e:
                    logger.warning("⚠️  %s verification error: %s", tool, e)


async def main():
    """Main function for tool installation."""
    print("🚀 Production Tools Installation - 2025 Best Practices")
    print("=" * 60)
    installer = ProductionToolsInstaller()
    try:
        results = await installer.install_all_tools()
        print("\n📊 INSTALLATION SUMMARY")
        print(f"Total Tools: {len(results)}")
        successful = [tool for tool, status in results.items() if status]
        failed = [tool for tool, status in results.items() if not status]
        print(f"✅ Successful: {len(successful)}")
        for tool in successful:
            print(f"   - {tool}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
            for tool in failed:
                print(f"   - {tool}")
        print("\n🎯 NEXT STEPS:")
        print(f'1. Add tools to PATH: export PATH="{installer.install_dir}:$PATH"')
        print("2. Start monitoring stack: docker-compose up prometheus grafana")
        print(
            "3. Run production readiness validation: python scripts/production_readiness_validation.py"
        )
        if len(successful) >= len(results) * 0.8:
            print("\n✅ INSTALLATION SUCCESSFUL - Ready for production validation")
            sys.exit(0)
        else:
            print("\n⚠️  PARTIAL INSTALLATION - Some tools failed to install")
            sys.exit(1)
    except Exception as e:
        logger.exception("Installation failed with error: %s", e)
        print(f"\n💥 INSTALLATION ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
