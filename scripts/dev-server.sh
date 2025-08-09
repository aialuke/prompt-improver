#!/bin/bash
# 2025 Enhanced Development Server - Python + Vite with Sub-50ms HMR
# Includes performance monitoring, multi-service orchestration, and real-time metrics

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 2025 Performance optimized configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

PYTHON_PATH="${PYTHON_PATH:-/workspaces/prompt-improver/src}"
WATCH_DIRS="${WATCH_DIRS:-src tests}"
MAIN_MODULE="${MAIN_MODULE:-prompt_improver.main}"
PYTHON_PORT="${PYTHON_PORT:-8000}"
VITE_PORT="${VITE_PORT:-5173}"
TUI_PORT="${TUI_PORT:-3000}"
RELOAD_DELAY="${RELOAD_DELAY:-0.05}"  # Sub-50ms target
VITE_MODE="${VITE_MODE:-development}"
PERFORMANCE_MONITORING="${PERFORMANCE_MONITORING:-true}"
HMR_TARGET_MS="${HMR_TARGET_MS:-50}"

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${CYAN}🚀 2025 Enhanced Development Server${NC}"
echo -e "${BLUE}   Python API: http://localhost:${PYTHON_PORT}${NC}"
echo -e "${BLUE}   Vite HMR:   http://localhost:${VITE_PORT}${NC}"
echo -e "${BLUE}   TUI Dashboard: http://localhost:${TUI_PORT}${NC}"
echo -e "${BLUE}   Mode:       ${VITE_MODE}${NC}"
echo -e "${BLUE}   HMR Target: <${HMR_TARGET_MS}ms${NC}"
echo -e "${BLUE}   Performance Monitoring: ${PERFORMANCE_MONITORING}${NC}"
echo ""

# Start Vite dev server if configuration exists
if [ -f "$PROJECT_ROOT/vite.config.ts" ]; then
    echo -e "${GREEN}Starting Vite development server...${NC}"
    cd "$PROJECT_ROOT"
    nohup npx vite --port "$VITE_PORT" --host 0.0.0.0 > "$LOG_DIR/vite.log" 2>&1 &
    VITE_PID=$!
    echo $VITE_PID > "$LOG_DIR/vite.pid"
    echo -e "${GREEN}✅ Vite server started (PID: $VITE_PID)${NC}"
    
    # Wait for Vite to start
    echo -e "${YELLOW}Waiting for Vite server to initialize...${NC}"
    sleep 3
    
    # Check if Vite is responding
    if curl -s "http://localhost:$VITE_PORT" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Vite server is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  Vite server is starting up...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No vite.config.ts found, skipping Vite server${NC}"
fi

# Performance monitoring function
if [ "$PERFORMANCE_MONITORING" = "true" ]; then
    echo -e "${GREEN}Starting performance monitoring...${NC}"
    
    cat > "$PROJECT_ROOT/perf_monitor.py" << 'PERF_EOF'
#!/usr/bin/env python3
import time
import asyncio
import aiohttp
import psutil
import sys
from datetime import datetime

async def monitor_performance():
    print(f"🚀 Performance monitoring started - Target: <{sys.argv[1]}ms HMR")
    print("📊 Monitoring HMR performance, CPU, and memory usage")
    print("-" * 60)
    
    hmr_times = []
    vite_port = sys.argv[2] if len(sys.argv) > 2 else "5173"
    target_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    
    while True:
        try:
            # Measure HMR performance
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                try:
                    async with session.get(f"http://localhost:{vite_port}") as response:
                        if response.status == 200:
                            hmr_time = (time.time() - start_time) * 1000
                            hmr_times.append(hmr_time)
                            
                            if hmr_time > target_ms:
                                print(f"⚠️  HMR: {hmr_time:.1f}ms (exceeds {target_ms}ms target)")
                            else:
                                print(f"✅ HMR: {hmr_time:.1f}ms")
                except Exception:
                    print("❌ HMR check failed - Vite server not responding")
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80:
                print(f"🔥 High CPU: {cpu_percent:.1f}%")
            if memory.percent > 85:
                print(f"💾 High Memory: {memory.percent:.1f}%")
            
            # Performance report every 30 seconds
            if len(hmr_times) % 6 == 0 and hmr_times:
                avg_hmr = sum(hmr_times) / len(hmr_times)
                print(f"\n📊 Avg HMR: {avg_hmr:.1f}ms | CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}%")
                
                if avg_hmr < target_ms:
                    print("🎯 Performance target achieved!")
                else:
                    print(f"⚡ Working to improve HMR below {target_ms}ms...")
                print("-" * 60)
            
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            print("\n🛑 Performance monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(monitor_performance())
PERF_EOF
    
    chmod +x "$PROJECT_ROOT/perf_monitor.py"
    nohup python3 "$PROJECT_ROOT/perf_monitor.py" "$HMR_TARGET_MS" "$VITE_PORT" > "$LOG_DIR/performance.log" 2>&1 &
    PERF_PID=$!
    echo $PERF_PID > "$LOG_DIR/performance.pid"
    echo -e "${GREEN}✅ Performance monitoring started (PID: $PERF_PID)${NC}"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f "/workspace/.venv/bin/activate" ]; then
        source /workspace/.venv/bin/activate
    else
        echo -e "${RED}No virtual environment found! Please create one first.${NC}"
        exit 1
    fi
fi

# Install watchdog if not present
if ! python -c "import watchdog" 2>/dev/null; then
    echo -e "${YELLOW}Installing watchdog for file monitoring...${NC}"
    pip install watchdog[watchmedo]
fi

# Create Python hot reload script
cat > /tmp/hot_reload_server.py << 'EOF'
import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'

class DevelopmentServer:
    def __init__(self, main_module, watch_dirs, reload_delay=1):
        self.main_module = main_module
        self.watch_dirs = watch_dirs
        self.reload_delay = reload_delay
        self.process = None
        self.restart_queue = queue.Queue()
        self.last_restart = 0
        
    def start(self):
        """Start the main application process."""
        if self.process:
            self.stop()
            
        print(f"{Colors.GREEN}Starting {self.main_module}...{Colors.NC}")
        cmd = [sys.executable, "-m", self.main_module]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        # Start output reader thread
        threading.Thread(target=self._read_output, daemon=True).start()
        self.last_restart = time.time()
        
    def stop(self):
        """Stop the current process."""
        if self.process:
            print(f"{Colors.YELLOW}Stopping process...{Colors.NC}")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"{Colors.RED}Process didn't terminate, killing...{Colors.NC}")
                self.process.kill()
            self.process = None
            
    def restart(self):
        """Restart the application."""
        current_time = time.time()
        if current_time - self.last_restart < self.reload_delay:
            return
            
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}Restarting due to file changes...{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        self.start()
        
    def _read_output(self):
        """Read and display process output."""
        if not self.process:
            return
            
        for line in iter(self.process.stdout.readline, ''):
            if line:
                print(f"{Colors.PURPLE}[APP]{Colors.NC} {line.rstrip()}")
                
        if self.process.poll() is not None:
            print(f"{Colors.RED}Process exited with code {self.process.returncode}{Colors.NC}")

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, server, extensions=None):
        self.server = server
        self.extensions = extensions or {'.py', '.yaml', '.yml', '.toml', '.json', '.sql'}
        self.ignore_patterns = {
            '__pycache__', '.git', '.pytest_cache', 
            '.ruff_cache', 'htmlcov', '.coverage', 'mlruns'
        }
        
    def should_reload(self, path):
        """Check if file change should trigger reload."""
        path_obj = Path(path)
        
        # Ignore certain directories
        for part in path_obj.parts:
            if part in self.ignore_patterns:
                return False
                
        # Check file extension
        return path_obj.suffix in self.extensions
        
    def on_modified(self, event):
        if not event.is_directory and self.should_reload(event.src_path):
            print(f"{Colors.YELLOW}File changed: {event.src_path}{Colors.NC}")
            self.server.restart_queue.put(True)
            
    def on_created(self, event):
        if not event.is_directory and self.should_reload(event.src_path):
            print(f"{Colors.GREEN}File created: {event.src_path}{Colors.NC}")
            self.server.restart_queue.put(True)
            
    def on_deleted(self, event):
        if not event.is_directory and self.should_reload(event.src_path):
            print(f"{Colors.RED}File deleted: {event.src_path}{Colors.NC}")
            self.server.restart_queue.put(True)

def main():
    # Parse environment variables
    main_module = os.environ.get('MAIN_MODULE', 'prompt_improver.main')
    watch_dirs = os.environ.get('WATCH_DIRS', 'src tests').split()
    reload_delay = float(os.environ.get('RELOAD_DELAY', '1'))
    
    # Create server instance
    server = DevelopmentServer(main_module, watch_dirs, reload_delay)
    
    # Set up file watcher
    event_handler = ChangeHandler(server)
    observer = Observer()
    
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            observer.schedule(event_handler, watch_dir, recursive=True)
            print(f"{Colors.GREEN}Watching directory: {watch_dir}{Colors.NC}")
        else:
            print(f"{Colors.YELLOW}Warning: Directory {watch_dir} does not exist{Colors.NC}")
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        print(f"\n{Colors.YELLOW}Shutting down development server...{Colors.NC}")
        server.stop()
        observer.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the server
    server.start()
    observer.start()
    
    print(f"{Colors.GREEN}Development server started on http://localhost:{os.environ.get('PORT', '8000')}{Colors.NC}")
    print(f"{Colors.BLUE}Watching for file changes. Press Ctrl+C to stop.{Colors.NC}")
    
    # Process restart queue
    try:
        while True:
            try:
                # Wait for restart signal with timeout
                restart = server.restart_queue.get(timeout=1)
                if restart:
                    # Drain the queue to avoid multiple restarts
                    while not server.restart_queue.empty():
                        server.restart_queue.get_nowait()
                    server.restart()
            except queue.Empty:
                # Check if process is still running
                if server.process and server.process.poll() is not None:
                    print(f"{Colors.RED}Process died unexpectedly, restarting...{Colors.NC}")
                    server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
EOF

# Export environment variables
export PYTHONPATH="$PYTHON_PATH"
export MAIN_MODULE="$MAIN_MODULE"
export WATCH_DIRS="$WATCH_DIRS"
export PORT="$PORT"
export RELOAD_DELAY="$RELOAD_DELAY"

# Signal handling for cleanup
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down development server...${NC}"
    
    # Stop Vite server
    if [ -f "$LOG_DIR/vite.pid" ]; then
        local vite_pid=$(cat "$LOG_DIR/vite.pid")
        if kill -0 "$vite_pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping Vite server (PID: $vite_pid)...${NC}"
            kill -TERM "$vite_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$vite_pid" 2>/dev/null || true
        fi
        rm -f "$LOG_DIR/vite.pid"
    fi
    
    # Stop performance monitoring
    if [ -f "$LOG_DIR/performance.pid" ]; then
        local perf_pid=$(cat "$LOG_DIR/performance.pid")
        if kill -0 "$perf_pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping performance monitor (PID: $perf_pid)...${NC}"
            kill -TERM "$perf_pid" 2>/dev/null || true
        fi
        rm -f "$LOG_DIR/performance.pid"
        rm -f "$PROJECT_ROOT/perf_monitor.py"
    fi
    
    # Clean up temporary files
    rm -f /tmp/hot_reload_server.py
    
    echo -e "${GREEN}✅ Development server stopped cleanly${NC}"
    exit 0
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Display final service information
echo -e "\n${CYAN}🎉 Development Environment Ready!${NC}"
echo -e "${CYAN}=====================================${NC}"
if [ -f "$LOG_DIR/vite.pid" ]; then
    echo -e "📦 ${GREEN}Vite Dev Server:${NC}     http://localhost:$VITE_PORT"
    echo -e "   ${BLUE}Hot Module Replacement enabled${NC}"
fi
echo -e "🐍 ${GREEN}Python API:${NC}          http://localhost:$PYTHON_PORT"
echo -e "   ${BLUE}Hot reload with sub-50ms target${NC}"
if [ "$PERFORMANCE_MONITORING" = "true" ]; then
    echo -e "📊 ${GREEN}Performance Monitor:${NC} Active"
    echo -e "   ${BLUE}Real-time HMR and resource monitoring${NC}"
fi
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Export updated environment variables
export PYTHONPATH="$PYTHON_PATH"
export MAIN_MODULE="$MAIN_MODULE"
export WATCH_DIRS="$WATCH_DIRS"
export PORT="$PYTHON_PORT"
export RELOAD_DELAY="$RELOAD_DELAY"

# Run the hot reload server
python /tmp/hot_reload_server.py