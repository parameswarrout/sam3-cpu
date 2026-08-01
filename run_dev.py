import os
import sys
import time
import subprocess

WATCH_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = os.path.join(WATCH_DIR, "app.py")

def get_mtimes():
    """Get dictionary of file modification times for all .py files in project."""
    mtimes = {}
    for root, _, files in os.walk(WATCH_DIR):
        if ".git" in root or "__pycache__" in root or ".idea" in root or "scratch" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    mtimes[filepath] = os.path.getmtime(filepath)
                except OSError:
                    pass
    return mtimes

def main():
    print("=" * 65)
    print("🔥 SAM 3 HOT-RELOAD DEV SERVER STARTED!")
    print(f"📁 Watching directory: {WATCH_DIR}")
    print("⚡ Saving any .py file will automatically restart the Web App!")
    print("=" * 65)

    last_mtimes = get_mtimes()
    process = subprocess.Popen([sys.executable, APP_FILE])

    try:
        while True:
            time.sleep(1.0)
            current_mtimes = get_mtimes()
            
            changed = False
            for path, mtime in current_mtimes.items():
                if path not in last_mtimes or mtime > last_mtimes[path]:
                    changed = True
                    rel_path = os.path.relpath(path, WATCH_DIR)
                    print(f"\n🔄 Code change detected in: [ {rel_path} ]")
                    break
            
            if not changed and len(current_mtimes) != len(last_mtimes):
                changed = True
                print("\n🔄 File structure change detected!")

            if changed:
                last_mtimes = current_mtimes
                print("♻️ Restarting SAM 3 Web App server...")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                process = subprocess.Popen([sys.executable, APP_FILE])
    except KeyboardInterrupt:
        print("\n🛑 Stopping SAM 3 Dev Server...")
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

if __name__ == "__main__":
    main()
