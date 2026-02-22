import os
import subprocess
import time
import webbrowser
from pathlib import Path

exe_dir = Path(getattr(__import__('sys'), 'executable', __file__)).resolve().parent
# If running from dist, project root is parent of dist.
project_root = exe_dir.parent if exe_dir.name.lower() == 'dist' else exe_dir

python = project_root / '.venv' / 'Scripts' / 'python.exe'
app = project_root / 'indexemaUserDefined.py'

env_path = project_root / '.env'
if not env_path.exists():
    print(f"Missing .env at {env_path}")
    print("Create it with: DHAN_TOKEN=YOUR_TOKEN")
    raise SystemExit(1)

if not python.exists():
    raise SystemExit(f"Python not found at {python}")
if not app.exists():
    raise SystemExit(f"App not found at {app}")

# Run Streamlit in headless mode so it doesn't open the default browser.
subprocess.Popen([
    str(python), '-m', 'streamlit', 'run', str(app),
    '--server.headless', 'true'
], cwd=str(project_root))

# Give Streamlit a moment to start

time.sleep(2)

url = 'http://localhost:8501'
# Prefer Chrome if installed, otherwise fall back to default browser.
chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
for path in chrome_paths:
    if Path(path).exists():
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(path))
        webbrowser.get('chrome').open(url)
        break
else:
    webbrowser.open(url)
