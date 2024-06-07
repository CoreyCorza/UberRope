import subprocess
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, script, python_interpreter):
        self.script = script
        self.python_interpreter = python_interpreter
        self.process = None
        self.run_script()

    def run_script(self):
        if self.process:
            self.process.kill()
        self.process = subprocess.Popen([self.python_interpreter, self.script])

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            self.run_script()

if __name__ == "__main__":
    
    
    # Path to the Python interpreter in your virtual environment
    python_interpreter = r'C:\\Rope Alucard\\Rope\\venv\\Scripts\\python.exe'  # Change this to your venv Python path
    script = r'C:\\Rope Alucard\\Rope\\Rope.py'  # Change this to your main script path
    
    
    
    
    path_to_watch = os.path.dirname(script)  # Directory to monitor

    event_handler = ChangeHandler(script, python_interpreter)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=True)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
