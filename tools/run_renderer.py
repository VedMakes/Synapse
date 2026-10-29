"""1-Click Launcher and Universal Server for SYNAPSE GP 2.5D Broadcast Renderer."""

import http.server
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8765
HOST = "127.0.0.1"


def get_paths():
    tools_dir = Path(__file__).resolve().parent
    repo_dir = tools_dir.parent
    renderer_dir = repo_dir / "renderer"
    return repo_dir, renderer_dir


class UniversalRendererHandler(http.server.SimpleHTTPRequestHandler):
    repo_dir, renderer_dir = get_paths()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/renderer/"):
            clean = clean[len("/renderer") :]
        elif clean == "/renderer":
            clean = "/"

        if clean in ("", "/"):
            return str(self.renderer_dir / "index.html")

        rel = clean.lstrip("/")
        candidate = (self.renderer_dir / rel).resolve()
        if candidate.exists():
            return str(candidate)

        root_candidate = (self.repo_dir / rel).resolve()
        if root_candidate.exists():
            return str(root_candidate)

        return str(candidate)

    def log_message(self, format, *args):
        pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, BrokenPipeError):
            pass


def main():
    repo_dir, renderer_dir = get_paths()
    if not renderer_dir.exists():
        print(f"Error: Renderer directory not found at {renderer_dir}", flush=True)
        sys.exit(1)

    os.chdir(renderer_dir)

    server = http.server.ThreadingHTTPServer((HOST, PORT), UniversalRendererHandler)

    url = f"http://{HOST}:{PORT}/"
    print("=" * 60, flush=True)
    print("  SYNAPSE GP 2.5D BROADCAST RENDERER", flush=True)
    print(f"  Live UI: {url}", flush=True)
    print("  (Universal mapping active: 404-free on all URLs)", flush=True)
    print("  (Press Ctrl+C to stop)", flush=True)
    print("=" * 60, flush=True)

    def open_browser():
        time.sleep(0.4)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping renderer server.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
