"""1-Click Launcher for the SYNAPSE GP 2.5D Broadcast Renderer."""

import http.server
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8765
HOST = "127.0.0.1"


def main():
    root = Path(__file__).resolve().parent.parent
    renderer_dir = root / "renderer"
    if not renderer_dir.exists():
        print(f"Error: Renderer directory not found at {renderer_dir}", flush=True)
        sys.exit(1)

    os.chdir(renderer_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer((HOST, PORT), CustomHandler)

    url = f"http://{HOST}:{PORT}/"
    print("=" * 60, flush=True)
    print("  SYNAPSE GP 2.5D BROADCAST RENDERER", flush=True)
    print(f"  Live UI: {url}", flush=True)
    print("  (Press Ctrl+C to stop)", flush=True)
    print("=" * 60, flush=True)

    def open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping renderer server.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
