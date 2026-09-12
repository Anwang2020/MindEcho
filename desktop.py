"""Start the optional MindEcho desktop client and its local FastAPI backend."""
from __future__ import annotations

import sys
import threading
import time

import uvicorn
from PySide6.QtWidgets import QApplication

from apps.main import app as backend_app
from interface.desktop import ChatWindow


HOST = "127.0.0.1"
PORT = 8000


def run_backend() -> None:
    uvicorn.run(backend_app, host=HOST, port=PORT, log_level="info")


def main() -> int:
    backend = threading.Thread(target=run_backend, daemon=True, name="mindecho-backend")
    backend.start()
    time.sleep(0.5)

    qt_app = QApplication(sys.argv)
    window = ChatWindow(host=HOST, port=PORT)
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
