"""A small desktop client compatible with MindEcho's FastAPI endpoints."""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

import requests
import websockets
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class StreamSignals(QObject):
    started = Signal()
    text = Signal(str)
    finished = Signal()
    failed = Signal(str)
    upload_finished = Signal()


class ChatWindow(QMainWindow):
    """One-request-per-WebSocket chat window matching the backend lifecycle."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        super().__init__()
        self.websocket_url = f"ws://{host}:{port}/apps/chat/ws/chat"
        self.upload_url = f"http://{host}:{port}/apps/rag/upload"
        self.session_id = str(uuid.uuid4())
        self.current_answer: QTextEdit | None = None
        self.signals = StreamSignals()
        self.signals.started.connect(self._start_answer)
        self.signals.text.connect(self._append_answer)
        self.signals.finished.connect(self._finish_answer)
        self.signals.failed.connect(self._show_error)
        self.signals.upload_finished.connect(lambda: self.upload_button.setEnabled(True))

        self.setWindowTitle("MindEcho")
        self.resize(860, 680)
        root = QWidget()
        layout = QVBoxLayout(root)

        self.messages = QWidget()
        self.messages_layout = QVBoxLayout(self.messages)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.messages)
        layout.addWidget(self.scroll)

        controls = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入问题；Ctrl+Enter 发送")
        self.input_box.setFixedHeight(82)
        controls.addWidget(self.input_box)
        self.upload_button = QPushButton("上传文件")
        self.upload_button.clicked.connect(self.upload_files)
        controls.addWidget(self.upload_button)
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        controls.addWidget(self.send_button)
        layout.addLayout(controls)
        self.setCentralWidget(root)

    def _add_message(self, text: str, sender: str) -> QTextEdit:
        label = QLabel(sender)
        label.setStyleSheet("font-weight: bold; color: #444;")
        self.messages_layout.addWidget(label)
        box = QTextEdit()
        box.setReadOnly(True)
        box.setFont(QFont("Microsoft YaHei", 10))
        box.setMarkdown(text)
        box.setMinimumHeight(54)
        self.messages_layout.addWidget(box)
        self._scroll_bottom()
        return box

    def _scroll_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def send_message(self) -> None:
        question = self.input_box.toPlainText().strip()
        if not question:
            return
        self.input_box.clear()
        self._add_message(question, "你")
        self.send_button.setEnabled(False)
        threading.Thread(target=self._chat_worker, args=(question,), daemon=True).start()

    def _chat_worker(self, question: str) -> None:
        async def stream() -> None:
            payload = {"type": "chat", "content": question, "session_id": self.session_id}
            async with websockets.connect(self.websocket_url) as socket:
                await socket.send(json.dumps(payload, ensure_ascii=False))
                self.signals.started.emit()
                async for fragment in socket:
                    self.signals.text.emit(fragment)

        try:
            asyncio.run(stream())
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.failed.emit(f"对话请求失败：{exc}")

    def _start_answer(self) -> None:
        self.current_answer = self._add_message("", "MindEcho")

    def _append_answer(self, fragment: str) -> None:
        if self.current_answer is None:
            self._start_answer()
        assert self.current_answer is not None
        self.current_answer.setMarkdown(self.current_answer.toPlainText() + fragment)
        self._scroll_bottom()

    def _finish_answer(self) -> None:
        self.current_answer = None
        self.send_button.setEnabled(True)

    def _show_error(self, message: str) -> None:
        self._add_message(message, "系统")
        self.current_answer = None
        self.send_button.setEnabled(True)

    def upload_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择知识库文件", "", "Documents (*.pdf *.doc *.docx *.xls *.xlsx *.txt *.md *.html *.py *.log)"
        )
        if paths:
            self.upload_button.setEnabled(False)
            threading.Thread(target=self._upload_worker, args=(paths,), daemon=True).start()

    def _upload_worker(self, paths: list[str]) -> None:
        opened = []
        try:
            for path in paths:
                opened.append(("files", (Path(path).name, open(path, "rb"))))
            response = requests.post(self.upload_url, files=opened, timeout=300)
            response.raise_for_status()
            self.signals.failed.emit(f"文件上传结果：{response.json().get('message', 'unknown')}")
        except Exception as exc:
            self.signals.failed.emit(f"文件上传失败：{exc}")
        finally:
            for _, (_, handle) in opened:
                handle.close()
            self.signals.upload_finished.emit()
