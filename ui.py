"""Tkinter/ttk front-end of the YouTube downloader.

The window is organised as a small set of labelled frames:

* **URL row** — an entry plus the «Получить информацию» button.
* **Content tree** — a :class:`CheckTreeview` (a ``ttk.Treeview`` with
  checkbox glyphs) showing ``плейлист → видео → фрагменты`` or, for a single
  video, ``видео → фрагменты``.  Ticking a parent propagates down, ticking a
  child propagates up (``nd`` = partially selected).
* **Format list** — a second ``CheckTreeview`` in single-selection mode that
  lists code | разрешение | кодеки | FPS | размер.  One format is chosen for
  the whole batch.
* **Destination row** — download directory + «Обзор…», persisted in
  ``config.json``.
* **Double progress bar** — the overall playlist progress («Видео 3 из 12»)
  and the per-file progress fed by yt-dlp's ``progress_hooks``.
* **Log pane** — a read-only text widget.

Every long running operation happens in a worker thread; results travel back
through a :class:`queue.Queue` that the Tk event loop drains via ``after``, so
the interface never blocks and no Tk object is touched off the main thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import Config, load_config, save_config
from downloader import DownloadJob, Downloader, check_ffmpeg, plan_jobs
from extractor import ExtractError, Extractor, build_format_list_from_video, parse_url, validate_url
from models import FormatInfo, Fragment, Playlist, Video, format_timestamp
from timestamp_parser import fragments_or_whole_video

#: Poll interval (ms) used to drain the worker -> UI message queue.
_POLL_MS = 100

#: Maximum number of lines kept in the journal window.
_MAX_LOG_LINES = 800

_CHECK_OFF = "☐"
_CHECK_ON = "☑"
_CHECK_ND = "◪"

_STATE_TO_GLYPH = {False: _CHECK_OFF, True: _CHECK_ON, None: _CHECK_ND}
_GLYPH_TO_STATE = {_CHECK_OFF: False, _CHECK_ON: True, _CHECK_ND: None}


def node_state(text: str) -> bool | None:
    """Interpret the leading checkbox glyph of *text*.

    Returns ``True`` when checked, ``False`` when unchecked and ``None`` for
    the mixed ("partially selected") state.
    """
    return _GLYPH_TO_STATE.get((text or "").split(" ", 1)[0], False)


def set_node_state(text: str, state: bool | None) -> str:
    """Replace the leading checkbox glyph of *text* with the one for *state*."""
    remainder = (text or "").split(" ", 1)
    body = remainder[1] if len(remainder) > 1 else ""
    glyph = _STATE_TO_GLYPH.get(state if state is not None else False, _CHECK_OFF)
    return f"{glyph} {body}".strip()


class CheckTreeview(ttk.Treeview):
    """A ``ttk.Treeview`` whose first column carries a checkbox glyph.

    The widget itself knows nothing about playlists: it only renders
    ``☐ / ☑ / ◪`` prefixes and reports toggle events.  Cascade logic (parent →
    children and child → parents) lives in :class:`App`, which owns several
    instances of this view.
    """

    def __init__(self, master, *, columns: tuple[str, ...] = (), on_toggle=None, **kwargs) -> None:
        show = "tree" + (" headings" if columns else "")
        super().__init__(master, columns=columns, show=show, selectmode="browse", **kwargs)
        self.heading("#0", text="")
        self.column("#0", width=420, stretch=True)
        for name in columns:
            self.heading(name, text=name)
            self.column(name, width=110, stretch=False, anchor="w")
        self._on_toggle = on_toggle
        self.bind("<Button-1>", self._handle_click, add=True)
        self.bind("<KeyRelease-Left>", self._handle_space, add=True)
        self.bind("<KeyRelease-Right>", self._handle_space, add=True)

    # ------------------------------------------------------------------ click
    def _handle_click(self, event) -> None:
        """Toggle the checkbox under the cursor / select a format row."""
        region = self.identify_region(event.x, event.y)
        item = self.identify_row(event.y)
        if not item:
            return
        if region == "separator":
            return
        if region in ("tree", "cell") and self._on_toggle is not None:
            self._on_toggle(item)
        else:
            self.selection_set(item)

    def _handle_space(self, event) -> None:
        item = self.focus()
        if item and self._on_toggle is not None:
            self._on_toggle(item)

    # ------------------------------------------------------------------- api
    def state_of(self, item: str) -> bool | None:
        """Checkbox state stored in the ``#0`` text of *item*."""
        return node_state(self.item(item, "text"))

    def set_state(self, item: str, state: bool | None) -> None:
        """Write the checkbox state into the ``#0`` text of *item*."""
        self.item(item, text=set_node_state(self.item(item, "text"), state))

    def toggle(self, item: str) -> bool | None:
        """Flip *item* (mixed counts as unchecked); return the new state."""
        state = True if self.state_of(item) is not True else False
        self.set_state(item, state)
        return state


class App(tk.Tk):
    """Main application window wiring the UI to the extractor/downloader."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Загрузчик контента с YouTube")
        self.geometry("1080x760")
        self.minsize(880, 600)

        self.config_data: Config = load_config()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()

        #: Content model currently displayed.
        self.playlist: Playlist | None = None
        self.single_video: Video | None = None
        #: Tree item id -> domain object.
        self._video_items: dict[str, Video] = {}
        self._fragment_items: dict[str, tuple[str, Fragment]] = {}
        self._fragments_loaded: set[str] = set()
        #: Format list shared by the whole batch.
        self._formats: list[FormatInfo] = []
        self._format_items: dict[str, FormatInfo] = {}

        self._worker: threading.Thread | None = None
        self._downloader: Downloader | None = None
        self._pending_jobs: list[DownloadJob] = []
        self._busy = False
        self._cancel_requested = False

        self._build_ui()
        self._apply_config()
        self._check_environment()
        self.after(_POLL_MS, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================ construction
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(1, weight=5)
        root.rowconfigure(2, weight=2)
        root.rowconfigure(4, weight=1)

        # ---------------------------------------------------------- URL row
        url_frame = ttk.Frame(root)
        url_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        url_frame.columnconfigure(0, weight=1)
        ttk.Label(url_frame, text="Ссылка на YouTube:").grid(row=0, column=0, sticky="w")
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(url_frame, textvariable=self.url_var)
        self.url_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.fetch_button = ttk.Button(url_frame, text="Получить информацию", command=self.on_fetch)
        self.fetch_button.grid(row=1, column=1, sticky="e")
        self.url_entry.bind("<Return>", lambda _e: self.on_fetch())

        # --------------------------------------------- content / format panes
        left = ttk.LabelFrame(root, text="Содержимое ссылки", padding=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=4)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tree = CheckTreeview(left, on_toggle=self._on_tree_toggle)
        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")

        right = ttk.LabelFrame(root, text="Формат (один на всю загрузку)", padding=6)
        right.grid(row=1, column=1, sticky="nsew", pady=4)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.format_tree = CheckTreeview(
            right,
            columns=("Разрешение", "Кодеки", "FPS", "Размер"),
            on_toggle=self._on_format_toggle,
        )
        fmt_scroll = ttk.Scrollbar(right, orient="vertical", command=self.format_tree.yview)
        self.format_tree.configure(yscrollcommand=fmt_scroll.set)
        self.format_tree.grid(row=0, column=0, sticky="nsew")
        fmt_scroll.grid(row=0, column=1, sticky="ns")
        self.format_hint = ttk.Label(
            right,
            text="Формат применяется ко всем выбранным видео; при отсутствии — ближайшее.",
            wraplength=300,
        )
        self.format_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # ------------------------------------------------------- actions row
        action_frame = ttk.Frame(root)
        action_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        action_frame.columnconfigure(1, weight=1)
        ttk.Label(action_frame, text="Каталог:").grid(row=0, column=0, padx=(0, 6))
        self.dir_var = tk.StringVar()
        ttk.Entry(action_frame, textvariable=self.dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(action_frame, text="Обзор…", command=self.on_browse).grid(row=0, column=2, padx=(0, 12))
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(action_frame, text="Перезаписывать", variable=self.overwrite_var).grid(
            row=0, column=3, padx=(0, 12)
        )
        self.download_button = ttk.Button(action_frame, text="Скачать выбранное", command=self.on_download)
        self.download_button.grid(row=0, column=4, padx=(0, 6))
        self.cancel_button = ttk.Button(action_frame, text="Отменить", command=self.on_cancel, state="disabled")
        self.cancel_button.grid(row=0, column=5)

        # ------------------------------------------------------ progress bars
        progress = ttk.Frame(root)
        progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        progress.columnconfigure(1, weight=1)
        ttk.Label(progress, text="Всего:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.overall_bar = ttk.Progressbar(progress, maximum=100.0)
        self.overall_bar.grid(row=0, column=1, sticky="ew")
        self.overall_label = ttk.Label(progress, text="—", width=22, anchor="e")
        self.overall_label.grid(row=0, column=2, padx=(6, 0))
        ttk.Label(progress, text="Файл:").grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.file_bar = ttk.Progressbar(progress, maximum=100.0)
        self.file_bar.grid(row=1, column=1, sticky="ew")
        self.file_label = ttk.Label(progress, text="—", width=22, anchor="e")
        self.file_label.grid(row=1, column=2, padx=(6, 0))

        # ------------------------------------------------------------ log area
        log_frame = ttk.LabelFrame(root, text="Журнал", padding=4)
        log_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        status = ttk.Frame(root)
        status.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.status_var = tk.StringVar(value="Готово.")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    def _apply_config(self) -> None:
        """Restore the last directory / URL from ``config.json``."""
        self.dir_var.set(self.config_data.resolved_download_dir)
        if self.config_data.last_url:
            self.url_var.set(self.config_data.last_url)

    def _check_environment(self) -> None:
        """Warn loudly (but non-fatally) about a missing ffmpeg."""
        ok, message = check_ffmpeg()
        if not ok:
            self.log(message)
            messagebox.showwarning("ffmpeg не найден", message)
        else:
            self.log("ffmpeg найден, можно работать.")

    # =================================================================== logging
    def log(self, message: str) -> None:
        """Append a line to the journal (main thread only).

        The journal is trimmed to ``_MAX_LOG_LINES`` lines so long batches
        with ffmpeg chatter cannot grow it without bound.
        """
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        excess = int(self.log_text.index("end-1c").split(".")[0]) - _MAX_LOG_LINES
        if excess > 0:
            self.log_text.delete("1.0", f"{excess + 1}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_busy(self, busy: bool, *, downloading: bool = False) -> None:
        """Enable/disable the buttons while a worker thread runs."""
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.fetch_button.configure(state=state)
        self.download_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if (busy and downloading) else "disabled")
        if not busy:
            self._cancel_requested = False

    # ============================================================ queue plumbing
    def _post(self, kind: str, payload: object = None) -> None:
        """Thread-safe way for workers to ask the UI to do something."""
        self.messages.put((kind, payload))

    def _drain_queue(self) -> None:
        """Apply every queued UI update, then reschedule ourselves."""
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                self._handle_message(kind, payload)
        except queue.Empty:
            pass
        self.after(_POLL_MS, self._drain_queue)

    def _handle_message(self, kind: str, payload: object) -> None:
        if kind == "log":
            self.log(str(payload))
        elif kind == "error":
            self._set_busy(False)
            self._set_status("Ошибка.")
            messagebox.showerror("Ошибка", str(payload))
        elif kind == "fetch_done":
            assert isinstance(payload, tuple)
            # The fetch worker is finished — release the busy flag *before*
            # filling the tree so follow-up lazy loads (description/formats)
            # are not silently skipped by the "busy" guard.
            self._set_busy(False)
            self._populate_tree(*payload)
        elif kind == "details_done":
            assert isinstance(payload, tuple)
            self._set_busy(False)
            self._apply_video_details(*payload)
        elif kind == "video_progress":
            index, total, title = payload  # type: ignore[misc]
            self._overall(index - 1, total, f"Видео {index} из {total}")
            self.file_bar.configure(value=0)
            self.file_label.configure(text="—")
            self._set_status(f"Скачиваю: {title}")
        elif kind == "file_progress":
            job, percent, text = payload  # type: ignore[misc]
            self._file(percent, text, job)
        elif kind == "job_done":
            job, status = payload  # type: ignore[misc]
            self._set_status(f"{status}: {job.basename}")
            # Keep the overall bar moving even when a job produced no hook
            # callbacks (e.g. it was skipped or failed before downloading).
            self._advance_overall(job)
        elif kind == "done":
            self._overall_complete()
            self._set_busy(False)
            self._set_status("Загрузка завершена.")
            self.log(str(payload))
        elif kind == "cancelled":
            self._set_busy(False)
            self._set_status("Отменено пользователем.")

    # ================================================================ fetching
    def on_fetch(self) -> None:
        """Validate the URL and start metadata extraction in a thread."""
        if self._busy:
            self.log("Дождитесь окончания текущей операции.")
            return
        raw = self.url_var.get().strip()
        try:
            url = validate_url(raw)
            video_id, playlist_id = parse_url(url)
            if video_id is None and playlist_id is None:
                raise ExtractError("В ссылке не найдены ни идентификатор видео, ни идентификатор плейлиста.")
        except ExtractError as exc:
            messagebox.showerror("Некорректная ссылка", str(exc))
            self.log(f"Ошибка ссылки: {exc}")
            return

        self.config_data.last_url = raw
        save_config(self.config_data)
        self._clear_content()
        self._set_busy(True)
        self._set_status("Получаю метаданные…")
        self._start_thread("fetch", lambda: self._fetch_worker(url))

    def _fetch_worker(self, url: str) -> None:
        try:
            playlist, video = Extractor(logger=lambda m: self._post("log", m)).fetch(url)
        except ExtractError as exc:
            self._post("error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never crash the app silently
            self._post("error", f"Непредвиденная ошибка: {exc}")
            return
        self._post("fetch_done", (playlist, video))

    def _start_thread(self, name: str, target) -> None:
        self._worker = threading.Thread(target=target, name=f"yt-{name}", daemon=True)
        self._worker.start()

    def _clear_content(self) -> None:
        """Forget the previous link's data."""
        self.tree.delete(*self.tree.get_children())
        self.format_tree.delete(*self.format_tree.get_children())
        self.playlist = None
        self.single_video = None
        self._video_items.clear()
        self._fragment_items.clear()
        self._fragments_loaded.clear()
        self._formats = []
        self._format_items.clear()
        self._reset_progress()

    # ============================================================== tree build
    def _populate_tree(self, playlist: Playlist | None, video: Video | None) -> None:
        """Fill the content tree and start loading details for one entry."""
        if playlist is not None:
            self.playlist = playlist
            header_iid = f"playlist:{playlist.playlist_id}" if playlist.playlist_id else "playlist"
            if self.tree.exists(header_iid):  # defensive: never clash on re-fill
                header_iid = "playlist"
            header = self.tree.insert(
                "",
                "end",
                iid=header_iid,
                text=f"{_CHECK_ON} Плейлист: {playlist.title} ({playlist.count} видео)",
                open=True,
            )
            target_iid: str | None = None
            for number, item in enumerate(playlist.videos, start=1):
                duration = format_timestamp(item.duration)
                label = f"{number:02d}. {item.title}" + (f"  [{duration}]" if duration else "")
                iid = f"v:{item.video_id}"
                self._video_items[iid] = item
                self.tree.insert(header, "end", iid=iid, text=f"{_CHECK_ON} {label}")
                if video is not None and item.video_id == video.video_id:
                    # The link pointed at this very entry: show its chapters.
                    target_iid = iid
            self.log(f"Отображён плейлист «{playlist.title}» ({playlist.count} видео).")
            if target_iid is None and playlist.videos:
                target_iid = f"v:{playlist.videos[0].video_id}"
            if target_iid is not None:
                self.tree.focus(target_iid)
                self.tree.see(target_iid)
                if video is not None and target_iid == f"v:{video.video_id}":
                    # Replace the flat entry with the fully-fetched object so
                    # its description/formats drive the tree from now on.
                    self._video_items[target_iid] = video
                    if video.details_loaded:
                        # Full metadata already arrived with the fetch —
                        # publish chapters and formats without a new request.
                        self._apply_video_details(
                            target_iid, video, build_format_list_from_video(video)
                        )
                    else:
                        self._load_details(target_iid)
                else:
                    # Flat playlist entries: load description + formats lazily.
                    self._load_details(target_iid)
        elif video is not None:
            self.single_video = video
            iid = "single"
            label = video.title + (f"  [{format_timestamp(video.duration)}]" if video.duration else "")
            self.tree.insert("", "end", iid=iid, text=f"{_CHECK_ON} Видео: {label}", open=True)
            self._video_items[iid] = video
            self._fragments_loaded.add(iid)
            self._apply_fragments(iid, video)
            if video.formats_loaded:
                # Full metadata (incl. formats) already fetched by the
                # extractor — publish it right away, no extra network call.
                self._publish_formats(video, build_format_list_from_video(video))
            else:
                self._load_formats(iid, video)
            self.log(f"Отображено видео «{video.title}».")
        self._set_status("Данные получены, выберите содержимое и формат.")

    def _load_details(self, iid: str) -> None:
        """Fetch description/formats for one playlist entry (lazily, in a thread)."""
        video = self._video_items.get(iid)
        if video is None:
            return
        if self._busy:
            return
        if video.details_loaded and iid in self._fragments_loaded:
            # Description already shown; only the format table may be missing.
            if not self._formats and not video.formats_loaded:
                self._load_formats(iid, video)
            return
        self._set_busy(True)
        self._set_status(f"Загружаю данные: {video.title}")
        self._start_thread("details", lambda: self._details_worker(iid, video))

    def _load_formats(self, iid: str, video: Video) -> None:
        """Request the format table of *video* when it is still unknown."""
        if self._formats or video.formats_loaded:
            return
        if self._busy:
            return
        self._set_busy(True)
        self._set_status(f"Загружаю форматы: {video.title}")
        self._start_thread("formats", lambda: self._details_worker(iid, video))

    def _details_worker(self, iid: str, video: Video) -> None:
        try:
            loaded, formats = Extractor(logger=lambda m: self._post("log", m)).load_video_details(video)
        except ExtractError as exc:
            self._post("log", f"Не удалось получить данные видео: {exc}")
            self._post("details_done", (iid, None, []))
            return
        except Exception as exc:  # noqa: BLE001
            self._post("log", f"Ошибка загрузки данных видео: {exc}")
            self._post("details_done", (iid, None, []))
            return
        self._post("details_done", (iid, loaded, formats))

    def _apply_video_details(self, iid: str, video: Video | None, formats: list[FormatInfo]) -> None:
        if video is None:
            return
        self._apply_fragments(iid, video)
        self._publish_formats(video, formats)

    def _apply_fragments(self, iid: str, video: Video) -> None:
        """(Re)build the fragment children of *iid* from its description."""
        for child in self.tree.get_children(iid):
            self.tree.delete(child)
            self._fragment_items.pop(child, None)
        fragments = fragments_or_whole_video(video.description, video.duration)
        video.fragments = fragments
        self._fragments_loaded.add(iid)
        for fragment in fragments:
            fid = f"{iid}#f{fragment.start}:{id(fragment)}"
            self._fragment_items[fid] = (iid, fragment)
            self.tree.insert(
                iid,
                "end",
                iid=fid,
                text=f"{_CHECK_ON} {format_timestamp(fragment.start)} {fragment.title}",
            )
        if len(fragments) == 1 and fragments[0].whole_video:
            self.log(f"«{video.title}»: тайм-кодов не найдено — доступно всё видео целиком.")

    def _publish_formats(self, video: Video, formats: list[FormatInfo]) -> None:
        """Show the format table once; later videos reuse the same selection."""
        if not formats:
            if not self._formats:
                self.log("Форматы ещё не получены — они появятся при выборе видео.")
            return
        if self._formats:
            return  # one format list per download batch (per the spec)
        self._formats = formats
        remembered = self.config_data.last_format_token
        best = 0
        for index, fmt in enumerate(formats):
            iid = f"fmt{index}"
            self._format_items[iid] = fmt
            self.format_tree.insert(
                "",
                "end",
                iid=iid,
                text=f"{_CHECK_ON} {fmt.note or fmt.token}",
                values=(fmt.resolution, fmt.codecs, fmt.fps_text, fmt.size_text),
            )
            if fmt.token == remembered:
                best = index
        self.format_tree.set_state(f"fmt{best}", True)
        for other in self._format_items:
            if other != f"fmt{best}":
                self.format_tree.set_state(other, False)
        self.format_tree.focus(f"fmt{best}")
        self.log(
            f"Доступно форматов: {len(formats)} (по видео «{video.title}»). "
            "Выбранный формат будет применён ко всей загрузке."
        )

    # ============================================================ tree toggling
    def _on_tree_toggle(self, iid: str) -> None:
        """Cascade a checkbox change down/up the content tree."""
        if not self.tree.exists(iid):
            return
        state = self.tree.toggle(iid)
        children = self.tree.get_children(iid)
        if children:
            for child in self._walk(iid):
                self.tree.set_state(child, state)
        self._refresh_ancestors(iid)

    def _walk(self, iid: str):
        for child in self.tree.get_children(iid):
            yield child
            yield from self._walk(child)

    def _refresh_ancestors(self, iid: str) -> None:
        """Recompute every ancestor's checkbox from its children."""
        parent = self.tree.parent(iid)
        while parent:
            states = [self.tree.state_of(child) for child in self.tree.get_children(parent)]
            if all(s is True for s in states):
                state: bool | None = True
            elif all(s is False for s in states):
                state = False
            else:
                state = None
            self.tree.set_state(parent, state)
            parent = self.tree.parent(parent)

    def _on_format_toggle(self, iid: str) -> None:
        """Radio-like behaviour: exactly one format stays checked."""
        if not self.format_tree.exists(iid):
            return
        for other in self._format_items:
            self.format_tree.set_state(other, other == iid)
        fmt = self._format_items.get(iid)
        if fmt is not None:
            self.config_data.last_format_token = fmt.token
            save_config(self.config_data)
            self._set_status(f"Формат: {fmt.resolution} · {fmt.codecs} · {fmt.fps_text} fps")

    # ================================================================= browsing
    def on_browse(self) -> None:
        directory = filedialog.askdirectory(title="Каталог для скачивания", initialdir=self.dir_var.get())
        if not directory:
            return
        self.dir_var.set(directory)
        self.config_data.download_dir = directory
        if not save_config(self.config_data):
            self.log("Не удалось сохранить config.json (нет прав на запись?).")

    # =============================================================== downloading
    def _selected_content(self) -> list[tuple[Video, list[Fragment]]]:
        """Collect ``(video, fragments)`` pairs marked in the tree."""
        result: list[tuple[Video, list[Fragment]]] = []
        for iid, video in self._video_items.items():
            if not self.tree.exists(iid):
                continue
            children = self.tree.get_children(iid)
            if not children:
                if node_state(self.tree.item(iid, "text")) is True:
                    fragments = video.fragments or [Fragment(0, video.duration, video.title, True)]
                    result.append((video, fragments))
                continue
            chosen: list[Fragment] = []
            for child in children:
                fragment = self._fragment_items.get(child)
                if fragment and node_state(self.tree.item(child, "text")):
                    chosen.append(fragment[1])
            if chosen:
                result.append((video, chosen))
        order = {vid.video_id: index for index, vid in enumerate(self._ordered_videos())}
        result.sort(key=lambda pair: order.get(pair[0].video_id, 1 << 30))
        return result

    def _ordered_videos(self) -> list[Video]:
        if self.playlist is not None:
            return list(self.playlist.videos)
        return [self.single_video] if self.single_video else []

    def _chosen_format(self) -> FormatInfo | None:
        for iid, fmt in self._format_items.items():
            if node_state(self.format_tree.item(iid, "text")) is True:
                return fmt
        return None

    def on_download(self) -> None:
        """Validate the selection and launch the download worker thread."""
        if self._busy:
            return
        items = self._selected_content()
        if not items:
            messagebox.showinfo("Нет выбора", "Отметьте галочками видео или фрагменты для скачивания.")
            return
        fmt = self._chosen_format()
        if fmt is None:
            messagebox.showinfo("Формат не выбран", "Выберите один формат в правом списке.")
            return
        ok, message = check_ffmpeg()
        if not ok:
            messagebox.showerror("ffmpeg не найден", message)
            self.log(message)
            return

        root_dir = Path(self.dir_var.get() or self.config_data.resolved_download_dir).expanduser()
        try:
            root_dir.mkdir(parents=True, exist_ok=True)
            probe = root_dir / ".write-test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            messagebox.showerror("Каталог недоступен", f"Нельзя записывать в {root_dir}:\n{exc}")
            return

        jobs = plan_jobs(items, root_dir, self.playlist.title if self.playlist else None)
        if not jobs:
            messagebox.showinfo("Нет выбора", "Ни один фрагмент не отмечен.")
            return

        self.config_data.download_dir = str(root_dir)
        save_config(self.config_data)

        self._downloader = Downloader(
            fmt,
            on_log=lambda m: self._post("log", m),
            on_video_progress=lambda i, t, title: self._post("video_progress", (i, t, title)),
            on_file_progress=lambda job, percent, text: self._post("file_progress", (job, percent, text)),
            on_job_done=lambda job, status: self._post("job_done", (job, status)),
            overwrite=bool(self.overwrite_var.get()),
        )
        self._pending_jobs = jobs
        self._set_busy(True, downloading=True)
        self._reset_progress()
        self._overall(0, len({job.position for job in jobs}), "Подготовка…")
        self.log(f"Начинаю загрузку: {len(jobs)} файл(ов), формат {fmt.resolution} · {fmt.codecs}.")
        self._start_thread("download", self._download_worker)

    def _download_worker(self) -> None:
        assert self._downloader is not None
        jobs = list(self._pending_jobs)
        summary = self._downloader.run(jobs)
        self._post("cancelled" if self._cancel_requested else "done", summary.as_text())

    def on_cancel(self) -> None:
        if self._downloader is None:
            return
        self._cancel_requested = True
        self._downloader.cancel()
        self.log("Запрошена отмена загрузки…")

    # ================================================================= progress
    def _reset_progress(self) -> None:
        self.overall_bar.configure(value=0, maximum=100.0)
        self.file_bar.configure(value=0)
        self.overall_label.configure(text="—")
        self.file_label.configure(text="—")
        #: Fraction of the overall bar already accounted for by finished jobs.
        self._overall_done = 0.0

    def _overall(self, done: int, total: int, text: str) -> None:
        """Switch the overall bar to per-video mode («Видео N из M»)."""
        self._overall_total_videos = max(total, 1)
        self._overall_done = float(done)
        self.overall_bar.configure(maximum=100.0)
        self.overall_bar.configure(value=min(done / self._overall_total_videos * 100.0, 100.0))
        self.overall_label.configure(text=text)

    def _overall_complete(self) -> None:
        self.overall_bar.configure(value=100.0)
        self.overall_label.configure(text="Завершено")
        self.file_bar.configure(value=100.0)
        self.file_label.configure(text="Готово")

    def _advance_overall(self, job: DownloadJob) -> None:
        """Credit one fully processed video to the overall bar."""
        total = getattr(self, "_overall_total_videos", job.videos_total) or 1
        self._overall_done = min(self._overall_done + 1.0, float(total))
        percent = self._overall_done / total * 100.0
        self.overall_bar.configure(maximum=100.0, value=min(percent, 100.0))
        self.overall_label.configure(
            text=f"Видео {min(int(self._overall_done) + 1, total)} из {total}"
        )

    def _file(self, percent: float, text: str, job: DownloadJob) -> None:
        """Update the per-file bar and blend it into the overall bar.

        ``percent`` is relative to the *current* file — for a chapter cut
        that means the fragment itself (yt-dlp downloads only the requested
        byte range), so the bar really shows the fragment's progress.  The
        label additionally shows how many fragments of this video are done.
        """
        self.file_bar.configure(value=max(0.0, min(percent, 100.0)))
        self.file_label.configure(text=text[:28])
        total_videos = max(job.videos_total, 1)
        step = 100.0 / total_videos
        base = getattr(self, "_overall_done", float(job.position - 1))
        value = min((base + percent / 100.0) * step, 100.0)
        self.overall_bar.configure(maximum=100.0, value=value)
        if job.fragment is not None and job.fragments_of_video > 1:
            caption = (
                f"Видео {job.position} из {total_videos} · "
                f"фрагмент {job.fragment_index}/{job.fragments_of_video}"
            )
        else:
            caption = f"Видео {job.position} из {total_videos}"
        self.overall_label.configure(text=caption[:40])

    # ==================================================================== close
    def _on_close(self) -> None:
        self.config_data.download_dir = self.dir_var.get()
        save_config(self.config_data)
        if self._downloader is not None:
            self._downloader.cancel()
        self.destroy()


def run() -> None:
    """Create the application and enter the Tk main loop."""
    App().mainloop()


if __name__ == "__main__":
    run()
