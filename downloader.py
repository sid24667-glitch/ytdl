"""Download engine built on top of the ``yt-dlp`` Python API.

The module turns the user's tree selection into concrete work items and runs
them sequentially inside a worker thread (so the UI stays responsive):

* :func:`plan_jobs` builds one :class:`DownloadJob` **per fragment** when
  chapters were ticked, or a single full-video job otherwise.  File names
  follow ``NN [time-code] Title [ID].mp4`` and are sanitised; playlist
  downloads go into ``<catalogue>/<playlist name>/``.
* :class:`Downloader` executes the jobs with yt-dlp.  Every chapter is fetched
  by its **own** yt-dlp run whose ``download_ranges`` callback yields exactly
  one time range — the programmatic equivalent of a single
  ``--download-sections "*start-end"`` on the command line.  yt-dlp then sets
  ``section_start``/``section_end``, downloads only that part (HTTP byte
  ranges / matching DASH-HLS segments) and writes it to a file named after
  *that* chapter, so N selected fragments always produce N distinct files.
  Each piece is muxed/remuxed to ``mp4`` (or converted to ``mp3`` for
  audio-only).
* The one format chosen by the user applies to the entire batch; when a video
  does not offer that exact resolution the nearest available height is used
  and the substitution is written to the log.

All callbacks may be invoked from the worker thread, therefore the UI passes
queue-based callables.  Per-item failures (private / deleted / age-restricted
videos) are logged and the batch continues; :class:`DownloadSummary` reports
downloaded / skipped / failed counts at the end.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from models import FormatInfo, Fragment, Video

#: Characters forbidden in Windows file names plus control codes.
_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MAX_FRAGMENT_TITLE = 60
_MAX_NAME_LEN = 180


@dataclass
class DownloadJob:
    """A single download: one video, optionally restricted to a time range."""

    video: Video
    fragment: Fragment | None           # ``None`` -> the whole video
    position: int                       # 1-based index of the video in the batch
    videos_total: int                   # number of videos in the batch
    output_dir: Path
    basename: str
    fragment_index: int = 0             # 1-based chapter number within the video
    fragments_of_video: int = 1

    @property
    def label(self) -> str:
        """Human readable description used in logs and progress lines."""
        if self.fragment is None:
            return f"{self.position}/{self.videos_total} · {self.video.title}"
        return (
            f"{self.position}/{self.videos_total} · {self.video.title} · "
            f"{self.fragment_index}/{self.fragments_of_video} {self.fragment.title}"
        )


@dataclass
class DownloadSummary:
    """Result counters of one batch."""

    downloaded: int = 0
    skipped: int = 0
    failed: int = 0

    def as_text(self) -> str:
        return (
            f"Итог: скачано {self.downloaded}, пропущено (уже существует) {self.skipped}, "
            f"ошибок {self.failed}"
        )


def sanitize(name: str, max_length: int = 90) -> str:
    """Return *name* made safe for use as a file or directory component."""
    cleaned = _UNSAFE_RE.sub("_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "untitled"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    upper = cleaned.upper()
    if upper in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(COM|LPT)\d", upper):
        cleaned = f"_{cleaned}"
    return cleaned


def timecode_name(seconds: int) -> str:
    """File-name friendly time code: ``HH-MM-SS`` or ``MM-SS``."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}" if hours else f"{minutes:02d}-{secs:02d}"


def build_basename(
    video: Video,
    fragment: Fragment | None,
    position: int,
    videos_total: int,
) -> str:
    """Compose ``NN [time-code] Title [ID]`` for one download."""
    pieces: list[str] = []
    if videos_total > 1:
        pieces.append(f"{position:02d}")
    if fragment is not None:
        pieces.append(f"[{timecode_name(fragment.start)}]")
        pieces.append(sanitize(fragment.title, _MAX_FRAGMENT_TITLE))
    else:
        pieces.append(sanitize(video.title))
    pieces.append(f"[{video.video_id}]")
    return " ".join(pieces)[:_MAX_NAME_LEN]


def plan_jobs(
    items: list[tuple[Video, list[Fragment]]],
    root_dir: Path,
    playlist_title: str | None,
) -> list[DownloadJob]:
    """Translate the tree selection into individual download jobs.

    Args:
        items: ``(video, selected fragments)`` pairs in playlist order.  A
            fragment flagged ``whole_video`` produces a single full download.
        root_dir: Directory chosen by the user.
        playlist_title: Playlist name, or ``None`` for a standalone video.
    """
    items = [(video, frags) for video, frags in items if frags]
    videos_total = len(items)
    output_dir = root_dir / sanitize(playlist_title) if playlist_title else root_dir

    jobs: list[DownloadJob] = []
    for position, (video, fragments) in enumerate(items, start=1):
        ordered = sorted(fragments, key=lambda f: f.start)
        if len(ordered) == 1 and ordered[0].whole_video:
            jobs.append(
                DownloadJob(
                    video=video,
                    fragment=None,
                    position=position,
                    videos_total=videos_total,
                    output_dir=output_dir,
                    basename=build_basename(video, None, position, videos_total),
                )
            )
            continue
        for number, fragment in enumerate(ordered, start=1):
            jobs.append(
                DownloadJob(
                    video=video,
                    fragment=fragment,
                    position=position,
                    videos_total=videos_total,
                    output_dir=output_dir,
                    basename=build_basename(video, fragment, position, videos_total),
                    fragment_index=number,
                    fragments_of_video=len(ordered),
                )
            )
    return jobs


LogFn = Callable[[str], None]
VideoProgressFn = Callable[[int, int, str], None]
FileProgressFn = Callable[[DownloadJob, float, str], None]
DoneFn = Callable[[DownloadJob, str], None]


class _YtDlpLogBridge:
    """Route yt-dlp's console/warning/error output into the application log.

    Implemented for two reasons: the user sees what happens during ffmpeg
    post-processing (remux / mp3 conversion / metadata) instead of a silent
    freeze, and error text is no longer swallowed by ``quiet=True``.  The
    noisy per-chunk download lines are skipped — the progress bar already
    shows them.
    """

    def __init__(self, sink: LogFn) -> None:
        self._sink = sink

    def debug(self, msg: str) -> None:
        self._emit(msg)

    def info(self, msg: str) -> None:
        self._emit(msg)

    def warning(self, msg: str) -> None:
        self._emit(msg)

    def error(self, msg: str) -> None:
        self._emit(msg)

    @staticmethod
    def _is_noise(msg: str) -> bool:
        if "[download] Destination:" in msg:
            return True
        # Per-chunk percentage lines produced by the built-in progress meter.
        return bool(re.search(r"\[download\]\s+[0-9.]+% of", msg))

    def _emit(self, msg: str) -> None:
        text = msg.strip()
        if not text or self._is_noise(text):
            return
        if len(text) > 240:
            text = text[:240] + "…"
        self._sink(text)


class Downloader:
    """Run a list of :class:`DownloadJob` objects through yt-dlp."""

    def __init__(
        self,
        fmt: FormatInfo,
        *,
        on_log: LogFn | None = None,
        on_video_progress: VideoProgressFn | None = None,
        on_file_progress: FileProgressFn | None = None,
        on_job_done: DoneFn | None = None,
        overwrite: bool = False,
    ) -> None:
        self._fmt = fmt
        self._log = on_log or (lambda message: None)
        self._video_progress = on_video_progress or (lambda index, total, title: None)
        self._file_progress = on_file_progress or (lambda job, percent, text: None)
        self._job_done = on_job_done or (lambda job, status: None)
        self._overwrite = overwrite
        #: Set from the UI thread to abort between jobs / inside progress hooks.
        self.cancelled = False

    # ------------------------------------------------------------------- api
    def run(self, jobs: list[DownloadJob]) -> DownloadSummary:
        """Execute every job sequentially and return aggregate counters."""
        summary = DownloadSummary()
        if not jobs:
            self._log("Ничего не выбрано для скачивания.")
            return summary

        try:
            import yt_dlp
        except ImportError:
            self._log("Модуль yt-dlp не установлен. Выполните: pip install -r requirements.txt")
            summary.failed = len(jobs)
            return summary

        self._prepare_directories(jobs)
        current_position = 0
        processed = 0
        for group in self._group_by_video(jobs):
            if self.cancelled:
                self._log("Загрузка остановлена пользователем.")
                break
            lead = group[0]
            if lead.position != current_position:
                current_position = lead.position
                self._video_progress(lead.position, lead.videos_total, lead.video.title)
            statuses = self._run_group(group, yt_dlp)
            for job, status in zip(group, statuses):
                processed += 1
                self._job_done(job, status)
                if status == "downloaded":
                    summary.downloaded += 1
                elif status == "skipped":
                    summary.skipped += 1
                elif status == "failed":
                    summary.failed += 1

        if self.cancelled and processed < len(jobs):
            self._log(f"Не выполнено из-за отмены: {len(jobs) - processed}")
        return summary

    def cancel(self) -> None:
        """Ask the running batch to stop as soon as possible."""
        self.cancelled = True

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _group_by_video(jobs: list[DownloadJob]) -> list[list[DownloadJob]]:
        """Group consecutive jobs of the same video into one batch.

        All chapters of one video are then fetched by a *single* yt-dlp run
        with several ``download_ranges`` — exactly like passing multiple
        ``--download-sections`` on the command line.
        """
        groups: list[list[DownloadJob]] = []
        for job in jobs:
            if groups and groups[-1][0].video.video_id == job.video.video_id:
                groups[-1].append(job)
            else:
                groups.append([job])
        return groups

    def _prepare_directories(self, jobs: list[DownloadJob]) -> None:
        for directory in sorted({job.output_dir for job in jobs}):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._log(f"Не удалось создать каталог {directory}: {exc}")

    def _expected_path(self, job: DownloadJob) -> Path:
        """Expected final file of *job*.

        Every job — full video or a single chapter — owns a deterministic
        file name built from its own ``basename`` (which already contains the
        chapter time code and title), so three selected chapters produce
        three distinct files.  The same template is handed to yt-dlp as
        ``outtmpl`` for each per-chapter run.
        """
        suffix = ".mp3" if self._fmt.is_audio_only else ".mp4"
        return job.output_dir / f"{job.basename}{suffix}"

    def _existing_output(self, expected: Path) -> bool:
        """``True`` when a finished file for this path is already on disk."""
        if expected.exists():
            return True
        stem = expected.with_suffix("").name
        media = {".mp4", ".mp3", ".mkv", ".webm", ".m4a"}
        directory = expected.parent
        if not directory.is_dir():
            return False
        return any(
            candidate.suffix.lower() in media
            for candidate in directory.glob(f"{stem}*")
        )

    def _format_selector(self, job: DownloadJob) -> str:
        """Map the globally chosen format onto this specific video.

        ``video.available_heights`` is filled by the UI for videos whose full
        metadata was already fetched; for the rest the requested ceiling is
        trusted and yt-dlp itself degrades to the best available stream.
        """
        if self._fmt.is_audio_only:
            return "bestaudio/best"
        limit = self._fmt.height_limit
        if limit is None:
            return "bv+ba/b"
        available = set(getattr(job.video, "available_heights", None) or ())
        if available and limit not in available:
            nearest = min(available, key=lambda h: (abs(h - limit), -h))
            self._log(
                f"«{job.video.title}»: разрешение {limit}p недоступно, "
                f"беру ближайшее {nearest}p."
            )
            return f"bv*[height<={nearest}]+ba/b[height<={nearest}]"
        return self._fmt.token

    def _postprocessors(self) -> list[dict]:
        if self._fmt.is_audio_only:
            return [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"},
                {"key": "FFmpegMetadata"},
            ]
        return [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
            {"key": "FFmpegMetadata"},
        ]

    def _make_hook(self, job: DownloadJob) -> Callable[[dict], None]:
        """Build a yt-dlp ``progress_hooks`` entry reporting to the UI.

        Each run (full video or one chapter cut) gets its own hook bound to
        exactly one :class:`DownloadJob`, so attribution is unambiguous.  For
        chapter cuts yt-dlp reports progress of the *cut piece* only (just
        the requested byte ranges / fragments are fetched), therefore the
        percentages are relative to the fragment size — which is what the
        per-file bar should show.
        """

        def hook(data: dict) -> None:
            if self.cancelled:
                raise KeyboardInterrupt("Отменено пользователем")
            status = data.get("status")
            if status == "downloading":
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                done = data.get("downloaded_bytes") or 0
                percent = (done / total * 100.0) if total else 0.0
                text = f"{percent:5.1f}% из {human_size(total)}"
                speed = data.get("speed")
                if speed:
                    text += f" · {human_size(speed)}/с"
                eta = data.get("eta")
                if eta:
                    text += f" · осталось ~{eta} с"
                self._file_progress(job, percent, text)
            elif status == "finished":
                self._file_progress(job, 100.0, "Обработка через ffmpeg…")

        return hook

    # ------------------------------------------------------------ execution
    def _run_group(self, group: list[DownloadJob], yt_dlp_module) -> list[str]:
        """Process every job of one video in a single logical step.

        Cutting happens per **fragment**: each chapter job is fetched by its
        own yt-dlp run whose ``download_ranges`` callback yields exactly one
        time range — the programmatic equivalent of a single
        ``--download-sections "*START-END"`` on the command line.  yt-dlp then
        downloads only that part (HTTP byte ranges / matching DASH-HLS
        segments) and writes it to a file named after *that* chapter, so N
        selected fragments always produce N distinct files with correct names.
        Returns one status string per job of *group*.
        """
        statuses: list[str] = []
        pending: list[tuple[int, DownloadJob]] = []   # (index into statuses, job)
        for job in group:
            path = self._expected_path(job)
            if self._existing_output(path):
                if not self._overwrite:
                    self._log(f"Файл уже существует, пропускаю: {path.name}")
                    statuses.append("skipped")
                    continue
                # Leftovers of an interrupted run would otherwise block retry.
                try:
                    path.unlink()
                except OSError:
                    pass
            statuses.append("pending")
            pending.append((len(statuses) - 1, job))
        if not pending:
            return statuses

        for position, job in pending:
            if self.cancelled:
                break
            statuses[position] = self._run_single(job, yt_dlp_module)
        return statuses

    def _base_options(self, job: DownloadJob) -> dict:
        """yt-dlp options shared by full-video and chapter runs."""
        options: dict = {
            "format": self._format_selector(job),
            "paths": {"home": str(job.output_dir)},
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "progress_hooks": [self._make_hook(job)],
            "postprocessors": self._postprocessors(),
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 3,
            # yt-dlp's own messages (ffmpeg post-processing passes, cut
            # results) are routed into the app journal so the user can see
            # what ffmpeg is doing between download chunks.
            "logger": _YtDlpLogBridge(self._log),
        }
        if self._overwrite:
            options["overwrites"] = True
        if not self._fmt.is_audio_only:
            options["merge_output_format"] = "mp4"
        return options

    def _run_single(self, job: DownloadJob, yt_dlp_module) -> str:
        """Execute one job (whole video or one chapter cut); return status."""
        expected = self._expected_path(job)
        options = self._base_options(job)
        # One deterministic template per run: the job's own basename, which
        # already contains the chapter time code and title.
        options["outtmpl"] = {"default": f"{job.basename}.%(ext)s"}

        if job.fragment is not None:
            # CLI-equivalent of --download-sections "*START-END".  The Python
            # API parameter is ``download_ranges`` and must be a callback
            # yielding the time ranges; passing a string under any name is
            # silently ignored by YoutubeDL (that bug made us download the
            # whole file instead of only the selected chapters).
            start = float(max(int(job.fragment.start), 0))
            end = (
                float(job.fragment.end)
                if job.fragment.end is not None
                else float("inf")
            )
            options["download_ranges"] = _single_range(start, end)
            options["force_keyframes_at_cuts"] = True
            self._log(
                "Скачиваю только фрагмент (--download-sections-эквивалент): "
                f"{job.label} [{timecode_name(int(start))}–"
                f"{timecode_name(int(end)) if end != float('inf') else 'конец'}]"
            )
        else:
            self._log(f"Скачиваю всё видео: {job.label}")

        ok = True
        try:
            with yt_dlp_module.YoutubeDL(options) as ydl:
                ydl.download([job.video.watch_url])
        except KeyboardInterrupt:
            self._log("Текущая загрузка прервана.")
            return "cancelled"
        except Exception as exc:  # noqa: BLE001 - one bad item must not kill the batch
            self._log(f"Ошибка на «{job.label}»: {_first_line(str(exc))}")
            ok = False

        if not ok:
            return "failed"
        produced = self._find_output(job, expected)
        if produced:
            self._log(f"Сохранено: {produced}")
            return "downloaded"
        self._log(f"Фрагмент не найден среди результатов: {expected.name}")
        return "failed"

    def _find_output(self, job: DownloadJob, expected: Path) -> str | None:
        """Locate the real file, whose extension may differ from the guess."""
        if expected.exists():
            return str(expected)
        stem = expected.with_suffix("").name
        media = {".mp4", ".mp3", ".mkv", ".webm", ".m4a"}
        for candidate in sorted(job.output_dir.glob(f"{stem}*")):
            if candidate.suffix.lower() in media:
                return str(candidate)
        return None


def _single_range(start: float, end: float) -> Callable[[dict, object], list[dict]]:
    """Build a ``download_ranges`` callback for exactly one time range.

    Mirrors one ``--download-sections "*START-END"`` option: yt-dlp sets
    ``section_start``/``section_end`` on the entry and downloads **only**
    that part of the media.  An open-ended chapter (``end == inf``) maps to
    "up to the end of the video".
    """

    def provider(_info_dict: dict, _ydl: object) -> list[dict]:
        return [{"start_time": start, "end_time": end}]

    return provider


def human_size(size: float | None) -> str:
    """Render byte counts (and byte/second rates) compactly."""
    if not size:
        return "?"
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024:
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} ПБ"


def _first_line(text: str) -> str:
    """Keep only the first non-empty line of a (possibly long) error message."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:300]
    return text.strip()[:300]


def check_ffmpeg() -> tuple[bool, str]:
    """Verify that ``ffmpeg``/``ffprobe`` are reachable; return an explanation."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        return False, (
            "Не найдены утилиты: " + ", ".join(missing) + ".\n"
            "ffmpeg обязателен — установите его и добавьте в PATH:\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )
    return True, ""
