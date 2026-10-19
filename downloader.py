"""Download engine built on top of the ``yt-dlp`` Python API.

The module turns the user's tree selection into concrete work items and runs
them sequentially inside a worker thread (so the UI stays responsive):

* :func:`plan_jobs` builds one :class:`DownloadJob` **per fragment** when
  chapters were ticked, or a single full-video job otherwise.  File names
  follow ``NN [time-code] Title [ID].mp4`` and are sanitised; playlist
  downloads go into ``<catalogue>/<playlist name>/``.
* :class:`Downloader` executes the jobs with yt-dlp, cutting chapters through
  ``download_sections="*START-END"`` so the whole file is never fetched, and
  muxing/remuxing to ``mp4`` (or converting to ``mp3`` for audio-only).
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
        for job in jobs:
            if self.cancelled:
                self._log("Загрузка остановлена пользователем.")
                break
            processed += 1
            if job.position != current_position:
                current_position = job.position
                self._video_progress(job.position, job.videos_total, job.video.title)
            status = self._run_job(job, yt_dlp)
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
    def _prepare_directories(self, jobs: list[DownloadJob]) -> None:
        for directory in sorted({job.output_dir for job in jobs}):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._log(f"Не удалось создать каталог {directory}: {exc}")

    def _target_path(self, job: DownloadJob) -> Path:
        suffix = ".mp3" if self._fmt.is_audio_only else ".mp4"
        return job.output_dir / f"{job.basename}{suffix}"

    def _existing_output(self, job: DownloadJob, expected: Path) -> bool:
        """``True`` when a finished file for this job is already on disk."""
        if expected.exists():
            return True
        stem = expected.with_suffix("").name
        media = {".mp4", ".mp3", ".mkv", ".webm", ".m4a"}
        return any(
            candidate.suffix.lower() in media
            for candidate in job.output_dir.glob(f"{stem}.*")
        ) if job.output_dir.is_dir() else False

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
        """Build a yt-dlp ``progress_hooks`` entry reporting to the UI."""

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
    def _run_job(self, job: DownloadJob, yt_dlp_module) -> str:
        target = self._target_path(job)
        if self._existing_output(job, target):
            if not self._overwrite:
                self._log(f"Файл уже существует, пропускаю: {target.name}")
                return "skipped"
            # Leftovers of an interrupted run would otherwise block the retry.
            try:
                target.unlink()
            except OSError:
                pass

        options: dict = {
            "format": self._format_selector(job),
            "paths": {"home": str(job.output_dir)},
            "outtmpl": {"default": f"{job.basename}%(ext)s"},
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
        }
        if self._overwrite:
            options["overwrites"] = True
        if not self._fmt.is_audio_only:
            options["merge_output_format"] = "mp4"
        if job.fragment is not None:
            end = job.fragment.end
            options["download_sections"] = (
                f"*{job.fragment.start}-{end}" if end else f"*{job.fragment.start}-"
            )
            options["force_keyframes_at_cuts"] = True

        self._log(f"Скачиваю: {job.label}")
        try:
            with yt_dlp_module.YoutubeDL(options) as ydl:
                ydl.download([job.video.watch_url])
        except KeyboardInterrupt:
            self._log("Текущая загрузка прервана.")
            return "cancelled"
        except Exception as exc:  # noqa: BLE001 - one bad video must not kill the batch
            self._log(f"Ошибка на «{job.label}»: {_first_line(str(exc))}")
            return "failed"

        produced = self._find_output(job, target)
        self._log(f"Сохранено: {produced or target}")
        return "downloaded"

    def _find_output(self, job: DownloadJob, expected: Path) -> str | None:
        """Locate the real file, whose extension may differ from the guess."""
        if expected.exists():
            return str(expected)
        stem = expected.with_suffix("").name
        for candidate in sorted(job.output_dir.glob(f"{stem}.*")):
            if candidate.suffix.lower() in {".mp4", ".mp3", ".mkv", ".webm", ".m4a"}:
                return str(candidate)
        return None


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
