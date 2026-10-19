"""Download engine built on top of the ``yt-dlp`` Python API.

The module turns the user's tree selection into concrete work items and runs
them sequentially inside a worker thread (so the UI stays responsive):

* :func:`plan_jobs` builds one :class:`DownloadJob` **per fragment** when
  chapters were ticked, or a single full-video job otherwise.  File names
  follow ``NN [time-code] Title [ID].mp4`` and are sanitised; playlist
  downloads go into ``<catalogue>/<playlist name>/``.
* :class:`Downloader` executes the jobs with yt-dlp.  All chapters of one
  video are fetched by a **single** yt-dlp run, exactly like several
  ``--download-sections "*start-end"`` options on the command line: the
  documented ``download_ranges`` callback yields every chapter range at once,
  yt-dlp sets ``section_start``/``section_end`` per split entry and downloads
  only that part (HTTP byte ranges / matching DASH-HLS segments), while the
  ``section_title``/``section_number`` output template keeps the pieces in
  separate files.  Each piece is muxed/remuxed to ``mp4`` (or converted to
  ``mp3`` for audio-only).
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

    def _target_path(self, job: DownloadJob) -> Path:
        """Expected final file for a full-video download."""
        suffix = ".mp3" if self._fmt.is_audio_only else ".mp4"
        return job.output_dir / f"{job.basename}{suffix}"

    def _expected_path(self, job: DownloadJob) -> Path:
        """Expected final file of *job*, chapter-aware.

        For chapters the name mirrors the group ``outtmpl``: the stem of the
        video's first chapter plus the sanitised section title in brackets —
        e.g. ``01 [00-00] Intro [vidAAA] [Intro].mp4``.  yt-dlp fills
        ``%(section_title)s`` from the range's ``title``, so every piece maps
        back to exactly one :class:`DownloadJob`.
        """
        suffix = ".mp3" if self._fmt.is_audio_only else ".mp4"
        if job.fragment is None:
            return self._target_path(job)
        return job.output_dir / f"{self._chapter_stem(job)}{suffix}"

    def _expected_paths(self, group: list[DownloadJob]) -> list[Path]:
        """One expected output file per job of the group, in job order."""
        return [self._expected_path(job) for job in group]

    @staticmethod
    def _chapter_stem(job: DownloadJob) -> str:
        """File-name stem of one chapter, mirroring the ``outtmpl`` used.

        ``%(section_title)s`` is filled from the range's ``title``, which we
        sanitise ourselves; the stem therefore matches yt-dlp's real output
        even when two chapters share a title (yt-dlp then appends the section
        number inside the brackets — handled by the glob in
        :meth:`_find_output`).
        """
        assert job.fragment is not None
        base = build_basename(job.video, job.fragment, job.position, job.videos_total)
        stem = re.sub(r"\[[^\[\]]*\]\s*$", "", base).strip()
        title = sanitize(job.fragment.title, _MAX_FRAGMENT_TITLE)
        return f"{stem} [{title}]"

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

    def _make_hook(
        self, pending: list[tuple[DownloadJob, Path]]
    ) -> Callable[[dict], None]:
        """Build a yt-dlp ``progress_hooks`` entry reporting to the UI.

        For chapter cuts yt-dlp reports the progress of the *cut piece* (only
        the requested byte range / fragments are fetched), so percentages are
        relative to the fragment size, not to the full video.  The current
        piece is recognised via its ``section_title``/``section_start``; for a
        plain full-video download the only job of the group is used.
        """

        def find_job(data: dict) -> DownloadJob:
            title = data.get("section_title")
            if title:
                for job, _ in pending:
                    if (
                        job.fragment is not None
                        and sanitize(job.fragment.title, _MAX_FRAGMENT_TITLE) == title
                    ):
                        return job
            start = data.get("section_start")
            end = data.get("section_end")
            if start is not None:
                for job, _ in pending:
                    frag = job.fragment
                    if frag is None:
                        continue
                    same_start = abs(frag.start - start) < 1.0
                    same_end = (
                        (end is None and frag.end is None)
                        or (end is not None and frag.end is not None
                            and abs(frag.end - end) < 1.0)
                    )
                    if same_start and same_end:
                        return job
            return pending[0][0]

        def hook(data: dict) -> None:
            if self.cancelled:
                raise KeyboardInterrupt("Отменено пользователем")
            status = data.get("status")
            if status == "downloading":
                job = find_job(data)
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
                self._file_progress(find_job(data), 100.0, "Обработка через ffmpeg…")

        return hook

    # ------------------------------------------------------------ execution
    def _run_group(self, group: list[DownloadJob], yt_dlp_module) -> list[str]:
        """Fetch one video (whole or all its chapters) in a single yt-dlp run.

        Every chapter of the video is passed to ``download_ranges`` at once —
        exactly what several ``--download-sections "*start-end"`` options do on
        the command line.  yt-dlp then splits the entry per range, sets
        ``section_start``/``section_end`` and downloads **only** the requested
        time ranges instead of the whole file.  Returns one status string per
        job of *group* (``downloaded`` / ``skipped`` / ``failed``).
        """
        lead = group[0]
        statuses: list[str] = []
        pending: list[tuple[DownloadJob, Path]] = []
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
            pending.append((job, path))
        if not pending:
            return statuses

        cutting = any(job.fragment is not None for job, _ in pending)
        options: dict = {
            "format": self._format_selector(lead),
            "paths": {"home": str(lead.output_dir)},
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "progress_hooks": [self._make_hook(pending)],
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

        if cutting:
            # CLI-equivalent of repeated --download-sections "*START-END".
            # The Python API parameter is ``download_ranges`` and must be a
            # callback yielding the time ranges; passing a string under any
            # name is silently ignored by YoutubeDL, which made us download
            # the whole file instead of only the selected chapters.
            ranges = [
                {
                    "start": job.fragment.start,
                    "end": job.fragment.end,
                    "title": sanitize(job.fragment.title, _MAX_FRAGMENT_TITLE),
                    "job": job,
                }
                for job, _ in pending
                if job.fragment is not None
            ]
            options["download_ranges"] = _ranges_callback(ranges)
            options["force_keyframes_at_cuts"] = True
            # Output template mirrors the *first* chapter's expected name with
            # the trailing ``[ID]`` replaced by ``[%(section_title)s]``, so
            # yt-dlp writes every chapter into its own file named after that
            # chapter (same scheme as the CLI ``--split-chapters``).
            first_stem = self._chapter_stem(ranges[0]["job"])
            options["outtmpl"] = {"default": f"{first_stem}.%(ext)s"}
            self._log(
                "Скачиваю только выбранные фрагменты "
                f"(--download-sections-эквивалент): {lead.video.title} "
                f"[{len(ranges)} фрагм.(ов)]"
            )
        else:
            options["outtmpl"] = {"default": f"{lead.basename}.%(ext)s"}
            self._log(f"Скачиваю всё видео: {lead.label}")

        ok = True
        try:
            with yt_dlp_module.YoutubeDL(options) as ydl:
                ydl.download([lead.video.watch_url])
        except KeyboardInterrupt:
            self._log("Текущая загрузка прервана.")
            return ["cancelled" if status == "pending" else status
                    for status in statuses]
        except Exception as exc:  # noqa: BLE001 - one bad video must not kill the batch
            self._log(f"Ошибка на «{lead.video.title}»: {_first_line(str(exc))}")
            ok = False

        for position, (job, path) in enumerate(pending):
            if not ok:
                statuses[position] = "failed"
                continue
            produced = self._find_output(job, path)
            if produced:
                statuses[position] = "downloaded"
                self._log(f"Сохранено: {produced}")
            else:
                statuses[position] = "failed"
                self._log(f"Фрагмент не найден среди результатов: {path.name}")
        return statuses

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


def _ranges_callback(
    ranges: list[dict],
) -> Callable[[dict, object], list[dict]]:
    """Build a ``download_ranges`` callback for one or several chapters.

    Mirrors what the CLI does for repeated ``--download-sections
    "*START-END"`` options (and ``--split-chapters`` naming): every entry of
    *ranges* carries ``start``/``end`` in seconds plus the chapter ``title``,
    which yt-dlp exposes to the output template as ``%(section_title)s`` and
    to progress hooks as ``section_title``.  An open ended chapter (``end`` of
    ``None``) maps to ``inf``, i.e. "up to the end of the video".
    """

    def provider(_info_dict: dict, _ydl: object) -> list[dict]:
        return [
            {
                "start_time": float(max(int(item["start"]), 0)),
                "end_time": (
                    float(item["end"]) if item.get("end") else float("inf")
                ),
                "title": item.get("title"),
            }
            for item in ranges
        ]

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
