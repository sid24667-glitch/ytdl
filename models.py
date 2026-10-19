"""Dataclasses describing the domain model of the application.

The module holds plain data containers shared between the extractor,
the UI and the downloader: :class:`Fragment`, :class:`Video`,
:class:`FormatInfo` and :class:`Playlist`.  Keeping them free of any
third-party imports makes the package easy to test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fragment:
    """A single time-coded chapter extracted from a video description.

    Attributes:
        start: Start of the fragment in seconds.
        end: End of the fragment in seconds (``None`` -> end of video).
        title: Human readable chapter title.
        whole_video: ``True`` for the synthetic "All video" entry used when
            a description carries no time codes at all.
    """

    start: int
    end: int | None
    title: str
    whole_video: bool = False

    @property
    def label(self) -> str:
        """Compact textual representation shown in the tree view."""
        return f"{format_timestamp(self.start)} {self.title}".strip()


@dataclass
class Video:
    """Metadata of one YouTube video inside a playlist or on its own."""

    video_id: str
    title: str
    duration: int | None = None
    url: str = ""
    #: Lazily filled by the extractor once the user selects the item.
    description: str | None = None
    fragments: list[Fragment] = field(default_factory=list)
    formats_loaded: bool = False
    #: ``True`` when the metadata (description/formats) was already fetched.
    details_loaded: bool = False
    #: Resolutions offered by this video once its full metadata is known.
    available_heights: list[int] = field(default_factory=list)
    #: Raw yt-dlp format table kept for building the selectable format list.
    raw_formats: list[dict] = field(default_factory=list, repr=False)

    @property
    def watch_url(self) -> str:
        """Canonical watch URL for the video."""
        return self.url or f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class FormatInfo:
    """One selectable media format offered to the user.

    Attributes:
        token: yt-dlp format selector used for the download.
        height: Vertical resolution in pixels (``0`` for audio only).
        vcodec / acodec: Codec identifiers as reported by yt-dlp.
        fps: Frame rate (``None`` when unknown / audio only).
        filesize: Approximate size in bytes (``None`` when unknown).
        is_audio_only: ``True`` for the "audio only (mp3)" pseudo format.
        note: Short human readable summary.
    """

    token: str
    height: int
    vcodec: str
    acodec: str
    fps: int | None
    filesize: int | None
    is_audio_only: bool = False
    note: str = ""

    @property
    def resolution(self) -> str:
        return "audio" if self.is_audio_only else (str(self.height) if self.height else "?")

    @property
    def codecs(self) -> str:
        parts = [c for c in (self.vcodec, self.acodec) if c and c != "none"]
        return ", ".join(parts) if parts else "-"

    @property
    def fps_text(self) -> str:
        return str(self.fps) if self.fps else "-"

    @property
    def size_text(self) -> str:
        return human_size(self.filesize)

    @property
    def height_limit(self) -> int | None:
        """Maximum video height implied by the format (``None`` = best)."""
        if self.is_audio_only or not self.height:
            return None
        return self.height


@dataclass
class Playlist:
    """A YouTube playlist with its (lazily enriched) list of videos."""

    playlist_id: str
    title: str
    videos: list[Video] = field(default_factory=list)
    url: str = ""

    @property
    def count(self) -> int:
        return len(self.videos)


def format_timestamp(seconds: int | None) -> str:
    """Return ``HH:MM:SS`` or ``MM:SS`` for a non-negative amount of seconds."""
    if seconds is None or seconds < 0:
        return ""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def human_size(size: int | None) -> str:
    """Render a byte count as a short human readable string."""
    if not size or size <= 0:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "-"
