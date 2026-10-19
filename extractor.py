"""Thin wrapper around ``yt-dlp`` used to inspect YouTube resources.

The module exposes three responsibilities:

* :func:`parse_url` — recognise a YouTube link and split it into a video id
  and/or a playlist id (including the combined ``watch?v=..&list=..`` form).
* :class:`Extractor` — fetch metadata.  A playlist is first read in *flat*
  mode (only ids/titles/durations, one network round trip) and the heavier
  per-video data (description + format list) is retrieved on demand through
  :meth:`Extractor.load_video_details`, which keeps startup fast even for
  playlists with hundreds of entries.
* Format selection helpers (:func:`build_format_list`) that turn the raw
  yt-dlp format table into a short, de-duplicated list of user choices.

All network errors are translated into :class:`ExtractError` carrying a
human readable Russian message so the UI never has to show a traceback.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from models import FormatInfo, Playlist, Video

#: Hosts accepted as YouTube links.
ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID_RE = re.compile(r"^(?:PL|UU|LL|RD|OL|FL|PU|LP)[A-Za-z0-9_-]{11,}$")


class ExtractError(Exception):
    """Raised for any recoverable metadata-extraction problem."""


def validate_url(url: str) -> str:
    """Normalise *url* and make sure it points to an allowed YouTube host."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ExtractError("Ссылка пуста.")
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ExtractError(f"Не удалось разобрать ссылку: {exc}") from exc
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ExtractError(
            "Ссылка не является ссылкой на YouTube "
            f"(домен: {host or 'не определён'}).\n"
            "Поддерживаются youtube.com, m.youtube.com, music.youtube.com и youtu.be."
        )
    return cleaned


def parse_url(url: str) -> tuple[str | None, str | None]:
    """Return ``(video_id, playlist_id)`` extracted from a YouTube *url*.

    Either element may be ``None``; a bare playlist page yields only a
    playlist id, a plain video link only a video id, and
    ``watch?v=ID&list=PL...`` yields both.
    """
    parsed = urlparse(validate_url(url))
    host = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)

    playlist_id = (query.get("list") or [None])[0]
    if playlist_id:
        # A "watch mix" (RD...) embeds the real playlist id in its tail.
        playlist_id = _mix_playlist_id(playlist_id) or playlist_id

    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        video_id = candidate if _VIDEO_ID_RE.match(candidate) else None
    elif host == "music.youtube.com":
        video_id = (query.get("v") or [None])[0]
        if not video_id and parsed.path.startswith("/playlist"):
            playlist_id = playlist_id or _path_segment(parsed.path)
    else:
        path = parsed.path
        if path in ("/playlist", "/playlist/"):
            video_id = None
        elif path in ("/watch", "/shorts/", "/live/") or path.startswith(("/shorts/", "/live/", "/embed/", "/v/")):
            parts = [p for p in path.split("/") if p]
            video_id = (query.get("v") or [None])[0]
            if video_id is None and len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
                video_id = parts[1]
        else:
            video_id = (query.get("v") or [None])[0]

    if video_id and not _VIDEO_ID_RE.match(video_id):
        video_id = None
    if playlist_id and not _PLAYLIST_ID_RE.match(playlist_id):
        # Keep any list= value that looks like an identifier at all; yt-dlp
        # will report a precise error for genuinely malformed ids.
        if not re.match(r"^[A-Za-z0-9_-]{10,}$", playlist_id):
            playlist_id = None
    return video_id, playlist_id


def _path_segment(path: str) -> str | None:
    """Best-effort extraction of a playlist id embedded in a URL path."""
    parts = [p for p in path.split("/") if p and p not in ("playlist", "watch")]
    return parts[0] if parts and _PLAYLIST_ID_RE.match(parts[0]) else None


def _mix_playlist_id(value: str) -> str | None:
    """Return the playlist id carried by a YouTube ``RD`` mix identifier."""
    if value.startswith("RD") and len(value) > 13:
        candidate = value[2:]
        if candidate.startswith(("PL", "UU", "OL")) or _PLAYLIST_ID_RE.match(candidate):
            return candidate
    return None


def build_format_list(raw_formats: list[dict]) -> list[FormatInfo]:
    """Condense the yt-dlp format table into selectable choices.

    Video formats are grouped by ``(height, fps)`` keeping the best entry of
    each group, ordered from the highest resolution down, and a synthetic
    "audio only (mp3)" option is appended.
    """
    groups: dict[tuple[int, int], dict] = {}
    for fmt in raw_formats:
        height = fmt.get("height") or 0
        vcodec = fmt.get("vcodec") or "none"
        if not height or vcodec == "none":
            continue
        if fmt.get("acodec") in (None, "none"):
            continue  # video-only stream cannot be downloaded standalone
        fps = int(fmt.get("fps") or 0)
        key = (height, fps)
        current = groups.get(key)
        size = fmt.get("filesize") or fmt.get("filesize_approx") or 0
        current_size = 0
        if current is not None:
            current_size = current.get("filesize") or current.get("filesize_approx") or 0
        if current is None or size > current_size:
            groups[key] = fmt

    result: list[FormatInfo] = []
    for (height, fps), fmt in sorted(groups.items(), key=lambda item: (-item[0][0], -item[0][1])):
        size = fmt.get("filesize") or fmt.get("filesize_approx")
        itag = fmt.get("format_id", "")
        result.append(
            FormatInfo(
                token=f"bv*[height<={height}]+ba/b[height<={height}]",
                height=height,
                vcodec=_short_codec(fmt.get("vcodec") or ""),
                acodec=_short_codec(fmt.get("acodec") or ""),
                fps=fps or None,
                filesize=size,
                note=f"{itag}",
            )
        )

    audio = next((f for f in raw_formats if (f.get("vcodec") in (None, "none")) and f.get("acodec")), None)
    if audio is not None:
        result.append(
            FormatInfo(
                token="bestaudio/best",
                height=0,
                vcodec="none",
                acodec=_short_codec(audio.get("acodec") or ""),
                fps=None,
                filesize=None,
                is_audio_only=True,
                note="mp3 (перекодирование ffmpeg)",
            )
        )
    return result


def _short_codec(codec: str) -> str:
    """Trim yt-dlp codec strings (``av01.0.05M.08`` -> ``av01``)."""
    if not codec or codec == "none":
        return "none"
    return codec.split(".")[0]


class Extractor:
    """Fetch YouTube metadata through the yt-dlp Python API."""

    def __init__(self, logger=None) -> None:
        #: Optional callable ``(str) -> None`` receiving progress messages.
        self._log = logger or (lambda message: None)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _ydl_options(**extra) -> dict:
        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "skip_download": True,
            "socket_timeout": 30,
        }
        options.update(extra)
        return options

    # --------------------------------------------------------------- lookups
    def fetch(self, url: str) -> tuple[Playlist | None, Video | None]:
        """Resolve *url* into either a :class:`Playlist` or a single :class:`Video`.

        A link containing both ``v=`` and ``list=`` returns the playlist plus
        the video pointing at the specific entry, so the caller can highlight
        it in the tree.  When YouTube does not report a playlist (for example
        a lone ``RD`` mix id), the result degrades gracefully to a single video.
        """
        video_id, playlist_id = parse_url(url)
        if playlist_id is None and video_id is None:
            raise ExtractError("В ссылке не найдены ни идентификатор видео, ни идентификатор плейлиста.")

        if playlist_id:
            # Flat playlist page: one request returns every entry's id/title.
            target = f"https://www.youtube.com/playlist?list={playlist_id}"
        else:
            target = f"https://www.youtube.com/watch?v={video_id}"

        self._log(f"Запрашиваю метаданные: {target}")
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ExtractError("Модуль yt-dlp не установлен. Выполните: pip install -r requirements.txt") from exc

        try:
            with yt_dlp.YoutubeDL(self._ydl_options(extract_flat="in_playlist")) as ydl:
                info = ydl.extract_info(target, download=False)
        except yt_dlp.utils.DownloadError as exc:
            message = _first_line(str(exc))
            if playlist_id and video_id and "could not find" in message.lower():
                # A mix/unsupported list id: fall back to the plain video page.
                self._log(f"Плейлист не найден ({message}), пробую как одиночное видео…")
                return None, self._fetch_single_video(video_id)
            raise ExtractError(f"Не удалось получить данные: {message}") from exc
        except Exception as exc:  # noqa: BLE001 - surface anything as a friendly error
            raise ExtractError(f"Ошибка извлечения метаданных: {_first_line(str(exc))}") from exc

        if info is None:
            raise ExtractError("YouTube вернул пустой ответ.")

        entries = [e for e in (info.get("entries") or []) if e] if info.get("_type") == "playlist" else []
        if entries:
            playlist = self._playlist_from_info(info, playlist_id or "", url)
            highlighted: Video | None = None
            if video_id:
                highlighted = next((v for v in playlist.videos if v.video_id == video_id), None)
            return playlist, highlighted

        if info.get("_type") == "playlist":
            raise ExtractError("Плейлист пуст или недоступен (приватный / удалённый).")

        return None, self._video_from_info(info, video_id or info.get("id", ""))

    def _fetch_single_video(self, video_id: str) -> Video:
        """Fetch full metadata of one video (fallback path)."""
        import yt_dlp

        with yt_dlp.YoutubeDL(self._ydl_options()) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        if not info:
            raise ExtractError("YouTube вернул пустой ответ.")
        return self._video_from_info(info, video_id)

    def _playlist_from_info(self, info: dict, playlist_id: str, url: str) -> Playlist:
        entries = [e for e in (info.get("entries") or []) if e]
        videos: list[Video] = []
        for entry in entries:
            if entry.get("_type") != "url" and not entry.get("id"):
                continue
            videos.append(
                Video(
                    video_id=str(entry.get("id")),
                    title=str(entry.get("title") or "(без названия)"),
                    duration=entry.get("duration"),
                    url=str(entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}"),
                )
            )
        playlist = Playlist(
            playlist_id=playlist_id or str(info.get("playlist_id") or ""),
            title=str(info.get("title") or "Плейлист"),
            videos=videos,
            url=url,
        )
        self._log(f"Плейлист «{playlist.title}»: {playlist.count} видео")
        return playlist

    @staticmethod
    def _video_from_info(info: dict, video_id: str) -> Video:
        return Video(
            video_id=str(info.get("id") or video_id),
            title=str(info.get("title") or "(без названия)"),
            duration=info.get("duration"),
            url=str(info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"),
            description=info.get("description") or "",
            formats_loaded=bool(info.get("formats")),
            details_loaded=True,
            available_heights=_available_heights(info),
        )

    # ------------------------------------------------------------ lazy loads
    def load_video_details(self, video: Video) -> tuple[Video, list[FormatInfo]]:
        """Fetch description, duration and formats for one flat playlist entry."""
        import yt_dlp

        self._log(f"Загружаю данные видео «{video.title}»…")
        try:
            with yt_dlp.YoutubeDL(self._ydl_options()) as ydl:
                info = ydl.extract_info(video.watch_url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise ExtractError(f"Не удалось загрузить данные видео: {_first_line(str(exc))}") from exc

        if not info:
            raise ExtractError("YouTube вернул пустой ответ для видео.")

        video.description = info.get("description") or ""
        video.duration = info.get("duration") or video.duration
        video.url = info.get("webpage_url") or video.url
        video.details_loaded = True
        video.available_heights = _available_heights(info)
        formats = build_format_list(info.get("formats") or [])
        video.formats_loaded = bool(formats)
        return video, formats


def _available_heights(info: dict) -> list[int]:
    """Distinct vertical resolutions offered by the raw format table."""
    heights = {
        int(fmt["height"])
        for fmt in (info.get("formats") or [])
        if fmt.get("height")
    }
    return sorted(heights, reverse=True)


def _first_line(text: str) -> str:
    """Keep error messages short: only the first meaningful line."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return text.strip()[:300]
