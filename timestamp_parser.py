"""Parsing of time-coded chapters ("timestamps") from video descriptions.

Supported spellings are ``H:MM:SS`` / ``HH:MM:SS`` and ``M:SS`` / ``MM:SS``.
A line only counts as a chapter when a non-empty textual label follows the
time code, which filters out dates (``2024:01:05``), plain numbers and
similar noise.  Successive marks are turned into :class:`~models.Fragment`
objects whose end equals the start of the next chapter; the last fragment is
closed with the total duration of the video.
"""

from __future__ import annotations

import re

from models import Fragment

#: Decorative characters that may precede a chapter mark on a line.
_PREFIX_CHARS = " \t-–—*•·>|"

#: A time code sits at the very beginning of a (cleaned) line and is followed
#: either by the end of the line or by a non-digit / non-colon character.
_HMS_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>[0-5]\d):(?P<s>[0-5]\d)(?![\d:])")
_MS_RE = re.compile(r"^(?P<m>\d{1,3}):(?P<s>[0-5]\d)(?![\d:])")

#: Words that never make a good chapter title on their own.
_STOP_WORDS = {"а", "и", "the", "a", "an", "of", "to", "and"}


def _match_timecode(line: str) -> tuple[int, str] | None:
    """Return ``(start_seconds, remainder)`` when *line* begins with a time code."""
    stripped = line.strip().lstrip(_PREFIX_CHARS).strip()
    if not stripped:
        return None

    match = _HMS_RE.match(stripped)
    hours: int | None
    if match:
        hours = int(match.group("h"))
        minutes, seconds = int(match.group("m")), int(match.group("s"))
    else:
        match = _MS_RE.match(stripped)
        if not match:
            return None
        hours = None
        minutes, seconds = int(match.group("m")), int(match.group("s"))

    total = (hours or 0) * 3600 + minutes * 60 + seconds
    text = stripped[match.end():]
    title = text.strip(" \t-–—:•·>|[](){}«»\"'")
    if not title or title.lower() in _STOP_WORDS:
        return None
    # Pure numbers / years ("2024", "1 500", "№ 12") are not chapter titles.
    if re.fullmatch(r"[№#]?[\d\s.,'x×%$€₽]+", title):
        return None
    return total, title


def parse_timestamps(description: str | None, duration: int | None = None) -> list[Fragment]:
    """Extract ordered chapters from *description*.

    Args:
        description: Raw YouTube description text (may be ``None``).
        duration: Total video length in seconds, used to close the last chapter.

    Returns:
        An empty list when the description carries no usable time codes,
        otherwise a chronologically sorted list of :class:`Fragment`.
    """
    if not description:
        return []

    pairs: list[tuple[int, str]] = []
    for raw_line in description.splitlines():
        parsed = _match_timecode(raw_line)
        if parsed:
            pairs.append(parsed)

    if not pairs:
        return []

    # Descriptions are chapter lists in practice, but never rely on it.
    pairs.sort(key=lambda item: item[0])

    fragments: list[Fragment] = []
    for index, (start, title) in enumerate(pairs):
        if index + 1 < len(pairs):
            end: int | None = pairs[index + 1][0]
            if end <= start:  # overlapping / unordered marks
                end = None
        else:
            end = None
        fragments.append(Fragment(start=start, end=end, title=title))

    last = fragments[-1]
    if last.end is None and duration and duration > last.start:
        last.end = duration

    return fragments


def fragments_or_whole_video(
    description: str | None,
    duration: int | None,
    title: str = "Все видео",
) -> list[Fragment]:
    """Chapters of a video, falling back to a single "whole video" fragment."""
    fragments = parse_timestamps(description, duration)
    if fragments:
        return fragments
    return [Fragment(start=0, end=None, title=title, whole_video=True)]
