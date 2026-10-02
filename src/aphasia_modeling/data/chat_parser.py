"""Parse AphasiaBank CHAT (.cha) files to extract participant utterances.

Replicates the CHAI Lab preprocessing pipeline from
BeyondBinary-ParaphasiaDetection. Parses raw *PAR: lines directly to
preserve CHAT annotations (error codes, IPA forms, target replacements)
that pylangacq's tokenizer would strip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Pattern to extract speaker number from Fridriksson session IDs
# e.g., "fridriksson01a" -> "fridriksson01", "fridriksson03b" -> "fridriksson03"
_FRIDRIKSSON_SPK_PATTERN = re.compile(r"^(fridriksson\d+)[a-z]$")

# Fridriksson-2 sessions are named like "1003-1", "1003-LARC" -> speaker "1003"
_FRIDRIKSSON2_SPK_PATTERN = re.compile(r"^(\d+)-(?:\d+|LARC)$")

# Scripts-Fridriksson sessions are named like "P1_B2_SA_C1" -> speaker "P1"
_SCRIPTS_SPK_PATTERN = re.compile(r"^(P\d+)_")


@dataclass
class Utterance:
    """A single participant utterance extracted from a CHAT file."""

    utterance_id: str
    speaker: str
    # Raw CHAT-coded text before cleaning
    raw_text: str
    # Word tokens after cleaning (no paraphasia labels)
    words: list[str] = field(default_factory=list)
    # Per-word paraphasia labels: "c", "p", "n", "s"
    labels: list[str] = field(default_factory=list)
    # Target words for paraphasic tokens (from [: target] annotations)
    targets: list[str | None] = field(default_factory=list)
    # Audio timing
    audio_path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    # Metadata
    session_id: str = ""
    database: str = ""
    speaker_id: str = ""


# Databases excluded by CHAI
EXCLUDED_DATABASES = {"kempler", "garrett"}

# Sessions whose TalkBank media is unusable. In Scripts-Fridriksson, each pair
# below shares a byte-identical video despite different transcripts/timings,
# so at least one per pair has the wrong audio (and the P13 video is corrupt
# past 30.7s). We can't tell which is correct, so both are dropped.
EXCLUDED_SESSIONS = {
    "P12_B2_SV_C1", "P12_P2_SV_C4",
    "P13_B2_SE_C1", "P13_P2_SE_C4",
}

# Timing marker pattern: \x15start_end\x15
TIMING_PATTERN = re.compile(r"\x15(\d+)_(\d+)\x15")


def parse_cha_file(
    cha_path: str | Path,
    audio_dir: str | Path | None = None,
) -> list[Utterance]:
    """Parse a single .cha file and return participant utterances.

    Reads the raw file directly to preserve CHAT annotations (error codes,
    IPA transcriptions, target replacements) that are needed for paraphasia
    label extraction.

    Args:
        cha_path: Path to the .cha file.
        audio_dir: Directory containing corresponding audio files.
            If None, audio_path will be empty.

    Returns:
        List of Utterance objects with raw_text populated.
        Call preprocess_utterance() on each to fill words/labels/targets.
    """
    cha_path = Path(cha_path)
    session_id = cha_path.stem

    # Infer database from the dir above PWA, which may be nested
    # (data/Fridriksson/PWA/file.cha or data/Fridriksson/PWA/P1/eggs/file.cha
    # -> Fridriksson). Fall back to parent if there is no PWA dir.
    dirs = [p.name for p in cha_path.parents]
    pwa_idx = next((i for i, d in enumerate(dirs) if d.upper() == "PWA"), None)
    if pwa_idx is not None and pwa_idx + 1 < len(dirs):
        database = dirs[pwa_idx + 1]
    else:
        database = cha_path.parent.name

    if audio_dir is not None:
        audio_dir = Path(audio_dir)

    # Read raw file and extract *PAR: lines with continuation
    raw_lines = cha_path.read_text(encoding="utf-8").splitlines()
    par_lines = _extract_speaker_lines(raw_lines, "PAR")

    utterances = []
    for utt_idx, (raw_text, start_ms, end_ms) in enumerate(par_lines):
        if not raw_text.strip():
            continue
        # Untimed utterances have no audio segment; loading them would read
        # the whole session file against a short transcript.
        if end_ms <= start_ms:
            continue

        utt_id = f"{session_id}_{utt_idx:04d}"
        utt = Utterance(
            utterance_id=utt_id,
            speaker="PAR",
            raw_text=raw_text,
            session_id=session_id,
            database=database,
            speaker_id=_session_to_speaker(session_id),
            start_time=start_ms / 1000.0,
            end_time=end_ms / 1000.0,
        )

        if audio_dir is not None:
            wav_path = audio_dir / f"{session_id}.wav"
            if wav_path.exists():
                utt.audio_path = str(wav_path)

        utterances.append(utt)

    return utterances


def _extract_speaker_lines(
    lines: list[str], speaker: str
) -> list[tuple[str, int, int]]:
    """Extract utterance lines for a given speaker from raw CHAT lines.

    CHAT format allows continuation lines (starting with \\t) after the
    main *SPEAKER: line. Dependent tiers (%mor:, %gra:, etc.) are skipped.

    Returns:
        List of (raw_text, start_ms, end_ms) tuples. Timing may be 0 if absent.
    """
    prefix = f"*{speaker}:"
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(prefix):
            # Collect the main line content (after *PAR:\t)
            content = line[len(prefix):].strip()

            # Collect continuation lines (start with \t, not a tier like %mor:)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith("\t") and not next_line.startswith("%"):
                    content += " " + next_line.strip()
                    i += 1
                elif next_line.startswith("%") or next_line.startswith("@"):
                    # Dependent tier or header — skip
                    i += 1
                    continue
                else:
                    break

            # Extract timing markers
            start_ms, end_ms = 0, 0
            timing_match = TIMING_PATTERN.search(content)
            if timing_match:
                start_ms = int(timing_match.group(1))
                end_ms = int(timing_match.group(2))
                # Remove timing marker from text
                content = TIMING_PATTERN.sub("", content).strip()

            results.append((content, start_ms, end_ms))
        else:
            i += 1

    return results


def _session_to_speaker(session_id: str) -> str:
    """Derive speaker ID from session ID.

    Fridriksson sessions like 'fridriksson01a' and 'fridriksson01b' belong
    to the same speaker 'fridriksson01'. For other datasets, session ID
    is used as-is.
    """
    m = _FRIDRIKSSON_SPK_PATTERN.match(session_id)
    if m:
        return m.group(1)
    m = _FRIDRIKSSON2_SPK_PATTERN.match(session_id)
    if m:
        return m.group(1)
    m = _SCRIPTS_SPK_PATTERN.match(session_id)
    if m:
        return m.group(1)
    return session_id


def parse_cha_directory(
    cha_dir: str | Path,
    audio_dir: str | Path | None = None,
    exclude_databases: set[str] | None = None,
) -> list[Utterance]:
    """Parse all .cha files in a directory tree.

    Args:
        cha_dir: Root directory to search for .cha files.
        audio_dir: Directory containing audio files.
        exclude_databases: Database names to skip (default: kempler, garrett per CHAI).

    Returns:
        List of all Utterance objects across all files.
    """
    cha_dir = Path(cha_dir)
    if exclude_databases is None:
        exclude_databases = EXCLUDED_DATABASES

    all_utterances = []
    for cha_path in sorted(cha_dir.rglob("*.cha")):
        # Check if this file belongs to an excluded database
        db_name = cha_path.parent.name.lower()
        if db_name in {d.lower() for d in exclude_databases}:
            continue
        if cha_path.stem in EXCLUDED_SESSIONS:
            continue

        utterances = parse_cha_file(cha_path, audio_dir)
        all_utterances.extend(utterances)

    return all_utterances
