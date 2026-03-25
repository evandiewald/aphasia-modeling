"""Preprocess CHAT utterances to produce cleaned text with paraphasia labels.

Replicates the CHAI Lab `clean_transcript_best_word.py` pipeline:
1. Skip unintelligible/empty utterances
2. Normalize brackets and whitespace
3. Remove terminators, punctuation, pauses
4. Handle repetitions, fragments, compound words, special forms
5. Convert IPA/UNIBET neologisms to pseudo-English words
6. Extract per-word paraphasia labels from [* ...] error codes
7. Produce single-seq format: "word [p] word word [n]"
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .chat_parser import Utterance

# IPA/UNIBET to English grapheme mapping (from CHAI's ipa2cmu/unibet2arpabet)
# Simplified mapping covering the most common cases
IPA_TO_GRAPHEME = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "f": "f", "v": "v", "s": "s", "z": "z", "h": "h", "m": "m",
    "n": "n", "l": "l", "r": "r", "w": "w", "j": "y",
    "ʃ": "sh", "ʒ": "zh", "tʃ": "ch", "dʒ": "j",
    "θ": "th", "ð": "th",
    "ŋ": "ng",
    # Vowels (approximate grapheme mapping)
    "i": "ee", "ɪ": "i", "e": "ay", "ɛ": "e", "æ": "a",
    "ɑ": "ah", "ɔ": "aw", "o": "oh", "ʊ": "oo", "u": "oo",
    "ʌ": "u", "ə": "uh", "ɝ": "er", "ɚ": "er",
    "aɪ": "ai", "aʊ": "ow", "ɔɪ": "oy", "eɪ": "ay", "oʊ": "oh",
    # UNIBET/CHAT-specific symbols
    "E": "e", "I": "i", "O": "oh", "U": "oo", "A": "ah",
    "W": "ow", "Y": "ai", "R": "er",
    "S": "sh", "Z": "zh", "T": "th", "D": "th", "N": "ng",
    "C": "ch", "J": "j",
}

# Patterns that cause the entire utterance to be skipped
SKIP_UTTERANCE_MARKERS = {"[?]", "xxx", "[<>]", "[+ exc]"}

# Vocal events to remove
VOCAL_EVENT_PATTERN = re.compile(r"&=[a-zA-Z_:.]+")

# Fragment pattern (false starts)
FRAGMENT_PATTERN = re.compile(r"&[-+][a-zA-Z]*")

# Special form codes that cause word to be skipped
SKIP_SPECIAL_FORMS = {"@n", "@o", "@s", "@b", "@si"}

# Special form codes where we just remove the marker
STRIP_SPECIAL_FORMS = {"@q", "@a", "@d", "@u", "@wp"}

# Letter/spelling forms
LETTER_SPECIAL_FORMS = {"@l", "@k"}

# Interjection form
INTERJECTION_FORM = "@i"


@dataclass
class _Token:
    """Internal token during preprocessing."""
    text: str
    error_code: str | None = None  # p, n, s, etc.
    target_word: str | None = None  # From [: target]


def preprocess_utterance(utt: Utterance) -> bool:
    """Preprocess an utterance in place, populating words/labels/targets.

    Returns True if the utterance is valid after preprocessing, False if it
    should be skipped (unintelligible, overlapping speech, etc.).

    Modifies utt.words, utt.labels, utt.targets in place.
    """
    raw = utt.raw_text.strip()
    if not raw:
        return False

    # Step 1: Check for skip conditions
    if _should_skip(raw):
        return False

    # Step 2: Tokenize and normalize
    tokens = _tokenize(raw)

    # Step 3: Process tokens through the cleaning pipeline
    tokens = _remove_terminators(tokens)
    tokens = _process_brackets(tokens)
    tokens = _process_repetitions(tokens)
    tokens = _remove_vocal_events(tokens)
    tokens = _remove_fragments(tokens)
    tokens = _process_compound_words(tokens)
    tokens = _process_special_forms(tokens)
    tokens = _process_error_codes(tokens)
    tokens = _finalize_tokens(tokens)

    if not tokens:
        return False

    # Populate utterance fields
    utt.words = [t.text for t in tokens]
    utt.labels = [t.error_code or "c" for t in tokens]
    utt.targets = [t.target_word for t in tokens]

    return True


def to_single_seq(words: list[str], labels: list[str]) -> str:
    """Convert words and labels to single-seq format.

    Example: words=["the", "cat", "sat"], labels=["c", "p", "c"]
    Returns: "the cat [p] sat"

    Matches CHAI's format: word followed by [p]/[n]/[s] tag only for
    paraphasic words, no tag for correct words.
    """
    parts = []
    for word, label in zip(words, labels):
        parts.append(word)
        if label in ("p", "n"):
            parts.append(f"[{label}]")
    return " ".join(parts)


def parse_single_seq(text: str) -> tuple[list[str], list[str]]:
    """Parse single-seq text back into words and labels.

    Example: "the cat [p] sat" -> (["the", "cat", "sat"], ["c", "p", "c"])
    """
    tokens = text.strip().split()
    words = []
    labels = []
    for token in tokens:
        if token in ("[p]", "[n]"):
            # Attach label to the preceding word
            if labels:
                labels[-1] = token[1]  # "p" or "n"
        else:
            words.append(token)
            labels.append("c")
    return words, labels


def preprocess_dataset(utterances: list[Utterance]) -> list[Utterance]:
    """Preprocess a list of utterances, returning only valid ones."""
    valid = []
    for utt in utterances:
        if preprocess_utterance(utt):
            valid.append(utt)
    return valid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _should_skip(text: str) -> bool:
    """Check if the entire utterance should be skipped."""
    # Empty or only whitespace
    if not text.strip():
        return True
    # Fully unintelligible
    stripped = text.strip().rstrip(".!?")
    if stripped in ("xxx", "yyy", "www", "0"):
        return True
    # Contains skip markers
    for marker in SKIP_UTTERANCE_MARKERS:
        if marker in text:
            return True
    # Contains overlapping speech indicator
    if re.search(r"\[<\d*\]|\[>\d*\]", text):
        return True
    # Contains paralinguistic material
    if re.search(r"\[=!", text):
        return True
    return False


def _tokenize(text: str) -> list[str]:
    """Normalize and tokenize raw CHAT text."""
    # Separate [...] blocks from adjacent text
    text = re.sub(r"(\S)\[", r"\1 [", text)
    text = re.sub(r"\](\S)", r"] \1", text)
    # Separate < from words
    text = re.sub(r"(\S)<", r"\1 <", text)
    text = re.sub(r">(\S)", r"> \1", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _remove_terminators(tokens: list[str]) -> list[str]:
    """Remove punctuation, terminators, pauses, and retrace markers."""
    skip_tokens = {
        ".", ",", ";", "?", "!", ":",
        "+...", "+/.", "+//.", "+//?", "+..?",
        "[/]", "[//]", "[///]",
        "(.)", "(..)", "(...)",
        "+<", "+^",
        "‡", "„",
    }
    return [t for t in tokens if t not in skip_tokens]


def _process_brackets(tokens: list[str]) -> list[str]:
    """Process bracket-enclosed annotations.

    Handles:
    - [* X] error codes -> attached to preceding word
    - [: target] replacement -> replaces preceding word
    - [=! ...] paralinguistic -> removed
    - [% ...] comments -> removed
    - [+ ...] postcodes -> removed (except [+ exc] caught earlier)
    - [- ...] precodes -> removed
    - [=  ...] explanations -> removed
    """
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Start of a bracket block
        if token == "[*" or token.startswith("[*"):
            # Error code: [* p], [* n], [* s], etc.
            # Collect until closing ]
            block = [token]
            if "]" not in token:
                i += 1
                while i < len(tokens):
                    block.append(tokens[i])
                    if "]" in tokens[i]:
                        break
                    i += 1
            code = _extract_error_code(block)
            # Attach to preceding word token
            if result:
                last = result[-1]
                if isinstance(last, _Token):
                    last.error_code = code
                else:
                    # Convert string to _Token
                    result[-1] = _Token(text=last, error_code=code)
            i += 1
            continue

        if token == "[:" or token.startswith("[:"):
            # Target/replacement: [: target_word]
            block = [token]
            if "]" not in token:
                i += 1
                while i < len(tokens):
                    block.append(tokens[i])
                    if "]" in tokens[i]:
                        break
                    i += 1
            target = _extract_target(block)
            # Attach target to preceding word
            if result and target:
                last = result[-1]
                if isinstance(last, _Token):
                    last.target_word = target
                else:
                    result[-1] = _Token(text=last, target_word=target)
            i += 1
            continue

        # Pass through repetition markers [x N] for later processing
        if token == "[x" or token.startswith("[x"):
            result.append(token)
            i += 1
            continue

        # Skip other bracket annotations
        if token.startswith("[") and token not in ("[", "<"):
            # Consume until matching ]
            if "]" not in token:
                i += 1
                while i < len(tokens) and "]" not in tokens[i]:
                    i += 1
            i += 1
            continue

        # Angular brackets (retrace/overlap scope markers)
        if token in ("<", ">"):
            i += 1
            continue

        result.append(token)
        i += 1

    # Ensure all items are _Token objects
    final = []
    for item in result:
        if isinstance(item, _Token):
            final.append(item)
        else:
            final.append(_Token(text=item))
    return final


def _extract_error_code(block: list[str]) -> str:
    """Extract paraphasia type from an error code block like ['[*', 'p]']."""
    text = " ".join(block)
    # Remove brackets
    text = text.replace("[", "").replace("]", "").replace("*", "").strip()
    # Take the first character/code
    parts = text.split(":")
    code = parts[0].strip().lower()
    # Map to our simplified labels
    if code.startswith("p"):
        return "p"
    elif code.startswith("n"):
        return "n"
    else:
        # Semantic paraphasias (s) are handled by Stage 2 LLM, not the
        # acoustic model. All other error types (d, m, f, s, etc.) are
        # treated as correct for Stage 1.
        return "c"


def _extract_target(block: list[str]) -> str | None:
    """Extract target word from [: target] block."""
    text = " ".join(block)
    text = text.replace("[:", "").replace("]", "").strip()
    return text if text else None


def _process_repetitions(tokens: list[_Token | str]) -> list[_Token | str]:
    """Expand repetition codes [x N]."""
    result = []
    i = 0
    items = tokens
    while i < len(items):
        item = items[i]
        text = item.text if isinstance(item, _Token) else item
        if text == "[x" or (isinstance(text, str) and text.startswith("[x")):
            # [x N] — repeat the preceding token N times
            # Collect the number
            block = text
            if "]" not in text:
                i += 1
                while i < len(items):
                    next_text = items[i].text if isinstance(items[i], _Token) else items[i]
                    block += " " + next_text
                    if "]" in next_text:
                        break
                    i += 1
            match = re.search(r"\d+", block)
            if match and result:
                count = int(match.group()) - 1  # -1 because word already in result
                for _ in range(count):
                    result.append(result[-1])
            i += 1
            continue
        result.append(item)
        i += 1
    return result


def _remove_vocal_events(tokens: list[_Token | str]) -> list[_Token | str]:
    """Remove vocal events like &=laughs, &=coughs."""
    result = []
    for t in tokens:
        text = t.text if isinstance(t, _Token) else t
        if VOCAL_EVENT_PATTERN.fullmatch(text):
            continue
        result.append(t)
    return result


def _remove_fragments(tokens: list[_Token | str]) -> list[_Token | str]:
    """Remove fragments (false starts) like &-uh, &+well."""
    result = []
    for t in tokens:
        text = t.text if isinstance(t, _Token) else t
        if FRAGMENT_PATTERN.fullmatch(text):
            continue
        result.append(t)
    return result


def _process_compound_words(tokens: list[_Token | str]) -> list[_Token | str]:
    """Handle compound words, @ markers with replacements."""
    result = []
    for t in tokens:
        tok = t if isinstance(t, _Token) else _Token(text=t)
        word = tok.text

        # Words with @ — handle special form codes
        if "@" in word:
            base, _, form_code = word.partition("@")
            form_code = "@" + form_code

            if form_code == "@u":
                # UNIBET/IPA transcription — convert to pseudoword
                # Do NOT use target_word here: the model should see what
                # the speaker produced, not what they intended to say
                tok.text = ipa_to_pseudoword(base)
                if tok.text:
                    result.append(tok)
                continue
            elif tok.target_word:
                # Non-IPA forms with a target replacement (e.g., dialect @d)
                tok.text = tok.target_word.lower()
                result.append(tok)
            elif form_code in SKIP_SPECIAL_FORMS:
                # No replacement available, skip entirely
                continue
            else:
                # Skip words with @ and no replacement
                continue
            continue

        # Break compounds on + and _
        if "+" in word or "_" in word:
            parts = re.split(r"[+_]", word)
            for j, part in enumerate(parts):
                if part:
                    new_tok = _Token(
                        text=part,
                        error_code=tok.error_code if j == 0 else None,
                        target_word=tok.target_word if j == 0 else None,
                    )
                    result.append(new_tok)
            continue

        result.append(tok)
    return result


def _process_special_forms(tokens: list[_Token | str]) -> list[_Token | str]:
    """Handle remaining special form codes (@l, @i, etc.)."""
    result = []
    for t in tokens:
        tok = t if isinstance(t, _Token) else _Token(text=t)
        word = tok.text

        if "@" in word:
            base, _, form_code = word.partition("@")
            form_code = "@" + form_code

            if form_code == INTERJECTION_FORM:
                continue  # Skip interjections
            if form_code in LETTER_SPECIAL_FORMS:
                # Spell out letters
                for ch in base:
                    if ch.isalpha():
                        result.append(_Token(text=ch))
                continue
            if form_code in STRIP_SPECIAL_FORMS:
                tok.text = base
                result.append(tok)
                continue
            # Default: use base without @ marker
            tok.text = base
            if base:
                result.append(tok)
            continue

        result.append(tok)
    return result


def _process_error_codes(tokens: list[_Token | str]) -> list[_Token | str]:
    """Ensure all tokens are _Token objects with proper error codes."""
    result = []
    for t in tokens:
        if isinstance(t, _Token):
            result.append(t)
        else:
            result.append(_Token(text=t))
    return result


def _finalize_tokens(tokens: list[_Token | str]) -> list[_Token]:
    """Final cleanup: lowercase, remove non-alphabetic chars, convert OOVs."""
    result = []
    for t in tokens:
        tok = t if isinstance(t, _Token) else _Token(text=t)
        word = tok.text

        # Remove possessive markers, special chars
        word = word.replace("'s", "s").replace("'", "")
        word = word.replace("$on", "")

        # Remove colons (used for elongation in CHAT)
        word = word.replace(":", "")

        # Remove syllable pause markers
        word = word.replace("^", "")

        # Handle partial omissions: (ing) -> ing
        word = re.sub(r"\(([a-zA-Z]+)\)", r"\1", word)

        # Normalize unicode
        word = unicodedata.normalize("NFKD", word)

        # Keep only alphabetic characters and hyphens
        word = re.sub(r"[^a-zA-Z-]", "", word)

        # Split on hyphens
        if "-" in word:
            parts = [p for p in word.split("-") if p]
            for j, part in enumerate(parts):
                new_tok = _Token(
                    text=part.lower(),
                    error_code=tok.error_code if j == 0 else "c",
                    target_word=tok.target_word if j == 0 else None,
                )
                if new_tok.text:
                    result.append(new_tok)
            continue

        word = word.lower()
        if word:
            tok.text = word
            result.append(tok)

    return result


def ipa_to_pseudoword(ipa_str: str) -> str:
    """Convert an IPA/UNIBET transcription to a pseudo-English word.

    Used for neologistic paraphasias where the original word is transcribed
    in IPA rather than standard orthography.

    Example: "fEkts" -> "fekts" (approximate)
    """
    result = []
    i = 0
    while i < len(ipa_str):
        # Try two-character sequences first
        if i + 1 < len(ipa_str):
            digraph = ipa_str[i:i + 2]
            if digraph in IPA_TO_GRAPHEME:
                result.append(IPA_TO_GRAPHEME[digraph])
                i += 2
                continue
        # Single character
        ch = ipa_str[i]
        if ch in IPA_TO_GRAPHEME:
            result.append(IPA_TO_GRAPHEME[ch])
        elif ch.isalpha():
            result.append(ch.lower())
        # Skip non-alphabetic IPA diacritics
        i += 1
    return "".join(result)
