"""Tests for raw CHAT file parsing."""

from aphasia_modeling.data.chat_parser import parse_cha_file

CHA = (
    "@Begin\n"
    "*PAR:\tthe cat sat . \x151000_2500\x15\n"
    "%mor:\tdet|the n|cat v|sit .\n"
    "*INV:\tokay . \x152500_3000\x15\n"
    "*PAR:\thmhm &=head:no .\n"
    "*PAR:\tand then . \x153000_4000\x15\n"
    "@End\n"
)


def test_untimed_utterances_dropped(tmp_path):
    path = tmp_path / "fridriksson01a.cha"
    path.write_text(CHA, encoding="utf-8")

    utts = parse_cha_file(path)

    assert [u.raw_text for u in utts] == ["the cat sat .", "and then ."]
    assert (utts[0].start_time, utts[0].end_time) == (1.0, 2.5)
    # IDs keep their position in the file, so the untimed line leaves a gap
    assert [u.utterance_id for u in utts] == ["fridriksson01a_0000", "fridriksson01a_0002"]
    assert utts[0].speaker_id == "fridriksson01"
    assert utts[0].database == tmp_path.name


def test_nested_scripts_layout(tmp_path):
    path = tmp_path / "Fridriksson" / "PWA" / "P1" / "eggs" / "P1_B2_SE_C1.cha"
    path.parent.mkdir(parents=True)
    path.write_text(CHA, encoding="utf-8")

    utts = parse_cha_file(path)

    assert utts[0].speaker_id == "P1"
    assert utts[0].database == "Fridriksson"
