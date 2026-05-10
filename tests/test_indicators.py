"""
Unit tests for minerva_indicators.

Run from repo root:
    python -m pytest tests/test_indicators.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from minerva_indicators import (extract_indicators, fired_codes,
                                 named_features,
                                 detect_emo, detect_urg, detect_anon,
                                 detect_miss, detect_pol, detect_cons,
                                 detect_disc, detect_rev, detect_endo,
                                 detect_recf, detect_fab, detect_imp)


def test_emo_fires_on_loaded_words():
    h = detect_emo("This is a betrayed and outraged scandal!")
    assert h.fired
    assert any(w in h.evidence for w in ["betrayed", "outraged"])


def test_emo_does_not_fire_on_neutral():
    h = detect_emo("The Senate met today to discuss the bill.")
    assert not h.fired


def test_urg_fires_on_caps_and_share_now():
    h = detect_urg("URGENT! SHARE NOW BEFORE IT'S DELETED!")
    assert h.fired


def test_anon_fires_on_sources_say():
    h = detect_anon("Sources say the candidate is in trouble.")
    assert h.fired
    assert any("sources say" in e.lower() for e in h.evidence)


def test_anon_fires_on_tagalog_hearsay():
    h = detect_anon("Diumano si X ay nagpa-foul play. Nakuha daw ang pera.")
    assert h.fired


def test_miss_fires_when_no_url_and_no_named_source():
    h = detect_miss("The senator promised many things in his speech.")
    assert h.fired


def test_miss_does_not_fire_when_url_present():
    h = detect_miss("Read the full statement at https://example.gov.ph/123.")
    assert not h.fired


def test_miss_does_not_fire_when_named_source_present():
    h = detect_miss("Sen. Maria Cruz said the bill passed today.")
    assert not h.fired


def test_pol_fires_on_us_vs_them():
    h = detect_pol("Real Filipinos vs traitors to the nation!")
    assert h.fired


def test_cons_fires_on_they_dont_want_you():
    h = detect_cons("They don't want you to know about this hidden agenda.")
    assert h.fired


def test_disc_fires_on_red_tagging():
    h = detect_disc("She is an NPA-supporter, a communist sympathizer.")
    assert h.fired


def test_rev_fires_on_golden_age_claim():
    h = detect_rev("During the golden age there were no martial law abuses.")
    assert h.fired


def test_endo_fires_on_unanchored_survey():
    h = detect_endo("85% of Filipinos already support the candidate!")
    assert h.fired


def test_recf_fires_on_invented_credentials():
    h = detect_recf("He is a Harvard graduate and a Nobel Peace Prize winner.")
    assert h.fired


def test_fab_fires_on_long_quote_no_url():
    h = detect_fab('She declared, "I will rebuild the country single-handedly within twelve months."')
    assert h.fired


def test_fab_does_not_fire_when_url_present():
    h = detect_fab('She declared, "I will rebuild." Full transcript at https://news.example.com/transcript.')
    assert not h.fired


def test_imp_fires_on_lookalike_domain():
    h = detect_imp("Read more at rappler.co (not rappler.com)")
    assert h.fired


def test_extract_indicators_returns_all_12():
    out = extract_indicators("Just some text.")
    assert set(out.keys()) == {"EMO","URG","ANON","MISS","FAB","POL",
                                "CONS","DISC","IMP","REV","ENDO","RECF"}


def test_credible_post_fires_nothing():
    txt = ("According to Sen. Maria Cruz, the Senate passed the bill on third "
           "reading today. Full transcript at https://www.senate.gov.ph/J123.")
    assert fired_codes(txt) == []


def test_named_features_dict_contains_indicator_keys():
    feats = named_features("URGENT! Sources say things!")
    assert "ind_urg_fired" in feats
    assert "ind_anon_fired" in feats
    assert "len_words" in feats
    assert "num_urls" in feats
    assert feats["ind_urg_fired"] == 1.0
    assert feats["ind_anon_fired"] == 1.0


def test_named_features_no_pca_components():
    """The named-feature replacement should NOT include PCA names."""
    feats = named_features("Some text with no indicators.")
    assert all(not k.startswith("rpca") for k in feats)
    assert all(not k.startswith("dpca") for k in feats)
    assert all(not k.startswith("gpca") for k in feats)


def test_indicators_deterministic():
    """Same input → same output."""
    txt = "URGENT! Sources say things!"
    a = fired_codes(txt)
    b = fired_codes(txt)
    assert a == b
