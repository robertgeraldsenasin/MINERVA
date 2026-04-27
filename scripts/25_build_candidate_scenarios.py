
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_index(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % size


def choose(seed: str, items: List[str]) -> str:
    return items[stable_index(seed, len(items))]


DEFAULT_PROFILES = {
    "A": {"name": "Aurelia Santos", "aliases": ["Candidate A", "Aurelia Santos"]},
    "B": {"name": "Bruno Villanueva", "aliases": ["Candidate B", "Bruno Villanueva"]},
    "C": {"name": "Celia Navarro", "aliases": ["Candidate C", "Celia Navarro"]},
}


def load_profiles(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return dict(DEFAULT_PROFILES)
    payload = read_json(path)
    if isinstance(payload, dict) and "candidates" in payload:
        out = {}
        for item in payload["candidates"]:
            cid = str(item.get("candidate_id") or item.get("id") or item.get("code"))
            if cid:
                out[cid] = dict(item)
        for k, v in payload.items():
            if k != "candidates" and isinstance(v, dict):
                out[str(k)] = dict(v)
        return out
    if isinstance(payload, list):
        out = {}
        for item in payload:
            cid = str(item.get("candidate_id") or item.get("id") or item.get("code"))
            if cid:
                out[cid] = dict(item)
        return out
    if isinstance(payload, dict):
        return {str(k): dict(v) for k, v in payload.items()}
    raise ValueError("Unsupported candidate profile JSON structure.")


def load_blue_truths(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "cards" in payload and isinstance(payload["cards"], list):
        seen = {}
        for card in payload["cards"]:
            bt = card.get("linked_blue_truth")
            if isinstance(bt, dict) and bt.get("id") and bt["id"] not in seen:
                seen[bt["id"]] = bt
        return list(seen.values())
    raise ValueError("Unsupported blue-truth JSON structure; provide a list of blue-truth objects.")


def infer_category(bt: Dict[str, Any]) -> str:
    text = str(bt.get("text", "") or "").lower()
    if "debate schedules" in text or "election notices" in text:
        return "general_notice"
    if "completed one full term" in text:
        return "experience_claim"
    if "announced his mayoral bid" in text or "no prior citywide elected office experience" in text:
        return "experience_claim"
    if "law-and-order" in text or "budget plan" in text:
        return "platform_without_budget"
    if "phase 1 rollout" in text or "public health expansion ordinance" in text:
        return "program_rollout"
    if "requires city council approval" in text or "approval before implementation" in text:
        return "approval_pending"
    if "endorsement record" in text or "construction firm" in text:
        return "endorsement_denial"
    if "national statistics" in text or "city-specific crime records" in text:
        return "statistics_context"
    if "open-data" in text or "procurement summaries every quarter" in text:
        return "open_data_proposal"
    if "crime rate decreased" in text:
        return "crime_statistic"
    if "joined 34 of 36" in text or "barangay town halls" in text:
        return "attendance_record"
    if "published costings" in text:
        return "costings_available"
    if "has not posted a formal procurement-transparency plan" in text:
        return "missing_transparency_plan"
    return "generic_election_claim"


OPENERS = {
    "real": ["Ulat:", "Update:", "Fact check:", "Balitang beripikado:"],
    "fake": ["BREAKING:", "Trending:", "Babala:", "Viral post:"],
    "neutral": ["Paalala:", "Sinusuri pa:", "Mabilis na tala:", "Need verification:"],
}


def confidence_for(verdict: str, idx: int) -> float:
    if verdict == "real":
        vals = [6.0, 11.0, 14.0]
    elif verdict == "fake":
        vals = [91.0, 95.0, 97.0]
    else:
        vals = [52.0, 55.0, 58.0]
    return vals[idx % len(vals)]


def difficulty_for(verdict: str, tactic: str) -> str:
    if verdict == "neutral":
        return "hard"
    if tactic in {"false_endorsement", "unsupported_number", "policy_distortion", "contextless_media"}:
        return "medium"
    return "easy"


def candidate_name(bt: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> str:
    cid = bt.get("candidate")
    if cid and cid in profiles:
        return str(profiles[cid].get("name") or f"Candidate {cid}")
    return "the candidates"


def render_templates(bt: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    cid = bt.get("candidate")
    name = candidate_name(bt, profiles)
    text = str(bt.get("text", "") or "")
    city = "San Isidro City"

    def wrap(verdict: str, tactic: str, body: str) -> Tuple[str, str, str]:
        return verdict, tactic, body

    category = infer_category(bt)

    if category == "general_notice":
        return [
            wrap("real", "traceable_update", f"Ayon sa official election bulletin ng {city}, naka-post ang iskedyul ng mga debate at city election notices. Makikita rin ito sa city website at pareho ang detalye sa dalawang source."),
            wrap("real", "traceable_update", f"Nilinaw ng city election office na ang petsa ng public debate ay mababasa lamang sa official bulletin at city website. Walang binagong iskedyul na inilabas sa mga verified channel."),
            wrap("fake", "contextless_media", f"Kumalat ang screenshot na nagsasabing inilipat daw ang debate schedule nang walang abiso sa publiko, pero walang bulletin number, walang opisyal na link, at hindi tugma ang format sa city notice."),
            wrap("fake", "virality_without_evidence", f"Nag-viral ang post na nagsasabing kanselado raw ang lahat ng debate dahil sa 'internal memo,' ngunit walang memo number, walang city seal, at wala ring ulat sa official bulletin."),
            wrap("neutral", "ambiguous_or_uncertain", f"May lumabas na post na sinasabing may dagdag na forum para sa mga kandidato, pero wala pang nakikitang opisyal na city notice. Kailangan pang i-check ang bulletin at city website bago paniwalaan."),
        ]

    if category == "experience_claim":
        if cid == "A":
            return [
                wrap("real", "traceable_update", f"Sa candidate background page, nakasaad na si {name} ang incumbent mayor at nakatapos na ng isang buong apat-na-taong termino. Tugma rin ito sa city archive."),
                wrap("real", "traceable_update", f"Sa debate primer ng {city}, inilista si {name} bilang incumbent mayor na may isang natapos na termino. May pareho ring record sa city archive."),
                wrap("fake", "unsupported_number", f"Nag-viral ang claim na si {name} ay limang termino nang mayor at may mahigit dalawampung taong tuloy-tuloy na city hall leadership. Walang source na ibinigay ang post."),
                wrap("fake", "policy_distortion", f"May post na nagsasabing si {name} ay hindi kailanman naging halal na opisyal ng lungsod. Salungat ito sa city archive ngunit walang dokumentong inilakip ang post."),
                wrap("neutral", "ambiguous_or_uncertain", f"Isang fan page ang nagsasabing mas matagal pa raw sa official record ang public service ni {name}, pero walang direktang link sa city archive. Kailangan pa itong i-verify."),
            ]
        if cid == "B":
            return [
                wrap("real", "traceable_update", f"Nasa candidate profile na si {name} ay inanunsyo ang mayoral bid anim na buwan na ang nakalipas at wala pang dating citywide elected office. Checkable ito sa campaign timeline."),
                wrap("real", "traceable_update", f"Sa debate introduction ng {city}, inilalarawan si {name} bilang bagong mayoral challenger na walang prior citywide elected office experience."),
                wrap("fake", "unsupported_number", f"May post na nagsasabing tatlong termino na raw citywide elected official si {name} bago pa man tumakbo ngayong cycle. Walang office history o city record na kalakip."),
                wrap("fake", "false_endorsement", f"Isang viral graphic ang naglalarawan kay {name} bilang dating vice mayor ng {city}, pero walang official city roster o archive link na sumusuporta rito."),
                wrap("neutral", "ambiguous_or_uncertain", f"May lumang repost na parang nagpapakitang may citywide office experience si {name}, pero hindi malinaw kung advisory role ba iyon o elective position. Kailangan ng mas malinaw na record."),
            ]

    if category == "platform_without_budget":
        return [
            wrap("real", "traceable_update", f"Sa campaign materials ni {name}, malinaw na law-and-order ang pangunahing plataporma, pero wala pang full line-item budget plan na naka-post sa official campaign page."),
            wrap("real", "traceable_update", f"Nilinaw sa debate summary na inuuna ni {name} ang peace-and-order agenda, ngunit wala pang detalyadong itemized budget na inilalabas sa verified campaign sources."),
            wrap("fake", "false_endorsement", f"May post na nagsasabing naglabas na raw si {name} ng kumpletong line-item budget na inaprubahan na rin ng budget office. Wala namang link o dokumentong ibinigay."),
            wrap("fake", "unsupported_number", f"Kumalat ang claim na may nakahandang 'fully funded' law-and-order package si {name} na aabot sa eksaktong halagang binanggit sa post, pero walang costing sheet o official plan."),
            wrap("neutral", "ambiguous_or_uncertain", f"May teaser graphic na nagsasabing ilalabas 'soon' ang budget details ni {name}, pero hanggang ngayon wala pang full document na makikita sa official page."),
        ]

    if category == "program_rollout":
        return [
            wrap("real", "traceable_update", f"Ayon sa city council archive, nasa Phase 1 rollout na ang public health expansion ordinance ni {name}. May dokumentong tumutukoy sa petsa ng pagpapatupad."),
            wrap("real", "traceable_update", f"Kinumpirma sa city bulletin na ang health expansion ordinance na iniuugnay kay {name} ay nasa unang yugto pa lamang ng rollout at hindi pa tapos ang buong programa."),
            wrap("fake", "policy_distortion", f"May viral post na nagsasabing tapos na raw sa buong lungsod ang lahat ng health expansion clinics ni {name}, kahit Phase 1 pa lamang ang makikitang official record."),
            wrap("fake", "unsupported_number", f"Isang card graphic ang nagdedeklarang daan-daan na raw agad ang bagong health sites sa ilalim ni {name}, pero walang ordinance reference, budget line, o city report."),
            wrap("neutral", "ambiguous_or_uncertain", f"May mga residenteng nagpo-post na may rollout na raw sa kanilang barangay ang health expansion program, pero walang naka-attach na official ordinance or bulletin link."),
        ]

    if category == "approval_pending":
        return [
            wrap("real", "traceable_update", f"Ang student transit subsidy na iniuugnay kay {name} ay proposal pa lamang at kailangan pa ng city council approval bago maipatupad. Nasa minutes at proposal summary ang status na ito."),
            wrap("real", "traceable_update", f"Nasa campaign page ni {name} ang transit subsidy proposal, ngunit malinaw na pending pa ito sa city council at wala pang final implementation date."),
            wrap("fake", "policy_distortion", f"May post na nagsasabing magsisimula na raw sa susunod na linggo ang student transit subsidy ni {name}, kahit wala pang city council resolution na naipapakita."),
            wrap("fake", "false_endorsement", f"Kumalat ang claim na aprubado na raw agad ng city council ang transit subsidy plan ni {name}, pero walang resolution number at walang minutes na inilakip."),
            wrap("neutral", "ambiguous_or_uncertain", f"May screenshot ng umano'y launch calendar para sa transit subsidy ni {name}, pero hindi malinaw kung draft lang ba ito o official schedule na talaga."),
        ]

    if category == "endorsement_denial":
        return [
            wrap("real", "traceable_update", f"Batay sa public filings ng {city}, walang formal endorsement record sa pagitan ni {name} at anumang private construction firm."),
            wrap("real", "traceable_update", f"Nilinaw sa public filings review na wala pang naitatalang pormal na construction-firm endorsement para kay {name}."),
            wrap("fake", "false_endorsement", f"Isang viral post ang nagsasabing lihim na ineendorso raw ng isang construction firm si {name}, ngunit wala namang filing, statement, o verified source."),
            wrap("fake", "smear_or_accusation", f"May card na nag-uugnay kay {name} sa isang private construction firm at sinasabing may kapalit na pabor sa proyekto, pero walang kontrata, filing, o official statement na ipinakita."),
            wrap("neutral", "ambiguous_or_uncertain", f"May litrato mula sa isang event na sinasabing patunay ng business endorsement kay {name}, pero wala pang pormal na filing o statement na makapagpapatunay nito."),
        ]

    if category == "statistics_context":
        return [
            wrap("real", "traceable_update", f"Sa mga speech ni {name}, madalas gamitin ang national crime statistics kaysa city-specific crime records. Mahalaga itong ihiwalay kapag sinusuri ang claim."),
            wrap("real", "traceable_update", f"Nasa debate notes na ang crime figures na binabanggit ni {name} ay kadalasang national-level at hindi awtomatikong tumutukoy sa {city}."),
            wrap("fake", "unsupported_number", f"May post na nagsasabing bumagsak daw nang eksaktong porsiyento ang crime sa {city} dahil sa plataporma ni {name}, pero national figure lang ang ginamit at walang city record."),
            wrap("fake", "contextless_media", f"Kumalat ang infographic na iniuugnay kay {name} at sinasabing city-specific daw ang crime numbers, pero wala namang source note at mukhang galing sa mas malawak na national dataset."),
            wrap("neutral", "ambiguous_or_uncertain", f"May shared image na gumagamit ng crime chart habang binabanggit si {name}, pero hindi malinaw kung local data ba ito o national statistics lamang."),
        ]

    if category == "open_data_proposal":
        return [
            wrap("real", "traceable_update", f"Ang open-data proposal ni {name} ay magsasapubliko ng city procurement summaries kada quarter kung maipapasa. Nasa proposal page ang pangakong ito."),
            wrap("real", "traceable_update", f"Sa campaign website ni {name}, nakasaad na quarterly procurement summaries ang target ng open-data plan kung maaaprubahan."),
            wrap("fake", "policy_distortion", f"May post na nagsasabing live na raw ang procurement transparency portal ni {name}, kahit proposal pa lamang ang nasa campaign materials."),
            wrap("fake", "unsupported_number", f"Isang graphic ang nagsasabing nakapaglabas na raw si {name} ng buong procurement database, pero walang portal link o official rollout notice."),
            wrap("neutral", "ambiguous_or_uncertain", f"May screenshot ng tila procurement dashboard na iniuugnay kay {name}, pero walang malinaw na official URL o city notice na nagpapatunay kung live na ito."),
        ]

    if category == "crime_statistic":
        return [
            wrap("real", "traceable_update", f"Ayon sa city report, bumaba ang reported crime rate sa nakaraang dalawang taon sa ilalim ng administrasyon ni {name}. Dapat pa ring tingnan ang timeframe at methodology."),
            wrap("real", "traceable_update", f"May city annual report na nagsasabing bumaba ang reported crime rate sa loob ng dalawang taon habang nanunungkulan si {name}."),
            wrap("fake", "unsupported_number", f"May viral image na nagsasabing doble raw ang crime sa ilalim ni {name}, pero walang city report, walang year labels, at walang methodology."),
            wrap("fake", "contextless_media", f"Isang cropped chart ang ipinapakalat laban kay {name} na tila nagpapakitang sumipa ang crime rate, ngunit hindi nakikita ang original source, axis labels, o date range."),
            wrap("neutral", "ambiguous_or_uncertain", f"May graphics war online tungkol sa crime record ni {name}, pero hindi malinaw kung alin ang gumagamit ng official city report at alin ang edit lamang."),
        ]

    if category == "attendance_record":
        return [
            wrap("real", "traceable_update", f"Nasa attendance records na dumalo si {name} sa 34 sa 36 na scheduled barangay town halls sa kasalukuyang termino."),
            wrap("real", "traceable_update", f"Makikita sa barangay schedule records na halos lahat ng town hall sessions ay nadaluhan ni {name}, maliban sa dalawang petsa."),
            wrap("fake", "unsupported_number", f"May post na nagsasabing hindi raw kailanman sumipot si {name} sa town halls, kahit may attendance records na kabaligtaran ang sinasabi."),
            wrap("fake", "policy_distortion", f"Isang montage ang nag-aangkin na perpekto raw ang attendance ni {name} sa higit limampung town halls, pero hindi tugma ang bilang sa official schedule."),
            wrap("neutral", "ambiguous_or_uncertain", f"May collage ng event photos na ginagamit bilang patunay ng attendance ni {name}, pero walang buong schedule at walang kumpletong date list."),
        ]

    if category == "costings_available":
        return [
            wrap("real", "traceable_update", f"Na-publish na sa official campaign website ni {name} ang costings para sa top three proposals niya. May downloadable summary ang page."),
            wrap("real", "traceable_update", f"Sa verified campaign site ni {name}, makikita ang costing breakdown ng tatlong pangunahing pangako sa kampanya."),
            wrap("fake", "unsupported_number", f"May post na nagsasabing walang anumang costings si {name}, kahit may naka-post na costing summaries sa official campaign website."),
            wrap("fake", "policy_distortion", f"Isang viral card ang nag-uugnay kay {name} sa sobrang laking budget figure ngunit walang link sa actual costing sheet o campaign source."),
            wrap("neutral", "ambiguous_or_uncertain", f"May screenshot ng costing table na iniuugnay kay {name}, pero dahil walang URL o verified page reference, kailangan pa ring i-check ang official site."),
        ]

    if category == "missing_transparency_plan":
        return [
            wrap("real", "traceable_update", f"Sa latest campaign update, wala pang pormal na procurement-transparency plan na naka-post para kay {name}."),
            wrap("real", "traceable_update", f"Na-verify sa campaign updates na wala pang formal transparency-plan document si {name} kahit madalas itong itanong sa debate."),
            wrap("fake", "false_endorsement", f"May post na nagsasabing naaprubahan na raw ang formal procurement-transparency plan ni {name}, pero walang plan document at walang official announcement."),
            wrap("fake", "smear_or_accusation", f"Isang viral claim ang naglalabas ng umano'y 'internal memo' tungkol sa transparency plan ni {name}, pero walang document provenance at walang verified source."),
            wrap("neutral", "ambiguous_or_uncertain", f"May supporters na nagsasabing may draft na raw ng transparency plan ni {name}, pero wala pa ring public release na puwedeng suriin."),
        ]

    # generic fallback
    return [
        wrap("real", "traceable_update", text),
        wrap("real", "traceable_update", f"Ayon sa official election materials, {text}"),
        wrap("fake", "ambiguous_or_uncertain", f"May viral post na nagbibigay ng ibang bersyon ng claim na ito, pero walang source na mahanap."),
        wrap("fake", "ambiguous_or_uncertain", f"May graphic na nagmamalaking siguradong totoo raw ang claim na ito, ngunit wala ring dokumentong ipinakita."),
        wrap("neutral", "ambiguous_or_uncertain", f"May post na inuulit ang claim na ito ngunit hindi malinaw kung alin ang original source."),
    ]


def make_card(bt: Dict[str, Any], profiles: Dict[str, Dict[str, Any]], verdict: str, tactic: str, body: str, seq: int) -> Dict[str, Any]:
    cid = bt.get("candidate")
    opener = choose(f"{bt.get('id')}-{verdict}-{seq}", OPENERS[verdict])
    fake_likelihood = confidence_for(verdict, seq)
    credibility = round(100.0 - fake_likelihood, 3)
    label = verdict
    if verdict == "neutral":
        label = "neutral"
    card_id = f"tpl_{bt.get('id','BT')}_{verdict}_{seq:02d}"
    targets = [cid] if cid else []

    return {
        "id": card_id,
        "target_label": label,
        "text": f"{opener} {body}",
        "verdict": label,
        "fake_likelihood_percent": fake_likelihood,
        "credibility_percent": credibility,
        "difficulty_bin": difficulty_for(verdict, tactic),
        "metadata": {
            "generation_mode": "rule_constrained_template",
            "score_source": "template_rule",
            "verdict_source": "template_rule",
            "needs_model_rescore": True,
            "tactic": tactic,
            "linked_blue_truth_id": bt.get("id"),
        },
        "p_fake": round(fake_likelihood / 100.0, 4),
        "targets": targets,
        "theme_flags": {
            "is_on_theme": True,
            "candidate_targets": targets,
            "candidate_focus": "single" if len(targets) == 1 else ("none" if not targets else "multi"),
            "election_keywords": ["candidate", "campaign", "city", "official"],
        },
        "linked_blue_truth": bt,
        "classification": {
            "truth_type": "blue" if verdict == "real" else ("red" if verdict == "fake" else "neutral"),
            "is_misinformation": verdict == "fake",
            "targets": targets,
            "linked_blue_truth_id": bt.get("id"),
            "explanation": "Rule-constrained candidate scenario. Run the teaching-curation script for detailed student feedback.",
        },
        "scenario_tactic": tactic,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate coherent three-candidate election cards from blue truths using rule-constrained templates."
    )
    ap.add_argument("--blue_truths", required=True, help="Blue truths JSON list.")
    ap.add_argument("--out_file", required=True, help="Output card pool JSON.")
    ap.add_argument("--candidate_profiles", default=None)
    args = ap.parse_args()

    blue_truths = load_blue_truths(Path(args.blue_truths))
    profiles = load_profiles(Path(args.candidate_profiles) if args.candidate_profiles else None)

    out: List[Dict[str, Any]] = []
    for bt in blue_truths:
        scenarios = render_templates(bt, profiles)
        for idx, (verdict, tactic, body) in enumerate(scenarios, start=1):
            out.append(make_card(bt, profiles, verdict, tactic, body, idx))

    write_json(Path(args.out_file), out)
    print(f"[OK] Wrote {len(out)} rule-constrained cards to {args.out_file}")


if __name__ == "__main__":
    main()
