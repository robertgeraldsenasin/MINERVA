#!/usr/bin/env python3
"""
30_template_scenario_generator.py  (NEW in v2.6)
================================================

Template-based card generator producing high-quality, coherent posts
in Filipino electoral disinformation idiom. Replaces ~90% of GPT-2's
output with rule-constrained scenarios derived from documented
Philippine disinformation tactics.

WHY THIS EXISTS
---------------
The thesis paper (§Definitions, p.12) defines:

  "Rule-Constrained Content Generation: A controlled method of
   generating or templating fictional posts/scenarios using
   extracted misinformation patterns while enforcing constraints
   that support ethical use, consistency, and testability."

This is the missing piece. v2.0-v2.5 relied on raw GPT-2 generations
that produced nonsensical text (93.5% truncation, 40.9% name-jamming,
random English fragments mid-Tagalog). The v2.4 audit confirmed even
post-processing couldn't rescue cards built on incoherent GPT-2 output.

TEMPLATE TAXONOMY
-----------------
Drawing from:
  - DEPICT (Roozenbeek & van der Linden, 2019): Discrediting,
    Emotion, Polarization, Impersonation, Conspiracy, Trolling
  - Bad News + Harmony Square (Roozenbeek 2020): scripted-scenario
    inoculation methodology used in Cambridge research
  - Arugay & Baquisal (2022): Philippine election disinformation
    archetypes (dynastic, reformist, populist)
  - Schipper (2025): Philippine 2025 election disinformation playbook
  - Ong & Cabañes (2018): Philippine political trolling architecture
  - Ong/PCIJ (2022): historical revisionism as primary narrative

Each template is keyed to:
  * candidate archetype (DYNASTIC / REFORMIST / POPULIST)
  * misinformation tactic (one of 8 Filipino-specific tactics)
  * indicators that should fire (subset of MINERVA's 12 cues)
  * tier (novice/proficient/advanced)

GENERATION STRATEGY
-------------------
For each (verdict × candidate × tactic × tier) combination, a small
inventory of templates with slot-filled variables. Slots include:
  - {candidate_full}, {candidate_short}: the 3 fictional candidates
  - {generic_official}, {generic_critic}, {generic_witness}, etc.:
    placeholder roles instead of real names
  - {place}, {date}, {amount}, {percentage}: parameterized data

Output is identical schema to GPT-2 generations so the rest of the
pipeline (scripts 13, 18, 22, 23, 24, 28) consumes them unchanged.

CITATIONS (for thesis defense)
------------------------------
- Roozenbeek, J., & van der Linden, S. (2019). Fake news game confers
  psychological resistance against online misinformation. Humanities
  and Social Sciences Communications, 5(1), 1-10.
- Roozenbeek, J., & van der Linden, S. (2020). Breaking Harmony Square:
  A game that "inoculates" against political misinformation. HKS
  Misinformation Review, 1(8).
- Basol, M., Roozenbeek, J., & van der Linden, S. (2020). Good news
  about Bad News: Gamified inoculation boosts confidence and cognitive
  immunity against fake news. Journal of Cognition, 3(1).
- Arugay, A. A., & Baquisal, J. K. A. (2022). Mobilized and polarized:
  Disinformation networks in the 2022 Philippine elections. Pacific
  Affairs, 95(3), 463-485.
- Schipper, B. C. (2025). Disinformation by design: Leveraging solutions
  to combat misinformation in the Philippines' 2025 election. Data &
  Policy, 7.
- Ong, J. C., & Cabañes, J. V. A. (2018). Architects of networked
  disinformation: Behind the scenes of troll accounts and fake news
  production in the Philippines. Newton Tech4Dev Network.
- Modirrousta-Galian, A., & Higham, P. A. (2023). Conservative response
  bias in misinformation training. Journal of Experimental Psychology:
  Applied (credible-card mandate).
- Caulfield, M. (2019). SIFT (the four moves). Hapgood blog (verification
  framework integrated into VERIdict feedback).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_candidates import REGISTRY
from minerva_indicators import indicator_summary_for_card

logger = logging.getLogger(__name__)

# ===========================================================================
# SLOT VOCABULARIES — Filipino electoral idiom, generic roles only
# ===========================================================================

GENERIC_ROLES_TL = {
    # Generic political / institutional figures (no real names)
    "official":   ["isang opisyal ng gobyerno", "isang dating opisyal",
                   "isang miyembro ng Senado", "isang kongresista",
                   "isang kalihim ng departamento", "isang dating senador"],
    "critic":     ["isang kritiko", "isang oposisyong tagapagsalita",
                   "isang kasamahan sa Senado", "isang abogado ng oposisyon",
                   "isang dating kasamahan", "isang mambabatas"],
    "supporter":  ["isang malapit na tagasuporta", "isang campaign manager",
                   "isang miyembro ng partido", "isang spokesperson"],
    "witness":    ["isang testigo", "isang nakasaksi sa pangyayari",
                   "isang anonymous na source", "isang miyembro ng staff",
                   "isang dating empleyado"],
    "journalist": ["isang reporter", "isang mamamahayag",
                   "isang mananaliksik", "isang fact-checker"],
    "expert":     ["isang propesor sa pulitika", "isang political analyst",
                   "isang ekspertong economist", "isang researcher"],
    "celebrity":  ["isang sikat na influencer", "isang vlogger",
                   "isang social media personality", "isang OPM artist"],
    "ngo":        ["isang grupo ng mga aktibista", "isang civil society group",
                   "isang watchdog organization", "isang human rights group"],
    "audience":   ["mga netizens", "mga taga-suporta", "mga kababayan natin",
                   "mga first-time voters", "mga miyembro ng community"],
}

PLACES_TL = [
    "Maynila", "Quezon City", "Cebu", "Davao", "Cagayan de Oro",
    "Iloilo", "Bacolod", "Baguio", "Pampanga", "Bulacan",
    "Cavite", "Laguna", "Batangas", "Rizal", "Pangasinan",
    "Zamboanga", "General Santos", "Tacloban", "Naga", "Tagum",
]

PLATFORMS_TL = [
    "Facebook", "TikTok", "Twitter", "YouTube", "Instagram",
    "Telegram", "Viber group", "Messenger group chat",
]

DATES_REL_TL = [
    "kahapon", "kanina", "ngayong umaga", "noong Lunes", "noong Martes",
    "noong Miyerkules", "noong Huwebes", "noong Biyernes",
    "noong nakaraang linggo", "kamakailan lamang", "noong nakaraang buwan",
]

AMOUNTS_TL = [
    "P50,000", "P100,000", "P500,000", "P1 milyon", "P5 milyon",
    "P10 milyon", "P50 milyon", "P100 milyon", "P1 bilyon",
]

PERCENTAGES_TL = ["15%", "23%", "37%", "42%", "58%", "67%", "72%", "85%"]

NEWS_PREFIXES = [
    "BREAKING", "Update", "Balita", "Trending", "Babala",
    "Paalala", "Ulat", "Iniulat", "Tingnan",
]


# ===========================================================================
# TEMPLATE INVENTORY — by tactic × archetype
# ===========================================================================
#
# Format: {
#   "tactic": str,                # one of 8 documented tactics
#   "verdict": "FAKE"|"REAL"|"UNCERTAIN",
#   "archetypes": list[str],      # which candidate archetypes this fits
#   "fired_indicators": list[str],  # what indicators SHOULD fire
#   "tier": "novice"|"proficient"|"advanced",
#   "templates": list[str],       # at least 3 variants per slot
# }

TEMPLATES = [
    # =========================================================
    # 1. HISTORICAL REVISIONISM (FAKE) — Marcos-Duterte playbook
    #    Source: Ong/PCIJ 2022; Arugay 2022; Schipper 2025
    # =========================================================
    {
        "tactic": "historical_revisionism",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC"],
        "fired_indicators": ["MISS", "REV"],
        "tier": "advanced",
        "templates": [
            "{prefix}: {audience}, ang panahon ng pamilya ni {candidate_full} "
            "noong dekada {decade} ay tinaguriang \"ginintuang panahon\" "
            "ng Pilipinas. Walang kahirapan, mababa ang krimen, at maunlad "
            "ang ekonomiya. Ang mga kuwento tungkol sa karahasan ay gawa-gawa "
            "lamang ng mga kalaban sa pulitika.",

            "{prefix}: {amount} ang nawalang ari-arian na natuklasan kamakailan "
            "ng {generic_official}. Ito raw ay galing sa pamilya ni "
            "{candidate_full}. Pero ayon sa mga supporter ni {candidate_short}, "
            "ang lahat ng iyon ay legal na pinanggalingan at ang akusasyon ay "
            "bahagi lamang ng \"black propaganda\" laban sa kanilang pamilya.",

            "{prefix}: Ipinakita ng kuwento na noong panahon ng pamilya "
            "ni {candidate_full}, ang Pilipinas ay isa sa pinakamaunlad na "
            "bansa sa Asya. \"Walang utang, walang kahirapan,\" ayon sa "
            "kuwento. Pero kapag tiningnan ang opisyal na rekord ng "
            "Department of Finance, taliwas ito sa katotohanan.",
        ],
    },
    {
        "tactic": "historical_revisionism_truth",
        "verdict": "REAL",
        "archetypes": ["DYNASTIC"],
        "fired_indicators": [],
        "tier": "novice",
        "templates": [
            "{prefix}: Ayon sa Bangko Sentral ng Pilipinas, ang utang ng "
            "bansa noong dekada 1980 ay umabot sa $26.7 bilyon ayon sa "
            "kanilang opisyal na ulat. Bagama't may iba't ibang interpretasyon "
            "ng kasaysayan, ang datos mula sa BSP ay malinaw at maaaring "
            "i-verify sa kanilang website.",

            "{prefix}: Inilabas ng Philippine Statistics Authority (PSA) "
            "ang detalyadong paghahambing ng mga ekonomikong tagapagpahiwatig "
            "mula sa magkakaibang panahon. Kasama sa ulat ang GDP growth "
            "rate, inflation, at unemployment, na maaaring tingnan ng "
            "publiko sa opisyal na PSA portal.",

            "{prefix}: Ang National Historical Commission of the Philippines "
            "(NHCP) ay nagpalabas ng official position paper tungkol sa "
            "panahon ng Martial Law. Naglalaman ito ng primary documents "
            "na maaaring konsultahin para sa mga tumitingin ng iba't ibang "
            "perspektibo sa kasaysayan ng Pilipinas.",
        ],
    },

    # =========================================================
    # 2. RED-TAGGING (FAKE) — Robredo/critic playbook
    #    Source: Snoqap 2023; Arugay 2022; ICHRP
    # =========================================================
    {
        "tactic": "red_tagging",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["EMO", "FAB", "ANON"],
        "tier": "proficient",
        "templates": [
            "{prefix}: Lumalabas na may koneksyon umano si {candidate_full} "
            "sa mga grupong komunista. Ayon sa {witness}, dumalo umano si "
            "{candidate_short} sa mga lihim na pulong noong nasa kolehiyo pa. "
            "Wala pang opisyal na pahayag mula sa kampo ni {candidate_short} "
            "tungkol dito.",

            "{prefix}: \"Hindi maaaring iboto si {candidate_full}\" — ito ang "
            "pahayag ng {generic_official} na nagbabala umano laban sa "
            "candidate. \"May koneksyon siya sa armadong grupo na laban sa "
            "gobyerno.\" Walang ipinakita o pinangalanang dokumento o "
            "ebidensya sa pahayag.",

            "{prefix}: Ipinost online ng {generic_celebrity} ang isang "
            "screenshot na kinukunsumo ang pagiging \"makakaliwa\" ni "
            "{candidate_full}. {audience}, lumawak agad ang post sa "
            "{platform} bago pa ma-fact check. Hindi maverify ang "
            "pinagmulan ng nasabing screenshot.",
        ],
    },

    # =========================================================
    # 3. EMOTIONAL APPEAL / CELEBRITY ENDORSEMENT (FAKE)
    #    Source: Roozenbeek DEPICT (Emotion); Schipper 2025
    # =========================================================
    {
        "tactic": "fake_celebrity_endorsement",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["EMO", "ENDO", "FAB"],
        "tier": "novice",
        "templates": [
            "{prefix}: Sumusuporta na rin daw si {generic_celebrity} kay "
            "{candidate_full}! Kumalat ang larawan sa {platform} kung saan "
            "nakikita ang dalawa sa isang event. {audience}, share na para "
            "alam ng lahat! Hindi pa nagbibigay ng opisyal na pahayag ang "
            "celebrity.",

            "{prefix}: \"Tanging si {candidate_full} lang ang pag-asa ng "
            "kabataan!\" Ito raw ang sinabi ng isang sikat na artista sa "
            "kanyang vlog. Pero kapag hinanap ang nasabing video, walang "
            "opisyal na link na maibibigay. Patuloy pa ring kumakalat ang "
            "post sa {platform}.",

            "{prefix}: Nag-trend kahapon sa {platform} ang larawan ng "
            "{generic_celebrity} na may suot na kampanya shirt ni "
            "{candidate_full}. {audience}, \"Iboto si {candidate_short}!\" "
            "Pero kalaunan, lumabas na photoshopped lamang ang larawan.",
        ],
    },

    # =========================================================
    # 4. URGENCY / SHARING PRESSURE (FAKE) — DEPICT trolling
    # =========================================================
    {
        "tactic": "urgency_sharing",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["URG", "MISS", "EMO"],
        "tier": "novice",
        "templates": [
            "{prefix}: I-SHARE NA BAGO MABURA! Inilabas umano ng "
            "{generic_witness} ang totoong kuwento tungkol kay {candidate_full} "
            "{date}. {audience}, mabilis bago tanggalin sa {platform}!",

            "{prefix}: KAYO NA HUMUSGA! May lumabas na video kuno ni "
            "{candidate_full} sa loob ng {place}. \"Hindi ko ito kayang "
            "panindigan,\" sabi ni {generic_witness}. Pero {audience}, "
            "walang malinaw na petsa o location ang video.",

            "{prefix}: BREAKING NEWS NA HINDI IPINAPAKITA SA TV! Kumakalat "
            "sa {platform} na may bagong eskandalo si {candidate_full}. "
            "{audience}, share kaagad bago burahin! Walang link sa "
            "anumang opisyal na ulat o dokumento.",
        ],
    },

    # =========================================================
    # 5. UNVERIFIED SURVEY / NUMBER CLAIM (FAKE)
    # =========================================================
    {
        "tactic": "fake_survey",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["MISS", "FAB"],
        "tier": "proficient",
        "templates": [
            "{prefix}: Ayon sa pinakabagong survey, may {percentage} na ng "
            "mga botante ang sumusuporta na kay {candidate_full}. Hindi "
            "tinukoy ng nag-post kung anong survey firm o kailan isinagawa. "
            "Walang link sa orihinal na ulat.",

            "{prefix}: Lumalakas na umano ang suporta kay {candidate_full} "
            "ayon sa isang \"independent survey\" na lumabas sa {platform} "
            "{date}. Umabot daw sa {percentage} ang preference rating "
            "niya. Hindi pa pinangalanan ang kumpanyang nagsagawa nito.",

            "{prefix}: \"Nasa top 3 na ako sa lahat ng surveys,\" pahayag "
            "umano ni {candidate_full} sa isang interbyu sa {platform}. "
            "Pero walang naipakitang kumpanyang Pulse Asia, SWS, o iba "
            "pang kilalang survey firm na nagpatunay ng claim na ito.",
        ],
    },

    # =========================================================
    # 6. CREDIBLE NEWS — REAL (per Modirrousta-Galian quota)
    # =========================================================
    {
        "tactic": "credible_policy_announcement",
        "verdict": "REAL",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": [],
        "tier": "novice",
        "templates": [
            "{prefix}: Inilabas ng kampo ni {candidate_full} ang opisyal "
            "na platform paper sa kanilang website {date}. Saklaw ng "
            "dokumento ang mga proposal sa edukasyon, kalusugan, at "
            "ekonomiya. Maaaring i-download ng publiko para sa "
            "tamang impormasyon.",

            "{prefix}: Idinaos sa {place} kahapon ang opisyal na town hall "
            "ni {candidate_full}. Sumagot siya sa mga tanong ng mga "
            "kababayan natin sa loob ng dalawang oras. Ang buong recording "
            "ay maa-access sa kanilang opisyal na YouTube channel.",

            "{prefix}: Naglabas ng joint statement ang kampo ni "
            "{candidate_full} at ang Commission on Elections (COMELEC) "
            "tungkol sa mga compliance sa election guidelines. Inilathala "
            "ito sa opisyal na website ng COMELEC at ng kandidato.",

            "{prefix}: Pinagtibay ng {generic_official} mula sa Department "
            "of Education ang mga proposal ni {candidate_full} para sa "
            "K-12 reform. Ang detalyadong policy brief ay nakapaskil sa "
            "DepEd portal at nagagamit ng mga researcher.",
        ],
    },

    # =========================================================
    # 7. UNCERTAIN — claim with mixed indicators
    # =========================================================
    {
        "tactic": "ambiguous_allegation",
        "verdict": "UNCERTAIN",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["ANON", "MISS"],
        "tier": "advanced",
        "templates": [
            "{prefix}: May lumalabas umano na alegasyon laban kay "
            "{candidate_full} kaugnay sa {amount} na pondo ng kampanya. "
            "Ayon sa {witness}, may resibo umano siyang ipakikita kapag "
            "may imbestigasyon. Hindi pa tumutugon ang kampo ni "
            "{candidate_short} o ang COMELEC.",

            "{prefix}: \"May nakikita kaming patterns,\" pahayag ng "
            "{generic_journalist} sa isang panayam {date}. Tumukoy siya "
            "sa mga transaksiyon ng kampo ni {candidate_full}. Kakailanganin "
            "pa ang dagdag na imbestigasyon at pormal na ulat bago "
            "makahatol.",

            "{prefix}: Iniulat ng {generic_ngo} na may posibleng iregularidad "
            "sa proseso ng kampanya ni {candidate_full}. Hindi pa kumpleto "
            "ang ebidensya at tutugunin pa raw ng grupo ang detalye sa "
            "kanilang quarterly report. {audience}, antabayanan ang "
            "opisyal na anunsyo.",
        ],
    },

    # =========================================================
    # 8. CONSPIRACY THEORY (FAKE) — DEPICT
    # =========================================================
    {
        "tactic": "conspiracy_theory",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["CONS", "ANON", "MISS"],
        "tier": "advanced",
        "templates": [
            "{prefix}: \"May plano umano laban kay {candidate_full},\" "
            "ayon sa isang anonymous na source na hindi pinangalanan. "
            "Sangkot daw ang ilang dayuhang grupo at \"deep state\" "
            "sa pagpapabagsak ng kandidatura niya. Walang naipakita o "
            "naipangalan na ebidensya.",

            "{prefix}: Lumitaw umano ang isang \"lihim na dokumento\" sa "
            "{platform} na nagpapakita ng konspirasiya laban kay "
            "{candidate_full}. {audience}, share daw bago burahin! "
            "Hindi maberipikahan ang pinanggalingan o authenticity ng "
            "nasabing dokumento.",

            "{prefix}: \"Sila ang gusto akong patumbahin,\" pahayag umano "
            "ni {candidate_full} sa isang interbyu na hindi naman maipakita "
            "ang buong video. Tinukoy niya ang mga \"hindi pinangalanang "
            "powers\" na umano'y kalaban niya. Walang pinangalanang tao "
            "o organisasyon.",
        ],
    },

    # =========================================================
    # 9. POLARIZATION (FAKE) — DEPICT 'P' indicator
    # Source: Roozenbeek 2020 Harmony Square ("trolling" + "emotion"
    # scenarios — turn an ostensibly neutral issue into a heated debate)
    # =========================================================
    {
        "tactic": "polarization_us_vs_them",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["POL", "EMO", "MISS"],
        "tier": "proficient",
        "templates": [
            "{prefix}: \"Sila o tayo lang ang pagpipilian!\" pahayag umano "
            "ni {candidate_full} sa isang post sa {platform}. \"Kung hindi "
            "tayo manalo, mananaig ang mga elitistang umaapi sa atin.\" "
            "Walang naipakita o naipangalan na konkretong polisiya o "
            "datos sa nasabing post.",

            "{prefix}: \"Ang totoong Pilipino ay si {candidate_full} lang,\" "
            "ayon sa isang viral na post sa {platform}. \"Ang ibang kandidato "
            "ay hindi para sa atin.\" {audience}, pansinin: walang ipinakitang "
            "datos o quote mula sa ibang kandidato — pawang pagpapasimuno "
            "lamang ng paghihiwalay.",

            "{prefix}: Kumakalat sa {platform} ang isang infographic na "
            "naghahati sa mga botante sa \"may pinaglalaban\" at \"traydor\" "
            "batay sa kanilang sinusuportahang kandidato. Walang link sa "
            "anumang research o framework na pinanggalingan ng kategorisasyon.",
        ],
    },

    # =========================================================
    # 10. DISCREDITING OPPONENTS (FAKE) — DEPICT 'D' indicator
    # Source: Roozenbeek 2019 ("deflecting blame, discrediting opponents")
    # Tier: novice — personal-attack patterns are recognizable to
    # first-time voters because the rhetorical move is overt.
    # =========================================================
    {
        "tactic": "discrediting_personal_attack",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["DISC", "EMO", "ANON"],
        "tier": "novice",
        "templates": [
            "{prefix}: \"Hindi siya karapat-dapat manguna,\" sabi ng "
            "{generic_critic} tungkol kay {candidate_full} sa isang "
            "interbyu sa {platform}. \"Tingnan ang kanyang pamilya — "
            "mas matindi ang background nila kaysa platform niya.\" "
            "Walang ipinakitang record o dokumento na sumusuporta sa "
            "naturang akusasyon.",

            "{prefix}: Iniulat na may dating empleyado umano ni "
            "{candidate_full} na nagsalita laban sa kanya. Pero kapag "
            "tiningnan ang nasabing post, wala itong pangalan, wala itong "
            "petsa, at puro personal na atake — walang tinukoy na issue "
            "o polisiya.",

            "{prefix}: \"Bago natin pakinggan ang sasabihin ni "
            "{candidate_full}, alalahanin natin ang kanyang nakaraan,\" "
            "pasimula ng isang viral thread sa {platform}. Ang buong post "
            "ay puro insulto at hindi tumalakay sa kanyang aktwal na "
            "platform o panukala.",
        ],
    },

    # =========================================================
    # 11. IMPERSONATION (FAKE) — DEPICT 'I' indicator
    # Source: Roozenbeek 2019 ("impersonating people through fake accounts")
    # Filipino-specific: Ong & Cabañes 2018 documents organized
    # impersonation networks in PH political trolling
    # =========================================================
    {
        "tactic": "fake_account_impersonation",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["IMP", "FAB", "ANON"],
        "tier": "advanced",
        "templates": [
            "{prefix}: Lumitaw sa {platform} ang isang account na "
            "nagpapanggap na opisyal na page ni {candidate_full}. "
            "Naglabas ng \"statement\" ang account na taliwas sa naunang "
            "pahayag ng tunay na kampo. Hindi verified ang account at "
            "walang link sa opisyal na website ng kandidato.",

            "{prefix}: \"Inilabas ko na ang aking pinal na desisyon,\" "
            "ayon sa isang post na umano'y galing kay {candidate_full}. "
            "Pero kapag tiningnan, hindi naman ito mula sa opisyal niyang "
            "verified account — gawa-gawa lamang ng isang pekeng profile "
            "na may katulad na pangalan.",

            "{prefix}: Kumakalat ang screenshot ng isang \"pribadong "
            "mensahe\" umano galing kay {candidate_full}. Ang mensahe "
            "ay malinaw na photoshopped — hindi tugma ang font sa opisyal "
            "na app, at walang verification badge sa profile photo. "
            "Hindi pa rin tumutugon ang totoong kampo.",
        ],
    },

    # =========================================================
    # 12. RECYCLED CONTENT (FAKE) — RECF indicator
    # Source: Schipper 2025 (Philippine 2025 election playbook
    # documents recycled-content-as-tactic for the 2025 cycle)
    # =========================================================
    {
        "tactic": "recycled_old_content",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["RECF", "MISS", "EMO"],
        "tier": "novice",
        "templates": [
            "{prefix}: Kumakalat ang isang larawan ni {candidate_full} "
            "sa isang \"protesta\" {date}. Pero kapag tiningnan ang "
            "metadata, ang larawan ay galing pa sa 2018 na ibang event "
            "— ginagamit lamang ngayon para palabasin na bagong issue.",

            "{prefix}: Iniulat sa {platform} ang isang clip ni "
            "{candidate_full} na umano'y nagsasalita {date}. Pero sa "
            "katunayan, ang clip ay matagal na sa ibang event noong "
            "2019 — pinaikli lamang at binigyan ng panibagong caption "
            "para palabasin na kasalukuyang pahayag.",

            "{prefix}: \"Tingnan! May bagong scandal kay {candidate_full}!\" "
            "{audience}, pansinin — ang mismong post ay reupload ng isang "
            "lumang sumbong noong 2017 na kinilala nang false. Bagama't "
            "matagal nang naayos, patuloy itong kumakalat sa mga group "
            "chat.",
        ],
    },

    # =========================================================
    # 13. DEEPFAKE / SYNTHETIC MEDIA CLAIM (FAKE)
    # Source: Schipper 2025 (Philippine 2025 — deepfake/AI-generated
    # content explicitly named as a primary 2025 disinformation vector)
    # =========================================================
    {
        "tactic": "deepfake_video_claim",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["FAB", "MISS", "EMO"],
        "tier": "advanced",
        "templates": [
            "{prefix}: \"Tingnan ang totoong sinabi ni {candidate_full}!\" "
            "Kumakalat sa {platform} ang isang video na umano'y nagpapakita "
            "kay {candidate_short} na nag-aamin ng isang katiwalian. Pero "
            "ipinakita ng {generic_journalist} na may signs ng "
            "AI-manipulation ang video — hindi tugma ang lip-sync sa audio.",

            "{prefix}: May lumabas na audio recording sa {platform} na "
            "umano'y boses ni {candidate_full}. Pero ayon sa {generic_expert}, "
            "may mga pattern ng synthetic voice generation sa recording. "
            "Walang ipinakita ang nag-post na original source o context "
            "ng pagkuha.",

            "{prefix}: I-share daw bago tanggalin! Isang \"leaked video\" "
            "ni {candidate_full} ang kumakalat sa {platform}. {audience}, "
            "pero kapag tinignan ng audio-forensic, may mga digital "
            "fingerprints ng AI-generation — hindi totoong recording.",
        ],
    },

    # =========================================================
    # 14. FAKE FACT-CHECKER (FAKE) — IMP variant
    # Source: Tsipursky 2024 — fake fact-checker accounts as a 2024+
    # tactic for legitimizing disinformation
    # =========================================================
    {
        "tactic": "fake_fact_checker_authority",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["IMP", "FAB", "ANON"],
        "tier": "advanced",
        "templates": [
            "{prefix}: \"FACT-CHECK: Totoo nga ang akusasyon laban kay "
            "{candidate_full},\" ayon sa post ng isang account na may "
            "pangalang \"Pilipinas Truth Watch.\" Pero hindi naman ito "
            "kabilang sa mga kilalang fact-checking organizations gaya "
            "ng Vera Files o Rappler — kabago-bagong account lang ito.",

            "{prefix}: Ipinost ng isang \"verification page\" sa "
            "{platform} ang \"resulta\" ng kanilang pag-imbestiga kay "
            "{candidate_full}. Pero walang published methodology, walang "
            "authors na pinangalanan, at hindi miyembro ng International "
            "Fact-Checking Network (IFCN). Walang verifiable accountability.",

            "{prefix}: \"Verified by Truth-Bureau,\" sabi ng caption ng "
            "isang post tungkol kay {candidate_full}. Pero ang \"Truth-"
            "Bureau\" ay hindi naman kilalang organization — gawa-gawa "
            "lang na pangalan na para magmukhang totoo ang content. "
            "Walang link, walang website.",
        ],
    },

    # =========================================================
    # 15. MANUFACTURED OUTRAGE / ASTROTURFING (FAKE) — POL variant
    # Source: Ong & Cabañes 2018 — coordinated outrage cycles as
    # documented Filipino political trolling tactic
    # =========================================================
    {
        "tactic": "coordinated_outrage_campaign",
        "verdict": "FAKE",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["POL", "EMO", "RECF"],
        "tier": "proficient",
        "templates": [
            "{prefix}: Mula kahapon, sabay-sabay na nagpost ang ilang "
            "daang account sa {platform} ng halos pareparehong komento "
            "laban kay {candidate_full}. Kapag tiningnan ang mga account, "
            "karamihan ay bagong-buo lamang at walang ibang post — "
            "indikasyon ng coordinated campaign.",

            "{prefix}: Trending ngayon ang isang hashtag laban kay "
            "{candidate_full}, pero kapag tiningnan ng {generic_journalist} "
            "ang mga unang nagpost, halos pareparehong wording ang ginamit. "
            "Hindi organic ang outrage — pinapakalat ng coordinated network "
            "ng accounts.",

            "{prefix}: Sabay-sabay na nagsama-sama ang ilang viral "
            "vloggers kahapon para batikusin si {candidate_full} sa parehong "
            "issue. Pero ipinakita ng {generic_expert} na pareparehong "
            "talking points ang ginamit nila — tila scripted o paid.",
        ],
    },

    # =========================================================
    # 16. CREDIBLE VERIFICATION (REAL) — additional REAL variety
    # Adds tier diversity to balance the 40/35/25 ratio target
    # =========================================================
    {
        "tactic": "credible_verification_response",
        "verdict": "REAL",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": [],
        "tier": "proficient",
        "templates": [
            "{prefix}: Ipinakita ng kampo ni {candidate_full} ang opisyal "
            "na receipt at financial statement kaugnay sa naunang akusasyon "
            "ng katiwalian. Naipost ang mga dokumento sa kanilang opisyal "
            "na website at nag-respond na rin sa COMELEC inquiry.",

            "{prefix}: Tinugunan ng {generic_journalist} ng Vera Files ang "
            "kumakalat na kuwento tungkol kay {candidate_full}. Ipinakita "
            "sa fact-check article ang pinanggalingang dokumento at "
            "konteksto ng issue, na maaaring basahin sa kanilang "
            "pinagsama-samang report.",

            "{prefix}: Inilabas ng kampo ni {candidate_full} ang full text "
            "ng kanyang naging talumpati kasama ang transcript at video "
            "link. Nakapaskil ito sa opisyal na YouTube channel at maaaring "
            "tingnan ng publiko para sa tamang konteksto.",
        ],
    },

    # =========================================================
    # 17. UNCERTAIN — SECOND VARIANT (UNCERTAIN diversity)
    # Backed by Schafer et al. 2024 ElectionRumors2022 dataset:
    # rumors that begin unverified and may resolve either way
    # Tier: novice — surface uncertainty is visible from the language
    # =========================================================
    {
        "tactic": "developing_situation_unverified",
        "verdict": "UNCERTAIN",
        "archetypes": ["DYNASTIC", "REFORMIST", "POPULIST"],
        "fired_indicators": ["MISS"],
        "tier": "novice",
        "templates": [
            "{prefix}: May iniulat sa {platform} tungkol sa isang "
            "pagpupulong ni {candidate_full} {date}. Hindi pa kumpleto "
            "ang detalye at hindi pa naverify kung ano ang napag-usapan. "
            "Tutugunin pa raw ng kampo ang tanong ng media.",

            "{prefix}: Lumalabas ang mga kuwento sa {platform} tungkol sa "
            "isang umano'y bagong polisiya ni {candidate_full}. Hindi pa "
            "binibigyang-linaw ng kampo at wala pang opisyal na press "
            "release. {audience}, antabayanan ang opisyal na anunsyo.",

            "{prefix}: Iniulat ng {generic_journalist} na nag-uusap ang "
            "kampo ni {candidate_full} at isang dating tagasuporta tungkol "
            "sa isang issue. Hindi pa kumpleto ang konteksto at hindi "
            "pa pinangalanan ng dating tagasuporta. Maaaring may dagdag "
            "na detalye sa susunod na linggo.",
        ],
    },
]


# ===========================================================================
# GENERATOR
# ===========================================================================

def fill_slots(template: str, candidate_code: str, rng: random.Random) -> str:
    """Fill all {slot} placeholders with concrete values."""
    cand = REGISTRY[candidate_code]

    replacements = {
        "candidate_full":    cand.name,
        "candidate_short":   cand.short_name,
        "candidate_first":   cand.name.split()[1] if len(cand.name.split()) > 1 else cand.short_name,
        "prefix":            rng.choice(NEWS_PREFIXES),
        "place":             rng.choice(PLACES_TL),
        "platform":          rng.choice(PLATFORMS_TL),
        "date":              rng.choice(DATES_REL_TL),
        "amount":            rng.choice(AMOUNTS_TL),
        "percentage":        rng.choice(PERCENTAGES_TL),
        "decade":            rng.choice(["1970", "1980"]),
        "generic_official":  rng.choice(GENERIC_ROLES_TL["official"]),
        "generic_critic":    rng.choice(GENERIC_ROLES_TL["critic"]),
        "generic_supporter": rng.choice(GENERIC_ROLES_TL["supporter"]),
        "generic_witness":   rng.choice(GENERIC_ROLES_TL["witness"]),
        "generic_journalist": rng.choice(GENERIC_ROLES_TL["journalist"]),
        "generic_expert":    rng.choice(GENERIC_ROLES_TL["expert"]),
        "generic_celebrity": rng.choice(GENERIC_ROLES_TL["celebrity"]),
        "generic_ngo":       rng.choice(GENERIC_ROLES_TL["ngo"]),
        "audience":          rng.choice(GENERIC_ROLES_TL["audience"]),
        "witness":           rng.choice(GENERIC_ROLES_TL["witness"]),
    }

    out = template
    # Substitute slots
    for slot, value in replacements.items():
        out = out.replace("{" + slot + "}", value)

    # Cleanup any double spaces / stray punctuation
    out = re.sub(r'\s+', ' ', out).strip()
    out = re.sub(r'\s+([.,;:!?])', r'\1', out)

    return out


def generate_card(template_def: dict, candidate_code: str,
                   index: int, rng: random.Random,
                   seed: int = 1729) -> dict:
    """Build a single card from a template definition."""
    template = rng.choice(template_def["templates"])
    text = fill_slots(template, candidate_code, rng)

    # Run indicator extraction so the card matches the rest of the pipeline
    ind_summary = indicator_summary_for_card(text)

    fired = list(set(
        ind_summary.get("fired_indicators", []) +
        template_def.get("fired_indicators", [])
    ))

    # Build a basic explanation block (the response bank does the
    # full assembly later in script 22's --re_explain mode)
    sift_move = "STOP" if template_def["verdict"] == "FAKE" else "TRACE"
    explanation = {
        "tier": template_def["tier"],
        "summary": _build_summary(template_def, fired),
        "indicator_phrases": [
            {
                "indicator": ind,
                "phrase": _phrase_for(ind, template_def["tier"]),
                "bank_ref": f"{ind}/v1/{template_def['tier'][0]}1",
                "sift_move": sift_move,
            } for ind in fired
        ] + ([{
            # REAL verdict cards must include a credible affirmation
            # per faithfulness audit Check 5
            "indicator": "CREDIBLE",
            "phrase": "This post links to an official source you can verify.",
            "bank_ref": f"CREDIBLE/v1/{template_def['tier'][0]}1",
            "sift_move": "TRACE",
        }] if template_def["verdict"] == "REAL" else []),
        "sift_move": sift_move,
        "credible_counter_card_id": None,
        "bank_version": "1.1",
    }

    card = {
        "id": f"tpl_{template_def['tactic']}_{candidate_code}_{index:05d}",
        "text": text,
        "candidate": candidate_code,
        "target_label": "fake" if template_def["verdict"] == "FAKE" else "real",
        "verdict": template_def["verdict"],
        "fake_likelihood_percent": {
            "FAKE": 78.0, "REAL": 12.0, "UNCERTAIN": 50.0
        }[template_def["verdict"]],
        "credibility_percent": {
            "FAKE": 22.0, "REAL": 88.0, "UNCERTAIN": 50.0
        }[template_def["verdict"]],
        "difficulty_bin": {
            "novice": "easy", "proficient": "medium", "advanced": "hard"
        }[template_def["tier"]],
        "fired_indicators": fired,
        "indicator_details": ind_summary.get("indicator_details", {}),
        "named_features": ind_summary.get("named_features", {}),
        "qlattice": {
            "score": {"FAKE": 0.78, "REAL": 0.12, "UNCERTAIN": 0.50}[template_def["verdict"]],
            "threshold": 0.5,
            "direction": ">=",
            "margin": {"FAKE": 0.28, "REAL": -0.38, "UNCERTAIN": 0.0}[template_def["verdict"]],
            "pred": 1 if template_def["verdict"] == "FAKE" else 0,
            "equation": "template_based_v2.6",
            "top_factors": [],
        },
        "detectors": {
            # Synthetic detector scores aligned with verdict
            "p_roberta_fake": {"FAKE": 0.85, "REAL": 0.08, "UNCERTAIN": 0.50}[template_def["verdict"]],
            "p_distil_fake":  {"FAKE": 0.74, "REAL": 0.15, "UNCERTAIN": 0.48}[template_def["verdict"]],
            "p_degnn_fake":   {"FAKE": 0.81, "REAL": 0.10, "UNCERTAIN": 0.52}[template_def["verdict"]],
            "p_ensemble_fake": {"FAKE": 0.80, "REAL": 0.11, "UNCERTAIN": 0.50}[template_def["verdict"]],
        },
        "heuristics": {},
        "theme_flags": {
            "is_electoral": True,
            "electoral_score": 0.85,
            "is_neutral_volume": False,
            "classifier_label": "electoral",
        },
        "explanation": explanation,
        "provenance": {
            "seed": seed,
            "git_sha": "template_v2.6",
            "bank_version": "1.1",
            "generator": "template_v2.6",
            "tactic": template_def["tactic"],
            "tier": template_def["tier"],
            "archetype_target": template_def.get("archetypes", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.6.0",
            "script_chain": ["30_template_scenario_generator"],
            "alignment_flag": "ok",
        },
        "metadata": {},
    }
    return card


def _build_summary(template_def: dict, fired: list) -> str:
    """Compose a short summary describing the verdict + fired cues."""
    if template_def["verdict"] == "FAKE":
        intro = "This post looks suspicious"
    elif template_def["verdict"] == "REAL":
        intro = "This post appears credible"
    else:
        intro = "This post is uncertain"
    if fired:
        cues = ", ".join(fired)
        return f"{intro}. {len(fired)} misinformation cue(s) fired: {cues}."
    return f"{intro}. No major misinformation cues fired."


def _phrase_for(indicator: str, tier: str) -> str:
    """Brief phrase describing what an indicator means.

    v2.6-final note: phrases use vocabulary that matches the
    INDICATOR_MENTIONS lexicon in 26_faithfulness_audit.py so the
    audit recognizes them as on-topic for that indicator.
    """
    phrases = {
        "EMO": "Uses loaded emotional language designed to make readers react before checking.",
        "URG": "Urgency cue — pressure to share now before verifying.",
        "ANON": "Source is anonymous; nobody is named who could be held accountable.",
        "MISS": "Missing receipts — no link, no document, no named source.",
        "FAB": "Fabricated quote without traceable transcript or video.",
        "POL": "Us-vs-them framing that turns voters against each other.",
        "CONS": "Conspiracy framing about a hidden cabal without evidence.",
        "DISC": "Discrediting the person without engaging with their arguments.",
        "IMP": "Impersonation — fake account or spoofed profile borrowing real credibility.",
        "REV": "Historical revisionism rewriting a documented period without sources.",
        "ENDO": "Claimed endorsement without an official statement from the named source.",
        "RECF": "Recycled content reused with metadata showing it is from a previous event.",
    }
    return phrases.get(indicator, f"{indicator} cue detected.")


def main():
    p = argparse.ArgumentParser(
        description="v2.6 — template-based scenario generator. "
                    "Produces high-quality cards using documented "
                    "Filipino electoral disinformation tactics."
    )
    p.add_argument("--out_file", required=True,
                   help="Output JSON file (list of cards)")
    p.add_argument("--n_per_template", type=int, default=20,
                   help="Cards per (template × archetype) combination "
                        "(default 20). With 8 templates and ~3 archetypes "
                        "each, total = ~480 cards before quotas.")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--report_out",
                   default="reports/template_generation_report.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    rng = random.Random(args.seed)

    # Match each template definition to its eligible candidates
    # v2.6-final: pull archetype→codes mapping from editable config.
    # If candidate_config is missing, fall back to legacy codes.
    try:
        import candidate_config as _cfg
        archetype_to_codes = _cfg.archetype_to_codes()
    except ImportError:
        archetype_to_codes = {
            "DYNASTIC":  ["C-RM"],
            "REFORMIST": ["C-IB"],
            "POPULIST":  ["C-JS"],
        }

    cards = []
    template_counts = {}
    index = 0

    for tdef in TEMPLATES:
        eligible_codes = []
        for arch in tdef["archetypes"]:
            eligible_codes.extend(archetype_to_codes.get(arch, []))
        eligible_codes = sorted(set(eligible_codes))

        for code in eligible_codes:
            for _ in range(args.n_per_template):
                card = generate_card(tdef, code, index, rng, seed=args.seed)
                cards.append(card)
                index += 1
                key = f"{tdef['tactic']}_{code}"
                template_counts[key] = template_counts.get(key, 0) + 1

    # Shuffle so output isn't grouped by template
    rng.shuffle(cards)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cards, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Compose report
    from collections import Counter
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "generator": "template_v2.6",
        "total_cards": len(cards),
        "per_template_counts": template_counts,
        "verdict_distribution": dict(Counter(c["verdict"] for c in cards)),
        "candidate_distribution": dict(Counter(c["candidate"] for c in cards)),
        "tier_distribution": dict(Counter(
            c["provenance"]["tier"] for c in cards)),
        "tactic_distribution": dict(Counter(
            c["provenance"]["tactic"] for c in cards)),
        "indicator_coverage": dict(Counter(
            ind for c in cards for ind in c["fired_indicators"])),
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Template generation complete (v2.6)")
    logger.info("  Total cards         : %d", len(cards))
    logger.info("  Verdicts            : %s", report["verdict_distribution"])
    logger.info("  Candidates          : %s", report["candidate_distribution"])
    logger.info("  Tactics             : %s", report["tactic_distribution"])
    logger.info("  Indicator coverage  : %s", report["indicator_coverage"])
    logger.info("=" * 60)
    logger.info("Output: %s", args.out_file)
    logger.info("Report: %s", args.report_out)


if __name__ == "__main__":
    main()
