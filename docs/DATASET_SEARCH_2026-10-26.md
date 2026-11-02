# Dataset search 2026-10-26 (open data similar to the BAR-ATH Metrica data)

Method: web search + fetch of primary pages. "V" = page fetched and content confirmed; "U" = unverified (search snippet / secondary source only).
Excluded (already held): Metrica sample Games 1-3, IDSSE 7 matches, SkillCorner open data, PFF WC2022.

## Headline finding
No additional Metrica-produced EPTS tracking or Smart Tagging exports are publicly available beyond the three sample games (V: github.com/metrica-sports/sample-data lists only Games 1-3; Game 3 is EPTS FIFA + JSON events; "be responsible", credit the source, no formal licence). No public "More than a Hack" dataset from earlier editions was found. The only route to more Metrica-style data is asking the organisers / Metrica directly (see recommendations).

## Ranked table (best similarity first)

| # | Dataset | URL | Licence / terms | Format | Matches / fps | Ball | Events / possession | Tracking nature | Access | Sim. (1-5) | Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Metrica private/academic data via organisers (ask) | contact MTAH2027 organisers / Metrica Sports | unknown | EPTS + Smart Tagging | unknown | yes | yes | identical pipeline | email request only | 5 | U (no evidence it exists; worth one email) |
| 2 | SoccerTrack v2 (BePro wide-angle 4K) | https://huggingface.co/datasets/atomscott/soccertrack-v2 ; https://github.com/AtomScott/SoccerTrack-v2 | CC BY 4.0 (data), MIT (code); gated by accepting conditions; pseudonymised | video + GSR (2D pitch coords, IDs, roles, teams) + BAS 12 action classes + `ball/` auxiliary ball tracks | 10 full amateur/university matches, 20 half videos, 145 GB (v1.2) | auxiliary ball tracks, partly | BAS events (Pass, Shot, ...) -> possession derivable from passes; no dead-ball labels | manual/curated annotation, not automatic occlusion; amateur level | HF login + accept conditions (user must do) | 3 (best for a ball detector on a tactical/wide camera, like our tactical video; weak for possession semantics) | V |
| 3 | SoccerNet-GAR "Pixels or Positions" (WC2022 tracking + broadcast) | https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR | "other": NDA, research/non-commercial, no sharing | 2D players + 3D ball, 30 fps, from Gradient/PFF, auto-extracted then refined | all 64 WC2022 matches (same underlying data as the PFF set we already hold) | ball in 93.4% frames | group-activity events (87,939), not possession | broadcast-derived, 6.6% ball missing | HF request + sign personalised NDA (user must do) | 2 (little new beyond PFF; NDA restricts) | V |
| 4 | SoccerNet-Tracking (MOT) | https://github.com/SoccerNet/sn-tracking | not stated in repo; SoccerNet data usually research-only/NDA-gated for video (U) | video + bboxes + tracklet IDs (players, GK, referees, ball) | 12 full games, 200 x 30 s clips, + 1 annotated half | ball boxes included | no events | broadcast, manual annotation | `pip install SoccerNet` + SoccerNetDownloader (video password/NDA likely, U) | 2 (ball detector training from broadcast; no possession) | V (repo), U (licence) |
| 5 | SoccerNet Game State Reconstruction (GSR 2024) | https://github.com/SoccerNet/sn-gamestate | code GPL-3.0; dataset licence not stated | 2D pitch positions, role, jersey, team side; v1.3 | 200 x 30 s clips | NO ball annotation | no | manual | SoccerNetDownloader task `gamestate-2024` | 1-2 | V |
| 6 | Hammarby (Signality) | listed in https://github.com/metrica-sports/sample-data (link to David Sumpter course repo) | unknown | tracking + video (unsynced) | 3 games | unclear | no events | Signality camera tracking (automatic) | direct per listing | 3 (automatic tracking, but no events -> no labels) | V (listing only; files not inspected) |
| 7 | Last Row (Liverpool 2019) | https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row (datasets/positional_data/liverpool_2019.csv confirmed via a third-party notebook) | unknown | CSV positions | 19 goals / short clips | players+ball | no | automatic | direct | 1 (clips around goals only) | U (my direct repo fetch 404'd with the other org name) |
| 8 | Sportec Open DFL (Bassek et al., Sci Data) = our IDSSE | figshare, Scientific Data 2025 | CC BY | | 7 matches | | | | | already held; no newer release found | V (via secondary papers) |
| 9 | FOOTPASS (play-by-play action spotting) | https://arxiv.org/abs/2511.16183 | not stated | | not stated | | | | | unknown; data source/availability not found in abstract | U |
| 10 | StatsBomb 360 open data | https://github.com/statsbomb/open-data | CC-style, attribution | freeze frames, events only | several comps | no tracking | events | n/a | direct | 1 (no continuous tracking) | U (not fetched) |
| 11 | Gradient Sports/PFF newer seasons | https://fc.pff.com | commercial (paid products) | | | | | | sales contact | 2 if obtainable; only WC2022 is free | V (WC2022 page; form content not visible) |

Notes on items we already hold (corrections): the SkillCorner repo (https://github.com/SkillCorner/opendata, MIT) currently lists 10 A-League 2024/25 matches (not 20) with tracking, dynamic events, phases of play, and body pose for 2 matches; recheck local copy count. PFF WC2022 page (https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) gives access via a form; terms not shown on the page.

Not found at all: Second Spectrum / Hawk-Eye / Sportlogiq open releases, Wyscout+tracking, La Liga / Barcelona / Athletic-specific open tracking, any earlier hackathon dataset from Metrica/FCB BIHUB. Search tooling returned nothing for these (absence of evidence, not proof).

## Recommendation: top 3
1. **Email the organisers / Metrica (similarity 5, if it exists).** Ask whether any other Metrica tactical-camera tracking + Smart Tagging from past editions or other matches (even a few minutes) is permitted for training, and whether dead-ball/possession labels exist. The user must send this; costs nothing.
2. **SoccerTrack v2 (similarity 3, for a ball detector).** Only real wide-camera video + ball tracks + pass events with a permissive CC BY 4.0 licence. User must: log into Hugging Face, accept the conditions on https://huggingface.co/datasets/atomscott/soccertrack-v2, then download `ball/`, `bas/`, `gsr/` first (small) before the 145 GB of video; cite arXiv 2508.01802. Useful mainly for Wave 3 (#11/#12) and for deriving possession from BAS passes at amateur level; domain gap (amateur, different camera).
3. **Hammarby/Signality 3 games (similarity 3, automatic camera tracking).** Closest in nature to automatic-camera tracking with realistic occlusions, but has no events, so labels would have to be derived heuristically (nearest-player-to-ball), which is weak for D. Check the link in the Metrica sample-data README (https://github.com/metrica-sports/sample-data) and licence before use. Fall-back: degrade IDSSE/Metrica with our #19 degrader, which already models the 41% ball gap.

Not worth pursuing: SoccerNet-GAR (same underlying PFF data, NDA), SoccerNet GSR (no ball), StatsBomb 360 (no tracking).

Nothing was registered, accepted or downloaded by this search.
