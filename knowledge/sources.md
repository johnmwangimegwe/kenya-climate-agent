# Knowledge Base Sources

This file is the citation register for every document in `knowledge/`. It does
two jobs: it tells anyone reading the repo where the agent's grounding comes
from, and it provides the legal/licensing record so the project only
redistributes documents it is allowed to.

## How to cite each document (format)

For every file you drop into `knowledge/pdfs/` or `knowledge/notes/`, add one
entry below using this template:

```
### <short id>  —  <filename in knowledge/>
- Title: <full document title>
- Author / Publisher: <author(s) or issuing organization>
- Year: <publication year>
- Type: <government report | journal article | NGO assessment | dataset notes>
- URL: <direct link to the source>
- License: <e.g. CC-BY 4.0 | Open Government data | © publisher — link only>
- Redistributable in this repo?: <yes / no — if no, cite + link only>
- Used for: <what the agent draws from it, e.g. "county drought phases">
```

**Rule of thumb on the last two fields:** government reports (NDMA, Kenya Met),
UN/OCHA, ACAPS, and CC-BY journal articles are generally safe to commit. For
anything paywalled or "all rights reserved" (e.g. Wiley, ScienceDirect
abstracts), set *Redistributable* to **no**, keep only this citation + URL, and
do **not** put the PDF in the repo.

---

## Recommended documents to download and add

These are real, currently-available Kenyan climate-risk sources. Download the
open ones into `knowledge/pdfs/`; for the closed ones, keep the citation only.

### ndma-bulletin  :  ndma_drought_bulletin.pdf
- Title: National Drought Early Warning Bulletin (monthly; e.g. February 2026)
- Author / Publisher: National Drought Management Authority (NDMA), Kenya
- Year: 2026
- Type: government report
- URL: https://www.ndma.go.ke/drought-information/  (monthly bulletins)
- License: Kenya Government / open public information
- Redistributable in this repo?: yes (government public information)
- Used for: per-county drought phase (Normal/Alert/Alarm/Emergency), rainfall
  and vegetation conditions, food-insecurity figures

### ndma-monthly-update  :  ndma_national_update_jan2026.pdf
- Title: National Monthly Drought Update, January 2026
- Author / Publisher: National Drought Management Authority (NDMA), Kenya
- Year: 2026
- Type: government report
- URL: https://www.ndma.go.ke/drought-situation-update-2/
- License: Kenya Government / open public information
- Redistributable in this repo?: yes
- Used for: which ASAL counties are worsening; affected-population counts

### acaps-floods  :  acaps_kenya_floods_briefing.pdf
- Title: Kenya — Heavy Rainfall and Floods (Briefing Note)
- Author / Publisher: ACAPS
- Year: 2024
- Type: NGO assessment
- URL: https://www.acaps.org/fileadmin/Data_Product/Main_media/20240514_ACAPS_Briefing_note_Kenya_Floods.pdf
- License: ACAPS — free to use with attribution (check current terms)
- Redistributable in this repo?: yes (attribution)
- Used for: Tana River flood warnings; Garissa/Lamu/Tana River exposure;
  cholera and waterborne-disease links to flooding

### odoyo-tana-flood  :  flood_vulnerability_study.pdf
- Title: Assessment of Flood Vulnerability and Coping Strategies of Communities
  Living along River Tana in Madogo Ward, Kenya
- Author / Publisher: Odoyo, E.; Huho, J. M.; Mohamed, A. M.; Mbugua, J. M. —
  American Journal of Climatic Studies, 4(1), 1–17
- Year: 2024
- Type: journal article
- URL: https://doi.org/10.47672/ajcs.1931
- License: CC-BY 4.0 (open access — redistribution permitted with attribution)
- Redistributable in this repo?: yes (CC-BY 4.0)
- Used for: flood recurrence (~every 2 years in Madogo), topography as the main
  vulnerability factor, community coping strategies

### mugatha-tana-inundation  :  tana_delta_inundation_mapping.pdf
- Title: Flood Inundation Mapping for the Tana River Delta in Kenya
- Author / Publisher: Mugatha, A. N. et al. — Tanzania Journal of Engineering
  and Technology, 44(1)
- Year: 2025
- Type: journal article
- URL: https://doi.org/10.52339/tjet.v44i1.981
  (PDF: https://www.ajol.info/index.php/tjet/article/download/292972/275730)
- License: AJOL / open access (verify the article's stated license)
- Redistributable in this repo?: yes if open access — otherwise cite + link
- Used for: DEM-based inundation extent, gauging stations (Garissa, Hola,
  Garsen), flood quantiles in the Tana delta

### okoko-flood-governance
- Title: Diversified flood governance and related socio-spatial vulnerability in
  Tana River County, Kenya
- Author / Publisher: Okoko — Disasters (Wiley)
- Year: 2024
- Type: journal article
- URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/disa.12648
- License: © Wiley — all rights reserved
- Redistributable in this repo?: NO — citation and link only
- Used for: socio-spatial vulnerability differentiated by age, gender,
  disability (supports the fairness framing — paraphrase, don't copy)

---

## Notes
- Always prefer the **county-level** government bulletins for the live numbers —
  they are the most authoritative and the safest to redistribute.
- When you add a PDF, re-run `build_index.py` so the new document is embedded
  into the retrieval index.
- Keep this file in sync with the actual contents of `knowledge/` — every file
  present should have an entry here, and every "cite only" entry should NOT have
  a file in the repo.
