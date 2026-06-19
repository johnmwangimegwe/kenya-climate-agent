# **🌍 Reasoning Over Earth Engine: An AI Agent for Kenya's Climate Risk**

Ask a question : *"Which counties in Kenya face the highest
flood risk this season, and who is most exposed?"* , and get a grounded, ranked,
explainable answer drawn from satellite data, population exposure, and
authoritative Kenyan documents.

This project is an **open replica of the Gemini-powered Geospatial Reasoning
pattern announced at Google I/O 2026**. It runs on **Gemini 3.5 Flash** and the
**free Earth Engine tier** , it is *not* Google's gated Geospatial Reasoning
agent, but it demonstrates the same architecture, end to end, on Kenyan data.

---

## **Overview**

A natural-language question is decomposed by a Gemini planner into a tool plan.
An orchestrator runs the tools, each an expert "sub-agent", and a fusion layer
synthesizes a grounded answer:

- **Knowledge Retrieval (RAG)**: retrieves Kenyan climate documents with sources
- **Earth Engine**: flood (Sentinel-1 SAR), rainfall (CHIRPS), elevation (SRTM)
- **Population**: exposed population per county (WorldPop)
- **Risk Scoring**: a transparent, auditable weighted score (no ML)

Everything degrades gracefully: with no API key or Earth Engine auth, the agent
still runs using deterministic fallbacks, so it never hard-fails in a demo.

## **Architecture**

```
                          Question
                             │
                             ▼
                   Gemini 3.5 Flash  (planner.py)
                             │   structured JSON plan
                             ▼
                   Orchestrator  (orchestrator.py)
                             │   dispatches tools (sub-agent pattern)
        ┌────────────────────┼────────────────────┬───────────────────┐
        ▼                    ▼                    ▼                   ▼
 search_knowledge   get_earth_engine_layers  get_population_    compute_risk_
   (RAG / FAISS)     (Sentinel-1/CHIRPS/SRTM)   exposure          score
        │                    │                    │                   │
        └────────────────────┴──────────┬─────────┴───────────────────┘
                                         ▼
                          Fusion  (fusion.py, Gemini 3.5 Flash)
                                         │  ranked + explained + cited
                                         ▼
                        Answer → Visualization (viz.py) → Streamlit UI
```

## **I/O 2026 concepts used**

Gemini 3.5 Flash · agent orchestration · `AGENTS.md` / `SKILL.md` agent &
skill definitions · tool-based reasoning · sub-agent pattern · grounded
retrieval (RAG) · WebMCP-compatible tool design · Managed-Agents-style
architecture.

---

## **Project structure**

```
kenya-climate-agent/
├── src/
│   ├── agent/        planner.py · orchestrator.py · fusion.py
│   ├── tools/        earth_engine.py · population.py · risk_score.py · knowledge.py
│   ├── rag/          build_index.py · retriever.py
│   └── utils/        geo.py · viz.py
├── app/              streamlit_app.py
├── knowledge/        pdfs/ · notes/ · sources.md   (RAG knowledge base)
├── data/             kenya_counties.geojson · settlements.geojson
├── config.yaml · requirements.txt · .env.example
├── AGENTS.md · SKILL.md · demo.ipynb · LICENSE · README.md
```

## **Installation**

Requires **Python 3.11+**.

```bash
git clone <your-repo-url> kenya-climate-agent
cd kenya-climate-agent
pip install -r requirements.txt
# On Colab / Debian / Ubuntu you may need:
# pip install -r requirements.txt --break-system-packages
```

## **Setup**

Copy the environment template and add your credentials:

```bash
touch .env
```

```
GOOGLE_API_KEY=your-gemini-api-key
EE_PROJECT_ID=your-earthengine-project-id
```

Both are optional — without them the agent runs in deterministic fallback mode —
but you need them for live Gemini reasoning and real satellite data.

### **Gemini setup**

Create an API key in **Google AI Studio** (<https://aistudio.google.com/apikey>)
and put it in `.env` as `GOOGLE_API_KEY`. The planner and fusion layers use
Gemini 3.5 Flash; the RAG index uses Gemini embeddings when the key is present
(and falls back to local TF-IDF when it is not).

### **Earth Engine authentication**

1. Sign up for Earth Engine and create/enable a Cloud project:
   <https://code.earthengine.google.com/>
2. Put the project id in `.env` as `EE_PROJECT_ID`.
3. Authenticate once:

```bash
earthengine authenticate
```

In Colab, run `import ee; ee.Authenticate()` in the first cell instead (see the
notebook).

### **County boundaries**

`data/kenya_counties.geojson` (all 47 counties) ships with the repo as a Voronoi
approximation — good enough for choropleths and region clipping. For
publication-grade boundaries, replace it with an official **GADM** or **HDX**
Kenya counties file (keep a `COUNTY` or `shapeName` property); everything else
works unchanged.

## **Running the Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL. Type a question (or pick an example), click **Run
analysis**, and you'll get the answer, a ranked county table, an interactive
map, a risk bar chart, and an expandable execution log. The sidebar shows your
environment status (API key / Earth Engine / knowledge index).

## **Running the notebook**

Open `demo.ipynb` locally or in Colab and run the cells top to bottom:

1. Install dependencies
2. Authenticate Earth Engine
3. Build the RAG index
4. Run the agent on a question
5. Display the ranked results
6. Visualize the risk map

The notebook is a thin runner — all logic lives in `src/`, so the notebook just
imports and calls it.

## **Demo screenshots**

> _Add screenshots here before presenting._

- `docs/screenshot_app.png` — Streamlit app with answer + map _(placeholder)_
- `docs/screenshot_map.png` — county risk choropleth _(placeholder)_
- `docs/screenshot_log.png` — execution log / plan _(placeholder)_

## **How the risk score works**

```
risk_score = 0.4·flood + 0.3·rainfall + 0.2·population_exposure + 0.1·elevation_vulnerability
```

All four inputs are normalized to [0, 1]; the weights are normalized to sum to 1,
so the score stays in [0, 1]. It is **deterministic and explainable** — no
machine learning — and every county's score breaks down into per-factor
contributions that sum exactly to the total. Tune the weights in `config.yaml`.

## **Future improvements**

- Swap the Voronoi county boundaries for official GADM/HDX geometries.
- Add a trained, explainable risk model (e.g. GNN over counties) as an optional
  sub-agent, alongside the transparent baseline.
- Expose the tools to a browser agent via **WebMCP** so Gemini in Chrome can
  drive the analysis directly.
- Add drought- and rainfall-specific scoring profiles and seasonal forecasting.
- Cache Earth Engine results per county to speed up repeat queries.

## **Disclaimer**

This is a **decision-support** tool, not an emergency-response system. When run
without Earth Engine or a Gemini key it uses deterministic fallback values that
are **not** real measurements. Always validate against official sources such as
the Kenya Meteorological Department and the National Drought Management Authority.

## **License**

Source code: **MIT** (see `LICENSE`). Knowledge-base documents retain their own
licenses — see `knowledge/sources.md`.
