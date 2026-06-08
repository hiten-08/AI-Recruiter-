# AI-Recruiter — Intelligent Candidate Ranking System

> Hackathon submission for the **Intelligent Candidate Discovery Challenge**
> Built by a team of 4 · ML Lead · Backend · Frontend (×2)

---

## What it does

AI-Recruiter takes a job description and ranks the top 100 best-fit candidates from a pool of 100,000+ using a multi-signal scoring engine — not keyword matching.

Most recruiter tools rank by "how many skill keywords match." We don't. We analyze career history, behavioral engagement signals, role trajectory, and semantic similarity to find candidates who have **actually done the work**, not just listed the buzzwords.

---

## Architecture

```
Job Description (text)
        │
        ▼
┌─────────────────────┐
│   JD Parser         │  Extracts required skills, experience range
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FAISS Retrieval   │  Fast semantic search — narrows 100K → 2000
│   (Stage 1)         │  all-MiniLM-L6-v2 embeddings + IndexFlatIP
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Honeypot Filter    │  Removes fake/impossible candidate profiles
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  8-Signal Scorer    │  Re-ranks 2000 → top 100
│  (Stage 2)          │  Career · Skill · Semantic · Role ·
│                     │  Experience · Availability · Education · Cert
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Explanation Engine │  Generates specific, grounded reasoning
│                     │  for each candidate (no hallucination)
└────────┬────────────┘
         │
         ▼
    submission.csv  +  frontend_results.json
```

---

## Scoring breakdown

| Signal | Weight | Why |
|---|---|---|
| Career history | **25%** | Did they actually build ML/AI systems? |
| Skill match | 18% | Weighted by criticality, not just count |
| Semantic similarity | 20% | Full profile vs. JD embedding match |
| Role alignment | 12% | Are their job titles ML-relevant? |
| Experience range | 10% | Right years, right type |
| Availability | 8% | Active? Open to work? Notice period? |
| Education | 4% | Degree, field, institution tier |
| Certifications | 3% | Relevant certs only |

> **Why career history is weighted highest:** The hackathon spec explicitly warns that the right answer is NOT finding candidates whose skills section contains the most AI keywords. A candidate who built a recommendation engine at a startup beats an HR Manager with "RAG" listed in skills.

---

## Honeypot detection

The dataset contains ~80 fake candidates designed to fool keyword-based systems. We detect them by checking:

- Skill list longer than 30 items (real people don't have 35 skills)
- Claims "Advanced" proficiency in 8+ skills with 0 years used
- Non-tech headline (HR Manager, Content Writer, etc.) with zero ML/AI career history

More than 10 honeypots in your top 100 = disqualification at Stage 3.

---

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector search | `faiss-cpu` — `IndexFlatIP` |
| Scoring | Pure Python + NumPy |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project structure

```
ai-recruiter/
├── config.py               # All constants, weights, keyword lists
├── ranking_engine.py       # FAISS retrieval + 8-signal scoring
├── explanation_engine.py   # Per-candidate reasoning generation
├── test_pipeline.py        # End-to-end run → submission.csv
├── api.py                  # FastAPI backend (3 endpoints)
├── frontend/
│   ├── index.html          # Landing page
│   ├── submit.html         # JD submission form
│   ├── results.html        # Ranked results view
│   ├── style.css           # Shared styles
│   └── app.js              # API integration + shared utils
├── requirements.txt
└── README.md
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add the dataset**
```bash
# Place the hackathon dataset in the project root
cp /path/to/candidates.jsonl .
```

**3. Build the cache (first run only — takes ~5 minutes)**
```bash
python test_pipeline.py
# After this, cache/ is created and all future runs load in seconds
```

**4. Run the backend API**
```bash
uvicorn api:app --reload --port 8000
```

**5. Open the frontend**
```bash
# Just open frontend/index.html in a browser
# Make sure the API is running on port 8000
```

---

## API reference

### `POST /rank`
Submit a JD and get back 100 ranked candidates.

**Request**
```json
{
  "jd": "Senior AI Engineer... (full JD text)",
  "top_n": 100
}
```

**Response**
```json
{
  "results": [
    {
      "candidate_id": "CAND_0042871",
      "rank": 1,
      "score": 0.847,
      "reasoning": "Has hands-on production experience as ML Engineer at Flipkart; shows advanced proficiency in RAG, FAISS. Availability: actively looking, 30-day notice.",
      "score_breakdown": {
        "semantic": 0.71,
        "skill": 0.68,
        "career": 0.82,
        "role": 0.90,
        "experience": 1.0,
        "availability": 0.75,
        "education": 0.80,
        "certification": 0.40
      }
    }
  ],
  "total_ranked": 100,
  "elapsed_seconds": 3.8
}
```

### `GET /candidate/:id`
Get the full profile for a single candidate.

### `GET /health`
Check if the server has finished loading candidates and is ready to accept requests.

---

## Output format

`submission.csv` — produced by `test_pipeline.py`

| Column | Description |
|---|---|
| `candidate_id` | Unique candidate identifier |
| `rank` | 1–100, no duplicates |
| `score` | Float in [0.0, 1.0], monotonically non-increasing |
| `reasoning` | 1–2 sentence specific explanation |

---

## Submission checklist

Before uploading `submission.csv`, `test_pipeline.py` runs these checks automatically:

- [ ] Exactly 100 rows
- [ ] Ranks 1–100 with no duplicates
- [ ] Scores non-increasing (rank 1 = highest score)
- [ ] No empty reasoning strings
- [ ] All scores in [0.0, 1.0]
- [ ] No duplicate candidate IDs

---

## Team

| Role | Responsibility |
|---|---|
| ML Lead | `ranking_engine.py`, `explanation_engine.py`, `config.py` |
| Backend | `api.py`, FastAPI server, deployment |
| Frontend (×2) | `index.html`, `submit.html`, `results.html`, `app.js` |

---

## Key design decisions

**Why two-stage retrieval?**
Running 8 scoring functions on all 100K candidates would take hours. FAISS narrows the pool to 2000 in milliseconds, then we do expensive scoring on just those. This fits the 5-minute CPU budget easily.

**Why is career history weighted at 25%?**
The JD explicitly warned that keyword-stuffers will score high on naive skill-match systems. The most reliable signal of future AI work is past AI work — which lives in career history descriptions, not the skills section.

**Why are weights tunable in `config.py`?**
Different JDs need different weights. A data science role should weight education higher. A startup role should weight availability higher. Having all weights in one file makes tuning easy without touching logic code.
