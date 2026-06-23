import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import ranking_engine
from ranking_engine import initialize_skills, initialize_embeddings, rank_candidates
from explanation_engine import explain_all

candidates = []      # holds all candidates in memory
is_ready   = False   # flag: True once initialization is complete

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs when the server starts up.
    We load the full candidate database and build the FAISS index here.
    After this, every /rank request is fast because the index is ready.
    """
    global candidates, is_ready

    print("Server starting up — loading candidates...")
    t0 = time.time()

    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    initialize_skills(candidates)
    initialize_embeddings(candidates)

    is_ready = True
    print("Server ready!")

    yield   

app = FastAPI(
    title="AI-Recruiter API",
    description="Ranks candidates for a given job description",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_methods=["*"],
    allow_headers=["*"],
)

class RankRequest(BaseModel):
    jd: str                          
    top_n: Optional[int] = 100       


class ScoreBreakdown(BaseModel):
    semantic:      float
    skill:         float
    career:        float
    role:          float
    experience:    float
    availability:  float
    education:     float
    certification: float


class CandidateResult(BaseModel):
    candidate_id:    str
    rank:            int
    score:           float
    reasoning:       str
    headline:        str
    years_experience: int
    matched_skills:  list[str]
    missing_skills:  list[str]
    bonus_skills:    list[str]
    availability:    str
    score_breakdown: dict
    open_to_work:    bool
    notice_days:     int
    response_rate:   float


class RankResponse(BaseModel):
    results:        list[CandidateResult]
    total_ranked:   int
    elapsed_seconds: float

@app.get("/health")
def health_check():
    """
    Simple check: is the server ready to accept ranking requests?
    The frontend should poll this on load and show a "loading" state
    until is_ready = True.
    """
    return {
        "status":      "ready" if is_ready else "initializing",
        "candidates":  len(candidates),
        "faiss_vectors": ranking_engine.faiss_index.ntotal if ranking_engine.faiss_index else 0,
    }


@app.post("/rank", response_model=RankResponse)
def rank_endpoint(request: RankRequest):
    """
    Main endpoint: given a JD, return the top N ranked candidates.

    Request body:
    {
        "jd": "Senior AI Engineer... (full JD text)",
        "top_n": 100
    }

    Response:
    {
        "results": [ { candidate data... }, ... ],
        "total_ranked": 100,
        "elapsed_seconds": 4.2
    }
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="Server is still initializing. Try again in a moment.")

    if not request.jd or len(request.jd.strip()) < 50:
        raise HTTPException(status_code=400, detail="JD is too short. Please provide a full job description.")

    top_n = min(max(request.top_n, 10), 100)   

    t0 = time.time()
    ranked_results = rank_candidates(request.jd, candidates, top_n=top_n)
    explanations   = explain_all(ranked_results, request.jd)
    elapsed = time.time() - t0

    return {
        "results":         explanations,
        "total_ranked":    len(explanations),
        "elapsed_seconds": round(elapsed, 2),
    }


@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: str):
    """
    Return the full profile for a single candidate.
    The frontend calls this when the user clicks "View Profile".
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="Server is still initializing.")

    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    candidate = candidate_lookup.get(candidate_id)

    if not candidate:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found.")

    return candidate

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)