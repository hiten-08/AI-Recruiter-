import re
import os
import pickle
import numpy as np
import faiss
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config import *   

model = SentenceTransformer("all-MiniLM-L6-v2")

candidate_embeddings = {}  
candidate_ids        = []   
faiss_index          = None 
all_skills           = set()

def initialize_skills(candidates):
    """
    Walk every candidate and collect all skill names into a global set.
    We need this vocabulary to know which skills to look for in the JD.
    """
    global all_skills
    all_skills = set()
    for candidate in candidates:
        for skill in candidate.get("skills", []):
            all_skills.add(skill["name"])
    return all_skills


def build_candidate_profile_text(candidate):
    """
    Combine a candidate's headline, summary, skills, and career history
    into ONE long string.  This is what we'll embed into a vector.

    Why combine everything?
    A candidate who "built a recommendation engine at Flipkart" should match
    a JD about ranking systems even if they don't have "FAISS" in their skills.
    The semantic model can only work with text, so we give it all the text.
    """
    parts = []

    profile = candidate.get("profile", {})
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    # Skills
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    # Career history — titles AND descriptions
    for job in candidate.get("career_history", []):
        if job.get("title"):
            parts.append(job["title"])
        if job.get("description"):
            parts.append(job["description"])

    return " ".join(parts)


def build_candidate_embeddings(candidates):
    """
    Convert every candidate's profile text into a vector.
    This takes a while for 100K candidates but we cache the result.
    """
    embeddings = {}
    total = len(candidates)
    for i, candidate in enumerate(candidates):
        if i % 1000 == 0:
            print(f"  Embedding candidate {i}/{total}...")
        cid = candidate["candidate_id"]
        text = build_candidate_profile_text(candidate)
        embeddings[cid] = model.encode(text)
    return embeddings


def build_faiss_index(candidates):
    """
    FAISS is a library from Meta that lets you find "nearest neighbors"
    in a list of vectors extremely fast.

    Think of it like this: every candidate is a point in 384-dimensional
    space.  The JD is also a point.  FAISS finds the 2000 closest candidates
    to the JD point in milliseconds, even with 100K candidates.

    IndexFlatIP = "Inner Product" (dot product after L2 normalization,
    which equals cosine similarity).  It's exact — no approximation.
    """
    global faiss_index, candidate_ids

    candidate_ids = []
    matrix = []

    for candidate in candidates:
        cid = candidate["candidate_id"]
        if cid not in candidate_embeddings:
            continue
        candidate_ids.append(cid)
        matrix.append(candidate_embeddings[cid])

    matrix = np.array(matrix, dtype=np.float32)
    faiss.normalize_L2(matrix)   # normalize so dot product = cosine similarity

    dim = matrix.shape[1]        # 384 for all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(matrix)

    print(f"  FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")


def initialize_embeddings(candidates):
    """
    Master initialization function.
    Checks for a cache first — if we already computed embeddings,
    load them from disk (much faster than recomputing).
    """
    global candidate_embeddings, faiss_index, candidate_ids

    os.makedirs("cache", exist_ok=True)
    emb_path   = "cache/candidate_embeddings.pkl"
    index_path = "cache/candidate.index"
    ids_path   = "cache/candidate_ids.pkl"

    if os.path.exists(emb_path) and os.path.exists(index_path):
        print("Loading cached embeddings (this is fast)...")
        with open(emb_path, "rb") as f:
            candidate_embeddings = pickle.load(f)
        with open(ids_path, "rb") as f:
            candidate_ids = pickle.load(f)
        faiss_index = faiss.read_index(index_path)
        print(f"  Loaded {len(candidate_embeddings)} embeddings, "
              f"{faiss_index.ntotal} FAISS vectors")
        return

    print("Building embeddings for the first time (may take a few minutes)...")
    candidate_embeddings = build_candidate_embeddings(candidates)
    build_faiss_index(candidates)

    with open(emb_path, "wb") as f:
        pickle.dump(candidate_embeddings, f)
    with open(ids_path, "wb") as f:
        pickle.dump(candidate_ids, f)
    faiss.write_index(faiss_index, index_path)
    print("  Cache saved!")

def extract_skills_from_jd(jd_text):
    """
    Find which skills from our vocabulary appear in the JD.
    Also applies aliases so "Large Language Models" → "llms".
    """
    jd_lower = jd_text.lower()
    found = set()

    for skill in all_skills:
        if skill.lower() in jd_lower:
            found.add(skill)

    # Also check aliases — catches paraphrased skill names
    for phrase, canonical in skill_aliases.items():
        if phrase in jd_lower:
            found.add(canonical)

    return list(found)


def extract_required_experience(jd_text):
    """
    Parse the JD for experience requirements.
    Bug fix from original: we now take the MAX match, not the first.
    "3 years Python experience ... minimum 5 years total" → returns 5, not 3.

    Also handles ranges like "5-9 years" → returns the minimum (5).
    """
    jd_lower = jd_text.lower()

    # Match ranges like "5-9 years" or "5–9 years"
    range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*(?:years|yrs)', jd_lower)
    if range_match:
        return int(range_match.group(1))   # use the lower bound of the range

    # Match all "N years" / "N+ years" mentions, take the max
    all_matches = re.findall(r'(\d+)\+?\s*(?:years|yrs)', jd_lower)
    if all_matches:
        return max(int(x) for x in all_matches)

    return 5   # sensible default


def build_jd_embedding(jd_text):
    """
    Convert the JD into a vector.  Called ONCE per request, result
    passed into all scoring functions so we don't re-embed 100 times.
    """
    return model.encode(jd_text)

def is_honeypot(candidate):
    """
    Detect candidates with impossible or suspicious profiles.

    The hackathon has ~80 honeypot candidates designed to fool
    keyword-based systems.  If >10% of your top-100 are honeypots
    → disqualified.

    We check three things:
    1. Too many skills listed (real people don't have 35 skills)
    2. Claims to be "expert" in many skills with 0 years used
    3. Experience at a company that didn't exist yet (if company_founded_year
       is in the data — not always available)
    """
    skills = candidate.get("skills", [])

    # Check 1: Suspiciously long skill list
    if len(skills) > MAX_REASONABLE_SKILLS:
        return True

    # Check 2: Expert in too many skills
    # Real experts have depth in a few things, not everything
    expert_count = sum(
        1 for s in skills
        if s.get("proficiency", "").lower() == "advanced"
    )
    if expert_count >= MAX_SKILL_PROFICIENCY_AT_EXPERT:
        # Also check if years_used is 0 for all those expert skills
        zero_years_expert = sum(
            1 for s in skills
            if s.get("proficiency", "").lower() == "advanced"
            and s.get("years_used", 1) == 0
        )
        if zero_years_expert >= 5:
            return True

    # Check 3: Profile headline is completely unrelated to tech
    headline = candidate.get("profile", {}).get("headline", "").lower()
    for weak in WEAK_ROLE_KEYWORDS:
        if weak in headline:
            # Extra check: do they have ANY real ML career history?
            career_text = " ".join(
                job.get("title", "") + " " + job.get("description", "")
                for job in candidate.get("career_history", [])
            ).lower()
            has_ml_career = any(kw in career_text for kw in AI_CAREER_KEYWORDS[:8])
            if not has_ml_career:
                return True  # HR Manager with no ML history in career → honeypot

    return False

def score_semantic(jd_embedding, candidate):
    """
    How similar is the candidate's full profile to the JD?
    Uses cosine similarity between embedding vectors.

    Score = 0.0  →  completely unrelated
    Score = 1.0  →  nearly identical text (basically impossible in practice)
    Score ≈ 0.5-0.7  →  strong match
    """
    cid = candidate["candidate_id"]
    if cid not in candidate_embeddings:
        return 0.0
    cand_emb = candidate_embeddings[cid]
    sim = cosine_similarity([jd_embedding], [cand_emb])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def proficiency_weight(level):
    """Convert a proficiency level string to a 0–1 score."""
    return {"advanced": 1.0, "intermediate": 0.7, "beginner": 0.4}.get(
        level.lower(), 0.5
    )


def score_skills(candidate, required_skills):
    """
    How many of the required skills does the candidate have?

    IMPORTANT CHANGE from the original:
    We do NOT just count skill keywords.  We weight by:
    - Is this a CRITICAL skill for this JD? (e.g., embeddings, vector DBs) → 3x
    - Is this an IMPORTANT skill? (e.g., Python) → 2x
    - What is the candidate's proficiency level?

    This means a candidate with Advanced FAISS + Advanced RAG scores
    MUCH higher than someone with Beginner Python + Beginner SQL.
    """
    if not required_skills:
        return 0.0

    cand_skill_map = {}
    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        # Normalize through aliases
        name = skill_aliases.get(name, name)
        cand_skill_map[name] = proficiency_weight(skill.get("proficiency", "beginner"))

    total_possible = 0.0
    total_scored   = 0.0

    for req in required_skills:
        req_lower = skill_aliases.get(req.lower(), req.lower())

        # Weight by importance tier
        if req_lower in {s.lower() for s in CRITICAL_SKILLS}:
            weight = 3.0
        elif req_lower in {s.lower() for s in IMPORTANT_SKILLS}:
            weight = 2.0
        else:
            weight = 1.0

        total_possible += weight

        # Check for match (also check if required skill is substring of cand skill)
        matched_prof = 0.0
        for cand_skill, prof in cand_skill_map.items():
            if req_lower in cand_skill or cand_skill in req_lower:
                matched_prof = prof
                break

        total_scored += matched_prof * weight

    return total_scored / total_possible if total_possible > 0 else 0.0


def score_career(candidate):
    """
    THIS IS THE MOST IMPORTANT SCORE FOR THIS JD.

    We analyze the candidate's entire career history text to see if they've
    actually worked on ML/AI systems, not just listed AI skills.

    The JD says: a candidate who built a recommendation system at a startup
    (no AI keywords in skills) beats an HR Manager with RAG listed.

    We also check for negative signals: consulting firms, irrelevant domains.
    """
    career_text = ""
    for job in candidate.get("career_history", []):
        career_text += " " + job.get("title", "")
        career_text += " " + job.get("description", "")
    career_text = career_text.lower()

    if not career_text.strip():
        return 0.0

    # Count positive AI/ML keyword hits
    positive_hits = sum(1 for kw in AI_CAREER_KEYWORDS if kw in career_text)
    positive_score = min(positive_hits / (len(AI_CAREER_KEYWORDS) * 0.5), 1.0)
    # ^ We cap at 50% hit rate = full score, so you don't need ALL keywords

    # Penalty for negative signals
    negative_hits = sum(1 for kw in NEGATIVE_CAREER_KEYWORDS if kw in career_text)
    negative_penalty = min(negative_hits * 0.15, 0.5)

    # Extra bonus: did they explicitly work on ranking/retrieval/recommendation?
    high_value_keywords = [
        "ranking", "retrieval", "recommendation", "search engine",
        "candidate ranking", "vector search", "embedding", "ndcg",
    ]
    high_value_hits = sum(1 for kw in high_value_keywords if kw in career_text)
    high_value_bonus = min(high_value_hits * 0.1, 0.3)

    # Consulting firm penalty
    consulting_penalty = 0.0
    employer_text = " ".join(
        job.get("company", "") for job in candidate.get("career_history", [])
    ).lower()
    if any(firm in employer_text for firm in CONSULTING_FIRMS):
        consulting_penalty = 0.15

    final = positive_score + high_value_bonus - negative_penalty - consulting_penalty
    return float(np.clip(final, 0.0, 1.0))


def score_role_alignment(candidate):
    """
    Does the candidate's job TITLE trajectory match what we need?

    A "Machine Learning Engineer → Senior ML Engineer" career path scores
    higher than "Software Engineer → Tech Lead" which scores higher than
    "Content Writer → Marketing Manager".
    """
    all_titles = candidate.get("profile", {}).get("headline", "").lower()
    for job in candidate.get("career_history", []):
        all_titles += " " + job.get("title", "").lower()

    good_matches = sum(1 for role in GOOD_ROLE_KEYWORDS if role in all_titles)
    weak_matches = sum(1 for role in WEAK_ROLE_KEYWORDS if role in all_titles)

    base_score = min(good_matches / 2.0, 1.0)   # 2 matching titles = full score
    penalty    = min(weak_matches * 0.3, 0.6)

    return float(np.clip(base_score - penalty, 0.0, 1.0))


def score_experience(candidate, required_exp):
    """
    Is the candidate's experience in the right range?

    This JD wants 5-9 years.  We reward being in the range and
    slightly penalize being too junior OR too senior (over-qualified
    candidates often don't want startup roles).
    """
    years = candidate.get("profile", {}).get("years_of_experience", 0)

    if required_exp is None or required_exp == 0:
        required_exp = 5   # default for this JD

    # Sweet spot: required_exp to required_exp+4
    ideal_max = required_exp + 4   # 5-9 years → sweet spot is 5 to 9

    if required_exp <= years <= ideal_max:
        return 1.0
    elif years < required_exp:
        # Under-experienced: scale down
        ratio = years / required_exp
        return max(0.2, ratio)
    else:
        # Over-experienced: slight penalty (too senior for early-stage startup)
        excess = years - ideal_max
        return max(0.6, 1.0 - excess * 0.05)


def score_availability(candidate):
    """
    A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is, for hiring purposes, not available.

    We use 10 of the 23 behavioral signals here.
    """
    signals = candidate.get("redrob_signals", {})

    # ---- Signal 1: Open to work? ----
    open_to_work = signals.get("open_to_work_flag", False)
    open_score = 1.0 if open_to_work else 0.4
    # Not open doesn't mean impossible — they might still respond

    # ---- Signal 2: Last active date ----
    last_active_str = signals.get("last_active_date", "")
    recency_score = 0.5   # default if no date
    if last_active_str:
        try:
            # Parse the date string (handles both date-only and datetime)
            last_active_str_clean = last_active_str[:10]  # "YYYY-MM-DD"
            last_active = datetime.strptime(last_active_str_clean, "%Y-%m-%d")
            # Make both naive (no timezone)
            now = datetime.now()
            days_inactive = (now - last_active).days
            if days_inactive <= 7:
                recency_score = 1.0
            elif days_inactive <= 30:
                recency_score = 0.9
            elif days_inactive <= STALE_DAYS_THRESHOLD:
                recency_score = 0.7
            elif days_inactive <= VERY_STALE_DAYS:
                recency_score = 0.4
            else:
                recency_score = 0.15   # very stale
        except Exception:
            recency_score = 0.5

    # ---- Signal 3: Recruiter response rate ----
    response_rate = signals.get("recruiter_response_rate", 0.5)
    # This is already 0.0–1.0, use directly

    # ---- Signal 4: Interview completion rate ----
    interview_rate = signals.get("interview_completion_rate", 0.5)

    # ---- Signal 5: Notice period ----
    notice_days = signals.get("notice_period_days", 60)
    if notice_days <= IDEAL_NOTICE_DAYS:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.7
    elif notice_days <= MAX_NOTICE_DAYS:
        notice_score = 0.4
    else:
        notice_score = 0.2

    # ---- Signal 6: Location / willing to relocate ----
    # JD wants Pune/Noida; open to other Tier-1 Indian cities
    willing_relocate = signals.get("willing_to_relocate", True)
    work_mode = signals.get("preferred_work_mode", "flexible")
    relocation_score = 1.0 if (willing_relocate or work_mode == "remote") else 0.6

    # ---- Signal 7: Offer acceptance rate ----
    # -1 means no prior offers; otherwise 0.0–1.0
    offer_rate = signals.get("offer_acceptance_rate", -1)
    offer_score = 0.7 if offer_rate == -1 else max(float(offer_rate), 0.2)

    # ---- Signal 8: Platform engagement ----
    profile_completeness = signals.get("profile_completeness_score", 50) / 100.0
    github_raw = signals.get("github_activity_score", -1)
    github_score = (github_raw / 100.0) if github_raw >= 0 else 0.3

    # ---- Combine all signals ----
    availability = (
        0.20 * open_score +
        0.20 * recency_score +
        0.15 * response_rate +
        0.10 * interview_rate +
        0.10 * notice_score +
        0.10 * relocation_score +
        0.08 * offer_score +
        0.04 * profile_completeness +
        0.03 * github_score
    )

    return float(np.clip(availability, 0.0, 1.0))


def score_education(candidate):
    """
    Degree + field + institution tier.
    Less important than career history for this JD but still a signal.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3   # no info, give a middle score

    score = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "").lower()

        # Relevant field?
        if any(kw in field for kw in [
            "computer", "software", "information technology",
            "artificial intelligence", "machine learning", "data science",
            "electrical", "electronics", "mathematics", "statistics"
        ]):
            score += 0.5

        # Higher degree is better for an AI role
        if "phd" in degree or "ph.d" in degree:
            score += 0.4
        elif any(m in degree for m in ["m.tech", "m.e.", "m.s.", "master"]):
            score += 0.3
        elif any(b in degree for b in ["b.tech", "b.e.", "b.s.", "bachelor"]):
            score += 0.2

        # Institution quality
        if tier == "tier_1":
            score += 0.2
        elif tier == "tier_2":
            score += 0.1

    return float(np.clip(score, 0.0, 1.0))


def score_certifications(candidate):
    """Relevant certifications are a small positive signal."""
    certs = candidate.get("certifications", [])
    if not certs:
        return 0.0

    score = 0.0
    relevant_keywords = [
        "machine learning", "ai", "artificial intelligence",
        "data science", "cloud", "aws", "azure", "gcp",
        "deep learning", "nlp",
    ]
    for cert in certs:
        name = cert.get("name", "").lower()
        if any(kw in name for kw in relevant_keywords):
            score += 0.4
        else:
            score += 0.1

    return float(np.clip(score, 0.0, 1.0))

def compute_final_score(candidate, jd_embedding, required_skills, required_exp):
    """
    Combine all 8 sub-scores using the weights from config.py.

    Returns a single float in [0.0, 1.0].
    Higher = better fit.
    """
    scores = {
        "semantic":      score_semantic(jd_embedding, candidate),
        "skill":         score_skills(candidate, required_skills),
        "career":        score_career(candidate),
        "role":          score_role_alignment(candidate),
        "experience":    score_experience(candidate, required_exp),
        "availability":  score_availability(candidate),
        "education":     score_education(candidate),
        "certification": score_certifications(candidate),
    }

    final = sum(SCORE_WEIGHTS[k] * v for k, v in scores.items())
    return float(np.clip(final, 0.0, 1.0)), scores

def retrieve_candidates(jd_text, candidates, top_k=2000):
    """
    Stage 1 of the pipeline: fast FAISS search.
    Returns the top_k most semantically similar candidates.

    We use 2000 (not 1000) to give the re-ranker more to work with.
    Even 2000 FAISS results is almost instantaneous.
    """
    jd_emb = model.encode(jd_text)
    jd_emb_matrix = np.array([jd_emb], dtype=np.float32)
    faiss.normalize_L2(jd_emb_matrix)

    top_k = min(top_k, len(candidate_ids))
    _, indices = faiss_index.search(jd_emb_matrix, top_k)

    lookup = {c["candidate_id"]: c for c in candidates}
    retrieved = []
    for idx in indices[0]:
        if idx == -1 or idx >= len(candidate_ids):
            continue
        cid = candidate_ids[idx]
        if cid in lookup:
            retrieved.append(lookup[cid])

    return retrieved

def rank_candidates(jd_text, candidates, top_n=100):
    """
    Full pipeline:  JD text → ranked list of top_n candidates.

    Returns a list of dicts:
    [
        {
            "candidate_id": "CAND_0042871",
            "rank": 1,
            "score": 0.847,
            "score_breakdown": { "semantic": 0.72, "skill": 0.65, ... },
        },
        ...
    ]
    """

    # --- Step 1: Parse the JD ---
    print("Parsing JD...")
    required_skills = extract_skills_from_jd(jd_text)
    required_exp    = extract_required_experience(jd_text)
    jd_embedding    = build_jd_embedding(jd_text)
    print(f"  Found {len(required_skills)} required skills, {required_exp}+ yrs exp")

    # --- Step 2: Fast FAISS retrieval ---
    print("Running FAISS retrieval...")
    retrieved = retrieve_candidates(jd_text, candidates, top_k=2000)
    print(f"  Retrieved {len(retrieved)} candidates")

    # --- Step 3: Honeypot filter ---
    print("Filtering honeypots...")
    clean_candidates = [c for c in retrieved if not is_honeypot(c)]
    removed = len(retrieved) - len(clean_candidates)
    print(f"  Removed {removed} honeypot/suspicious candidates, {len(clean_candidates)} remain")

    # --- Step 4: Score every remaining candidate ---
    print("Scoring candidates...")
    scored = []
    for candidate in clean_candidates:
        final_score, breakdown = compute_final_score(
            candidate, jd_embedding, required_skills, required_exp
        )
        scored.append({
            "candidate_id":    candidate["candidate_id"],
            "score":           final_score,
            "score_breakdown": breakdown,
            "_candidate":      candidate,   # kept for explanation engine
        })

    # --- Step 5: Sort and return top_n ---
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_n]

    # Assign ranks (1-indexed)
    for i, item in enumerate(top):
        item["rank"] = i + 1

    print(f"Ranking complete. Top score: {top[0]['score']:.4f}, "
          f"Bottom score: {top[-1]['score']:.4f}")

    return top