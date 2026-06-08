from config import *
from ranking_engine import (
    extract_skills_from_jd,
    extract_required_experience,
    skill_aliases,
    AI_CAREER_KEYWORDS,
    GOOD_ROLE_KEYWORDS,
    CONSULTING_FIRMS,
)
from datetime import datetime

def analyze_skills(candidate, required_skills):
    """
    Compare candidate skills against required skills.
    Returns matched skills (with proficiency), missing skills, and bonus skills.

    'bonus_skills' = candidate has important skills the JD didn't even ask for
    — this is a strong positive signal worth mentioning in the reasoning.
    """
    # Build a normalized map: canonical_skill_name → proficiency
    cand_skill_map = {}
    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        canonical = skill_aliases.get(name, name)
        cand_skill_map[canonical] = {
            "original_name": skill.get("name", ""),
            "proficiency":   skill.get("proficiency", "beginner"),
            "years_used":    skill.get("years_used", 0),
        }

    matched  = []
    missing  = []

    for req in required_skills:
        req_lower    = req.lower()
        req_canonical = skill_aliases.get(req_lower, req_lower)

        # Check for exact or partial match
        found = False
        for cand_canonical, info in cand_skill_map.items():
            if req_canonical in cand_canonical or cand_canonical in req_canonical:
                matched.append({
                    "skill":      info["original_name"],
                    "proficiency": info["proficiency"],
                    "years_used":  info["years_used"],
                })
                found = True
                break

        if not found:
            missing.append(req)

    # Bonus skills: critical/important skills candidate has beyond what JD asked
    bonus = []
    all_critical_lower = {s.lower() for s in CRITICAL_SKILLS}
    for cand_canonical, info in cand_skill_map.items():
        if (cand_canonical in all_critical_lower
                and cand_canonical not in {skill_aliases.get(r.lower(), r.lower())
                                           for r in required_skills}
                and info["proficiency"].lower() in ("advanced", "intermediate")):
            bonus.append(info["original_name"])

    return matched, missing, bonus[:3]  

def extract_career_highlights(candidate):
    """
    Pull out 1-2 concrete facts from the candidate's career history
    that are relevant to this JD.

    We look for:
    - Most recent (or most senior) relevant title
    - A description snippet that mentions ranking/retrieval/embedding/etc.
    - Company names (product companies are a positive signal)
    """
    career = candidate.get("career_history", [])
    if not career:
        return None, None, None

    # Most recent relevant job
    relevant_jobs = []
    for job in career:
        title = job.get("title", "").lower()
        desc  = job.get("description", "").lower()
        combined = title + " " + desc
        if any(kw in combined for kw in AI_CAREER_KEYWORDS[:12]):
            relevant_jobs.append(job)

    best_job = relevant_jobs[0] if relevant_jobs else career[0]

    title   = best_job.get("title", "")
    company = best_job.get("company", "")

    # Find a specific keyword hit in the description to cite
    desc       = best_job.get("description", "")
    desc_lower = desc.lower()

    highlight_keywords = [
        "ranking", "retrieval", "recommendation", "search",
        "embedding", "vector", "fine-tun", "deployed", "production",
        "ndcg", "a/b", "latency", "scale",
    ]
    found_highlight = None
    for kw in highlight_keywords:
        if kw in desc_lower:
            found_highlight = kw
            break

    return title, company, found_highlight


def get_availability_summary(candidate):
    """
    Return a short human-readable string about candidate availability.
    E.g. "actively looking, 30-day notice" or "not open to work, last active 6 months ago"
    """
    signals = candidate.get("redrob_signals", {})

    open_to_work  = signals.get("open_to_work_flag", False)
    notice_days   = signals.get("notice_period_days", 60)
    last_active   = signals.get("last_active_date", "")

    parts = []

    if open_to_work:
        parts.append("actively looking")
    else:
        parts.append("not marked open to work")

    if notice_days <= 15:
        parts.append("immediately available")
    elif notice_days <= 30:
        parts.append(f"{notice_days}-day notice")
    elif notice_days <= 60:
        parts.append(f"{notice_days}-day notice (buyout possible)")
    else:
        parts.append(f"{notice_days}-day notice (long)")

    if last_active:
        try:
            last_dt   = datetime.strptime(last_active[:10], "%Y-%m-%d")
            days_ago  = (datetime.now() - last_dt).days
            if days_ago <= 7:
                parts.append("active this week")
            elif days_ago <= 30:
                parts.append(f"active {days_ago}d ago")
            elif days_ago <= 90:
                parts.append(f"active ~{days_ago//30}mo ago")
            else:
                parts.append(f"inactive {days_ago//30}mo")
        except Exception:
            pass

    return ", ".join(parts)

def build_reasoning(candidate, score, score_breakdown, required_skills):
    """
    Generate a specific, data-grounded 1-2 sentence reasoning string.

    RULES (from hackathon spec):
    - Must be specific to this candidate — no generic templates
    - Must NOT hallucinate (only say things the data supports)
    - Tone should match rank: top candidates get enthusiastic, bottom get honest
    - Length: 1-2 sentences only

    HOW THIS WORKS:
    We have a set of sentence "fragments" for each component (skills, career,
    availability, experience).  We pick the most relevant 2-3 fragments
    and join them into a natural-sounding sentence.
    """
    profile    = candidate.get("profile", {})
    headline   = profile.get("headline", "this candidate")
    years_exp  = profile.get("years_of_experience", 0)

    matched_skills, missing_skills, bonus_skills = analyze_skills(
        candidate, required_skills
    )
    title, company, career_highlight = extract_career_highlights(candidate)
    availability_str = get_availability_summary(candidate)

    breakdown = score_breakdown
    strengths = []

    # Career signal — strongest for this JD
    if breakdown.get("career", 0) >= 0.6:
        if title and company:
            strengths.append(
                f"has hands-on production experience as {title} at {company}"
            )
        elif career_highlight:
            strengths.append(
                f"has demonstrated {career_highlight}-related work in their career history"
            )

    # Skill signal
    if matched_skills:
        # Highlight the 2-3 most impressive matched skills
        advanced_matched = [
            s["skill"] for s in matched_skills
            if s["proficiency"].lower() == "advanced"
        ][:3]
        if advanced_matched:
            skills_str = ", ".join(advanced_matched)
            strengths.append(f"shows advanced proficiency in {skills_str}")
        elif len(matched_skills) >= 3:
            skills_str = ", ".join(s["skill"] for s in matched_skills[:3])
            strengths.append(f"matches key required skills: {skills_str}")

    # Bonus skills
    if bonus_skills:
        strengths.append(f"also brings {', '.join(bonus_skills[:2])}")

    # Experience
    if 5 <= years_exp <= 10:
        strengths.append(f"{years_exp} years of relevant experience")

    # Semantic — if very high, note the overall profile alignment
    if breakdown.get("semantic", 0) >= 0.65 and not strengths:
        strengths.append("strong overall profile alignment with this JD")

    if strengths:
        sentence1 = headline + " " + "; ".join(strengths[:2]) + "."
    else:
        sentence1 = f"{headline} has {years_exp} years of experience with partial skill overlap."

    sentence2 = ""

    critical_missing = [
        s for s in missing_skills
        if s.lower() in {c.lower() for c in CRITICAL_SKILLS}
    ]

    if critical_missing and score < 0.6:
        gap = critical_missing[0]
        sentence2 = f"Gap to probe: no explicit {gap} experience found; verify in interview."
    elif availability_str:
        sentence2 = f"Availability: {availability_str}."

    reasoning = sentence1
    if sentence2:
        reasoning = reasoning + " " + sentence2

    return reasoning

def explain_candidate(ranked_item, required_skills):
    """
    Takes one item from rank_candidates() output and returns a
    complete explanation dict ready for the frontend and the CSV.

    ranked_item looks like:
    {
        "candidate_id": "CAND_0042871",
        "rank": 1,
        "score": 0.847,
        "score_breakdown": { "semantic": 0.72, "skill": 0.65, ... },
        "_candidate": { ... full candidate dict ... }
    }
    """
    candidate      = ranked_item["_candidate"]
    score          = ranked_item["score"]
    score_breakdown = ranked_item["score_breakdown"]
    rank           = ranked_item["rank"]

    matched_skills, missing_skills, bonus_skills = analyze_skills(
        candidate, required_skills
    )

    reasoning = build_reasoning(
        candidate, score, score_breakdown, required_skills
    )

    profile   = candidate.get("profile", {})
    signals   = candidate.get("redrob_signals", {})

    return {
        # ---- Core output fields (required by submission spec) ----
        "candidate_id": candidate["candidate_id"],
        "rank":         rank,
        "score":        round(score, 4),
        "reasoning":    reasoning,

        # ---- Rich data for frontend display ----
        "headline":         profile.get("headline", ""),
        "years_experience": profile.get("years_of_experience", 0),
        "matched_skills":   [s["skill"] for s in matched_skills],
        "missing_skills":   missing_skills[:5],
        "bonus_skills":     bonus_skills,
        "availability":     get_availability_summary(candidate),

        # ---- Score breakdown for frontend score bars ----
        "score_breakdown": {
            k: round(v, 3) for k, v in score_breakdown.items()
        },

        # ---- Flags ----
        "open_to_work":   signals.get("open_to_work_flag", False),
        "notice_days":    signals.get("notice_period_days", 60),
        "response_rate":  signals.get("recruiter_response_rate", 0.5),
    }


def explain_all(ranked_results, jd_text):
    """
    Run explain_candidate on all ranked results.
    This is the function your backend endpoint calls.

    Returns a list of explanation dicts, one per candidate.
    """
    required_skills = extract_skills_from_jd(jd_text)

    explanations = []
    for item in ranked_results:
        exp = explain_candidate(item, required_skills)
        explanations.append(exp)

    return explanations