from ranking_engine import *
from config import *

def explain_candidate(candidate, jd, required_skills, required_exp):

    candidate_skills = [
        skill["name"].lower()
        for skill in candidate["skills"]
    ]

    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        if skill.lower() in candidate_skills:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    years_exp = candidate["profile"]["years_of_experience"]

    semantic_score = semantic_similarity_cached(jd, candidate)
    skill_score = skill_match_score_with_proficiency(
        candidate["skills"],
        required_skills
    )

    exp_score = experience_score(
        years_exp,
        required_exp
    )

    llm_score = llm_bonus(candidate)

    return {
        "candidate_id": candidate["candidate_id"],
        "headline": candidate["profile"]["headline"],
        "experience": years_exp,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "semantic_score": round(float(semantic_score), 3),
        "skill_score": round(float(skill_score), 3),
        "experience_score": round(float(exp_score), 3),
        "llm_score": round(float(llm_score), 3)
    }

def print_explanation(explanation):

    print("=" * 60)

    print("Candidate:")
    print(explanation["headline"])

    print()

    print("Experience:")
    print(explanation["experience"], "years")

    print()

    print("Matched Skills:")
    for skill in explanation["matched_skills"]:
        print("✓", skill)

    print()

    print("Missing Skills:")
    for skill in explanation["missing_skills"]:
        print("✗", skill)

    print()

    print("Scores:")
    print("Semantic:", explanation["semantic_score"])
    print("Skill:", explanation["skill_score"])
    print("Experience:", explanation["experience_score"])
    print("LLM:", explanation["llm_score"])

