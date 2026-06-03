import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import *

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

candidate_embeddings = {}
all_skills = set()

def extract_skills_from_jd(jd, skill_vocabulary):

    jd_lower = jd.lower()

    found_skills = set()

    for skill in skill_vocabulary:

        if skill.lower() in jd_lower:
            found_skills.add(skill)

    for phrase, skill in skill_aliases.items():

        if phrase.lower() in jd_lower:
            found_skills.add(skill)

    return list(found_skills)

def extract_required_experience(jd):
    
    jd = jd.lower()

    match = re.search(r'(\d+)\+?\s*(?:years|yrs)',jd)

    if match:
        return int(match.group(1))

    return None

def get_candidate_skills(candidate):

    skills = []

    for skill in candidate["skills"]:
        skills.append(skill["name"])

    return skills

def proficiency_to_score(level):

    level = level.lower()

    mapping = {"advanced":1.0,"intermediate":0.7,"beginner":0.4}

    return mapping.get(level,0.5)

def skill_match_score(candidate_skills,required_skills):

    matches = 0

    for skill in required_skills:

        if skill in candidate_skills:
            matches += 1

    return matches / len(required_skills)


def skill_match_score_with_proficiency(candidate_skills,required_skills):

    if len(required_skills) == 0:
        return 0

    score = 0

    for skill in candidate_skills:

        skill_name = skill["name"].lower()

        for req_skill in required_skills:

            if req_skill.lower() in skill_name:

                score += proficiency_to_score(
                    skill["proficiency"]
                )

                break

    return score / len(required_skills)

def experience_score(candidate_exp, required_exp):

    if required_exp is None:
        return 0.5

    ratio = candidate_exp / required_exp

    if ratio >= 1:
        return 1.0

    elif ratio >= 0.8:
        return 0.8

    elif ratio >= 0.5:
        return 0.5

    else:
        return 0.2

def candidate_quality_score(candidate):

    signals = candidate["redrob_signals"]

    profile_score = (signals["profile_completeness_score"]/ 100)

    response_score = (signals["recruiter_response_rate"])

    interview_score = (signals["interview_completion_rate"])

    github_score = (signals["github_activity_score"]/ 100)

    return (0.35 * profile_score + 0.25 * response_score + 0.25 * interview_score + 0.15 * github_score)

def career_relevance_score(candidate):

    text = ""

    for job in candidate["career_history"]:
        text += job["title"] + " "
        text += job["description"] + " "

    text = text.lower()

    matches = 0

    for keyword in AI_KEYWORDS:

        if keyword in text:
            matches += 1

    return matches / len(AI_KEYWORDS)

def role_alignment_score(candidate):

    text = candidate["profile"]["headline"].lower()

    for job in candidate["career_history"]:
        text += " " + job["title"].lower()

    ai_roles = ["ai engineer","ml engineer","machine learning engineer","data scientist","nlp engineer","llm engineer","ai research engineer"]

    matches = 0

    for role in ai_roles:
        if role in text:
            matches += 1

    return min(matches / 3, 1.0)

def llm_bonus(candidate):

    text = build_candidate_profile(candidate).lower()

    matches = 0

    for skill in LLM_SKILLS:
        if skill in text:
            matches += 1

    return matches / len(LLM_SKILLS)

def build_candidate_profile(candidate):

    profile = ""

    profile += candidate["profile"]["headline"] + " "

    profile += candidate["profile"]["summary"] + " "

    for skill in candidate["skills"]:
        profile += skill["name"] + " "

    for job in candidate["career_history"]:
        profile += job["title"] + " "
        profile += job["description"] + " "

    return profile

def build_candidate_embeddings(candidates):

    candidate_embeddings = {}

    for candidate in candidates:

        candidate_embeddings[
            candidate["candidate_id"]
        ] = model.encode(
            build_candidate_profile(candidate)
        )

    return candidate_embeddings    

def initialize_embeddings(candidates):

    global candidate_embeddings

    candidate_embeddings = build_candidate_embeddings(
        candidates
    )

def initialize_skills(candidates):

    global all_skills

    all_skills = set()

    for candidate in candidates:
        for skill in candidate["skills"]:
            all_skills.add(skill["name"])

    return all_skills    

def semantic_similarity_cached(jd_text,candidate):

    jd_embedding = model.encode(jd_text)

    candidate_embedding = candidate_embeddings[candidate["candidate_id"]]

    similarity = cosine_similarity([jd_embedding],[candidate_embedding])[0][0]

    return similarity

def final_candidate_score(jd,candidate,required_skills,required_exp):

    semantic_score = semantic_similarity_cached(jd,candidate)

    skill_score = skill_match_score_with_proficiency(candidate["skills"],required_skills)

    exp_score = experience_score(candidate["profile"]["years_of_experience"],required_exp)

    quality_score = candidate_quality_score(candidate)

    career_score = career_relevance_score(candidate)

    role_score = role_alignment_score(candidate)

    llm_score = llm_bonus(candidate)

    return (0.35 * semantic_score + 0.20 * skill_score + 0.10 * exp_score + 0.15 * quality_score + 0.10 * career_score + 0.05 * role_score + 0.05 * llm_score)

def rank_candidates_semantic(jd,candidates,top_n=10):

    required_skills = extract_skills_from_jd(jd,all_skills)

    required_exp = extract_required_experience(jd)

    scores = []

    for candidate in candidates:

        score = final_candidate_score(jd,candidate,required_skills,required_exp)

        scores.append((candidate["candidate_id"],score))

    return sorted(scores,key=lambda x: x[1],reverse=True)[:top_n]

def build_candidate_embeddings(candidates):

    candidate_embeddings = {}

    for candidate in candidates:

        candidate_embeddings[
            candidate["candidate_id"]
        ] = model.encode(
            build_candidate_profile(candidate)
        )

    return candidate_embeddings
 
def rank_candidates(jd,candidates,top_n=10):
    return rank_candidates_semantic(jd,candidates,top_n)

