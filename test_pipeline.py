# =============================================================================
# test_pipeline.py  —  Run the full pipeline end to end
# =============================================================================
#
# HOW TO RUN THIS:
#   python test_pipeline.py
#
# WHAT IT DOES:
#   1. Loads candidates from candidates.jsonl
#   2. Initializes embeddings (loads from cache if available)
#   3. Runs the full ranking pipeline on the sample JD
#   4. Generates submission.csv (the file you submit to the hackathon)
#   5. Prints a detailed breakdown of the top 10 for human inspection
#
# =============================================================================

import json
import csv
import time

import ranking_engine
from ranking_engine import initialize_skills, initialize_embeddings, rank_candidates
from explanation_engine import explain_all

# =============================================================================
# STEP 1: Load candidates
# =============================================================================
print("=" * 60)
print("AI-RECRUITER  —  Full Pipeline Test")
print("=" * 60)

print("\n[1/5] Loading candidates...")
candidates = []
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"  Loaded {len(candidates):,} candidates")

print("\n[2/5] Initializing skill vocabulary...")
initialize_skills(candidates)
print(f"  Skill vocabulary: {len(ranking_engine.all_skills):,} unique skills")

print("\n[3/5] Initializing embeddings...")
t0 = time.time()
initialize_embeddings(candidates)
print(f"  Done in {time.time() - t0:.1f}s")

print("\n[4/5] Running ranking pipeline...")

JD = """
Senior AI Engineer – Generative AI / RAG Systems

We are looking for a highly skilled Senior AI Engineer to join our AI Platform team
and help build production-grade Generative AI applications.

Responsibilities:
- Design and develop scalable Retrieval-Augmented Generation (RAG) systems.
- Build and optimize LLM-powered applications for enterprise use cases.
- Develop semantic search pipelines using vector databases.
- Create document ingestion, chunking, embedding, and retrieval pipelines.
- Fine-tune open-source language models for domain-specific tasks.
- Build evaluation frameworks for AI systems.
- Collaborate with data engineers and software engineers to deploy AI solutions.
- Optimize latency, retrieval quality, and inference performance.
- Develop AI agents capable of tool usage and reasoning.
- Ensure production reliability, monitoring, and scalability of AI services.

Required Skills:
- Strong Python programming skills.
- Expert in Large Language Models (LLMs).
- Must have Retrieval-Augmented Generation (RAG) experience.
- Expert in Vector Databases such as Milvus, Pinecone, Weaviate, or ChromaDB.
- Strong understanding of Embeddings and Semantic Search.
- Experience with LangChain or LlamaIndex.
- Experience with Prompt Engineering.
- Experience building AI Agents.
- Knowledge of Transformer architectures.
- Experience with Hugging Face ecosystem.
- Experience with REST APIs and backend development.

Preferred Skills:
- Fine-tuning LLMs using LoRA or QLoRA.
- Experience with OpenAI, Anthropic, or open-source LLMs.
- Knowledge of MLOps and model deployment.
- Experience with Docker and Kubernetes.
- Experience with AWS, Azure, or GCP.

Experience:
- Minimum 5 years of software engineering experience.
- At least 3 years of experience building AI/ML systems.
- Proven experience delivering production AI solutions.

Important:
Strong experience in LLMs, RAG, and Vector Databases is required.
Production deployment experience is essential.
"""

t0 = time.time()

# rank_candidates returns top 100 by default
ranked_results = rank_candidates(JD, candidates, top_n=100)

elapsed = time.time() - t0
print(f"  Ranking complete in {elapsed:.2f}s")
print(f"  Ranked {len(ranked_results)} candidates")

# Generate explanations for all 100
print("\n[5/5] Generating explanations...")
explanations = explain_all(ranked_results, JD)

print("\n" + "=" * 60)
print("TOP 10 CANDIDATES")
print("=" * 60)

for exp in explanations[:10]:
    print(f"\nRank #{exp['rank']}  |  Score: {exp['score']}")
    print(f"  {exp['headline']}")
    print(f"  Experience: {exp['years_experience']} years")
    print(f"  Matched skills: {', '.join(exp['matched_skills'][:5]) or 'None'}")
    if exp['missing_skills']:
        print(f"  Missing skills: {', '.join(exp['missing_skills'][:3])}")
    print(f"  Availability: {exp['availability']}")
    print(f"  Score breakdown:")
    for component, val in exp['score_breakdown'].items():
        bar = "█" * int(val * 20)
        print(f"    {component:>12}: {val:.3f}  {bar}")
    print(f"  Reasoning: {exp['reasoning']}")

print("\n" + "=" * 60)
print("Writing submission.csv...")

with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["candidate_id", "rank", "score", "reasoning"]
    )
    writer.writeheader()
    for exp in explanations:
        writer.writerow({
            "candidate_id": exp["candidate_id"],
            "rank":         exp["rank"],
            "score":        exp["score"],
            "reasoning":    exp["reasoning"],
        })

print(f"  submission.csv written  ({len(explanations)} rows)")

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

scores = [exp["score"] for exp in explanations]
ranks  = [exp["rank"]  for exp in explanations]

# Check 1: Exactly 100 rows
check1 = len(explanations) == 100
print(f"  [{'✓' if check1 else '✗'}] Exactly 100 candidates ranked")

# Check 2: Ranks are 1-100 with no duplicates
check2 = sorted(ranks) == list(range(1, 101))
print(f"  [{'✓' if check2 else '✗'}] Ranks are 1–100 with no duplicates")

# Check 3: Scores are monotonically non-increasing
check3 = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f"  [{'✓' if check3 else '✗'}] Scores are non-increasing (rank 1 has highest score)")

# Check 4: No empty reasoning strings
check4 = all(exp["reasoning"].strip() for exp in explanations)
print(f"  [{'✓' if check4 else '✗'}] All candidates have non-empty reasoning")

# Check 5: Score range
check5 = all(0.0 <= s <= 1.0 for s in scores)
print(f"  [{'✓' if check5 else '✗'}] All scores in [0.0, 1.0]")

# Check 6: No duplicate candidate IDs
ids    = [exp["candidate_id"] for exp in explanations]
check6 = len(ids) == len(set(ids))
print(f"  [{'✓' if check6 else '✗'}] No duplicate candidate IDs")

print(f"\n  Score range: {scores[-1]:.4f} – {scores[0]:.4f}")
print(f"  Mean score:  {sum(scores)/len(scores):.4f}")

all_passed = all([check1, check2, check3, check4, check5, check6])
if all_passed:
    print("\n   All checks passed — submission.csv is ready!")
else:
    print("\n   Some checks failed — review output above before submitting.")

with open("frontend_results.json", "w", encoding="utf-8") as f:
    # Remove the internal _candidate key before writing to JSON
    clean = []
    for item in ranked_results:
        clean_item = {k: v for k, v in item.items() if k != "_candidate"}
        clean.append(clean_item)
    json.dump(clean, f, indent=2)

print("\n  frontend_results.json written (for the frontend team)")
print("\nDone!")