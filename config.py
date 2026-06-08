skill_aliases = {
    "large language models": "llms",
    "llm":                   "llms",
    "vector database":       "vector databases",
    "vector db":             "vector databases",
    "vectordb":              "vector databases",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "natural language processing":    "nlp",
    "backend apis":                   "rest apis",
    "huggingface":                    "hugging face",
    "fine-tuning":                    "fine tuning",
    "finetuning":                     "fine tuning",
    "lora":                           "fine tuning",
    "qlora":                          "fine tuning",
}

CRITICAL_SKILLS = {
    "embeddings", "sentence transformers", "vector databases", "faiss",
    "pinecone", "milvus", "weaviate", "qdrant", "opensearch",
    "rag", "llms", "ranking", "information retrieval", "hybrid search",
    "ndcg", "map", "mrr", "a/b testing", "evaluation frameworks",
}

IMPORTANT_SKILLS = {
    "python", "pytorch", "transformers", "hugging face",
    "fine tuning", "langchain", "llamaindex", "mlops",
    "elasticsearch", "rest apis", "docker", "kubernetes",
}

GOOD_ROLE_KEYWORDS = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "llm engineer",
    "search engineer", "ranking engineer", "recommendation engineer",
    "applied scientist", "research engineer", "mlops engineer",
    "genai engineer", "generative ai engineer",
]

# The JD explicitly says these are disqualifiers or weak signals.
WEAK_ROLE_KEYWORDS = [
    "marketing manager", "hr manager", "content writer",
    "graphic designer", "accountant", "civil engineer",
    "mechanical engineer", "sales executive", "business analyst",
    "project manager",
]

# Consulting firms the JD explicitly says are a poor fit signal
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis",
]

AI_CAREER_KEYWORDS = [
    "machine learning", "deep learning", "neural network",
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "search", "nlp", "transformer", "fine-tun", "llm",
    "model", "inference", "training", "pytorch", "tensorflow",
    "a/b test", "evaluation", "precision", "recall", "ndcg",
    "production", "deployed", "scaled", "latency",
]

# These in career descriptions suggest the candidate is NOT a good fit
NEGATIVE_CAREER_KEYWORDS = [
    "sap", "cobol", "erp", "crm", "salesforce admin",
    "manual testing", "qa testing", "content writing",
    "social media", "digital marketing", "seo", "graphic design",
]

SCORE_WEIGHTS = {
    "semantic":     0.20,   
    "skill":        0.18,   
    "career":       0.25,   
    "role":         0.12,   
    "experience":   0.10,   
    "availability": 0.08,   
    "education":    0.04,   
    "certification":0.03,   
}

_weight_sum = sum(SCORE_WEIGHTS.values())
assert abs(_weight_sum - 1.0) < 0.001, f"Weights sum to {_weight_sum}, must be 1.0"

STALE_DAYS_THRESHOLD    = 90    
VERY_STALE_DAYS         = 180   
IDEAL_NOTICE_DAYS       = 30
MAX_NOTICE_DAYS         = 90    
MAX_SKILL_PROFICIENCY_AT_EXPERT = 8 
MAX_REASONABLE_SKILLS = 30            