from config import *

skill_aliases = {
    "Large Language Models": "LLMs","LLM": "LLMs","Vector Database": "Milvus","Vector Databases": "Milvus","Retrieval Augmented Generation": "RAG","Retrieval-Augmented Generation": "RAG","Computer Vision": "Computer Vision","Natural Language Processing": "NLP","Backend APIs": "REST APIs"
}

role_skill_map = {

    "genai engineer": ["LLMs","RAG","Prompt Engineering","Milvus","FAISS","LoRA"],

    "ml engineer": ["Python","MLOps","PyTorch"],

    "computer vision engineer": ["Computer Vision","Image Classification","GANs"],

    "data engineer": ["Kafka","Snowflake","ETL","dbt"]
}

AI_KEYWORDS = ["machine learning","ml","ai","deep learning","llm","nlp","computer vision","rag","transformer","neural network","python"]

ai_roles = ["ai engineer","ml engineer","machine learning engineer","data scientist","nlp engineer","llm engineer","ai research engineer","genai engineer","generative ai engineer","mlops engineer","machine learning scientist","ai scientist","data engineer","backend engineer"]

LLM_SKILLS = ["llms","rag","vector databases","vector search","langchain","fine tuning","hugging face","lora","openai"]

print(len(ai_roles))

