# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, get_shop_stats_from_df, get_shop_stats_from_chroma
import re

model = OllamaLLM(model="llama3.2", base_url="http://192.168.51.147:11434")

template = """
You are an expert in answering questions about pizzas based on customer reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def is_rating_aggregation_question(q: str) -> bool:
    q_lower = q.lower()
    # simple keyword-based intent detection
    patterns = [
        r"which shops have the best rating",
        r"which shop(s)? (have|has) the best rating",
        r"top rated shops",
        r"best rated (shops|restaurants)"
    ]
    for p in patterns:
        if re.search(p, q_lower):
            return True
    return False

def format_stats(stats):
    lines = []
    for i, (shop, avg, cnt) in enumerate(stats, start=1):
        lines.append(f"{i}. {shop} — avg rating: {avg:.2f}, reviews: {cnt}")
    return "\n".join(lines)

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    if is_rating_aggregation_question(question):
        # Option A: compute from CSV (fast and accurate)
        stats = get_shop_stats_from_df(n_top=10, min_reviews=1)
        # Option B: compute from Chroma metadata instead
        # stats = get_shop_stats_from_chroma(n_top=10, min_reviews=1)
        print("Top shops by average rating:")
        print(format_stats(stats))
        continue

    # fallback to semantic retrieval + LLM
    reviews = retriever.invoke(question)
    print("Relevant reviews:")
    for r in reviews:
        print(r.page_content)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
