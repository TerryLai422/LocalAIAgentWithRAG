# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, get_shop_stats_from_df, is_aggregation_semantic
import json

model = OllamaLLM(model="llama3.2", base_url="http://192.168.51.147:11434")

template = """
You are an expert in answering questions about pizzas based on customer reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

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

    # Semantic intent detection
    is_agg, debug = is_aggregation_semantic(question)
    # Optional: print debug to tune threshold during development
    # print("DEBUG intent:", json.dumps(debug, indent=2))

    if is_agg:
        # compute from CSV (accurate) or from Chroma metadata
        stats = get_shop_stats_from_df(n_top=10, min_reviews=1)
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
