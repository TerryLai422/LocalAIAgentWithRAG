# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, get_shop_stats_from_df, is_aggregation_semantic  # keep semantic fallback
import json
import re

# LLM used both for answering and for intent classification
llm = OllamaLLM(model="llama3.2", base_url="http://192.168.51.147:11434")

# Existing QA prompt (unchanged)
qa_template = """
You are an expert in answering questions about pizzas based on customer reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_chain = qa_prompt | llm

# -------------------------
# LLM intent classifier
# -------------------------
# This prompt instructs the LLM to return ONLY a JSON object with keys: label and confidence.
# label must be either "aggregation" or "other".
# confidence must be a float between 0.0 and 1.0.
CLASSIFIER_PROMPT = """
You are a classifier. Given a single user question about restaurant reviews, decide whether the user is
asking for an aggregation/statistics query (for example: "Which shops have the best rating?",
"Top rated shops", "Which restaurants are highest rated?", "Show me shops with most 5-star reviews")
or a non-aggregation question (for example: "What do customers say about the pizza at X?",
"Summarize reviews for shop Y", "Is the pizza spicy?").

Return ONLY a JSON object with two fields:
- "label": either "aggregation" or "other"
- "confidence": a number between 0.0 and 1.0 indicating how confident you are

Example valid output:
{"label":"aggregation","confidence":0.95}

Now classify this question:
{question}
"""

def classify_with_llm(question: str, min_confidence: float = 0.75) -> (bool, dict):
    """
    Returns (is_aggregation, debug_info)
    debug_info contains parsed label, confidence, and raw LLM text.
    Falls back to semantic exemplar method if parsing fails or confidence < min_confidence.
    """
    prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
    chain = prompt | llm

    # Call the LLM classifier
    try:
        raw = chain.invoke({"question": question})
        # chain.invoke may return a string; ensure we have text
        if isinstance(raw, dict) and "content" in raw:
            raw_text = raw["content"]
        else:
            raw_text = str(raw).strip()
    except Exception as e:
        # LLM call failed; fallback to semantic method
        fallback_info = {"reason": "llm_call_failed", "error": str(e)}
        is_agg, sem_debug = is_aggregation_semantic(question)
        fallback_info["semantic_debug"] = sem_debug
        return is_agg, {"raw": None, "parsed": None, "fallback": fallback_info}

    # Try to extract JSON from the LLM output
    parsed = None
    try:
        # Some LLMs may wrap JSON in backticks or markdown; extract first {...}
        m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        json_text = m.group(0) if m else raw_text
        parsed = json.loads(json_text)
        label = parsed.get("label", "").strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        # parsing failed -> fallback to semantic exemplar method
        is_agg, sem_debug = is_aggregation_semantic(question)
        return is_agg, {"raw": raw_text, "parsed": None, "fallback": sem_debug}

    debug = {"raw": raw_text, "parsed": parsed}

    # Decide based on label and confidence threshold
    if label == "aggregation" and confidence >= min_confidence:
        return True, debug
    elif label == "other" and confidence >= min_confidence:
        return False, debug
    else:
        # low confidence -> fallback to semantic exemplar method
        is_agg, sem_debug = is_aggregation_semantic(question)
        debug["fallback"] = sem_debug
        return is_agg, debug

# -------------------------
# Helper to format stats
# -------------------------
def format_stats(stats):
    lines = []
    for i, (shop, avg, cnt) in enumerate(stats, start=1):
        lines.append(f"{i}. {shop} — avg rating: {avg:.2f}, reviews: {cnt}")
    return "\n".join(lines)

# -------------------------
# Main loop
# -------------------------
if __name__ == "__main__":
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        print("\n\n")
        if question == "q":
            break

        # Use LLM classifier
        is_agg, debug = classify_with_llm(question, min_confidence=0.75)
        # Optional: print debug while tuning
        # import pprint; pprint.pprint(debug)

        if is_agg:
            stats = get_shop_stats_from_df(n_top=10, min_reviews=1)
            print("Top shops by average rating:")
            print(format_stats(stats))
            continue

        # fallback to semantic retrieval + LLM answer
        reviews = retriever.invoke(question)
        print("Relevant reviews:")
        for r in reviews:
            print(r.page_content)
        result = qa_chain.invoke({"reviews": reviews, "question": question})
        print(result)
