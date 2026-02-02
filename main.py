from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2", base_url="http://192.168.51.147:11434")

template = """
You are an exeprt in answering questions about pizzas based on customer reviews.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    print("Relevant reviews:")
    for r in reviews:
        print(r.page_content)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)