"""Answer policy questions using retrieved chunks (RAG), with page citations."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from vector_store import search_policy

MAX_DISTANCE = 1.0
NOT_FOUND = "I don't know. The policy document doesn't cover this."

llm = ChatOllama(model="llama3.1:8b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions about a health insurance policy for insurance staff.\n"
        "Use ONLY the policy excerpts below. Do not use outside knowledge.\n"
        "After each fact, cite the page it came from, like this: [Page 2].\n"
        "If the excerpts do not contain the answer, reply exactly: {not_found}\n"
        "If costs differ for network and out-of-network providers, state both separately.\n"
        "Keep the answer to 3 sentences or fewer.\n\n"
        "Policy excerpts:\n{context}",
    ),
    ("human", "{question}"),
])


def format_context(results):
    """Join the retrieved chunks into one text block, each labelled with its page."""
    parts = []
    for chunk, distance in results:
        parts.append(f"[Page {chunk.metadata['page']}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, k=3):
    """Retrieve policy chunks, then ask the LLM to answer using only those chunks."""
    results = search_policy(question, k=k)
    best_distance = results[0][1]

    if best_distance > MAX_DISTANCE:
        return {
            "answer": NOT_FOUND,
            "pages": [],
            "best_distance": round(best_distance, 3),
            "used_llm": False,
        }

    chain = prompt | llm
    response = chain.invoke({
        "context": format_context(results),
        "question": question,
        "not_found": NOT_FOUND,
    })

    pages = sorted({chunk.metadata["page"] for chunk, distance in results})
    return {
        "answer": response.content,
        "pages": pages,
        "best_distance": round(best_distance, 3),
        "used_llm": True,
    }


if __name__ == "__main__":
    questions = [
        "How much does it cost to go to the ER?",
        "What is the deductible?",
        "Is there a limit on how much I pay in a year?",
        "Does this plan cover dental care for adults?",
        "What is the capital of France?",
        "What do I pay for a specialist visit?",
        "Does the plan cover pet insurance?",
        "Who won the hockey game last night?",
    ]

    for question in questions:
        result = answer_question(question)
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("ANSWER:  ", result["answer"])
        print("Pages searched:", result["pages"])
        print("Best distance: ", result["best_distance"])
        print("AI called:     ", result["used_llm"])