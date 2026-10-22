"""First AI calls: talk to a local Llama model through LangChain."""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", temperature=0)

# 1. The simplest call: one question, one answer
response = llm.invoke("In two sentences, what does a health insurance company do?")
print("1. Simple question:")
print(response.content)

# 2. A system message controls how the AI behaves
messages = [
    SystemMessage(
        content=(
            "You are an assistant for insurance staff. "
            "Answer in plain language, in at most 3 sentences. "
            "If you are not sure about something, say so."
        )
    ),
    HumanMessage(content="Why might a smoker cost a health insurer more?"),
]

start = time.time()
response = llm.invoke(messages)
seconds = time.time() - start

print("\n2. With a system message:")
print(response.content)

# 3. Tokens and speed
print("\n3. Tokens used:", response.usage_metadata)
print(f"Time taken: {seconds:.1f} seconds")