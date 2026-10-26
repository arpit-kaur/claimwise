"""Agent step 1: the LLM chooses a tool and its arguments, then our code runs it."""

from langchain_ollama import ChatOllama

from tools import ask_policy, predict_risk

tools = [ask_policy, predict_risk]
tools_by_name = {t.name: t for t in tools}

llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def choose_and_run(question):
    """Ask the LLM which tool to use, then run that tool ourselves."""
    response = llm_with_tools.invoke(question)

    if not response.tool_calls:
        print("No tool chosen. The model replied directly:")
        print(response.content)
        return

    for call in response.tool_calls:
        print("Tool chosen:", call["name"])
        print("Arguments:  ", call["args"])
        chosen_tool = tools_by_name[call["name"]]
        result = chosen_tool.invoke(call["args"])
        print("Tool result:", result)


if __name__ == "__main__":
    questions = [
        "What is the deductible for this plan?",
        "Is a 45 year old male smoker with a BMI of 31, 2 children, "
        "living in the southeast, likely to be high cost?",
        "Check the risk for a 30 year old female non-smoker, BMI 24, "
        "1 child, northwest region.",
        "Check the risk for a 15 year old male smoker, BMI 30, "
        "no children, southeast region.",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        choose_and_run(question)