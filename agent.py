"""ClaimWise agent: a LangGraph loop with human approval and a step limit."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ask_policy, predict_risk

MAX_TOOL_CALLS = 3
NEEDS_APPROVAL = {"predict_risk"}

SYSTEM_PROMPT = (
    "You are ClaimWise, an assistant for health insurance staff.\n"
    "Use the ask_policy tool for any question about the insurance policy.\n"
    "Use the predict_risk tool when customer details are given and risk is asked.\n"
    "Never guess policy facts or risk yourself. Always use a tool.\n"
    "Call a tool only once per question. Do not look up extra questions on your own.\n"
    "After you get the tool result, answer the user's original question using that result.\n"
    "Keep any page citations from the tool results.\n"
    "If a tool returns an error, explain it simply.\n"
    "Only if the question has nothing to do with insurance or customer risk, reply: "
    "I can only help with policy questions and customer risk."
)
ANSWER_PROMPT = (
    "You are ClaimWise, an assistant for health insurance staff.\n"
    "Answer the user's question using only the tool result in the conversation.\n"
    "Include the numbers and any page citations from the tool result.\n"
    "If the tool returned an error, explain it simply.\n"
    "Keep the answer to 3 sentences or fewer."
)

tools = [ask_policy, predict_risk]
llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState):
    """The agent node: pick a tool first, then write the answer from the tool result."""
    last_message = state["messages"][-1]

    if last_message.type == "tool":
        messages = [SystemMessage(content=ANSWER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["tools"])


def ask_agent(question, thread_id):
    """Run the agent for one question, pausing for approval before risky tools."""
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"messages": [HumanMessage(content=question)]}, config)

    tool_calls_made = 0
    while graph.get_state(config).next:
        tool_calls_made += 1
        if tool_calls_made > MAX_TOOL_CALLS:
            return f"Stopped: the agent used more than {MAX_TOOL_CALLS} tool calls."

        last_message = graph.get_state(config).values["messages"][-1]
        for call in last_message.tool_calls:
            print("  Agent wants to use:", call["name"], call["args"])

            if call["name"] in NEEDS_APPROVAL:
                choice = input("  Approve this tool call? (y/n): ")
                if choice.strip().lower() != "y":
                    return "Cancelled: staff did not approve the tool call."

        graph.invoke(None, config)

    for message in graph.get_state(config).values["messages"]:
        if message.type == "tool":
            print("  Tool result:", message.content[:150])
    return graph.get_state(config).values["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "What is the deductible for this plan?",
        "Is a 45 year old male smoker, BMI 31, 2 children, in the southeast, high risk?",
        "Check the risk for a 30 year old female non-smoker, BMI 24, 1 child, northwest.",
        "What is the capital of France?",
    ]

    for number, question in enumerate(questions, start=1):
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        answer = ask_agent(question, thread_id=str(number))
        print("FINAL ANSWER:", answer)