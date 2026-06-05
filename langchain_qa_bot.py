"""
LangChain Q&A Bot
=================
A conversational Q&A bot built with LangChain and OpenAI.

Covers:
  - Prompt templates
  - LLMChain (basic chain)
  - ConversationChain with memory
  - Sequential chains
  - A simple ReAct agent with tools
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain, ConversationChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to a .env file or export it as an environment variable."
    )


# Model initialisation

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY,
)

# Basic LLMChain with a custom prompt template

def build_qa_chain() -> LLMChain:
    """
    A straightforward Q&A chain.
    The system prompt establishes a concise, accurate assistant persona.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a knowledgeable assistant. "
        "Answer questions clearly and concisely. "
        "If you are unsure, say so rather than guessing."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{question}")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return LLMChain(llm=llm, prompt=chat_prompt)


# ConversationChain with memory

def build_conversation_chain() -> ConversationChain:
    """
    A stateful conversation chain that remembers prior exchanges
    using ConversationBufferMemory.
    """
    memory = ConversationBufferMemory(return_messages=True)

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful, thoughtful assistant who remembers "
        "the context of the conversation."
    )
    history_placeholder = MessagesPlaceholder(variable_name="history")
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt, history_placeholder, human_prompt]
    )

    return ConversationChain(llm=llm, prompt=chat_prompt, memory=memory, verbose=False)


# Sequential chain 

def build_sequential_chain() -> SequentialChain:
    """
    A two-step sequential chain:
      Step 1 — summarise the provided text.
      Step 2 — evaluate the quality of that summary.

    Input  : {"text": "..."}
    Outputs: {"summary": "...", "evaluation": "..."}
    """
    # Step 1: summarise
    summarise_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a professional editor. Summarise the following text "
            "in 2-3 sentences, preserving the key points."
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ])
    summarise_chain = LLMChain(
        llm=llm,
        prompt=summarise_prompt,
        output_key="summary",
    )

    # Step 2: evaluate the summary
    evaluate_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a critical reviewer. "
            "Rate the following summary on clarity and completeness (1-10) "
            "and explain your rating in one sentence."
        ),
        HumanMessagePromptTemplate.from_template("{summary}"),
    ])
    evaluate_chain = LLMChain(
        llm=llm,
        prompt=evaluate_prompt,
        output_key="evaluation",
    )

    return SequentialChain(
        chains=[summarise_chain, evaluate_chain],
        input_variables=["text"],
        output_variables=["summary", "evaluation"],
        verbose=False,
    )


# ReAct agent with a search tool
def build_agent():
    """
    A ReAct-style agent that can use DuckDuckGo to answer questions
    that require up-to-date information.

    Note: DuckDuckGoSearchRun does not require an API key.
    """
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description=(
                "Useful for answering questions about current events, "
                "facts, or anything that benefits from a live web search."
            ),
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True,
        handle_parsing_errors=True,
    )


# Run five sample queries
def run_sample_queries():
    print("=" * 70)
    print("LangChain Q&A Bot — Sample Query Runner")
    print("=" * 70)

    # --- A. Basic LLMChain ---
    qa_chain = build_qa_chain()

    sample_queries = [
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of a vector database in simple terms.",
        "What are the main differences between LangChain and LlamaIndex?",
    ]

    print("\n[ Section A: Basic LLMChain — Standalone Q&A ]\n")
    for i, question in enumerate(sample_queries, 1):
        print(f"Q{i}: {question}")
        response = qa_chain.run(question=question)
        print(f"A{i}: {response.strip()}")
        print("-" * 60)

    # --- B. ConversationChain with memory ---
    conv_chain = build_conversation_chain()

    print("\n[ Section B: ConversationChain — Multi-turn with memory ]\n")
    turns = [
        "My name is Mohammed and I am building a RAG system.",
        "What framework would you recommend for my project, given what I just told you?",
    ]
    for i, turn in enumerate(turns, 1):
        print(f"Turn {i}: {turn}")
        response = conv_chain.predict(input=turn)
        print(f"Bot   : {response.strip()}")
        print("-" * 60)

    # --- C. Sequential chain ---
    seq_chain = build_sequential_chain()

    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large "
        "language models by allowing them to retrieve relevant documents from an "
        "external knowledge base at inference time. Instead of relying solely on "
        "parametric knowledge baked into the model weights, RAG grounds responses "
        "in retrieved context, reducing hallucinations and enabling knowledge updates "
        "without full retraining."
    )

    print("\n[ Section C: Sequential Chain — Summarise then Evaluate ]\n")
    print(f"Input text:\n{sample_text}\n")
    result = seq_chain({"text": sample_text})
    print(f"Summary   : {result['summary'].strip()}")
    print(f"Evaluation: {result['evaluation'].strip()}")
    print("-" * 60)


def interactive_chat():
    print("\n[ Interactive Chat Mode ]")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    conv_chain = build_conversation_chain()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Session ended.")
            break
        response = conv_chain.predict(input=user_input)
        print(f"Bot: {response.strip()}\n")


if __name__ == "__main__":
    # Run the five sample queries to demonstrate all three chain types
    run_sample_queries()

    # Then drop into an interactive session
    interactive_chat()
