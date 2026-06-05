# LangChain Q&A Bot

A minimal but complete LangChain project demonstrating prompt templates,
chains, memory, sequential chains, and a ReAct agent.

---

## Files

| File | Purpose |
|------|---------|
| `langchain_qa_bot.py` | Standalone Python script — run directly from the terminal |
| `.env.example` | Template for environment variables |

---

## Quick Start

### 1. Clone / copy the project

```bash
cd your-project-folder
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install langchain langchain-openai langchain-community python-dotenv duckduckgo-search
```

### 4. Set your API key

Copy the example file and fill in your key in `.env`:

I used Free OpenAI modell from Openrouter.com
```
OPENAI_API_KEY=sk-your-actual-key-here
```
---

## What is demonstrated

### Prompt Templates
`ChatPromptTemplate` with `SystemMessagePromptTemplate` and
`HumanMessagePromptTemplate`. Templates accept named variables
(`{question}`, `{text}`) that are filled at runtime.

### LLMChain (basic chain)
The simplest chain: prompt template + LLM. Input variables are
passed via `.run()` or `.predict()`.

### ConversationChain with memory
`ConversationBufferMemory` stores the full message history.
The chain injects it automatically on every call, giving the model
full context of the current session. Memory resets when the chain
object is re-created.

### SequentialChain
Pipes the output of one chain into the input of the next.
Useful for multi-step reasoning: summarise -> evaluate,
draft -> review -> rewrite, etc.

### ReAct Agent
An agent that chooses from a set of tools using the
Thought / Action / Observation loop. The example agent can
call DuckDuckGo to answer questions requiring live data.

---

## Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (`sk-...`) |

`DuckDuckGoSearchRun` does not need an API key.

---

