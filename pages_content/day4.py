import streamlit as st

from components.code_block import code_block
from components.evidence import evidence
from components.real_world_example import real_world_example
from components.dashboard_tabs import dashboard_tabs, stat_row, resource_list
from components.sandboxes import (
    pretrain_loss_sandbox, scaling_law_sandbox, tool_building_sandbox,
    routing_sandbox, workflow_pattern_sandbox, cartpole_sandbox,
)
from data.day4_data import (
    TOOL_BUILDING_CODE, STATEFUL_TOOL_CODE, CODE_AGENT_VS_TOOL_CALLING_AGENT,
    TRAVEL_AGENT_MCP_CODE, PLANNING_HITL_CODE, CART_POLE_CODE, MEMORY_RESET_NOTE,
    CONTEXT_ENGINEERING_TECHNIQUES, AGENT_COMPONENTS, DAY4_RESOURCES,
)


def render():
    st.title("Day 4 — Pre-training & Agentic AI")
    st.caption(
        "Module 4.1: pre-training vs fine-tuning. Module 4.2: Agentic AI with smolagents — building "
        "custom tools, CodeAgent vs ToolCallingAgent, planning with human-in-the-loop, MCP-based agents, "
        "and a reinforcement-learning case study (CartPole Q-learning)."
    )

    stat_row([
        {"label": "SEA-LION pretrain tokens", "value": "200B"},
        {"label": "Chinchilla D/N ratio", "value": "~20x"},
        {"label": "Agent base class", "value": "MultiStepAgent"},
        {"label": "RL state dims", "value": "4 (continuous)"},
    ])

    st.header("Pre-training vs fine-tuning")
    st.write("Pre-training builds a model's general language ability from massive unlabeled corpora; "
             "fine-tuning (Day 2's LoRA workshop) specializes an already pre-trained model.")
    evidence("course", "Day 4 slides, p.10 — SEA-LION v3 case study",
             "SEA-LION v3 8B/70B: continued pre-trained on 200B tokens of SEA-language data, then instruction-tuned.")
    evidence("external", "Hoffmann et al. 2022 — not in Day 4 slides", "The deck has no loss function/scaling-law formula anywhere.")

    st.header("Agentic AI dashboard")

    def _pretrain():
        pretrain_loss_sandbox()
        scaling_law_sandbox()
        real_world_example("scaling-law decisions real labs make", [
            {"label": "SEA-LION's choice", "scenario": "AI Singapore wants a strong SEA-language model with a fixed compute budget.",
             "outcome": "Continue-pretraining Llama 3.1 on 200B tokens of SEA data instead of training from scratch."},
            {"label": "GPT-3's mistake, Chinchilla's fix", "scenario": "GPT-3 (175B, 300B tokens) and Chinchilla (70B, 1.4T tokens) spent similar compute.",
             "outcome": "Chinchilla beat GPT-3 — GPT-3 over-invested in parameters relative to data."},
        ])

    def _concepts():
        st.write("**What is an AI Agent?**")
        st.table(AGENT_COMPONENTS)
        st.write("**Context Engineering**")
        st.table(CONTEXT_ENGINEERING_TECHNIQUES)
        real_world_example("agent vs plain chatbot, in production", [
            {"label": "Travel booking", "scenario": 'A user says "book me a flight to Tokyo next Friday under $800."',
             "outcome": "An agent with search tools actually calls APIs and executes the booking — advice vs action."},
            {"label": "Why context engineering matters", "scenario": "A coding agent accumulates 50 tool-call results in a long session.",
             "outcome": "Compress and Isolate keep the agent from running out of context or getting confused."},
        ])

    def _tooluse():
        st.write("`@tool` turns a type-hinted Python function into a tool the LLM can call.")
        code_block(TOOL_BUILDING_CODE, "chuk_day04-tools-blank.ipynb — @tool functions")
        tool_building_sandbox()
        st.write("Stateful tools (holding a DB connection, a lock) subclass `Tool` instead:")
        code_block(STATEFUL_TOOL_CODE, "chuk_day04-tools-blank.ipynb — SQLiteTool(Tool)")
        real_world_example("deciding when to call a tool", [
            {"label": "Stock price question", "scenario": "A fact that changes every minute and isn't in training data.",
             "outcome": "The model calls a stock-price tool — tool_use → execute → tool_result loop."},
            {"label": "Arithmetic question", "scenario": '"What is 847 × 23?"',
             "outcome": "A well-prompted agent answers directly instead of wasting a tool round-trip."},
        ])

    def _codevtool():
        st.write("Both inherit the same `MultiStepAgent` base — they differ in how the agent takes action each step:")
        st.table(CODE_AGENT_VS_TOOL_CALLING_AGENT)
        st.caption(MEMORY_RESET_NOTE)
        real_world_example("picking the right agent type", [
            {"label": "Data analysis agent", "scenario": "Find the 3 products with the biggest sales drop from a 50,000-row CSV.",
             "outcome": "CodeAgent — writing a pandas filter/groupby is more reliable than chaining many fixed tool calls."},
            {"label": "Customer-facing booking agent", "scenario": "An agent calling exactly 2 vetted, rate-limited APIs.",
             "outcome": "ToolCallingAgent — a fixed, auditable action space is safer for user-facing agents."},
        ])

    def _routing():
        routing_sandbox()
        workflow_pattern_sandbox()
        real_world_example("routing and workflows in real systems", [
            {"label": "Tiered support routing", "scenario": 'A password reset question vs an architecture trade-off question.',
             "outcome": "Simple → cheap-model path; complex → expensive-model path."},
            {"label": "Content moderation pipeline", "scenario": "Every post needs text/image/spam checks simultaneously.",
             "outcome": "Parallelization — run checks concurrently, then aggregate."},
        ])

    def _planning():
        st.write("With `planning_interval` set, the agent periodically produces a `PlanningStep` that a human can review before it acts.")
        code_block(PLANNING_HITL_CODE, "cli_travelagent.py — review_plan + planning_interval")
        real_world_example("why human-in-the-loop isn't optional for some agents", [
            {"label": "Financial transaction agent", "scenario": "An agent plans to transfer $5,000 — real money movement.",
             "outcome": "planning_interval surfaces the plan for human approval before any transfer executes."},
            {"label": "Low-stakes research agent", "scenario": "An agent plans to search and summarize papers — read-only, reversible.",
             "outcome": "No HITL gate needed — the cost of a wrong step is just a wasted search."},
        ])

    def _mcp():
        st.write("`ToolCallingAgent` wired to two independent MCP servers, each a local subprocess over stdio:")
        code_block(TRAVEL_AGENT_MCP_CODE, "travelagent.py — MCP + ToolCallingAgent + GradioUI")
        st.caption("**Model Context Protocol (MCP)** is an open standard for connecting agents to external tools/data sources in a model-agnostic way.")
        real_world_example("MCP solving a real integration headache", [
            {"label": "Before MCP", "scenario": "5 agents each need custom integration code for Slack/CRM/ticketing — 15 integrations.",
             "outcome": "N agents × M tools = N×M custom integrations, each breaking independently on API changes."},
            {"label": "With MCP", "scenario": "The same 5 agents, but each tool ships one MCP server.",
             "outcome": "N agents × M tools = N+M total integration points."},
        ])

    def _rl():
        st.write("A different kind of agent: no LLM at all. The agent learns a policy purely from trial and error.")
        code_block(CART_POLE_CODE, "cart_pole_qlearning.py — linear Q-learning")
        cartpole_sandbox()
        real_world_example("RL beyond the simulator", [
            {"label": "Data center cooling", "scenario": "Google DeepMind needed to control data-center cooling without overheating servers.",
             "outcome": "An RL agent learned a cooling policy, reportedly cutting energy use by ~40% — same reward-driven loop as CartPole."},
            {"label": "RLHF — training the chat model itself", "scenario": "How does GPT-OSS/Qwen get tuned to prefer helpful responses?",
             "outcome": "Human preference ratings train a reward model, then the LLM updates via policy gradient (PPO)."},
        ])

    dashboard_tabs([
        {"label": "Pre-training math", "badge": True, "render": _pretrain},
        {"label": "Agent concepts", "render": _concepts},
        {"label": "Tool-use loop", "badge": True, "render": _tooluse},
        {"label": "CodeAgent vs ToolCalling", "render": _codevtool},
        {"label": "Routing & patterns", "badge": True, "render": _routing},
        {"label": "Planning & HITL", "render": _planning},
        {"label": "MCP agent", "render": _mcp},
        {"label": "RL: CartPole", "badge": True, "render": _rl},
    ])

    st.header("This is what powers the search bar →")
    st.write(
        "The search bar on top of every page runs the exact pattern taught above: a guardrail check "
        "gates every message against this site's own topics first, then an in-scope message is sent "
        "to a small free model (Groq) with a web_search tool it can use only to support an in-scope "
        "answer — the same detect-tool-call → execute → return-result loop as ToolCallingAgent above, "
        "just in Python instead of smolagents."
    )

    st.header("Resources")
    resource_list(DAY4_RESOURCES)
