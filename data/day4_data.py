TOOL_BUILDING_CODE = '''from smolagents import tool
import pandas as pd, requests

df = pd.read_csv("cities_latlng.csv")

@tool
def get_latlng(city: str) -> any:
  """
  Get the latitude, longitude and altitude of a city.

  Args:
    city: The name of the city.
  """
  r = df.query(f"city.str.lower() == '{city.lower()}'")
  return { 'city': city, 'latitude': r.iloc[0]['latitude'], 'longitude': r.iloc[0]['longitude'] }

@tool
def get_temperature(latitude: float, longitude: float) -> any:
  """
  Get the temperature at the given latitude and longitude.

  Args:
    latitude: The latitude of the location.
    longitude: The longitude of the location.
  """
  url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
  resp = requests.get(url)
  j = resp.json()
  return { "temperature": j['current']['temperature_2m'] }'''

STATEFUL_TOOL_CODE = '''import sqlite3, threading
from smolagents import Tool

class SQLiteTool(Tool):
  name = "albums"
  description = "Perform SQL queries on the album_track table."
  inputs = {"query": {"type": "string", "description": "A valid SQL query.", "required": True}}
  output_type = "string"

  def __init__(self, db_file):
    self.db_file = db_file
    self.setup()

  def setup(self):
    super().setup()
    self.conn = sqlite3.connect(self.db_file)
    self.lock = threading.Lock()

  def forward(self, query: str) -> list:
    with self.lock:
        cursor = self.conn.cursor()
        rows = cursor.execute(query)
        return rows.fetchall()

db = SQLiteTool("/content/data/chinook_sqlite.sqlite")'''

CODE_AGENT_VS_TOOL_CALLING_AGENT = [
    {"aspect": "Action format", "codeAgent": "Python code snippet, between <code> and </code> tags", "toolCalling": "Structured JSON tool calls via the provider's native function-calling API"},
    {"aspect": "How tools appear to the LLM", "codeAgent": "As Python functions it can call in code", "toolCalling": "As JSON schemas (name, description, parameter types)"},
    {"aspect": "Tool calls per step", "codeAgent": "Any number, with loops/conditionals/nested calls", "toolCalling": "One or more; each result goes back to the LLM before reuse"},
    {"aspect": "State between steps", "codeAgent": "Python variables persist across steps", "toolCalling": "Only what is in the message history"},
    {"aspect": "Best for", "codeAgent": "Problem solving, chaining tools, data wrangling", "toolCalling": "Dispatchers, simple independent API calls, strict validation"},
]

TRAVEL_AGENT_MCP_CODE = '''from mcp import StdioServerParameters
from smolagents import GradioUI, MCPClient, ToolCallingAgent, OpenAIServerModel

airbnb_params = StdioServerParameters(command="npx", args=["-y", "@openbnb/mcp-server-airbnb"])
fli_params = StdioServerParameters(command="../.venv/bin/fli-mcp")

airbnb_client = MCPClient(airbnb_params, structured_output=True)
fli_client = MCPClient(fli_params, structured_output=True)

llm_model = OpenAIServerModel(model_id="gpt-5.6-luna", api_key=OPENAI_API_KEY)

agent = ToolCallingAgent(
        tools=[*airbnb_client.get_tools(), *fli_client.get_tools()],
        model=llm_model,
        max_tool_threads=2, max_steps=15,
        add_base_tools=True,
        planning_interval=5)

ui = GradioUI(agent, reset_agent_memory=False)
ui.launch(share=False, server_name="0.0.0.0", server_port=7890)'''

PLANNING_HITL_CODE = '''from smolagents import FinalAnswerStep, MCPClient, PlanningStep, ToolCallingAgent, OpenAIServerModel

def review_plan(step: PlanningStep, agent: ToolCallingAgent) -> None:
    print(f"*** Proposed plan: {step.plan}")
    choice = input("[A]pprove, [E]dit, [C]ancel: ").strip().lower()
    if choice == "c":
        raise Exception("User rejected this plan")

agent = ToolCallingAgent(
        tools=[*airbnb_client.get_tools(), *fli_client.get_tools()],
        model=llm_model,
        max_tool_threads=2, max_steps=50,
        planning_interval=5,
        step_callbacks={PlanningStep: review_plan})

result = agent.run("Plan a 3 day trip to Tokyo, departing Singapore next Sunday.")'''

CART_POLE_CODE = '''import gymnasium as gym
import numpy as np

GAMMA = 0.99
ALPHA = 0.001
DECAY = 0.002
EPISODES = 2000

env = gym.make('CartPole-v1')
weights = np.zeros((4, 2))
bias = np.zeros(2)

def predict_q_values(st):
    return np.dot(st, weights) + bias

def policy(st, eps):
    if np.random.random() > eps:
        return int(np.argmax(predict_q_values(st)))
    return np.random.randint(0, 2)

for e in range(EPISODES):
    eps = 0.02 + (1.0 - 0.02) * np.exp(-e * DECAY)
    state, _ = env.reset()
    action = policy(state, eps)
    done = False
    total_reward = 0
    while not done:
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            td_target = -1.0 if terminated and total_reward < 500 else reward
        else:
            next_action = policy(new_state, eps)
            td_target = reward + GAMMA * np.amax(predict_q_values(new_state))
        current_prediction = predict_q_values(state)[action]
        td_error = td_target - current_prediction
        weights[:, action] += ALPHA * td_error * state
        bias[action] += ALPHA * td_error
        if not done:
            action, state = next_action, new_state'''

MEMORY_RESET_NOTE = (
    "The reset parameter controls whether an agent remembers past turns. agent.run() defaults to "
    "reset=True — every call starts fresh. Passing reset=False makes the next run() see the "
    "previous run's full message history, enabling follow-up questions."
)

CONTEXT_ENGINEERING_TECHNIQUES = [
    {"name": "Write", "desc": "Persist information outside the immediate context window (scratchpads, memory stores)."},
    {"name": "Select", "desc": "Pull only the most relevant subset of available context into the prompt."},
    {"name": "Compress", "desc": "Summarize or distill long context to fit within token budgets."},
    {"name": "Isolate", "desc": "Partition context across sub-agents so unrelated information doesn't interfere."},
]

AGENT_COMPONENTS = [
    {"name": "Base LLM", "desc": "The underlying reasoning/generation engine driving the agent's decisions."},
    {"name": "Tools", "desc": "External functions, APIs, or retrieval systems the agent can invoke."},
    {"name": "Decision Framework", "desc": "Logic for choosing actions — ReAct-style reasoning, planning, or routing."},
    {"name": "Memory", "desc": "Short-term (conversation) and long-term (persisted) state the agent can draw on."},
    {"name": "Goal-oriented behavior", "desc": "The agent pursues an objective across multiple steps rather than responding once."},
]

DAY4_RESOURCES = [
    {"title": "smolagents documentation", "url": "https://huggingface.co/docs/smolagents",
     "desc": "HuggingFace's lightweight agent framework.",
     "headings": ["CodeAgent", "ToolCallingAgent", "Planning", "MCP integration"],
     "summary": "smolagents is deliberately minimal: an agent is a short ReAct loop around a model and a list of tools.",
     "howTo": "Default to ToolCallingAgent for small/local models or simple dispatch; CodeAgent for chaining computation."},
    {"title": "Model Context Protocol (MCP)", "url": "https://modelcontextprotocol.io/",
     "desc": "Open standard for connecting agents to tools and data sources.",
     "headings": ["What is MCP", "Why MCP matters"],
     "summary": "MCP is a standard protocol so a tool integration is built once and reused across any MCP-supporting agent framework.",
     "howTo": "Point smolagents' MCPClient at an MCP server's StdioServerParameters to launch it as a subprocess."},
    {"title": "Open-Meteo API", "url": "https://open-meteo.com/",
     "desc": "Free, keyless weather API — powers the live temperature lookup tool.",
     "headings": ["Current weather", "No API key required"],
     "summary": "A free weather API with no signup/API key and permissive CORS.",
     "howTo": "GET https://api.open-meteo.com/v1/forecast?latitude=...&longitude=...&current=temperature_2m."},
    {"title": "Gymnasium — CartPole-v1", "url": "https://gymnasium.farama.org/environments/classic_control/cart_pole/",
     "desc": "The reinforcement-learning environment behind the Q-learning demo.",
     "headings": ["Observation space", "Rewards"],
     "summary": "A 4D continuous observation, 2 discrete actions, +1 reward per timestep survived.",
     "howTo": "A tabular Q-table can't index a continuous 4D state, so Q(s,a) is approximated linearly."},
    {"title": "LangChain — Context Engineering", "url": "https://www.langchain.com/blog/context-engineering-for-agents",
     "desc": "Write/Select/Compress/Isolate framework for managing agent context.",
     "headings": ["Write Context", "Select Context", "Compress Context", "Isolate Context"],
     "summary": "A context window is finite, and long-running agents accumulate tokens that cause context poisoning/distraction.",
     "howTo": "Pick one field in your agent's state and have a tool call append structured notes to it after each step."},
]
