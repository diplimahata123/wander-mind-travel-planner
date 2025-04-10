# --- WanderMind: Full Travel Guide Project with User Input ---

# 📦 Imports
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# --- 1. Define State ---
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Chat memory"]
    city: str
    country: str
    interests: List[str]
    travel_dates: str
    budget: str
    itinerary: str

# --- 2. Load LLM ---
llm=ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

# --- 3. Define Agents ---
def memory_agent(state: PlannerState) -> PlannerState:
    return state  # Extend with ChromaDB lookup

def local_expert_agent(state: PlannerState) -> PlannerState:
    prompt = f" You are a travel expert. Provide safety, culture, and must-know info for visiting {state['city']}, {state['country']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)
    return state

def experience_curator_agent(state: PlannerState) -> PlannerState:
    prompt = f"Based on interests: {state['interests']}, plan activities in {state['city']} that match"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)
    return state

def weather_agent(state: PlannerState) -> PlannerState:
    response = AIMessage(content="Checked weather: Mostly sunny, 25°C")
    state["messages"].append(response)
    return state

def logistics_agent(state: PlannerState) -> PlannerState:
    prompt = f"Organize the activities into a logical 5-day itinerary for {state['city']} in {state['travel_dates']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["itinerary"] = response.content
    state["messages"].append(response)
    return state

def budget_agent(state: PlannerState) -> PlannerState:
    prompt = f"""
    Ensure this itinerary fits within a {state['budget']} budget.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)
    return state

def event_agent(state: PlannerState) -> PlannerState:
    response = AIMessage(content="Local Event: Tokyo Ramen Festa on Day 3")
    state["messages"].append(response)
    return state

# --- 4. Build Graph ---
graph = StateGraph(PlannerState)

graph.add_node("Memory", RunnableLambda(memory_agent))
graph.add_node("LocalExpert", RunnableLambda(local_expert_agent))
graph.add_node("ExperienceCurator", RunnableLambda(experience_curator_agent))
graph.add_node("Weather", RunnableLambda(weather_agent))
graph.add_node("Logistics", RunnableLambda(logistics_agent))
graph.add_node("Budget", RunnableLambda(budget_agent))
graph.add_node("Event", RunnableLambda(event_agent))

graph.set_entry_point("Memory")
graph.add_edge("Memory", "LocalExpert")
graph.add_edge("LocalExpert", "ExperienceCurator")
graph.add_edge("ExperienceCurator", "Weather")
graph.add_edge("Weather", "Logistics")
graph.add_edge("Logistics", "Budget")
graph.add_edge("Budget", "Event")
graph.add_edge("Event", END)

app = graph.compile()

# --- 5. User Input ---
print("Welcome to WanderMind: Your AI Travel Planner")
city = input("Enter your destination city:")
country = input("Enter the country:")
interests = input("List your interests (comma-separated):").split(",")
travel_dates = input("Enter your travel dates (e.g., July 2025):")
budget = input("Enter your budget level (low, medium, high):")

example_input: PlannerState = {
    "messages": [],
    "city": city.strip(),
    "country": country.strip(),
    "interests": [i.strip() for i in interests],
    "travel_dates": travel_dates.strip(),
    "budget": budget.strip(),
    "itinerary": ""
}

# --- 6. Run ---
result = app.invoke(example_input)

# --- 7. Print Output ---
print("\n--- Final Itinerary ---\n")
print(result['itinerary'])

print("\n--- Conversation Log ---\n")
for msg in result['messages']:
    print(msg.type.upper() + ":", msg.content)