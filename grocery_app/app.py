import sys
import os
import asyncio
from typing import TypedDict, List
from dotenv import load_dotenv
import chromadb
import gradio as gr
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────
# LLM and Search
# ─────────────────────────────────────────
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)
search_tool = TavilySearch(max_results=3, topic="general")

# ─────────────────────────────────────────
# Simple in-memory grocery search (no ChromaDB)
# ─────────────────────────────────────────
grocery_items = [
    "chicken breast - protein, lean meat, 165 calories per 100g",
    "salmon fillet - protein, omega-3, healthy fats, 208 calories per 100g",
    "tofu firm - plant protein, vegan, low calorie, 76 calories per 100g",
    "eggs large - protein, vitamin D, 155 calories per 100g",
    "lentils - plant protein, fiber, iron, 116 calories per 100g",
    "chickpeas - plant protein, fiber, vegan, 164 calories per 100g",
    "greek yogurt - protein, probiotics, calcium, 59 calories per 100g",
    "spinach fresh - iron, vitamins, low calorie, 23 calories per 100g",
    "broccoli - vitamin C, fiber, anti-inflammatory, 34 calories per 100g",
    "sweet potato - complex carbs, vitamin A, fiber, 86 calories per 100g",
    "avocado - healthy fats, potassium, fiber, 160 calories per 100g",
    "bell peppers - vitamin C, antioxidants, low calorie, 31 calories per 100g",
    "kale - superfood, vitamins K C A, calcium, 49 calories per 100g",
    "carrots - beta carotene, fiber, vitamin A, 41 calories per 100g",
    "quinoa - complete protein, gluten free, fiber, 120 calories per 100g",
    "brown rice - complex carbs, fiber, gluten free, 216 calories per 100g",
    "oats rolled - fiber, beta glucan, heart healthy, 389 calories per 100g",
    "blueberries - antioxidants, vitamin C, low sugar, 57 calories per 100g",
    "banana - potassium, energy, natural sugar, 89 calories per 100g",
    "almonds - healthy fats, protein, vitamin E, 579 calories per 100g",
    "chia seeds - omega-3, fiber, plant protein, 486 calories per 100g",
    "almond milk - low calorie, vegan, calcium fortified, 17 calories per 100g",
    "olive oil - healthy fats, anti-inflammatory, 884 calories per 100g",
]

def search_grocery_inventory_simple(query: str) -> str:
    """Simple keyword search through grocery items."""
    query_words = query.lower().split()
    matches = []
    for item in grocery_items:
        item_lower = item.lower()
        score = sum(1 for word in query_words if word in item_lower)
        if score > 0:
            matches.append((score, item))
    matches.sort(reverse=True)
    top_matches = [item for _, item in matches[:5]]
    if not top_matches:
        top_matches = grocery_items[:5]
    return f"Found {len(top_matches)} matching items:\n" + "\n".join([f"- {item}" for item in top_matches])

# ─────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────
class AgentState(TypedDict):
    dietary_needs: str
    meal_plan: str
    health_critique: str
    search_results: List[str]
    ingredients: List[str]
    inventory_matches: List[str]
    cart: List[str]
    approved: bool
    iterations: int

# ─────────────────────────────────────────
# Grocery Search Tool
# ─────────────────────────────────────────
@tool
def search_grocery_inventory(query: str) -> str:
    """Search the grocery store inventory for items matching dietary needs."""
    return search_grocery_inventory_simple(query)

# ─────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────
def chef_agent(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are a top class home cooking chef.
        Your cooking style matches mom's cooking which is very personal and emotional.
        Help pick healthy ingredients to make an awesome meal that's quick, easy and tasty.
        Always list the specific ingredients needed at the end of your response."""),
        HumanMessage(content=f"Dietary needs: {state['dietary_needs']}")
    ])
    return {"meal_plan": response.content, "iterations": state.get("iterations", 0) + 1}

def dr_ganapathy_agent(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="""You are Dr. Advice Ganapathy, an experienced nutritionist.
        Review the proposed meal plan critically:
        1. Check if ingredients align with dietary needs
        2. Evaluate overall health quotient
        3. Consider appropriateness for time of day
        4. Flag any health concerns
        5. Suggest healthier alternatives where needed
        If nutritionally sound, start with 'APPROVED'."""),
        HumanMessage(content=f"""
Patient dietary needs: {state['dietary_needs']}
Proposed meal plan: {state['meal_plan']}
Please review and provide assessment.""")
    ])
    return {"health_critique": response.content}

def grocery_matcher_agent(state: AgentState) -> dict:
    ingredient_response = llm.invoke([
        SystemMessage(content="Extract only ingredient names from the meal plan. Return a simple comma separated list. No quantities or instructions."),
        HumanMessage(content=state["meal_plan"])
    ])
    ingredients = [i.strip() for i in ingredient_response.content.split(",")]
    inventory_matches = []
    for ingredient in ingredients[:8]:
        results = search_grocery_inventory.invoke(ingredient)
        inventory_matches.append(f"{ingredient}: {results.split(chr(10))[1] if chr(10) in results else results}")
    cart = [f"- {ingredient}" for ingredient in ingredients[:8]]
    return {"ingredients": ingredients, "inventory_matches": inventory_matches, "cart": cart}

def should_continue(state: AgentState) -> str:
    health_critique = state.get("health_critique", "")
    if "APPROVED" in health_critique or state.get("iterations", 0) >= 3:
        return "grocery"
    return "chef"

# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
workflow = StateGraph(AgentState)
workflow.add_node("chef_agent", chef_agent)
workflow.add_node("dr_ganapathy_agent", dr_ganapathy_agent)
workflow.add_node("grocery_matcher_agent", grocery_matcher_agent)
workflow.add_edge("chef_agent", "dr_ganapathy_agent")
workflow.add_edge("grocery_matcher_agent", END)
workflow.add_conditional_edges("dr_ganapathy_agent", should_continue, {
    "chef": "chef_agent",
    "grocery": "grocery_matcher_agent",
})
workflow.set_entry_point("chef_agent")
app = workflow.compile()

# ─────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────
def run_grocery_agent(dietary_input: str) -> tuple:
    if not dietary_input.strip():
        return "Please enter your dietary needs!", "", ""
    try:
        result = app.invoke({
            "dietary_needs": dietary_input,
            "meal_plan": "",
            "health_critique": "",
            "search_results": [],
            "ingredients": [],
            "inventory_matches": [],
            "cart": [],
            "approved": False,
            "iterations": 0
        })
        meal_plan = result.get("meal_plan", "No meal plan generated")
        health_critique = result.get("health_critique", "No review available")
        cart_items = result.get("cart", [])
        cart_text = "\n".join(cart_items) if cart_items else "No items in cart"
        return meal_plan, health_critique, cart_text
    except Exception as e:
        return f"Error: {str(e)}", "", ""

with gr.Blocks(title="Personal Grocery & Meal Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Personal Grocery & Meal Planning Agent
    ### Powered by Chef AI + Dr. Ganapathy Health Review
    *Tell me your dietary needs and I'll plan your perfect meal and shopping list!*
    """)
    with gr.Row():
        with gr.Column():
            dietary_input = gr.Textbox(
                label="What would you like to cook today?",
                placeholder="e.g. High protein vegan lunch, quick vegetarian dinner, healthy breakfast with eggs...",
                lines=3
            )
            submit_btn = gr.Button("Get My Meal Plan!", variant="primary", size="lg")
    gr.Markdown("---")
    with gr.Row():
        meal_output = gr.Markdown(label="Chef's Meal Plan")
    with gr.Row():
        health_output = gr.Markdown(label="Dr. Ganapathy's Review")
    with gr.Row():
        cart_output = gr.Textbox(label="Your Shopping Cart", lines=10, interactive=False)
    gr.Examples(
        examples=[
            ["High protein vegan meal for afternoon energy"],
            ["Quick vegetarian dinner ready in 20 minutes"],
            ["Healthy breakfast with eggs and plant based sides"],
            ["Low calorie lunch for weight loss"],
            ["Post workout meal with high protein"],
        ],
        inputs=dietary_input,
        label="Try these examples"
    )
    submit_btn.click(fn=run_grocery_agent, inputs=dietary_input, outputs=[meal_output, health_output, cart_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        auth=[("bala", "groceryai2026")]
    )