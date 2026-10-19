"""Mock tools with fixed, deterministic outputs.

Because these never change between calls, any variation across runs
comes from the agent, not the environment.
"""
import ast
import operator

from langchain_core.tools import tool

WEATHER = {
    "denver": {"temp_f": 54, "condition": "partly cloudy"},
    "boston": {"temp_f": 47, "condition": "rainy"},
    "miami": {"temp_f": 82, "condition": "sunny"},
}

PRODUCTS = [
    {"id": "p101", "name": "UltraBook 14", "category": "laptop", "price_usd": 1199.00},
    {"id": "p102", "name": "WorkBook 15", "category": "laptop", "price_usd": 899.00},
    {"id": "p103", "name": "GameBook 17", "category": "laptop", "price_usd": 1599.00},
    {"id": "p201", "name": "PhoneX", "category": "phone", "price_usd": 799.00},
    {"id": "p202", "name": "PhoneMini", "category": "phone", "price_usd": 499.00},
]

RATES = {("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("USD", "JPY"): 150.0}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    data = WEATHER.get(city.strip().lower())
    return data or {"error": f"No weather data for '{city}'"}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(12 + 3) * 4'. Supports + - * / **."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"error: {e}"


@tool
def search_products(query: str) -> list:
    """Search the product catalog. Matches against product name and category.
    Returns a list of products with id, name, category, and price in USD."""
    q = query.strip().lower()
    hits = [p for p in PRODUCTS
            if q in p["name"].lower() or q in p["category"].lower()
            or q.rstrip("s") in p["category"]]
    return hits or []


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Get the exchange rate between two currencies (3-letter codes, e.g. USD, EUR)."""
    key = (from_currency.upper(), to_currency.upper())
    if key in RATES:
        return {"from": key[0], "to": key[1], "rate": RATES[key]}
    return {"error": f"No rate for {key[0]}->{key[1]}"}


ALL_TOOLS = [get_weather, calculator, search_products, get_exchange_rate]
