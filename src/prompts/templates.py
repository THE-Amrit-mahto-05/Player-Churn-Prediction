SYSTEM_PROMPT = """You are a Senior Gaming Retention Expert. You provide data-driven, 
ethical, and actionable advice. 
CRITICAL: Only use provided data. Do not invent player actions.
If data is insufficient, say "Insufficient data to determine..." """

ANALYZER_PROMPT = """Analyze this player's behavior:
DATA: {player_data}
CHURN PROBABILITY: {churn_prob}%

TASK: 
1. Summarize engagement patterns.
2. Identify engagement trend direction (Increasing ↑, Decreasing ↓, or Stable →).
3. Identify one specific 'Red Flag' contributing to churn risk.

FORMAT: 3-4 sentences maximum."""

STRATEGY_PROMPT = """Based on this player analysis: 
{analysis}

CHURN RISK LEVEL: {risk_level}
GAME GENRE: {genre}

INDUSTRY CONTEXT (Use these proven strategies if relevant to the player):
{context}

TASK: Generate 3 personalized retention strategies.
FORMAT: Respond ONLY with a valid JSON object:
{{
  "strategies": [
    {{
      "action": "Short title",
      "rationale": "Why this works for this player",
      "priority": "High/Medium/Low"
    }}
  ]
}}"""

REPORT_PROMPT = """Combine the analysis and strategies into a final executive report.
PLAYER DATA: {player_data}
CHURN PROBABILITY: {churn_prob}%
ANALYSIS: {analysis}
STRATEGIES: {strategies}

TASK: Generate a structured report.
FORMAT: Respond ONLY with a valid JSON object:
{{
  "player_profile": {{ "summary": "2-3 sentence overview" }},
  "risk_analysis": {{ "level": "High/Medium/Low", "trend": "↑/↓/→", "red_flag": "..." }},
  "recommendations": [
    {{ "action": "...", "rationale": "...", "priority": "High/Medium/Low" }}
  ],
  "disclaimers": "Include AI accuracy and ethical usage notes here."
}}"""
