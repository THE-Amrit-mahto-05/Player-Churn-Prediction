import json
from src.llm_client import ask_llm
from src.prompts.templates import SYSTEM_PROMPT, STRATEGY_PROMPT


class StrategyAgent:
    """Generates retention strategies based on player analysis."""

    def generate(self, analysis: str, churn_prob: float, genre: str = "Unknown") -> dict:
        """
        Args:
            analysis: behavior summary from AnalyzerAgent
            churn_prob: churn probability (0-100) to determine risk level
            genre: game genre (e.g., RPG, FPS, Strategy)
        Returns:
            dict with 'strategies' list (parsed from LLM JSON)
        """

        if churn_prob >= 70:
            risk_level = "High"
        elif churn_prob >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        prompt = STRATEGY_PROMPT.format(
            analysis=analysis,
            risk_level=risk_level,
            genre=genre
        )

        response = ask_llm(prompt, system_msg=SYSTEM_PROMPT)

        
        try:
            return json.loads(response)
        except json.JSONDecodeError:

            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                return json.loads(response[start:end])
            except (ValueError, json.JSONDecodeError):
                return {
                    "strategies": [
                        {
                            "action": "Unable to parse strategies",
                            "rationale": response[:200],
                            "priority": "Medium"
                        }
                    ]
                }
