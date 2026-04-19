import json
from src.llm_client import ask_llm
from src.prompts.templates import SYSTEM_PROMPT, REPORT_PROMPT
from src.output_models import validate_report_json


class ReportAgent:
    """Generates the final structured engagement report."""

    def generate(self, player_data: dict, churn_prob: float, analysis: str, strategies: dict) -> dict:
        """
        Args:
            player_data: dict of player features
            churn_prob: churn probability (0-100)
            analysis: behavior summary from AnalyzerAgent
            strategies: strategies dict from StrategyAgent
        Returns:
            dict with player_profile, risk_analysis, recommendations, disclaimers
        """
        data_str = ", ".join(f"{k}={v}" for k, v in player_data.items())

        prompt = REPORT_PROMPT.format(
            player_data=data_str,
            churn_prob=round(churn_prob, 1),
            analysis=analysis,
            strategies=json.dumps(strategies, indent=2)
        )

        response = ask_llm(prompt, system_msg=SYSTEM_PROMPT)

        try:
            raw_report = json.loads(response)
            return validate_report_json(raw_report)
        except json.JSONDecodeError:
            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                raw_report = json.loads(response[start:end])
                return validate_report_json(raw_report)
            except (ValueError, json.JSONDecodeError):
                return {
                    "player_profile": {"summary": analysis},
                    "risk_analysis": {
                        "level": "High" if churn_prob >= 70 else "Medium" if churn_prob >= 40 else "Low",
                        "trend": "↓ Declining",
                        "red_flag": "Unable to generate detailed analysis"
                    },
                    "recommendations": strategies.get("strategies", []),
                    "disclaimers": "AI-generated predictions. Use as supplementary decision support only."
                }
