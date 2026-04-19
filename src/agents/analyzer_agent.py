from src.llm_client import ask_llm
from src.prompts.templates import SYSTEM_PROMPT, ANALYZER_PROMPT


class AnalyzerAgent:
    """Analyzes player behavior and identifies churn red flags."""

    def analyze(self, player_data: dict, churn_prob: float) -> str:
        """
        Args:
            player_data: dict of player features (e.g., PlayTimeHours, SessionsPerWeek)
            churn_prob: churn probability from ML model (0-100)
        Returns:
            Behavior summary string (3-4 sentences)
        """
 
        data_str = ", ".join(f"{k}={v}" for k, v in player_data.items())

        prompt = ANALYZER_PROMPT.format(
            player_data=data_str,
            churn_prob=round(churn_prob, 1)
        )

        return ask_llm(prompt, system_msg=SYSTEM_PROMPT)
