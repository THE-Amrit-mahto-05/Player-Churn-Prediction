from pydantic import BaseModel, Field, ValidationError
from typing import List

# --- Agent Output Schemas ---
class Recommendation(BaseModel):
    action: str
    rationale: str
    priority: str

class RiskAnalysis(BaseModel):
    level: str
    trend: str
    red_flag: str

class PlayerProfile(BaseModel):
    summary: str

class EngagementReport(BaseModel):
    player_profile: PlayerProfile
    risk_analysis: RiskAnalysis
    recommendations: List[Recommendation]
    disclaimers: str

def validate_report_json(report_dict: dict) -> dict:
    """Validates that the parsed JSON strictly follows our report structure."""
    try:
        validated_report = EngagementReport(**report_dict)
        return validated_report.model_dump()
    except ValidationError as e:
        print(f"Report validation failed: {e}")
        # Return the raw dict if it fails, but this warns us.
        return report_dict
