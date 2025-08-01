import numpy as np
from datetime import datetime
from typing import Tuple
from ..models.match import MatchEvent, Prediction


def calculate_win_probability_poisson(
    home_goals: int, away_goals: int, minute: int
) -> Tuple[float, float]:
    """
    Simple Poisson-based model for win probability calculation.
    In production, this would be replaced with a trained ML model.
    """
    base_rate_home = 1.5
    base_rate_away = 1.2
    
    # Adjust rates based on current match state
    time_factor = min(minute / 90.0, 1.0)
    goal_diff = home_goals - away_goals
    
    # Dynamic lambda based on current state
    lambda_home = base_rate_home * (1 + time_factor * 0.2) + max(goal_diff * 0.3, 0)
    lambda_away = base_rate_away * (1 + time_factor * 0.2) + max(-goal_diff * 0.3, 0)
    
    # Calculate probabilities using Poisson distribution
    prob_home_no_score = np.exp(-lambda_away)
    prob_away_no_score = np.exp(-lambda_home)
    
    # Normalize probabilities (simplified approach)
    total_prob = prob_home_no_score + prob_away_no_score
    win_prob_home = prob_home_no_score / total_prob
    win_prob_away = prob_away_no_score / total_prob
    
    return win_prob_home, win_prob_away


class PredictorService:
    """
    Service for generating match outcome predictions.
    Can be extended to use ML models trained on historical data.
    """
    
    def __init__(self):
        self.model_type = "poisson"
        
    async def generate_prediction(self, match_event: MatchEvent) -> Prediction:
        """Generate prediction based on current match state."""
        
        win_prob_home, win_prob_away = calculate_win_probability_poisson(
            match_event.home_goals,
            match_event.away_goals,
            match_event.minute
        )
        
        return Prediction(
            match_id=match_event.match_id,
            win_prob_home=win_prob_home,
            win_prob_away=win_prob_away,
            timestamp=datetime.utcnow()
        )
    
    def calculate_expected_value(
        self, prediction_prob: float, market_odds: float
    ) -> float:
        """Calculate expected value of a bet."""
        implied_prob = 1.0 / market_odds
        return (prediction_prob * market_odds) - 1.0
    
    async def analyze_match_value(
        self, match_event: MatchEvent
    ) -> Tuple[float, float]:
        """Analyze expected value for both home and away bets."""
        prediction = await self.generate_prediction(match_event)
        
        ev_home = self.calculate_expected_value(
            prediction.win_prob_home, match_event.market_odds_home
        )
        ev_away = self.calculate_expected_value(
            prediction.win_prob_away, match_event.market_odds_away
        )
        
        return ev_home, ev_away