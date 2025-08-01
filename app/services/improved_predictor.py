import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from scipy.stats import poisson
from ..models.match import MatchEvent, Prediction


class ImprovedPredictorService:
    """
    Improved prediction service with proper statistical modeling.
    """
    
    def __init__(self):
        # Realistic home advantage and goal rates from football analytics
        self.home_advantage = 0.3  # Home teams score ~30% more
        self.avg_goals_per_team = 1.35  # Average goals per team per match
        self.draw_probability_base = 0.28  # ~28% of matches end in draws
        
        # Team strength database (in real system, this would be from database)
        self.team_ratings = self._initialize_team_ratings()
    
    def _initialize_team_ratings(self) -> Dict[str, float]:
        """Initialize team strength ratings (1.0 = average)."""
        return {
            # Premier League examples - would be data-driven in production
            "Arsenal": 1.25, "Chelsea": 1.15, "Liverpool": 1.30, "Manchester City": 1.35,
            "Team A": 1.0, "Team B": 1.0,  # Default for test teams
        }
    
    def get_team_strength(self, team: str) -> float:
        """Get team strength rating, default to average if unknown."""
        return self.team_ratings.get(team, 1.0)
    
    def calculate_expected_goals(
        self, 
        home_team: str, 
        away_team: str, 
        minute: int, 
        current_score_diff: int
    ) -> Tuple[float, float]:
        """
        Calculate expected goals using team strength and match context.
        """
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)
        
        # Base expected goals adjusted for team strength
        home_xg = self.avg_goals_per_team * home_strength * (1 + self.home_advantage)
        away_xg = self.avg_goals_per_team * away_strength
        
        # Time adjustment - remaining time affects expected goals
        time_remaining = max(90 - minute, 0) / 90.0
        home_xg *= time_remaining
        away_xg *= time_remaining
        
        # Score state adjustment - teams trailing tend to attack more
        if current_score_diff > 0:  # Home team leading
            home_xg *= 0.8  # Defensive approach
            away_xg *= 1.3  # More attacking
        elif current_score_diff < 0:  # Away team leading
            home_xg *= 1.3  # More attacking  
            away_xg *= 0.8  # Defensive approach
        
        return home_xg, away_xg
    
    def calculate_match_probabilities(
        self, 
        home_xg: float, 
        away_xg: float, 
        max_goals: int = 5
    ) -> Tuple[float, float, float]:
        """
        Calculate win/draw probabilities using proper Poisson distribution.
        """
        win_home = 0.0
        win_away = 0.0
        draw = 0.0
        
        # Calculate probabilities for all realistic score combinations
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact score
                prob_score = (
                    poisson.pmf(home_goals, home_xg) * 
                    poisson.pmf(away_goals, away_xg)
                )
                
                if home_goals > away_goals:
                    win_home += prob_score
                elif away_goals > home_goals:
                    win_away += prob_score
                else:
                    draw += prob_score
        
        # Normalize to ensure probabilities sum to 1
        total = win_home + win_away + draw
        if total > 0:
            win_home /= total
            win_away /= total
            draw /= total
        
        return win_home, win_away, draw
    
    async def generate_prediction(self, match_event: MatchEvent) -> Prediction:
        """Generate improved prediction using proper statistical modeling."""
        
        # Calculate expected goals
        current_score_diff = match_event.home_goals - match_event.away_goals
        home_xg, away_xg = self.calculate_expected_goals(
            match_event.home_team,
            match_event.away_team,
            match_event.minute,
            current_score_diff
        )
        
        # Calculate match outcome probabilities
        win_prob_home, win_prob_away, draw_prob = self.calculate_match_probabilities(
            home_xg, away_xg
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
        """Calculate expected value with improved accuracy."""
        if market_odds <= 1.0 or prediction_prob <= 0:
            return -1.0
        
        # Expected value = (probability * odds) - 1
        return (prediction_prob * market_odds) - 1.0
    
    def calculate_implied_probability(self, odds: float) -> float:
        """Convert odds to implied probability."""
        if odds <= 1.0:
            return 1.0
        return 1.0 / odds
    
    def calculate_market_edge(
        self, 
        prediction_prob: float, 
        market_odds: float,
        overround_adjustment: float = 0.05
    ) -> float:
        """
        Calculate true edge accounting for bookmaker overround.
        """
        implied_prob = self.calculate_implied_probability(market_odds)
        
        # Adjust for typical bookmaker margin (5-10%)
        fair_implied_prob = implied_prob * (1 + overround_adjustment)
        
        # True edge is our prediction vs fair market probability
        return prediction_prob - fair_implied_prob
    
    async def analyze_match_value(
        self, match_event: MatchEvent
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Comprehensive value analysis for betting decisions.
        Returns: (ev_home, ev_away, analysis_metrics)
        """
        prediction = await self.generate_prediction(match_event)
        
        ev_home = self.calculate_expected_value(
            prediction.win_prob_home, match_event.market_odds_home
        )
        ev_away = self.calculate_expected_value(
            prediction.win_prob_away, match_event.market_odds_away
        )
        
        # Calculate market edges
        edge_home = self.calculate_market_edge(
            prediction.win_prob_home, match_event.market_odds_home
        )
        edge_away = self.calculate_market_edge(
            prediction.win_prob_away, match_event.market_odds_away
        )
        
        analysis = {
            "prediction_home": prediction.win_prob_home,
            "prediction_away": prediction.win_prob_away,
            "implied_prob_home": self.calculate_implied_probability(match_event.market_odds_home),
            "implied_prob_away": self.calculate_implied_probability(match_event.market_odds_away),
            "market_edge_home": edge_home,
            "market_edge_away": edge_away,
            "confidence_home": abs(edge_home) * prediction.win_prob_home,
            "confidence_away": abs(edge_away) * prediction.win_prob_away,
        }
        
        return ev_home, ev_away, analysis


class CalibrationService:
    """
    Service to calibrate model predictions against historical results.
    """
    
    def __init__(self):
        self.prediction_history = []
        self.actual_results = []
    
    def add_result(self, prediction: float, actual_outcome: bool):
        """Add a prediction and actual result for calibration."""
        self.prediction_history.append(prediction)
        self.actual_results.append(1.0 if actual_outcome else 0.0)
    
    def calculate_calibration_score(self) -> float:
        """
        Calculate Brier score - lower is better (0 = perfect).
        Brier Score = average((prediction - outcome)^2)
        """
        if len(self.prediction_history) < 10:
            return 0.5  # Default for insufficient data
        
        squared_errors = [
            (pred - actual) ** 2 
            for pred, actual in zip(self.prediction_history, self.actual_results)
        ]
        return sum(squared_errors) / len(squared_errors)
    
    def get_calibration_adjustment(self) -> float:
        """
        Get adjustment factor based on historical calibration.
        Values > 1.0 mean model is underconfident, < 1.0 means overconfident.
        """
        if len(self.prediction_history) < 20:
            return 1.0
        
        avg_prediction = sum(self.prediction_history) / len(self.prediction_history)
        avg_actual = sum(self.actual_results) / len(self.actual_results)
        
        if avg_prediction == 0:
            return 1.0
        
        return avg_actual / avg_prediction