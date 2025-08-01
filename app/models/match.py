from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class MatchEvent(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    minute: int
    home_goals: int
    away_goals: int
    market_odds_home: float
    market_odds_away: float
    timestamp: datetime


class Prediction(BaseModel):
    match_id: str
    win_prob_home: float
    win_prob_away: float
    timestamp: datetime


class Bet(BaseModel):
    match_id: str
    selection: str  # "home" or "away"
    stake: float
    odds_taken: float
    expected_value: float
    timestamp: datetime
    result: Optional[str] = None  # "win", "loss", or None if pending
    payout: Optional[float] = None


class TradingMetrics(BaseModel):
    total_bets: int
    winning_bets: int
    total_staked: float
    total_payout: float
    roi: float
    win_rate: float
    average_odds: float
    timestamp: datetime


class LatencyMetrics(BaseModel):
    match_id: str
    ingestion_time: datetime
    prediction_time: datetime
    decision_time: datetime
    end_to_end_latency_ms: float