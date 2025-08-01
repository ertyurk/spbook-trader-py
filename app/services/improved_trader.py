from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
from ..models.match import MatchEvent, Bet, Prediction, TradingMetrics
from .improved_predictor import ImprovedPredictorService


class ImprovedTradingStrategy:
    """Base class for improved trading strategies with better risk management."""
    
    def __init__(
        self, 
        min_edge: float = 0.03,
        min_confidence: float = 0.15,
        max_stake_percentage: float = 0.01,
        min_odds: float = 1.5,
        max_odds: float = 10.0
    ):
        self.min_edge = min_edge  # Minimum market edge required
        self.min_confidence = min_confidence  # Minimum confidence level
        self.max_stake_percentage = max_stake_percentage
        self.min_odds = min_odds  # Avoid betting on heavy favorites
        self.max_odds = max_odds  # Avoid betting on long shots
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float,
        analysis: Dict[str, float]
    ) -> Optional[Bet]:
        raise NotImplementedError


class ValueBettingStrategy(ImprovedTradingStrategy):
    """
    Conservative value betting strategy focusing on high-probability edges.
    """
    
    def __init__(
        self,
        min_edge: float = 0.05,
        min_confidence: float = 0.20,
        max_stake_percentage: float = 0.005,
        **kwargs
    ):
        super().__init__(
            min_edge=min_edge,
            min_confidence=min_confidence,
            max_stake_percentage=max_stake_percentage,
            **kwargs
        )
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float,
        analysis: Dict[str, float]
    ) -> Optional[Bet]:
        """Conservative betting only on high-confidence edges."""
        
        # Check home bet opportunity
        if (analysis["market_edge_home"] > self.min_edge and
            analysis["confidence_home"] > self.min_confidence and
            self.min_odds <= match_event.market_odds_home <= self.max_odds):
            
            # Scale stake by confidence
            confidence_multiplier = min(analysis["confidence_home"] / 0.3, 1.0)
            stake = bankroll * self.max_stake_percentage * confidence_multiplier
            
            return Bet(
                match_id=match_event.match_id,
                selection="home",
                stake=stake,
                odds_taken=match_event.market_odds_home,
                expected_value=analysis["market_edge_home"],
                timestamp=datetime.utcnow()
            )
        
        # Check away bet opportunity  
        if (analysis["market_edge_away"] > self.min_edge and
            analysis["confidence_away"] > self.min_confidence and
            self.min_odds <= match_event.market_odds_away <= self.max_odds):
            
            confidence_multiplier = min(analysis["confidence_away"] / 0.3, 1.0)
            stake = bankroll * self.max_stake_percentage * confidence_multiplier
            
            return Bet(
                match_id=match_event.match_id,
                selection="away", 
                stake=stake,
                odds_taken=match_event.market_odds_away,
                expected_value=analysis["market_edge_away"],
                timestamp=datetime.utcnow()
            )
        
        return None


class AdaptiveKellyStrategy(ImprovedTradingStrategy):
    """
    Kelly Criterion with risk adjustments and market efficiency filters.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            min_edge=0.03,
            min_confidence=0.1,
            max_stake_percentage=0.02,  # Cap Kelly at 2%
            **kwargs
        )
        self.kelly_multiplier = 0.25  # Quarter Kelly for safety
    
    def calculate_kelly_fraction(
        self, win_prob: float, odds: float, edge: float
    ) -> float:
        """Calculate Kelly fraction with safety constraints."""
        if win_prob <= 0 or odds <= 1.0 or edge <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = win_prob, q = 1 - win_prob
        b = odds - 1.0
        p = win_prob
        q = 1.0 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety multiplier
        kelly_fraction *= self.kelly_multiplier
        
        # Cap at maximum stake percentage
        return min(max(kelly_fraction, 0), self.max_stake_percentage)
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float,
        analysis: Dict[str, float]
    ) -> Optional[Bet]:
        """Kelly betting with market efficiency and risk filters."""
        
        # Home bet analysis
        if (analysis["market_edge_home"] > self.min_edge and
            analysis["confidence_home"] > self.min_confidence and
            self.min_odds <= match_event.market_odds_home <= self.max_odds):
            
            kelly_fraction = self.calculate_kelly_fraction(
                prediction.win_prob_home,
                match_event.market_odds_home,
                analysis["market_edge_home"]
            )
            
            if kelly_fraction > 0:
                stake = bankroll * kelly_fraction
                
                return Bet(
                    match_id=match_event.match_id,
                    selection="home",
                    stake=stake,
                    odds_taken=match_event.market_odds_home,
                    expected_value=analysis["market_edge_home"],
                    timestamp=datetime.utcnow()
                )
        
        # Away bet analysis
        if (analysis["market_edge_away"] > self.min_edge and
            analysis["confidence_away"] > self.min_confidence and
            self.min_odds <= match_event.market_odds_away <= self.max_odds):
            
            kelly_fraction = self.calculate_kelly_fraction(
                prediction.win_prob_away,
                match_event.market_odds_away,
                analysis["market_edge_away"]
            )
            
            if kelly_fraction > 0:
                stake = bankroll * kelly_fraction
                
                return Bet(
                    match_id=match_event.match_id,
                    selection="away",
                    stake=stake,
                    odds_taken=match_event.market_odds_away,
                    expected_value=analysis["market_edge_away"],
                    timestamp=datetime.utcnow()
                )
        
        return None


class ImprovedTradingService:
    """Enhanced trading service with improved prediction and risk management."""
    
    def __init__(
        self, 
        strategy: ImprovedTradingStrategy, 
        initial_bankroll: float = 10000.0,
        max_exposure_percentage: float = 0.1  # Max 10% of bankroll at risk
    ):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_exposure = initial_bankroll * max_exposure_percentage
        
        self.predictor = ImprovedPredictorService()
        self.bets: List[Bet] = []
        self.settled_bets: List[Bet] = []
        
        # Track performance
        self.total_staked = 0.0
        self.current_exposure = 0.0  # Outstanding bet stakes
        
    def calculate_current_exposure(self) -> float:
        """Calculate current exposure from active bets."""
        return sum(bet.stake for bet in self.bets if bet.result is None)
    
    async def process_match_event(self, match_event: MatchEvent) -> Optional[Bet]:
        """Process match event with improved analysis and risk management."""
        
        # Check exposure limits
        current_exposure = self.calculate_current_exposure()
        if current_exposure >= self.max_exposure:
            return None  # Skip betting due to exposure limits
        
        # Generate prediction and analysis
        ev_home, ev_away, analysis = await self.predictor.analyze_match_value(match_event)
        
        # Get prediction for strategy
        prediction = await self.predictor.generate_prediction(match_event)
        
        # Check if we should bet
        bet = self.strategy.should_bet(prediction, match_event, self.current_bankroll, analysis)
        
        if bet:
            # Final risk checks
            if bet.stake > self.current_bankroll * 0.02:  # Never risk more than 2%
                bet.stake = self.current_bankroll * 0.02
            
            if current_exposure + bet.stake > self.max_exposure:
                # Reduce stake to stay within exposure limits
                bet.stake = max(0, self.max_exposure - current_exposure)
                
            if bet.stake > 0:
                self.bets.append(bet)
                self.current_bankroll -= bet.stake
                self.total_staked += bet.stake
                self.current_exposure += bet.stake
                return bet
        
        return None
    
    def settle_bet(self, match_id: str, home_goals: int, away_goals: int) -> List[Bet]:
        """Settle bets with improved tracking."""
        
        if home_goals < 0 or away_goals < 0:
            raise ValueError("Goals cannot be negative")
        
        settled_bets_this_match = []
        remaining_bets = []
        
        for bet in self.bets:
            if bet.match_id == match_id:
                # Determine winner
                if home_goals > away_goals:
                    winner = "home"
                elif away_goals > home_goals:
                    winner = "away"
                else:
                    winner = "draw"
                
                # Settle bet
                if bet.selection == winner:
                    bet.result = "win"
                    bet.payout = bet.stake * bet.odds_taken
                    self.current_bankroll += bet.payout
                else:
                    bet.result = "loss"
                    bet.payout = 0.0
                
                # Reduce exposure
                self.current_exposure -= bet.stake
                
                self.settled_bets.append(bet)
                settled_bets_this_match.append(bet)
            else:
                remaining_bets.append(bet)
        
        self.bets = remaining_bets
        return settled_bets_this_match
    
    def get_trading_metrics(self) -> TradingMetrics:
        """Get enhanced trading metrics."""
        
        if not self.settled_bets:
            return TradingMetrics(
                total_bets=0,
                winning_bets=0,
                total_staked=0.0,
                total_payout=0.0,
                roi=0.0,
                win_rate=0.0,
                average_odds=0.0,
                timestamp=datetime.utcnow()
            )
        
        total_bets = len(self.settled_bets)
        winning_bets = len([bet for bet in self.settled_bets if bet.result == "win"])
        total_staked = sum(bet.stake for bet in self.settled_bets)
        total_payout = sum(bet.payout or 0 for bet in self.settled_bets)
        
        roi = ((total_payout - total_staked) / total_staked * 100) if total_staked > 0 else 0
        win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
        average_odds = sum(bet.odds_taken for bet in self.settled_bets) / total_bets if total_bets > 0 else 0
        
        return TradingMetrics(
            total_bets=total_bets,
            winning_bets=winning_bets,
            total_staked=total_staked,
            total_payout=total_payout,
            roi=roi,
            win_rate=win_rate,
            average_odds=average_odds,
            timestamp=datetime.utcnow()
        )
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get detailed risk and performance metrics."""
        
        current_exposure = self.calculate_current_exposure()
        
        return {
            "current_bankroll": self.current_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "profit_loss": self.current_bankroll - self.initial_bankroll,
            "profit_loss_pct": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100,
            "current_exposure": current_exposure,
            "exposure_percentage": current_exposure / self.current_bankroll * 100 if self.current_bankroll > 0 else 0,
            "max_exposure_limit": self.max_exposure,
            "active_bets": len([bet for bet in self.bets if bet.result is None]),
            "total_volume": self.total_staked,
            "volume_turnover": self.total_staked / self.initial_bankroll if self.initial_bankroll > 0 else 0,
        }