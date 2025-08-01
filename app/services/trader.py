from datetime import datetime
from typing import Optional, List
from ..models.match import MatchEvent, Bet, Prediction, TradingMetrics
from .predictor import PredictorService


class TradingStrategy:
    """Base class for trading strategies."""
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float
    ) -> Optional[Bet]:
        raise NotImplementedError


class ExpectedValueStrategy(TradingStrategy):
    """Strategy that bets when expected value exceeds threshold."""
    
    def __init__(self, ev_threshold: float = 0.05, stake_percentage: float = 0.01):
        self.ev_threshold = ev_threshold
        self.stake_percentage = stake_percentage
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float
    ) -> Optional[Bet]:
        """Decide whether to place a bet based on expected value."""
        
        # Calculate expected values
        ev_home = (prediction.win_prob_home * match_event.market_odds_home) - 1.0
        ev_away = (prediction.win_prob_away * match_event.market_odds_away) - 1.0
        
        stake = bankroll * self.stake_percentage
        
        # Check home bet
        if ev_home > self.ev_threshold:
            return Bet(
                match_id=match_event.match_id,
                selection="home",
                stake=stake,
                odds_taken=match_event.market_odds_home,
                expected_value=ev_home,
                timestamp=datetime.utcnow()
            )
        
        # Check away bet
        if ev_away > self.ev_threshold:
            return Bet(
                match_id=match_event.match_id,
                selection="away",
                stake=stake,
                odds_taken=match_event.market_odds_away,
                expected_value=ev_away,
                timestamp=datetime.utcnow()
            )
        
        return None


class KellyStrategy(TradingStrategy):
    """Kelly Criterion-based betting strategy."""
    
    def __init__(self, min_edge: float = 0.02, max_stake_percentage: float = 0.05):
        self.min_edge = min_edge
        self.max_stake_percentage = max_stake_percentage
    
    def calculate_kelly_stake(
        self, win_prob: float, odds: float, bankroll: float
    ) -> float:
        """Calculate optimal stake using Kelly Criterion."""
        b = odds - 1.0  # Net odds
        p = win_prob
        q = 1.0 - p
        
        # Kelly fraction: (bp - q) / b
        kelly_fraction = (b * p - q) / b
        
        # Cap at maximum stake percentage for risk management
        kelly_fraction = min(kelly_fraction, self.max_stake_percentage)
        kelly_fraction = max(kelly_fraction, 0)  # No negative stakes
        
        return bankroll * kelly_fraction
    
    def should_bet(
        self, 
        prediction: Prediction, 
        match_event: MatchEvent, 
        bankroll: float
    ) -> Optional[Bet]:
        """Decide bet using Kelly Criterion."""
        
        # Calculate edges
        implied_prob_home = 1.0 / match_event.market_odds_home
        implied_prob_away = 1.0 / match_event.market_odds_away
        
        edge_home = prediction.win_prob_home - implied_prob_home
        edge_away = prediction.win_prob_away - implied_prob_away
        
        # Check home bet
        if edge_home > self.min_edge:
            stake = self.calculate_kelly_stake(
                prediction.win_prob_home, 
                match_event.market_odds_home, 
                bankroll
            )
            if stake > 0:
                ev = (prediction.win_prob_home * match_event.market_odds_home) - 1.0
                return Bet(
                    match_id=match_event.match_id,
                    selection="home",
                    stake=stake,
                    odds_taken=match_event.market_odds_home,
                    expected_value=ev,
                    timestamp=datetime.utcnow()
                )
        
        # Check away bet
        if edge_away > self.min_edge:
            stake = self.calculate_kelly_stake(
                prediction.win_prob_away, 
                match_event.market_odds_away, 
                bankroll
            )
            if stake > 0:
                ev = (prediction.win_prob_away * match_event.market_odds_away) - 1.0
                return Bet(
                    match_id=match_event.match_id,
                    selection="away",
                    stake=stake,
                    odds_taken=match_event.market_odds_away,
                    expected_value=ev,
                    timestamp=datetime.utcnow()
                )
        
        return None


class TradingService:
    """Main trading service that coordinates predictions and betting decisions."""
    
    def __init__(self, strategy: TradingStrategy, initial_bankroll: float = 1000.0):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.predictor = PredictorService()
        self.bets: List[Bet] = []
        self.settled_bets: List[Bet] = []
    
    async def process_match_event(self, match_event: MatchEvent) -> Optional[Bet]:
        """Process a match event and potentially place a bet."""
        
        # Generate prediction
        prediction = await self.predictor.generate_prediction(match_event)
        
        # Check if we should bet
        bet = self.strategy.should_bet(prediction, match_event, self.current_bankroll)
        
        if bet:
            self.bets.append(bet)
            self.current_bankroll -= bet.stake
            
        return bet
    
    def settle_bet(self, match_id: str, home_goals: int, away_goals: int) -> None:
        """Settle bets for a completed match."""
        
        # Find unsettled bets for this match
        unsettled_bets = [bet for bet in self.bets if bet.match_id == match_id and bet.result is None]
        
        for bet in unsettled_bets:
            # Determine result
            if home_goals > away_goals:
                winner = "home"
            elif away_goals > home_goals:
                winner = "away"
            else:
                winner = "draw"  # Handle draws (no winner)
            
            # Update bet result
            if bet.selection == winner:
                bet.result = "win"
                bet.payout = bet.stake * bet.odds_taken
                self.current_bankroll += bet.payout
            else:
                bet.result = "loss"
                bet.payout = 0
            
            self.settled_bets.append(bet)
        
        # Remove settled bets from active bets
        self.bets = [bet for bet in self.bets if bet.match_id != match_id]
    
    def get_trading_metrics(self) -> TradingMetrics:
        """Calculate current trading performance metrics."""
        
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