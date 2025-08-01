import asyncio
import aiohttp
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from ..models.match import MatchEvent


class SportsDataAPI:
    """
    Service to fetch real sports data from various APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_urls = {
            "odds_api": "https://api.the-odds-api.com/v4",
            "football_data": "https://api.football-data.org/v4", 
            "rapidapi": "https://api-football-v1.p.rapidapi.com/v3"
        }
    
    async def fetch_live_odds(
        self, 
        sport: str = "soccer_epl",
        markets: List[str] = ["h2h"]  # head-to-head
    ) -> List[Dict[str, Any]]:
        """
        Fetch live odds from The Odds API.
        Requires API key: https://the-odds-api.com/
        """
        if not self.api_key:
            return self._mock_live_odds()
        
        url = f"{self.base_urls['odds_api']}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "uk,us",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return self._mock_live_odds()
        except Exception as e:
            print(f"Error fetching live odds: {e}")
            return self._mock_live_odds()
    
    def _mock_live_odds(self) -> List[Dict[str, Any]]:
        """Mock data for development/testing."""
        return [
            {
                "id": "mock_match_001",
                "sport_key": "soccer_epl",
                "sport_title": "Premier League",
                "commence_time": "2025-08-01T15:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "title": "Bet365",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.1},
                                    {"name": "Chelsea", "price": 1.9},
                                    {"name": "Draw", "price": 3.2}
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "id": "mock_match_002", 
                "sport_key": "soccer_epl",
                "sport_title": "Premier League",
                "commence_time": "2025-08-01T17:30:00Z",
                "home_team": "Liverpool",
                "away_team": "Manchester City",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "title": "Bet365", 
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Liverpool", "price": 2.8},
                                    {"name": "Manchester City", "price": 2.4},
                                    {"name": "Draw", "price": 3.1}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    
    async def fetch_historical_results(
        self, 
        league: str = "PL",  # Premier League
        season: str = "2023-24",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical match results for backtesting.
        """
        if not self.api_key:
            return self._mock_historical_data()
        
        # Implementation would use football-data.org API
        # For now, return mock data
        return self._mock_historical_data()
    
    def _mock_historical_data(self) -> List[Dict[str, Any]]:
        """Mock historical data for development."""
        matches = []
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Tottenham", "Manchester United"]
        
        # Generate 50 realistic historical matches
        for i in range(50):
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            
            if home_team == away_team:
                away_team = teams[(i + 2) % len(teams)]
            
            # Simulate realistic scores
            home_goals = max(0, int(np.random.poisson(1.4)))  # Slightly higher for home
            away_goals = max(0, int(np.random.poisson(1.1)))
            
            matches.append({
                "id": f"hist_{i:03d}",
                "date": "2024-01-01",  # Would be real dates
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_goals,
                "away_score": away_goals,
                "result": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
            })
        
        return matches
    
    def convert_to_match_events(self, odds_data: List[Dict]) -> List[MatchEvent]:
        """Convert API odds data to MatchEvent objects."""
        events = []
        
        for match in odds_data:
            if not match.get("bookmakers"):
                continue
                
            # Get best odds across bookmakers
            best_home_odds = 0
            best_away_odds = 0
            
            for bookmaker in match["bookmakers"]:
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == match["home_team"]:
                                best_home_odds = max(best_home_odds, outcome["price"])
                            elif outcome["name"] == match["away_team"]:
                                best_away_odds = max(best_away_odds, outcome["price"])
            
            if best_home_odds > 0 and best_away_odds > 0:
                event = MatchEvent(
                    match_id=match["id"],
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    minute=0,  # Pre-match
                    home_goals=0,
                    away_goals=0,
                    market_odds_home=best_home_odds,
                    market_odds_away=best_away_odds,
                    timestamp=datetime.utcnow()
                )
                events.append(event)
        
        return events


class HistoricalDataManager:
    """
    Manages historical match data for backtesting and model training.
    """
    
    def __init__(self, data_source: SportsDataAPI):
        self.data_source = data_source
        self.historical_matches = []
    
    async def load_historical_data(self, seasons: List[str] = ["2023-24"]):
        """Load historical match data."""
        all_matches = []
        
        for season in seasons:
            matches = await self.data_source.fetch_historical_results(season=season)
            all_matches.extend(matches)
        
        self.historical_matches = all_matches
        print(f"Loaded {len(all_matches)} historical matches")
    
    def get_team_historical_performance(self, team: str) -> Dict[str, float]:
        """Get team's historical performance metrics."""
        if not self.historical_matches:
            return {"win_rate": 0.5, "avg_goals_for": 1.3, "avg_goals_against": 1.3}
        
        team_matches = [
            m for m in self.historical_matches 
            if m["home_team"] == team or m["away_team"] == team
        ]
        
        if not team_matches:
            return {"win_rate": 0.5, "avg_goals_for": 1.3, "avg_goals_against": 1.3}
        
        wins = 0
        total_goals_for = 0
        total_goals_against = 0
        
        for match in team_matches:
            if match["home_team"] == team:
                goals_for = match["home_score"]
                goals_against = match["away_score"]
                if match["result"] == "H":
                    wins += 1
            else:
                goals_for = match["away_score"]
                goals_against = match["home_score"]
                if match["result"] == "A":
                    wins += 1
        
        total_goals_for += goals_for
        total_goals_against += goals_against
        
        return {
            "win_rate": wins / len(team_matches),
            "avg_goals_for": total_goals_for / len(team_matches),
            "avg_goals_against": total_goals_against / len(team_matches)
        }
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Create dataset for ML model training."""
        if not self.historical_matches:
            return pd.DataFrame()
        
        data = []
        for match in self.historical_matches:
            # Create features for ML model
            home_perf = self.get_team_historical_performance(match["home_team"])
            away_perf = self.get_team_historical_performance(match["away_team"])
            
            row = {
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_win_rate": home_perf["win_rate"],
                "away_win_rate": away_perf["win_rate"],
                "home_avg_goals": home_perf["avg_goals_for"],
                "away_avg_goals": away_perf["avg_goals_for"],
                "home_goals": match["home_score"],
                "away_goals": match["away_score"],
                "result": match["result"]
            }
            data.append(row)
        
        return pd.DataFrame(data)


class BacktestingEngine:
    """
    Engine for backtesting trading strategies on historical data.
    """
    
    def __init__(self, historical_manager: HistoricalDataManager):
        self.historical_manager = historical_manager
        self.results = []
    
    async def run_backtest(
        self,
        trading_service,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        """
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Reset trading service
        initial_bankroll = trading_service.initial_bankroll
        trading_service.current_bankroll = initial_bankroll
        trading_service.bets = []
        trading_service.settled_bets = []
        
        total_bets = 0
        total_wins = 0
        
        # Simulate historical matches
        for match in self.historical_manager.historical_matches:
            # Create match event
            event = MatchEvent(
                match_id=match["id"],
                home_team=match["home_team"],
                away_team=match["away_team"],
                minute=0,
                home_goals=0,
                away_goals=0,
                market_odds_home=2.0,  # Would use real historical odds
                market_odds_away=2.0,
                timestamp=datetime.utcnow()
            )
            
            # Process bet decision
            bet = await trading_service.process_match_event(event)
            
            if bet:
                total_bets += 1
                
                # Settle bet based on actual result
                home_goals = match["home_score"]
                away_goals = match["away_score"]
                
                if ((bet.selection == "home" and home_goals > away_goals) or
                    (bet.selection == "away" and away_goals > home_goals)):
                    total_wins += 1
                
                # Settle the bet
                trading_service.settle_bet(event.match_id, home_goals, away_goals)
        
        # Calculate performance metrics
        final_bankroll = trading_service.current_bankroll
        profit_loss = final_bankroll - initial_bankroll
        roi = (profit_loss / initial_bankroll) * 100
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        results = {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": final_bankroll,
            "profit_loss": profit_loss,
            "roi_percentage": roi,
            "total_bets": total_bets,
            "winning_bets": total_wins,
            "win_rate_percentage": win_rate,
            "total_matches": len(self.historical_manager.historical_matches)
        }
        
        self.results.append(results)
        return results