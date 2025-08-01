import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .models.match import MatchEvent, Prediction, Bet, TradingMetrics, LatencyMetrics
from .services.stream import RedisStreamService, EventProcessor
from .services.trader import TradingService, ExpectedValueStrategy, KellyStrategy
from .services.improved_trader import ImprovedTradingService, ValueBettingStrategy, AdaptiveKellyStrategy
from .services.predictor import PredictorService
from .services.improved_predictor import ImprovedPredictorService


# Global services
stream_service: Optional[RedisStreamService] = None
trading_service: Optional[TradingService] = None
event_processor: Optional[EventProcessor] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global stream_service, trading_service, event_processor
    
    # Startup
    logger.info("Starting Sports Betting Simulator...")
    
    # Initialize services
    stream_service = RedisStreamService()
    await stream_service.connect()
    
    # Initialize IMPROVED trading service with conservative strategy
    strategy = ValueBettingStrategy(min_edge=0.05, min_confidence=0.15, max_stake_percentage=0.005)
    trading_service = ImprovedTradingService(strategy, initial_bankroll=10000.0)
    
    # Initialize event processor
    event_processor = EventProcessor(stream_service)
    
    # Start background processing
    asyncio.create_task(event_processor.start_processing())
    
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if stream_service:
        await stream_service.disconnect()


app = FastAPI(
    title="Sports Betting Simulator",
    description="Real-time sports data simulation with predictive modeling",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.post("/match-events", response_model=dict)
async def publish_match_event(match_event: MatchEvent):
    """Publish a new match event to the stream."""
    if not stream_service:
        raise HTTPException(status_code=500, detail="Stream service not initialized")
    
    try:
        message_id = await stream_service.publish_match_event(match_event)
        
        # Also process directly for immediate response
        if trading_service:
            bet = await trading_service.process_match_event(match_event)
            if bet:
                await stream_service.publish_bet(bet)
        
        return {
            "message_id": message_id,
            "match_id": match_event.match_id,
            "status": "published"
        }
    except Exception as e:
        logger.error(f"Error publishing match event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live-matches", response_model=List[dict])
async def get_live_matches():
    """Get information about live matches from streams."""
    if not stream_service:
        raise HTTPException(status_code=500, detail="Stream service not initialized")
    
    try:
        # Get stream information
        match_events_info = await stream_service.get_stream_info(
            stream_service.MATCH_EVENTS_STREAM
        )
        predictions_info = await stream_service.get_stream_info(
            stream_service.PREDICTIONS_STREAM
        )
        
        return [
            {
                "stream": "match_events",
                "length": match_events_info.get("length", 0),
                "last_entry_id": match_events_info.get("last-generated-id", "0-0")
            },
            {
                "stream": "predictions", 
                "length": predictions_info.get("length", 0),
                "last_entry_id": predictions_info.get("last-generated-id", "0-0")
            }
        ]
    except Exception as e:
        logger.error(f"Error getting live matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bets", response_model=List[Bet])
async def get_bets():
    """Get all bets placed by the trading service."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    return trading_service.bets + trading_service.settled_bets


@app.get("/bets/active", response_model=List[Bet])
async def get_active_bets():
    """Get only active (unsettled) bets."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    return trading_service.bets


@app.get("/bets/settled", response_model=List[Bet])
async def get_settled_bets():
    """Get only settled bets."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    return trading_service.settled_bets


@app.post("/bets/{match_id}/settle")
async def settle_match_bets(match_id: str, home_goals: int, away_goals: int):
    """Settle bets for a completed match."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    try:
        trading_service.settle_bet(match_id, home_goals, away_goals)
        return {
            "match_id": match_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "status": "settled"
        }
    except Exception as e:
        logger.error(f"Error settling bets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=TradingMetrics)
async def get_trading_metrics():
    """Get current trading performance metrics."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    return trading_service.get_trading_metrics()


@app.get("/bankroll")
async def get_bankroll():
    """Get current bankroll information."""
    if not trading_service:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    
    # Check if using improved service
    if hasattr(trading_service, 'get_risk_metrics'):
        return trading_service.get_risk_metrics()
    
    # Fallback for old service
    return {
        "initial_bankroll": trading_service.initial_bankroll,
        "current_bankroll": trading_service.current_bankroll,
        "profit_loss": trading_service.current_bankroll - trading_service.initial_bankroll,
        "profit_loss_percentage": (
            (trading_service.current_bankroll - trading_service.initial_bankroll) / 
            trading_service.initial_bankroll * 100
        )
    }


@app.post("/predictions", response_model=Prediction)
async def generate_prediction(match_event: MatchEvent):
    """Generate a prediction for a match event."""
    try:
        predictor = PredictorService()
        prediction = await predictor.generate_prediction(match_event)
        
        if stream_service:
            await stream_service.publish_prediction(prediction)
        
        return prediction
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/compare")
async def compare_predictions(match_event: MatchEvent):
    """Compare old vs improved prediction models."""
    try:
        # Old model
        old_predictor = PredictorService()
        old_prediction = await old_predictor.generate_prediction(match_event)
        
        # Improved model
        improved_predictor = ImprovedPredictorService()
        improved_prediction = await improved_predictor.generate_prediction(match_event)
        
        # Get detailed analysis
        ev_home, ev_away, analysis = await improved_predictor.analyze_match_value(match_event)
        
        return {
            "match_info": {
                "match_id": match_event.match_id,
                "teams": f"{match_event.home_team} vs {match_event.away_team}",
                "minute": match_event.minute,
                "score": f"{match_event.home_goals}-{match_event.away_goals}",
                "odds": f"Home: {match_event.market_odds_home}, Away: {match_event.market_odds_away}"
            },
            "old_model": {
                "home_prob": old_prediction.win_prob_home,
                "away_prob": old_prediction.win_prob_away,
                "draw_prob": 1 - old_prediction.win_prob_home - old_prediction.win_prob_away
            },
            "improved_model": {
                "home_prob": improved_prediction.win_prob_home,
                "away_prob": improved_prediction.win_prob_away,
                "draw_prob": 1 - improved_prediction.win_prob_home - improved_prediction.win_prob_away
            },
            "market_analysis": analysis,
            "betting_recommendation": {
                "should_bet_home": analysis["market_edge_home"] > 0.05 and analysis["confidence_home"] > 0.15,
                "should_bet_away": analysis["market_edge_away"] > 0.05 and analysis["confidence_away"] > 0.15,
                "home_edge": analysis["market_edge_home"],
                "away_edge": analysis["market_edge_away"]
            }
        }
    except Exception as e:
        logger.error(f"Error comparing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streams/info")
async def get_streams_info():
    """Get information about all Redis streams."""
    if not stream_service:
        raise HTTPException(status_code=500, detail="Stream service not initialized")
    
    try:
        streams = [
            stream_service.MATCH_EVENTS_STREAM,
            stream_service.PREDICTIONS_STREAM,
            stream_service.BETS_STREAM,
            stream_service.METRICS_STREAM
        ]
        
        info = {}
        for stream in streams:
            try:
                info[stream] = await stream_service.get_stream_info(stream)
            except Exception:
                info[stream] = {"error": "Stream does not exist or is empty"}
        
        return info
    except Exception as e:
        logger.error(f"Error getting streams info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/match")
async def simulate_match(
    home_team: str = "Team A",
    away_team: str = "Team B", 
    duration_minutes: int = 90
):
    """Simulate a complete match with events."""
    if not stream_service or not trading_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        match_id = f"match_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate match events
        events = []
        home_goals = 0
        away_goals = 0
        
        for minute in range(0, duration_minutes + 1, 10):
            # Randomly generate goals
            if minute > 0 and minute % 20 == 0:
                if minute % 40 == 0:
                    home_goals += 1
                else:
                    away_goals += 1
            
            # Create match event
            match_event = MatchEvent(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                minute=minute,
                home_goals=home_goals,
                away_goals=away_goals,
                market_odds_home=2.1 - (home_goals - away_goals) * 0.1,
                market_odds_away=2.1 - (away_goals - home_goals) * 0.1,
                timestamp=datetime.utcnow()
            )
            
            # Publish event
            await stream_service.publish_match_event(match_event)
            
            # Process for betting
            bet = await trading_service.process_match_event(match_event)
            if bet:
                await stream_service.publish_bet(bet)
                events.append({"minute": minute, "bet_placed": True, "bet": bet.model_dump()})
            else:
                events.append({"minute": minute, "bet_placed": False})
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.1)
        
        # Settle the match
        trading_service.settle_bet(match_id, home_goals, away_goals)
        
        return {
            "match_id": match_id,
            "final_score": f"{home_team} {home_goals} - {away_goals} {away_team}",
            "events": events,
            "metrics": trading_service.get_trading_metrics().model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error simulating match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)