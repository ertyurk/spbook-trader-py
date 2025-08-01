import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime

from redis.asyncio import Redis

from ..models.match import MatchEvent, Prediction, Bet, LatencyMetrics


class RedisStreamService:
    """Service for publishing and consuming events via Redis Streams."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.logger = logging.getLogger(__name__)
        
        # Stream names
        self.MATCH_EVENTS_STREAM = "match_events"
        self.PREDICTIONS_STREAM = "predictions" 
        self.BETS_STREAM = "bets"
        self.METRICS_STREAM = "metrics"
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis = Redis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.logger.info("Disconnected from Redis")
    
    async def publish_match_event(self, match_event: MatchEvent) -> str:
        """Publish a match event to the stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        event_data = {
            "event_type": "match_event",
            "data": match_event.model_dump_json(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await self.redis.xadd(
            self.MATCH_EVENTS_STREAM, 
            event_data
        )
        
        self.logger.info(f"Published match event {match_event.match_id} with ID {message_id}")
        return message_id
    
    async def publish_prediction(self, prediction: Prediction) -> str:
        """Publish a prediction to the stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        event_data = {
            "event_type": "prediction",
            "data": prediction.model_dump_json(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await self.redis.xadd(
            self.PREDICTIONS_STREAM, 
            event_data
        )
        
        self.logger.info(f"Published prediction for match {prediction.match_id}")
        return message_id
    
    async def publish_bet(self, bet: Bet) -> str:
        """Publish a bet to the stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        event_data = {
            "event_type": "bet",
            "data": bet.model_dump_json(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await self.redis.xadd(
            self.BETS_STREAM, 
            event_data
        )
        
        self.logger.info(f"Published bet for match {bet.match_id}")
        return message_id
    
    async def publish_latency_metrics(self, metrics: LatencyMetrics) -> str:
        """Publish latency metrics to the stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        event_data = {
            "event_type": "latency_metrics",
            "data": metrics.model_dump_json(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await self.redis.xadd(
            self.METRICS_STREAM, 
            event_data
        )
        
        return message_id
    
    async def consume_match_events(
        self, 
        consumer_group: str = "processors",
        consumer_name: str = "processor-1",
        last_id: str = ">"
    ) -> AsyncGenerator[MatchEvent, None]:
        """Consumer for match events stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Create consumer group if it doesn't exist
        try:
            await self.redis.xgroup_create(
                self.MATCH_EVENTS_STREAM, 
                consumer_group, 
                id="0", 
                mkstream=True
            )
        except Exception:
            # Group likely already exists
            pass
        
        while True:
            try:
                # Read from stream
                messages = await self.redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {self.MATCH_EVENTS_STREAM: last_id},
                    count=1,
                    block=5000  # 5 second timeout
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            # Parse the match event
                            data = json.loads(fields["data"])
                            match_event = MatchEvent.model_validate(data)
                            
                            # Acknowledge message
                            await self.redis.xack(
                                self.MATCH_EVENTS_STREAM, 
                                consumer_group, 
                                msg_id
                            )
                            
                            yield match_event
                            
                        except Exception as e:
                            self.logger.error(f"Error processing message {msg_id}: {e}")
                            # Could implement dead letter queue here
            
            except asyncio.TimeoutError:
                # No new messages, continue polling
                continue
            except Exception as e:
                self.logger.error(f"Error consuming match events: {e}")
                await asyncio.sleep(1)
    
    async def consume_predictions(
        self, 
        consumer_group: str = "traders",
        consumer_name: str = "trader-1",
        last_id: str = ">"
    ) -> AsyncGenerator[Prediction, None]:
        """Consumer for predictions stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Create consumer group if it doesn't exist
        try:
            await self.redis.xgroup_create(
                self.PREDICTIONS_STREAM, 
                consumer_group, 
                id="0", 
                mkstream=True
            )
        except Exception:
            pass
        
        while True:
            try:
                messages = await self.redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {self.PREDICTIONS_STREAM: last_id},
                    count=1,
                    block=5000
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            data = json.loads(fields["data"])
                            prediction = Prediction.model_validate(data)
                            
                            await self.redis.xack(
                                self.PREDICTIONS_STREAM, 
                                consumer_group, 
                                msg_id
                            )
                            
                            yield prediction
                            
                        except Exception as e:
                            self.logger.error(f"Error processing prediction {msg_id}: {e}")
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error consuming predictions: {e}")
                await asyncio.sleep(1)
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        return await self.redis.xinfo_stream(stream_name)
    
    async def get_stream_length(self, stream_name: str) -> int:
        """Get the length of a stream."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        return await self.redis.xlen(stream_name)


class EventProcessor:
    """Processes events from streams and coordinates different services."""
    
    def __init__(self, stream_service: RedisStreamService):
        self.stream_service = stream_service
        self.logger = logging.getLogger(__name__)
    
    async def start_processing(self) -> None:
        """Start processing events from streams."""
        
        # Start multiple concurrent processors
        tasks = [
            asyncio.create_task(self._process_match_events()),
            asyncio.create_task(self._monitor_latency())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _process_match_events(self) -> None:
        """Process match events and generate predictions."""
        from .predictor import PredictorService
        
        predictor = PredictorService()
        
        async for match_event in self.stream_service.consume_match_events():
            try:
                # Generate prediction
                prediction = await predictor.generate_prediction(match_event)
                
                # Publish prediction
                await self.stream_service.publish_prediction(prediction)
                
                self.logger.info(f"Processed match event for {match_event.match_id}")
                
            except Exception as e:
                self.logger.error(f"Error processing match event: {e}")
    
    async def _monitor_latency(self) -> None:
        """Monitor end-to-end latency of the system."""
        
        # This would track timestamps from ingestion to decision
        # and publish latency metrics
        
        while True:
            try:
                # Monitor stream lengths and processing delays
                match_events_len = await self.stream_service.get_stream_length(
                    self.stream_service.MATCH_EVENTS_STREAM
                )
                predictions_len = await self.stream_service.get_stream_length(
                    self.stream_service.PREDICTIONS_STREAM
                )
                
                self.logger.info(
                    f"Stream lengths - Match Events: {match_events_len}, "
                    f"Predictions: {predictions_len}"
                )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring latency: {e}")
                await asyncio.sleep(10)