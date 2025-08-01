# 🏟️ Sports Betting Simulator - !EXPERIMENTATION do not rely on it.

A real-time sports data simulation platform with predictive modeling and automated trading capabilities. Built with FastAPI, Redis Streams, and modern async Python patterns.

## 🚀 Features

- **Real-time Data Streaming**: Live sports match data processing via Redis Streams
- **Predictive Modeling**: Poisson-based win probability calculations (extensible to ML models)
- **Automated Trading**: Multiple betting strategies (Expected Value, Kelly Criterion)
- **Event-Driven Architecture**: Scalable microservices with async processing
- **API-First Design**: RESTful endpoints for monitoring and control
- **Docker Support**: Containerized deployment with Redis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Feed     │───▶│  Prediction     │───▶│   Trading       │
│   Service       │    │   Service       │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Redis Streams                                │
│  • match_events    • predictions    • bets    • metrics        │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, asyncio
- **Streaming**: Redis Streams for event processing
- **Data**: Pandas, NumPy, Scikit-learn
- **Validation**: Pydantic v2 for data models
- **Containerization**: Docker, Docker Compose
- **Package Management**: uv (modern pip replacement)

## 📦 Project Structure

```
sports_simulator/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── models/
│   │   └── match.py         # Pydantic data models
│   ├── services/
│   │   ├── predictor.py     # Win probability calculations
│   │   ├── trader.py        # Betting strategies & risk management
│   │   └── stream.py        # Redis stream pub/sub services
│   ├── api/
│   └── core/
├── tests/                   # Test suite (to be implemented)
├── datasets/               # Historical match data
├── docker-compose.yml      # Multi-service deployment
├── Dockerfile             # Container definition
└── pyproject.toml         # Python project configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server (or use Docker)
- [uv](https://github.com/astral-sh/uv) package manager

### Local Development

1. **Clone and setup environment**:
```bash
git clone <your-repo>
cd sports_simulator
uv venv
source .venv/bin/activate
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Start Redis** (if not using Docker):
```bash
redis-server
```

4. **Run the application**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Start all services**:
```bash
docker-compose up --build
```

2. **Access services**:
   - API: http://localhost:8000
   - Redis: localhost:6379

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/match-events` | Publish new match event |
| `GET` | `/live-matches` | Get stream information |
| `GET` | `/bets` | Retrieve all bets |
| `GET` | `/metrics` | Trading performance metrics |
| `POST` | `/simulate/match` | Simulate complete match |

### Example Usage

**Publish a match event**:
```bash
curl -X POST "http://localhost:8000/match-events" \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": "match_001",
    "home_team": "Arsenal",
    "away_team": "Chelsea", 
    "minute": 45,
    "home_goals": 1,
    "away_goals": 0,
    "market_odds_home": 2.1,
    "market_odds_away": 1.8,
    "timestamp": "2024-01-01T15:30:00"
  }'
```

**Get trading metrics**:
```bash
curl http://localhost:8000/metrics
```

## 🎯 Trading Strategies

### Expected Value Strategy
```python
# Bets when expected value > threshold (default 5%)
strategy = ExpectedValueStrategy(ev_threshold=0.05, stake_percentage=0.01)
```

### Kelly Criterion Strategy  
```python
# Optimal bet sizing based on edge and bankroll
strategy = KellyStrategy(min_edge=0.02, max_stake_percentage=0.05)
```

## 📊 Predictive Models

### Current: Poisson Model
- Simple time-adjusted Poisson distribution
- Factors in current score and match time
- Suitable for demonstration and testing

### Future: ML Models
- Logistic regression on historical data
- Feature engineering (team stats, player ratings)
- Model training pipeline with scikit-learn

## 🔄 Event Flow

1. **Data Ingestion**: Match events published to `match_events` stream
2. **Prediction**: Event processor generates win probabilities  
3. **Trading Decision**: Strategy evaluates expected value vs market odds
4. **Bet Execution**: Qualified bets placed and tracked
5. **Settlement**: Match completion triggers profit/loss calculation

## 🐳 Docker Services

- **api**: FastAPI application server
- **processor**: Background event processing worker  
- **redis**: Redis server with persistence

## 🔧 Configuration

Key environment variables:

```bash
REDIS_URL=redis://localhost:6379
INITIAL_BANKROLL=10000.0
MIN_EDGE=0.02
MAX_STAKE_PERCENTAGE=0.02
LOG_LEVEL=INFO
```

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 📈 Monitoring

- **Health Checks**: `/health` endpoint for service monitoring
- **Metrics**: Trading performance via `/metrics` endpoint
- **Stream Info**: Redis stream statistics via `/streams/info`
- **Latency Tracking**: End-to-end processing time measurement
