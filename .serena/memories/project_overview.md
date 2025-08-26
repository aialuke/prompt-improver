# APES Project Overview

## Purpose
Adaptive Prompt Enhancement System (APES) - Intelligent prompt improvement system using ML-driven rule learning and clean architecture principles.

## Tech Stack
- **Python 3.11+** with clean architecture patterns
- **FastAPI** for API layer
- **AsyncPG** for PostgreSQL database (standardized from psycopg)
- **Redis/CoreDIS** for caching with graceful fallback
- **ML Stack**: scikit-learn, optuna, mlflow, pandas, numpy
- **Testing**: pytest with testcontainers (no mocks policy)
- **Monitoring**: OpenTelemetry observability stack

## Architecture (2025)
Clean Architecture with:
- Protocol-based dependency injection
- Repository pattern for all data access
- Service facades (*Facade, *Service, *Manager patterns)
- Multi-level caching (L1/L2/L3) <2ms response
- Real behavior testing (87.5% success rate)

## Development Standards
- No mocks policy - testcontainers for real services
- Protocol-based DI (typing.Protocol)
- Repository boundaries enforced via import-linter
- Clean architecture layer separation
- Performance: P95 <100ms, cache >80% hit rates