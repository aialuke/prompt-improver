"""Real Behavior Testing Infrastructure.

Comprehensive testcontainer suite for real behavior testing of all decomposed services.
Provides infrastructure for testing with actual services instead of mocks.

Features:
- PostgreSQL testcontainer for database testing
- Redis testcontainer for cache testing  
- ML model testing with real data scenarios
- Network failure simulation for retry testing
- Error injection for error handling testing
- Performance benchmarking with real services
- Integration with existing test architecture

Performance Targets:
- Database operations: <10ms for simple queries, <100ms for complex operations
- Cache operations: L1 <1ms, L2 <10ms, L3 <50ms
- ML operations: <100ms for predictions, <5000ms for training
- API endpoints: <100ms for P95, <500ms for P99
"""

__version__ = "1.0.0"