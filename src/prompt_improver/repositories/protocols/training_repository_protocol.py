"""Training Repository Protocol.

Defines the interface for training session management, progress tracking,
and training-related database operations following repository pattern.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable  
class TrainingRepositoryProtocol(Protocol):
    """Protocol for training data persistence and management operations."""
    
    async def create_training_session(self, training_config: Dict[str, Any]) -> str:
        """Create a new training session with configuration.
        
        Args:
            training_config: Training configuration dictionary
            
        Returns:
            Created training session ID
        """
        ...
    
    async def update_training_progress(
        self,
        session_id: str,
        iteration: int,
        performance_metrics: Dict[str, float],
        improvement_score: float = 0.0,
    ) -> bool:
        """Update training progress for a session.
        
        Args:
            session_id: Training session ID
            iteration: Current iteration number
            performance_metrics: Performance metrics dictionary
            improvement_score: Improvement score for this iteration
            
        Returns:
            True if updated successfully
        """
        ...
    
    async def get_training_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session context and current state.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Training session context if found
        """
        ...
    
    async def save_training_progress(self, session_id: str) -> bool:
        """Save current training progress to persistent storage.
        
        Args:
            session_id: Training session ID
            
        Returns:
            True if progress saved successfully
        """
        ...
    
    async def create_checkpoint(self, session_id: str) -> str:
        """Create emergency training checkpoint.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Checkpoint ID if created successfully
        """
        ...
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore training session from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore from
            
        Returns:
            Restored session context if successful
        """
        ...
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active training sessions.
        
        Returns:
            List of active training session dictionaries
        """
        ...
    
    async def terminate_session(self, session_id: str, reason: str = "manual_termination") -> bool:
        """Terminate a training session.
        
        Args:
            session_id: Session ID to terminate
            reason: Reason for termination
            
        Returns:
            True if terminated successfully
        """
        ...
    
    async def get_session_history(
        self, 
        session_id: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get training session history.
        
        Args:
            session_id: Specific session ID, or None for all sessions
            limit: Maximum number of records to return
            
        Returns:
            List of training session records
        """
        ...
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old completed training sessions.
        
        Args:
            days_old: Sessions older than this many days will be cleaned up
            
        Returns:
            Number of sessions cleaned up
        """
        ...
    
    async def store_training_iteration_data(
        self,
        session_id: str,
        iteration: int,
        iteration_data: Dict[str, Any]
    ) -> bool:
        """Store detailed training iteration data.
        
        Args:
            session_id: Training session ID
            iteration: Iteration number
            iteration_data: Detailed iteration data
            
        Returns:
            True if stored successfully
        """
        ...
    
    async def get_training_iteration_data(
        self, 
        session_id: str, 
        iteration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get training iteration data for analysis.
        
        Args:
            session_id: Training session ID
            iteration: Specific iteration number, or None for all iterations
            
        Returns:
            List of iteration data records
        """
        ...
    
    async def verify_training_tables(self) -> bool:
        """Verify required training tables exist and are accessible.
        
        Returns:
            True if all required tables are available
        """
        ...
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health for training operations.
        
        Returns:
            Database health status
        """
        ...