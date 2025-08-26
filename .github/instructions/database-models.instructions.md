---
applyTo: "**/models/**/*.py"
---

## Database Models Development Guidelines

### SQLModel Best Practices
- Use SQLModel for all database models to leverage both Pydantic and SQLAlchemy features
- Include proper type annotations for all fields
- Implement table relationships with proper foreign key constraints
- Add database-level validation constraints

### Required Model Pattern
```python
from typing import Optional, List, DateTime
from sqlmodel import SQLModel, Field, Relationship, Column, Index
from datetime import datetime
from uuid import UUID, uuid4

class PromptAnalysisResult(SQLModel, table=True):
    """Database model for storing prompt analysis results."""
    
    __tablename__ = "prompt_analysis_results"
    __table_args__ = (
        Index("idx_user_id_created_at", "user_id", "created_at"),
        Index("idx_prompt_hash", "prompt_hash"),
        Index("idx_analysis_type", "analysis_type"),
    )
    
    # Primary key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Foreign keys with proper relationships
    user_id: UUID = Field(foreign_key="users.id", nullable=False)
    user: "User" = Relationship(back_populates="analysis_results")
    
    # Required fields with validation
    prompt_text: str = Field(min_length=1, max_length=10000, nullable=False)
    prompt_hash: str = Field(max_length=64, nullable=False, unique=True)
    analysis_type: str = Field(max_length=50, nullable=False)
    
    # Analysis results (JSON fields)
    quality_scores: dict = Field(default_factory=dict, sa_column=Column(JSON))
    improvement_suggestions: List[dict] = Field(default_factory=list, sa_column=Column(JSON))
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps with proper defaults
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: Optional[datetime] = Field(default=None)
    
    # Performance metrics
    processing_time_ms: Optional[int] = Field(default=None, ge=0)
    model_version: Optional[str] = Field(default=None, max_length=50)
```

### Indexing Strategy
- Add indexes for all foreign keys
- Create composite indexes for common query patterns
- Use partial indexes for filtered queries
- Monitor query performance and add indexes as needed

### Field Validation Rules
- Use appropriate field types (UUID for IDs, datetime for timestamps)
- Add length constraints for string fields
- Use proper nullable settings based on business requirements
- Include range validation for numeric fields (ge, le, gt, lt)

### Relationship Definitions
- Use proper foreign key relationships with cascading rules
- Define back_populates for bidirectional relationships
- Use lazy loading for large collections
- Consider using select_related for frequently accessed relationships

### JSON Field Usage
- Use JSON fields for flexible schema data (metadata, configurations)
- Validate JSON structure using Pydantic models where possible
- Index JSON fields using PostgreSQL JSON operators when needed
- Consider separate tables for complex relational data

### Audit Trail Pattern
```python
class AuditMixin(SQLModel):
    """Mixin for audit trail fields."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: Optional[datetime] = Field(default=None)
    created_by: Optional[UUID] = Field(foreign_key="users.id")
    updated_by: Optional[UUID] = Field(foreign_key="users.id")
    version: int = Field(default=1, ge=1)
```

### Migration Considerations
- Design models with future extensibility in mind
- Use nullable fields for new columns in existing tables
- Consider data migration scripts for complex changes
- Test migrations thoroughly in staging environments

### Performance Optimization
- Use appropriate field types for performance (UUID vs Integer for IDs)
- Consider partitioning for large tables
- Use database-level defaults where possible
- Monitor table sizes and query performance

### Required Imports
```python
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Relationship, Column, Index, JSON
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import validator, root_validator
```

### Validation Examples
```python
@validator('email')
def validate_email(cls, v):
    """Validate email format."""
    if '@' not in v:
        raise ValueError('Invalid email format')
    return v.lower()

@root_validator
def validate_date_range(cls, values):
    """Validate date ranges."""
    start_date = values.get('start_date')
    end_date = values.get('end_date')
    if start_date and end_date and start_date > end_date:
        raise ValueError('Start date must be before end date')
    return values
```
