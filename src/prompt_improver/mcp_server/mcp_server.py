"""Pure MCP Server implementation for the Adaptive Prompt Enhancement System (APES).
Provides prompt enhancement via Model Context Protocol with stdio transport.
"""

import asyncio
import time
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from prompt_improver.database import get_session
from prompt_improver.services.analytics import AnalyticsService
from prompt_improver.services.prompt_improvement import PromptImprovementService

# Initialize the MCP server
mcp = FastMCP(
    name="APES - Adaptive Prompt Enhancement System",
    description="AI-powered prompt optimization service using ML-driven rules",
)

# Initialize services
prompt_service = PromptImprovementService()
analytics_service = AnalyticsService()


class PromptEnhancementRequest(BaseModel):
    """Request model for prompt enhancement"""

    prompt: str = Field(..., description="The prompt to enhance")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context for enhancement"
    )
    session_id: str | None = Field(default=None, description="Session ID for tracking")


class PromptStorageRequest(BaseModel):
    """Request model for storing prompt data"""

    original: str = Field(..., description="The original prompt")
    enhanced: str = Field(..., description="The enhanced prompt")
    metrics: dict[str, Any] = Field(..., description="Success metrics")
    session_id: str | None = Field(default=None, description="Session ID for tracking")


@mcp.tool()
async def improve_prompt(
    prompt: str = Field(..., description="The prompt to enhance"),
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context"
    ),
    session_id: str | None = Field(default=None, description="Session ID for tracking"),
    ctx: Context = None,
) -> dict[str, Any]:
    """Enhance a prompt using ML-optimized rules.

    This tool applies data-driven rules to improve prompt clarity, specificity,
    and effectiveness. Response time is optimized for <200ms.

    Args:
        prompt: The prompt text to enhance
        context: Optional context information
        session_id: Optional session ID for tracking
        ctx: MCP context (provided by framework)

    Returns:
        Enhanced prompt with processing metrics and applied rules
    """
    start_time = time.time()

    try:
        # Get database session
        async with get_session() as db_session:
            # Use the existing prompt improvement service
            result = await prompt_service.improve_prompt(
                prompt=prompt,
                user_context=context,
                session_id=session_id,
                db_session=db_session,
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Add processing metrics
            result["processing_time_ms"] = processing_time
            result["mcp_transport"] = "stdio"

            # Store the prompt data for ML training (real data priority 100)
            if result.get("improved_prompt") and result["improved_prompt"] != prompt:
                # Store asynchronously to not block response
                asyncio.create_task(
                    _store_prompt_data(
                        original=prompt,
                        enhanced=result["improved_prompt"],
                        metrics=result.get("metrics", {}),
                        session_id=session_id or result.get("session_id"),
                        priority=100,  # Real data priority
                    )
                )

            return result

    except Exception as e:
        # Return graceful error response
        return {
            "improved_prompt": prompt,  # Fallback to original
            "error": str(e),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "fallback": True,
        }


@mcp.tool()
async def store_prompt(
    original: str = Field(..., description="The original prompt"),
    enhanced: str = Field(..., description="The enhanced prompt"),
    metrics: dict[str, Any] = Field(..., description="Success metrics"),
    session_id: str | None = Field(default=None, description="Session ID"),
    ctx: Context = None,
) -> dict[str, Any]:
    """Store prompt interaction data for ML training.

    This tool captures real prompt data with priority 100 for training
    the ML models to improve rule effectiveness over time.

    Args:
        original: The original prompt text
        enhanced: The enhanced prompt text
        metrics: Performance and success metrics
        session_id: Optional session ID
        ctx: MCP context (provided by framework)

    Returns:
        Storage confirmation with priority level
    """
    try:
        await _store_prompt_data(
            original=original,
            enhanced=enhanced,
            metrics=metrics,
            session_id=session_id,
            priority=100,  # Real data priority
        )

        return {
            "status": "stored",
            "priority": 100,
            "data_source": "real",
            "session_id": session_id,
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "priority": 0}


@mcp.resource("apes://rule_status")
async def get_rule_status() -> dict[str, Any]:
    """Get current rule effectiveness and status.

    This resource provides real-time information about rule performance,
    effectiveness scores, and active experiments.

    Returns:
        Current rule status and effectiveness metrics
    """
    try:
        async with get_session() as db_session:
            # Get rule effectiveness stats
            rule_stats = await analytics_service.get_rule_effectiveness(
                days=7, min_usage=10, db_session=db_session
            )

            # Get active rules metadata
            rule_metadata = await prompt_service.get_rules_metadata(
                enabled_only=True, db_session=db_session
            )

            return {
                "active_rules": len(rule_metadata),
                "rule_effectiveness": [
                    {
                        "rule_id": stat.rule_id,
                        "effectiveness_score": stat.effectiveness_score,
                        "usage_count": stat.usage_count,
                        "improvement_rate": stat.improvement_rate,
                    }
                    for stat in rule_stats
                ],
                "last_updated": time.time(),
                "status": "operational",
            }

    except Exception as e:
        return {"status": "error", "error": str(e), "active_rules": 0}


async def _store_prompt_data(
    original: str,
    enhanced: str,
    metrics: dict[str, Any],
    session_id: str | None,
    priority: int,
) -> None:
    """Internal helper to store prompt data asynchronously."""
    try:
        async with get_session() as db_session:
            # Store in training_prompts table with real data priority
            await db_session.execute(
                """
                INSERT INTO training_prompts (
                    prompt_text, enhancement_result, 
                    data_source, training_priority, created_at
                ) VALUES (
                    :prompt_text, :enhancement_result::jsonb,
                    :data_source, :training_priority, NOW()
                )
                """,
                {
                    "prompt_text": original,
                    "enhancement_result": {
                        "enhanced_prompt": enhanced,
                        "metrics": metrics,
                        "session_id": session_id,
                    },
                    "data_source": "real",
                    "training_priority": priority,
                },
            )
            await db_session.commit()

    except Exception as e:
        # Log error but don't raise - this is async background task
        print(f"Error storing prompt data: {e}")


# Main entry point for stdio transport
if __name__ == "__main__":
    # Run the MCP server with stdio transport
    print("Starting APES MCP Server with stdio transport...")
    mcp.run(transport="stdio")
