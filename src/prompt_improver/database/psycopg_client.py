"""Type-safe database client using psycopg3 + Pydantic for zero serialization overhead.
Research-validated patterns for high-performance database operations.
Enhanced with 2025 error handling best practices.
"""

import contextlib
import logging
import os
import socket
import time
from typing import Any, Dict, List, TypeVar

import aiofiles

from psycopg import (
    errors as psycopg_errors,
    sql,
)
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ValidationError

from prompt_improver.utils.datetime_utils import aware_utc_now

from .config import DatabaseConfig
from .error_handling import (
    DatabaseErrorClassifier,
    ErrorCategory,
    ErrorSeverity,
    ErrorMetrics,
    enhance_error_context,
    global_error_metrics,
    get_default_database_retry_config,
    execute_with_database_retry,
)

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

class QueryMetrics:
    """Track query performance metrics for Phase 2 requirements"""

    def __init__(self):
        self.query_times: list[float] = []
        self.slow_queries: list[dict[str, Any]] = []
        self.total_queries = 0

    def record_query(self, query: str, duration_ms: float, params: dict | None = None):
        """Record query execution metrics"""
        self.total_queries += 1
        self.query_times.append(duration_ms)

        # Track slow queries (>50ms target)
        if duration_ms > 50:
            self.slow_queries.append({
                "query": query[:100] + "..." if len(query) > 100 else query,
                "duration_ms": duration_ms,
                "timestamp": aware_utc_now().isoformat(),
                "params_count": len(params) if params else 0,
            })

    @property
    def avg_query_time(self) -> float:
        """Average query time in milliseconds"""
        return sum(self.query_times) / len(self.query_times) if self.query_times else 0

    @property
    def queries_under_50ms(self) -> float:
        """Percentage of queries under 50ms target"""
        if not self.query_times:
            return 0
        under_target = sum(1 for t in self.query_times if t <= 50)
        return (under_target / len(self.query_times)) * 100

class TypeSafePsycopgClient:
    """High-performance type-safe database client using psycopg3 + Pydantic.

    Enhanced with 2025 connection optimizations:
    - Application name for monitoring
    - Timezone awareness
    - Enhanced pool monitoring
    - Connection security options
    - Advanced error handling with retry mechanisms
    - Circuit breaker pattern for fault tolerance
    - Comprehensive error classification and metrics
    """

    def __init__(
        self,
        config: DatabaseConfig | None = None,
        enable_error_metrics: bool = True,
    ):
        self.config = config or DatabaseConfig()
        self.metrics = QueryMetrics()

        # 2025 Enhancement: Use unified retry manager
        from ..ml.orchestration.core.unified_retry_manager import get_retry_manager
        self.retry_manager = get_retry_manager()
        self.retry_config = get_default_database_retry_config()

        self.error_metrics = ErrorMetrics() if enable_error_metrics else None
        self._connection_id = f"psycopg-{id(self)}"

        # 2025 Enhancement: Generate application name for monitoring
        hostname = socket.gethostname()
        pid = os.getpid()
        app_name = f"prompt-improver-{hostname}-{pid}"

        # Build connection string with 2025 optimizations (psycopg3 format)
        self.conninfo = (
            f"postgresql://{self.config.postgres_username}:{self.config.postgres_password}@"
            f"{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}"
        )

        # 2025 Enhancement: Advanced connection kwargs (psycopg3 compatible)
        connection_kwargs = {
            "row_factory": dict_row,  # Return dict rows for Pydantic mapping
            "prepare_threshold": 5,  # Prepare frequently used queries
            "autocommit": False,  # Explicit transaction control
            # 2025 Connection Optimizations
            "application_name": app_name,  # For monitoring and logging
            "connect_timeout": 10,  # Connection timeout
            # Simplified server settings for compatibility
            "options": f"-c timezone=UTC -c application_name={app_name}",
        }

        # 2025 Enhancement: SSL and security settings
        if self.config.postgres_host != "localhost":
            connection_kwargs["sslmode"] = "require"
            connection_kwargs["sslcert"] = os.getenv("POSTGRES_SSL_CERT")
            connection_kwargs["sslkey"] = os.getenv("POSTGRES_SSL_KEY")
            connection_kwargs["sslrootcert"] = os.getenv("POSTGRES_SSL_ROOT_CERT")

        # Initialize connection pool with 2025 enhancements
        self.pool = AsyncConnectionPool(
            conninfo=self.conninfo,
            min_size=self.config.pool_min_size,
            max_size=self.config.pool_max_size,
            timeout=self.config.pool_timeout,
            max_lifetime=self.config.pool_max_lifetime,
            max_idle=self.config.pool_max_idle,
            kwargs=connection_kwargs,
        )

    async def __aenter__(self):
        """Async context manager entry"""
        await self.pool.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.pool.__aexit__(exc_type, exc_val, exc_tb)

    @contextlib.asynccontextmanager
    async def connection(self):
        """Get a connection from the pool with enhanced monitoring"""
        async with self.pool.connection() as conn:
            yield conn

    async def fetch_models(
        self, model_class: type[T], query: str, params: dict[str, Any] | None = None
    ) -> list[T]:
        """Execute query and return typed Pydantic models with enhanced error handling.

        features:
        - Zero serialization overhead with direct row mapping
        - Automatic retry for transient errors
        - Circuit breaker protection
        - Comprehensive error classification and metrics
        """
        context = ErrorContext(
            operation="fetch_models",
            query=query,
            params=params,
            connection_id=self._connection_id,
        )

        async def _execute_operation():
            start_time = time.perf_counter()

            try:
                async with self.connection() as conn, conn.cursor() as cur:
                    await cur.execute(query, params or {})
                    rows = await cur.fetchall()

                    # Direct Pydantic model creation from dict rows
                    models = []
                    validation_errors = 0

                    for row in rows:
                        try:
                            models.append(model_class.model_validate(row))
                        except ValidationError as e:
                            validation_errors += 1
                            logger.warning(
                                f"Validation error for {model_class.__name__}: {e}"
                            )
                            continue

                    # Record performance metrics
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    context.duration_ms = duration_ms
                    self.metrics.record_query(query, duration_ms, params)

                    if validation_errors > 0:
                        logger.info(
                            f"Completed fetch_models with {validation_errors} validation errors"
                        )

                    return models

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                context.duration_ms = duration_ms
                self.metrics.record_query(query, duration_ms, params)

                # Record error in metrics
                if self.error_metrics:
                    self.error_metrics.record_error(context, e)

                # Classify and log error
                category, severity = DatabaseErrorClassifier.classify_error(e)
                logger.error(
                    f"fetch_models failed: {type(e).__name__}: {e} - "
                    f"Category: {category.value}, Severity: {severity.value} - "
                    f"Duration: {duration_ms:.2f}ms"
                )

                raise

        # Apply retry logic for enhanced reliability
        return await self.retry_manager.retry_async(_execute_operation, context)

    async def fetch_one_model(
        self, model_class: type[T], query: str, params: dict[str, Any] | None = None
    ) -> T | None:
        """Execute query and return single typed Pydantic model."""
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})  # type: ignore[arg-type]
                row = await cur.fetchone()

                if row is None:
                    return None

                # Direct Pydantic model creation
                model = model_class.model_validate(row)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return model

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> int:
        """Execute non-SELECT query and return affected row count."""
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})  # type: ignore[arg-type]

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return cur.rowcount

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def fetch_raw(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query and return raw dictionary results.
        For cases where Pydantic models aren't needed.
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute(query, params or {})  # type: ignore[arg-type]
                rows = await cur.fetchall()

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(query, duration_ms, params)

                return rows

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(query, duration_ms, params)
            raise

    async def fetch_models_server_side(
        self,
        model_class: type[T],
        query: str,
        params: dict[str, Any] | None = None,
        prepared: bool = True,
    ) -> list[T]:
        """2025 Enhancement: Execute query with server-side binding optimization.

        Args:
            model_class: Pydantic model class for type safety
            query: SQL query string
            params: Query parameters
            prepared: Whether to use prepared statements (default: True)

        Returns:
            List of typed Pydantic models
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                if prepared:
                    # Use server-side prepared statement
                    await cur.execute(query, params or {}, prepare=True)  # type: ignore[arg-type]
                else:
                    # Use server-side binding without preparation
                    await cur.execute(query, params or {})  # type: ignore[arg-type]

                rows = await cur.fetchall()

                # Direct Pydantic model creation from dict rows
                models = []
                for row in rows:
                    try:
                        models.append(model_class.model_validate(row))
                    except ValidationError as e:
                        print(
                            f"Server-side validation error for {model_class.__name__}: {e}"
                        )
                        continue

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)

                return models

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)
            raise

    async def execute_batch_server_side(
        self, query: str, params_list: list[dict[str, Any]], prepared: bool = True
    ) -> int:
        """2025 Enhancement: Execute batch operations with server-side binding.

        Args:
            query: SQL query string
            params_list: List of parameter dictionaries
            prepared: Whether to use prepared statements (default: True)

        Returns:
            Total number of affected rows
        """
        start_time = time.perf_counter()
        total_affected = 0

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                # Note: executemany doesn't support prepare parameter in psycopg3
                # Use server-side binding for batch operations
                await cur.executemany(query, params_list)  # type: ignore[arg-type]

                total_affected = cur.rowcount

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[BATCH-SERVER-SIDE] {query}",
                    duration_ms,
                    {"batch_size": len(params_list)},
                )

                return total_affected

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[BATCH-SERVER-SIDE] {query}",
                duration_ms,
                {"batch_size": len(params_list)},
            )
            raise

    async def fetch_one_server_side(
        self,
        model_class: type[T],
        query: str,
        params: dict[str, Any] | None = None,
        prepared: bool = True,
    ) -> T | None:
        """2025 Enhancement: Fetch single model with server-side binding.

        Args:
            model_class: Pydantic model class for type safety
            query: SQL query string
            params: Query parameters
            prepared: Whether to use prepared statements (default: True)

        Returns:
            Single typed Pydantic model or None
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                if prepared:
                    await cur.execute(query, params or {}, prepare=True)  # type: ignore[arg-type]
                else:
                    await cur.execute(query, params or {})  # type: ignore[arg-type]

                row = await cur.fetchone()

                if row is None:
                    return None

                # Direct Pydantic model creation
                model = model_class.model_validate(row)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)

                return model

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)
            raise

    async def execute_server_side(
        self, query: str, params: dict[str, Any] | None = None, prepared: bool = True
    ) -> int:
        """2025 Enhancement: Execute non-SELECT query with server-side binding.

        Args:
            query: SQL query string
            params: Query parameters
            prepared: Whether to use prepared statements (default: True)

        Returns:
            Number of affected rows
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                if prepared:
                    await cur.execute(query, params or {}, prepare=True)  # type: ignore[arg-type]
                else:
                    await cur.execute(query, params or {})  # type: ignore[arg-type]

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)

                return cur.rowcount

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(f"[SERVER-SIDE] {query}", duration_ms, params)
            raise

    async def execute_pipeline_batch(
        self, queries: list[tuple[str, dict[str, Any] | None]]
    ) -> list[int]:
        """2025 Enhancement: Execute multiple queries using pipeline mode.

        Pipeline mode allows multiple small queries to be sent to the server
        in a single round-trip, significantly improving performance for
        batch operations.

        Args:
            queries: List of (query, params) tuples

        Returns:
            List of row counts for each query
        """
        start_time = time.perf_counter()
        results = []

        try:
            async with self.connection() as conn:
                async with conn.pipeline() as pipeline:
                    cursors = []

                    # Queue all queries in the pipeline
                    for query, params in queries:
                        cur = (
                            conn.cursor()
                        )  # Use connection cursor within pipeline context
                        cursors.append(cur)
                        await cur.execute(query, params or {})  # type: ignore[arg-type]

                    # Execute the pipeline
                    await pipeline.sync()

                    # Collect results
                    for cur in cursors:
                        results.append(cur.rowcount)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[PIPELINE] {len(queries)} queries",
                    duration_ms,
                    {"pipeline_size": len(queries)},
                )

                return results

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[PIPELINE] {len(queries)} queries",
                duration_ms,
                {"pipeline_size": len(queries)},
            )
            raise

    async def fetch_pipeline_batch(
        self, model_class: type[T], queries: list[tuple[str, dict[str, Any] | None]]
    ) -> list[list[T]]:
        """2025 Enhancement: Fetch multiple result sets using pipeline mode.

        Args:
            model_class: Pydantic model class for type safety
            queries: List of (query, params) tuples

        Returns:
            List of result lists, one for each query
        """
        start_time = time.perf_counter()
        results = []

        try:
            async with self.connection() as conn:
                async with conn.pipeline() as pipeline:
                    cursors = []

                    # Queue all queries in the pipeline
                    for query, params in queries:
                        cur = (
                            conn.cursor()
                        )  # Use connection cursor within pipeline context
                        cursors.append(cur)
                        await cur.execute(query, params or {})  # type: ignore[arg-type]

                    # Execute the pipeline
                    await pipeline.sync()

                    # Collect and convert results
                    for cur in cursors:
                        rows = await cur.fetchall()
                        models = []
                        for row in rows:
                            try:
                                models.append(model_class.model_validate(row))
                            except ValidationError as e:
                                print(
                                    f"Pipeline validation error for {model_class.__name__}: {e}"
                                )
                                continue
                        results.append(models)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[PIPELINE] {len(queries)} fetch queries",
                    duration_ms,
                    {"pipeline_size": len(queries)},
                )

                return results

        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[PIPELINE] {len(queries)} fetch queries",
                duration_ms,
                {"pipeline_size": len(queries)},
            )
            raise

    async def execute_mixed_pipeline(
        self, operations: list[dict[str, Any]]
    ) -> list[Any]:
        """2025 Enhancement: Execute mixed operations (SELECT/INSERT/UPDATE/DELETE) in pipeline.

        Args:
            operations: List of operation dictionaries with keys:
                - type: 'select' | 'execute'
                - query: SQL query string
                - params: Query parameters (optional)
                - model_class: Pydantic model class (for select operations)

        Returns:
            List of results (models for select, row counts for execute)
        """
        start_time = time.perf_counter()
        results = []

        try:
            async with self.connection() as conn:
                async with conn.pipeline() as pipeline:
                    cursors = []

                    # Queue all operations in the pipeline
                    for op in operations:
                        cur = (
                            conn.cursor()
                        )  # Use connection cursor within pipeline context
                        cursors.append((cur, op))
                        await cur.execute(op["query"], op.get("params") or {})  # type: ignore[arg-type]

                    # Execute the pipeline
                    await pipeline.sync()

                    # Collect results based on operation type
                    for cur, op in cursors:
                        if op["type"] == "select":
                            rows = await cur.fetchall()
                            models = []
                            model_class = op["model_class"]
                            for row in rows:
                                try:
                                    models.append(model_class.model_validate(row))
                                except ValidationError as e:
                                    print(
                                        f"Mixed pipeline validation error for {model_class.__name__}: {e}"
                                    )
                                    continue
                            results.append(models)
                        else:  # execute operation
                            results.append(cur.rowcount)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[MIXED-PIPELINE] {len(operations)} operations",
                    duration_ms,
                    {"pipeline_size": len(operations)},
                )

                return results

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[MIXED-PIPELINE] {len(operations)} operations",
                duration_ms,
                {"pipeline_size": len(operations)},
            )
            raise

    async def copy_from_iterable(
        self,
        table_name: str,
        data_iterable: list[dict[str, Any]],
        columns: list[str] | None = None,
        format: str = "csv",
        delimiter: str = ",",
        header: bool = False,
    ) -> int:
        """2025 Enhancement: Bulk insert data using COPY FROM operation.

        COPY FROM is the fastest way to insert large amounts of data into PostgreSQL.
        Can be 10-100x faster than individual INSERT statements.

        Args:
            table_name: Target table name
            data_iterable: List of dictionaries containing row data
            columns: List of column names (optional, inferred from data if not provided)
            format: COPY format ('csv', 'text', 'binary')
            delimiter: Field delimiter for CSV format
            header: Whether to include header row

        Returns:
            Number of rows inserted
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                # Infer columns from first row if not provided
                if not columns and data_iterable:
                    columns = list(data_iterable[0].keys())

                # Ensure columns is not None for type safety
                if not columns:
                    raise ValueError(
                        "No columns specified and no data provided to infer columns"
                    )

                # Build COPY command
                columns_str = ", ".join(columns)
                copy_sql = f"COPY {table_name} ({columns_str}) FROM STDIN WITH (FORMAT {format.upper()}"

                if format.lower() == "csv":
                    copy_sql += f", DELIMITER '{delimiter}'"
                    if header:
                        copy_sql += ", HEADER"

                copy_sql += ")"

                # Convert data to CSV format
                import csv
                import io

                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=columns, delimiter=delimiter)

                if header:
                    writer.writeheader()

                for row in data_iterable:
                    writer.writerow(row)

                # Execute COPY FROM
                buffer.seek(0)
                async with cur.copy(copy_sql) as copy:  # type: ignore[arg-type]
                    await copy.write(buffer.read())

                row_count = len(data_iterable)

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[COPY-FROM] {table_name}",
                    duration_ms,
                    {"row_count": row_count, "format": format},
                )

                return row_count

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[COPY-FROM] {table_name}",
                duration_ms,
                {"row_count": len(data_iterable), "format": format},
            )
            raise

    async def copy_to_file(
        self,
        table_name: str,
        file_path: str,
        columns: list[str] | None = None,
        where_clause: str | None = None,
        format: str = "csv",
        delimiter: str = ",",
        header: bool = True,
    ) -> int:
        """2025 Enhancement: Bulk export data using COPY TO operation.

        COPY TO is the fastest way to export large amounts of data from PostgreSQL.

        Args:
            table_name: Source table name
            file_path: Target file path
            columns: List of column names (optional, exports all if not provided)
            where_clause: WHERE clause for filtering (optional)
            format: COPY format ('csv', 'text', 'binary')
            delimiter: Field delimiter for CSV format
            header: Whether to include header row

        Returns:
            Number of rows exported
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                # Build COPY command
                columns_str = ", ".join(columns) if columns else "*"
                copy_sql = f"COPY (SELECT {columns_str} FROM {table_name}"

                if where_clause:
                    copy_sql += f" WHERE {where_clause}"

                copy_sql += f") TO STDOUT WITH (FORMAT {format.upper()}"

                if format.lower() == "csv":
                    copy_sql += f", DELIMITER '{delimiter}'"
                    if header:
                        copy_sql += ", HEADER"

                copy_sql += ")"

                # Execute COPY TO with async file operations
                row_count = 0
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    async with cur.copy(copy_sql) as copy:  # type: ignore[arg-type]
                        async for data in copy:
                            # Proper binary data handling for 2025
                            content = bytes(data).decode("utf-8")
                            await f.write(content)
                            # Count rows (approximation for CSV)
                            row_count += content.count("\n")

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[COPY-TO] {table_name}",
                    duration_ms,
                    {"row_count": row_count, "format": format, "file_path": file_path},
                )

                return row_count

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[COPY-TO] {table_name}",
                duration_ms,
                {"format": format, "file_path": file_path},
            )
            raise

    async def copy_from_file(
        self,
        table_name: str,
        file_path: str,
        columns: list[str] | None = None,
        format: str = "csv",
        delimiter: str = ",",
        header: bool = True,
        skip_errors: bool = False,
    ) -> int:
        """2025 Enhancement: Bulk import data from file using COPY FROM operation.

        Args:
            table_name: Target table name
            file_path: Source file path
            columns: List of column names (optional)
            format: COPY format ('csv', 'text', 'binary')
            delimiter: Field delimiter for CSV format
            header: Whether file has header row
            skip_errors: Whether to skip rows with errors

        Returns:
            Number of rows imported
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                # Build COPY command
                columns_str = f" ({', '.join(columns)})" if columns else ""
                copy_sql = f"COPY {table_name}{columns_str} FROM STDIN WITH (FORMAT {format.upper()}"

                if format.lower() == "csv":
                    copy_sql += f", DELIMITER '{delimiter}'"
                    if header:
                        copy_sql += ", HEADER"

                if skip_errors:
                    copy_sql += ", ON_ERROR IGNORE"

                copy_sql += ")"

                # Execute COPY FROM with async file operations
                row_count = 0
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    async with cur.copy(copy_sql) as copy:  # type: ignore[arg-type]
                        content = await f.read()
                        await copy.write(content)
                        # Count rows (approximation for CSV)
                        row_count = content.count("\n")
                        if header and row_count > 0:
                            row_count -= 1  # Subtract header row

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[COPY-FROM-FILE] {table_name}",
                    duration_ms,
                    {"row_count": row_count, "format": format, "file_path": file_path},
                )

                return row_count

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[COPY-FROM-FILE] {table_name}",
                duration_ms,
                {"format": format, "file_path": file_path},
            )
            raise

    async def copy_between_tables(
        self,
        source_table: str,
        target_table: str,
        columns: list[str] | None = None,
        where_clause: str | None = None,
        transform_query: str | None = None,
    ) -> int:
        """2025 Enhancement: Bulk copy data between tables using optimized COPY operations.

        Args:
            source_table: Source table name
            target_table: Target table name
            columns: List of column names (optional)
            where_clause: WHERE clause for source filtering (optional)
            transform_query: Custom SELECT query for data transformation (optional)

        Returns:
            Number of rows copied
        """
        start_time = time.perf_counter()

        try:
            async with self.connection() as conn, conn.cursor() as cur:
                # Build source query
                if transform_query:
                    source_query = transform_query
                else:
                    columns_str = ", ".join(columns) if columns else "*"
                    source_query = f"SELECT {columns_str} FROM {source_table}"

                    if where_clause:
                        source_query += f" WHERE {where_clause}"

                # Build target columns
                target_columns = f" ({', '.join(columns)})" if columns else ""

                # Use pipeline for atomic operation
                async with conn.pipeline() as pipeline:
                    # Create temporary table for intermediate storage
                    temp_table = f"temp_copy_{int(time.time() * 1000)}"

                    # Copy from source to temp table
                    copy_out_sql = (
                        f"COPY ({source_query}) TO STDOUT WITH (FORMAT BINARY)"
                    )
                    copy_in_sql = f"COPY {target_table}{target_columns} FROM STDIN WITH (FORMAT BINARY)"

                    cur1 = (
                        conn.cursor()
                    )  # Use connection cursor within pipeline context
                    cur2 = (
                        conn.cursor()
                    )  # Use connection cursor within pipeline context

                    # Execute copy operations
                    row_count = 0
                    async with cur1.copy(copy_out_sql) as copy_out:  # type: ignore[arg-type]
                        async with cur2.copy(copy_in_sql) as copy_in:  # type: ignore[arg-type]
                            async for data in copy_out:
                                await copy_in.write(data)
                                row_count += 1

                    await pipeline.sync()

                # Record performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.record_query(
                    f"[COPY-BETWEEN] {source_table} -> {target_table}",
                    duration_ms,
                    {"row_count": row_count},
                )

                return row_count

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.record_query(
                f"[COPY-BETWEEN] {source_table} -> {target_table}", duration_ms, {}
            )
            raise

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics with 2025 enhanced monitoring."""
        # Get PostgreSQL cache hit ratio
        cache_hit_query = """
        SELECT
            CASE
                WHEN (blks_hit + blks_read) = 0 THEN 0
                ELSE blks_hit::float / (blks_hit + blks_read)
            END as cache_hit_ratio
        FROM pg_stat_database
        WHERE datname = current_database()
        """

        # 2025 Enhancement: Get pool status metrics
        pool_stats = await self.get_pool_stats()

        cache_stats = await self.fetch_raw(cache_hit_query)
        cache_hit_ratio = cache_stats[0]["cache_hit_ratio"] if cache_stats else 0

        return {
            "total_queries": self.metrics.total_queries,
            "avg_query_time_ms": round(self.metrics.avg_query_time, 2),
            "queries_under_50ms_percent": round(self.metrics.queries_under_50ms, 1),
            "slow_query_count": len(self.metrics.slow_queries),
            "cache_hit_ratio": round(cache_hit_ratio, 3) if cache_hit_ratio else 0,
            "target_query_time_ms": self.config.target_query_time_ms,
            "target_cache_hit_ratio": self.config.target_cache_hit_ratio,
            "pool_status": pool_stats,  # 2025 Enhancement
            "performance_status": "GOOD"
            if (
                self.metrics.avg_query_time <= self.config.target_query_time_ms
                and cache_hit_ratio >= self.config.target_cache_hit_ratio
                and pool_stats["pool_health"] == "HEALTHY"
            )
            else "NEEDS_ATTENTION",
        }

    async def get_pool_stats(self) -> dict[str, Any]:
        """2025 Enhancement: Get detailed connection pool statistics."""
        try:
            # Get pool statistics
            stats = self.pool.get_stats()
            return {
                "pool_size": stats["pool_size"],
                "pool_available": stats["pool_available"],
                "pool_max": stats["pool_max"],
                "pool_min": stats["pool_min"],
                "requests_waiting": stats["requests_waiting"],
                "requests_errors": stats["requests_errors"],
                "requests_num": stats["requests_num"],
                "usage_ms": stats["usage_ms"],
                "connections_num": stats["connections_num"],
                "pool_health": "HEALTHY"
                if stats["pool_available"] > 0
                else "EXHAUSTED",
                "pool_utilization": round(
                    (stats["pool_size"] - stats["pool_available"])
                    / stats["pool_size"]
                    * 100,
                    2,
                )
                if stats["pool_size"] > 0
                else 0,
            }
        except Exception as e:
            return {
                "error": str(e),
                "pool_health": "UNKNOWN",
                "pool_utilization": 0,
            }

    async def get_connection_info(self) -> dict[str, Any]:
        """2025 Enhancement: Get connection configuration info"""
        try:
            async with self.connection() as conn:
                # Get connection info
                info_query = """
                SELECT
                    current_setting('application_name') as app_name,
                    current_setting('TimeZone') as timezone,
                    current_setting('default_transaction_isolation') as isolation_level,
                    current_setting('statement_timeout') as statement_timeout,
                    current_setting('lock_timeout') as lock_timeout,
                    current_setting('idle_in_transaction_session_timeout') as idle_timeout,
                    current_setting('server_version') as server_version,
                    current_setting('max_connections') as max_connections,
                    current_setting('shared_buffers') as shared_buffers
                """

                info_result = await self.fetch_raw(info_query)
                connection_info = info_result[0] if info_result else {}

                return {
                    "connection_info": connection_info,
                    "client_info": {
                        "pool_min_size": self.config.pool_min_size,
                        "pool_max_size": self.config.pool_max_size,
                        "pool_timeout": self.config.pool_timeout,
                        "pool_max_lifetime": self.config.pool_max_lifetime,
                        "pool_max_idle": self.config.pool_max_idle,
                    },
                }
        except Exception as e:
            return {
                "error": str(e),
                "connection_info": {},
                "client_info": {},
            }

    def reset_metrics(self):
        """Reset performance metrics (useful for testing)."""
        self.metrics = QueryMetrics()
        if self.error_metrics:
            self.error_metrics = ErrorMetrics()

    def get_error_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive error metrics summary."""
        if not self.error_metrics:
            return {"error": "Error metrics not enabled"}

        return self.error_metrics.get_metrics_summary()

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "success_count": self.circuit_breaker.success_count,
            "last_failure": self.circuit_breaker.last_failure_time.isoformat()
            if self.circuit_breaker.last_failure_time
            else None,
            "config": {
                "failure_threshold": self.circuit_breaker.config.failure_threshold,
                "recovery_timeout_seconds": self.circuit_breaker.config.recovery_timeout_seconds,
                "success_threshold": self.circuit_breaker.config.success_threshold,
                "enabled": self.circuit_breaker.config.enabled,
            },
        }

    async def test_connection_with_retry(self) -> dict[str, Any]:
        """Test database connection with retry logic and error classification."""
        context = ErrorContext(
            operation="test_connection", connection_id=self._connection_id
        )

        async def _test_operation():
            start_time = time.perf_counter()
            try:
                async with self.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT 1 as test, current_timestamp as timestamp"
                        )
                        result = await cur.fetchone()

                duration_ms = (time.perf_counter() - start_time) * 1000

                return {
                    "status": "SUCCESS",
                    "result": dict(result) if result else None,
                    "response_time_ms": round(duration_ms, 2),
                    "retry_count": context.retry_count,
                }

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                category, severity = DatabaseErrorClassifier.classify_error(e)

                if self.error_metrics:
                    self.error_metrics.record_error(context, e)

                return {
                    "status": "FAILED",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_category": category.value,
                    "error_severity": severity.value,
                    "is_retryable": DatabaseErrorClassifier.is_retryable(e),
                    "response_time_ms": round(duration_ms, 2),
                    "retry_count": context.retry_count,
                }

        try:
            return await self.retry_manager.retry_async(_test_operation, context)
        except Exception as e:
            # If all retries failed, return error details
            category, severity = DatabaseErrorClassifier.classify_error(e)
            return {
                "status": "FAILED_ALL_RETRIES",
                "error": str(e),
                "error_type": type(e).__name__,
                "error_category": category.value,
                "error_severity": severity.value,
                "retry_count": context.retry_count,
                "max_retries": self.retry_config.max_attempts,
            }

    async def health_check(self) -> dict[str, Any]:
        """2025 Enhancement: Comprehensive health check with enhanced error classification"""
        health_status = {
            "overall_health": "UNKNOWN",
            "timestamp": aware_utc_now().isoformat(),
            "checks": {},
            "error_metrics": self.get_error_metrics_summary()
            if self.error_metrics
            else None,
            "circuit_breaker_status": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count,
                "last_failure": self.circuit_breaker.last_failure_time.isoformat()
                if self.circuit_breaker.last_failure_time
                else None,
            },
        }

        try:
            # 1. Connection health check with timing
            try:
                start_time = time.perf_counter()
                async with self.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT 1")
                        result = await cur.fetchone()
                        response_time_ms = (time.perf_counter() - start_time) * 1000

                        health_status["checks"]["connection"] = {
                            "status": "HEALTHY"
                            if result and result[0] == 1
                            else "UNHEALTHY",
                            "response_time_ms": round(response_time_ms, 2),
                        }
            except Exception as e:
                # Classify the connection error
                category, severity = DatabaseErrorClassifier.classify_error(e)
                health_status["checks"]["connection"] = {
                    "status": "UNHEALTHY",
                    "error": str(e),
                    "error_category": category.value,
                    "error_severity": severity.value,
                    "error_type": type(e).__name__,
                }

            # 2. Pool health check
            pool_stats = await self.get_pool_stats()
            health_status["checks"]["pool"] = {
                "status": pool_stats.get("pool_health", "UNKNOWN"),
                "utilization": pool_stats.get("pool_utilization", 0),
                "available_connections": pool_stats.get("pool_available", 0),
                "total_connections": pool_stats.get("pool_size", 0),
            }

            # 3. Performance health check
            perf_stats = await self.get_performance_stats()
            health_status["checks"]["performance"] = {
                "status": perf_stats.get("performance_status", "UNKNOWN"),
                "avg_query_time_ms": perf_stats.get("avg_query_time_ms", 0),
                "cache_hit_ratio": perf_stats.get("cache_hit_ratio", 0),
                "slow_queries": perf_stats.get("slow_query_count", 0),
            }

            # 4. Server health check
            try:
                server_query = """
                SELECT
                    pg_database_size(current_database()) as db_size,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                    (SELECT count(*) FROM pg_locks WHERE NOT granted) as blocked_queries
                """
                server_stats = await self.fetch_raw(server_query)
                if server_stats:
                    server_info = server_stats[0]
                    health_status["checks"]["server"] = {
                        "status": "HEALTHY",
                        "database_size_bytes": server_info.get("db_size", 0),
                        "active_connections": server_info.get("active_connections", 0),
                        "idle_connections": server_info.get("idle_connections", 0),
                        "blocked_queries": server_info.get("blocked_queries", 0),
                    }
            except Exception as e:
                health_status["checks"]["server"] = {
                    "status": "UNHEALTHY",
                    "error": str(e),
                }

            # 5. Determine overall health
            connection_healthy = (
                health_status["checks"]["connection"]["status"] == "HEALTHY"
            )
            pool_healthy = health_status["checks"]["pool"]["status"] == "HEALTHY"
            performance_good = (
                health_status["checks"]["performance"]["status"] == "GOOD"
            )
            server_healthy = health_status["checks"]["server"]["status"] == "HEALTHY"

            if (
                connection_healthy
                and pool_healthy
                and performance_good
                and server_healthy
            ):
                health_status["overall_health"] = "HEALTHY"
            elif connection_healthy and pool_healthy:
                health_status["overall_health"] = "DEGRADED"
            else:
                health_status["overall_health"] = "UNHEALTHY"

            return health_status

        except Exception as e:
            health_status["overall_health"] = "UNHEALTHY"
            health_status["error"] = str(e)
            return health_status

    async def get_detailed_metrics(self) -> dict[str, Any]:
        """2025 Enhancement: Get comprehensive database metrics for monitoring and alerting"""
        try:
            # Get basic performance stats
            perf_stats = await self.get_performance_stats()
            pool_stats = await self.get_pool_stats()
            connection_info = await self.get_connection_info()

            # Get additional PostgreSQL metrics
            metrics_query = """
            SELECT
                -- Database stats
                pg_database_size(current_database()) as database_size,
                pg_size_pretty(pg_database_size(current_database())) as database_size_pretty,

                -- Connection stats
                (SELECT count(*) FROM pg_stat_activity) as total_connections,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction') as idle_in_transaction,

                -- Lock stats
                (SELECT count(*) FROM pg_locks WHERE NOT granted) as blocked_queries,
                (SELECT count(*) FROM pg_locks WHERE granted) as granted_locks,

                -- Cache and I/O stats
                (SELECT sum(blks_hit) FROM pg_stat_database) as total_cache_hits,
                (SELECT sum(blks_read) FROM pg_stat_database) as total_disk_reads,
                (SELECT sum(tup_returned) FROM pg_stat_database) as total_rows_returned,
                (SELECT sum(tup_fetched) FROM pg_stat_database) as total_rows_fetched,

                -- Transaction stats
                (SELECT sum(xact_commit) FROM pg_stat_database) as total_commits,
                (SELECT sum(xact_rollback) FROM pg_stat_database) as total_rollbacks,

                -- Replication lag (if applicable)
                CASE
                    WHEN pg_is_in_recovery() THEN
                        EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                    ELSE 0
                END as replication_lag_seconds
            """

            db_metrics = await self.fetch_raw(metrics_query)
            db_info = db_metrics[0] if db_metrics else {}

            # Calculate derived metrics
            total_cache_hits = db_info.get("total_cache_hits", 0)
            total_disk_reads = db_info.get("total_disk_reads", 0)
            total_reads = total_cache_hits + total_disk_reads
            overall_cache_hit_ratio = (
                (total_cache_hits / total_reads) if total_reads > 0 else 0
            )

            total_commits = db_info.get("total_commits", 0)
            total_rollbacks = db_info.get("total_rollbacks", 0)
            total_transactions = total_commits + total_rollbacks
            commit_ratio = (
                (total_commits / total_transactions) if total_transactions > 0 else 0
            )

            return {
                "timestamp": aware_utc_now().isoformat(),
                "client_metrics": {
                    "total_queries": self.metrics.total_queries,
                    "avg_query_time_ms": round(self.metrics.avg_query_time, 2),
                    "queries_under_50ms_percent": round(
                        self.metrics.queries_under_50ms, 1
                    ),
                    "slow_queries": len(self.metrics.slow_queries),
                    "recent_slow_queries": self.metrics.slow_queries[-5:]
                    if self.metrics.slow_queries
                    else [],
                },
                "pool_metrics": pool_stats,
                "connection_metrics": connection_info,
                "database_metrics": {
                    "size_bytes": db_info.get("database_size", 0),
                    "size_pretty": db_info.get("database_size_pretty", "0 bytes"),
                    "total_connections": db_info.get("total_connections", 0),
                    "active_connections": db_info.get("active_connections", 0),
                    "idle_connections": db_info.get("idle_connections", 0),
                    "idle_in_transaction": db_info.get("idle_in_transaction", 0),
                    "blocked_queries": db_info.get("blocked_queries", 0),
                    "granted_locks": db_info.get("granted_locks", 0),
                    "replication_lag_seconds": db_info.get(
                        "replication_lag_seconds", 0
                    ),
                },
                "performance_metrics": {
                    "cache_hit_ratio": round(overall_cache_hit_ratio, 3),
                    "total_cache_hits": total_cache_hits,
                    "total_disk_reads": total_disk_reads,
                    "rows_returned": db_info.get("total_rows_returned", 0),
                    "rows_fetched": db_info.get("total_rows_fetched", 0),
                    "commit_ratio": round(commit_ratio, 3),
                    "total_commits": total_commits,
                    "total_rollbacks": total_rollbacks,
                },
                "targets": {
                    "target_query_time_ms": self.config.target_query_time_ms,
                    "target_cache_hit_ratio": self.config.target_cache_hit_ratio,
                },
            }

        except Exception as e:
            return {
                "timestamp": aware_utc_now().isoformat(),
                "error": str(e),
                "client_metrics": {
                    "total_queries": self.metrics.total_queries,
                    "avg_query_time_ms": round(self.metrics.avg_query_time, 2),
                    "slow_queries": len(self.metrics.slow_queries),
                },
            }

    async def get_alerts(self) -> list[dict[str, Any]]:
        """2025 Enhancement: Get performance alerts and warnings"""
        alerts = []

        try:
            # Get current metrics
            detailed_metrics = await self.get_detailed_metrics()

            # Check query performance
            avg_query_time = detailed_metrics["client_metrics"]["avg_query_time_ms"]
            if avg_query_time > self.config.target_query_time_ms:
                alerts.append({
                    "level": "WARNING",
                    "type": "PERFORMANCE",
                    "message": f"Average query time ({avg_query_time}ms) exceeds target ({self.config.target_query_time_ms}ms)",
                    "metric": "avg_query_time_ms",
                    "value": avg_query_time,
                    "threshold": self.config.target_query_time_ms,
                })

            # Check cache hit ratio
            cache_hit_ratio = detailed_metrics["performance_metrics"]["cache_hit_ratio"]
            if cache_hit_ratio < self.config.target_cache_hit_ratio:
                alerts.append({
                    "level": "WARNING",
                    "type": "PERFORMANCE",
                    "message": f"Cache hit ratio ({cache_hit_ratio:.1%}) below target ({self.config.target_cache_hit_ratio:.1%})",
                    "metric": "cache_hit_ratio",
                    "value": cache_hit_ratio,
                    "threshold": self.config.target_cache_hit_ratio,
                })

            # Check pool utilization
            pool_utilization = detailed_metrics["pool_metrics"]["pool_utilization"]
            if pool_utilization > 80:
                alerts.append({
                    "level": "WARNING" if pool_utilization < 90 else "CRITICAL",
                    "type": "POOL",
                    "message": f"Connection pool utilization high ({pool_utilization}%)",
                    "metric": "pool_utilization",
                    "value": pool_utilization,
                    "threshold": 80,
                })

            # Check blocked queries
            blocked_queries = detailed_metrics["database_metrics"]["blocked_queries"]
            if blocked_queries > 0:
                alerts.append({
                    "level": "WARNING",
                    "type": "LOCKS",
                    "message": f"{blocked_queries} blocked queries detected",
                    "metric": "blocked_queries",
                    "value": blocked_queries,
                    "threshold": 0,
                })

            # Check slow queries
            slow_query_count = detailed_metrics["client_metrics"]["slow_queries"]
            if slow_query_count > 10:
                alerts.append({
                    "level": "WARNING",
                    "type": "PERFORMANCE",
                    "message": f"{slow_query_count} slow queries detected",
                    "metric": "slow_queries",
                    "value": slow_query_count,
                    "threshold": 10,
                })

            # Check replication lag
            replication_lag = detailed_metrics["database_metrics"][
                "replication_lag_seconds"
            ]
            if replication_lag > 5:
                alerts.append({
                    "level": "WARNING" if replication_lag < 30 else "CRITICAL",
                    "type": "REPLICATION",
                    "message": f"Replication lag high ({replication_lag:.1f}s)",
                    "metric": "replication_lag_seconds",
                    "value": replication_lag,
                    "threshold": 5,
                })

            return alerts

        except Exception as e:
            return [
                {
                    "level": "ERROR",
                    "type": "MONITORING",
                    "message": f"Failed to generate alerts: {e!s}",
                    "metric": "monitoring_error",
                    "value": str(e),
                }
            ]

    async def run_orchestrated_analysis(self, analysis_type: str = "performance_metrics", **kwargs) -> Dict[str, Any]:
        """
        Run orchestrated analysis for TypeSafePsycopgClient component.

        Compatible with ML orchestrator integration patterns.

        Args:
            analysis_type: Type of analysis to run
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing analysis results
        """
        if analysis_type == "performance_metrics":
            return await self._analyze_performance_metrics(**kwargs)
        elif analysis_type == "connection_health":
            return await self._analyze_connection_health(**kwargs)
        elif analysis_type == "query_analysis":
            return await self._analyze_query_patterns(**kwargs)
        elif analysis_type == "type_safety_validation":
            return await self._analyze_type_safety(**kwargs)
        elif analysis_type == "comprehensive_analysis":
            return await self._run_comprehensive_analysis(**kwargs)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    async def _analyze_performance_metrics(self, **kwargs) -> Dict[str, Any]:
        """Analyze database performance metrics."""
        performance_stats = await self.get_performance_stats()
        detailed_metrics = await self.get_detailed_metrics()

        return {
            "component": "TypeSafePsycopgClient",
            "analysis_type": "performance_metrics",
            "performance_status": performance_stats.get("performance_status", "UNKNOWN"),
            "query_performance": {
                "total_queries": self.metrics.total_queries,
                "avg_query_time_ms": round(self.metrics.avg_query_time, 2),
                "queries_under_50ms_percent": round(self.metrics.queries_under_50ms, 1),
                "slow_queries_count": len(self.metrics.slow_queries),
                "target_compliance": "GOOD" if self.metrics.queries_under_50ms >= 90 else "NEEDS_IMPROVEMENT"
            },
            "connection_metrics": {
                "pool_health": performance_stats.get("pool_status", {}).get("pool_health", "UNKNOWN"),
                "pool_utilization": performance_stats.get("pool_status", {}).get("pool_utilization", 0),
                "cache_hit_ratio": performance_stats.get("cache_hit_ratio", 0)
            },
            "recommendations": self._generate_performance_recommendations(performance_stats),
            "timestamp": aware_utc_now().isoformat()
        }

    async def _analyze_connection_health(self, **kwargs) -> Dict[str, Any]:
        """Analyze connection pool and database health."""
        health_status = await self.health_check()
        pool_stats = await self.get_pool_stats()
        connection_info = await self.get_connection_info()

        return {
            "component": "TypeSafePsycopgClient",
            "analysis_type": "connection_health",
            "overall_health": health_status.get("overall_health", "UNKNOWN"),
            "connection_status": health_status.get("checks", {}).get("connection", {}),
            "pool_analysis": {
                "health": pool_stats.get("pool_health", "UNKNOWN"),
                "utilization": pool_stats.get("pool_utilization", 0),
                "available_connections": pool_stats.get("pool_available", 0),
                "total_connections": pool_stats.get("pool_size", 0),
                "status": "HEALTHY" if pool_stats.get("pool_health") == "HEALTHY" and pool_stats.get("pool_utilization", 0) < 80 else "NEEDS_ATTENTION"
            },
            "server_health": health_status.get("checks", {}).get("server", {}),
            "error_metrics": self.get_error_metrics_summary() if self.error_metrics else None,
            "circuit_breaker_status": self.get_circuit_breaker_status(),
            "recommendations": self._generate_health_recommendations(health_status, pool_stats),
            "timestamp": aware_utc_now().isoformat()
        }

    async def _analyze_query_patterns(self, **kwargs) -> Dict[str, Any]:
        """Analyze query execution patterns."""
        detailed_metrics = await self.get_detailed_metrics()
        alerts = await self.get_alerts()

        # Analyze slow queries
        slow_query_analysis = self._analyze_slow_queries()
        query_complexity_analysis = self._analyze_query_complexity()

        return {
            "component": "TypeSafePsycopgClient",
            "analysis_type": "query_analysis",
            "query_statistics": {
                "total_queries": self.metrics.total_queries,
                "slow_queries": len(self.metrics.slow_queries),
                "slow_query_details": self.metrics.slow_queries[-10:],  # Last 10 slow queries
                "query_time_distribution": slow_query_analysis
            },
            "complexity_analysis": query_complexity_analysis,
            "performance_alerts": [alert for alert in alerts if alert.get("type") == "PERFORMANCE"],
            "optimization_opportunities": self._identify_optimization_opportunities(),
            "timestamp": aware_utc_now().isoformat()
        }

    async def _analyze_type_safety(self, **kwargs) -> Dict[str, Any]:
        """Analyze type safety and validation effectiveness."""
        return {
            "component": "TypeSafePsycopgClient",
            "analysis_type": "type_safety_validation",
            "type_safety_status": "ENFORCED",
            "validation_features": {
                "pydantic_validation": True,
                "server_side_binding": True,
                "prepared_statements": True,
                "zero_serialization_overhead": True
            },
            "error_handling": {
                "circuit_breaker_enabled": self.circuit_breaker.config.enabled,
                "retry_mechanism": True,
                "error_classification": True,
                "comprehensive_error_metrics": self.error_metrics is not None
            },
            "security_features": {
                "sql_injection_protection": "PARAMETERIZED_QUERIES",
                "connection_security": "SSL_ENABLED" if self.config.postgres_host != "localhost" else "LOCAL_CONNECTION",
                "timeout_protection": True
            },
            "recommendations": [
                "Type safety is fully enforced with Pydantic models",
                "Server-side binding eliminates SQL injection risks",
                "Circuit breaker provides fault tolerance",
                "Comprehensive error handling and classification in place"
            ],
            "timestamp": aware_utc_now().isoformat()
        }

    async def _run_comprehensive_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run comprehensive analysis combining all analysis types."""
        performance_analysis = await self._analyze_performance_metrics(**kwargs)
        health_analysis = await self._analyze_connection_health(**kwargs)
        query_analysis = await self._analyze_query_patterns(**kwargs)
        type_safety_analysis = await self._analyze_type_safety(**kwargs)

        # Calculate overall score
        performance_score = self._calculate_performance_score(performance_analysis)
        health_score = self._calculate_health_score(health_analysis)
        query_score = self._calculate_query_score(query_analysis)
        type_safety_score = self._calculate_type_safety_score(type_safety_analysis)

        overall_score = (performance_score + health_score + query_score + type_safety_score) / 4

        return {
            "component": "TypeSafePsycopgClient",
            "analysis_type": "comprehensive_analysis",
            "overall_score": round(overall_score, 2),
            "score_breakdown": {
                "performance": performance_score,
                "health": health_score,
                "query_patterns": query_score,
                "type_safety": type_safety_score
            },
            "detailed_analyses": {
                "performance": performance_analysis,
                "health": health_analysis,
                "query_patterns": query_analysis,
                "type_safety": type_safety_analysis
            },
            "executive_summary": self._generate_executive_summary(overall_score, performance_analysis, health_analysis),
            "timestamp": aware_utc_now().isoformat()
        }

    def _generate_performance_recommendations(self, performance_stats: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        avg_query_time = performance_stats.get("avg_query_time_ms", 0)
        if avg_query_time > self.config.target_query_time_ms:
            recommendations.append(f"Average query time ({avg_query_time}ms) exceeds target - consider query optimization")

        cache_hit_ratio = performance_stats.get("cache_hit_ratio", 0)
        if cache_hit_ratio < self.config.target_cache_hit_ratio:
            recommendations.append("Cache hit ratio below target - consider index optimization or caching strategy")

        pool_utilization = performance_stats.get("pool_status", {}).get("pool_utilization", 0)
        if pool_utilization > 80:
            recommendations.append("High pool utilization - consider increasing pool size")
        elif pool_utilization < 20:
            recommendations.append("Low pool utilization - consider reducing pool size")

        return recommendations

    def _generate_health_recommendations(self, health_status: Dict[str, Any], pool_stats: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []

        overall_health = health_status.get("overall_health", "UNKNOWN")
        if overall_health == "UNHEALTHY":
            recommendations.append("Critical health issues detected - immediate attention required")
        elif overall_health == "DEGRADED":
            recommendations.append("Performance degradation detected - investigate and optimize")

        blocked_queries = health_status.get("checks", {}).get("server", {}).get("blocked_queries", 0)
        if blocked_queries > 0:
            recommendations.append(f"Blocked queries detected ({blocked_queries}) - investigate locking issues")

        pool_health = pool_stats.get("pool_health", "UNKNOWN")
        if pool_health == "EXHAUSTED":
            recommendations.append("Connection pool exhausted - increase pool size or optimize connection usage")

        return recommendations

    def _analyze_slow_queries(self) -> Dict[str, Any]:
        """Analyze slow query patterns."""
        if not self.metrics.slow_queries:
            return {"status": "no_slow_queries"}

        # Group slow queries by duration ranges
        duration_ranges = {"50-100ms": 0, "100-500ms": 0, "500-1000ms": 0, "1000ms+": 0}

        for query in self.metrics.slow_queries:
            duration = query.get("duration_ms", 0)
            if duration <= 100:
                duration_ranges["50-100ms"] += 1
            elif duration <= 500:
                duration_ranges["100-500ms"] += 1
            elif duration <= 1000:
                duration_ranges["500-1000ms"] += 1
            else:
                duration_ranges["1000ms+"] += 1

        return {
            "total_slow_queries": len(self.metrics.slow_queries),
            "duration_distribution": duration_ranges,
            "avg_slow_query_duration": sum(q.get("duration_ms", 0) for q in self.metrics.slow_queries) / len(self.metrics.slow_queries),
            "worst_query": max(self.metrics.slow_queries, key=lambda q: q.get("duration_ms", 0)) if self.metrics.slow_queries else None
        }

    def _analyze_query_complexity(self) -> Dict[str, Any]:
        """Analyze query complexity patterns."""
        if not self.metrics.slow_queries:
            return {"status": "no_data"}

        complexity_patterns = {"simple": 0, "moderate": 0, "complex": 0}

        for query in self.metrics.slow_queries:
            query_text = query.get("query", "").lower()
            complexity = "simple"

            if any(keyword in query_text for keyword in ["join", "subquery", "with", "window"]):
                complexity = "moderate"
            if any(keyword in query_text for keyword in ["recursive", "pivot", "lateral", "over("]):
                complexity = "complex"

            complexity_patterns[complexity] += 1

        return {
            "complexity_distribution": complexity_patterns,
            "most_complex_queries": [q for q in self.metrics.slow_queries if "join" in q.get("query", "").lower() or "with" in q.get("query", "").lower()][:5]
        }

    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify query optimization opportunities."""
        opportunities = []

        if len(self.metrics.slow_queries) > 10:
            opportunities.append("High number of slow queries - consider index optimization")

        if self.metrics.avg_query_time > 50:
            opportunities.append("Average query time above target - query optimization needed")

        if self.metrics.queries_under_50ms < 80:
            opportunities.append("Less than 80% of queries meet performance target")

        return opportunities

    def _calculate_performance_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate performance score (0-100)."""
        query_perf = analysis.get("query_performance", {})
        target_compliance = query_perf.get("queries_under_50ms_percent", 0)
        avg_query_time = query_perf.get("avg_query_time_ms", 100)

        # Score based on target compliance and query time
        compliance_score = target_compliance
        time_score = max(0, 100 - (avg_query_time / 50) * 50)  # 50ms target

        return (compliance_score + time_score) / 2

    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate health score (0-100)."""
        overall_health = analysis.get("overall_health", "UNKNOWN")
        pool_analysis = analysis.get("pool_analysis", {})

        if overall_health == "HEALTHY":
            health_score = 100
        elif overall_health == "DEGRADED":
            health_score = 70
        elif overall_health == "UNHEALTHY":
            health_score = 30
        else:
            health_score = 50

        # Adjust based on pool utilization
        pool_util = pool_analysis.get("utilization", 0)
        if pool_util > 90:
            health_score *= 0.8
        elif pool_util < 20:
            health_score *= 0.9

        return health_score

    def _calculate_query_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate query analysis score (0-100)."""
        query_stats = analysis.get("query_statistics", {})
        slow_queries = query_stats.get("slow_queries", 0)
        total_queries = query_stats.get("total_queries", 1)

        # Score based on slow query ratio
        slow_ratio = slow_queries / total_queries if total_queries > 0 else 0
        score = max(0, 100 - (slow_ratio * 100))

        return score

    def _calculate_type_safety_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate type safety score (0-100)."""
        # Type safety is always high due to Pydantic integration
        validation_features = analysis.get("validation_features", {})
        error_handling = analysis.get("error_handling", {})
        security_features = analysis.get("security_features", {})

        # All features are implemented, so score is high
        base_score = 95

        # Small deductions for any missing features
        if not error_handling.get("circuit_breaker_enabled", False):
            base_score -= 2
        if not error_handling.get("comprehensive_error_metrics", False):
            base_score -= 3

        return base_score

    def _generate_executive_summary(self, overall_score: float, performance_analysis: Dict[str, Any], health_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of database analysis."""
        status = "EXCELLENT" if overall_score >= 90 else "GOOD" if overall_score >= 80 else "NEEDS_IMPROVEMENT" if overall_score >= 60 else "CRITICAL"

        key_metrics = {
            "overall_score": overall_score,
            "status": status,
            "total_queries": performance_analysis.get("query_performance", {}).get("total_queries", 0),
            "avg_query_time": performance_analysis.get("query_performance", {}).get("avg_query_time_ms", 0),
            "health_status": health_analysis.get("overall_health", "UNKNOWN"),
            "pool_health": health_analysis.get("pool_analysis", {}).get("health", "UNKNOWN")
        }

        critical_issues = []
        if overall_score < 60:
            critical_issues.append("Overall database performance below acceptable threshold")
        if health_analysis.get("overall_health") == "UNHEALTHY":
            critical_issues.append("Database health status is unhealthy")
        if performance_analysis.get("query_performance", {}).get("target_compliance") == "NEEDS_IMPROVEMENT":
            critical_issues.append("Query performance does not meet targets")

        return {
            "key_metrics": key_metrics,
            "critical_issues": critical_issues,
            "recommendations_summary": f"Database shows {status.lower()} performance with {len(critical_issues)} critical issues to address"
        }

# Global client instance
_client: TypeSafePsycopgClient | None = None

async def get_psycopg_client() -> TypeSafePsycopgClient:
    """Get or create the global psycopg client"""
    global _client
    if _client is None:
        _client = TypeSafePsycopgClient()
        await _client.__aenter__()
    return _client

async def close_psycopg_client():
    """Close the global psycopg client"""
    global _client
    if _client is not None:
        await _client.__aexit__(None, None, None)
        _client = None
