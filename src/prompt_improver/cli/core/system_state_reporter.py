"""
System State Reporter for Week 4 Smart Initialization
Provides comprehensive system state reporting with component status, data metrics, and recommendations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class SystemStateReporter:
    """
    Comprehensive system state reporting service implementing 2025 best practices.
    
    Features:
    - Rich console output with color-coded status indicators
    - Component health visualization
    - Data metrics dashboard
    - Intelligent recommendations display
    - Export capabilities for system state
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    def generate_comprehensive_report(self, initialization_results: Dict[str, Any]) -> None:
        """
        Generate and display comprehensive system state report.
        
        Args:
            initialization_results: Results from smart initialization
        """
        self.console.print("\n🔍 APES Training System - Comprehensive State Report", style="bold blue")
        self.console.print("=" * 80, style="dim")
        
        # Overall status summary
        self._display_overall_status(initialization_results)
        
        # Component status details
        self._display_component_status(initialization_results)
        
        # Database and rule status
        self._display_database_status(initialization_results)
        
        # Data availability and quality
        self._display_data_status(initialization_results)
        
        # Initialization plan and execution
        self._display_initialization_details(initialization_results)
        
        # Recommendations
        self._display_recommendations(initialization_results)
        
        # Performance metrics
        self._display_performance_metrics(initialization_results)
    
    def _display_overall_status(self, results: Dict[str, Any]) -> None:
        """Display overall system status summary."""
        success = results.get("success", False)
        status_color = "green" if success else "red"
        status_text = "✅ HEALTHY" if success else "❌ ISSUES DETECTED"
        
        # Create status panel
        status_content = [
            f"Status: {status_text}",
            f"Initialization Time: {results.get('initialization_time_seconds', 0):.2f}s",
            f"Components Initialized: {len(results.get('components_initialized', []))}",
            f"Timestamp: {results.get('timestamp', 'Unknown')}"
        ]
        
        panel = Panel(
            "\n".join(status_content),
            title="🎯 Overall System Status",
            border_style=status_color,
            expand=False
        )
        self.console.print(panel)
    
    def _display_component_status(self, results: Dict[str, Any]) -> None:
        """Display detailed component status."""
        component_status = results.get("component_status", {})
        
        table = Table(title="🔧 Component Health Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        for component, status_info in component_status.items():
            if component == "validation_time_ms":
                continue
            
            status = status_info.get("status", "unknown")
            details = status_info.get("details", {})
            
            # Color code status
            if status == "healthy":
                status_display = "[green]✅ Healthy[/green]"
            elif status == "not_initialized":
                status_display = "[yellow]⚠️  Not Initialized[/yellow]"
            elif status == "error":
                status_display = "[red]❌ Error[/red]"
            else:
                status_display = f"[dim]{status}[/dim]"
            
            # Format details
            if isinstance(details, dict):
                detail_text = ", ".join(f"{k}: {v}" for k, v in details.items() if k != "error")
                if "error" in details:
                    detail_text = f"Error: {details['error']}"
            else:
                detail_text = str(details)
            
            table.add_row(component.title(), status_display, detail_text[:50] + "..." if len(detail_text) > 50 else detail_text)
        
        self.console.print(table)
    
    def _display_database_status(self, results: Dict[str, Any]) -> None:
        """Display database and rule validation status."""
        database_status = results.get("database_status", {})
        
        # Database connectivity
        connectivity = database_status.get("connectivity", {})
        schema = database_status.get("schema_validation", {})
        rules = database_status.get("seeded_rules", {})
        
        db_table = Table(title="🗄️  Database & Rules Status", show_header=True, header_style="bold blue")
        db_table.add_column("Category", style="cyan")
        db_table.add_column("Status", justify="center")
        db_table.add_column("Details", style="dim")
        
        # Connectivity row
        conn_status = connectivity.get("status", "unknown")
        conn_color = "green" if conn_status == "healthy" else "red"
        db_table.add_row(
            "Database Connection",
            f"[{conn_color}]{conn_status.title()}[/{conn_color}]",
            "Connection successful" if conn_status == "healthy" else "Connection failed"
        )
        
        # Schema row
        schema_status = schema.get("status", "unknown")
        schema_details = schema.get("details", {})
        missing_tables = schema_details.get("missing_tables", [])
        schema_color = "green" if schema_status == "healthy" else "yellow"
        schema_detail_text = f"Missing: {', '.join(missing_tables)}" if missing_tables else "All tables present"
        db_table.add_row(
            "Database Schema",
            f"[{schema_color}]{schema_status.title()}[/{schema_color}]",
            schema_detail_text
        )
        
        # Rules row
        rules_status = rules.get("status", "unknown")
        rules_details = rules.get("details", {})
        rules_color = "green" if rules_status == "healthy" else "yellow"
        rules_text = f"{rules_details.get('valid_rules', 0)}/{rules_details.get('expected_rules', 0)} rules valid"
        db_table.add_row(
            "Seeded Rules",
            f"[{rules_color}]{rules_status.title()}[/{rules_color}]",
            rules_text
        )
        
        self.console.print(db_table)
    
    def _display_data_status(self, results: Dict[str, Any]) -> None:
        """Display data availability and quality status."""
        data_status = results.get("data_status", {})
        
        training_data = data_status.get("training_data", {})
        data_quality = data_status.get("data_quality", {})
        requirements = data_status.get("minimum_requirements", {})
        
        data_table = Table(title="📊 Training Data Status", show_header=True, header_style="bold green")
        data_table.add_column("Metric", style="cyan")
        data_table.add_column("Value", justify="right")
        data_table.add_column("Status", justify="center")
        
        # Training data metrics
        training_details = training_data.get("details", {})
        total_prompts = training_details.get("total_training_prompts", 0)
        synthetic_prompts = training_details.get("synthetic_prompts", 0)
        user_prompts = training_details.get("user_prompts", 0)
        
        data_table.add_row(
            "Total Training Prompts",
            str(total_prompts),
            "[green]✅[/green]" if total_prompts >= 100 else "[yellow]⚠️[/yellow]"
        )
        
        data_table.add_row(
            "Synthetic Data",
            str(synthetic_prompts),
            "[green]✅[/green]" if synthetic_prompts > 0 else "[red]❌[/red]"
        )
        
        data_table.add_row(
            "User Data",
            str(user_prompts),
            "[green]✅[/green]" if user_prompts > 0 else "[yellow]⚠️[/yellow]"
        )
        
        # Quality metrics
        quality_details = data_quality.get("details", {})
        overall_quality = quality_details.get("overall_quality_score", 0)
        
        data_table.add_row(
            "Data Quality Score",
            f"{overall_quality:.3f}",
            "[green]✅[/green]" if overall_quality > 0.7 else "[yellow]⚠️[/yellow]" if overall_quality > 0.5 else "[red]❌[/red]"
        )

        # Minimum requirements status
        requirements_details = requirements.get("details", {})
        requirements_status = requirements.get("status", "unknown")

        data_table.add_row(
            "Training Data Requirement",
            "≥100 prompts",
            "[green]✅[/green]" if requirements_details.get("training_data_count", False) else "[red]❌[/red]"
        )

        data_table.add_row(
            "Synthetic Data Available",
            "Required",
            "[green]✅[/green]" if requirements_details.get("synthetic_data_available", False) else "[red]❌[/red]"
        )

        data_table.add_row(
            "Overall Requirements",
            requirements_status.replace("_", " ").title(),
            "[green]✅[/green]" if requirements_status == "met" else "[red]❌[/red]"
        )

        self.console.print(data_table)
    
    def _display_initialization_details(self, results: Dict[str, Any]) -> None:
        """Display initialization plan and execution details."""
        plan = results.get("initialization_plan", {})
        execution = results.get("execution_results", {})
        
        # Initialization summary
        init_panel_content = []
        
        if plan:
            actions_count = len(plan.get("actions", []))
            estimated_time = plan.get("estimated_time_seconds", 0)
            init_panel_content.append(f"Planned Actions: {actions_count}")
            init_panel_content.append(f"Estimated Time: {estimated_time}s")
        
        if execution:
            completed = len(execution.get("actions_completed", []))
            failed = len(execution.get("actions_failed", []))
            actual_time = execution.get("execution_time_seconds", 0)
            init_panel_content.extend([
                f"Actions Completed: {completed}",
                f"Actions Failed: {failed}",
                f"Actual Time: {actual_time:.2f}s"
            ])
        
        if init_panel_content:
            panel = Panel(
                "\n".join(init_panel_content),
                title="⚙️ Initialization Execution",
                border_style="blue",
                expand=False
            )
            self.console.print(panel)
    
    def _display_recommendations(self, results: Dict[str, Any]) -> None:
        """Display system recommendations."""
        recommendations = results.get("recommendations", [])
        
        if recommendations:
            self.console.print("\n💡 System Recommendations:", style="bold yellow")
            for i, recommendation in enumerate(recommendations, 1):
                self.console.print(f"  {i}. {recommendation}", style="yellow")
        else:
            self.console.print("\n✅ No recommendations - system is optimally configured", style="green")
    
    def _display_performance_metrics(self, results: Dict[str, Any]) -> None:
        """Display performance metrics."""
        final_validation = results.get("final_validation", {})
        
        if final_validation:
            metrics_content = []
            
            overall_health = final_validation.get("overall_health", False)
            training_readiness = final_validation.get("training_readiness", False)
            validation_time = final_validation.get("validation_time_ms", 0)
            
            metrics_content.extend([
                f"Overall Health: {'✅ Healthy' if overall_health else '❌ Issues'}",
                f"Training Ready: {'✅ Ready' if training_readiness else '❌ Not Ready'}",
                f"Validation Time: {validation_time:.1f}ms"
            ])
            
            panel = Panel(
                "\n".join(metrics_content),
                title="📈 Performance Metrics",
                border_style="cyan",
                expand=False
            )
            self.console.print(panel)
    
    def export_state_report(self, results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Export system state report to file.
        
        Args:
            results: Initialization results
            filepath: Optional file path for export
            
        Returns:
            Path to exported file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"apes_system_state_{timestamp}.json"
        
        import json
        
        # Create exportable report
        export_data = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "system_state": results,
            "summary": {
                "overall_status": "healthy" if results.get("success") else "issues",
                "components_initialized": len(results.get("components_initialized", [])),
                "initialization_time": results.get("initialization_time_seconds", 0),
                "recommendations_count": len(results.get("recommendations", []))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath
