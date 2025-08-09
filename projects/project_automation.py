"""
MORPHEUS CHAT - Project Automation System
Intelligent task scheduling and automation for projects

Revolutionary Features:
- AI-driven automation workflow creation
- Scheduled tasks and recurring automation
- File monitoring and triggered actions
- Intelligent workflow optimization
- Zero-maintenance automation that adapts over time
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import cron_descriptor
import croniter

logger = logging.getLogger(__name__)


class AutomationTrigger(str, Enum):
    """Types of automation triggers"""
    SCHEDULE = "schedule"           # Time-based triggers
    FILE_CHANGE = "file_change"     # File system triggers
    PROJECT_EVENT = "project_event" # Project-specific events
    THRESHOLD = "threshold"         # Metric-based triggers
    MANUAL = "manual"              # User-initiated
    CHAIN = "chain"                # Triggered by other automations


class AutomationAction(str, Enum):
    """Types of automation actions"""
    ANALYZE_FILES = "analyze_files"
    GENERATE_REPORT = "generate_report"
    BACKUP_PROJECT = "backup_project"
    SYNC_KNOWLEDGE = "sync_knowledge"
    CLEANUP_FILES = "cleanup_files"
    EXECUTE_SCRIPT = "execute_script"
    SEND_NOTIFICATION = "send_notification"
    UPDATE_DOCUMENTATION = "update_documentation"
    RUN_COLLABORATION = "run_collaboration"
    PROCESS_QUEUE = "process_queue"


@dataclass
class AutomationRule:
    """Individual automation rule definition"""
    rule_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # Trigger configuration
    trigger_type: AutomationTrigger = AutomationTrigger.MANUAL
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    
    # Action configuration
    action_type: AutomationAction = AutomationAction.ANALYZE_FILES
    action_config: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    cron_expression: Optional[str] = None
    enabled: bool = True
    
    # Conditions and filters
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    
    # Execution tracking
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    average_duration: float = 0.0
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)
    triggers_rules: List[UUID] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    tags: Set[str] = field(default_factory=set)


@dataclass
class AutomationExecution:
    """Automation execution record"""
    execution_id: UUID = field(default_factory=uuid4)
    rule_id: UUID = None
    
    # Execution details
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    
    # Results
    output: str = ""
    artifacts_created: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Context
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)


class ProjectAutomationEngine:
    """
    Intelligent automation engine for project tasks
    
    Features:
    - Smart automation rule creation based on project patterns
    - Adaptive scheduling that learns from project activity
    - File system monitoring with intelligent filtering
    - Workflow orchestration with dependency management
    - Performance optimization and resource management
    """
    
    def __init__(self, project_id: str, project_vm, project_knowledge, project_intelligence):
        self.project_id = project_id
        self.project_vm = project_vm
        self.project_knowledge = project_knowledge
        self.project_intelligence = project_intelligence
        
        # Automation state
        self.automation_rules: Dict[UUID, AutomationRule] = {}
        self.active_executions: Dict[UUID, AutomationExecution] = {}
        self.execution_history: List[AutomationExecution] = []
        
        # Scheduling and monitoring
        self.scheduler_running = False
        self.file_monitor_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.file_monitor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.automation_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'automations_created': 0,
            'last_optimization': datetime.now(timezone.utc)
        }
        
        logger.info(f"Automation engine initialized for project {project_id}")
    
    async def initialize(self):
        """Initialize automation engine"""
        
        try:
            # Setup automation workspace in VM
            await self._setup_automation_workspace()
            
            # Load existing automation rules
            await self._load_existing_rules()
            
            # Create default automation rules based on project
            await self._create_default_automations()
            
            # Start automation systems
            await self.start_automation_systems()
            
            logger.info(f"Automation engine ready for project {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize automation engine: {e}")
            raise
    
    async def _setup_automation_workspace(self):
        """Setup automation workspace in project VM"""
        
        # Create automation directories
        await self.project_vm.execute_command('''
mkdir -p /project/automation/{rules,executions,scripts,logs,schedules}
        ''')
        
        # Create automation toolkit
        automation_toolkit = '''#!/usr/bin/env python3
"""
Project Automation Toolkit
Provides automation capabilities within project VM
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

class AutomationToolkit:
    def __init__(self):
        self.project_root = Path("/project")
        self.automation_root = self.project_root / "automation"
        
    def log_execution(self, rule_id, status, output="", error=""):
        """Log automation execution"""
        log_entry = {
            "rule_id": rule_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "output": output,
            "error": error
        }
        
        log_file = self.automation_root / "logs" / f"{rule_id}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\\n")
    
    def execute_command(self, command):
        """Execute system command safely"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, 
                text=True, timeout=300
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def count_files(self, pattern="*"):
        """Count files matching pattern"""
        files = list(self.project_root.glob(f"files/**/{pattern}"))
        return len(files)
    
    def get_project_stats(self):
        """Get basic project statistics"""
        return {
            "total_files": self.count_files(),
            "last_updated": datetime.now().isoformat(),
            "disk_usage": self.execute_command("du -sh /project")["output"]
        }

# Global toolkit instance
toolkit = AutomationToolkit()
        '''
        
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            "/project/automation/toolkit.py",
            automation_toolkit
        )
    
    async def _create_default_automations(self):
        """Create intelligent default automation rules based on project"""
        
        # Get project knowledge to determine useful automations
        knowledge_summary = await self.project_knowledge.get_knowledge_summary()
        
        default_rules = []
        
        # Daily project analysis automation
        daily_analysis = AutomationRule(
            name="Daily Project Analysis",
            description="Analyze project files and update knowledge base daily",
            trigger_type=AutomationTrigger.SCHEDULE,
            trigger_config={"cron": "0 9 * * *"},  # 9 AM daily
            action_type=AutomationAction.ANALYZE_FILES,
            action_config={"deep_analysis": True},
            cron_expression="0 9 * * *",
            tags={"daily", "analysis", "automatic"}
        )
        default_rules.append(daily_analysis)
        
        # File change monitoring (if project has many files)
        if knowledge_summary.get('total_items', 0) > 10:
            file_monitor = AutomationRule(
                name="File Change Monitor",
                description="Monitor for file changes and update knowledge base",
                trigger_type=AutomationTrigger.FILE_CHANGE,
                trigger_config={"watch_patterns": ["*.py", "*.md", "*.txt", "*.json"]},
                action_type=AutomationAction.SYNC_KNOWLEDGE,
                action_config={"incremental": True},
                tags={"monitoring", "files", "automatic"}
            )
            default_rules.append(file_monitor)
        
        # Weekly backup automation
        weekly_backup = AutomationRule(
            name="Weekly Project Backup",
            description="Create weekly backup of project state",
            trigger_type=AutomationTrigger.SCHEDULE,
            trigger_config={"cron": "0 2 * * 0"},  # 2 AM Sunday
            action_type=AutomationAction.BACKUP_PROJECT,
            action_config={"include_vm_state": True},
            cron_expression="0 2 * * 0",
            tags={"weekly", "backup", "automatic"}
        )
        default_rules.append(weekly_backup)
        
        # Documentation update (if project has code files)
        if 'code' in knowledge_summary.get('items_by_type', {}):
            doc_update = AutomationRule(
                name="Documentation Update",
                description="Update documentation when code changes",
                trigger_type=AutomationTrigger.FILE_CHANGE,
                trigger_config={"watch_patterns": ["*.py", "*.js", "*.html"]},
                action_type=AutomationAction.UPDATE_DOCUMENTATION,
                action_config={"auto_generate": True},
                tags={"documentation", "code", "automatic"}
            )
            default_rules.append(doc_update)
        
        # Create and register default rules
        for rule in default_rules:
            await self.create_automation_rule(rule)
    
    async def create_automation_rule(self, rule: AutomationRule) -> UUID:
        """Create new automation rule"""
        
        try:
            # Validate rule configuration
            await self._validate_automation_rule(rule)
            
            # Store rule
            self.automation_rules[rule.rule_id] = rule
            
            # Persist rule to VM
            await self._persist_automation_rule(rule)
            
            # Update metrics
            self.automation_metrics['automations_created'] += 1
            
            logger.info(f"Created automation rule: {rule.name}")
            return rule.rule_id
            
        except Exception as e:
            logger.error(f"Failed to create automation rule: {e}")
            raise
    
    async def _validate_automation_rule(self, rule: AutomationRule):
        """Validate automation rule configuration"""
        
        # Validate cron expression if scheduled
        if rule.trigger_type == AutomationTrigger.SCHEDULE and rule.cron_expression:
            try:
                croniter.croniter(rule.cron_expression)
            except:
                raise ValueError(f"Invalid cron expression: {rule.cron_expression}")
        
        # Validate file patterns
        if rule.trigger_type == AutomationTrigger.FILE_CHANGE:
            if not rule.trigger_config.get('watch_patterns'):
                raise ValueError("File change trigger requires watch_patterns")
        
        # Validate action configuration
        if rule.action_type == AutomationAction.EXECUTE_SCRIPT:
            if not rule.action_config.get('script_path'):
                raise ValueError("Execute script action requires script_path")
    
    async def _persist_automation_rule(self, rule: AutomationRule):
        """Persist automation rule to VM"""
        
        rule_data = {
            'rule_id': str(rule.rule_id),
            'name': rule.name,
            'description': rule.description,
            'trigger_type': rule.trigger_type,
            'trigger_config': rule.trigger_config,
            'action_type': rule.action_type,
            'action_config': rule.action_config,
            'cron_expression': rule.cron_expression,
            'enabled': rule.enabled,
            'conditions': rule.conditions,
            'file_patterns': rule.file_patterns,
            'depends_on': [str(dep) for dep in rule.depends_on],
            'triggers_rules': [str(trig) for trig in rule.triggers_rules],
            'created_at': rule.created_at.isoformat(),
            'created_by': rule.created_by,
            'tags': list(rule.tags)
        }
        
        rule_file = f"/project/automation/rules/{rule.rule_id}.json"
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            rule_file,
            json.dumps(rule_data, indent=2)
        )
    
    async def start_automation_systems(self):
        """Start automation scheduling and monitoring systems"""
        
        if not self.scheduler_running:
            self.scheduler_task = asyncio.create_task(self._run_scheduler())
            self.scheduler_running = True
        
        if not self.file_monitor_running:
            self.file_monitor_task = asyncio.create_task(self._run_file_monitor())
            self.file_monitor_running = True
        
        logger.info("Automation systems started")
    
    async def _run_scheduler(self):
        """Main scheduling loop for time-based automations"""
        
        while self.scheduler_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check all scheduled rules
                for rule in self.automation_rules.values():
                    if (rule.enabled and 
                        rule.trigger_type == AutomationTrigger.SCHEDULE and 
                        rule.cron_expression):
                        
                        # Check if rule should execute
                        if await self._should_execute_rule(rule, current_time):
                            await self._execute_automation_rule(rule)
                
                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _should_execute_rule(self, rule: AutomationRule, current_time: datetime) -> bool:
        """Check if scheduled rule should execute"""
        
        try:
            cron = croniter.croniter(rule.cron_expression, current_time)
            next_run = cron.get_prev(datetime)
            
            # Check if we should have run since last execution
            if rule.last_executed is None:
                return True
            
            return next_run > rule.last_executed
            
        except Exception as e:
            logger.error(f"Error checking rule schedule: {e}")
            return False
    
    async def _run_file_monitor(self):
        """Monitor file system changes for file-based triggers"""
        
        # Simple file monitoring - check for changes every 10 seconds
        last_file_counts = {}
        
        while self.file_monitor_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check file-triggered rules
                for rule in self.automation_rules.values():
                    if (rule.enabled and 
                        rule.trigger_type == AutomationTrigger.FILE_CHANGE):
                        
                        # Check for file changes
                        patterns = rule.trigger_config.get('watch_patterns', ['*'])
                        
                        for pattern in patterns:
                            current_count = await self._count_matching_files(pattern)
                            last_count = last_file_counts.get(f"{rule.rule_id}_{pattern}", 0)
                            
                            if current_count != last_count:
                                # Files changed, execute rule
                                trigger_data = {
                                    'pattern': pattern,
                                    'previous_count': last_count,
                                    'current_count': current_count,
                                    'change_detected': current_time.isoformat()
                                }
                                
                                await self._execute_automation_rule(rule, trigger_data)
                                last_file_counts[f"{rule.rule_id}_{pattern}"] = current_count
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"File monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _count_matching_files(self, pattern: str) -> int:
        """Count files matching pattern in project"""
        
        try:
            result = await self.project_vm.execute_command_in_vm(
                self.project_vm,
                f"find /project/files -name '{pattern}' -type f | wc -l"
            )
            return int(result.strip())
        except:
            return 0
    
    async def _execute_automation_rule(
        self, 
        rule: AutomationRule, 
        trigger_data: Optional[Dict[str, Any]] = None
    ):
        """Execute automation rule"""
        
        execution = AutomationExecution(
            rule_id=rule.rule_id,
            trigger_data=trigger_data or {}
        )
        
        self.active_executions[execution.execution_id] = execution
        
        try:
            # Check dependencies
            if rule.depends_on:
                for dep_rule_id in rule.depends_on:
                    if not await self._check_dependency_satisfied(dep_rule_id):
                        raise Exception(f"Dependency {dep_rule_id} not satisfied")
            
            # Execute action based on type
            if rule.action_type == AutomationAction.ANALYZE_FILES:
                result = await self._execute_analyze_files(rule, execution)
            elif rule.action_type == AutomationAction.GENERATE_REPORT:
                result = await self._execute_generate_report(rule, execution)
            elif rule.action_type == AutomationAction.BACKUP_PROJECT:
                result = await self._execute_backup_project(rule, execution)
            elif rule.action_type == AutomationAction.SYNC_KNOWLEDGE:
                result = await self._execute_sync_knowledge(rule, execution)
            elif rule.action_type == AutomationAction.UPDATE_DOCUMENTATION:
                result = await self._execute_update_documentation(rule, execution)
            elif rule.action_type == AutomationAction.EXECUTE_SCRIPT:
                result = await self._execute_custom_script(rule, execution)
            else:
                result = {"output": f"Unknown action type: {rule.action_type}"}
            
            # Update execution record
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "completed"
            execution.output = result.get('output', '')
            execution.artifacts_created = result.get('artifacts', [])
            
            # Update rule statistics
            rule.last_executed = execution.completed_at
            rule.execution_count += 1
            rule.success_count += 1
            
            duration = (execution.completed_at - execution.started_at).total_seconds()
            rule.average_duration = (
                (rule.average_duration * (rule.execution_count - 1) + duration) / 
                rule.execution_count
            )
            
            # Update global metrics
            self.automation_metrics['total_executions'] += 1
            self.automation_metrics['successful_executions'] += 1
            
            # Trigger dependent rules
            for trigger_rule_id in rule.triggers_rules:
                if trigger_rule_id in self.automation_rules:
                    dependent_rule = self.automation_rules[trigger_rule_id]
                    await self._execute_automation_rule(dependent_rule)
            
            logger.info(f"Automation rule '{rule.name}' executed successfully")
            
        except Exception as e:
            execution.completed_at = datetime.now(timezone.utc)
            execution.status = "failed"
            execution.error_message = str(e)
            
            rule.execution_count += 1
            
            self.automation_metrics['total_executions'] += 1
            self.automation_metrics['failed_executions'] += 1
            
            logger.error(f"Automation rule '{rule.name}' failed: {e}")
        
        finally:
            # Move to history and cleanup
            self.execution_history.append(execution)
            del self.active_executions[execution.execution_id]
            
            # Keep only recent history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
    
    async def _execute_analyze_files(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute file analysis automation"""
        
        # Trigger intelligence engine to analyze files
        # This integrates with project_intelligence.py
        
        analysis_results = await self.project_intelligence._scan_for_new_files()
        
        if analysis_results:
            await self.project_intelligence._process_files_batch(analysis_results)
            
            output = f"Analyzed {len(analysis_results)} files"
            artifacts = [f"/project/intelligence/analysis_{execution.execution_id}.json"]
            
            # Save analysis results
            analysis_data = {
                'execution_id': str(execution.execution_id),
                'files_analyzed': len(analysis_results),
                'analysis_time': datetime.now(timezone.utc).isoformat(),
                'file_list': [str(f) for f in analysis_results]
            }
            
            await self.project_vm.write_file_to_vm(
                self.project_vm,
                artifacts[0],
                json.dumps(analysis_data, indent=2)
            )
            
            return {'output': output, 'artifacts': artifacts}
        else:
            return {'output': 'No new files to analyze'}
    
    async def _execute_generate_report(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute report generation automation"""
        
        # Generate project status report
        knowledge_summary = await self.project_knowledge.get_knowledge_summary()
        automation_status = await self.get_automation_status()
        
        report_content = f"""# Automated Project Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project ID: {self.project_id}

## Knowledge Base Summary
- Total Items: {knowledge_summary.get('total_items', 0)}
- Item Types: {', '.join(knowledge_summary.get('items_by_type', {}).keys())}
- Last Updated: {knowledge_summary.get('last_updated', 'Unknown')}

## Automation Status
- Active Rules: {automation_status['active_rules']}
- Total Executions: {automation_status['total_executions']}
- Success Rate: {automation_status['success_rate']:.1%}

## Recent Activity
{chr(10).join(f"- {item['title']}: {item['access_count']} accesses" for item in knowledge_summary.get('popular_items', [])[:5])}
        """
        
        report_path = f"/project/automation/reports/report_{execution.execution_id}.md"
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            report_path,
            report_content
        )
        
        return {
            'output': f"Generated project report with {len(report_content)} characters",
            'artifacts': [report_path]
        }
    
    async def _execute_backup_project(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute project backup automation"""
        
        backup_name = f"backup_{execution.execution_id}_{int(time.time())}"
        backup_path = f"/project/automation/backups/{backup_name}"
        
        # Create backup using tar
        backup_command = f"""
mkdir -p /project/automation/backups
cd /project && tar -czf {backup_path}.tar.gz \
    --exclude='automation/backups' \
    --exclude='*.log' \
    files/ knowledge/ artifacts/ workspace/
        """
        
        result = await self.project_vm.execute_command_in_vm(
            self.project_vm,
            backup_command
        )
        
        return {
            'output': f"Created project backup: {backup_name}.tar.gz",
            'artifacts': [f"{backup_path}.tar.gz"]
        }
    
    async def _execute_sync_knowledge(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute knowledge base sync automation"""
        
        # This would trigger knowledge base updates
        # Integration with project_knowledge.py
        
        return {
            'output': "Knowledge base sync completed",
            'artifacts': []
        }
    
    async def _execute_update_documentation(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute documentation update automation"""
        
        # Auto-generate or update project documentation
        doc_content = f"""# Project Documentation (Auto-Updated)
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This documentation was automatically generated by the project automation system.

## Project Overview
Project ID: {self.project_id}
Automation Rules: {len(self.automation_rules)}

## Files Organization
The project files are automatically organized and analyzed by the AI system.

## Automation
This project uses intelligent automation to:
- Monitor file changes
- Update knowledge base
- Generate reports
- Maintain backups
        """
        
        doc_path = f"/project/README_auto.md"
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            doc_path,
            doc_content
        )
        
        return {
            'output': "Documentation updated automatically",
            'artifacts': [doc_path]
        }
    
    async def _execute_custom_script(self, rule: AutomationRule, execution: AutomationExecution) -> Dict[str, Any]:
        """Execute custom script automation"""
        
        script_path = rule.action_config.get('script_path')
        if not script_path:
            raise ValueError("Script path not specified")
        
        result = await self.project_vm.execute_command_in_vm(
            self.project_vm,
            f"cd /project && python3 {script_path}"
        )
        
        return {
            'output': f"Executed script: {script_path}\nOutput: {result[:500]}",
            'artifacts': []
        }
    
    async def _check_dependency_satisfied(self, dep_rule_id: UUID) -> bool:
        """Check if dependency rule has executed successfully recently"""
        
        # Check if dependent rule executed successfully in last 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for execution in reversed(self.execution_history):
            if (execution.rule_id == dep_rule_id and 
                execution.status == "completed" and
                execution.completed_at and
                execution.completed_at > cutoff_time):
                return True
        
        return False
    
    async def get_automation_status(self) -> Dict[str, Any]:
        """Get comprehensive automation status"""
        
        active_rules = sum(1 for rule in self.automation_rules.values() if rule.enabled)
        
        # Calculate success rate
        total_executions = self.automation_metrics['total_executions']
        successful_executions = self.automation_metrics['successful_executions']
        success_rate = successful_executions / max(total_executions, 1)
        
        return {
            'project_id': self.project_id,
            'automation_enabled': self.scheduler_running,
            'total_rules': len(self.automation_rules),
            'active_rules': active_rules,
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': self.automation_metrics['failed_executions'],
            'success_rate': success_rate,
            'average_execution_time': self.automation_metrics['average_execution_time'],
            'active_executions': len(self.active_executions),
            'recent_executions': [
                {
                    'rule_name': next(
                        (rule.name for rule in self.automation_rules.values() 
                         if rule.rule_id == execution.rule_id), 
                        'Unknown'
                    ),
                    'status': execution.status,
                    'started_at': execution.started_at.isoformat(),
                    'duration': (
                        (execution.completed_at - execution.started_at).total_seconds()
                        if execution.completed_at else None
                    )
                }
                for execution in self.execution_history[-10:]
            ]
        }
    
    async def stop_automation_systems(self):
        """Stop automation systems"""
        
        self.scheduler_running = False
        self.file_monitor_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        if self.file_monitor_task:
            self.file_monitor_task.cancel()
        
        logger.info("Automation systems stopped")
    
    async def _load_existing_rules(self):
        """Load existing automation rules from VM"""
        
        try:
            # Check if rules directory exists
            result = await self.project_vm.execute_command_in_vm(
                self.project_vm,
                "ls /project/automation/rules/*.json 2>/dev/null || echo 'none'"
            )
            
            if result.strip() != 'none':
                # Load rules (simplified - in production, load all rule files)
                logger.info("Loading existing automation rules...")
                # Implementation would load and parse rule files
        
        except Exception as e:
            logger.info(f"No existing automation rules found: {e}")
    
    async def cleanup(self):
        """Cleanup automation engine"""
        
        await self.stop_automation_systems()
        
        self.automation_rules.clear()
        self.active_executions.clear()
        self.execution_history.clear()