"""Execution Agent for running generated code in a sandboxed environment."""

import os
import sys
import traceback
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import uuid

from .base import BaseAgent
from ..models.agents import ExecutionRequest, ExecutionResponse
from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..utils.config import config


class ExecutionAgent(BaseAgent):
    """Agent responsible for executing generated Python code in a sandboxed environment."""
    
    def __init__(self):
        super().__init__(
            name="ExecutionAgent",
            description="Executes generated Python code safely with sandboxing and monitoring"
        )
        
        # Allowed modules for code execution
        self.allowed_modules = {
            'pandas', 'numpy', 'openpyxl', 'xlrd', 'datetime', 
            'json', 'csv', 'pathlib', 'collections', 'itertools',
            'math', 'statistics', 're', 'copy'
        }
        
        # Create sandbox directory
        self.sandbox_dir = Path(config.temp_dir) / "sandbox"
        self.sandbox_dir.mkdir(exist_ok=True)
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process code execution request."""
        if not isinstance(request, ExecutionRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected ExecutionRequest, got {type(request)}"
            )
        
        try:
            if request.dry_run:
                # Perform dry run validation
                result = await self._dry_run_validation(request.code)
                
                return ExecutionResponse(
                    agent_id=self.name,
                    request_id=request.request_id,
                    status=AgentStatus.SUCCESS,
                    result_file=None,
                    output="Dry run completed successfully",
                    execution_log=[f"Dry run validation: {result}"],
                    result={
                        "dry_run": True,
                        "validation_result": result,
                        "code_analysis": self._analyze_code(request.code)
                    }
                )
            
            else:
                # Execute code in sandbox
                execution_result = await self._execute_code_safely(
                    request.file_id,
                    request.code
                )
                
                self.logger.info(f"Code execution completed with status: {execution_result['status']}")
                
                return ExecutionResponse(
                    agent_id=self.name,
                    request_id=request.request_id,
                    status=execution_result['status'],
                    result_file=execution_result.get('result_file'),
                    output=execution_result.get('output'),
                    execution_log=execution_result.get('execution_log', []),
                    result=execution_result
                )
                
        except Exception as e:
            self.logger.error(f"Error during code execution: {e}")
            return self.create_error_response(request, str(e))
    
    async def _execute_code_safely(
        self, 
        file_id: str, 
        code: str
    ) -> Dict[str, Any]:
        """Execute code safely in a sandboxed environment."""
        execution_id = str(uuid.uuid4())
        sandbox_path = self.sandbox_dir / execution_id
        sandbox_path.mkdir()
        
        execution_log = []
        
        try:
            # Create isolated execution environment
            exec_globals = self._create_safe_globals()
            exec_locals = {}
            
            # Capture stdout and stderr
            stdout_buffer = StringIO()
            stderr_buffer = StringIO()
            
            # Prepare code with safety wrappers
            wrapped_code = self._wrap_code_for_safety(code, file_id, str(sandbox_path))
            
            execution_log.append(f"Executing code in sandbox: {execution_id}")
            
            # Execute code with redirection
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                try:
                    exec(wrapped_code, exec_globals, exec_locals)
                    execution_status = AgentStatus.SUCCESS
                    execution_log.append("Code executed successfully")
                    
                except Exception as e:
                    execution_status = AgentStatus.FAILED
                    execution_log.append(f"Execution error: {str(e)}")
                    execution_log.append(f"Traceback: {traceback.format_exc()}")
            
            # Capture output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            # Find result files
            result_files = list(sandbox_path.glob("*.xlsx")) + list(sandbox_path.glob("*.csv"))
            result_file = str(result_files[0]) if result_files else None
            
            # Prepare output
            output_parts = []
            if stdout_content:
                output_parts.append(f"STDOUT:\n{stdout_content}")
            if stderr_content:
                output_parts.append(f"STDERR:\n{stderr_content}")
            
            output = "\n\n".join(output_parts) if output_parts else "Code executed with no output"
            
            return {
                'status': execution_status,
                'output': output,
                'result_file': result_file,
                'execution_log': execution_log,
                'sandbox_path': str(sandbox_path),
                'locals': {k: str(v)[:100] for k, v in exec_locals.items() if not k.startswith('_')}
            }
            
        except Exception as e:
            execution_log.append(f"Sandbox execution failed: {str(e)}")
            return {
                'status': AgentStatus.FAILED,
                'output': f"Execution failed: {str(e)}",
                'result_file': None,
                'execution_log': execution_log,
                'sandbox_path': str(sandbox_path)
            }
        
        finally:
            # Cleanup sandbox (optional - keep for debugging)
            # shutil.rmtree(sandbox_path, ignore_errors=True)
            pass
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global namespace for code execution."""
        safe_builtins = {
            'len', 'range', 'enumerate', 'zip', 'list', 'dict', 'set', 'tuple',
            'str', 'int', 'float', 'bool', 'min', 'max', 'sum', 'sorted',
            'print', 'type', 'isinstance', 'hasattr', 'getattr'
        }
        
        # Create restricted builtins
        restricted_builtins = {
            name: getattr(__builtins__, name) 
            for name in safe_builtins 
            if hasattr(__builtins__, name)
        }
        
        # Import allowed modules
        safe_globals = {
            '__builtins__': restricted_builtins,
        }
        
        # Import common modules safely
        try:
            import pandas as pd
            import numpy as np
            import openpyxl
            from pathlib import Path
            from datetime import datetime, date
            import json
            import math
            import re
            
            safe_globals.update({
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'openpyxl': openpyxl,
                'Path': Path,
                'datetime': datetime,
                'date': date,
                'json': json,
                'math': math,
                're': re
            })
            
        except ImportError as e:
            self.logger.warning(f"Could not import module for safe globals: {e}")
        
        return safe_globals
    
    def _wrap_code_for_safety(
        self, 
        code: str, 
        file_id: str, 
        sandbox_path: str
    ) -> str:
        """Wrap code with safety measures and path restrictions."""
        
        # Add path restrictions and file ID context
        wrapper = f'''
import os
import sys
from pathlib import Path

# Sandbox configuration
SANDBOX_PATH = Path("{sandbox_path}")
FILE_ID = "{file_id}"

# Override Path to restrict to sandbox
original_path = Path
class RestrictedPath(original_path):
    def __new__(cls, *args, **kwargs):
        path = original_path(*args, **kwargs)
        # Allow only paths within sandbox or specific file paths
        if not (str(path).startswith(str(SANDBOX_PATH)) or 
                str(path).endswith('.xlsx') or 
                str(path).endswith('.xls') or
                str(path).endswith('.csv')):
            # For now, allow paths but log them
            print(f"WARNING: Accessing path outside sandbox: {{path}}")
        return path

Path = RestrictedPath

# Original user code starts here:
{code}
'''
        return wrapper
    
    async def _dry_run_validation(self, code: str) -> Dict[str, Any]:
        """Perform dry run validation without execution."""
        validation_result = {
            'syntax_valid': False,
            'imports_valid': False,
            'estimated_safety': 'unknown',
            'warnings': []
        }
        
        try:
            # Check syntax
            compile(code, '<string>', 'exec')
            validation_result['syntax_valid'] = True
            
            # Analyze imports
            import ast
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check if all imports are allowed
            disallowed_imports = [
                imp for imp in imports 
                if not any(imp.startswith(allowed) for allowed in self.allowed_modules)
            ]
            
            if not disallowed_imports:
                validation_result['imports_valid'] = True
            else:
                validation_result['warnings'].append(
                    f"Potentially unsafe imports: {disallowed_imports}"
                )
            
            # Estimate safety
            if validation_result['syntax_valid'] and validation_result['imports_valid']:
                validation_result['estimated_safety'] = 'safe'
            else:
                validation_result['estimated_safety'] = 'unsafe'
            
        except SyntaxError as e:
            validation_result['warnings'].append(f"Syntax error: {e}")
        except Exception as e:
            validation_result['warnings'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for complexity and resource usage estimation."""
        lines = code.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Count different types of operations
        operations = {
            'data_loading': sum(1 for line in lines if 'read_excel' in line or 'read_csv' in line),
            'data_saving': sum(1 for line in lines if 'to_excel' in line or 'to_csv' in line),
            'aggregations': sum(1 for line in lines if 'groupby' in line or 'agg' in line),
            'filters': sum(1 for line in lines if '[' in line and ']' in line),
            'loops': sum(1 for line in lines if line.strip().startswith('for ') or line.strip().startswith('while ')),
        }
        
        # Estimate complexity
        complexity_score = (
            len(non_empty_lines) + 
            operations['aggregations'] * 2 + 
            operations['loops'] * 3
        )
        
        if complexity_score < 20:
            complexity_level = 'low'
        elif complexity_score < 50:
            complexity_level = 'medium'
        else:
            complexity_level = 'high'
        
        return {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'operations': operations,
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'estimated_memory_usage': self._estimate_memory_usage(operations),
            'estimated_execution_time': self._estimate_execution_time(complexity_score)
        }
    
    def _estimate_memory_usage(self, operations: Dict[str, int]) -> str:
        """Estimate memory usage based on operations."""
        base_memory = 50  # MB
        data_loading_memory = operations['data_loading'] * 100  # MB per file
        aggregation_memory = operations['aggregations'] * 50  # MB per aggregation
        
        total_memory = base_memory + data_loading_memory + aggregation_memory
        
        if total_memory < 200:
            return "< 200 MB"
        elif total_memory < 500:
            return "200-500 MB"
        else:
            return "> 500 MB"
    
    def _estimate_execution_time(self, complexity_score: int) -> str:
        """Estimate execution time based on complexity score."""
        if complexity_score < 20:
            return "< 10 seconds"
        elif complexity_score < 50:
            return "10-60 seconds"
        else:
            return "> 60 seconds"