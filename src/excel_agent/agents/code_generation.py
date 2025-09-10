"""Code Generation Agent for converting user requests into executable pandas/openpyxl code."""

import ast
import json
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..models.agents import CodeGenerationRequest, CodeGenerationResponse
from ..models.base import AgentRequest, AgentResponse, AgentStatus
from ..utils.config import config


class CodeGenerationAgent(BaseAgent):
    """Agent responsible for converting user requests into executable pandas/openpyxl code."""
    
    def __init__(self):
        super().__init__(
            name="CodeGenerationAgent",
            description="Converts user requests into executable pandas/openpyxl code with safety checks"
        )
        
        # Code templates for common operations
        self.templates = {
            'read_excel': """
import pandas as pd
import openpyxl
from pathlib import Path

# Read Excel file
file_path = "{file_path}"
sheet_name = "{sheet_name}"
df = pd.read_excel(file_path, sheet_name=sheet_name)
""",
            'filter_data': """
# Filter data based on condition
filtered_df = df[{condition}]
""",
            'aggregate_data': """
# Aggregate data
result = df.groupby('{group_by}').{agg_func}()
""",
            'sort_data': """
# Sort data
sorted_df = df.sort_values(by='{sort_column}', ascending={ascending})
""",
            'save_result': """
# Save result to Excel
output_path = "{output_path}"
{data_var}.to_excel(output_path, sheet_name='{sheet_name}', index=False)
print(f"Result saved to: {{output_path}}")
"""
        }
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process code generation request."""
        if not isinstance(request, CodeGenerationRequest):
            return self.create_error_response(
                request,
                f"Invalid request type. Expected CodeGenerationRequest, got {type(request)}"
            )
        
        try:
            # Generate code based on user request
            generated_code = await self._generate_code(request)
            
            # Create dry run plan
            dry_run_plan = await self._create_dry_run_plan(request, generated_code)
            
            # Validate generated code
            validation_result = self._validate_code(generated_code)
            if not validation_result['is_valid']:
                return self.create_error_response(
                    request,
                    f"Generated code validation failed: {validation_result['error']}"
                )
            
            self.logger.info(f"Code generation completed successfully")
            
            return CodeGenerationResponse(
                agent_id=self.name,
                request_id=request.request_id,
                status=AgentStatus.SUCCESS,
                code=generated_code,
                dry_run_plan=dry_run_plan,
                dependencies=validation_result.get('dependencies', []),
                result={
                    "code": generated_code,
                    "dry_run_plan": dry_run_plan,
                    "dependencies": validation_result.get('dependencies', []),
                    "estimated_execution_time": self._estimate_execution_time(generated_code)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during code generation: {e}")
            return self.create_error_response(request, str(e))
    
    async def _generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate code based on user request using LLM."""
        # Extract context information
        file_info = request.file_info or {}
        context = request.context or {}
        
        # Build system prompt
        system_prompt = """You are an expert Python code generator specializing in Excel data processing using pandas and openpyxl.

Generate clean, safe, and efficient Python code based on user requests. Follow these guidelines:

1. SECURITY: Never include code that could:
   - Execute arbitrary system commands
   - Access files outside the specified directory
   - Import dangerous modules (os, subprocess, eval, exec, etc.)
   - Perform network operations

2. SAFETY: Always include error handling and data validation

3. EFFICIENCY: Use pandas vectorized operations when possible

4. CLARITY: Include helpful comments and variable names

5. OUTPUT: Return only the Python code, no explanations

Available information:
- File info: {file_info}
- Context: {context}

Generate code for this request: {user_request}
"""
        
        # Build user message
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    file_info=json.dumps(self._make_json_serializable(file_info), indent=2),
                    context=json.dumps(self._make_json_serializable(context), indent=2),
                    user_request=request.user_request
                )
            },
            {
                "role": "user",
                "content": f"Generate Python code for: {request.user_request}"
            }
        ]
        
        # Get LLM response
        response = await self.llm_completion(
            messages=messages,
            temperature=0.1,  # Low temperature for consistent code generation
            max_tokens=2000
        )
        
        # Extract generated code
        generated_code = response['choices'][0]['message']['content']
        
        # Clean up the code
        generated_code = self._clean_generated_code(generated_code)
        
        return generated_code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and format generated code."""
        # Remove markdown code blocks
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Ensure proper imports are at the top
        lines = code.split('\n')
        import_lines = []
        other_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct with imports at top
        if import_lines:
            cleaned_code = '\n'.join(import_lines) + '\n\n' + '\n'.join(other_lines)
        else:
            cleaned_code = '\n'.join(other_lines)
        
        return cleaned_code
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code for safety and syntax."""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_nodes = []
            dependencies = set()
            
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        dependencies.add(module_name)
                        if self._is_dangerous_module(module_name):
                            dangerous_nodes.append(f"Import of dangerous module: {module_name}")
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    dependencies.add(module_name)
                    if self._is_dangerous_module(module_name):
                        dangerous_nodes.append(f"Import from dangerous module: {module_name}")
                
                # Check function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if self._is_dangerous_function(func_name):
                            dangerous_nodes.append(f"Dangerous function call: {func_name}")
            
            if dangerous_nodes:
                return {
                    'is_valid': False,
                    'error': f"Code contains dangerous operations: {', '.join(dangerous_nodes)}",
                    'dependencies': list(dependencies)
                }
            
            return {
                'is_valid': True,
                'dependencies': list(dependencies),
                'ast_node_count': len(list(ast.walk(tree)))
            }
            
        except SyntaxError as e:
            return {
                'is_valid': False,
                'error': f"Syntax error: {e}",
                'dependencies': []
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f"Validation error: {e}",
                'dependencies': []
            }
    
    def _is_dangerous_module(self, module_name: str) -> bool:
        """Check if module is potentially dangerous."""
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'urllib',
            'requests', 'httpx', 'eval', 'exec', 'compile',
            'importlib', '__import__', 'builtins'
        }
        
        # Check exact match or starts with dangerous module
        return (
            module_name in dangerous_modules or
            any(module_name.startswith(dm + '.') for dm in dangerous_modules)
        )
    
    def _is_dangerous_function(self, func_name: str) -> bool:
        """Check if function call is potentially dangerous."""
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'open',  # We'll handle file operations explicitly
            'input',  # Interactive input not suitable for automation
        }
        return func_name in dangerous_functions
    
    async def _create_dry_run_plan(
        self, 
        request: CodeGenerationRequest, 
        generated_code: str
    ) -> str:
        """Create a dry run execution plan."""
        # Analyze the generated code to create execution plan
        plan_prompt = f"""
Analyze this Python code and create a step-by-step execution plan describing what the code will do:

```python
{generated_code}
```

Create a concise plan with:
1. Data loading steps
2. Processing operations
3. Output operations
4. Estimated impact/changes

Format as a numbered list. Be specific about what data will be affected.
"""
        
        messages = [
            {"role": "user", "content": plan_prompt}
        ]
        
        try:
            response = await self.llm_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            return response['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            self.logger.warning(f"Could not generate dry run plan: {e}")
            return f"Execution plan generation failed: {e}"
    
    def _estimate_execution_time(self, code: str) -> str:
        """Estimate execution time based on code complexity."""
        # Simple heuristic based on code structure
        lines = len(code.split('\n'))
        
        # Count potentially expensive operations
        expensive_ops = [
            'read_excel', 'to_excel', 'groupby', 'merge', 'join',
            'sort_values', 'pivot_table', 'apply'
        ]
        
        expensive_count = sum(
            1 for op in expensive_ops 
            if op in code
        )
        
        if lines < 20 and expensive_count < 2:
            return "< 5 seconds"
        elif lines < 50 and expensive_count < 5:
            return "5-30 seconds"
        else:
            return "30+ seconds"
    
    def get_code_template(self, operation_type: str) -> Optional[str]:
        """Get code template for common operations."""
        return self.templates.get(operation_type)
    
    def build_code_from_template(
        self, 
        template_name: str, 
        **kwargs
    ) -> str:
        """Build code from template with parameters."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing parameter for template: {e}")