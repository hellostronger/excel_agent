"""Optimized prompt templates for efficient token usage (inspired by ST-Raptor)."""

from typing import Dict, Any, List, Optional


class PromptTemplates:
    """Centralized prompt templates with token optimization."""
    
    # Query decomposition template (ST-Raptor inspired)
    QUERY_DECOMPOSE = """Decompose the complex query into simpler sub-queries for table analysis.

TABLE SCHEMA:
{schema}

USER QUERY: {query}

RULES:
1. Each sub-query should target a specific part of the table
2. Use format: [Query] <query text> [Retrieve] <true/false>
3. Mark [Retrieve] true if data retrieval needed, false for calculation only
4. Maximum 3 sub-queries

OUTPUT:
[Query] <sub-query-1> [Retrieve] <true/false>
[Query] <sub-query-2> [Retrieve] <true/false>
[Query] <sub-query-3> [Retrieve] <true/false>"""

    # Code generation template (optimized)
    CODE_GENERATION = """Generate Python pandas code for the given query on Excel data.

DATA CONTEXT:
- File: {file_path}
- Columns: {columns}
- Data types: {data_types}
- Sample data: {sample_data}

QUERY: {query}

REQUIREMENTS:
- Import: pandas, openpyxl only
- Variable: df (DataFrame)
- Output: final result only
- Error handling: try/except
- Max 15 lines

CODE:
```python
"""

    # Minimal code generation for simple queries
    SIMPLE_CODE_GENERATION = """Generate pandas code for: {query}

Context: {context}

Code (5 lines max):
```python
"""

    # Column profiling template (compact)
    COLUMN_PROFILING = """Analyze column: {column_name}

Data sample (first 10): {sample_values}
Data types detected: {data_types}

Output JSON:
{{"data_type": "string/number/date", "null_count": N, "unique_values": N, "description": "brief"}}"""

    # Structure analysis template
    STRUCTURE_ANALYSIS = """Analyze Excel table structure:

Sheet: {sheet_name}
Dimensions: {rows}x{cols}
Merged cells: {merged_cells}

Key features (3 max):
1. 
2. 
3. """

    # Relation discovery template
    RELATION_DISCOVERY = """Find relationships between tables:

Table A columns: {table_a_cols}
Table B columns: {table_b_cols}

Common columns: {common_cols}

Best join strategy:
- Join type: inner/left/outer
- Join columns: [column list]
- Confidence: 0-100%"""

    # Verification templates
    FORWARD_VERIFICATION = """Verify if code execution result matches the query intent:

QUERY: {query}
CODE: {code}
RESULT: {result}

Valid? YES/NO
Reason: <brief explanation>"""

    BACKWARD_VERIFICATION = """Check if the answer satisfies the original question:

QUESTION: {original_query}
ANSWER: {answer}
CONTEXT: {context}

Confidence score (0-100): 
Issues (if any): """

    # Memory templates
    MEMORY_UPDATE = """Update user memory with new interaction:

Previous memory: {previous_memory}
New interaction: Query="{query}" Result="{result}"

Updated memory (key points only):
- Preference: 
- Pattern: 
- Context: """

    # Error handling template
    ERROR_ANALYSIS = """Analyze execution error:

Code: {code}
Error: {error}

Fix strategy:
1. Issue: 
2. Solution: 
3. New code: """


class PromptOptimizer:
    """Optimize prompts to reduce token usage."""
    
    @staticmethod
    def compress_data_context(data_info: Dict[str, Any], max_tokens: int = 500) -> Dict[str, Any]:
        """Compress data context to fit within token limit."""
        compressed = {}
        
        # Compress column info
        if 'columns' in data_info:
            cols = data_info['columns']
            if len(cols) > 10:
                compressed['columns'] = cols[:8] + ['...'] + cols[-2:]
            else:
                compressed['columns'] = cols
        
        # Compress sample data
        if 'sample_data' in data_info:
            sample = data_info['sample_data']
            if isinstance(sample, list) and len(sample) > 5:
                compressed['sample_data'] = sample[:3] + ['...', sample[-1]]
            else:
                compressed['sample_data'] = sample
        
        # Keep essential fields
        for key in ['file_path', 'data_types', 'sheet_name']:
            if key in data_info:
                compressed[key] = data_info[key]
        
        return compressed
    
    @staticmethod
    def truncate_schema(schema: str, max_lines: int = 20) -> str:
        """Truncate schema to maximum lines."""
        lines = schema.split('\n')
        if len(lines) <= max_lines:
            return schema
        
        # Keep first 15 lines and last 3 lines
        return '\n'.join(lines[:15]) + '\n...\n' + '\n'.join(lines[-3:])
    
    @staticmethod
    def choose_template(query_complexity: str, data_size: str) -> str:
        """Choose appropriate template based on complexity."""
        if query_complexity == 'simple' and data_size == 'small':
            return PromptTemplates.SIMPLE_CODE_GENERATION
        else:
            return PromptTemplates.CODE_GENERATION
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    @staticmethod
    def optimize_for_model(prompt: str, model_name: str, max_tokens: int = 4000) -> str:
        """Optimize prompt for specific model constraints."""
        estimated_tokens = PromptOptimizer.estimate_tokens(prompt)
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Progressive compression
        lines = prompt.split('\n')
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Compress examples if present
        if 'EXAMPLE:' in prompt:
            example_start = None
            for i, line in enumerate(lines):
                if 'EXAMPLE:' in line:
                    example_start = i
                    break
            
            if example_start and len(lines) > example_start + 5:
                lines = lines[:example_start + 3] + ['...'] + lines[-2:]
        
        compressed = '\n'.join(lines)
        
        # If still too long, truncate
        if PromptOptimizer.estimate_tokens(compressed) > max_tokens:
            char_limit = max_tokens * 4
            compressed = compressed[:char_limit] + '...'
        
        return compressed


class ContextualPrompts:
    """Context-aware prompt generation."""
    
    @staticmethod
    def generate_query_prompt(query: str, context: Dict[str, Any], query_type: str = "general") -> str:
        """Generate contextual query prompt."""
        
        # Compress context
        optimizer = PromptOptimizer()
        compressed_context = optimizer.compress_data_context(context)
        
        if query_type == "simple":
            template = PromptTemplates.SIMPLE_CODE_GENERATION
            return template.format(query=query, context=str(compressed_context))
        
        elif query_type == "complex":
            template = PromptTemplates.CODE_GENERATION
            return template.format(
                query=query,
                file_path=compressed_context.get('file_path', 'unknown'),
                columns=compressed_context.get('columns', []),
                data_types=compressed_context.get('data_types', {}),
                sample_data=compressed_context.get('sample_data', [])
            )
        
        else:
            # Auto-detect complexity
            if len(query.split()) < 10 and 'join' not in query.lower():
                return ContextualPrompts.generate_query_prompt(query, context, "simple")
            else:
                return ContextualPrompts.generate_query_prompt(query, context, "complex")
    
    @staticmethod
    def generate_decomposition_prompt(query: str, schema: str) -> str:
        """Generate query decomposition prompt."""
        optimizer = PromptOptimizer()
        truncated_schema = optimizer.truncate_schema(schema, max_lines=15)
        
        return PromptTemplates.QUERY_DECOMPOSE.format(
            query=query,
            schema=truncated_schema
        )
    
    @staticmethod
    def generate_verification_prompt(query: str, code: str, result: str, verification_type: str = "forward") -> str:
        """Generate verification prompt."""
        if verification_type == "forward":
            return PromptTemplates.FORWARD_VERIFICATION.format(
                query=query,
                code=code[:500] + ('...' if len(code) > 500 else ''),
                result=str(result)[:200] + ('...' if len(str(result)) > 200 else '')
            )
        else:
            return PromptTemplates.BACKWARD_VERIFICATION.format(
                original_query=query,
                answer=str(result)[:300] + ('...' if len(str(result)) > 300 else ''),
                context="Table analysis context"
            )


# Global prompt manager
prompt_manager = ContextualPrompts()
prompt_optimizer = PromptOptimizer()
templates = PromptTemplates()