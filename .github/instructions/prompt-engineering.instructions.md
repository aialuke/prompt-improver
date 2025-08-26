# Prompt Engineering Instructions for APES

## Overview
This file provides GitHub Copilot with specific guidance for creating, optimizing, and managing prompts within the Adaptive Prompt Enhancement System (APES). Follow 2025 best practices for prompt engineering including structured outputs, chain-of-thought reasoning, and multi-modal prompt design.

## 2025 Prompt Engineering Best Practices

### 1. Structured Output Patterns
When creating prompts that require specific output formats:

```python
# Good: Structured prompt with clear output specification
prompt_template = """
You are an expert code reviewer. Analyze the provided code and return your response in the following JSON format:

{
    "overall_score": <1-10>,
    "issues": [
        {
            "type": "performance|security|style|logic",
            "severity": "low|medium|high|critical",
            "line": <line_number>,
            "description": "<clear description>",
            "suggestion": "<actionable fix>"
        }
    ],
    "strengths": ["<strength1>", "<strength2>"],
    "recommendations": ["<rec1>", "<rec2>"]
}

Code to review:
{code_content}
"""
```

### 2. Chain-of-Thought Prompting
Use step-by-step reasoning for complex tasks:

```python
# Good: Chain-of-thought structure
cot_prompt = """
Let's analyze this API performance issue step by step:

1. **Data Analysis**: First, examine the metrics data to identify patterns
2. **Root Cause Investigation**: Then, correlate timing with system events
3. **Impact Assessment**: Evaluate the severity and user impact
4. **Solution Design**: Propose specific optimization strategies
5. **Implementation Plan**: Create actionable next steps

Context: {performance_data}

Please work through each step systematically:
"""
```

### 3. Few-Shot Learning Patterns
Provide 2-3 high-quality examples for complex tasks:

```python
# Good: Few-shot prompt with diverse examples
few_shot_prompt = """
Create FastAPI route handlers following these patterns:

Example 1 - Simple CRUD:
```python
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user
```

Example 2 - Complex Query with Validation:
```python
@app.post("/analytics/query", response_model=AnalyticsResponse)
async def analytics_query(
    query: AnalyticsQuery,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Validate permissions
    if not await PermissionService.can_access_analytics(db, current_user.id):
        raise HTTPException(403, "Insufficient permissions")
    
    # Execute query with caching
    cache_key = f"analytics:{hash(query.json())}"
    if cached := await redis_client.get(cache_key):
        return AnalyticsResponse.parse_raw(cached)
    
    result = await AnalyticsService.execute_query(db, query)
    await redis_client.setex(cache_key, 300, result.json())
    return result
```

Now create a similar handler for: {task_description}
"""
```

### 4. XML-Tagged Structure
Use XML tags for complex, multi-part prompts:

```python
# Good: XML-structured prompt for clarity
xml_prompt = """
<task>Optimize the database query performance</task>

<context>
<current_query>{sql_query}</current_query>
<performance_metrics>
- Execution time: {exec_time}ms
- Rows examined: {rows_examined}
- Index usage: {index_info}
</performance_metrics>
<constraints>
- Must maintain data accuracy
- Cannot change table structure
- PostgreSQL 14+ features available
</constraints>
</context>

<instructions>
1. Analyze the query execution plan
2. Identify performance bottlenecks
3. Propose specific optimizations
4. Provide the optimized query
5. Explain the performance improvements
</instructions>

<output_format>
Return your analysis in markdown format with clear sections for each step.
</output_format>
"""
```

### 5. System Message Patterns
Create clear system prompts for consistent behavior:

```python
# Good: Comprehensive system prompt
system_prompt = """
You are a senior software engineer specializing in Python FastAPI applications with expertise in:
- Async/await patterns and performance optimization
- SQLModel/SQLAlchemy ORM best practices  
- Redis caching and session management
- PostgreSQL query optimization
- OpenTelemetry observability
- Production-ready error handling and logging

When providing code:
- Always use type hints and async patterns
- Include proper error handling with custom exceptions
- Add logging statements for debugging
- Consider caching opportunities
- Follow the existing project structure in APES
- Use dependency injection patterns
- Include docstrings for complex functions

When analyzing issues:
- Start with the most likely causes
- Provide specific, actionable solutions
- Consider performance and security implications
- Reference relevant logs or metrics when available
"""
```

### 6. Multi-Modal Prompt Design
For prompts that include images, audio, or other media:

```python
# Good: Multi-modal prompt structure
multimodal_prompt = """
Analyze this system architecture diagram and the accompanying error logs:

Image: {base64_image_data}

Error Logs:
```
{error_logs}
```

Please provide:
1. **Visual Analysis**: What components and data flows do you see in the diagram?
2. **Error Correlation**: How do the errors relate to the architecture shown?
3. **Root Cause**: What's the most likely source of these issues?
4. **Recommendations**: Specific changes to improve reliability

Focus on the interaction between the API Gateway, microservices, and database components.
"""
```

### 7. Prefilling and Completion Patterns
Start assistant responses to guide output format:

```python
# Good: Prefilled response format
prefilled_prompt = """
Review this pull request and provide feedback:

PR Details: {pr_details}
Changed Files: {file_changes}

## Code Review Analysis

### Summary
This pull request introduces 

### Key Changes
1. **New Features**: 
2. **Bug Fixes**: 
3. **Performance Improvements**: 

### Detailed Feedback

#### Security Considerations
- 

#### Performance Impact
- 

#### Code Quality
- 

### Recommendation
Based on this analysis, I recommend: [APPROVE/REQUEST_CHANGES/COMMENT] because
"""
```

## APES-Specific Patterns

### 1. Prompt Evaluation Templates
```python
# Template for creating evaluation prompts
evaluation_prompt = """
Evaluate this prompt for effectiveness in the APES system:

**Original Prompt:**
{original_prompt}

**Evaluation Criteria:**
1. **Clarity**: Is the instruction clear and unambiguous?
2. **Specificity**: Does it provide enough context and constraints?
3. **Structure**: Is it well-organized and easy to follow?
4. **Output Format**: Does it specify the expected response format?
5. **Error Handling**: Does it account for edge cases?

**Performance Metrics:**
- Current success rate: {success_rate}%
- Average response time: {avg_time}ms
- User satisfaction: {satisfaction_score}/10

**Suggest 3 specific improvements:**
1. 
2. 
3. 

**Provide an optimized version:**
```optimized_prompt
[Your improved prompt here]
```
"""
```

### 2. Context-Aware Prompt Generation
```python
# Dynamic prompt generation based on context
context_prompt = """
Generate a specialized prompt for the following context:

**Task Type**: {task_type}
**User Expertise**: {user_level}  # beginner, intermediate, advanced
**Domain**: {domain}  # api_development, data_analysis, ml_training, etc.
**Output Format**: {output_format}  # code, analysis, documentation, etc.
**Constraints**: {constraints}

**Requirements:**
- Adapt language complexity to user expertise level
- Include relevant examples for the domain
- Structure for the specified output format
- Incorporate the given constraints

**Template Structure:**
1. Clear role definition
2. Context setting
3. Specific instructions
4. Output format specification
5. Quality criteria

Generate the optimized prompt:
"""
```

### 3. A/B Testing Prompt Variants
```python
# Template for creating prompt variants
ab_test_prompt = """
Create two variants of this prompt for A/B testing:

**Original Prompt**: {original_prompt}
**Test Hypothesis**: {hypothesis}
**Success Metric**: {metric}

**Variant A - Control**: 
[Minimal changes to original]

**Variant B - Experimental**: 
[Implement the hypothesis]

**Test Plan**:
- Sample size: {sample_size}
- Duration: {test_duration}
- Success criteria: {success_criteria}

**Predicted Outcomes**:
- Variant A: {prediction_a}
- Variant B: {prediction_b}
"""
```

## Best Practices for APES

### 1. Prompt Versioning
- Always version prompts with semantic versioning (v1.2.3)
- Document changes and performance impacts
- Maintain backwards compatibility when possible

### 2. Performance Optimization
- Keep prompts concise but complete
- Use caching for expensive prompt generation
- Monitor token usage and optimize for cost

### 3. Error Handling
- Include fallback behaviors in prompts
- Handle edge cases gracefully
- Provide clear error messages

### 4. Testing and Validation
- Create test cases for each prompt template
- Validate outputs against expected formats
- Monitor real-world performance metrics

### 5. Documentation
- Document prompt intent and expected outcomes
- Include usage examples and common pitfalls
- Maintain a prompt library with categories

## Common Anti-Patterns to Avoid

### ❌ Avoid: Vague Instructions
```python
# Bad
prompt = "Make this code better"
```

### ❌ Avoid: No Output Format
```python
# Bad
prompt = "Analyze the performance data and tell me what you think"
```

### ❌ Avoid: Too Many Tasks
```python
# Bad
prompt = "Review the code, fix bugs, optimize performance, update docs, and deploy"
```

### ❌ Avoid: No Context
```python
# Bad
prompt = "Debug this error: {error_message}"
```

### ✅ Good: Clear, Structured, Contextual
```python
# Good
prompt = """
You are a Python performance expert. Analyze the provided profiling data and identify the top 3 performance bottlenecks.

Context:
- Application: FastAPI web service
- Load: 1000 requests/second
- Environment: Production

Profiling Data:
{profiling_data}

Required Output Format:
1. **Bottleneck 1**: Description | Impact | Solution
2. **Bottleneck 2**: Description | Impact | Solution  
3. **Bottleneck 3**: Description | Impact | Solution

For each bottleneck, provide:
- Specific line numbers or functions
- Quantified performance impact
- Concrete optimization steps
"""
```

## Integration with MCP

When creating prompts for MCP servers:

1. **Use Structured Arguments**: Define clear, typed parameters
2. **Support Completion**: Implement argument auto-completion
3. **Handle Resources**: Embed relevant resources when needed
4. **Multi-Modal Support**: Include image/audio content types when applicable
5. **Error Responses**: Return meaningful error messages for invalid inputs

Follow these patterns to create effective, maintainable, and performant prompts within the APES ecosystem.
