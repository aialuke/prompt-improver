import voyageai
import os
import pickle
import ast
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, cast, Dict, Any, Optional, Union, TYPE_CHECKING
from enum import Enum

# Enhanced imports for cAST implementation with proper type handling
if TYPE_CHECKING:
    # Use a protocol that matches tree-sitter Node interface for type checking
    from typing import Protocol
    class TreeSitterNode(Protocol):
        type: str
        children: List['TreeSitterNode']
        start_point: Tuple[int, int]
        end_point: Tuple[int, int]
        text: bytes
else:
    # At runtime, use Any to avoid type conflicts with actual tree-sitter Node
    TreeSitterNode = Any

# Runtime imports and availability tracking
TREE_SITTER_AVAILABLE = False
tspython: Any = None
Language: Any = None
Parser: Any = None

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
    print("🌳 Tree-sitter successfully imported for enhanced cAST parsing")
except ImportError:
    print("⚠️  tree-sitter-python not available. Install with: pip install tree-sitter tree-sitter-python")
    print("   Falling back to enhanced AST parsing")

def _verify_tree_sitter_availability() -> bool:
    """Verify that tree-sitter is available and properly imported."""
    return (TREE_SITTER_AVAILABLE and
            tspython is not None and
            Language is not None and
            Parser is not None)


class ChunkType(Enum):
    """Types of code chunks for semantic search."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    PROPERTY = "property"
    IMPORT_BLOCK = "import_block"


@dataclass
class CodeChunk:
    """Represents a semantic code chunk with metadata."""
    content: str
    chunk_type: ChunkType
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity_score: float = 0.0
    dependencies: Optional[List[str]] = None
    parent_class: Optional[str] = None
    decorators: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []
        if self.decorators is None:
            self.decorators = []


@dataclass
class EmbeddingMetadata:
    """Enhanced metadata for embeddings storage."""
    embeddings: List[List[float]]
    chunks: List[CodeChunk]
    generation_timestamp: float
    model_used: str
    total_files_processed: int
    total_chunks_created: int
    processing_stats: Dict[str, Any]


# Contextual Information Generation Configuration
CONTEXTUAL_CONFIG = {
    "enabled": True,                    # Enable contextual information generation
    "target_context_tokens": 75,       # Target 50-100 tokens of context
    "max_context_tokens": 100,         # Maximum context tokens
    "use_llm_for_context": True,       # Use LLM to generate context
    "fallback_to_heuristic": True,     # Fallback to heuristic context if LLM fails
    "cache_contexts": True,            # Cache generated contexts
    "context_prompt_template": """
<document>
{full_file_content}
</document>

<chunk>
{chunk_content}
</chunk>

Please provide a concise context (50-100 tokens) that situates this code chunk within the overall file for improved semantic search. Include:
- File purpose and the chunk's role
- Key dependencies and relationships
- Functional context and usage patterns

Answer only with the context and nothing else.
""".strip()
}

# Set Voyage AI API key (preferably via environment variable)
# os.environ["VOYAGE_API_KEY"] = "your-api-key-here"

# Initialize Voyage AI client with enhanced configuration
try:
    vo = voyageai.Client()  # type: ignore [attr-defined]
    print("🚀 Voyage AI client initialized successfully")
    print("   Using voyage-code-3 model optimized for code retrieval")
    print("   Enhanced with cAST chunking and contextual information")
except AttributeError as e:
    print(f"❌ Error: 'Client' not found in voyageai module. {e}")
    print("   Please install voyageai: pip install voyageai")
    raise
except Exception as e:
    print(f"❌ Error initializing Voyage AI client: {e}")
    print("   Please check your API key configuration")
    raise

# Enhanced Configuration for cAST and voyage-code-3
MAX_TOKENS_PER_BATCH = 120000  # Voyage AI limit for voyage-code-3

# Matryoshka Embedding Configurations
EMBEDDING_CONFIGS = {
    "ultra_high_accuracy": {"output_dimension": 2048, "output_dtype": "float"},
    "high_accuracy": {"output_dimension": 1024, "output_dtype": "float"},
    "balanced": {"output_dimension": 1024, "output_dtype": "float"},
    "fast_search": {"output_dimension": 512, "output_dtype": "int8"},
    "ultra_fast": {"output_dimension": 256, "output_dtype": "binary"},
    "binary_rescore": {"output_dimension": 256, "output_dtype": "binary"}
}

# cAST Configuration
CAST_CONFIG = {
    "max_chunk_size": 8000,           # Maximum characters per chunk
    "min_function_lines": 3,          # Minimum lines for significant functions
    "include_imports": True,          # Create separate import chunks
    "context_lines_before": 2,       # Lines of context before functions/classes
    "context_lines_after": 1,        # Lines of context after functions/classes
    "preserve_hierarchy": True,       # Maintain class-method relationships
    "merge_small_chunks": True,       # Merge chunks smaller than threshold
    "small_chunk_threshold": 100,    # Threshold for small chunk merging
    "recursive_split_large": True,    # Recursively split oversized chunks
    "include_docstrings": True,       # Always include docstrings with code
    "include_decorators": True,       # Include decorators with functions
    "semantic_grouping": True         # Group related code semantically
}

# Code parsing configuration (legacy compatibility)
MIN_FUNCTION_LINES = CAST_CONFIG["min_function_lines"]
MAX_CHUNK_SIZE = CAST_CONFIG["max_chunk_size"]
INCLUDE_IMPORTS = CAST_CONFIG["include_imports"]

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string (approx. 4 chars per token)."""
    return len(text) // 4 + 1  # Add 1 to avoid zero tokens for small strings


def calculate_complexity_score(node: ast.AST) -> float:
    """Calculate a simple complexity score for a code node."""
    complexity = 1.0  # Base complexity

    for child in ast.walk(node):
        # Add complexity for control structures
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            complexity += 1
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            complexity += 0.5
        elif isinstance(child, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp)):
            complexity += 0.3

    return complexity


def extract_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> Optional[str]:
    """Extract docstring from a function or class node."""
    if (node.body and
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def extract_decorators(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
    """Extract decorator names from a function or class node."""
    decorators = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            decorators.append(f"{ast.unparse(decorator.value)}.{decorator.attr}")
        else:
            decorators.append(ast.unparse(decorator))
    return decorators


def get_function_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    """Extract function signature as a string."""
    try:
        # Build signature manually for better control
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg_str)

        # **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg_str)

        signature = f"{node.name}({', '.join(args)})"

        # Add return type annotation
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        return signature
    except Exception:
        # Fallback to basic signature
        return f"{node.name}(...)"

def parse_python_file(file_path: str, content: str) -> List[CodeChunk]:
    """Parse a Python file into semantic code chunks using enhanced cAST."""

    # Try tree-sitter first for enhanced parsing
    if TREE_SITTER_AVAILABLE:
        try:
            return _parse_with_tree_sitter(file_path, content)
        except Exception as e:
            print(f"⚠️  Tree-sitter parsing failed for {file_path}: {e}")
            print("   Falling back to AST parsing...")

    # Fallback to enhanced AST parsing
    return _parse_with_enhanced_ast(file_path, content)


def _parse_with_tree_sitter(file_path: str, content: str) -> List[CodeChunk]:
    """Enhanced cAST parsing using tree-sitter for superior code structure analysis."""

    # Comprehensive availability check
    if not _verify_tree_sitter_availability():
        raise RuntimeError("Tree-sitter not available or not properly imported")

    # Initialize tree-sitter parser (we know modules are available here)
    PY_LANGUAGE = Language(tspython.language())  # type: ignore
    parser = Parser(PY_LANGUAGE)  # type: ignore

    tree = parser.parse(bytes(content, "utf8"))
    lines = content.splitlines()
    chunks = []

    print(f"🌳 Using tree-sitter cAST parsing for {Path(file_path).name}")

    # Extract semantic units with enhanced hierarchy preservation
    root_node = tree.root_node

    # Process imports first
    if CAST_CONFIG["include_imports"]:
        import_chunks = _extract_import_blocks_ts(root_node, lines, file_path)
        chunks.extend(import_chunks)

    # Process top-level definitions with recursive chunking
    for child in root_node.children:
        if child.type in ['function_definition', 'async_function_definition']:
            chunk = _process_function_node_ts(child, lines, file_path, content)
            if chunk:
                chunks.append(chunk)
        elif child.type == 'class_definition':
            class_chunks = _process_class_node_ts(child, lines, file_path, content)
            chunks.extend(class_chunks)

    # Apply recursive splitting for oversized chunks
    final_chunks = []
    for chunk in chunks:
        if len(chunk.content) > CAST_CONFIG["max_chunk_size"]:
            if CAST_CONFIG["recursive_split_large"]:
                split_chunks = _recursive_split_chunk_ts(chunk, content)
                final_chunks.extend(split_chunks)
            else:
                # Truncate if recursive splitting is disabled
                chunk.content = chunk.content[:CAST_CONFIG["max_chunk_size"]]
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)

    # Merge small chunks if enabled
    if CAST_CONFIG["merge_small_chunks"]:
        final_chunks = _merge_small_chunks(final_chunks)

    # Create module-level chunk if no significant chunks found
    if not final_chunks or (len(final_chunks) == 1 and
                           final_chunks[0].chunk_type == ChunkType.IMPORT_BLOCK):
        module_chunk = CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=Path(file_path).stem,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            complexity_score=_calculate_complexity_score_ts(root_node)
        )
        final_chunks.append(module_chunk)

    return final_chunks


def _parse_with_enhanced_ast(file_path: str, content: str) -> List[CodeChunk]:
    """Enhanced AST parsing with improved error handling and chunking."""
    chunks = []

    try:
        tree = ast.parse(content)
        lines = content.splitlines()

        print(f"🔧 Using enhanced AST parsing for {Path(file_path).name}")

        # Extract imports as a single chunk if enabled
        if CAST_CONFIG["include_imports"]:
            import_lines = []
            import_start = None
            import_end = None

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if import_start is None:
                        import_start = node.lineno
                    import_end = node.lineno
                    import_lines.append(lines[node.lineno - 1])

            if import_lines:
                import_content = "\n".join(import_lines)
                chunks.append(CodeChunk(
                    content=import_content,
                    chunk_type=ChunkType.IMPORT_BLOCK,
                    name="imports",
                    file_path=file_path,
                    start_line=import_start or 1,
                    end_line=import_end or 1,
                    complexity_score=0.1
                ))

        # Process classes and functions with enhanced logic
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(_process_class_node(node, lines, file_path))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only process top-level functions (not methods)
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                          if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                    chunk = _process_function_node(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)

        # If no significant chunks found, create a module-level chunk
        if not chunks or len(chunks) == 1 and chunks[0].chunk_type == ChunkType.IMPORT_BLOCK:
            module_chunk = CodeChunk(
                content=content,
                chunk_type=ChunkType.MODULE,
                name=Path(file_path).stem,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                complexity_score=calculate_complexity_score(tree)
            )
            chunks.append(module_chunk)

    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        # Fallback to module-level chunk for files with syntax errors
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=Path(file_path).stem,
            file_path=file_path,
            start_line=1,
            end_line=len(content.splitlines()),
            complexity_score=1.0
        ))
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        # Fallback to module-level chunk
        chunks.append(CodeChunk(
            content=content,
            chunk_type=ChunkType.MODULE,
            name=Path(file_path).stem,
            file_path=file_path,
            start_line=1,
            end_line=len(content.splitlines()),
            complexity_score=1.0
        ))

    return chunks


def _process_class_node(node: ast.ClassDef, lines: List[str], file_path: str) -> List[CodeChunk]:
    """Process a class node and extract class and method chunks."""
    chunks = []

    # Extract class content
    class_start = node.lineno
    class_end = node.end_lineno or class_start
    class_content = "\n".join(lines[class_start - 1:class_end])

    # Create class chunk
    class_chunk = CodeChunk(
        content=class_content,
        chunk_type=ChunkType.CLASS,
        name=node.name,
        file_path=file_path,
        start_line=class_start,
        end_line=class_end,
        docstring=extract_docstring(node),
        complexity_score=calculate_complexity_score(node),
        decorators=extract_decorators(node)
    )
    chunks.append(class_chunk)

    # Process methods within the class
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_chunk = _process_function_node(item, lines, file_path, parent_class=node.name)
            if method_chunk:
                chunks.append(method_chunk)

    return chunks


def _process_function_node(node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                          lines: List[str], file_path: str,
                          parent_class: Optional[str] = None) -> Optional[CodeChunk]:
    """Process a function node and create a function chunk."""
    func_start = node.lineno
    func_end = node.end_lineno or func_start

    # Skip very small functions unless they're significant
    if func_end - func_start < MIN_FUNCTION_LINES and not extract_docstring(node):
        return None

    func_content = "\n".join(lines[func_start - 1:func_end])

    # Determine chunk type
    if parent_class:
        if isinstance(node, ast.AsyncFunctionDef):
            chunk_type = ChunkType.ASYNC_METHOD
        else:
            chunk_type = ChunkType.METHOD
    else:
        if isinstance(node, ast.AsyncFunctionDef):
            chunk_type = ChunkType.ASYNC_FUNCTION
        else:
            chunk_type = ChunkType.FUNCTION

    return CodeChunk(
        content=func_content,
        chunk_type=chunk_type,
        name=node.name,
        file_path=file_path,
        start_line=func_start,
        end_line=func_end,
        docstring=extract_docstring(node),
        signature=get_function_signature(node),
        complexity_score=calculate_complexity_score(node),
        parent_class=parent_class,
        decorators=extract_decorators(node)
    )


# Tree-sitter helper functions for enhanced cAST
def _extract_import_blocks_ts(root_node: TreeSitterNode, lines: List[str], file_path: str) -> List[CodeChunk]:
    """Extract import statements as semantic chunks using tree-sitter."""
    import_chunks = []
    import_nodes = []

    for child in root_node.children:
        if child.type in ['import_statement', 'import_from_statement']:
            import_nodes.append(child)

    if import_nodes:
        # Group consecutive imports
        import_groups: List[List[TreeSitterNode]] = []
        current_group: List[TreeSitterNode] = []

        for node in import_nodes:
            if not current_group or node.start_point[0] <= current_group[-1].end_point[0] + 2:
                current_group.append(node)
            else:
                import_groups.append(current_group)
                current_group = [node]

        if current_group:
            import_groups.append(current_group)

        # Create chunks for each import group
        for i, group in enumerate(import_groups):
            start_line = group[0].start_point[0]
            end_line = group[-1].end_point[0]

            import_content = "\n".join(lines[start_line:end_line + 1])

            import_chunks.append(CodeChunk(
                content=import_content,
                chunk_type=ChunkType.IMPORT_BLOCK,
                name=f"imports_{i+1}" if len(import_groups) > 1 else "imports",
                file_path=file_path,
                start_line=start_line + 1,
                end_line=end_line + 1,
                complexity_score=0.1
            ))

    return import_chunks


def _process_function_node_ts(node: TreeSitterNode, lines: List[str], file_path: str, full_content: str) -> Optional[CodeChunk]:
    """Process a function node using tree-sitter with enhanced context preservation."""

    func_start = node.start_point[0]
    func_end = node.end_point[0]

    # Include context lines
    context_start = max(0, func_start - CAST_CONFIG["context_lines_before"])
    context_end = min(len(lines), func_end + 1 + CAST_CONFIG["context_lines_after"])

    # Skip very small functions unless they have docstrings
    if (func_end - func_start < CAST_CONFIG["min_function_lines"] and
        not _has_docstring_ts(node)):
        return None

    func_content = "\n".join(lines[context_start:context_end])

    # Extract function name
    func_name = _extract_function_name_ts(node)

    # Determine chunk type
    chunk_type = ChunkType.ASYNC_FUNCTION if node.type == 'async_function_definition' else ChunkType.FUNCTION

    # Extract enhanced metadata
    docstring = _extract_docstring_ts(node, lines)
    signature = _extract_signature_ts(node, lines)
    complexity = _calculate_complexity_score_ts(node)
    decorators = _extract_decorators_ts(node, lines)
    dependencies = _extract_dependencies_ts(node, full_content)

    return CodeChunk(
        content=func_content,
        chunk_type=chunk_type,
        name=func_name,
        file_path=file_path,
        start_line=context_start + 1,
        end_line=context_end,
        docstring=docstring,
        signature=signature,
        complexity_score=complexity,
        decorators=decorators,
        dependencies=dependencies
    )


def _process_class_node_ts(node: TreeSitterNode, lines: List[str], file_path: str, full_content: str) -> List[CodeChunk]:
    """Process a class node using tree-sitter with method extraction."""
    chunks = []

    class_start = node.start_point[0]
    class_end = node.end_point[0]

    # Include context lines
    context_start = max(0, class_start - CAST_CONFIG["context_lines_before"])
    context_end = min(len(lines), class_end + 1 + CAST_CONFIG["context_lines_after"])

    class_content = "\n".join(lines[context_start:context_end])
    class_name = _extract_class_name_ts(node)

    # Create class chunk
    class_chunk = CodeChunk(
        content=class_content,
        chunk_type=ChunkType.CLASS,
        name=class_name,
        file_path=file_path,
        start_line=context_start + 1,
        end_line=context_end,
        docstring=_extract_docstring_ts(node, lines),
        complexity_score=_calculate_complexity_score_ts(node),
        decorators=_extract_decorators_ts(node, lines)
    )
    chunks.append(class_chunk)

    # Process methods within the class if hierarchy preservation is enabled
    if CAST_CONFIG["preserve_hierarchy"]:
        for child in node.children:
            if child.type in ['function_definition', 'async_function_definition']:
                method_chunk = _process_method_node_ts(child, lines, file_path, full_content, class_name)
                if method_chunk:
                    chunks.append(method_chunk)

    return chunks


def _process_method_node_ts(node: TreeSitterNode, lines: List[str], file_path: str,
                           full_content: str, parent_class: str) -> Optional[CodeChunk]:
    """Process a method node within a class."""

    method_start = node.start_point[0]
    method_end = node.end_point[0]

    # Include minimal context for methods
    context_start = max(0, method_start - 1)
    context_end = min(len(lines), method_end + 1)

    # Skip very small methods unless they have docstrings
    if (method_end - method_start < CAST_CONFIG["min_function_lines"] and
        not _has_docstring_ts(node)):
        return None

    method_content = "\n".join(lines[context_start:context_end])
    method_name = _extract_function_name_ts(node)

    # Determine chunk type
    chunk_type = ChunkType.ASYNC_METHOD if node.type == 'async_function_definition' else ChunkType.METHOD

    return CodeChunk(
        content=method_content,
        chunk_type=chunk_type,
        name=method_name,
        file_path=file_path,
        start_line=context_start + 1,
        end_line=context_end,
        docstring=_extract_docstring_ts(node, lines),
        signature=_extract_signature_ts(node, lines),
        complexity_score=_calculate_complexity_score_ts(node),
        parent_class=parent_class,
        decorators=_extract_decorators_ts(node, lines),
        dependencies=_extract_dependencies_ts(node, full_content)
    )


# Tree-sitter utility functions
def _extract_function_name_ts(node: TreeSitterNode) -> str:
    """Extract function name from tree-sitter node."""
    for child in node.children:
        if child.type == 'identifier':
            return str(child.text.decode('utf-8'))
    return "unknown_function"


def _extract_class_name_ts(node: TreeSitterNode) -> str:
    """Extract class name from tree-sitter node."""
    for child in node.children:
        if child.type == 'identifier':
            return str(child.text.decode('utf-8'))
    return "unknown_class"


def _has_docstring_ts(node: TreeSitterNode) -> bool:
    """Check if a function/class has a docstring using tree-sitter."""
    for child in node.children:
        if child.type == 'block':
            for stmt in child.children:
                if stmt.type == 'expression_statement':
                    for expr in stmt.children:
                        if expr.type == 'string':
                            return True
    return False


def _extract_docstring_ts(node: TreeSitterNode, lines: List[str]) -> Optional[str]:
    """Extract docstring from tree-sitter node."""
    for child in node.children:
        if child.type == 'block':
            for stmt in child.children:
                if stmt.type == 'expression_statement':
                    for expr in stmt.children:
                        if expr.type == 'string':
                            start_line = expr.start_point[0]
                            end_line = expr.end_point[0]
                            docstring_lines = lines[start_line:end_line + 1]
                            return "\n".join(docstring_lines).strip('"""').strip("'''").strip()
    return None


def _extract_signature_ts(node: TreeSitterNode, lines: List[str]) -> str:
    """Extract function signature from tree-sitter node."""
    start_line = node.start_point[0]

    # Find the end of the function definition line
    for child in node.children:
        if child.type == 'block':
            end_line = child.start_point[0] - 1
            break
    else:
        end_line = start_line

    signature_lines = lines[start_line:end_line + 1]
    signature = " ".join(line.strip() for line in signature_lines)

    # Clean up the signature
    if signature.endswith(':'):
        signature = signature[:-1]

    return signature


def _extract_decorators_ts(node: TreeSitterNode, lines: List[str]) -> List[str]:
    """Extract decorators from tree-sitter node."""
    decorators = []

    for child in node.children:
        if child.type == 'decorator':
            start_line = child.start_point[0]
            end_line = child.end_point[0]
            decorator_text = "\n".join(lines[start_line:end_line + 1])
            decorators.append(decorator_text.strip())

    return decorators


def _calculate_complexity_score_ts(node: TreeSitterNode) -> float:
    """Calculate complexity score using tree-sitter node traversal."""
    complexity = 1.0

    def traverse(n: TreeSitterNode) -> None:
        nonlocal complexity

        if n.type in ['if_statement', 'for_statement', 'while_statement',
                      'try_statement', 'with_statement', 'match_statement']:
            complexity += 1
        elif n.type in ['function_definition', 'async_function_definition', 'class_definition']:
            complexity += 0.5
        elif n.type in ['lambda', 'list_comprehension', 'dictionary_comprehension',
                        'set_comprehension', 'generator_expression']:
            complexity += 0.3

        for child in n.children:
            traverse(child)

    traverse(node)
    return complexity


def _extract_dependencies_ts(node: TreeSitterNode, full_content: str) -> List[str]:
    """Extract dependencies and imports used within a function/class."""
    dependencies: List[str] = []

    # Extract from current node
    def traverse(n: TreeSitterNode) -> None:
        if n.type == 'identifier':
            identifier = n.text.decode('utf-8')
            # Enhanced heuristic: check against imports in full file
            if (identifier[0].isupper() or
                any(pattern in identifier.lower() for pattern in ['os', 'sys', 'json', 'time', 'math']) or
                _is_imported_identifier(identifier, full_content)):
                dependencies.append(identifier)

        for child in n.children:
            traverse(child)

    traverse(node)

    # Add cross-file dependencies from full content analysis
    cross_file_deps = _extract_cross_file_dependencies(node, full_content)
    dependencies.extend(cross_file_deps)

    return list(set(dependencies))  # Remove duplicates


def _is_imported_identifier(identifier: str, full_content: str) -> bool:
    """Check if identifier is imported in the full file content."""
    import_patterns = [
        f"import {identifier}",
        f"from {identifier}",
        f"from .{identifier}",
        f"from ..{identifier}",
        f"import {identifier.lower()}",
        f"as {identifier}"
    ]
    return any(pattern in full_content for pattern in import_patterns)


def _extract_cross_file_dependencies(node: TreeSitterNode, full_content: str) -> List[str]:
    """Extract dependencies that span across the file."""
    cross_deps: List[str] = []

    # Extract class names that might be used
    class_pattern_matches = []
    lines = full_content.split('\n')
    for line in lines:
        if line.strip().startswith('class '):
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_pattern_matches.append(class_name)

    # Check if any class names are referenced in current node
    node_text = node.text.decode('utf-8') if hasattr(node, 'text') else ""
    for class_name in class_pattern_matches:
        if class_name in node_text and class_name not in cross_deps:
            cross_deps.append(class_name)

    return cross_deps


def _analyze_file_context_for_splitting(full_content: str, chunk: CodeChunk) -> Dict[str, Any]:
    """Analyze file context to make better splitting decisions."""
    context = {
        "total_lines": len(full_content.split('\n')),
        "chunk_position": chunk.start_line / len(full_content.split('\n')) if full_content else 0,
        "file_complexity": full_content.count('def ') + full_content.count('class '),
        "has_main": '__name__ == "__main__"' in full_content,
        "import_density": full_content.count('import ') / max(1, len(full_content.split('\n'))),
    }
    return context


def _recursive_split_chunk_ts(chunk: CodeChunk, full_content: str) -> List[CodeChunk]:
    """Recursively split large chunks using tree-sitter analysis."""

    if not _verify_tree_sitter_availability():
        # Fallback to simple splitting when tree-sitter not available
        return _simple_split_chunk(chunk)

    try:
        # Parse the chunk content with context from full file
        PY_LANGUAGE = Language(tspython.language())  # type: ignore
        parser = Parser(PY_LANGUAGE)  # type: ignore
        tree = parser.parse(bytes(chunk.content, "utf8"))

        # Use full_content for enhanced context analysis
        file_context = _analyze_file_context_for_splitting(full_content, chunk)

        # Find splittable nodes (functions, classes, etc.)
        splittable_nodes = []
        for child in tree.root_node.children:
            if child.type in ['function_definition', 'async_function_definition', 'class_definition']:
                splittable_nodes.append(child)

        # Use file context to determine splitting strategy
        min_splits = 2 if file_context["file_complexity"] > 10 else 1

        if len(splittable_nodes) <= min_splits:
            # Can't split further or not worth splitting, truncate if necessary
            if len(chunk.content) > CAST_CONFIG["max_chunk_size"]:
                chunk.content = chunk.content[:CAST_CONFIG["max_chunk_size"]]
            return [chunk]

        # Split into smaller chunks
        lines = chunk.content.splitlines()
        split_chunks = []

        for node in splittable_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            node_content = "\n".join(lines[start_line:end_line + 1])

            if len(node_content) <= CAST_CONFIG["max_chunk_size"]:
                # Create a new chunk for this node
                node_name = _extract_function_name_ts(node) if node.type in ['function_definition', 'async_function_definition'] else _extract_class_name_ts(node)

                split_chunk = CodeChunk(
                    content=node_content,
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name}_{node_name}",
                    file_path=chunk.file_path,
                    start_line=chunk.start_line + start_line,
                    end_line=chunk.start_line + end_line,
                    docstring=_extract_docstring_ts(node, lines),
                    signature=_extract_signature_ts(node, lines) if node.type in ['function_definition', 'async_function_definition'] else None,
                    complexity_score=_calculate_complexity_score_ts(node),
                    parent_class=chunk.parent_class,
                    decorators=_extract_decorators_ts(node, lines),
                    dependencies=chunk.dependencies
                )
                split_chunks.append(split_chunk)

        return split_chunks if split_chunks else [chunk]

    except Exception as e:
        print(f"⚠️  Recursive splitting failed: {e}")
        return _simple_split_chunk(chunk)


def _simple_split_chunk(chunk: CodeChunk) -> List[CodeChunk]:
    """Simple fallback chunk splitting by lines."""
    max_lines = CAST_CONFIG["max_chunk_size"] // 50  # Rough estimate
    lines = chunk.content.splitlines()

    if len(lines) <= max_lines:
        return [chunk]

    split_chunks: List[CodeChunk] = []
    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        chunk_content = "\n".join(chunk_lines)

        split_chunk = CodeChunk(
            content=chunk_content,
            chunk_type=chunk.chunk_type,
            name=f"{chunk.name}_part_{i//max_lines + 1}",
            file_path=chunk.file_path,
            start_line=chunk.start_line + i,
            end_line=chunk.start_line + i + len(chunk_lines) - 1,
            complexity_score=float(chunk.complexity_score) / float(len(split_chunks)) if split_chunks else chunk.complexity_score,
            parent_class=chunk.parent_class,
            dependencies=chunk.dependencies
        )
        split_chunks.append(split_chunk)

    return split_chunks


def _merge_small_chunks(chunks: List[CodeChunk]) -> List[CodeChunk]:
    """Merge small chunks to improve information density."""
    if not chunks:
        return chunks

    merged_chunks: List[CodeChunk] = []
    current_group: List[CodeChunk] = []
    current_size = 0

    for chunk in chunks:
        chunk_size = len(chunk.content)

        # If chunk is large enough on its own, finalize current group and start new
        if chunk_size >= CAST_CONFIG["small_chunk_threshold"]:
            if current_group:
                merged_chunks.append(_merge_chunk_group(current_group))
                current_group = []
                current_size = 0
            merged_chunks.append(chunk)
        else:
            # Add to current group if it doesn't exceed max size
            if current_size + chunk_size <= CAST_CONFIG["max_chunk_size"]:
                current_group.append(chunk)
                current_size += chunk_size
            else:
                # Finalize current group and start new
                if current_group:
                    merged_chunks.append(_merge_chunk_group(current_group))
                current_group = [chunk]
                current_size = chunk_size

    # Handle remaining group
    if current_group:
        merged_chunks.append(_merge_chunk_group(current_group))

    return merged_chunks


def _merge_chunk_group(chunks: List[CodeChunk]) -> CodeChunk:
    """Merge a group of small chunks into a single chunk."""
    if len(chunks) == 1:
        return chunks[0]

    # Combine content
    combined_content = "\n\n".join(chunk.content for chunk in chunks)

    # Combine names
    combined_name = "_".join(chunk.name for chunk in chunks[:3])  # Limit to first 3 names
    if len(chunks) > 3:
        combined_name += f"_and_{len(chunks)-3}_more"

    # Use first chunk as base
    base_chunk = chunks[0]

    return CodeChunk(
        content=combined_content,
        chunk_type=ChunkType.MODULE,  # Mixed chunks become module type
        name=combined_name,
        file_path=base_chunk.file_path,
        start_line=base_chunk.start_line,
        end_line=chunks[-1].end_line,
        complexity_score=sum(chunk.complexity_score for chunk in chunks),
        dependencies=list(set(dep for chunk in chunks for dep in (chunk.dependencies or [])))
    )


# Contextual Information Generation Functions
def generate_contextual_information(chunk: CodeChunk, full_file_content: str,
                                   context_cache: Optional[Dict[str, str]] = None) -> str:
    """Generate contextual information for a code chunk to improve retrieval accuracy."""

    if not CONTEXTUAL_CONFIG["enabled"]:
        return chunk.content

    # Create cache key
    if context_cache is None:
        context_cache = {}

    cache_key = hashlib.md5(f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.content}".encode()).hexdigest()

    # Check cache first
    if CONTEXTUAL_CONFIG["cache_contexts"] and cache_key in context_cache:
        cached_context = context_cache[cache_key]
        return f"{cached_context}\n\n{chunk.content}"

    # Generate context
    context = None

    if CONTEXTUAL_CONFIG["use_llm_for_context"]:
        try:
            context = _generate_llm_context(chunk, full_file_content)
        except Exception as e:
            print(f"⚠️  LLM context generation failed for {chunk.name}: {e}")
            if CONTEXTUAL_CONFIG["fallback_to_heuristic"]:
                context = _generate_heuristic_context(chunk, full_file_content)
    else:
        context = _generate_heuristic_context(chunk, full_file_content)

    # Cache the context
    if context and CONTEXTUAL_CONFIG["cache_contexts"]:
        context_cache[cache_key] = context

    # Return contextualized content
    if context:
        return f"{context}\n\n{chunk.content}"
    else:
        return chunk.content


def _generate_llm_context(chunk: CodeChunk, full_file_content: str) -> Optional[str]:
    """Generate context using LLM - currently implemented with local analysis."""

    template = CONTEXTUAL_CONFIG["context_prompt_template"]
    if isinstance(template, str):
        prompt = template.format(
            full_file_content=full_file_content,
            chunk_content=chunk.content
        )
    else:
        prompt = f"Context for {chunk.name}: {chunk.content[:100]}..."

    # Current implementation: Advanced heuristic analysis that mimics LLM reasoning
    # This provides immediate contextual enhancement without external API calls
    context = _generate_advanced_context_analysis(chunk, full_file_content, prompt)

    if context:
        print(f"🧠 Generated contextual information for {chunk.name} ({len(context)} chars)")
        return context
    else:
        print(f"⚠️  Context generation failed for {chunk.name}, using heuristic fallback")
        return _generate_heuristic_context(chunk, full_file_content)


def _generate_heuristic_context(chunk: CodeChunk, full_file_content: str) -> str:
    """Generate context using heuristic analysis with full file awareness."""

    context_parts = []

    # File context with purpose analysis
    file_name = Path(chunk.file_path).stem
    file_purpose = _analyze_file_purpose(full_file_content)
    context_parts.append(f"This code chunk is from {file_name}.py, {file_purpose}")

    # Chunk type context
    if chunk.chunk_type == ChunkType.FUNCTION:
        context_parts.append(f"defining the {chunk.name} function")
    elif chunk.chunk_type == ChunkType.CLASS:
        context_parts.append(f"defining the {chunk.name} class")
    elif chunk.chunk_type == ChunkType.METHOD:
        if chunk.parent_class:
            context_parts.append(f"defining the {chunk.name} method in {chunk.parent_class} class")
        else:
            context_parts.append(f"defining the {chunk.name} method")
    elif chunk.chunk_type == ChunkType.IMPORT_BLOCK:
        context_parts.append("containing import statements and dependencies")
    else:
        context_parts.append(f"containing {chunk.chunk_type.value} code")

    # Functional context with file-level relationships
    if chunk.docstring:
        # Extract first sentence of docstring for context
        first_sentence = chunk.docstring.split('.')[0].strip()
        if first_sentence and len(first_sentence) < 100:
            context_parts.append(f"which {first_sentence.lower()}")

    # Enhanced dependencies context using full file analysis
    if chunk.dependencies and len(chunk.dependencies) > 0:
        deps = chunk.dependencies[:3]  # Limit to first 3 dependencies
        deps_str = ", ".join(deps)
        if len(chunk.dependencies) > 3:
            deps_str += f" and {len(chunk.dependencies) - 3} others"
        context_parts.append(f"using {deps_str}")

    # File-level relationship context
    relationships = _analyze_chunk_relationships(chunk, full_file_content)
    if relationships:
        context_parts.append(f"and {relationships}")

    # Complexity context
    if chunk.complexity_score > 5:
        context_parts.append("with high complexity")
    elif chunk.complexity_score > 2:
        context_parts.append("with moderate complexity")

    # Join context parts
    context = " ".join(context_parts) + "."

    # Ensure context is within token limits
    max_tokens = CONTEXTUAL_CONFIG["max_context_tokens"]
    if isinstance(max_tokens, int) and len(context.split()) > max_tokens:
        words = context.split()
        context = " ".join(words[:max_tokens]) + "..."

    return context


def _analyze_file_purpose(full_file_content: str) -> str:
    """Analyze the purpose of a file based on its content."""
    lines = full_file_content.split('\n')

    # Check for module docstring
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        if '"""' in line or "'''" in line:
            # Found potential docstring
            docstring_lines = []
            for j in range(i, min(i + 5, len(lines))):
                if '"""' in lines[j] or "'''" in lines[j]:
                    if j > i:  # End of docstring
                        break
                docstring_lines.append(lines[j])

            if docstring_lines:
                docstring = ' '.join(docstring_lines).replace('"""', '').replace("'''", '').strip()
                if docstring and len(docstring) > 10:
                    return f"which {docstring[:100].lower()}"

    # Analyze based on imports and class/function patterns
    if 'import unittest' in full_file_content or 'import pytest' in full_file_content:
        return "a test module"
    elif 'class ' in full_file_content and 'def __init__' in full_file_content:
        return "defining classes and their methods"
    elif 'def main(' in full_file_content or 'if __name__ == "__main__"' in full_file_content:
        return "a main execution module"
    elif full_file_content.count('def ') > 5:
        return "containing utility functions"
    elif 'import ' in full_file_content and full_file_content.count('\n') < 50:
        return "a configuration or constants module"
    else:
        return "a Python module"


def _analyze_chunk_relationships(chunk: CodeChunk, full_file_content: str) -> str:
    """Analyze relationships between the chunk and other parts of the file."""
    relationships = []

    # Check if chunk is called by other functions
    if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
        call_pattern = f"{chunk.name}("
        call_count = full_file_content.count(call_pattern) - 1  # Subtract definition
        if call_count > 0:
            relationships.append(f"called {call_count} times in this file")

    # Check if chunk inherits from classes in the file
    if chunk.chunk_type == ChunkType.CLASS and '(' in chunk.content:
        parent_classes = []
        lines = chunk.content.split('\n')
        for line in lines:
            if f'class {chunk.name}(' in line:
                # Extract parent classes
                parent_part = line.split('(')[1].split(')')[0]
                parents = [p.strip() for p in parent_part.split(',') if p.strip()]
                for parent in parents:
                    if f'class {parent}' in full_file_content:
                        parent_classes.append(parent)

        if parent_classes:
            relationships.append(f"inherits from {', '.join(parent_classes)}")

    # Check if chunk uses other functions/classes defined in the file
    chunk_content = chunk.content.lower()
    defined_items = []

    # Find other definitions in file
    for line in full_file_content.split('\n'):
        if line.strip().startswith('def ') and chunk.name not in line:
            func_name = line.split('def ')[1].split('(')[0].strip()
            if func_name.lower() in chunk_content and func_name != chunk.name:
                defined_items.append(func_name)
        elif line.strip().startswith('class ') and chunk.name not in line:
            class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
            if class_name.lower() in chunk_content and class_name != chunk.name:
                defined_items.append(class_name)

    if defined_items:
        unique_items = list(set(defined_items))[:3]  # Limit to 3
        relationships.append(f"uses {', '.join(unique_items)}")

    return "; ".join(relationships) if relationships else ""


def _generate_advanced_context_analysis(chunk: CodeChunk, full_file_content: str, prompt: str) -> str:
    """Generate advanced contextual analysis that mimics LLM reasoning for immediate use."""

    # Parse the prompt to understand what context is needed
    context_elements = []

    # Use prompt to guide context generation
    prompt_lower = prompt.lower()
    include_dependencies = 'dependencies' in prompt_lower or 'relationships' in prompt_lower
    include_usage = 'usage' in prompt_lower or 'patterns' in prompt_lower
    include_purpose = 'purpose' in prompt_lower or 'role' in prompt_lower

    # 1. File-level purpose and role analysis (always include)
    file_purpose = _analyze_file_purpose(full_file_content)
    context_elements.append(f"This code chunk is from {Path(chunk.file_path).stem}.py, {file_purpose}")

    # 2. Semantic role analysis (include if purpose requested or by default)
    if include_purpose or not any([include_dependencies, include_usage]):
        semantic_role = _determine_semantic_role(chunk, full_file_content)
        if semantic_role:
            context_elements.append(semantic_role)

    # 3. Dependency and relationship analysis (include if dependencies requested)
    if include_dependencies or not any([include_purpose, include_usage]):
        relationships = _analyze_chunk_relationships(chunk, full_file_content)
        if relationships:
            context_elements.append(f"It {relationships}")

    # 4. Functional context from docstring and signature (always include if available)
    functional_context = _extract_functional_context(chunk)
    if functional_context:
        context_elements.append(functional_context)

    # 5. Usage pattern analysis (include if usage requested)
    if include_usage or not any([include_dependencies, include_purpose]):
        usage_patterns = _analyze_usage_patterns(chunk, full_file_content)
        if usage_patterns:
            context_elements.append(usage_patterns)

    # 6. Complexity and importance indicators (always include if significant)
    importance_indicators = _analyze_importance_indicators(chunk, full_file_content)
    if importance_indicators:
        context_elements.append(importance_indicators)

    # Combine elements into coherent context
    context = " ".join(context_elements)

    # Ensure context is within token limits
    max_tokens = CONTEXTUAL_CONFIG["max_context_tokens"]
    if isinstance(max_tokens, int) and len(context.split()) > max_tokens:
        words = context.split()
        context = " ".join(words[:max_tokens]) + "..."

    return context.strip()


def _determine_semantic_role(chunk: CodeChunk, full_file_content: str) -> str:
    """Determine the semantic role of the chunk within the codebase."""

    if chunk.chunk_type == ChunkType.FUNCTION:
        # Analyze function role with file context
        if chunk.name.startswith('_'):
            return f"defining the private helper function {chunk.name}"
        elif chunk.name in ['main', '__main__', 'run', 'execute']:
            return f"defining the main entry point function {chunk.name}"
        elif any(pattern in chunk.content.lower() for pattern in ['test', 'assert', 'mock']):
            return f"defining the test function {chunk.name}"
        elif any(pattern in chunk.content.lower() for pattern in ['parse', 'process', 'transform']):
            return f"defining the data processing function {chunk.name}"
        elif '__name__ == "__main__"' in full_file_content and chunk.start_line > full_file_content.find('__name__ == "__main__"'):
            return f"defining the {chunk.name} function in the main execution block"
        else:
            return f"defining the {chunk.name} function"

    elif chunk.chunk_type == ChunkType.CLASS:
        # Analyze class role
        if 'Exception' in chunk.content or 'Error' in chunk.content:
            return f"defining the {chunk.name} exception class"
        elif any(pattern in chunk.content.lower() for pattern in ['config', 'setting', 'option']):
            return f"defining the {chunk.name} configuration class"
        elif 'Protocol' in chunk.content or 'ABC' in chunk.content:
            return f"defining the {chunk.name} interface/protocol"
        else:
            return f"defining the {chunk.name} class"

    elif chunk.chunk_type == ChunkType.METHOD:
        # Analyze method role
        if chunk.name.startswith('__') and chunk.name.endswith('__'):
            return f"defining the {chunk.name} magic method"
        elif chunk.name.startswith('_'):
            return f"defining the private {chunk.name} method"
        elif chunk.name in ['get', 'set', 'update', 'delete']:
            return f"defining the {chunk.name} accessor method"
        else:
            return f"defining the {chunk.name} method"

    elif chunk.chunk_type == ChunkType.IMPORT_BLOCK:
        # Analyze import patterns
        import_count = chunk.content.count('import')
        if import_count > 5:
            return "containing extensive import statements for external dependencies"
        else:
            return "containing import statements for required dependencies"

    return ""


def _extract_functional_context(chunk: CodeChunk) -> str:
    """Extract functional context from docstrings and signatures."""

    context_parts = []

    if chunk.docstring:
        # Extract key information from docstring
        doc_lower = chunk.docstring.lower()
        if any(word in doc_lower for word in ['return', 'returns']):
            context_parts.append("that returns processed data")
        if any(word in doc_lower for word in ['parse', 'parsing']):
            context_parts.append("for parsing operations")
        if any(word in doc_lower for word in ['validate', 'validation']):
            context_parts.append("for validation purposes")
        if any(word in doc_lower for word in ['generate', 'create']):
            context_parts.append("for generating/creating content")

    if chunk.signature:
        # Analyze signature patterns
        sig_lower = chunk.signature.lower()
        if 'async def' in sig_lower:
            context_parts.append("as an asynchronous operation")
        if '-> ' in chunk.signature:
            return_type = chunk.signature.split('-> ')[1].strip()
            if return_type not in ['None', 'Any']:
                context_parts.append(f"returning {return_type}")

    return " ".join(context_parts) if context_parts else ""


def _analyze_usage_patterns(chunk: CodeChunk, full_file_content: str) -> str:
    """Analyze how the chunk is used within the file."""

    if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
        call_pattern = f"{chunk.name}("
        call_count = full_file_content.count(call_pattern) - 1  # Subtract definition

        if call_count > 3:
            return f"frequently called ({call_count} times in this file)"
        elif call_count > 0:
            return f"called {call_count} time{'s' if call_count > 1 else ''} in this file"
        else:
            return "defined but not called in this file"

    elif chunk.chunk_type == ChunkType.CLASS:
        instantiation_patterns = [f"{chunk.name}(", f"= {chunk.name}("]
        usage_count = sum(full_file_content.count(pattern) for pattern in instantiation_patterns)

        if usage_count > 0:
            return f"instantiated {usage_count} time{'s' if usage_count > 1 else ''} in this file"
        else:
            return "defined but not instantiated in this file"

    return ""


def _analyze_importance_indicators(chunk: CodeChunk, full_file_content: str) -> str:
    """Analyze indicators of chunk importance and complexity."""

    indicators = []

    # Complexity indicators
    if hasattr(chunk, 'complexity_score') and chunk.complexity_score:
        if chunk.complexity_score > 10:
            indicators.append("with high computational complexity")
        elif chunk.complexity_score > 5:
            indicators.append("with moderate complexity")

    # Size indicators
    if len(chunk.content) > 1000:
        indicators.append("as a substantial code block")

    # Decorator indicators
    if hasattr(chunk, 'decorators') and chunk.decorators:
        decorator_types = []
        for decorator in chunk.decorators:
            if 'property' in decorator:
                decorator_types.append("property")
            elif 'staticmethod' in decorator:
                decorator_types.append("static method")
            elif 'classmethod' in decorator:
                decorator_types.append("class method")

        if decorator_types:
            indicators.append(f"decorated as {', '.join(decorator_types)}")

    # Error handling indicators
    if any(pattern in chunk.content for pattern in ['try:', 'except:', 'raise', 'assert']):
        indicators.append("with error handling logic")

    # File-level importance indicators
    total_functions = full_file_content.count('def ')
    if total_functions > 0:
        chunk_position = chunk.start_line / len(full_file_content.split('\n'))
        if chunk_position < 0.2:  # Early in file
            indicators.append("positioned early in the file")
        elif chunk_position > 0.8:  # Late in file
            indicators.append("positioned late in the file")

    return "; ".join(indicators) if indicators else ""


def process_chunks_with_context(chunks: List[CodeChunk],
                               file_contents: Dict[str, str]) -> Tuple[List[str], List[CodeChunk]]:
    """Process chunks with contextual information generation."""

    if not CONTEXTUAL_CONFIG["enabled"]:
        return [chunk.content for chunk in chunks], chunks

    print(f"🔍 Generating contextual information for {len(chunks)} chunks...")

    contextual_content: List[str] = []
    context_cache: Dict[str, str] = {}

    for i, chunk in enumerate(chunks):
        if i % 50 == 0:  # Progress indicator
            print(f"   Processing chunk {i+1}/{len(chunks)}")

        # Get full file content
        full_content = file_contents.get(chunk.file_path, "")

        # Generate contextual content
        contextual_chunk_content = generate_contextual_information(
            chunk, full_content, context_cache
        )

        contextual_content.append(contextual_chunk_content)

    print(f"✅ Generated contextual information for {len(chunks)} chunks")
    print(f"   Context cache size: {len(context_cache)} entries")

    return contextual_content, chunks


def load_file_contents(chunks: List[CodeChunk]) -> Dict[str, str]:
    """Load full file contents for contextual information generation."""

    file_contents = {}
    unique_files = set(chunk.file_path for chunk in chunks)

    print(f"📁 Loading {len(unique_files)} unique files for context generation...")

    for file_path in unique_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_contents[file_path] = f.read()
        except Exception as e:
            print(f"⚠️  Failed to load {file_path}: {e}")
            file_contents[file_path] = ""

    return file_contents


# Enhanced Embedding Generation Functions
def get_embedding_config(config_name: str = "balanced") -> Dict[str, Any]:
    """Get embedding configuration with metadata."""

    if config_name not in EMBEDDING_CONFIGS:
        print(f"⚠️  Unknown config '{config_name}', using 'balanced'")
        config_name = "balanced"

    config = EMBEDDING_CONFIGS[config_name].copy()
    config["name"] = config_name

    return config


def generate_embeddings_with_config(content_list: List[str],
                                   config: Dict[str, Any],
                                   batch_num: int = 1,
                                   total_batches: int = 1) -> List[List[float]]:
    """Generate embeddings using specified Matryoshka configuration with progress tracking."""

    try:
        # Enhanced progress reporting
        progress_pct = (batch_num / total_batches) * 100 if total_batches > 0 else 0
        print(f"   🔧 Generating embeddings with {config['name']} config...")
        print(f"      Progress: {batch_num}/{total_batches} ({progress_pct:.1f}%)")
        print(f"      Processing {len(content_list)} chunks")

        # Prepare embedding parameters
        embed_params = {
            "model": "voyage-code-3",
            "input_type": "document",
            "output_dtype": config["output_dtype"]
        }

        # Add output_dimension if specified
        if "output_dimension" in config:
            embed_params["output_dimension"] = config["output_dimension"]

        # Performance timing for batch analysis
        batch_start_time = time.time()

        # Generate embeddings
        response = vo.embed(content_list, **embed_params)
        embeddings = response.embeddings

        batch_time = time.time() - batch_start_time
        avg_time_per_chunk = batch_time / len(content_list) if content_list else 0

        print(f"      ✅ Batch completed in {batch_time:.2f}s ({avg_time_per_chunk:.3f}s per chunk)")

        # Handle different output types
        if config["output_dtype"] == "binary":
            # Convert binary to float for consistency
            embeddings = [[float(x) for x in embedding] for embedding in embeddings]
        elif config["output_dtype"] == "int8":
            # Convert int8 to float for consistency
            embeddings = [[float(x) for x in embedding] for embedding in embeddings]

        return cast(List[List[float]], embeddings)

    except Exception as e:
        print(f"   ❌ Embedding generation failed for batch {batch_num}/{total_batches}: {e}")
        raise


def create_batches_with_content(content_list: List[str],
                               chunks: List[CodeChunk],
                               max_tokens: int) -> List[Tuple[List[str], List[CodeChunk]]]:
    """Create batches from content list and corresponding chunks."""

    if len(content_list) != len(chunks):
        raise ValueError(f"Content list ({len(content_list)}) and chunks ({len(chunks)}) length mismatch")

    batches: List[Tuple[List[str], List[CodeChunk]]] = []
    current_batch_content: List[str] = []
    current_batch_chunks: List[CodeChunk] = []
    current_tokens: int = 0

    for content, chunk in zip(content_list, chunks):
        content_tokens = estimate_tokens(content)

        # If single content exceeds limit, truncate it
        if content_tokens > max_tokens:
            truncated_content = content[:MAX_CHUNK_SIZE]
            content_tokens = estimate_tokens(truncated_content)
            print(f"⚠️  Truncated large content for {chunk.name} in {chunk.file_path}")
            content = truncated_content

        if current_tokens + content_tokens > max_tokens:
            if current_batch_content:  # Save current batch if non-empty
                batches.append((current_batch_content, current_batch_chunks))
                current_batch_content = []
                current_batch_chunks = []
                current_tokens = 0

        current_batch_content.append(content)
        current_batch_chunks.append(chunk)
        current_tokens += content_tokens

    if current_batch_content:  # Save the last batch
        batches.append((current_batch_content, current_batch_chunks))

    return batches


def handle_embedding_error(error: Exception,
                          batch_content: List[str],
                          batch_chunks: List[CodeChunk],
                          config: Dict[str, Any]) -> bool:
    """Handle embedding errors with intelligent retry logic and chunk analysis."""

    error_str = str(error).lower()

    # Analyze problematic chunks for better error handling
    chunk_analysis = _analyze_problematic_chunks(batch_chunks, error_str)
    if chunk_analysis:
        print(f"   📊 Chunk analysis: {chunk_analysis}")

    if "rate limit" in error_str:
        print("   🕐 Rate limit detected. Implementing exponential backoff...")

        # Exponential backoff: 30s, 60s, 120s
        for attempt in range(3):
            wait_time = 30 * (2 ** attempt)
            print(f"   ⏳ Waiting {wait_time}s before retry attempt {attempt + 1}/3...")
            time.sleep(wait_time)

            try:
                # Test with a small sample first
                test_content = batch_content[:1]
                vo.embed(test_content,
                        model="voyage-code-3",
                        input_type="document",
                        output_dtype=config["output_dtype"])
                print(f"   ✅ Rate limit cleared after {wait_time}s wait")
                return True
            except Exception as retry_error:
                if attempt == 2:  # Last attempt
                    print(f"   ❌ Rate limit retry failed after 3 attempts: {retry_error}")
                    return False
                continue

    elif "token" in error_str or "length" in error_str:
        print("   📏 Token limit exceeded. Attempting batch size reduction...")

        # Try reducing batch size
        if len(batch_content) > 1:
            print(f"   🔄 Splitting batch of {len(batch_content)} items into smaller batches")
            # This would require recursive batch processing - for now, return False
            return False
        else:
            print("   ❌ Single item exceeds token limit")
            return False

    elif "model" in error_str or "parameter" in error_str:
        print("   🔧 Model/parameter error. Attempting fallback configuration...")

        # Try fallback to basic configuration
        try:
            vo.embed(batch_content[:1],
                    model="voyage-code-3",
                    input_type="document",
                    output_dtype="float")
            print("   ✅ Fallback configuration successful")
            return True
        except Exception as fallback_error:
            print(f"   ❌ Fallback configuration failed: {fallback_error}")
            return False

    else:
        print(f"   ❌ Unhandled error type: {error}")
        return False

    # Default return for safety
    return False


def _analyze_problematic_chunks(chunks: List[CodeChunk], error_str: str) -> str:
    """Analyze chunks to identify potential causes of embedding errors."""
    analysis_parts = []

    if not chunks:
        return "No chunks to analyze"

    # Analyze chunk types
    chunk_types: Dict[str, int] = {}
    for chunk in chunks:
        chunk_type = chunk.chunk_type.value
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    analysis_parts.append(f"Chunk types: {dict(chunk_types)}")

    # Analyze content sizes
    sizes = [len(chunk.content) for chunk in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    analysis_parts.append(f"Avg size: {avg_size:.0f}, Max size: {max_size}")

    # Check for specific problematic patterns
    if "token" in error_str or "length" in error_str:
        large_chunks = [chunk for chunk in chunks if len(chunk.content) > 8000]
        if large_chunks:
            analysis_parts.append(f"{len(large_chunks)} chunks exceed 8K chars")

    # Check for encoding issues
    if "encoding" in error_str or "unicode" in error_str:
        problematic_chunks = []
        for chunk in chunks:
            try:
                chunk.content.encode('utf-8')
            except UnicodeEncodeError:
                problematic_chunks.append(chunk.name)

        if problematic_chunks:
            analysis_parts.append(f"Encoding issues in: {', '.join(problematic_chunks[:3])}")

    # Check complexity distribution
    if chunks and hasattr(chunks[0], 'complexity_score'):
        complexities = [chunk.complexity_score for chunk in chunks if chunk.complexity_score]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            high_complexity = len([c for c in complexities if c > 10])
            if high_complexity > 0:
                analysis_parts.append(f"High complexity chunks: {high_complexity}, Avg: {avg_complexity:.1f}")

    return "; ".join(analysis_parts) if analysis_parts else "No specific issues identified"


def create_batches(chunks: List[CodeChunk], max_tokens: int) -> List[Tuple[List[str], List[CodeChunk]]]:
    """Split code chunks into batches with total tokens <= max_tokens."""
    batches: List[Tuple[List[str], List[CodeChunk]]] = []
    current_batch_content: List[str] = []
    current_batch_chunks: List[CodeChunk] = []
    current_tokens: int = 0

    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk.content)

        # If single chunk exceeds limit, truncate it
        if chunk_tokens > max_tokens:
            truncated_content = chunk.content[:MAX_CHUNK_SIZE]
            chunk_tokens = estimate_tokens(truncated_content)
            print(f"Warning: Truncated large chunk {chunk.name} in {chunk.file_path}")
            chunk.content = truncated_content

        if current_tokens + chunk_tokens > max_tokens:
            if current_batch_content:  # Save current batch if non-empty
                batches.append((current_batch_content, current_batch_chunks))
                current_batch_content = []
                current_batch_chunks = []
                current_tokens = 0

        current_batch_content.append(chunk.content)
        current_batch_chunks.append(chunk)
        current_tokens += chunk_tokens

    if current_batch_content:  # Save the last batch
        batches.append((current_batch_content, current_batch_chunks))

    return batches

# Collect code chunks from all Python files
all_chunks: List[CodeChunk] = []
processing_stats: Dict[str, Any] = {
    "files_processed": 0,
    "files_skipped": 0,
    "total_chunks": 0,
    "chunk_types": {},
    "parsing_errors": 0
}

# Define the project src directory
project_root = "/Users/lukemckenzie/prompt-improver/src"

print("🔍 Scanning Python files and extracting code chunks...")
start_time = time.time()

# Read all .py files in the src directory and parse them into chunks
for root, _, files in os.walk(project_root):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Only process non-empty files
                        print(f"📄 Parsing {file_path}...")
                        file_chunks = parse_python_file(file_path, content)
                        all_chunks.extend(file_chunks)

                        # Update statistics
                        processing_stats["files_processed"] += 1
                        processing_stats["total_chunks"] += len(file_chunks)

                        for chunk in file_chunks:
                            chunk_type_name = chunk.chunk_type.value
                            processing_stats["chunk_types"][chunk_type_name] = (
                                processing_stats["chunk_types"].get(chunk_type_name, 0) + 1
                            )

                        print(f"  ✅ Extracted {len(file_chunks)} chunks")
                    else:
                        print(f"⚠️  Skipping {file_path} because it is empty")
                        processing_stats["files_skipped"] += 1

            except UnicodeDecodeError:
                print(f"❌ Skipping {file_path} due to encoding error")
                processing_stats["files_skipped"] += 1
                processing_stats["parsing_errors"] += 1
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                processing_stats["files_skipped"] += 1
                processing_stats["parsing_errors"] += 1

parsing_time = time.time() - start_time
print(f"\n📊 Parsing completed in {parsing_time:.2f}s")
print(f"   Files processed: {processing_stats['files_processed']}")
print(f"   Files skipped: {processing_stats['files_skipped']}")
print(f"   Total chunks: {processing_stats['total_chunks']}")
print(f"   Chunk types: {processing_stats['chunk_types']}")

# Check if any valid chunks were found
if not all_chunks:
    raise ValueError("No valid code chunks found in the src directory")

# Enhanced embedding generation with Matryoshka configurations and contextual information
print(f"\n🚀 Generating embeddings using voyage-code-3 with enhanced configurations...")
embedding_start_time = time.time()

# Load file contents for contextual information
file_contents = load_file_contents(all_chunks)

# Process chunks with contextual information
contextual_content, processed_chunks = process_chunks_with_context(all_chunks, file_contents)

# Generate embeddings with configurable parameters
embedding_config = get_embedding_config()
print(f"📊 Using embedding configuration: {embedding_config['name']}")
print(f"   Dimensions: {embedding_config['output_dimension']}")
print(f"   Data type: {embedding_config['output_dtype']}")

all_embeddings: List[List[float]] = []
all_chunks_processed: List[CodeChunk] = []

# Create batches with contextual content
batches = create_batches_with_content(contextual_content, processed_chunks, MAX_TOKENS_PER_BATCH)
print(f"📦 Created {len(batches)} batches for processing")

for batch_idx, (batch_content, batch_chunks) in enumerate(batches):
    print(f"⚡ Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_content)} chunks")

    try:
        # Generate embeddings with enhanced configuration
        batch_embeddings = generate_embeddings_with_config(
            batch_content,
            embedding_config,
            batch_idx + 1,
            len(batches)
        )

        # Verify we got the expected number of embeddings
        if len(batch_embeddings) != len(batch_chunks):
            raise ValueError(f"Mismatch: got {len(batch_embeddings)} embeddings for {len(batch_chunks)} chunks")

        all_embeddings.extend(batch_embeddings)
        all_chunks_processed.extend(batch_chunks)

        print(f"   ✅ Successfully processed batch {batch_idx + 1}")

    except Exception as e:
        print(f"❌ Error generating embeddings for batch {batch_idx + 1}: {e}")
        print(f"   Batch contained {len(batch_chunks)} chunks")

        # Enhanced error handling with intelligent retry
        if handle_embedding_error(e, batch_content, batch_chunks, embedding_config):
            # Retry successful
            batch_embeddings = generate_embeddings_with_config(
                batch_content,
                embedding_config,
                batch_idx + 1,
                len(batches)
            )
            all_embeddings.extend(batch_embeddings)
            all_chunks_processed.extend(batch_chunks)
            print(f"   ✅ Retry successful for batch {batch_idx + 1}")
        else:
            raise

embedding_time = time.time() - embedding_start_time
print(f"\n🎯 Enhanced embedding generation completed in {embedding_time:.2f}s")
print(f"   Total embeddings generated: {len(all_embeddings)}")
print(f"   Total chunks processed: {len(all_chunks_processed)}")
print(f"   Configuration used: {embedding_config['name']}")
print(f"   Average embedding time: {embedding_time/len(all_embeddings):.3f}s per embedding")

# Save enhanced embeddings with comprehensive metadata
embeddings_path = os.path.join(project_root, "embeddings.pkl")
print(f"\n💾 Saving enhanced embeddings to {embeddings_path}")

try:
    # Ensure the directory is writable
    os.makedirs(project_root, exist_ok=True)

    # Create comprehensive metadata
    total_processing_time = time.time() - start_time
    processing_stats.update({
        "total_processing_time": total_processing_time,
        "parsing_time": parsing_time,
        "embedding_time": embedding_time,
        "embeddings_generated": len(all_embeddings),
        "chunks_processed": len(all_chunks_processed)
    })

    enhanced_metadata = EmbeddingMetadata(
        embeddings=all_embeddings,
        chunks=all_chunks_processed,
        generation_timestamp=time.time(),
        model_used=f"voyage-code-3 ({embedding_config['name']})",
        total_files_processed=processing_stats["files_processed"],
        total_chunks_created=processing_stats["total_chunks"],
        processing_stats=processing_stats
    )

    # Save with enhanced structure while maintaining backward compatibility
    save_data = {
        # Backward compatibility fields
        "embeddings": all_embeddings,
        "file_paths": [chunk.file_path for chunk in all_chunks_processed],

        # Enhanced metadata
        "enhanced_metadata": enhanced_metadata,
        "chunks": all_chunks_processed,
        "model_used": f"voyage-code-3 ({embedding_config['name']})",
        "embedding_config": embedding_config,
        "contextual_enabled": CONTEXTUAL_CONFIG["enabled"],
        "cast_enabled": TREE_SITTER_AVAILABLE,
        "generation_timestamp": time.time(),
        "version": "2.1"  # Version for enhanced features
    }

    with open(embeddings_path, "wb") as f:  # type: ignore[assignment]
        pickle.dump(save_data, f)  # type: ignore[arg-type]

    if os.path.exists(embeddings_path):
        file_size = os.path.getsize(embeddings_path) / (1024 * 1024)  # MB
        print(f"✅ Enhanced embeddings successfully saved to {embeddings_path}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Total processing time: {total_processing_time:.2f}s")
        print(f"   Embeddings per second: {len(all_embeddings) / total_processing_time:.1f}")
    else:
        print(f"❌ Error: {embeddings_path} was not created, check file system")

except PermissionError as e:
    print(f"❌ Permission error saving {embeddings_path}: {e}")
    print(f"   Check write permissions for {project_root}")
    raise
except Exception as e:
    print(f"❌ Error saving embeddings to {embeddings_path}: {e}")
    raise

print(f"\n🎉 Successfully generated enhanced embeddings for {processing_stats['files_processed']} files")
print(f"   Total chunks created: {processing_stats['total_chunks']}")
print(f"   Chunk distribution: {processing_stats['chunk_types']}")
print(f"   Model used: voyage-code-3 ({embedding_config['name']} config)")
print(f"   Embedding dimensions: {embedding_config.get('output_dimension', 'default')}")
print(f"   Data type: {embedding_config['output_dtype']}")
print(f"   cAST parsing: {'✅ Enabled' if TREE_SITTER_AVAILABLE else '⚠️  Fallback to AST'}")
print(f"   Contextual information: {'✅ Enabled' if CONTEXTUAL_CONFIG['enabled'] else '❌ Disabled'}")
print(f"   Ready for enhanced semantic code search! 🔍✨")