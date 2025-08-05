#!/usr/bin/env python3
"""
Incremental Embedding Updater for Voyage AI Search System
Only processes new/changed files to keep embeddings up-to-date efficiently.
"""

import os
import sys
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass

import json

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    with open(config_path, "r") as f:
        return json.load(f)  # type: ignore[no-any-return]

# Load configuration
CONFIG = load_config()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", CONFIG["project_root"]))

# Add project src to path for imports
project_src_path = os.path.join(PROJECT_ROOT, 'src')
if project_src_path not in sys.path:
    sys.path.insert(0, project_src_path)

from .generate_embeddings import (
    parse_python_file,
    generate_contextualized_embeddings_with_config,
    get_embedding_config
)


@dataclass
class FileMetadata:
    """Metadata for tracking file changes."""
    file_path: str
    last_modified: float
    file_size: int
    content_hash: str
    chunk_count: int


class IncrementalEmbeddingUpdater:
    """
    Manages incremental updates to embeddings when files change.
    Only processes new/modified files to maintain efficiency.
    """
    
    def __init__(self, embeddings_file: Optional[str] = None, metadata_file: Optional[str] = None):
        """Initialize the incremental updater."""
        if embeddings_file is None:
            embeddings_file = os.path.join(os.path.dirname(__file__), "..", CONFIG["embeddings_file"])
        if metadata_file is None:
            metadata_file = os.path.join(os.path.dirname(__file__), "..", CONFIG["metadata_file"])
        self.embeddings_file = embeddings_file
        self.metadata_file = metadata_file
        self.file_metadata: Dict[str, FileMetadata] = {}
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load existing file metadata."""
        if Path(self.metadata_file).exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    self.file_metadata = pickle.load(f)
                print(f"📋 Loaded metadata for {len(self.file_metadata)} files")
            except Exception as e:
                print(f"⚠️  Failed to load metadata: {e}")
                self.file_metadata = {}
        else:
            print("📋 No existing metadata found - will create new")
    
    def _save_metadata(self) -> None:
        """Save file metadata."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.file_metadata, f)
            print(f"💾 Saved metadata for {len(self.file_metadata)} files")
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get content hash for a file."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _get_file_info(self, file_path: str) -> Tuple[float, int, str]:
        """Get file modification time, size, and content hash."""
        try:
            stat = os.stat(file_path)
            return stat.st_mtime, stat.st_size, self._get_file_hash(file_path)
        except Exception:
            return 0.0, 0, ""
    
    def detect_changes(self, directory: str = ".", exclude_patterns: Optional[List[str]] = None) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Detect new, modified, and deleted files.
        
        Returns:
            Tuple of (new_files, modified_files, deleted_files)
        """
        print("🔍 Detecting file changes...")
        
        # Scan for current files
        current_files = set()
        exclude_patterns = exclude_patterns or ['.git', '__pycache__', '.pytest_cache', 'node_modules']

        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    current_files.add(file_path)
        
        # Get existing files from metadata
        existing_files = set(self.file_metadata.keys())
        
        # Detect new and deleted files
        new_files = current_files - existing_files
        deleted_files = existing_files - current_files
        
        # Detect modified files
        modified_files = set()
        for file_path in current_files & existing_files:
            mtime, size, content_hash = self._get_file_info(file_path)
            metadata = self.file_metadata[file_path]
            
            if (mtime != metadata.last_modified or 
                size != metadata.file_size or 
                content_hash != metadata.content_hash):
                modified_files.add(file_path)
        
        print(f"📊 Change detection results:")
        print(f"   New files: {len(new_files)}")
        print(f"   Modified files: {len(modified_files)}")
        print(f"   Deleted files: {len(deleted_files)}")
        
        return new_files, modified_files, deleted_files
    
    def update_embeddings(self, directory: str = ".", force_full_rebuild: bool = False) -> bool:
        """
        Update embeddings incrementally or perform full rebuild.
        
        Args:
            directory: Directory to scan for changes
            force_full_rebuild: Force complete rebuild of all embeddings
            
        Returns:
            True if updates were made, False otherwise
        """
        start_time = time.time()
        
        if force_full_rebuild:
            print("🔄 Performing full rebuild of embeddings...")
            # Remove existing files to force full rebuild
            if Path(self.embeddings_file).exists():
                os.remove(self.embeddings_file)
            if Path(self.metadata_file).exists():
                os.remove(self.metadata_file)
            self.file_metadata = {}
            
            # Run full embedding generation
            from .generate_embeddings import main as generate_main
            generate_main()
            
            # Update metadata for all files
            self._update_metadata_for_all_files(directory)
            return True
        
        # Detect changes
        new_files, modified_files, deleted_files = self.detect_changes(directory)
        
        # Check if any updates needed
        files_to_process = new_files | modified_files
        if not files_to_process and not deleted_files:
            print("✅ No changes detected - embeddings are up to date")
            return False
        
        print(f"🔄 Processing {len(files_to_process)} changed files...")
        
        # Load existing embeddings
        if not Path(self.embeddings_file).exists():
            print("❌ No existing embeddings found - run full generation first")
            return False
        
        try:
            with open(self.embeddings_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            embeddings = existing_data['embeddings']
            chunks = existing_data['chunks']
            metadata = existing_data['metadata']
            binary_embeddings = existing_data.get('binary_embeddings', [])
            
        except Exception as e:
            print(f"❌ Failed to load existing embeddings: {e}")
            return False
        
        # Remove chunks for deleted/modified files
        files_to_remove = deleted_files | modified_files
        if files_to_remove:
            print(f"🗑️  Removing chunks for {len(files_to_remove)} files...")
            
            # Filter out chunks from removed/modified files
            new_chunks = []
            new_embeddings = []
            new_binary_embeddings = []
            
            for i, chunk in enumerate(chunks):
                if chunk.file_path not in files_to_remove:
                    new_chunks.append(chunk)
                    new_embeddings.append(embeddings[i])
                    if i < len(binary_embeddings):
                        new_binary_embeddings.append(binary_embeddings[i])
            
            chunks = new_chunks
            embeddings = new_embeddings
            binary_embeddings = new_binary_embeddings
        
        # Process new/modified files
        if files_to_process:
            print(f"⚡ Processing {len(files_to_process)} files...")
            
            # Process files and generate chunks
            new_chunks = []
            for file_path in files_to_process:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if content:  # Only process non-empty files
                        file_chunks = parse_python_file(file_path, content)
                        new_chunks.extend(file_chunks)
                        print(f"   📄 {file_path}: {len(file_chunks)} chunks")
                    else:
                        print(f"   ⚠️  Skipping {file_path} (empty file)")
                except Exception as e:
                    print(f"   ❌ Failed to process {file_path}: {e}")
            
            if new_chunks:
                # Group new chunks by file for contextualized embeddings
                chunks_by_file: Dict[str, List[Any]] = {}
                for chunk in new_chunks:
                    if chunk.file_path not in chunks_by_file:
                        chunks_by_file[chunk.file_path] = []
                    chunks_by_file[chunk.file_path].append(chunk)
                
                # Generate embeddings for new chunks
                config = get_embedding_config("contextualized_high")
                new_embeddings, new_binary_embeddings = generate_contextualized_embeddings_with_config(
                    chunks_by_file, config, 1, 1
                )
                # Handle the case where binary embeddings might be None
                if new_binary_embeddings is None:
                    new_binary_embeddings = []
                
                # Add new chunks and embeddings
                chunks.extend(new_chunks)
                embeddings.extend(new_embeddings)
                if new_binary_embeddings:
                    binary_embeddings.extend(new_binary_embeddings)
        
        # Update metadata
        for file_path in files_to_process:
            mtime, size, content_hash = self._get_file_info(file_path)
            chunk_count = sum(1 for chunk in chunks if chunk.file_path == file_path)
            
            self.file_metadata[file_path] = FileMetadata(
                file_path=file_path,
                last_modified=mtime,
                file_size=size,
                content_hash=content_hash,
                chunk_count=chunk_count
            )
        
        # Remove metadata for deleted files
        for file_path in deleted_files:
            if file_path in self.file_metadata:
                del self.file_metadata[file_path]
        
        # Save updated embeddings
        updated_data = {
            'embeddings': embeddings,
            'chunks': chunks,
            'metadata': metadata,
            'binary_embeddings': binary_embeddings
        }
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(updated_data, f)
        
        # Save metadata
        self._save_metadata()
        
        update_time = time.time() - start_time
        print(f"✅ Incremental update completed in {update_time:.1f}s")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Total files tracked: {len(self.file_metadata)}")
        
        return True
    
    def _update_metadata_for_all_files(self, directory: str) -> None:
        """Update metadata for all files after full rebuild."""
        print("📋 Updating metadata for all files...")

        # Load the newly generated embeddings to get chunk info
        try:
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
            chunks = data['chunks']
        except Exception:
            print("⚠️  Could not load chunks for metadata update")
            return

        # Count chunks per file from embeddings
        chunk_counts: Dict[str, int] = {}
        for chunk in chunks:
            chunk_counts[chunk.file_path] = chunk_counts.get(chunk.file_path, 0) + 1

        # Scan directory for all Python files to ensure complete metadata
        all_files = set()
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    all_files.add(os.path.abspath(file_path))

        # Update metadata for all files (both in embeddings and in directory)
        all_tracked_files = set(chunk_counts.keys()) | all_files
        for file_path in all_tracked_files:
            if os.path.exists(file_path):
                mtime, size, content_hash = self._get_file_info(file_path)
                self.file_metadata[file_path] = FileMetadata(
                    file_path=file_path,
                    last_modified=mtime,
                    file_size=size,
                    content_hash=content_hash,
                    chunk_count=chunk_counts.get(file_path, 0)
                )
        
        self._save_metadata()
    
    def status(self) -> Dict[str, Any]:
        """Get status of the embedding system."""
        embeddings_exists = Path(self.embeddings_file).exists()
        metadata_exists = Path(self.metadata_file).exists()
        
        status: Dict[str, Union[bool, int, float, str]] = {
            "embeddings_file_exists": embeddings_exists,
            "metadata_file_exists": metadata_exists,
            "tracked_files": len(self.file_metadata),
            "last_update": max([m.last_modified for m in self.file_metadata.values()]) if self.file_metadata else 0
        }
        
        if embeddings_exists:
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                status["total_chunks"] = len(data.get('chunks', []))
                status["total_embeddings"] = len(data.get('embeddings', []))
            except Exception:
                status["total_chunks"] = -1  # Use -1 to indicate unknown
                status["total_embeddings"] = -1  # Use -1 to indicate unknown
        
        return status


def main() -> None:
    """CLI interface for incremental embedding updates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental Embedding Updater")
    parser.add_argument("--check", action="store_true", help="Check for changes without updating")
    parser.add_argument("--update", action="store_true", help="Update embeddings incrementally")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--directory", default=".", help="Directory to scan (default: current)")
    
    args = parser.parse_args()
    
    updater = IncrementalEmbeddingUpdater()
    
    if args.status:
        status = updater.status()
        print("📊 Embedding System Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    elif args.check:
        new_files, modified_files, deleted_files = updater.detect_changes(args.directory)
        total_changes = len(new_files) + len(modified_files) + len(deleted_files)
        
        if total_changes == 0:
            print("✅ No changes detected")
        else:
            print(f"📋 Changes detected: {total_changes} files need updates")
            if new_files:
                print(f"   New: {list(new_files)[:5]}{'...' if len(new_files) > 5 else ''}")
            if modified_files:
                print(f"   Modified: {list(modified_files)[:5]}{'...' if len(modified_files) > 5 else ''}")
            if deleted_files:
                print(f"   Deleted: {list(deleted_files)[:5]}{'...' if len(deleted_files) > 5 else ''}")
    
    elif args.rebuild:
        updater.update_embeddings(args.directory, force_full_rebuild=True)
    
    elif args.update:
        updated = updater.update_embeddings(args.directory)
        if not updated:
            print("ℹ️  No updates needed")
    
    else:
        print("Use --check, --update, --rebuild, or --status")


if __name__ == "__main__":
    main()
