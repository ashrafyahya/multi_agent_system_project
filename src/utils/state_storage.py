"""State storage utility for managing large state data externally.

This module provides functions to store large state data (reports, collected data)
externally to keep the state object size manageable. It uses file-based storage
with reference tracking and automatic cleanup of old data.

The storage system:
- Stores large data in JSON files with unique identifiers
- Returns references that can be stored in state instead of the full data
- Retrieves data by reference when needed
- Automatically cleans up data older than TTL
- Thread-safe operations for concurrent access
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)

# Thread lock for thread-safe operations
_storage_lock = Lock()

# Storage metadata file name
_METADATA_FILE = "_storage_metadata.json"


def store_large_data(
    data: Any,
    data_type: str = "generic",
    storage_dir: Path | None = None
) -> str:
    """Store large data externally and return a reference.
    
    This function stores data in a JSON file and returns a reference string
    that can be stored in the state instead of the full data. The reference
    format is: `__storage_ref:<data_type>:<hash>`.
    
    Args:
        data: Data to store (must be JSON-serializable)
        data_type: Type identifier for the data (e.g., "report", "collected_data")
            Used for organization and cleanup
        storage_dir: Optional storage directory. If None, uses config value
    
    Returns:
        Reference string in format `__storage_ref:<data_type>:<hash>`
        that can be stored in state instead of the full data
    
    Raises:
        ValueError: If data is not JSON-serializable
        OSError: If storage directory cannot be created or written to
    
    Example:
        ```python
        large_report = "Very long report text..." * 1000
        ref = store_large_data(large_report, data_type="report")
        # ref = "__storage_ref:report:abc123..."
        state["report"] = ref  # Store reference instead of full data
        ```
    """
    config = get_config()
    if storage_dir is None:
        storage_dir = config.state_storage_dir
    
    # Ensure storage directory exists
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate hash for data
    data_json = json.dumps(data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()[:16]
    
    # Create reference
    reference = f"__storage_ref:{data_type}:{data_hash}"
    
    # Check if data already exists (deduplication)
    file_path = storage_dir / f"{data_type}_{data_hash}.json"
    
    with _storage_lock:
        if file_path.exists():
            logger.debug(f"Data already stored at {file_path}, reusing reference")
            # Update metadata timestamp
            _update_metadata(storage_dir, reference, data_type)
            return reference
        
        # Store data
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Update metadata
            _update_metadata(storage_dir, reference, data_type)
            
            logger.info(
                f"Stored {data_type} data ({len(data_json)} bytes) "
                f"with reference {reference}"
            )
            
            return reference
            
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to store data: {e}")
            raise


def retrieve_large_data(
    reference: str,
    storage_dir: Path | None = None
) -> Any:
    """Retrieve large data by reference.
    
    This function retrieves data stored externally using the reference string.
    If the reference is not a storage reference (doesn't start with `__storage_ref:`),
    it returns the reference as-is (assuming it's already the data).
    
    Args:
        reference: Reference string in format `__storage_ref:<data_type>:<hash>`
            or the actual data if not a storage reference
        storage_dir: Optional storage directory. If None, uses config value
    
    Returns:
        Retrieved data (deserialized from JSON)
    
    Raises:
        ValueError: If reference format is invalid
        FileNotFoundError: If stored data file doesn't exist
        OSError: If file cannot be read
        json.JSONDecodeError: If stored data is not valid JSON
    
    Example:
        ```python
        ref = state.get("report")
        if ref and ref.startswith("__storage_ref:"):
            report = retrieve_large_data(ref)
        else:
            report = ref  # Already the actual data
        ```
    """
    # Check if it's a storage reference
    if not isinstance(reference, str) or not reference.startswith("__storage_ref:"):
        # Not a storage reference, return as-is
        return reference
    
    # Parse reference
    try:
        _, data_type, data_hash = reference.split(":", 2)
    except ValueError:
        raise ValueError(f"Invalid storage reference format: {reference}")
    
    config = get_config()
    if storage_dir is None:
        storage_dir = config.state_storage_dir
    
    # Construct file path
    file_path = storage_dir / f"{data_type}_{data_hash}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Stored data not found for reference {reference} at {file_path}"
        )
    
    # Retrieve data
    with _storage_lock:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Update metadata timestamp (mark as recently accessed)
            _update_metadata(storage_dir, reference, data_type)
            
            logger.debug(f"Retrieved {data_type} data from {file_path}")
            return data
            
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve data from {file_path}: {e}")
            raise


def cleanup_old_data(
    storage_dir: Path | None = None,
    ttl: int | None = None
) -> int:
    """Clean up old stored data based on TTL.
    
    This function removes stored data files that haven't been accessed
    within the TTL (time to live) period. It also removes orphaned files
    (files without metadata entries).
    
    Args:
        storage_dir: Optional storage directory. If None, uses config value
        ttl: Optional TTL in seconds. If None, uses config value
    
    Returns:
        Number of files cleaned up
    
    Example:
        ```python
        cleaned_count = cleanup_old_data()
        logger.info(f"Cleaned up {cleaned_count} old data files")
        ```
    """
    config = get_config()
    if storage_dir is None:
        storage_dir = config.state_storage_dir
    if ttl is None:
        ttl = config.state_storage_ttl
    
    if not storage_dir.exists():
        return 0
    
    metadata_file = storage_dir / _METADATA_FILE
    
    # Load metadata
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load metadata: {e}, starting fresh")
            metadata = {}
    
    current_time = time.time()
    cleaned_count = 0
    
    # Find files to clean up
    files_to_remove = []
    references_to_remove = []
    
    for reference, entry in metadata.items():
        last_accessed = entry.get("last_accessed", 0)
        age = current_time - last_accessed
        
        if age > ttl:
            # File is older than TTL, mark for removal
            try:
                _, data_type, data_hash = reference.split(":", 2)
                file_path = storage_dir / f"{data_type}_{data_hash}.json"
                if file_path.exists():
                    files_to_remove.append(file_path)
                references_to_remove.append(reference)
            except ValueError:
                # Invalid reference format, remove from metadata
                references_to_remove.append(reference)
    
    # Remove old files
    with _storage_lock:
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                cleaned_count += 1
                logger.debug(f"Removed old data file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        # Remove references from metadata
        for reference in references_to_remove:
            metadata.pop(reference, None)
        
        # Find orphaned files (files without metadata entries)
        if storage_dir.exists():
            for file_path in storage_dir.glob("*.json"):
                if file_path.name == _METADATA_FILE:
                    continue
                
                # Check if file has a corresponding metadata entry
                found = False
                for ref in metadata.keys():
                    try:
                        _, data_type, data_hash = ref.split(":", 2)
                        if file_path.name == f"{data_type}_{data_hash}.json":
                            found = True
                            break
                    except ValueError:
                        continue
                
                if not found:
                    # Orphaned file, remove it
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed orphaned file: {file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to remove orphaned file {file_path}: {e}")
        
        # Save updated metadata
        if metadata_file.exists() or metadata:
            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            except OSError as e:
                logger.warning(f"Failed to save metadata: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} old data files from {storage_dir}")
    
    return cleaned_count


def _update_metadata(
    storage_dir: Path,
    reference: str,
    data_type: str
) -> None:
    """Update storage metadata with access timestamp.
    
    Internal function to update metadata file with reference information
    and last accessed timestamp.
    
    Args:
        storage_dir: Storage directory
        reference: Storage reference string
        data_type: Type of data stored
    """
    metadata_file = storage_dir / _METADATA_FILE
    
    # Load existing metadata
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            metadata = {}
    
    # Update metadata
    current_time = time.time()
    metadata[reference] = {
        "data_type": data_type,
        "last_accessed": current_time,
        "created": metadata.get(reference, {}).get("created", current_time),
    }
    
    # Save metadata
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to update metadata: {e}")


def is_storage_reference(value: Any) -> bool:
    """Check if a value is a storage reference.
    
    Args:
        value: Value to check
    
    Returns:
        True if value is a storage reference string, False otherwise
    """
    return (
        isinstance(value, str)
        and value.startswith("__storage_ref:")
    )


def get_storage_stats(storage_dir: Path | None = None) -> dict[str, Any]:
    """Get statistics about stored data.
    
    Args:
        storage_dir: Optional storage directory. If None, uses config value
    
    Returns:
        Dictionary with statistics:
        - total_files: Number of stored data files
        - total_size: Total size of stored files in bytes
        - by_type: Dictionary mapping data types to counts
    """
    config = get_config()
    if storage_dir is None:
        storage_dir = config.state_storage_dir
    
    if not storage_dir.exists():
        return {
            "total_files": 0,
            "total_size": 0,
            "by_type": {},
        }
    
    metadata_file = storage_dir / _METADATA_FILE
    metadata = {}
    
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    
    total_files = 0
    total_size = 0
    by_type: dict[str, int] = {}
    
    for reference, entry in metadata.items():
        try:
            _, data_type, data_hash = reference.split(":", 2)
            file_path = storage_dir / f"{data_type}_{data_hash}.json"
            
            if file_path.exists():
                total_files += 1
                total_size += file_path.stat().st_size
                by_type[data_type] = by_type.get(data_type, 0) + 1
        except ValueError:
            continue
    
    return {
        "total_files": total_files,
        "total_size": total_size,
        "by_type": by_type,
    }

