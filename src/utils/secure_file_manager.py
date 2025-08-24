"""
Secure File Manager

Enhanced version of file_manager.py with security fixes and best practices.
Addresses path traversal vulnerabilities and improves input validation.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class SecureFileConfig:
    """Configuration for secure file operations."""
    allowed_directories: List[Path]
    max_file_size_mb: int = 100
    allowed_extensions: Set[str] = None
    enable_integrity_checks: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'.txt', '.md', '.csv', '.json'}
        
        # Convert to resolved paths
        self.allowed_directories = [Path(d).resolve() for d in self.allowed_directories]


class SecureFileManager:
    """
    Secure file manager with path validation and integrity checks.
    
    Addresses security vulnerabilities:
    - Path traversal prevention
    - Input validation
    - File integrity verification
    - Secure error handling
    """
    
    def __init__(self, config: Optional[SecureFileConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            # Default secure configuration
            self.config = SecureFileConfig(
                allowed_directories=[
                    Path('./corpus'),
                    Path('./data'),
                    Path('./models'), 
                    Path('./output'),
                    Path('.')  # Current directory only
                ],
                max_file_size_mb=100
            )
        else:
            self.config = config
    
    def validate_path(self, filepath: str) -> Tuple[bool, str, Optional[Path]]:
        """
        Validate file path against security policies.
        
        Args:
            filepath: File path to validate
            
        Returns:
            Tuple of (is_valid, error_message, resolved_path)
        """
        try:
            # Convert to Path and resolve (follows symlinks, removes ..)
            path = Path(filepath).resolve()
            
            # Check if path is within allowed directories
            is_allowed = any(
                str(path).startswith(str(allowed_dir)) 
                for allowed_dir in self.config.allowed_directories
            )
            
            if not is_allowed:
                return False, "File path not within allowed directories", None
            
            # Check file extension
            if path.suffix.lower() not in self.config.allowed_extensions:
                return False, f"File extension {path.suffix} not allowed", None
            
            # Additional security checks
            path_str = str(path)
            
            # Check for suspicious patterns
            suspicious_patterns = ['../', '..\\\\', '%2e%2e', '0x2e0x2e']
            if any(pattern in path_str.lower() for pattern in suspicious_patterns):
                return False, "Suspicious path pattern detected", None
            
            return True, "Path validated", path
            
        except (ValueError, OSError) as e:
            self.logger.warning(f"Path validation error: {e}")
            return False, "Invalid path format", None
    
    def validate_text_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Securely validate text file for training.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # First validate the path
            is_valid, error_msg, path = self.validate_path(filepath)
            if not is_valid:
                return False, error_msg
            
            # Check if file exists
            if not path.exists():
                return False, "File not found"
            
            if not path.is_file():
                return False, "Path is not a file"
            
            # Check file size limits
            file_size = path.stat().st_size
            max_size = self.config.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size:
                return False, f"File too large: {file_size / (1024*1024):.1f} MB (max: {self.config.max_file_size_mb} MB)"
            
            if file_size < 1000:
                return False, f"File too small: {file_size} bytes (minimum: 1000 bytes)"
            
            # Validate file content and encoding
            try:
                with open(path, 'r', encoding='utf-8', errors='strict') as f:
                    # Read sample for validation
                    sample = f.read(2000)  # Read more for better validation
                    
                    if len(sample.strip()) == 0:
                        return False, "File is empty or contains only whitespace"
                    
                    # Basic content validation
                    if not self._validate_text_content(sample):
                        return False, "File contains suspicious or invalid content"
                        
            except UnicodeDecodeError as e:
                return False, f"File encoding error: must be UTF-8"
            
            # Generate file integrity info if enabled
            if self.config.enable_integrity_checks:
                file_hash = self._calculate_file_hash(path)
                self.logger.info(f"File validated: {path} (SHA256: {file_hash[:16]}...)")
            
            return True, f"File validated: {file_size / 1024:.1f} KB"
            
        except PermissionError:
            return False, "Permission denied accessing file"
        except Exception as e:
            self.logger.error(f"Unexpected error validating file: {e}")
            return False, "File validation failed"
    
    def secure_read_text_file(self, filepath: str) -> Tuple[bool, str, Optional[str]]:
        """
        Securely read text file with validation.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Tuple of (success, message, content)
        """
        # Validate file first
        is_valid, validation_msg = self.validate_text_file(filepath)
        if not is_valid:
            return False, validation_msg, None
        
        try:
            # Re-validate path (defense in depth)
            is_valid, error_msg, path = self.validate_path(filepath)
            if not is_valid:
                return False, error_msg, None
            
            # Read file securely
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                content = f.read()
            
            # Final content validation
            if not self._validate_text_content(content):
                return False, "File content validation failed", None
            
            return True, f"Successfully read {len(content)} characters", content
            
        except Exception as e:
            self.logger.error(f"Error reading file securely: {e}")
            return False, "Failed to read file", None
    
    def find_text_files_secure(self, search_paths: Optional[List[str]] = None) -> List[str]:
        """
        Securely find text files within allowed directories.
        
        Args:
            search_paths: Optional list of paths to search
            
        Returns:
            List of validated file paths
        """
        if search_paths is None:
            search_paths = [str(d) for d in self.config.allowed_directories]
        
        text_files = []
        
        for search_path_str in search_paths:
            # Validate search path
            is_valid, _, search_path = self.validate_path(search_path_str)
            if not is_valid:
                self.logger.warning(f"Skipping invalid search path: {search_path_str}")
                continue
            
            if not search_path.exists() or not search_path.is_dir():
                continue
            
            # Find files with allowed extensions
            for extension in self.config.allowed_extensions:
                try:
                    pattern = f"*{extension}"
                    found_files = search_path.glob(pattern)
                    
                    for file_path in found_files:
                        # Double-check each file
                        is_valid, _ = self.validate_text_file(str(file_path))
                        if is_valid:
                            text_files.append(str(file_path))
                            
                except Exception as e:
                    self.logger.warning(f"Error searching in {search_path}: {e}")
        
        return sorted(text_files)
    
    def _validate_text_content(self, content: str) -> bool:
        """
        Validate text content for security and quality.
        
        Args:
            content: Text content to validate
            
        Returns:
            True if content is safe and valid
        """
        # Check for minimum content
        if len(content.strip()) < 10:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '<?php',  # PHP code
            '<script',  # JavaScript
            '#!/bin/',  # Shell scripts
            'import os',  # Python system imports
            'eval(',  # Code evaluation
            'exec(',  # Code execution
            '${',  # Template injection
        ]
        
        content_lower = content.lower()
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                self.logger.warning(f"Suspicious pattern detected: {pattern}")
                return False
        
        # Check for excessive special characters (possible binary data)
        special_char_ratio = sum(1 for c in content if not c.isprintable() and c not in '\\n\\r\\t') / len(content)
        if special_char_ratio > 0.1:  # More than 10% non-printable
            self.logger.warning(f"High non-printable character ratio: {special_char_ratio}")
            return False
        
        return True
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def verify_file_integrity(self, filepath: str, expected_hash: str) -> bool:
        """
        Verify file integrity using SHA256 hash.
        
        Args:
            filepath: Path to file
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if file integrity is verified
        """
        is_valid, _, path = self.validate_path(filepath)
        if not is_valid or not path.exists():
            return False
        
        actual_hash = self._calculate_file_hash(path)
        return actual_hash.lower() == expected_hash.lower()


# Secure singleton instance
_secure_file_manager_instance = None


def get_secure_file_manager() -> SecureFileManager:
    """Get singleton secure file manager instance."""
    global _secure_file_manager_instance
    if _secure_file_manager_instance is None:
        _secure_file_manager_instance = SecureFileManager()
    return _secure_file_manager_instance