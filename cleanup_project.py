#!/usr/bin/env python3
"""
Project cleanup script for Robo-Poet.

Removes obsolete files, consolidates duplicated functionality,
and organizes the codebase according to peer review recommendations.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectCleanup:
    """Automated project cleanup based on peer review."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / ".cleanup_backup"
        self.cleanup_report = {
            "files_removed": [],
            "files_moved": [],
            "files_consolidated": [],
            "directories_created": [],
            "issues_found": []
        }
    
    def run_cleanup(self):
        """Execute complete cleanup process."""
        logger.info("🧹 Starting Robo-Poet project cleanup...")
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        logger.info(f"Created backup directory: {self.backup_dir}")
        
        try:
            # Phase 1: Remove obsolete files
            self._remove_obsolete_files()
            
            # Phase 2: Consolidate GPU management
            self._consolidate_gpu_modules()
            
            # Phase 3: Organize test files
            self._organize_tests()
            
            # Phase 4: Clean up imports and dependencies
            self._clean_imports()
            
            # Phase 5: Create missing directories
            self._create_missing_directories()
            
            # Phase 6: Update project structure
            self._update_project_structure()
            
            # Generate report
            self._generate_report()
            
            logger.info("✅ Project cleanup completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            self._restore_backup()
            raise
    
    def _remove_obsolete_files(self):
        """Remove obsolete and duplicate files."""
        logger.info("Phase 1: Removing obsolete files...")
        
        # Files to remove (peer review identified duplicates/obsoletes)
        obsolete_files = [
            "src/gpu_detection.py",  # Replaced by gpu_optimization/
            "src/gpu_manager.py",    # Replaced by gpu_optimization/
            "src/evaluation_system.py",  # If exists, replaced by evaluation/
            "robo_poet_old.py",     # If exists
            "backup_*.py",          # Any backup files
            "temp_*.py",            # Temporary files
            "*.pyc",                # Compiled Python files
            "__pycache__/",         # Python cache directories
        ]
        
        for pattern in obsolete_files:
            files_to_remove = list(self.project_root.rglob(pattern))
            
            for file_path in files_to_remove:
                if file_path.exists():
                    # Backup before removing
                    self._backup_file(file_path)
                    
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                        logger.info(f"  Removed directory: {file_path.relative_to(self.project_root)}")
                    else:
                        file_path.unlink()
                        logger.info(f"  Removed file: {file_path.relative_to(self.project_root)}")
                    
                    self.cleanup_report["files_removed"].append(str(file_path.relative_to(self.project_root)))
        
        # Remove empty directories
        self._remove_empty_directories()
    
    def _consolidate_gpu_modules(self):
        """Consolidate GPU management modules."""
        logger.info("Phase 2: Consolidating GPU modules...")
        
        # Check if old GPU files still exist and need references updated
        old_gpu_files = [
            self.project_root / "src/gpu_detection.py",
            self.project_root / "src/gpu_manager.py"
        ]
        
        files_with_gpu_imports = [
            "src/interface/menu_system.py",
            "src/logger.py", 
            "src/orchestrator.py",
            "robo_poet.py"
        ]
        
        for file_path_str in files_with_gpu_imports:
            file_path = self.project_root / file_path_str
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    original_content = content
                    
                    # Replace old imports with new ones
                    replacements = {
                        "from gpu_detection import": "from src.gpu_optimization.mixed_precision import",
                        "from gpu_manager import": "from src.gpu_optimization.memory_manager import",
                        "import gpu_detection": "from src.gpu_optimization import mixed_precision",
                        "import gpu_manager": "from src.gpu_optimization import memory_manager",
                        "GPUDetector": "MixedPrecisionManager",
                        "GPUManager": "GPUMemoryManager"
                    }
                    
                    for old, new in replacements.items():
                        if old in content:
                            content = content.replace(old, new)
                            logger.info(f"    Updated import in {file_path.name}: {old} → {new}")
                    
                    # Only write if changes were made
                    if content != original_content:
                        # Backup original
                        self._backup_file(file_path)
                        file_path.write_text(content, encoding='utf-8')
                        self.cleanup_report["files_consolidated"].append(file_path_str)
                        
                except Exception as e:
                    logger.warning(f"    Could not update {file_path}: {e}")
                    self.cleanup_report["issues_found"].append(f"Import update failed: {file_path} - {e}")
    
    def _organize_tests(self):
        """Organize test files into proper structure."""
        logger.info("Phase 3: Organizing test files...")
        
        # Ensure tests directory exists
        tests_dir = self.project_root / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Test files should already be moved, but verify structure
        test_categories = {
            "unit": ["test_models.py", "test_config.py", "test_tokenizer.py"],
            "integration": ["test_training_pipeline.py", "test_generation_pipeline.py"],
            "system": ["test_gpu_optimization.py", "test_generation_system.py", "test_data_pipeline.py"]
        }
        
        for category, test_files in test_categories.items():
            category_dir = tests_dir / category
            category_dir.mkdir(exist_ok=True)
            self.cleanup_report["directories_created"].append(str(category_dir.relative_to(self.project_root)))
            
            # Create __init__.py files
            init_file = category_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""Test category: {category}"""\\n')
        
        # Create conftest.py for shared test fixtures
        conftest_path = tests_dir / "conftest.py"
        if not conftest_path.exists():
            conftest_content = '''"""
Shared test configuration and fixtures for Robo-Poet test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import tensorflow as tf


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture 
def mock_gpu():
    """Mock GPU for testing without hardware dependency."""
    with tf.device('/CPU:0'):
        yield Mock()


@pytest.fixture
def sample_text_corpus(temp_dir):
    """Create a sample text corpus for testing."""
    corpus_file = temp_dir / "sample_corpus.txt"
    sample_text = """
    This is a sample text corpus for testing.
    It contains multiple sentences and paragraphs.
    The text is designed to test tokenization and processing.
    """
    corpus_file.write_text(sample_text)
    return corpus_file


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Disable GPU for most tests
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    yield
    # Cleanup after test
    tf.keras.backend.clear_session()
'''
            conftest_path.write_text(conftest_content)
            logger.info("  Created conftest.py with shared fixtures")
    
    def _clean_imports(self):
        """Clean up imports and remove unused dependencies."""
        logger.info("Phase 4: Cleaning imports...")
        
        # This would be more complex in a real implementation
        # For now, just report files that might need attention
        python_files = list(self.project_root.rglob("*.py"))
        
        problematic_imports = [
            "from config import",  # Should use unified_config
            "import config",
            "from data_processor import",  # Should use data pipeline
            "import data_processor"
        ]
        
        for py_file in python_files:
            if "test" in str(py_file) or ".cleanup_backup" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for problematic in problematic_imports:
                    if problematic in content:
                        self.cleanup_report["issues_found"].append(
                            f"Problematic import in {py_file.relative_to(self.project_root)}: {problematic}"
                        )
                        
            except Exception as e:
                logger.debug(f"Could not analyze {py_file}: {e}")
    
    def _create_missing_directories(self):
        """Create missing directories for proper organization."""
        logger.info("Phase 5: Creating missing directories...")
        
        directories_to_create = [
            "config",
            "docs",
            "scripts", 
            "examples",
            "notebooks",
            "data/raw",
            "data/processed", 
            "models/checkpoints",
            "models/exports",
            "logs",
            "output"
        ]
        
        for dir_path in directories_to_create:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                
                # Create .gitkeep files for empty directories
                gitkeep = full_path / ".gitkeep"
                gitkeep.write_text("# This file ensures the directory is tracked by git\\n")
                
                logger.info(f"  Created directory: {dir_path}")
                self.cleanup_report["directories_created"].append(dir_path)
    
    def _update_project_structure(self):
        """Update project structure files."""
        logger.info("Phase 6: Updating project structure...")
        
        # Update .gitignore
        gitignore_path = self.project_root / ".gitignore"
        gitignore_additions = [
            "",
            "# Robo-Poet specific",
            "*.pkl",
            "*.h5",
            "*.model",
            "models/checkpoints/*",
            "!models/checkpoints/.gitkeep",
            "logs/*",
            "!logs/.gitkeep", 
            "output/*",
            "!output/.gitkeep",
            "data/raw/*",
            "!data/raw/.gitkeep",
            "data/processed/*",
            "!data/processed/.gitkeep",
            "",
            "# Cleanup backups",
            ".cleanup_backup/",
            "",
            "# IDE specific",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "",
            "# System files", 
            ".DS_Store",
            "Thumbs.db"
        ]
        
        if gitignore_path.exists():
            current_content = gitignore_path.read_text()
            new_content = current_content + "\\n" + "\\n".join(gitignore_additions)
        else:
            new_content = "\\n".join(gitignore_additions)
        
        gitignore_path.write_text(new_content)
        logger.info("  Updated .gitignore")
        
        # Create requirements.txt if not exists
        requirements_path = self.project_root / "requirements.txt"
        if not requirements_path.exists():
            requirements_content = """# Robo-Poet Dependencies
tensorflow>=2.20.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.7
rouge-score>=0.0.4
sacrebleu>=2.0.0
pydantic>=1.8.0
sqlalchemy>=1.4.0
typer>=0.4.0
rich>=10.0.0
fastapi>=0.70.0
uvicorn>=0.15.0

# Development dependencies
pytest>=6.0.0
pytest-cov>=3.0.0
black>=21.0.0
isort>=5.0.0
mypy>=0.900
flake8>=4.0.0

# Optional GPU dependencies
# pynvml>=11.0.0  # For NVIDIA GPU monitoring
"""
            requirements_path.write_text(requirements_content)
            logger.info("  Created requirements.txt")
    
    def _backup_file(self, file_path: Path):
        """Backup a file before modifying/removing it."""
        if not file_path.exists():
            return
            
        try:
            relative_path = file_path.relative_to(self.project_root)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.is_dir():
                shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
            else:
                shutil.copy2(file_path, backup_path)
                
        except Exception as e:
            logger.warning(f"Could not backup {file_path}: {e}")
    
    def _remove_empty_directories(self):
        """Remove empty directories after cleanup."""
        for dir_path in self.project_root.rglob("*"):
            if (dir_path.is_dir() and 
                dir_path != self.backup_dir and
                not any(dir_path.iterdir()) and
                dir_path.name != ".git"):
                try:
                    dir_path.rmdir()
                    logger.info(f"  Removed empty directory: {dir_path.relative_to(self.project_root)}")
                except OSError:
                    pass  # Directory not empty or permission issue
    
    def _restore_backup(self):
        """Restore from backup in case of failure."""
        logger.info("Restoring from backup due to cleanup failure...")
        
        if not self.backup_dir.exists():
            return
        
        try:
            for backup_file in self.backup_dir.rglob("*"):
                if backup_file.is_file():
                    relative_path = backup_file.relative_to(self.backup_dir)
                    target_path = self.project_root / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_path)
            
            logger.info("Backup restoration completed")
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
    
    def _generate_report(self):
        """Generate cleanup report."""
        report_path = self.project_root / "cleanup_report.json"
        
        # Add summary statistics
        self.cleanup_report["summary"] = {
            "total_files_removed": len(self.cleanup_report["files_removed"]),
            "total_files_moved": len(self.cleanup_report["files_moved"]),
            "total_files_consolidated": len(self.cleanup_report["files_consolidated"]),
            "total_directories_created": len(self.cleanup_report["directories_created"]),
            "total_issues_found": len(self.cleanup_report["issues_found"])
        }
        
        with open(report_path, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        logger.info(f"Generated cleanup report: {report_path}")
        
        # Print summary
        print("\\n" + "="*60)
        print("🧹 CLEANUP SUMMARY")
        print("="*60)
        print(f"Files removed: {self.cleanup_report['summary']['total_files_removed']}")
        print(f"Files consolidated: {self.cleanup_report['summary']['total_files_consolidated']}")
        print(f"Directories created: {self.cleanup_report['summary']['total_directories_created']}")
        print(f"Issues identified: {self.cleanup_report['summary']['total_issues_found']}")
        
        if self.cleanup_report["issues_found"]:
            print("\\n⚠️  Issues requiring manual attention:")
            for issue in self.cleanup_report["issues_found"]:
                print(f"  - {issue}")
        
        print(f"\\n📋 Full report saved to: cleanup_report.json")
        print(f"🗂️  Backup created in: {self.backup_dir}")
        print("="*60)


def main():
    """Main cleanup execution."""
    project_root = Path(__file__).parent
    
    print("🧹 Robo-Poet Project Cleanup")
    print("=" * 40)
    print("This script will clean up the project based on peer review recommendations.")
    print("A backup will be created before making any changes.")
    print("")
    
    # Confirm with user
    response = input("Continue with cleanup? (y/N): ").lower().strip()
    if response not in ('y', 'yes'):
        print("Cleanup cancelled.")
        return
    
    try:
        cleanup = ProjectCleanup(project_root)
        cleanup.run_cleanup()
        
        print("\\n✅ Project cleanup completed successfully!")
        print("\\nNext steps:")
        print("1. Review cleanup_report.json for any issues")
        print("2. Run tests to ensure everything works: python -m pytest tests/")
        print("3. Update imports in files flagged in the report")
        print("4. Commit changes: git add . && git commit -m 'Clean up project structure'")
        
    except Exception as e:
        print(f"\\n❌ Cleanup failed: {e}")
        print("Check the logs for more details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())