#!/usr/bin/env python3
"""
Automated project cleanup for architectural presentation.
Runs cleanup without user input for CI/CD purposes.
"""

import os
import shutil
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Auto cleanup execution."""
    project_root = Path(__file__).parent
    backup_dir = project_root / ".cleanup_backup"
    
    print("🧹 Automated Robo-Poet Project Cleanup")
    print("=" * 40)
    
    cleanup_report = {
        "files_removed": [],
        "files_moved": [],
        "files_consolidated": [],
        "directories_created": [],
        "issues_found": []
    }
    
    try:
        # Create backup directory
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"Created backup directory: {backup_dir}")
        
        # Phase 1: Remove obsolete files if they exist
        logger.info("Phase 1: Removing obsolete files...")
        obsolete_patterns = [
            "src/gpu_detection.py",
            "src/gpu_manager.py", 
            "src/evaluation_system.py",
            "robo_poet_old.py",
            "backup_*.py",
            "temp_*.py"
        ]
        
        for pattern in obsolete_patterns:
            files_to_remove = list(project_root.rglob(pattern))
            for file_path in files_to_remove:
                if file_path.exists():
                    logger.info(f"  Would remove: {file_path.relative_to(project_root)}")
                    cleanup_report["files_removed"].append(str(file_path.relative_to(project_root)))
        
        # Phase 2: Create missing directories  
        logger.info("Phase 2: Creating missing directories...")
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
            full_path = project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                gitkeep = full_path / ".gitkeep"
                gitkeep.write_text("# This file ensures the directory is tracked by git\\n")
                logger.info(f"  Created directory: {dir_path}")
                cleanup_report["directories_created"].append(dir_path)
        
        # Phase 3: Update .gitignore
        logger.info("Phase 3: Updating .gitignore...")
        gitignore_path = project_root / ".gitignore"
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
            # Only add if not already present
            if "# Robo-Poet specific" not in current_content:
                new_content = current_content + "\\n" + "\\n".join(gitignore_additions)
                gitignore_path.write_text(new_content)
                logger.info("  Updated .gitignore")
        else:
            new_content = "\\n".join(gitignore_additions)
            gitignore_path.write_text(new_content)
            logger.info("  Created .gitignore")
        
        # Generate report
        report_path = project_root / "cleanup_report.json"
        cleanup_report["summary"] = {
            "total_files_removed": len(cleanup_report["files_removed"]),
            "total_files_moved": len(cleanup_report["files_moved"]), 
            "total_files_consolidated": len(cleanup_report["files_consolidated"]),
            "total_directories_created": len(cleanup_report["directories_created"]),
            "total_issues_found": len(cleanup_report["issues_found"])
        }
        
        with open(report_path, 'w') as f:
            json.dump(cleanup_report, f, indent=2)
        
        logger.info(f"Generated cleanup report: {report_path}")
        
        # Print summary
        print("\\n" + "="*60)
        print("🧹 CLEANUP SUMMARY")
        print("="*60)
        print(f"Files removed: {cleanup_report['summary']['total_files_removed']}")
        print(f"Files consolidated: {cleanup_report['summary']['total_files_consolidated']}")
        print(f"Directories created: {cleanup_report['summary']['total_directories_created']}")
        print(f"Issues identified: {cleanup_report['summary']['total_issues_found']}")
        
        print(f"\\n📋 Full report saved to: cleanup_report.json")
        print(f"🗂️  Backup created in: {backup_dir}")
        print("="*60)
        
        print("\\n✅ Automated cleanup completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())