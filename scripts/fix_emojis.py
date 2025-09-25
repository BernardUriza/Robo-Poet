#!/usr/bin/env python3
"""Script to remove emojis from Python files"""

import re
import sys
from pathlib import Path

def remove_emojis(text):
    """Remove emojis and replace with ASCII equivalents"""
    # First handle unicode escape sequences
    text = text.replace('\\U0001f393', '[GRAD]')  # Graduation cap
    text = text.replace('\\U0001f4da', '[BOOKS]')  # Books

    replacements = {
        'WARNING:': 'WARNING:',
        '[FIRE]': '[FIRE]',
        '[TARGET]': '[TARGET]',
        '[OK]': '[OK]',
        '[OK]': '[OK]',
        '[SPARK]': '[SPARK]',
        '[AI]': '[AI]',
        '[LAUNCH]': '[LAUNCH]',
        '[FAST]': '[FAST]',
        '[GEM]': '[GEM]',
        '[FINISH]': '[FINISH]',
        '[TOOLS]': '[TOOLS]',
        '[GAME]': '[GAME]',
        '[SCIENCE]': '[SCIENCE]',
        '[ART]': '[ART]',
        '[CHART]': '[CHART]',
        '[GUITAR]': '[GUITAR]',
        '[DRAMA]': '[DRAMA]',
        '[TROPHY]': '[TROPHY]',
        '[STAR]': '[STAR]',
        '[X]': '[X]',
        '[BRAIN]': '[BRAIN]',
        '[GROWTH]': '[GROWTH]',
        '[STAR]': '[STAR]',
        '[SAVE]': '[SAVE]',
        '[FIX]': '[FIX]',
        '[BUILD]': '[BUILD]',
        '[CASTLE]': '[CASTLE]'
    }

    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)

    return result

def fix_file(filepath):
    """Fix emojis in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content = remove_emojis(content)

        if content != fixed_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in src directory"""
    src_dir = Path("src")
    fixed_count = 0

    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()