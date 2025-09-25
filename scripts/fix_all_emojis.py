#!/usr/bin/env python3
"""Comprehensive script to remove all emojis from Python files"""

import re
from pathlib import Path

def remove_all_emojis(text):
    """Remove all emojis and non-ASCII characters"""
    # Remove common emojis with specific replacements
    replacements = {
        '🎓': '[GRAD]',
        '📚': '[BOOKS]',
        '⚠️': 'WARNING:',
        '🔥': '[FIRE]',
        '🎯': '[TARGET]',
        '✓': '[OK]',
        '✅': '[OK]',
        '[SPARK]': '[SPARK]',
        '🤖': '[AI]',
        '🚀': '[LAUNCH]',
        '[FAST]': '[FAST]',
        '💎': '[GEM]',
        '🏁': '[FINISH]',
        '🛠️': '[TOOLS]',
        '🎮': '[GAME]',
        '🔬': '[SCIENCE]',
        '🎨': '[ART]',
        '📊': '[CHART]',
        '🎸': '[GUITAR]',
        '🎭': '[DRAMA]',
        '🏆': '[TROPHY]',
        '🌟': '[STAR]',
        '❌': '[X]',
        '🧠': '[BRAIN]',
        '📈': '[GROWTH]',
        '⭐': '[STAR]',
        '💾': '[SAVE]',
        '🔧': '[FIX]',
        '🏗️': '[BUILD]',
        '🏰': '[CASTLE]',
        '📝': '[DOC]',
        '💡': '[IDEA]',
        '🐛': '[BUG]',
        '🔍': '[SEARCH]',
        '📦': '[PACKAGE]',
        '📂': '[FOLDER]',
        '🔄': '[CYCLE]',
        '⏱️': '[TIME]',
        '🌈': '[RAINBOW]',
        '🔔': '[BELL]',
        '🔀': '[SHUFFLE]',
        '💻': '[COMPUTER]',
        '🖥️': '[DESKTOP]',
        '📱': '[PHONE]',
        '💪': '[STRONG]',
        '👍': '[THUMBUP]',
        '👎': '[THUMBDOWN]',
        '✔️': '[CHECK]',
        '❗': '[!]',
        '❓': '[?]',
        '➡️': '[ARROW]',
        '⬅️': '[ARROW]',
        '↩️': '[RETURN]',
        '🔗': '[LINK]'
    }

    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)

    # Remove any remaining emoji using regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

    result = emoji_pattern.sub('', result)

    return result

def fix_file(filepath):
    """Fix emojis in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        fixed_content = remove_all_emojis(content)

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
    """Fix all Python files in src directory and root"""
    fixed_count = 0

    # Fix files in src directory
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1

    # Fix files in root directory
    for py_file in Path(".").glob("*.py"):
        if py_file.name != "fix_all_emojis.py":
            if fix_file(py_file):
                fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()