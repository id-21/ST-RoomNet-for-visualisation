#!/bin/bash
#
# This script generates a digest of a code repository.
# 1. It creates a structure file (`repository_structure.md`) listing all .py and .md files.
# 2. It creates a zip archive (`markdown_files.zip`) of all markdown files.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
STRUCTURE_FILE="repository_structure.md"
ARCHIVE_FILE="markdown_files.zip"
# Directories/patterns to ignore in both tree and find commands.
IGNORE_PATTERNS=("venv" "__pycache__" ".git" "dist" "build" "*.egg-info" "node_modules")


# --- Step 1: Generate Repository Structure File ---
echo "INFO: Generating repository structure in '$STRUCTURE_FILE'..."

# Check if the 'tree' command is available.
if command -v tree &> /dev/null; then
    # 'tree' is installed, use it for a nice visual output.
    # Build the ignore string for the 'tree' command by joining the array with '|'.
    ignore_str=$(IFS="|"; echo "${IGNORE_PATTERNS[*]}")
    
    # -a: show all files (including dotfiles) that match the pattern
    # -P: filter by pattern (Python or Markdown files)
    # -I: ignore directories/patterns
    tree -a -P "*.py|*.md" -I "$ignore_str" > "$STRUCTURE_FILE"
    echo "SUCCESS: Created '$STRUCTURE_FILE' using 'tree'."
else
    # 'tree' is not installed, use 'find' as a fallback.
    echo "WARN: 'tree' command not found. Using 'find' for a simple file list."
    
    # Build the '-prune' arguments for the 'find' command.
    find_prune_args=()
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        find_prune_args+=(-o -path "./$pattern" -prune)
    done
    # Remove the initial spurious '-o'.
    unset 'find_prune_args[0]'

    # Find all files matching the patterns, excluding pruned paths, and sort the result.
    find . "${find_prune_args[@]}" -o -type f \( -name "*.py" -o -name "*.md" \) -print | sort > "$STRUCTURE_FILE"
    echo "SUCCESS: Created '$STRUCTURE_FILE' using 'find'. (Note: This is a flat list, not a tree)."
fi
echo ""


# --- Step 2: Archive All Markdown Files ---
echo "INFO: Archiving all Markdown files into '$ARCHIVE_FILE'..."

# Remove old archive if it exists to ensure a fresh start.
if [ -f "$ARCHIVE_FILE" ]; then
    rm "$ARCHIVE_FILE"
fi

# Build the '-prune' arguments for this 'find' command instance.
find_prune_args_zip=()
for pattern in "${IGNORE_PATTERNS[@]}"; do
    find_prune_args_zip+=(-o -path "./$pattern" -prune)
done
unset 'find_prune_args_zip[0]'

# Temporarily disable 'exit on error' to handle the 'zip' command's specific exit codes.
set +e
# Find all .md files (null-delimited for safety) and pipe them to zip.
find . "${find_prune_args_zip[@]}" -o -type f -name "*.md" -print0 | xargs -0 zip -q "$ARCHIVE_FILE"
ZIP_EXIT_CODE=$?
# Re-enable 'exit on error'.
set -e

# Check the exit code from the zip command.
if [ $ZIP_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Created '$ARCHIVE_FILE' with all found markdown files."
elif [ $ZIP_EXIT_CODE -eq 12 ]; then # Exit code 12 for zip is 'nothing to do'.
    echo "WARN: No markdown files were found. An empty archive '$ARCHIVE_FILE' has been created."
else
    echo "ERROR: Failed to create zip archive. Zip exited with code $ZIP_EXIT_CODE."
    exit 1
fi
echo ""

# --- Final Summary ---
echo "-------------------------------------"
echo "Repository digest complete!"
echo "-> Structure file: $STRUCTURE_FILE"
echo "-> Markdown archive: $ARCHIVE_FILE"
echo "-------------------------------------"
