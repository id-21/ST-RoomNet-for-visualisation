#!/bin/bash
#
# This script generates a digest of a code repository.
# 1. It creates a structure file (`repository_structure.md`) listing all .py and .md files.
# 2. It finds all markdown files, prepends their original path to the top of each,
#    and creates a zip archive (`markdown_files.zip`) of the modified files.

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
    ignore_str=$(IFS="|"; echo "${IGNORE_PATTERNS[*]}")
    tree -a -P "*.py|*.md|*.MD" -I "$ignore_str" > "$STRUCTURE_FILE"
    echo "SUCCESS: Created '$STRUCTURE_FILE' using 'tree'."
else
    # 'tree' is not installed, use 'find' as a fallback.
    echo "WARN: 'tree' command not found. Using 'find' for a simple file list."
    
    prune_paths=()
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        prune_paths+=(-o -path "./$pattern")
    done
    unset 'prune_paths[0]' # remove the first -o

    find . \( "${prune_paths[@]}" \) -prune -o -type f \( -name "*.py" -o -iname "*.md" \) -print | sort > "$STRUCTURE_FILE"
    echo "SUCCESS: Created '$STRUCTURE_FILE' using 'find'. (Note: This is a flat list, not a tree)."
fi
echo ""


# --- Step 2: Archive Markdown Files with Prepended Paths ---
echo "INFO: Preparing to archive all Markdown files into '$ARCHIVE_FILE'..."

# Remove old archive if it exists to ensure a fresh start.
if [ -f "$ARCHIVE_FILE" ]; then
    rm "$ARCHIVE_FILE"
fi

# Create a temporary directory for modified markdown files.
TEMP_DIR=$(mktemp -d)
# Ensure the temporary directory is cleaned up on script exit, error, or interrupt.
trap 'rm -rf "$TEMP_DIR"' EXIT

# --- FIX: Robust 'find' command logic ---
# Build a robust list of paths to prune.
prune_paths_zip=()
for pattern in "${IGNORE_PATTERNS[@]}"; do
    # For each pattern, add '-o -path ./pattern' to the array.
    prune_paths_zip+=(-o -path "./$pattern")
done
# The first element will be a spurious '-o', so we remove it.
unset 'prune_paths_zip[0]'

echo "INFO: Finding and modifying markdown files..."
file_count=0
# This corrected find command explicitly groups the prune paths.
# This tells find: ( find paths to prune ) OR ( find the files we want ). This is reliable.
find . \( "${prune_paths_zip[@]}" \) -prune -o -type f -iname "*.md" -print0 | while IFS= read -r -d $'\0' file; do
    # Create a new flat filename by replacing slashes with hyphens.
    flat_filename=$(echo "$file" | sed 's|^\./||' | tr '/' '-')
    
    # Prepend the path as a markdown comment and save to a flat file in the temp directory.
    (echo "<!-- Original Path: $file -->"; echo; cat "$file") > "$TEMP_DIR/$flat_filename"
    file_count=$((file_count + 1))
done

echo "INFO: Found $file_count markdown file(s) to process."

# Now, zip the contents of the temporary directory.
if [ $file_count -gt 0 ]; then
    echo "INFO: Creating zip archive '$ARCHIVE_FILE'..."
    ARCHIVE_FILE_PATH="$(pwd)/$ARCHIVE_FILE"
    (cd "$TEMP_DIR" && zip -r -q "$ARCHIVE_FILE_PATH" .)
    echo "SUCCESS: Created '$ARCHIVE_FILE' with file paths prepended."
else
    echo "WARN: No markdown files were found. An empty archive will not be created."
fi
echo ""

# --- Final Summary ---
echo "-------------------------------------"
echo "Repository digest complete!"
echo "-> Structure file: $STRUCTURE_FILE"
if [ $file_count -gt 0 ]; then
    echo "-> Markdown archive: $ARCHIVE_FILE"
fi
echo "-------------------------------------"

