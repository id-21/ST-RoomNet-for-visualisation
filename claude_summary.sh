#!/bin/bash
#
# This script generates a digest of a code repository.
# 1. It creates a structure file (`repository_structure.md`) listing all .py and .md files.
# 2. It finds all markdown files, prepends their original path to the top of each,
#    renames them to include the path, and creates a flat zip archive.

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
    
    prune_args=()
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        prune_args+=(-path "./$pattern" -prune -o)
    done

    find . "${prune_args[@]}" -type f \( -name "*.py" -o -iname "*.md" \) -print | sort > "$STRUCTURE_FILE"
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

echo "INFO: Finding and modifying markdown files..."

# Build prune arguments for find
prune_args=()
for pattern in "${IGNORE_PATTERNS[@]}"; do
    prune_args+=(-path "./$pattern" -prune -o)
done

# Create a temporary file to store the list of markdown files
TEMP_FILE_LIST=$(mktemp)
find . "${prune_args[@]}" -type f -iname "*.md" -print > "$TEMP_FILE_LIST"

# Count the files
file_count=$(wc -l < "$TEMP_FILE_LIST" | tr -d ' ')
echo "INFO: Found $file_count markdown file(s) to process."

# Process each file
while IFS= read -r file; do
    # Skip empty lines
    [ -z "$file" ] && continue
    
    # Remove leading './' from the path
    relative_path="${file#./}"
    
    # Create a flat filename by replacing slashes with double-hyphens
    flat_filename=$(echo "$relative_path" | tr '/' '-')
    
    # Prepend the path as a markdown comment and save to temp directory
    {
        echo "<!-- Original Path: $relative_path -->"
        echo ""
        cat "$file"
    } > "$TEMP_DIR/$flat_filename"
done < "$TEMP_FILE_LIST"

# Clean up the temporary file list
rm "$TEMP_FILE_LIST"

# Now, zip the contents of the temporary directory.
if [ "$file_count" -gt 0 ]; then
    echo "INFO: Creating zip archive '$ARCHIVE_FILE'..."
    ARCHIVE_FILE_PATH="$(pwd)/$ARCHIVE_FILE"
    (cd "$TEMP_DIR" && zip -r -q "$ARCHIVE_FILE_PATH" .)
    echo "SUCCESS: Created '$ARCHIVE_FILE' with $file_count file(s)."
else
    echo "WARN: No markdown files were found. An empty archive will not be created."
fi
echo ""

# --- Final Summary ---
echo "-------------------------------------"
echo "Repository digest complete!"
echo "-> Structure file: $STRUCTURE_FILE"
if [ "$file_count" -gt 0 ]; then
    echo "-> Markdown archive: $ARCHIVE_FILE (contains $file_count file(s))"
fi
echo "-------------------------------------"