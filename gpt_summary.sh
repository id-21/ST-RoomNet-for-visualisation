#!/bin/bash
# flat_summary.sh
# 1. Create repository_structure.md listing all .py and .md files.
# 2. Create a flat zip of all .md files with original paths prepended inside.

set -euo pipefail

STRUCTURE_FILE="repository_structure.md"
ARCHIVE_FILE="markdown_files.zip"
IGNORE_PATTERNS=("venv" "__pycache__" ".git" "dist" "build" "*.egg-info" "node_modules")

echo "INFO: Generating repository structure in '$STRUCTURE_FILE'..."

if command -v tree &> /dev/null; then
    ignore_str=$(IFS="|"; echo "${IGNORE_PATTERNS[*]}")
    tree -a -P "*.py|*.md" -I "$ignore_str" > "$STRUCTURE_FILE"
else
    find_prune_args=()
    for pattern in "${IGNORE_PATTERNS[@]}"; do
        find_prune_args+=(-o -path "./$pattern" -prune)
    done
    unset 'find_prune_args[0]'
    find . "${find_prune_args[@]}" -o -type f \( -name "*.py" -o -iname "*.md" \) -print | sort > "$STRUCTURE_FILE"
fi
echo "SUCCESS: Created '$STRUCTURE_FILE'."
echo ""

echo "INFO: Preparing to archive all Markdown files..."

if [ -f "$ARCHIVE_FILE" ]; then
    rm "$ARCHIVE_FILE"
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Build prune args for zip
find_prune_args_zip=()
for pattern in "${IGNORE_PATTERNS[@]}"; do
    find_prune_args_zip+=(-o -path "./$pattern" -prune)
done
unset 'find_prune_args_zip[0]'

file_count=0
while IFS= read -r -d '' file; do
    rel_path="${file#./}"
    flat_filename=$(echo "$rel_path" | tr '/' '-')
    # Ensure .md extension remains at the end
    if [[ "$flat_filename" != *.md && "$flat_filename" != *.MD ]]; then
        flat_filename="${flat_filename}.md"
    fi

    {
      echo "Original Path: $rel_path"
      echo ""
      cat "$file"
    } > "$TEMP_DIR/$flat_filename"

    ((file_count++))
done < <(find . "${find_prune_args_zip[@]}" -o -type f -iname "*.md" -print0)

echo "INFO: Found $file_count markdown file(s)."

if [ "$file_count" -gt 0 ]; then
    (cd "$TEMP_DIR" && zip -q -r "../$ARCHIVE_FILE" .)
    echo "SUCCESS: Created '$ARCHIVE_FILE'."
else
    echo "WARN: No markdown files found. Archive not created."
fi

echo ""
echo "-------------------------------------"
echo "Repository digest complete!"
echo "-> Structure file: $STRUCTURE_FILE"
if [ "$file_count" -gt 0 ]; then
    echo "-> Markdown archive: $ARCHIVE_FILE"
fi
echo "-------------------------------------"
