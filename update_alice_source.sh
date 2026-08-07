#!/bin/bash
# Hook to automatically update alice_unikernel.txt
TARGET_FILE="/home/killboxincorporated/alice_unikernel.txt"

echo "========================================" > "$TARGET_FILE"
echo "A.L.I.C.E. UNIKERNEL V10 SOURCE CODE DUMP" >> "$TARGET_FILE"
echo "========================================" >> "$TARGET_FILE"
echo "" >> "$TARGET_FILE"

# Find all .rs files excluding target, .git, and other cache directories
find /home/killboxincorporated -type d \( -name "target" -o -name ".git" -o -name "hf-venv" -o -name ".cache" -o -name ".rustup" -o -name ".cargo" \) -prune -o -type f -name "*.rs" -print | sort | while read -r file; do
    # Extract the top-level project directory name (e.g., aegis-uefi, aegis-forge)
    rel_path="${file#/home/killboxincorporated/}"
    project_dir=$(echo "$rel_path" | cut -d'/' -f1)
    
    echo "Directory: $project_dir" >> "$TARGET_FILE"
    echo "========================================" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    echo "File: $file" >> "$TARGET_FILE"
    echo "----------------------------------------" >> "$TARGET_FILE"
    cat "$file" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
done

echo "[AEGIS] Successfully updated source code dump at $TARGET_FILE"
