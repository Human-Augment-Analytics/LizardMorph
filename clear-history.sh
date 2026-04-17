#!/bin/bash

PROJECT_DIR="$HOME/.claude/projects/-Users-leyangloh-dev-LizardMorph"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "No Claude session directory found for this project."
    exit 1
fi

COUNT=$(ls "$PROJECT_DIR" | wc -l | tr -d ' ')

if [[ "$COUNT" -eq 0 ]]; then
    echo "No sessions to remove."
    exit 0
fi

echo "Found $COUNT files in $PROJECT_DIR"
read -p "Delete all? [y/N] " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$PROJECT_DIR"/*
    echo "Cleared $COUNT files."
else
    echo "Aborted."
fi
