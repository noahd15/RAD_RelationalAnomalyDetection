# Artifacts Layout

This directory is organized by dataset domain.

## Folders
- `amazon/`: Amazon-specific generated artifacts.
- `hm/`: H&M-specific generated artifacts.
- `shared/`: Cross-dataset experiment outputs.

## Backward compatibility
Legacy top-level artifact names are preserved as symbolic links to the new locations.

## Rule for new files
Write dataset-specific outputs to the matching dataset folder.
