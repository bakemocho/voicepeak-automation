# Architecture (Draft)

## Layers

1. CLI layer
2. Workflow layer (validation, queueing, retries)
3. Adapter layer (UI automation/backend-specific control)
4. Artifact layer (logs, outputs, metadata)

## Design Goal

Keep adapter-specific fragility isolated from workflow logic so backend changes do not break task schema.
