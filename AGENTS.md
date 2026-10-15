# Repository Guidance

## Product Scope

- The long-term goal is two-way translation between SIBI and Indonesian.
- The current working capability is local webcam detection of 24 static alphabet classes (A-I and K-Y); J and Z are excluded. Do not describe this as sentence translation or a complete translator.
- Webcam inference is a standalone ML script. Do not add web integration unless the task explicitly asks for it.
- The Next.js page is starter UI, and the FastAPI app currently exposes only a health endpoint.

## Project Layout

- `frontend/`: Next.js application.
- `backend/`: FastAPI application.
- `ml/data/raw/`: source datasets.
- `ml/data/processed/`: prepared detection images, labels, and manifests.
- `ml/datasets/sibi/`: generated fold lists and YOLO configuration files.
- `ml/exports/`: exported ONNX models used by the webcam test.
- `ml/scripts/`: data preparation, training, evaluation, export, and local webcam tools.

## ML Conventions

- Run Python commands from `ml/` with `.venv\Scripts\python.exe` on Windows. Prefer `python -m pip` through that interpreter to avoid stale `pip.exe` launchers.
- The ONNX models contain their class-name mapping in model metadata. Keep inference labels aligned with that mapping.
- Dataset splitting must keep frames from the same subject/video or duplicate group in one fold to avoid leakage.
- A synthetic-frame inference check validates code paths, not real-world camera accuracy. Verify webcam behavior on a machine with an actual camera and display.

## Documentation and Validation

- Check the implementation before marking a backlog story complete; distinguish local ML tools from frontend/backend capabilities.
- Keep changes scoped to the requested package and preserve the generated Next.js rules in `frontend/AGENTS.md`.
- Run a focused test or validation for each code change. For camera changes, test inference separately from the physical webcam when the environment has no camera/display.