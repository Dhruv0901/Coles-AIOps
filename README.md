# Gesture Vision Experiments

A modular Python scaffold for computer-vision and gesture-recognition experimentation focused on operator-only checkout workflows.

## Structure

- `src/`: reusable pipeline modules
- `configs/`: experiment configuration files
- `experiments/`: generated experiment summaries
- `artifacts/`: run artifacts and snapshots
- `logs/`: runtime logs
- `tests/`: import and wiring checks
- `main.py`: CLI entrypoint for running one or many experiments

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python main.py --config configs/hand_landmarks.yaml
python main.py --params params.yaml
```

## Testing

```bash
pytest
```
