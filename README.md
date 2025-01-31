# Electric Mesa
A "toy" idea of connecting an OOP (Mesa) model to a Jupyter notebook and discrete vectors therein.

This project demonstrates connecting a Mesa agent-based model to a Jupyter notebook interface, allowing users to:
1. Use familiar notebook-style parameter settings
2. Run Mesa's agent-based simulation engine
3. Convert Mesa outputs back to vector format for analysis
4. Generate plots matching the original implementation

## Quick Start
This project uses the [uv](https://docs.astral.sh/uv/) package manager. Not a requirement, but maybe check it out. To get started with Electric Mesa using `uv`, navigate your shell to the root of the project directory and enter:

`uv init`

#### Activate your virtual environment:
Mac/Linux: `source .venv/bin/activate`

Win: `.venv\Scripts\activate`

#### Install dependencies from pyproject.toml:
`uv sync`

## Usage
The main interface is `notebooks/experiment.ipynb`. When running:
1. Ensure your Jupyter environment uses the activated virtual environment
2. Parameters can be set individually or using defaults
3. Results match the format of the original implementation

## Structure
- `model.py`: Mesa model implementation
- `agents.py`: GenCo agent definition
- `notebooks/experiment.ipynb`: Main interface
- `docs/ElectricityAuctions.ipynb`: Original implementation (reference only)

## License
- Mesa components: Apache 2.0
- Original notebook (ElectricityAuctions.ipynb): Copyright J. Applegate / Arizona State University