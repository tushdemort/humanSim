# Master Creative Writing Simulation

Unified script for running solo or pairwise LLM creative-writing simulations with configurable personas, prompts, and conversation modes.

## Quick Start

```bash
# Pairwise (default): 2 agents, 25 runs, 1 prompt, multiple personas
python master_sim.py

# Solo writer: 1 agent writes alone
python master_sim.py --mode solo

# No personas (default writer only)
python master_sim.py --no-persona

# Single persona reused for every run
python master_sim.py --single-persona

# 5 different prompts, 10 total runs
python master_sim.py --num-prompts 5 --num-runs 10

# Custom model and output folder
python master_sim.py --model-id meta-llama/Llama-3.1-8B-Instruct --results-dir ./my_runs
