# Master Creative Writing Simulation

Run creative-writing simulations across **three independent dimensions**, giving **12 possible configurations**.

## Dimensions

| Flag | Choices | Description |
|------|---------|-------------|
| `--mode` | `solo`, `pairwise` | One writer alone, or two agents collaborating |
| `--persona` | `none`, `single`, `multiple` | `none`=default writer; `single`=same persona(s) every run; `multiple`=fresh persona(s) per run |
| `--prompts` | `single`, `multiple` | `single`=same prompt for all runs; `multiple`=new prompt per run |

## Quick-start recipes

```bash
# 1. Pairwise, multiple personas, single prompt (closest to original)
python master_sim.py --mode pairwise --persona multiple --prompts single

# 2. Solo writer, no persona, multiple prompts
python master_sim.py --mode solo --persona none --prompts multiple

# 3. Pairwise, same two personas reused, single prompt
python master_sim.py --mode pairwise --persona single --prompts single

# 4. Solo, new persona each run, one prompt
python master_sim.py --mode solo --persona multiple --prompts single

# 5. Full sweep: 25 runs, everything fresh each time
python master_sim.py --mode pairwise --persona multiple --prompts multiple --num-runs 25
