# 🧬 Genetic Algorithm: Evolve an 80-bit Pattern

BSD3513 Introduction to Artificial Intelligence, Lab Report 2 (Question 2).

A Streamlit app that runs a genetic algorithm (GA) to evolve 80-bit chromosomes whose number of ones equals 50.

**Live app:** https://ailabreport2-bjpgapp9gz29alguqytxejr.streamlit.app/

## Problem 

| Requirement | Value |
|---|---|
| Population size | 300 |
| Chromosome length | 80 bits |
| Optimum | number of ones = 50 |
| Fitness at optimum | 80 |
| Generations | 50 |

Fitness function: `f(x) = 80 - |ones(x) - 50|`

## GA design

| Phase (lecture notes) | Implementation |
|---|---|
| Initialization | 300 random 0/1 chromosomes |
| Fitness assignment | `f(x) = 80 - abs(ones - 50)` |
| Selection | Tournament (k = 3, default), roulette wheel or rank-based |
| Reproduction | One-point (default), two-point or uniform crossover (rate 0.9) + flip-bit mutation (rate 1/80 per bit), 2 elites kept |
| Termination | After 50 generations (optional early stop at fitness 80) |

Time complexity is O(G × P × L) = 50 × 300 × 80 bit operations; memory is O(P × L).

## Features

**Sidebar (left)**
- Problem setup: the fixed requirements table and the fitness function

**Run GA tab**
- **Live animation:** the chart, best bitstring and statistics update every generation, with adjustable speed
- **Choice of operators:** tournament / roulette wheel / rank-based selection and one-point / two-point / uniform crossover, plus crossover rate, mutation rate and elites
- Seeded runs, so every result is reproducible
- Best individual with its bitstring, fitness and the generation the optimum was first found
- Convergence chart (best, average, worst), average-ones chart, and final-population histogram
- Per-generation statistics table with CSV download

**How it works tab**
- A flowchart of the GA loop, plus tables of the five phases, the operator options and the complexity

## Results of comparing operators (10 seeds)

| Selection | One-point | Two-point | Uniform |
|---|---|---|---|
| Tournament (k=3) | 78.46 | 78.23 | 77.71 |
| Rank-based | 78.20 | 78.07 | 77.44 |
| Roulette wheel | 75.44 | 75.40 | 75.34 |

Final average fitness after 50 generations (seeds 0–9, run with the operator settings in the Run GA tab). Every setting found fitness 80 in all 10 runs.

## Screenshots

| Live animation | Best individual |
|---|---|
| ![Live](screenshots/02_live_animation.png) | ![Best](screenshots/03_best_and_convergence.png) |
| **Population charts** | **Problem setup (sidebar)** |
| ![Population](screenshots/04_population.png) | ![Setup](screenshots/01_setup.png) |
| **How it works** | |
| ![How it works](screenshots/06_how_it_works.png) | |

## Run locally
1. Download this repo (green Code button → Download ZIP) and unzip it.
2. Open a terminal in the folder and run:
```bash
pip install -r requirements.txt
streamlit run bfsdfs.py
```

## Project structure
```
bfsdfs.py            # Streamlit app
requirements.txt     # dependencies
screenshots/         # README images
```

