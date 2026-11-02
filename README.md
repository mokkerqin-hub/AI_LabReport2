# 🧬 Genetic Algorithm: Evolve an 80-bit Pattern

BSD3513 Introduction to Artificial Intelligence, Lab Report 2 (Question 2)
**Mok Ker Qin | SD24032**

A Streamlit app that runs a genetic algorithm (GA) to evolve 80-bit chromosomes whose number of ones equals 50.

**Live app:** https://ailabreport2-bjpgapp9gz29alguqytxejr.streamlit.app/
**Repository:** https://github.com/mokkerqin-hub/AI_LabReport2

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

| Phase | Implementation |
|---|---|
| Initialization | 300 random 0/1 chromosomes |
| Fitness assignment | `f(x) = 80 - abs(ones - 50)` |
| Selection | Tournament (k = 3, default), roulette wheel or rank-based |
| Reproduction | One-point (default), two-point or uniform crossover (rate 0.9) + flip-bit mutation (rate 1/80 per bit), 2 elites kept |
| Termination | After 50 generations (optional early stop at fitness 80) |

## Screenshots

### 1. Home page
![Home page](screenshot/cover.png)
The sidebar shows the fixed lab requirements and the fitness function. The **Run GA** tab lets the user set a random seed, turn the live animation on or off, adjust its speed, and open the optional GA operator settings before clicking **Run Genetic Algorithm**.

### 2. Best individual and convergence
![Best individual and convergence](screenshot/convergence.png)
After a run (seed 42), the best individual reaches the maximum fitness of **80 / 80** with exactly **50 ones** and 30 zeros, and its 80-bit string is displayed. The convergence chart plots the best, average and worst fitness for each generation: the best line stays at 80, while the average rises from about 70 to about 78 as the population improves.

### 3. Population moving towards 50 ones
![Population and final distribution](screenshot/chart.png)
The top chart shows the average number of ones climbing from about 40 to the target of 50 within the first few generations, with the green bars counting how many individuals have exactly 50 ones. The histogram shows the final population clustered around 50 ones.

### 4. GA flowchart
![GA flowchart](screenshot/flowchart.png)
The **How it works** tab shows the GA loop: initialization → fitness assignment → termination check → selection → crossover → mutation → new population (2 elites + 298 children), repeated for 50 generations.

### 5. Phases, operators and complexity
![Phases, operators and complexity](screenshot/hotitworks.png)
Tables explain the five GA phases with the function that implements each one, the available selection, crossover and mutation methods, and the complexity: time O(G × P × L) = 50 × 300 × 80 = 1.2 million bit operations, and memory O(P × L) = 24,000 bits.

## Run locally
1. Download this repo (green **Code** button → **Download ZIP**) and unzip it.
2. Open a terminal in the folder and run:
```bash
pip install -r requirements.txt
streamlit run AI_Lab2.py
```

## Project structure
```
AI_Lab2.py           # Streamlit app
requirements.txt     # dependencies
screenshot/          # screenshots of the live app
```
