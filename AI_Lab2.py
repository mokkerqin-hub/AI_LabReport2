import random
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
# ---- Fixed GA Parameters (Lab Report 2, Question 2) ----
POP_SIZE = 300          # 1) population setup is 300
CHROM_LEN = 80          # 3) length of all individuals is 80
TARGET_ONES = 50        # 2) optimum when number of ones == 50
MAX_FITNESS = 80        # 4) return value is 80 when ones == 50
N_GENERATIONS = 50      # 5) number of generations is 50
 
# ---- GA Hyperparameters (simple & sensible defaults) ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN  # ~1 bit flip expected per chromosome
ELITISM = 2                      # best individuals copied unchanged to next generation
 
SELECTION_METHODS = ["Tournament", "Roulette wheel", "Rank-based"]
CROSSOVER_METHODS = ["One-point", "Two-point", "Uniform"]
 
 
# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness peaks when the number of 1s equals TARGET_ONES.
    Max fitness is MAX_FITNESS (=80) at ones == 50.
    Every one above or below 50 costs 1 fitness point.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)
 
 
# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    """Phase 1 - Initialization: random 0/1 chromosomes."""
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)
 
 
# Phase 3 - Selection (three methods from the Chapter 2 slides)
def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    """Pick k random individuals and return the fittest."""
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()
 
 
def selection_probabilities(fits: np.ndarray, method: str):
    """
    Roulette wheel: chance proportional to fitness (all fitness values here are positive).
    Rank-based: chance proportional to rank (worst = 1, best = P).
    Computed once per generation; tournament selection does not need it.
    """
    if method == "Roulette wheel":
        return fits / fits.sum()
    if method == "Rank-based":
        ranks = np.empty(len(fits))
        ranks[np.argsort(fits)] = np.arange(1, len(fits) + 1)
        return ranks / ranks.sum()
    return None
 
 
def wheel_selection(pop: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Spin the wheel once: pick an individual with the given probabilities."""
    return pop[np.random.choice(len(pop), p=probs)].copy()
 
 
def select_parent(pop: np.ndarray, fits: np.ndarray, method: str, k: int, probs=None) -> np.ndarray:
    if method == "Tournament":
        return tournament_selection(pop, fits, k)
    return wheel_selection(pop, probs)
 
 
# Phase 4a - Reproduction: crossover (three methods)
def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    """Swap the tails of two parents after one random cut."""
    point = np.random.randint(1, CHROM_LEN)  # cut between [1 .. CHROM_LEN-1]
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2
 
 
def two_point_crossover(p1: np.ndarray, p2: np.ndarray):
    """Swap the middle segment between two random cuts."""
    a, b = sorted(np.random.choice(range(1, CHROM_LEN), size=2, replace=False))
    c1 = np.concatenate([p1[:a], p2[a:b], p1[b:]])
    c2 = np.concatenate([p2[:a], p1[a:b], p2[b:]])
    return c1, c2
 
 
def uniform_crossover(p1: np.ndarray, p2: np.ndarray):
    """Each bit comes from either parent with probability 0.5."""
    mask = np.random.rand(CHROM_LEN) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)
 
 
def crossover(p1: np.ndarray, p2: np.ndarray, method: str, rate: float):
    if np.random.rand() > rate:
        return p1.copy(), p2.copy()
    if method == "Two-point":
        return two_point_crossover(p1, p2)
    if method == "Uniform":
        return uniform_crossover(p1, p2)
    return single_point_crossover(p1, p2)
 
 
def mutate(individual: np.ndarray, rate: float) -> np.ndarray:
    """Phase 4b - Reproduction: flip-bit mutation keeps diversity."""
    mask = np.random.rand(CHROM_LEN) < rate
    individual[mask] = 1 - individual[mask]
    return individual
 
 
def evolve(pop: np.ndarray, generations: int, selection: str = "Tournament", k: int = TOURNAMENT_K,
           cx_method: str = "One-point", cx_rate: float = CROSSOVER_RATE, mut_rate: float = MUTATION_RATE,
           elitism: int = ELITISM, stop_at_max: bool = False, on_generation=None):
    """
    Run the GA and return the best individual plus a per-generation history.
    Phase 2 (fitness assignment) and Phase 5 (termination) happen inside the loop.
    on_generation(gen, history, best_individual) is called after each generation (used for live animation).
    """
    history = []
    best_individual = None
    best_f = -np.inf
    first_hit = None  # first generation where fitness == MAX_FITNESS
 
    for gen in range(1, generations + 1):
        # Phase 2 - Fitness assignment
        fits = np.array([fitness(ind) for ind in pop], dtype=float)
        ones = pop.sum(axis=1)
 
        # Track best this generation and overall
        gen_best_idx = int(np.argmax(fits))
        gen_best_f = float(fits[gen_best_idx])
        history.append({
            "Generation": gen,
            "Best fitness": gen_best_f,
            "Average fitness": float(fits.mean()),
            "Worst fitness": float(fits.min()),
            "Best ones": int(ones[gen_best_idx]),
            "Average ones": float(ones.mean()),
            "Optimal individuals": int((fits == MAX_FITNESS).sum()),
        })
 
        if gen_best_f > best_f:
            best_f = gen_best_f
            best_individual = pop[gen_best_idx].copy()
        if first_hit is None and gen_best_f == MAX_FITNESS:
            first_hit = gen
        if on_generation is not None:
            on_generation(gen, history, pop[gen_best_idx])
 
        # Phase 5 - Termination (optional early stop at the threshold fitness)
        if stop_at_max and gen_best_f == MAX_FITNESS:
            break
        if gen == generations:
            break
 
        # Elitism: carry the top individuals over unchanged
        elite_idx = np.argsort(fits)[-elitism:] if elitism > 0 else []
        new_pop = [pop[i].copy() for i in elite_idx]
 
        # Create next generation (selection -> crossover -> mutation)
        probs = selection_probabilities(fits, selection)
        while len(new_pop) < len(pop):
            p1 = select_parent(pop, fits, selection, k, probs)
            p2 = select_parent(pop, fits, selection, k, probs)
            c1, c2 = crossover(p1, p2, cx_method, cx_rate)
            c1 = mutate(c1, mut_rate)
            c2 = mutate(c2, mut_rate)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:len(pop)], dtype=np.int8)
 
    return best_individual, best_f, pd.DataFrame(history), pop, first_hit
 
 
def run_ga(seed: int, settings: dict, on_generation=None):
    """Seed, initialise and evolve with the given operator settings."""
    random.seed(seed)
    np.random.seed(seed)
    population = init_population(POP_SIZE, CHROM_LEN)
    return evolve(population, N_GENERATIONS, on_generation=on_generation, **settings)
 
 
def operator_inputs(prefix: str, defaults: dict) -> dict:
    """Streamlit widgets for one set of GA operator settings."""
    sel = st.selectbox("Selection", SELECTION_METHODS, index=SELECTION_METHODS.index(defaults["selection"]), key=prefix + "sel")
    k = st.slider("Tournament size (k)", 2, 10, defaults["k"], key=prefix + "k", disabled=sel != "Tournament")
    cx = st.selectbox("Crossover", CROSSOVER_METHODS, index=CROSSOVER_METHODS.index(defaults["cx_method"]), key=prefix + "cx")
    cx_rate = st.slider("Crossover rate", 0.0, 1.0, defaults["cx_rate"], 0.05, key=prefix + "cxr")
    mut = st.slider("Mutation rate (per bit)", 0.0, 0.10, defaults["mut_rate"], 0.0025, format="%.4f", key=prefix + "mut")
    el = st.slider("Elites per generation", 0, 10, defaults["elitism"], key=prefix + "el")
    return {"selection": sel, "k": k, "cx_method": cx, "cx_rate": cx_rate, "mut_rate": mut, "elitism": el}
 
 
DEFAULTS = {"selection": "Tournament", "k": TOURNAMENT_K, "cx_method": "One-point",
            "cx_rate": CROSSOVER_RATE, "mut_rate": round(MUTATION_RATE, 4), "elitism": ELITISM}
 
 
# ---- Streamlit UI ----
st.set_page_config(page_title="Bit Pattern GA (80 bits, target ones = 50)", page_icon="🧬", layout="centered")
 
st.title("🧬 Genetic Algorithm: Evolve an 80-bit Pattern")
st.caption(
    "BSD3513 LAB REPORT 2 | MOK KER QIN | SD24032"
)
 
# Problem setup and fixed requirements on the left (sidebar)
with st.sidebar:
    st.header("ℹ️ Problem setup")
    st.caption("Fixed by the lab requirement")
    st.table(pd.DataFrame({
        "Requirement": ["Population size", "Chromosome length", "Target number of ones",
                        "Max fitness at optimum", "Generations"],
        "Value": [POP_SIZE, CHROM_LEN, TARGET_ONES, MAX_FITNESS, N_GENERATIONS],
    }).set_index("Requirement"))
    st.markdown("**Fitness function**")
    st.code(f"f(x) = {MAX_FITNESS} - |ones(x) - {TARGET_ONES}|", language="text")
    st.caption(f"Maximum of {MAX_FITNESS} when the chromosome has exactly {TARGET_ONES} ones; "
               "each one above or below costs 1 point.")
 
tab_run, tab_compare, tab_info = st.tabs(["▶️ Run GA", "⚖️ Compare settings", "🧠 How it works"])
 
# ================= Tab 1: single run with live animation =================
with tab_run:
    with st.expander("⚙️ GA operators (optional tuning)"):
        st.caption("The fixed requirements in the sidebar cannot be changed.")
        settings = operator_inputs("run_", DEFAULTS)
        stop_at_max = st.checkbox("Stop early once fitness 80 is reached (termination criterion)", value=False)
 
    col1, col2, col3 = st.columns([2, 2, 1.6])
    with col1:
        seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)
    with col2:
        live = st.checkbox("Animate live", value=True)
        delay = st.slider("Animation delay (s/gen)", 0.0, 0.3, 0.05, 0.01, disabled=not live)
    with col3:
        st.write("")
        run_btn = st.button("Run Genetic Algorithm", type="primary")
 
    if run_btn:
        live_status, live_bits, live_chart, live_progress = st.empty(), st.empty(), st.empty(), st.progress(0)
 
        def show_generation(gen, history, gen_best):
            """Live update after every generation."""
            live_progress.progress(gen / N_GENERATIONS)
            if not live:
                return
            last = history[-1]
            live_status.markdown(
                f"**Generation {gen}/{N_GENERATIONS}** - best fitness **{last['Best fitness']:.0f}**, "
                f"average ones **{last['Average ones']:.2f}**, individuals with 50 ones **{last['Optimal individuals']}**"
            )
            live_bits.code("".join(map(str, gen_best.tolist())), language="text")
            live_chart.line_chart(pd.DataFrame(history).set_index("Generation")[["Best fitness", "Average fitness", "Worst fitness"]])
            time.sleep(delay)
 
        run_settings = dict(settings, stop_at_max=stop_at_max)
        result = run_ga(int(seed), run_settings, on_generation=show_generation)
        for area in (live_status, live_bits, live_chart, live_progress):
            area.empty()
        st.session_state["result"] = (int(seed), run_settings) + result
 
    if "result" in st.session_state:
        seed_used, used, best_ind, best_fit, history, final_pop, first_hit = st.session_state["result"]
 
        # Results
        ones_count = int(best_ind.sum())
        zeros_count = CHROM_LEN - ones_count
        bitstring = "".join(map(str, best_ind.tolist()))
 
        st.subheader("🏁 Best Individual Found")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Best Fitness", value=f"{best_fit:.0f} / {MAX_FITNESS}")
        m2.metric(label="Number of Ones", value=f"{ones_count}")
        m3.metric(label="Optimum first reached", value=f"Gen {first_hit}" if first_hit else "—")
        st.write(f"**Ones**: {ones_count} | **Zeros**: {zeros_count} | **Length**: {CHROM_LEN} | **Seed**: {seed_used}")
        st.caption(f"Operators: {used['selection']} selection, {used['cx_method']} crossover "
                   f"(rate {used['cx_rate']}), mutation {used['mut_rate']:.4f}, {used['elitism']} elites")
        st.code(bitstring, language="text")
 
        # Convergence plot
        st.subheader("📉 Convergence (Fitness per Generation)")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["Generation"], history["Best fitness"], linewidth=2, label="Best")
        ax.plot(history["Generation"], history["Average fitness"], linewidth=2, linestyle="--", label="Average")
        ax.plot(history["Generation"], history["Worst fitness"], linewidth=1, alpha=0.7, label="Worst")
        ax.axhline(MAX_FITNESS, color="grey", linestyle=":", label=f"Max = {MAX_FITNESS}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA Convergence")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "With 300 random individuals, one with exactly 50 ones often exists already in generation 1, "
            "so the best line can start at 80. The average line shows the whole population improving."
        )
 
        # Population drift towards the target
        st.subheader("🎯 Population Moving Towards 50 Ones")
        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
        ax3.plot(history["Generation"], history["Average ones"], linewidth=2, label="Average ones")
        ax3.axhline(TARGET_ONES, color="red", linestyle="--", label=f"Target = {TARGET_ONES}")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Average number of ones")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3b = ax3.twinx()
        ax3b.bar(history["Generation"], history["Optimal individuals"], alpha=0.25, color="green", label="Individuals with 50 ones")
        ax3b.set_ylabel("Individuals with 50 ones")
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3b.get_legend_handles_labels()
        ax3.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        st.pyplot(fig3)
        plt.close(fig3)
 
        # Final population distribution
        st.subheader("📊 Number of Ones in the Final Population")
        final_ones = final_pop.sum(axis=1)
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.hist(final_ones, bins=range(int(final_ones.min()), int(final_ones.max()) + 2), align="left", edgecolor="white")
        ax2.axvline(TARGET_ONES, color="red", linestyle="--", label=f"Target = {TARGET_ONES}")
        ax2.set_xlabel("Number of ones")
        ax2.set_ylabel("Individuals")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)
        optimal_share = float((final_ones == TARGET_ONES).mean())
        st.write(f"{optimal_share:.0%} of the final population has exactly {TARGET_ONES} ones.")
 
        # Generation table + download
        with st.expander("📋 Per-generation statistics"):
            st.dataframe(history, hide_index=True)
            st.download_button("Download CSV", history.to_csv(index=False), file_name="ga_history.csv", mime="text/csv")
 
        # Validation note
        if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
            st.success(f"Perfect match achieved: ones = {TARGET_ONES} and fitness = {MAX_FITNESS} ✅")
        else:
            st.info("GA may reach near-optimal solutions; try another seed to explore further.")
 
# ================= Tab 2: compare two settings =================
with tab_compare:
    st.write("Run two operator settings on the **same seeds** and compare how fast the whole population improves.")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Setting A**")
        set_a = operator_inputs("a_", DEFAULTS)
    with cb:
        st.markdown("**Setting B**")
        set_b = operator_inputs("b_", dict(DEFAULTS, selection="Roulette wheel"))
    n_seeds = st.slider("Number of seeds to average (seeds 0, 1, 2, ...)", 1, 10, 5)
 
    if st.button("Compare A vs B", type="primary"):
        progress = st.progress(0)
        curves, summary = {}, []
        for i, (name, cfg) in enumerate([("A", set_a), ("B", set_b)]):
            runs = []
            for s in range(n_seeds):
                runs.append(run_ga(s, cfg))
                progress.progress((i * n_seeds + s + 1) / (2 * n_seeds))
            hist = pd.concat([r[2] for r in runs]).groupby("Generation").mean()
            curves[name] = hist
            hits = [r[4] for r in runs if r[4] is not None]
            summary.append({
                "Setting": name,
                "Selection": cfg["selection"],
                "Crossover": cfg["cx_method"],
                "Reached 80": f"{sum(r[1] == MAX_FITNESS for r in runs)}/{n_seeds}",
                "1st gen at 80": f"{np.mean(hits):.1f}" if hits else "—",
                "Final avg fitness": round(float(hist["Average fitness"].iloc[-1]), 2),
                "% with 50 ones": f"{hist['Optimal individuals'].iloc[-1] / POP_SIZE:.0%}",
            })
        progress.empty()
        st.session_state["compare"] = (curves, summary, n_seeds)
 
    if "compare" in st.session_state:
        curves, summary, n_used = st.session_state["compare"]
        fig4, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
        for name, style in (("A", "-"), ("B", "--")):
            a1.plot(curves[name].index, curves[name]["Average fitness"], linestyle=style, linewidth=2, label=f"Setting {name}")
            a2.plot(curves[name].index, curves[name]["Optimal individuals"] / POP_SIZE * 100, linestyle=style, linewidth=2, label=f"Setting {name}")
        a1.set_title("Average fitness")
        a2.set_title("% of population with 50 ones")
        for a in (a1, a2):
            a.set_xlabel("Generation")
            a.grid(True, linestyle="--", alpha=0.5)
            a.legend()
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
        st.dataframe(pd.DataFrame(summary), hide_index=True)
        st.caption(f"Curves are averaged over {n_used} seed(s). Both settings use the same seeds, so they start from the same populations.")
 
# ================= Tab 3: explanation =================
with tab_info:
    st.subheader("GA flowchart")
    st.graphviz_chart(f"""
digraph GA {{
    rankdir=TB; nodesep=0.4; ranksep=0.3;
    node [shape=box, style="rounded,filled", fillcolor="#eef3fa", fontname="Helvetica", fontsize=11, width=2.6];
    edge [fontname="Helvetica", fontsize=10];
    start [label="Start", shape=oval, fillcolor="#dfe9d8", width=1.2];
    init  [label="1. Initialization\n{POP_SIZE} random {CHROM_LEN}-bit strings"];
    fit   [label="2. Fitness assignment\nf(x) = {MAX_FITNESS} - |ones - {TARGET_ONES}|"];
    stop  [label="5. Termination?\n{N_GENERATIONS} generations done", shape=diamond, fillcolor="#fdf1d8", width=2.8];
    out   [label="Return best individual", shape=oval, fillcolor="#dfe9d8", width=2.2];
    sel   [label="3. Selection\npick 2 parents (fitter = more likely)"];
    cx    [label="4a. Crossover\nmix the 2 parents into 2 children"];
    mut   [label="4b. Mutation\nflip each bit with small probability"];
    newp  [label="New population\n{ELITISM} elites + {POP_SIZE - ELITISM} children"];
    start -> init -> fit -> stop;
    stop -> out [label="  yes"];
    stop -> sel [label="  no"];
    sel -> cx -> mut;
    mut -> sel [label=" repeat until\n full", style=dashed];
    mut -> newp;
    newp -> fit [label="  next generation"];
    {{ rank=same; stop; out; }}
}}
""", use_container_width=False)
    st.caption("The loop runs once per generation: score everyone, then build a new population from the fittest.")
 
    st.subheader("The five phases")
    st.table(pd.DataFrame([
        ["1. Initialization", f"Create {POP_SIZE} random chromosomes of {CHROM_LEN} bits.", "init_population"],
        ["2. Fitness assignment", f"Score each chromosome: {TARGET_ONES} ones scores {MAX_FITNESS}, each one off costs 1 point.", "fitness"],
        ["3. Selection", "Choose parents, favouring fitter individuals.", "select_parent"],
        ["4. Reproduction", f"Crossover mixes two parents; mutation flips random bits. The {ELITISM} best are kept unchanged.", "crossover, mutate"],
        ["5. Termination", f"Stop after {N_GENERATIONS} generations (or early at fitness {MAX_FITNESS}) and return the best.", "evolve"],
    ], columns=["Phase", "What happens", "Function in code"]).set_index("Phase"))
 
    st.subheader("Operator options")
    st.table(pd.DataFrame([
        ["Selection", "Tournament", "The fittest of k random individuals wins."],
        ["Selection", "Roulette wheel", "Chance of being picked is proportional to fitness."],
        ["Selection", "Rank-based", "Chance is proportional to rank, so only the order of fitness matters."],
        ["Crossover", "One-point", "Swap the tails after one random cut."],
        ["Crossover", "Two-point", "Swap the middle segment between two random cuts."],
        ["Crossover", "Uniform", "Each bit comes from either parent with probability 0.5."],
        ["Mutation", "Flip-bit", "Each bit flips 0 ↔ 1 with the mutation rate, keeping diversity."],
    ], columns=["Step", "Method", "How it works"]).set_index("Step"))
 
    st.subheader("Complexity")
    st.table(pd.DataFrame([
        ["Time (one run)", "O(G × P × L)", f"{N_GENERATIONS} × {POP_SIZE} × {CHROM_LEN} = 1.2 million bit operations"],
        ["Memory", "O(P × L)", f"{POP_SIZE} × {CHROM_LEN} = 24,000 bits for the population"],
        ["Tournament selection", "O(k) per parent", "only k individuals are compared"],
        ["Roulette / rank selection", "O(P) per parent", "plus O(P log P) once per generation to sort for rank"],
    ], columns=["Item", "Complexity", "For this problem"]).set_index("Item"))
 
