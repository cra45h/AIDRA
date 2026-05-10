"""
main.py
AIDRA — Adaptive Intelligent Disaster Response Agent
Main entry point. Integrates all modules and runs the simulation.

Usage:
  python main.py              # Full Pygame simulation
  python main.py --headless   # Headless mode (charts only, no window)
  python main.py --benchmark  # Run all algorithms and print comparison

Author: AIDRA Team
Course: AIC-201 (Artificial Intelligence)
"""

import sys
import os
import time
import argparse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Core modules ──────────────────────────────────────────────────────────────
from environment import DisasterEnvironment
from search.bfs import bfs
from search.dfs import dfs
from search.greedy import greedy
from search.astar import astar, astar_compare
from search.local_search import compare_local_search
from csp.csp_solver import CSPSolver, compare_csp
from ml.ml_models import MLSystem
from fuzzy.fuzzy_system import FuzzyDisasterSystem
from replanning.replanner import Replanner
from utils.agent import AIDRAAgent
from datasets.generate_dataset import save_datasets
from visualization.kpi_charts import generate_all_charts


def run_headless(env, agent, ml_system, fuzzy_system, search_results,
                  csp_results, sa_result, hc_result, ml_results):
    """Run simulation headlessly for a fixed number of steps and generate charts."""
    print("\n" + "="*60)
    print("AIDRA — HEADLESS SIMULATION")
    print("="*60)

    agent.initialize()

    for step in range(100):
        events = agent.step()
        if events:
            for ev in events:
                print(f"  [Step {env.step_count}] {ev}")

    print("\n[Main] Generating KPI charts...")
    chart_paths = generate_all_charts(
        env, search_results, ml_results, csp_results, sa_result, hc_result
    )
    print(f"[Main] Charts saved: {chart_paths}")

    print("\n[Main] Decision Log (last 5 entries):")
    agent.replanner.print_log()

    print("\n[Main] KPIs:")
    kpis = env.get_kpis()
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    print("\n[Main] ML Comparison:")
    print(ml_system.compare_models())

    if csp_results:
        print("\n[Main] CSP Comparison:")
        print(csp_results.get("comparison", ""))

    local_comparison = compare_local_search(env.victims, env)
    print(f"\n[Main] Local Search: {local_comparison['comparison']}")


def run_pygame(env, agent, fuzzy_system, ml_system):
    """Run full interactive Pygame simulation."""
    try:
        import pygame
    except ImportError:
        print("[Main] Pygame not installed. Run: pip install pygame")
        return

    from visualization.pygame_vis import AIDRAVisualizer
    from search.astar import astar

    vis = AIDRAVisualizer(env, algorithm="A*")
    vis.init()

    agent.initialize()

    # Get initial route from ambulance 0 for display
    if env.ambulances[0].route:
        vis.set_path(env.ambulances[0].route)

    # Initial fuzzy assessment for first critical victim
    if env.victims:
        v = env.victims[0]
        r, c = v.pos
        fire_val = float(env.risk_map[r][c])
        fo = fuzzy_system.evaluate(fire_val, max(0.01, 1.0-fire_val/10), 0.3, v.condition)
        vis.set_fuzzy(fo)
        mo = agent.get_current_assessment(v.vid)
        vis.set_ml(mo)

    step_delay = 5  # steps between vis frames

    while vis.running or True:
        action = vis.handle_events()
        if action == 'quit':
            break
        if action == 'reset':
            env.__init__(seed=42)
            agent.initialize()
            vis.events_log = []

        if not vis.paused:
            events = agent.step()
            if events:
                for ev in events:
                    vis.add_event(ev)

            # Update visualization path
            pass  # paths drawn directly from env.ambulances in draw_path()

            # Update info panel
            info = [
                f"Step: {env.step_count}",
                f"Victims saved: {env.victims_saved}/5",
                f"Replanning: {env.replanning_triggers}",
                f"Algorithm: A* (balanced)",
            ]
            vis.set_search_info(info)

        vis.render()
        vis.tick(10)  # 10 FPS

    vis.quit()
    print("\n[Main] Simulation ended. Generating charts...")
    return


def benchmark_search(env):
    """Compare all search algorithms on a test route."""
    print("\n" + "="*60)
    print("SEARCH ALGORITHM BENCHMARK")
    print("="*60)
    start = env.base_pos
    goal  = env.victims[0].pos

    results = {}
    for name, fn in [("BFS", bfs), ("DFS", dfs), ("Greedy", greedy)]:
        r = fn(env, start, goal)
        results[name] = r
        print(f"\n{name}:")
        print(f"  Path length: {len(r['path'])} | Cost: {r['path_cost']:.2f} | "
              f"Nodes: {r['nodes_expanded']} | Time: {r['runtime']*1000:.3f}ms")
        print(f"  {r['explanation']}")

    for mode in ["speed", "safe", "balanced"]:
        r = astar(env, start, goal, mode=mode)
        key = f"A*-{mode}"
        results[key] = r
        print(f"\nA* ({mode}):")
        print(f"  Path length: {len(r['path'])} | Cost: {r['path_cost']:.2f} | "
              f"Risk: {r.get('path_risk',0):.2f} | Nodes: {r['nodes_expanded']} | "
              f"Time: {r['runtime']*1000:.3f}ms")
        print(f"  {r['explanation']}")

    cmp = astar_compare(env, start, goal)
    print(f"\nA* Trade-off: {cmp['tradeoff_analysis']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AIDRA — Disaster Response Agent")
    parser.add_argument("--headless",   action="store_true", help="No Pygame window")
    parser.add_argument("--benchmark",  action="store_true", help="Run algorithm benchmark only")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random)")
    args = parser.parse_args()

    print("="*60)
    print("AIDRA — Adaptive Intelligent Disaster Response Agent")
    print("AIC-201 | Artificial Intelligence")
    print("="*60)

    # ── 1. Environment ───────────────────────────────────────────────────────
    print("\n[Init] Creating disaster environment...")
    env = DisasterEnvironment(seed=args.seed)

    # ── 2. Datasets ──────────────────────────────────────────────────────────
    print("[Init] Generating/loading datasets...")
    victim_df, area_df = save_datasets("datasets")

    # ── 3. ML System ─────────────────────────────────────────────────────────
    print("[Init] Training ML models...")
    ml_system = MLSystem()
    ml_results = ml_system.train_all(victim_df, area_df)

    # ── 4. Fuzzy System ──────────────────────────────────────────────────────
    print("[Init] Building fuzzy logic system...")
    fuzzy_system = FuzzyDisasterSystem()

    # ── 5. Replanner ─────────────────────────────────────────────────────────
    replanner = Replanner(env, fuzzy_system=fuzzy_system, ml_system=ml_system)

    # ── 6. Agent ─────────────────────────────────────────────────────────────
    agent = AIDRAAgent(env, fuzzy_system, ml_system, replanner, CSPSolver)

    # ── 7. Search Benchmark ──────────────────────────────────────────────────
    print("[Init] Running search algorithm benchmark...")
    search_results = benchmark_search(env)

    # ── 8. CSP Comparison ────────────────────────────────────────────────────
    print("[Init] Running CSP comparison...")
    csp_results = compare_csp(env.victims, env)
    print(f"  {csp_results['comparison']}")

    # ── 9. Local Search ──────────────────────────────────────────────────────
    print("[Init] Running local search optimization...")
    local_results = compare_local_search(env.victims, env)
    sa_result = local_results["simulated_annealing"]
    hc_result = local_results["hill_climbing"]
    print(f"  SA: {sa_result['explanation']}")
    print(f"  HC: {hc_result['explanation']}")
    print(f"  Winner: {local_results['winner']}")

    if args.benchmark:
        print("\n[Benchmark mode] Done.")
        return

    # ── 10. Run simulation ───────────────────────────────────────────────────
    if args.headless:
        run_headless(env, agent, ml_system, fuzzy_system, search_results,
                     csp_results, sa_result, hc_result, ml_results)
    else:
        try:
            import pygame
            run_pygame(env, agent, fuzzy_system, ml_system)
        except ImportError:
            print("[Main] Pygame not available — running headless mode")
            run_headless(env, agent, ml_system, fuzzy_system, search_results,
                         csp_results, sa_result, hc_result, ml_results)

    # ── 11. Generate charts AFTER simulation so KPIs reflect actual results ──
    print("\n[Main] Simulation complete. Generating final KPI charts...")

    # Re-run search benchmark with post-simulation environment state
    # (risk map may have changed due to fire spread and aftershocks)
    post_search_results = benchmark_search(env)

    # Re-run local search with updated victim conditions
    post_local = compare_local_search(env.victims, env)
    post_sa = post_local["simulated_annealing"]
    post_hc = post_local["hill_climbing"]

    generate_all_charts(
        env,
        post_search_results,    # uses post-simulation risk map
        ml_results,             # ML metrics don't change (fixed training set)
        csp_results,            # CSP comparison doesn't change
        post_sa,                # SA on post-simulation victim conditions
        post_hc                 # HC on post-simulation victim conditions
    )

    # Print final KPI summary to console
    kpis = env.get_kpis()
    print("\n[Results] Final KPIs:")
    print(f"  Victims saved:       {kpis['victims_saved']}/{kpis['victims_total']} ({kpis['victims_pct']:.0f}%)")
    print(f"  Avg rescue time:     {kpis['avg_rescue_time']:.2f}s")
    print(f"  Total risk exposure: {kpis['total_risk_exposure']:.2f}")
    print(f"  Replanning events:   {kpis['replanning_triggers']}")
    print(f"  Simulation steps:    {kpis['simulation_steps']}")
    print(f"  Charts saved to:     results/")

if __name__ == "__main__":
    main()
