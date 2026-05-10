"""
search/astar.py
A* Search with custom weighted cost function for AIDRA.

f(n) = g(n) + h(n)
where:
  g(n) = distance + risk_weight + hazard_penalty
  h(n) = Manhattan distance to goal
"""

import heapq
from typing import Tuple, Dict
import time


def astar(env,
          start: Tuple[int,int],
          goal: Tuple[int,int],
          risk_weight: float = 1.0,
          hazard_penalty: float = 5.0,
          mode: str = "balanced") -> Dict:
    """
    A* with custom cost function.

    mode:
      "speed"   -> risk_weight=0.1, hazard_penalty=1.0   (prefer fast path)
      "safe"    -> risk_weight=3.0, hazard_penalty=15.0  (prefer safe path)
      "balanced"-> risk_weight=1.0, hazard_penalty=5.0   (trade-off)

    Returns path, metrics, and explanation of trade-off.
    """
    if mode == "speed":
        risk_weight = 0.1
        hazard_penalty = 1.0
        mode_desc = "SPEED mode: minimise rescue time, higher risk accepted"
    elif mode == "safe":
        risk_weight = 3.0
        hazard_penalty = 15.0
        mode_desc = "SAFE mode: minimise risk exposure, longer path accepted"
    else:
        mode_desc = "BALANCED mode: trade-off between time and risk"

    t0 = time.perf_counter()

    def g_cost(r: int, c: int, dist: float) -> float:
        base = dist + 1.0
        risk = env.risk_map[r][c] * risk_weight
        hazard = hazard_penalty if env.risk_map[r][c] >= 8.0 else 0.0
        return base + risk + hazard

    # heap: (f, g, pos, path)
    g_start = 0.0
    h_start = env.manhattan(start, goal)
    heap = [(g_start + h_start, g_start, start, [start])]
    best_g = {start: g_start}
    nodes_expanded = 0
    max_frontier = 1
    explored_nodes = []

    while heap:
        max_frontier = max(max_frontier, len(heap))
        f, g, pos, path = heapq.heappop(heap)
        nodes_expanded += 1
        explored_nodes.append(pos)

        if pos == goal:
            elapsed = time.perf_counter() - t0
            cost = env.path_cost(path)
            risk_total = env.path_risk(path)
            return {
                "algorithm": f"A* ({mode})",
                "path": path,
                "path_cost": cost,
                "path_risk": risk_total,
                "nodes_expanded": nodes_expanded,
                "frontier_size": max_frontier,
                "runtime": elapsed,
                "explored": explored_nodes,
                "found": True,
                "mode": mode,
                "explanation": (
                    f"{mode_desc}. "
                    f"Path length: {len(path)-1} steps, cost: {cost:.2f}, risk: {risk_total:.2f}. "
                    f"Expanded {nodes_expanded} nodes in {elapsed*1000:.2f}ms. "
                    f"A* selected this route because f(n)=distance+risk_weight*risk+hazard_penalty "
                    f"with risk_weight={risk_weight} and hazard_penalty={hazard_penalty}."
                )
            }

        if g > best_g.get(pos, float('inf')):
            continue

        for nb in env.neighbors(*pos):
            nr, nc = nb
            g_new = g + g_cost(nr, nc, 1.0)
            if g_new < best_g.get(nb, float('inf')):
                best_g[nb] = g_new
                h_nb = env.manhattan(nb, goal)
                heapq.heappush(heap, (g_new + h_nb, g_new, nb, path + [nb]))

    elapsed = time.perf_counter() - t0
    return {
        "algorithm": f"A* ({mode})",
        "path": [],
        "path_cost": float('inf'),
        "path_risk": float('inf'),
        "nodes_expanded": nodes_expanded,
        "frontier_size": max_frontier,
        "runtime": elapsed,
        "explored": explored_nodes,
        "found": False,
        "mode": mode,
        "explanation": f"A* ({mode_desc}) found no path to goal."
    }


def astar_compare(env, start: Tuple[int,int], goal: Tuple[int,int]) -> Dict:
    """Run A* in both speed and safe modes; return comparison."""
    speed_result = astar(env, start, goal, mode="speed")
    safe_result  = astar(env, start, goal, mode="safe")

    if speed_result["found"] and safe_result["found"]:
        speed_path_len = len(speed_result["path"])
        safe_path_len  = len(safe_result["path"])
        speed_risk = speed_result.get("path_risk", 0)
        safe_risk  = safe_result.get("path_risk", 0)
        tradeoff = (
            f"TRADE-OFF ANALYSIS: "
            f"Speed path: {speed_path_len-1} steps, risk={speed_risk:.1f}. "
            f"Safe path: {safe_path_len-1} steps, risk={safe_risk:.1f}. "
            f"Safe path is {safe_path_len-speed_path_len} steps longer "
            f"but reduces risk by {speed_risk-safe_risk:.1f} units."
        )
    else:
        tradeoff = "Could not complete trade-off comparison (no path found)."

    return {
        "speed": speed_result,
        "safe": safe_result,
        "tradeoff_analysis": tradeoff
    }
