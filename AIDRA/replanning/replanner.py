"""
replanning/replanner.py
Dynamic replanning engine for AIDRA.

Monitors environment for changes, invalidates routes,
triggers replanning, and logs every decision.
"""

from typing import List, Dict, Optional, Tuple
from search.astar import astar
from search.bfs import bfs
import time


class Replanner:
    """
    Monitors events, validates routes, and triggers replanning.
    Maintains a full decision log.
    """

    def __init__(self, env, fuzzy_system=None, ml_system=None):
        self.env = env
        self.fuzzy = fuzzy_system
        self.ml = ml_system
        self.replan_count = 0
        self.decision_log: List[Dict] = []
        self.current_routes: Dict[int, List[Tuple]] = {}  # amb_id -> route

    # ------------------------------------------------------------------ #
    #  ROUTE VALIDATION                                                    #
    # ------------------------------------------------------------------ #
    def validate_route(self, amb_id: int, route: List[Tuple]) -> bool:
        """Check if entire route is still passable."""
        for pos in route:
            r, c = pos
            if not self.env.is_passable(r, c):
                return False
        return True

    def check_all_routes(self) -> List[int]:
        """Return list of ambulance IDs whose routes are invalid."""
        invalid = []
        for amb in self.env.ambulances:
            if amb.route:
                if not self.validate_route(amb.aid, amb.route):
                    invalid.append(amb.aid)
        return invalid

    # ------------------------------------------------------------------ #
    #  REPLANNING                                                          #
    # ------------------------------------------------------------------ #
    def replan_ambulance(self, amb_id: int, reason: str,
                          algorithm: str = "astar") -> Optional[List[Tuple]]:
        """
        Replan route for ambulance. Uses A* by default (best for risk-aware routing).
        """
        amb = self.env.ambulances[amb_id]
        start = amb.pos

        # Determine target: first unrescued passenger pickup or medical center
        if amb.passengers:
            goal = self.env.nearest_medical_center(start)
            goal_desc = f"medical center at {goal}"
        else:
            # Find nearest unrescued victim
            unrescued = [v for v in self.env.victims if not v.rescued and
                         v.assigned_ambulance is None or v.assigned_ambulance == amb_id]
            if not unrescued:
                return []
            victim = min(unrescued, key=lambda v: self.env.manhattan(start, v.pos))
            goal = victim.pos
            goal_desc = f"victim {victim.vid} at {goal}"

        prev_route = list(amb.route)

        # Use fuzzy to decide mode
        mode = "balanced"
        if self.fuzzy:
            r, c = start
            fire_val = float(self.env.risk_map[r][c])
            fuzzy_out = self.fuzzy.evaluate(
                fire_intensity=fire_val,
                road_stability=0.5,
                aftershock_prob=0.3,
                victim_condition=0.5
            )
            if fuzzy_out["avoid_route"]:
                mode = "safe"
            elif fuzzy_out["route_danger"] < 3:
                mode = "speed"

        if algorithm == "astar":
            result = astar(self.env, start, goal, mode=mode)
        else:
            result = bfs(self.env, start, goal)

        new_route = result["path"] if result["found"] else []

        # Log the replanning decision
        log_entry = {
            "timestamp": round(time.time() - self.env.start_time, 2),
            "event": f"REPLANNING triggered for Ambulance {amb_id}",
            "reason": reason,
            "previous_decision": f"Following route of {len(prev_route)} steps",
            "new_decision": (
                f"A* ({mode}) planned {len(new_route)} steps to {goal_desc}"
                if new_route else "No path found — ambulance halted"
            ),
            "algorithm_used": f"{algorithm} ({mode})",
            "impact": (
                f"Route changed from {len(prev_route)} to {len(new_route)} steps. "
                f"Trade-off: {mode} mode selected based on fuzzy risk assessment."
                if new_route else "Mission incomplete due to no valid path."
            ),
            "tradeoff": (
                f"MODE={mode}: "
                + ("Prioritizing SAFETY over speed" if mode == "safe"
                   else "Prioritizing SPEED over safety" if mode == "speed"
                   else "Balancing time and risk")
            )
        }
        self.decision_log.append(log_entry)
        self.env.log_decision(
            event=log_entry["event"],
            reason=log_entry["reason"],
            prev=log_entry["previous_decision"],
            new=log_entry["new_decision"],
            impact=log_entry["impact"]
        )

        # Update ambulance route
        if new_route:
            amb.route = new_route
            amb.route_step = 0
            self.current_routes[amb_id] = new_route
            self.replan_count += 1

        return new_route

    # ------------------------------------------------------------------ #
    #  PROCESS EVENTS                                                      #
    # ------------------------------------------------------------------ #
    def process_events(self, events: List[str]) -> List[Dict]:
        """Process dynamic events and trigger replanning as needed."""
        responses = []

        if not events:
            return responses

        # Check which ambulances need replanning
        invalid_ambs = self.check_all_routes()

        for amb_id in invalid_ambs:
            reason = "; ".join(events)
            new_route = self.replan_ambulance(amb_id, reason=reason)
            responses.append({
                "ambulance": amb_id,
                "triggered_by": events,
                "new_route_length": len(new_route) if new_route else 0
            })
            self.env.replanning_triggers += 1

        # Log environmental events
        for event in events:
            if "FIRE" in event:
                self._handle_fire_event(event)
            elif "BLOCKED" in event:
                self._handle_blockage_event(event)
            elif "AFTERSHOCK" in event:
                self._handle_aftershock_event()

        return responses

    def _handle_fire_event(self, event: str):
        self.decision_log.append({
            "timestamp": round(time.time() - self.env.start_time, 2),
            "event": event,
            "reason": "Fire spread detected",
            "previous_decision": "Original route maintained",
            "new_decision": "Routes re-evaluated; avoid fire zones",
            "impact": "Risk map updated; A* will route around new fire cell",
            "tradeoff": "Safety prioritized — longer path to avoid fire"
        })

    def _handle_blockage_event(self, event: str):
        self.decision_log.append({
            "timestamp": round(time.time() - self.env.start_time, 2),
            "event": event,
            "reason": "Road blocked by aftershock/debris",
            "previous_decision": "Route through blocked road",
            "new_decision": "Recompute path avoiding blocked cell",
            "impact": "Rescue time increases; new path found via A*",
            "tradeoff": "No choice — blocked roads force detour"
        })

    def _handle_aftershock_event(self):
        self.decision_log.append({
            "timestamp": round(time.time() - self.env.start_time, 2),
            "event": "AFTERSHOCK DETECTED",
            "reason": "Seismic activity increased risk levels",
            "previous_decision": "Low-risk routing",
            "new_decision": "Switch to SAFE mode routing",
            "impact": "Longer paths chosen to minimize risk exposure",
            "tradeoff": "TIME sacrificed to minimize RISK"
        })

    # ------------------------------------------------------------------ #
    #  METRICS                                                             #
    # ------------------------------------------------------------------ #
    def get_summary(self) -> Dict:
        return {
            "total_replannings": self.replan_count,
            "decision_log_entries": len(self.decision_log),
            "log": self.decision_log
        }

    def print_log(self):
        print("\n" + "="*60)
        print("DECISION LOG")
        print("="*60)
        for entry in self.decision_log[-10:]:  # last 10
            print(f"[{entry['timestamp']:.1f}s] {entry['event']}")
            print(f"  Reason: {entry['reason']}")
            print(f"  New Decision: {entry['new_decision']}")
            print(f"  Trade-off: {entry['tradeoff']}")
            print()
