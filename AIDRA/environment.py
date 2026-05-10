"""
environment.py
Core disaster environment model for AIDRA.
Manages grid state, victims, hazards, resources, and dynamic events.
"""

import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# Cell types
EMPTY       = 0
BLOCKED     = 1
FIRE_ZONE   = 2
HIGH_RISK   = 3
BASE        = 4
MEDICAL_CTR = 5
VICTIM      = 6

CELL_RISK = {
    EMPTY: 0,
    BLOCKED: 999,
    FIRE_ZONE: 10,
    HIGH_RISK: 5,
    BASE: 0,
    MEDICAL_CTR: 0,
    VICTIM: 1,
}

SEVERITY_LABELS = {0: "MINOR", 1: "MODERATE", 2: "CRITICAL"}
SEVERITY_COLORS  = {0: (0,200,0), 1: (255,165,0), 2: (220,20,60)}


@dataclass
class Victim:
    vid: int
    pos: Tuple[int,int]
    severity: int            # 0=minor,1=moderate,2=critical
    condition: float         # 0..1 (1=worst)
    rescued: bool = False
    rescue_time: Optional[float] = None
    assigned_ambulance: Optional[int] = None
    survival_prob: float = 1.0
    area_risk: float = 0.0


@dataclass
class Ambulance:
    aid: int
    pos: Tuple[int,int]
    base_pos: Tuple[int,int]
    capacity: int = 2
    passengers: List[int] = field(default_factory=list)
    route: List[Tuple[int,int]] = field(default_factory=list)
    route_step: int = 0
    busy: bool = False
    returning: bool = False
    trips: int = 0
    risk_exposure: float = 0.0


@dataclass
class DecisionLog:
    timestamp: float
    event: str
    reason: str
    previous_decision: str
    new_decision: str
    impact: str


class DisasterEnvironment:
    GRID_SIZE = 15

    def __init__(self, seed: int = None):
        if seed is None:
            seed = random.randint(0, 99999)
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.size = self.GRID_SIZE
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.risk_map = np.zeros((self.size, self.size), dtype=float)
        self.step_count = 0
        self.start_time = time.time()
        self.decision_log: List[DecisionLog] = []
        self.dynamic_events: List[str] = []
        self.victims: List[Victim] = []
        self.ambulances: List[Ambulance] = []
        self.medical_centers: List[Tuple[int,int]] = []
        self.base_pos: Tuple[int,int] = (0, 0)
        self.rescue_team_busy: bool = False
        self.rescue_team_pos: Optional[Tuple[int,int]] = None
        self.medical_kits: int = 10
        self.replanning_triggers: int = 0
        self.victims_saved: int = 0
        self.rescue_times: List[float] = []
        self._fuzzy = None          # set after construction by agent
        self._setup_environment()

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #
    def _setup_environment(self):
        g = self.grid
        occupied = set()

        def place(r, c, cell_type, risk=0.0):
            g[r][c] = cell_type
            self.risk_map[r][c] = risk
            occupied.add((r, c))

        def rand_free(exclude_border=False):
            """Pick a random unoccupied cell."""
            while True:
                r = random.randint(1, self.size - 2) if exclude_border else random.randint(0, self.size - 1)
                c = random.randint(1, self.size - 2) if exclude_border else random.randint(0, self.size - 1)
                if (r, c) not in occupied:
                    return r, c

        # Base — always top-left corner for consistency
        self.base_pos = (0, 0)
        place(0, 0, BASE)

        # 2 Medical centers — random positions on opposite sides of grid
        mc1 = (random.randint(0, self.size-1), self.size - 1)
        mc2 = (self.size - 1, random.randint(self.size//3, 2*self.size//3))
        self.medical_centers = [mc1, mc2]
        for mc in self.medical_centers:
            place(mc[0], mc[1], MEDICAL_CTR)

        # Fire zones — random cluster of 4-7 cells
        fire_seed_r, fire_seed_c = rand_free(exclude_border=True)
        fire_count = random.randint(4, 7)
        fire_frontier = [(fire_seed_r, fire_seed_c)]
        placed_fire = 0
        while fire_frontier and placed_fire < fire_count:
            r, c = fire_frontier.pop(random.randint(0, len(fire_frontier)-1))
            if (r, c) not in occupied and 0 < r < self.size-1 and 0 < c < self.size-1:
                place(r, c, FIRE_ZONE, risk=10.0)
                placed_fire += 1
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb = (r+dr, c+dc)
                    if nb not in occupied:
                        fire_frontier.append(nb)

        # High-risk zones — random cluster of 4-6 cells, different area
        hr_seed_r, hr_seed_c = rand_free(exclude_border=True)
        hr_count = random.randint(4, 6)
        hr_frontier = [(hr_seed_r, hr_seed_c)]
        placed_hr = 0
        while hr_frontier and placed_hr < hr_count:
            r, c = hr_frontier.pop(random.randint(0, len(hr_frontier)-1))
            if (r, c) not in occupied and 0 < r < self.size-1 and 0 < c < self.size-1:
                place(r, c, HIGH_RISK, risk=5.0)
                placed_hr += 1
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb = (r+dr, c+dc)
                    if nb not in occupied:
                        hr_frontier.append(nb)

        # Blocked roads — 6-10 random cells
        for _ in range(random.randint(6, 10)):
            r, c = rand_free()
            place(r, c, BLOCKED)

        # 5 Victims: 2 critical, 2 moderate, 1 minor — random positions
        severities  = [2, 2, 1, 1, 0]
        conditions  = [
            round(random.uniform(0.75, 0.95), 2),   # critical
            round(random.uniform(0.70, 0.90), 2),   # critical
            round(random.uniform(0.45, 0.70), 2),   # moderate
            round(random.uniform(0.40, 0.65), 2),   # moderate
            round(random.uniform(0.20, 0.45), 2),   # minor
        ]
        for i, (sev, cond) in enumerate(zip(severities, conditions)):
            r, c = rand_free()
            v = Victim(vid=i, pos=(r, c), severity=sev, condition=cond)
            v.area_risk = float(self.risk_map[r][c])
            self.victims.append(v)
            place(r, c, VICTIM)

        # Ambulances start at base
        for i in range(2):
            amb = Ambulance(aid=i, pos=self.base_pos, base_pos=self.base_pos)
            self.ambulances.append(amb)

        # Fill remaining risk map
        for r in range(self.size):
            for c in range(self.size):
                if self.risk_map[r][c] == 0:
                    self.risk_map[r][c] = CELL_RISK.get(g[r][c], 0)

    # ------------------------------------------------------------------ #
    #  NAVIGATION HELPERS                                                  #
    # ------------------------------------------------------------------ #
    def is_passable(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False
        return self.grid[r][c] != BLOCKED

    def neighbors(self, r: int, c: int) -> List[Tuple[int,int]]:
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.is_passable(nr, nc):
                result.append((nr, nc))
        return result

    def move_cost(self, r: int, c: int) -> float:
        base = 1.0
        risk = self.risk_map[r][c]
        return base + risk * 0.5

    def manhattan(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def path_cost(self, path: List[Tuple[int,int]]) -> float:
        return sum(self.move_cost(r,c) for r,c in path)

    def path_risk(self, path: List[Tuple[int,int]]) -> float:
        return sum(self.risk_map[r][c] for r,c in path)

    def nearest_medical_center(self, pos: Tuple[int,int]) -> Tuple[int,int]:
        return min(self.medical_centers, key=lambda m: self.manhattan(pos, m))

    # ------------------------------------------------------------------ #
    #  DYNAMIC EVENTS                                                      #
    # ------------------------------------------------------------------ #
    def step(self) -> List[str]:
        """Advance simulation by one step; return list of events."""
        self.step_count += 1
        events = []

        # Spread fire every 15 steps
        if self.step_count % 15 == 0:
            ev = self._spread_fire()
            if ev:
                events.append(ev)

        # Random road block every 20 steps
        if self.step_count % 20 == 0:
            ev = self._random_blockage()
            if ev:
                events.append(ev)

        # Aftershock risk increase every 25 steps
        if self.step_count % 25 == 0:
            ev = self._aftershock_event()
            if ev:
                events.append(ev)

        # Move ambulances along routes
        self._move_ambulances()

        # Deteriorate victim conditions
        if self.step_count % 5 == 0:
            self._deteriorate_victims()

        self.dynamic_events.extend(events)
        return events

    def _spread_fire(self) -> Optional[str]:
        fire_cells = [(r,c) for r in range(self.size) for c in range(self.size)
                      if self.grid[r][c] == FIRE_ZONE]
        if not fire_cells:
            return None
        src = random.choice(fire_cells)
        neighbors = self.neighbors(*src)
        if not neighbors:
            return None
        target = random.choice(neighbors)
        r, c = target
        if self.grid[r][c] == EMPTY:
            self.grid[r][c] = FIRE_ZONE
            self.risk_map[r][c] = 10.0
            return f"FIRE SPREAD to ({r},{c})"
        return None

    def _random_blockage(self) -> Optional[str]:
        candidates = [(r,c) for r in range(self.size) for c in range(self.size)
                      if self.grid[r][c] == EMPTY and (r,c) != self.base_pos
                      and (r,c) not in self.medical_centers]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        self.grid[r][c] = BLOCKED
        return f"ROAD BLOCKED at ({r},{c})"

    def _aftershock_event(self) -> Optional[str]:
        high_risk = [(r,c) for r in range(self.size) for c in range(self.size)
                     if self.grid[r][c] == HIGH_RISK]
        for r,c in high_risk:
            self.risk_map[r][c] = min(self.risk_map[r][c] + 1.0, 15.0)
        if high_risk:
            return "AFTERSHOCK: risk levels increased in high-risk zones"
        return None

    def _deteriorate_victims(self):
        for v in self.victims:
            if not v.rescued:
                v.condition = min(1.0, v.condition + 0.02)
                if v.severity == 2:
                    v.condition = min(1.0, v.condition + 0.01)

    # ------------------------------------------------------------------ #
    #  AMBULANCE MOVEMENT                                                  #
    # ------------------------------------------------------------------ #
    def _move_ambulances(self):
        for amb in self.ambulances:
            if amb.route and amb.route_step < len(amb.route):
                next_pos = amb.route[amb.route_step]
                r, c = next_pos
                # Check if path still valid
                if not self.is_passable(r, c):
                    amb.route = []
                    amb.route_step = 0
                    self._log_decision(
                        event="PATH INVALIDATED",
                        reason=f"Cell {next_pos} became impassable",
                        prev=f"Ambulance {amb.aid} following route",
                        new="Route cleared; replanning needed",
                        impact="Rescue delayed; replanning triggered"
                    )
                    self.replanning_triggers += 1
                    continue
                risk = self.risk_map[r][c]
                amb.risk_exposure += risk
                amb.pos = next_pos
                amb.route_step += 1

                # Pick up any victim at current cell mid-route (not just at route end)
                for v in self.victims:
                    if (not v.rescued
                            and v.pos == amb.pos
                            and len(amb.passengers) < amb.capacity
                            and (v.assigned_ambulance is None
                                 or v.assigned_ambulance == amb.aid)):
                        amb.passengers.append(v.vid)
                        v.assigned_ambulance = amb.aid
                        # Clear grid immediately
                        if self.grid[v.pos[0]][v.pos[1]] == VICTIM:
                            self.grid[v.pos[0]][v.pos[1]] = EMPTY
                        # If now carrying passengers, make sure route ends at medical center
                        if amb.passengers and amb.route:
                            dest = amb.route[-1]
                            if dest not in self.medical_centers:
                                mc = self.nearest_medical_center(amb.pos)
                                self._route_ambulance_to(amb, mc)

                # Mid-route opportunistic pickup check (nearby victims within radius)
                if len(amb.passengers) < amb.capacity:
                    self._opportunistic_pickup(amb)

                # Reached end of route
                if amb.route_step >= len(amb.route):
                    self._on_ambulance_arrived(amb)

    def _opportunistic_pickup(self, amb: Ambulance):
        """
        While moving, if a victim is on the current cell or adjacent (within 1 step)
        and ambulance is under capacity, reroute to grab them before continuing.
        Only triggers if the detour is <= 2 steps extra.
        """
        PICKUP_RADIUS = 2       # manhattan distance to consider
        DETOUR_LIMIT  = 2       # max extra steps to accept

        if len(amb.passengers) >= amb.capacity:
            return

        unrescued = [
            v for v in self.victims
            if not v.rescued
            and v.assigned_ambulance is None
            and self.manhattan(amb.pos, v.pos) <= PICKUP_RADIUS
        ]
        if not unrescued:
            return

        # Find the remaining route cost to current destination
        remaining_route = amb.route[amb.route_step:]
        if not remaining_route:
            return
        current_dest = remaining_route[-1]
        direct_remaining = len(remaining_route)

        # Check if detouring to any nearby victim is worth it
        best = None
        best_extra = float('inf')
        for v in unrescued:
            detour_cost = (self.manhattan(amb.pos, v.pos)
                           + self.manhattan(v.pos, current_dest))
            extra = detour_cost - direct_remaining
            if extra <= DETOUR_LIMIT and extra < best_extra:
                best_extra = extra
                best = v

        if best:
            best.assigned_ambulance = amb.aid
            # Reroute: current pos → victim → original destination
            from search.astar import astar
            to_victim = astar(self, amb.pos, best.pos, mode="balanced")
            if to_victim["found"] and len(to_victim["path"]) > 1:
                # After picking up victim, continue to original destination
                to_dest = astar(self, best.pos, current_dest, mode="balanced")
                if to_dest["found"]:
                    combined = to_victim["path"] + to_dest["path"][1:]
                    amb.route = combined
                    amb.route_step = 1
                    self._log_decision(
                        event=f"OPPORTUNISTIC PICKUP: Amb {amb.aid} detouring to V{best.vid}",
                        reason=f"V{best.vid} is {self.manhattan(amb.pos, best.pos)} cells away, detour={best_extra} steps",
                        prev=f"Heading to {current_dest}",
                        new=f"Rerouted via V{best.vid} then to {current_dest}",
                        impact=f"Capacity utilised: {len(amb.passengers)+1}/2 passengers"
                    )

    def _on_ambulance_arrived(self, amb: Ambulance):
        pos = amb.pos
        # Note: mid-route pickups already handled in _move_ambulances every step.
        # On arrival, do a final pickup check for the destination cell itself.
        for v in self.victims:
            if (not v.rescued
                    and v.pos == pos
                    and len(amb.passengers) < amb.capacity
                    and (v.assigned_ambulance is None
                         or v.assigned_ambulance == amb.aid)):
                amb.passengers.append(v.vid)
                v.assigned_ambulance = amb.aid
                if self.grid[v.pos[0]][v.pos[1]] == VICTIM:
                    self.grid[v.pos[0]][v.pos[1]] = EMPTY

        # At medical center with passengers → rescue and dispatch again
        if pos in self.medical_centers and amb.passengers:
            for vid in amb.passengers:
                v = self.victims[vid]
                v.rescued = True
                v.rescue_time = time.time() - self.start_time
                self.rescue_times.append(v.rescue_time)
                self.victims_saved += 1
                v.assigned_ambulance = None      # clear assignment after rescue
                # Clear grid cell (should already be cleared at pickup, safety net)
                if self.grid[v.pos[0]][v.pos[1]] == VICTIM:
                    self.grid[v.pos[0]][v.pos[1]] = EMPTY
            amb.passengers = []
            amb.trips += 1
            amb.returning = False
            amb.busy = False
            self._dispatch_ambulance(amb)
            return

        # Has passengers but not at medical center → route there
        if amb.passengers:
            mc = self.nearest_medical_center(pos)
            self._route_ambulance_to(amb, mc)
        else:
            self._dispatch_ambulance(amb)

    def _dispatch_ambulance(self, amb: Ambulance):
        """
        Send ambulance to nearest unrescued victim.
        If already carrying 1 passenger, check if a second victim lies
        within DETOUR_THRESHOLD steps of the route to medical center —
        if so, pick them up first (capacity utilisation).
        """
        DETOUR_THRESHOLD = 4    # max extra steps acceptable for a detour pickup

        unrescued = [
            v for v in self.victims
            if not v.rescued and v.assigned_ambulance is None
        ]

        if not unrescued:
            if amb.pos != amb.base_pos:
                self._route_ambulance_to(amb, amb.base_pos)
            else:
                amb.busy = False
            return

        if len(amb.passengers) >= amb.capacity:
            # Already full — go straight to medical center
            mc = self.nearest_medical_center(amb.pos)
            self._route_ambulance_to(amb, mc)
            return

        if amb.passengers:
            # Carrying 1 — check if a second victim is near the path to medical center
            mc = self.nearest_medical_center(amb.pos)
            direct_dist = self.manhattan(amb.pos, mc)
            best_detour = None
            best_cost   = float('inf')
            for v in unrescued:
                # Cost: amb→victim + victim→mc vs direct amb→mc
                detour_cost = self.manhattan(amb.pos, v.pos) + self.manhattan(v.pos, mc)
                extra_steps = detour_cost - direct_dist
                if extra_steps <= DETOUR_THRESHOLD:
                    if detour_cost < best_cost:
                        best_cost   = detour_cost
                        best_detour = v
            if best_detour:
                # Pick up second victim on the way
                best_detour.assigned_ambulance = amb.aid
                self._route_ambulance_to(amb, best_detour.pos)
                return
            # No worthwhile detour — go straight to medical center
            self._route_ambulance_to(amb, mc)
            return

        # No passengers — go to nearest victim
        target = min(unrescued, key=lambda v: self.manhattan(amb.pos, v.pos))
        target.assigned_ambulance = amb.aid
        self._route_ambulance_to(amb, target.pos)

    def _route_ambulance_to(self, amb: Ambulance, goal: Tuple[int, int]):
        """Plan route using algorithm selected by fuzzy risk at current position."""
        from search.astar import astar
        r, c = amb.pos
        fire_val = float(self.risk_map[r][c])

        # Let fuzzy system decide routing mode
        if hasattr(self, '_fuzzy') and self._fuzzy:
            fuzzy_out = self._fuzzy.evaluate(
                fire_intensity=fire_val,
                road_stability=max(0.01, 1.0 - fire_val / 10),
                aftershock_prob=0.3,
                victim_condition=0.5
            )
            if fuzzy_out["avoid_route"]:
                mode = "safe"
            elif fuzzy_out["route_danger"] < 3.0:
                mode = "speed"
            else:
                mode = "balanced"
        else:
            mode = "balanced"

        result = astar(self, amb.pos, goal, mode=mode)
        if result["found"] and len(result["path"]) > 1:
            amb.route = result["path"]
            amb.route_step = 1
            amb.busy = True
            amb.last_algo = f"A* ({mode})"
        else:
            amb.busy = False

    # ------------------------------------------------------------------ #
    #  LOGGING                                                             #
    # ------------------------------------------------------------------ #
    def _log_decision(self, event, reason, prev, new, impact):
        log = DecisionLog(
            timestamp=time.time()-self.start_time,
            event=event, reason=reason,
            previous_decision=prev,
            new_decision=new,
            impact=impact
        )
        self.decision_log.append(log)

    def log_decision(self, event, reason, prev, new, impact):
        self._log_decision(event, reason, prev, new, impact)

    # ------------------------------------------------------------------ #
    #  METRICS                                                             #
    # ------------------------------------------------------------------ #
    def get_kpis(self) -> Dict:
        total = len(self.victims)
        saved = self.victims_saved
        avg_rescue = np.mean(self.rescue_times) if self.rescue_times else 0
        kits_used = 10 - self.medical_kits
        amb_util = sum(a.trips for a in self.ambulances) / max(1, sum(a.trips+1 for a in self.ambulances))
        risk_exp = sum(a.risk_exposure for a in self.ambulances)
        return {
            "victims_total": total,
            "victims_saved": saved,
            "victims_pct": saved/total*100 if total else 0,
            "avg_rescue_time": avg_rescue,
            "medical_kits_used": kits_used,
            "ambulance_utilization": amb_util,
            "total_risk_exposure": risk_exp,
            "replanning_triggers": self.replanning_triggers,
            "simulation_steps": self.step_count,
        }
