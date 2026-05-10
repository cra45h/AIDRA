"""
visualization/pygame_vis.py
Pygame-based real-time disaster simulation visualizer for AIDRA.
Professional, color-coded grid with live stats panel.
"""

import pygame
import sys
import math
from typing import List, Tuple, Optional, Dict

# Colors
BLACK       = (10,  10,  10)
WHITE       = (240, 240, 240)
DARK_GRAY   = (50,  50,  50)
LIGHT_GRAY  = (180, 180, 180)
RED         = (220, 40,  40)
ORANGE      = (255, 140, 0)
YELLOW      = (255, 210, 0)
GREEN       = (34,  180, 34)
DARK_GREEN  = (20,  120, 20)
BLUE        = (30,  120, 220)
LIGHT_BLUE  = (100, 180, 255)
CYAN        = (0,   200, 220)
PURPLE      = (140, 0,   200)
PINK        = (255, 100, 180)
TEAL        = (0,   180, 160)
DARK_RED    = (150, 0,   0)
CREAM       = (255, 255, 220)

# Cell type colors
CELL_COLORS = {
    0: (40,  45,  50),    # EMPTY
    1: (80,  80,  80),    # BLOCKED
    2: (200, 60,  20),    # FIRE_ZONE
    3: (180, 100, 0),     # HIGH_RISK
    4: (30,  120, 30),    # BASE
    5: (0,   100, 200),   # MEDICAL_CENTER
    6: (220, 200, 0),     # VICTIM
}

CELL_LABELS = {
    0: "", 1: "X", 2: "🔥", 3: "⚠", 4: "🏠", 5: "+", 6: "V"
}


class AIDRAVisualizer:
    GRID_PIXEL = 38        # pixels per cell
    PANEL_W    = 540       # right side panel width
    HEADER_H   = 50        # top header

    def __init__(self, env, algorithm: str = "A*"):
        self.env = env
        self.algorithm = algorithm
        self.running = False
        self.paused = False
        self.selected_algo = algorithm
        self.explored_nodes: List[Tuple] = []
        self.current_path: List[Tuple] = []
        self.info_lines: List[str] = []
        self.step_count = 0
        self.events_log: List[str] = []
        self.fuzzy_output: Optional[Dict] = None
        self.ml_output: Optional[Dict] = None

        size = env.GRID_SIZE
        self.GRID_W = size * self.GRID_PIXEL
        self.GRID_H = size * self.GRID_PIXEL
        self.WIDTH  = self.GRID_W + self.PANEL_W
        self.HEIGHT = self.GRID_H + self.HEADER_H + 30

    def init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AIDRA — Adaptive Intelligent Disaster Response Agent")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_md  = pygame.font.SysFont("Arial", 13)
        self.font_sm  = pygame.font.SysFont("Arial", 11)
        self.font_xl  = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ico = pygame.font.SysFont("Segoe UI Emoji", 18)

    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = c * self.GRID_PIXEL
        y = r * self.GRID_PIXEL + self.HEADER_H
        return pygame.Rect(x, y, self.GRID_PIXEL-1, self.GRID_PIXEL-1)

    def draw_grid(self):
        env = self.env
        for r in range(env.size):
            for c in range(env.size):
                cell = env.grid[r][c]
                rect = self._cell_rect(r, c)
                color = CELL_COLORS.get(cell, DARK_GRAY)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (70, 70, 70), rect, 1)

                # Draw cell label
                label = CELL_LABELS.get(cell, "")
                if label and label not in ["🔥", "⚠"]:
                    surf = self.font_md.render(label, True, WHITE)
                    self.screen.blit(surf,
                        (rect.x + rect.w//2 - surf.get_width()//2,
                         rect.y + rect.h//2 - surf.get_height()//2))

                # Risk overlay (light tint)
                risk = env.risk_map[r][c]
                if risk > 0 and cell not in [1, 2, 4, 5]:
                    alpha = min(int(risk / 15.0 * 80), 80)
                    s = pygame.Surface((self.GRID_PIXEL-1, self.GRID_PIXEL-1), pygame.SRCALPHA)
                    s.fill((255, 80, 0, alpha))
                    self.screen.blit(s, (rect.x, rect.y))

    def draw_victims(self):
        for v in self.env.victims:
            if v.rescued:
                continue          # already removed from grid, don't draw
            r, c = v.pos
            rect = self._cell_rect(r, c)
            sev = v.severity
            color = [(0,200,0),(255,165,0),(220,40,40)][sev]

            pygame.draw.circle(self.screen, color,
                               (rect.x + rect.w//2, rect.y + rect.h//2), 10)
            pygame.draw.circle(self.screen, WHITE,
                               (rect.x + rect.w//2, rect.y + rect.h//2), 10, 2)
            surf = self.font_sm.render(str(v.vid), True, BLACK)
            self.screen.blit(surf,
                (rect.x + rect.w//2 - surf.get_width()//2,
                 rect.y + rect.h//2 - surf.get_height()//2))

    def draw_ambulances(self):
        for amb in self.env.ambulances:
            r, c = amb.pos
            rect = self._cell_rect(r, c)
            color = CYAN if amb.aid == 0 else PINK
            pygame.draw.rect(self.screen, color,
                             (rect.x+4, rect.y+4, rect.w-8, rect.h-8), border_radius=4)
            surf = self.font_sm.render(f"A{amb.aid}", True, BLACK)
            self.screen.blit(surf,
                (rect.x + rect.w//2 - surf.get_width()//2,
                 rect.y + rect.h//2 - surf.get_height()//2))

    def draw_path(self):
        """Draw active routes for ALL ambulances."""
        colors = [(0, 255, 150), (255, 180, 0)]   # Amb0=green, Amb1=orange
        for amb in self.env.ambulances:
            route_remaining = amb.route[amb.route_step:]
            if len(route_remaining) < 2:
                continue
            points = []
            for r, c in route_remaining:
                rect = self._cell_rect(r, c)
                points.append((rect.x + rect.w//2, rect.y + rect.h//2))
            color = colors[amb.aid % len(colors)]
            pygame.draw.lines(self.screen, color, False, points, 3)
            pygame.draw.circle(self.screen, color, points[-1], 6)

    def draw_header(self):
        """Top header bar."""
        pygame.draw.rect(self.screen, (20, 20, 40), (0, 0, self.WIDTH, self.HEADER_H))
        title = self.font_xl.render("AIDRA — Adaptive Intelligent Disaster Response Agent",
                                    True, (200, 220, 255))
        self.screen.blit(title, (10, 14))

        step_surf = self.font_md.render(f"Step: {self.env.step_count}  |  "
                                         f"Saved: {self.env.victims_saved}/5  |  "
                                         f"Algo: {self.selected_algo}  |  "
                                         f"[SPACE] Pause  [R] Reset  [Q] Quit",
                                        True, LIGHT_GRAY)
        self.screen.blit(step_surf, (10, self.HEADER_H - 16))

    def draw_panel(self):
        """Right side info panel."""
        px = self.GRID_W + 5
        pw = self.PANEL_W - 10

        # Background
        pygame.draw.rect(self.screen, (20, 25, 35),
                         (self.GRID_W, 0, self.PANEL_W, self.HEIGHT))
        pygame.draw.line(self.screen, (80, 80, 120),
                         (self.GRID_W, 0), (self.GRID_W, self.HEIGHT), 2)

        y = self.HEADER_H + 8

        def section(title, color=(160, 200, 255)):
            nonlocal y
            surf = self.font_lg.render(title, True, color)
            self.screen.blit(surf, (px, y))
            y += 20
            pygame.draw.line(self.screen, (60, 70, 90),
                             (px, y-2), (px + pw, y-2), 1)

        def line(text, color=LIGHT_GRAY, indent=0):
            nonlocal y
            surf = self.font_sm.render(text, True, color)
            self.screen.blit(surf, (px + indent, y))
            y += 15

        # VICTIMS
        section("VICTIMS")
        sev_label = ["Minor", "Moderate", "Critical"]
        sev_color = [(100,200,100),(255,165,0),(220,60,60)]
        for v in self.env.victims:
            status = "RESCUED ✓" if v.rescued else f"cond={v.condition:.2f}"
            c = sev_color[v.severity]
            line(f"V{v.vid} [{sev_label[v.severity]}] {status}",
                 color=c if not v.rescued else GREEN)
        y += 5

        # AMBULANCES
        section("AMBULANCES")
        for amb in self.env.ambulances:
            col = CYAN if amb.aid == 0 else PINK
            line(f"Amb {amb.aid}: pos={amb.pos}  trips={amb.trips}", color=col)
            line(f"  passengers={amb.passengers}  risk={amb.risk_exposure:.1f}",
                 color=LIGHT_GRAY, indent=8)
        y += 5

        # RESOURCES
        section("RESOURCES")
        line(f"Medical Kits: {self.env.medical_kits}/10")
        line(f"Rescue Team: {'BUSY' if self.env.rescue_team_busy else 'Available'}")
        line(f"Replanning: {self.env.replanning_triggers}")
        y += 5

        # SEARCH INFO
        section("SEARCH INFO")
        for info in self.info_lines[-6:]:
            line(info, color=(200, 230, 200))
        y += 5

        # FUZZY OUTPUT
        if self.fuzzy_output:
            section("FUZZY ASSESSMENT", (255, 200, 100))
            fo = self.fuzzy_output
            line(f"Route Danger: {fo.get('route_danger', 0):.1f}/10",
                 color=RED if fo.get('avoid_route') else GREEN)
            line(f"Rescue Urgency: {fo.get('rescue_urgency', 0):.1f}/10",
                 color=ORANGE if fo.get('prioritize_victim') else GREEN)
            line(f"Overall Risk: {fo.get('overall_risk', 0):.1f}/10")
            if fo.get('replan_recommended'):
                line("⚡ REPLAN RECOMMENDED", color=YELLOW)
            y += 5

        # ML OUTPUT
        if self.ml_output:
            section("ML ASSESSMENT", (180, 140, 255))
            mo = self.ml_output
            line(f"Risk Class: {mo.get('risk_label','?')}", color=PURPLE)
            line(f"Survival Prob: {mo.get('survival_probability', 0):.2f}")
            line(f"Priority: {mo.get('priority', '?')[:30]}", color=ORANGE)
            y += 5

        # EVENTS LOG
        section("RECENT EVENTS", (255, 140, 100))
        for ev in self.events_log[-5:]:
            line(ev[:38], color=(255, 160, 80))

        # LEGEND
        
        legend_offset_x = 350   # move everything right
        
        # temporarily shift panel x-position
        original_px = px
        px += legend_offset_x
        
        y = self.HEIGHT - 200
        
        section("LEGEND", (180, 180, 180))
        
        legend_items = [
            ((200, 60, 20), "Fire Zone"),
            ((180, 100, 0), "High Risk"),
            ((80, 80, 80), "Blocked Road"),
            ((30, 120, 30), "Base"),
            ((0, 100, 200), "Medical Center"),
            ((0, 180, 120), "Path"),
            ((40, 80, 100), "Explored"),
        ]
        
        for color, label in legend_items:
        
            pygame.draw.rect(
                self.screen,
                color,
                (px, y, 12, 12)
            )
        
            surf = self.font_sm.render(label, True, LIGHT_GRAY)
        
            self.screen.blit(
                surf,
                (px + 16, y)
            )
        
            y += 14
        
        # restore original px
        px = original_px
            
    def draw_buttons(self):
        """Draw control buttons at bottom."""
        y = self.HEIGHT - 28
        buttons = [
            ("START", (30, y, 70, 22), (0, 160, 80)),
            ("PAUSE", (110, y, 70, 22), (200, 140, 0)),
            ("RESET", (190, y, 70, 22), (180, 0, 0)),
        ]
        for label, rect, color in buttons:
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            surf = self.font_sm.render(label, True, WHITE)
            self.screen.blit(surf, (rect[0] + rect[2]//2 - surf.get_width()//2,
                                    rect[1] + 4))

    def render(self):
        self.screen.fill(DARK_GRAY)
        self.draw_header()
        self.draw_grid()
        self.draw_path()
        self.draw_victims()
        self.draw_ambulances()
        self.draw_panel()
        self.draw_buttons()
        pygame.display.flip()

    def handle_events(self) -> str:
        """Process pygame events. Returns: 'quit','reset','pause',or ''."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return 'quit'
                if event.key == pygame.K_r:
                    return 'reset'
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    return 'pause'
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Check button clicks
                if 30 <= x <= 100 and self.HEIGHT-28 <= y <= self.HEIGHT-6:
                    self.paused = False
                    return 'start'
                if 110 <= x <= 180 and self.HEIGHT-28 <= y <= self.HEIGHT-6:
                    self.paused = not self.paused
                    return 'pause'
                if 190 <= x <= 260 and self.HEIGHT-28 <= y <= self.HEIGHT-6:
                    return 'reset'
        return ''

    def set_search_info(self, info: List[str]):
        self.info_lines = info

    def set_path(self, path: List[Tuple], explored: List[Tuple] = None):
        self.current_path = path or []
        self.explored_nodes = set(explored or [])

    def add_event(self, event: str):
        self.events_log.append(event)
        if len(self.events_log) > 20:
            self.events_log.pop(0)

    def set_fuzzy(self, output: Dict):
        self.fuzzy_output = output

    def set_ml(self, output: Dict):
        self.ml_output = output

    def tick(self, fps: int = 10):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
