import random
import time
import os

# ═══════════════════════════════════════════════════════════
#  CAMPUS INTERSECTION JAYWALKING SIMULATION
#  Visuals: box-drawing grid, lane dividers, vehicles
#  Logic:   time-of-day cycles, bursty arrivals,
#           social contagion, light-aware pedestrians
# ═══════════════════════════════════════════════════════════

# ─────────────────────────────────────────
#  Grid & intersection geometry
# ─────────────────────────────────────────
WIDTH  = 57
HEIGHT = 29

MID_X = WIDTH  // 2   # 28
MID_Y = HEIGHT // 2   # 14

ROAD_HW = 4   # half-width  of vertical   road arm
ROAD_HH = 2   # half-height of horizontal road arm

LEFT_EDGE  = MID_X - ROAD_HW   # 24
RIGHT_EDGE = MID_X + ROAD_HW   # 32
TOP_EDGE   = MID_Y - ROAD_HH   # 12
BOT_EDGE   = MID_Y + ROAD_HH   # 16

CURB_LEFT  = LEFT_EDGE  - 1    # 23
CURB_RIGHT = RIGHT_EDGE + 1    # 33
CURB_TOP   = TOP_EDGE   - 1    # 11
CURB_BOT   = BOT_EDGE   + 1    # 17

# Pedestrian path offsets
CW_OFFSET = 2   # left-of-centre for legal crossers
JW_OFFSET = 3   # further off-centre for jaywalkers

# Vehicle lane offset from centreline
LANE = 1

# Animation speed
FRAME_DELAY = 0.10

# ─────────────────────────────────────────
#  Time-of-day cycle definitions
#  (from the other simulation's model)
# ─────────────────────────────────────────
TIME_CYCLES = [
    {
        "name":          "Cycle 1",
        "time":          "8:50 AM",
        "label":         "Morning class change",
        "burst_chance":  0.85,
        "arrival_range": (14, 22),
        "base_jaywalk":  0.08,
        "light_seq":     [("GREEN_NS", 14), ("GREEN_EW", 14),
                          ("GREEN_NS", 14), ("GREEN_EW", 14)],
    },
    {
        "name":          "Cycle 2",
        "time":          "11:00 AM",
        "label":         "Late morning",
        "burst_chance":  0.65,
        "arrival_range": (10, 17),
        "base_jaywalk":  0.10,
        "light_seq":     [("GREEN_NS", 12), ("GREEN_EW", 12),
                          ("GREEN_NS", 12), ("GREEN_EW", 12)],
    },
    {
        "name":          "Cycle 3",
        "time":          "2:30 PM",
        "label":         "Afternoon",
        "burst_chance":  0.40,
        "arrival_range": (6, 12),
        "base_jaywalk":  0.11,
        "light_seq":     [("GREEN_NS", 12), ("GREEN_EW", 12),
                          ("GREEN_NS", 12), ("GREEN_EW", 12)],
    },
    {
        "name":          "Cycle 4",
        "time":          "5:10 PM",
        "label":         "Late afternoon",
        "burst_chance":  0.18,
        "arrival_range": (2, 7),
        "base_jaywalk":  0.14,
        "light_seq":     [("GREEN_NS", 10), ("GREEN_EW", 10),
                          ("GREEN_NS", 10), ("GREEN_EW", 10)],
    },
]

# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def lerp_path(waypoints, num_steps):
    if len(waypoints) < 2:
        return list(waypoints) * num_steps
    segs = len(waypoints) - 1
    steps_per = max(1, num_steps // segs)
    pts = []
    for i in range(segs):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        for s in range(steps_per):
            t = s / steps_per
            pts.append((round(x0 + (x1 - x0) * t),
                        round(y0 + (y1 - y0) * t)))
    pts.append(waypoints[-1])
    return pts


def social_probability(base_prob, jaywalkers_so_far, seen_so_far):
    """Jaywalking becomes more likely the more others have done it."""
    if seen_so_far == 0:
        return base_prob
    first_bonus = 0.12 if jaywalkers_so_far > 0 else 0.0
    herd_bonus  = 0.28 * (jaywalkers_so_far / seen_so_far)
    return clamp(base_prob + first_bonus + herd_bonus)


# ─────────────────────────────────────────
#  Pedestrian paths
# ─────────────────────────────────────────
# Legal crosswalk paths – left-of-centreline, full sidewalk-to-sidewalk.
# Each path is (start, end); lerp_path fills in all steps.
CROSSWALK_PATHS = {
    # N→S: walk south, keeping left (west side of vertical road)
    "N_S": [(MID_X - CW_OFFSET, 0),          (MID_X - CW_OFFSET, HEIGHT - 1)],
    # S→N: walk north, keeping left (east side of vertical road)
    "S_N": [(MID_X + CW_OFFSET, HEIGHT - 1), (MID_X + CW_OFFSET, 0)],
    # W→E: walk east, keeping left (north side of horizontal road)
    "W_E": [(0,         MID_Y - CW_OFFSET),  (WIDTH - 1, MID_Y - CW_OFFSET)],
    # E→W: walk west, keeping left (south side of horizontal road)
    "E_W": [(WIDTH - 1, MID_Y + CW_OFFSET),  (0,         MID_Y + CW_OFFSET)],
}

# Jaywalk paths – side of road, away from the crosswalk stripes.
JAYWALK_PATHS = {
    "N_right": [(MID_X + JW_OFFSET, CURB_TOP),  (MID_X + JW_OFFSET, CURB_BOT)],
    "N_left":  [(MID_X - JW_OFFSET, CURB_TOP),  (MID_X - JW_OFFSET, CURB_BOT)],
    "S_right": [(MID_X + JW_OFFSET, CURB_BOT),  (MID_X + JW_OFFSET, CURB_TOP)],
    "S_left":  [(MID_X - JW_OFFSET, CURB_BOT),  (MID_X - JW_OFFSET, CURB_TOP)],
    "E_up":    [(CURB_RIGHT, MID_Y - JW_OFFSET), (CURB_LEFT, MID_Y - JW_OFFSET)],
    "E_down":  [(CURB_RIGHT, MID_Y + JW_OFFSET), (CURB_LEFT, MID_Y + JW_OFFSET)],
    "W_up":    [(CURB_LEFT, MID_Y - JW_OFFSET),  (CURB_RIGHT, MID_Y - JW_OFFSET)],
    "W_down":  [(CURB_LEFT, MID_Y + JW_OFFSET),  (CURB_RIGHT, MID_Y + JW_OFFSET)],
}

# Which crosswalk paths are legal given each light state
# GREEN_NS = north/south pedestrians may cross; GREEN_EW = east/west may cross
LEGAL_PATHS_FOR_LIGHT = {
    "GREEN_NS": ["N_S", "S_N"],
    "GREEN_EW": ["W_E", "E_W"],
}

# Vehicle paths (drive straight through, faster than peds)
VEHICLE_PATHS = {
    "N->S": ([(MID_X + LANE, 0),          (MID_X + LANE, HEIGHT - 1)], "v"),
    "S->N": ([(MID_X - LANE, HEIGHT - 1), (MID_X - LANE, 0)],          "^"),
    "W->E": ([(0,            MID_Y + LANE),(WIDTH - 1, MID_Y + LANE)],  ">"),
    "E->W": ([(WIDTH - 1,    MID_Y - LANE),(0,         MID_Y - LANE)],  "<"),
}

# Vehicles active for each light phase (green means go)
VEHICLES_FOR_LIGHT = {
    "GREEN_NS": ["N->S", "S->N"],
    "GREEN_EW": ["W->E", "E->W"],
}

# ─────────────────────────────────────────
#  Pedestrian class
# ─────────────────────────────────────────
# _PED_LETTERS = list("P")
_pid_counter = 0

# Pedestrian states
WAITING  = "waiting"   # standing on start sidewalk, not yet crossing
CROSSING = "crossing"  # actively moving along path
ARRIVED  = "arrived"   # reached destination, standing on far sidewalk

class Pedestrian:
    def __init__(self, jaywalking, light_state):
        global _pid_counter
        self.char       = "P"
        _pid_counter   += 1
        self.jaywalking = jaywalking
        self.step       = 0
        self.state      = CROSSING   # start moving immediately
        self.done       = False      # kept for compatibility; True == ARRIVED
        self._build_path(light_state)

    def _build_path(self, light_state):
        if self.jaywalking:
            self.char = "J"
            key   = random.choice(list(JAYWALK_PATHS.keys()))
            wpts  = JAYWALK_PATHS[key]
            steps = 20
        else:
            self.char = "P"
            valid = LEGAL_PATHS_FOR_LIGHT.get(light_state, list(CROSSWALK_PATHS.keys()))
            key   = random.choice(valid)
            wpts  = CROSSWALK_PATHS[key]
            steps = 36
        self.path     = lerp_path(wpts, steps)
        self.start_pos = self.path[0]
        self.end_pos   = self.path[-1]

    def pos(self):
        """Current display position — never returns None; clamps to endpoint."""
        if self.state == ARRIVED:
            return self.end_pos
        idx = min(self.step, len(self.path) - 1)
        return self.path[idx]

    def advance(self, blocked_cells):
        """Move one step unless blocked by a vehicle."""
        if self.state == ARRIVED:
            return

        next_step = self.step + 1

        if next_step >= len(self.path):
            self.state = ARRIVED
            self.done  = True
            return

        next_pos = self.path[next_step]

        # Jaywalkers pause if their next cell is occupied by a vehicle
        if self.jaywalking and next_pos in blocked_cells:
            return   # hold position this frame

        self.step = next_step

    @property
    def display_char(self):
        # Jaywalking status is tracked internally; all pedestrians print
        # as the same uppercase letter so they are visually indistinguishable
        return self.char


# ─────────────────────────────────────────
#  Vehicle class
# ─────────────────────────────────────────
class Vehicle:
    def __init__(self, direction):
        wpts, self.char = VEHICLE_PATHS[direction]
        self.path = lerp_path(wpts, 18)
        self.step = 0
        self.done = False

    def pos(self):
        if self.step >= len(self.path):
            self.done = True
            return None
        return self.path[self.step]

    def advance(self):
        self.step += 1


# ─────────────────────────────────────────
#  Grid builder
# ─────────────────────────────────────────
def build_base_grid(light_state):
    grid = [[' '] * WIDTH for _ in range(HEIGHT)]

    # Sidewalk
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if not (LEFT_EDGE <= x <= RIGHT_EDGE or TOP_EDGE <= y <= BOT_EDGE):
                grid[y][x] = '░'

    # Crosswalk stripes
    for y in range(TOP_EDGE, BOT_EDGE + 1):
        for x in (MID_X - 1, MID_X, MID_X + 1):
            if grid[y][x] == ' ':
                grid[y][x] = '┆'
    for x in range(LEFT_EDGE, RIGHT_EDGE + 1):
        for y in (MID_Y - 1, MID_Y, MID_Y + 1):
            if grid[y][x] == ' ':
                grid[y][x] = '┄'

    # Intersection centre box
    for y in range(MID_Y - 1, MID_Y + 2):
        for x in range(MID_X - 1, MID_X + 2):
            grid[y][x] = '+'

    # Curb lines
    for x in range(WIDTH):
        if grid[CURB_TOP][x] == '░': grid[CURB_TOP][x] = '─'
        if grid[CURB_BOT][x] == '░': grid[CURB_BOT][x] = '─'
    for y in range(HEIGHT):
        if grid[y][CURB_LEFT]  == '░': grid[y][CURB_LEFT]  = '│'
        if grid[y][CURB_RIGHT] == '░': grid[y][CURB_RIGHT] = '│'

    # Corner joints
    for cy, cx, ch in [(CURB_TOP, CURB_LEFT,  '┘'), (CURB_TOP, CURB_RIGHT, '└'),
                        (CURB_BOT, CURB_LEFT,  '┐'), (CURB_BOT, CURB_RIGHT, '┌')]:
        grid[cy][cx] = ch

    # Dashed centre-line lane dividers
    for y in range(HEIGHT):
        if grid[y][MID_X] == ' ':
            grid[y][MID_X] = ':' if y % 2 == 0 else ' '
    for x in range(WIDTH):
        if grid[MID_Y][x] in (' ', ':'):
            grid[MID_Y][x] = '-' if x % 3 == 0 else ' '

    # Jaywalk guide dots
    for y in range(TOP_EDGE, BOT_EDGE + 1):
        for xoff in (MID_X - JW_OFFSET, MID_X + JW_OFFSET):
            if grid[y][xoff] == ' ':
                grid[y][xoff] = '·'
    for x in range(LEFT_EDGE, RIGHT_EDGE + 1):
        for yoff in (MID_Y - JW_OFFSET, MID_Y + JW_OFFSET):
            if grid[yoff][x] == ' ':
                grid[yoff][x] = '·'

    # ── Traffic lights in all four corners of the intersection ───────
    # NS = north/south signal; EW = east/west signal
    ns_sym = 'G' if light_state == 'GREEN_NS' else 'R'
    ew_sym = 'G' if light_state == 'GREEN_EW' else 'R'

    # Place lights just inside the sidewalk corners
    tl_positions = {
        # (y, x): symbol
        (CURB_TOP - 1, CURB_LEFT  - 1): ns_sym,   # NW corner → NS light
        (CURB_TOP - 1, CURB_RIGHT + 1): ew_sym,   # NE corner → EW light
        (CURB_BOT + 1, CURB_LEFT  - 1): ew_sym,   # SW corner → EW light
        (CURB_BOT + 1, CURB_RIGHT + 1): ns_sym,   # SE corner → NS light
    }
    for (ly, lx), sym in tl_positions.items():
        if 0 <= ly < HEIGHT and 0 <= lx < WIDTH:
            grid[ly][lx] = sym

    return grid


# ─────────────────────────────────────────
#  Vehicle collision helpers
# ─────────────────────────────────────────
def vehicle_blocked_cells(vehicles):
    """Return set of (x,y) cells currently occupied by active vehicles."""
    cells = set()
    for v in vehicles:
        if v.done:
            continue
        p = v.pos()
        if p:
            cells.add(p)
    return cells


# ─────────────────────────────────────────
#  Frame renderer
# ─────────────────────────────────────────
def render_frame(base_grid, pedestrians, vehicles, cyc, light_state,
                 frame_num, total_peds, total_jw, cycle_peds, cycle_jw):

    frame = [row[:] for row in base_grid]

    for v in vehicles:
        if v.done:
            continue
        p = v.pos()
        if p and 0 <= p[1] < HEIGHT and 0 <= p[0] < WIDTH:
            frame[p[1]][p[0]] = v.char

    # Draw ALL pedestrians including ARRIVED ones parked on far sidewalk
    for ped in pedestrians:
        p = ped.pos()   # always returns a valid position now
        if p and 0 <= p[1] < HEIGHT and 0 <= p[0] < WIDTH:
            frame[p[1]][p[0]] = ped.display_char

    bw      = WIDTH + 2
    pct_cyc = (cycle_jw / cycle_peds * 100) if cycle_peds > 0 else 0.0
    pct_tot = (total_jw  / total_peds * 100) if total_peds > 0 else 0.0

    ns_label = "GO " if light_state == "GREEN_NS" else "---"
    ew_label = "GO " if light_state == "GREEN_EW" else "---"

    lines = []
    lines.append("╔" + "═" * bw + "╗")
    title = f"  CAMPUS JAYWALKING SIM  ·  {cyc['time']}  ·  {cyc['label']}"
    lines.append("║" + title.ljust(bw) + "║")
    lines.append("╠" + "═" * bw + "╣")
    for row in frame:
        lines.append("║ " + "".join(row) + " ║")
    lines.append("╠" + "═" * bw + "╣")

    # Stats line 1: cycle info
    s1 = (f"  {cyc['name']}  frame {frame_num:>3}  "
          f"light: NS={ns_label} EW={ew_label}  "
          f"G=green  R=red  (corners)")
    lines.append("║" + s1.ljust(bw) + "║")

    # Stats line 2: pedestrian counts
    s2 = (f"  This cycle → peds: {cycle_peds:>3}  jaywalkers: {cycle_jw:>3}"
          f"  rate: {pct_cyc:4.1f}%"
          f"   All-day rate: {pct_tot:4.1f}%")
    lines.append("║" + s2.ljust(bw) + "║")

    # Stats line 3: legend
    s3 = ("  A-Z = pedestrian (jaywalkers indistinguishable by design)   "
          "^v<> = vehicle   · = jaywalk lane   ┆┄ = crosswalk")
    lines.append("║" + s3.ljust(bw) + "║")
    lines.append("╚" + "═" * bw + "╝")

    return "\n".join(lines)


# ─────────────────────────────────────────
#  Cycle runner
# ─────────────────────────────────────────
def run_cycle(cyc, day_peds, day_jw):
    """Run one time-of-day cycle.  Returns updated (day_peds, day_jw)."""

    pedestrians  = []
    vehicles     = []
    cycle_peds   = 0
    cycle_jw     = 0
    frame_num    = 0

    # Frames at which new pedestrian bursts spawn
    BURST_FRAMES = {1, 2, 3, 8, 16}

    for light_state, duration in cyc["light_seq"]:

        # Spawn vehicles appropriate for this light phase
        veh_dirs = VEHICLES_FOR_LIGHT.get(light_state, [])
        vehicles = [Vehicle(d) for d in veh_dirs
                    for _ in range(random.randint(1, 2))]

        for _ in range(duration):
            frame_num += 1

            # Bursty pedestrian arrivals
            if frame_num in BURST_FRAMES:
                count = random.randint(*cyc["arrival_range"])
                if random.random() < cyc["burst_chance"]:
                    count += random.randint(4, 9)

                for _ in range(count):
                    # Recompute probability for each arrival so that every
                    # jaywalker spawned in this burst raises the odds for
                    # the next person — true within-burst social contagion
                    prob  = social_probability(cyc["base_jaywalk"],
                                               cycle_jw,
                                               max(cycle_peds, 1))
                    is_jw = random.random() < prob
                    ped   = Pedestrian(is_jw, light_state)
                    pedestrians.append(ped)
                    cycle_peds += 1
                    if is_jw:
                        cycle_jw += 1
                    # cycle_jw and cycle_peds are now updated immediately,
                    # so the next pedestrian in this same burst already sees
                    # the influence of this one's decision

            # Compute which cells vehicles currently occupy
            blocked = vehicle_blocked_cells(vehicles)

            # Advance pedestrians (jaywalkers pause if a vehicle is in their way)
            for p in pedestrians:
                p.advance(blocked)
            for v in vehicles:
                v.advance()

            # Render
            os.system('cls' if os.name == 'nt' else 'clear')
            grid = build_base_grid(light_state)
            print(render_frame(grid, pedestrians, vehicles,
                               cyc, light_state, frame_num,
                               day_peds + cycle_peds,
                               day_jw   + cycle_jw,
                               cycle_peds, cycle_jw))
            time.sleep(FRAME_DELAY)

        # Keep all pedestrians (ARRIVED ones stay visible on far sidewalk)

    return day_peds + cycle_peds, day_jw + cycle_jw, cycle_peds, cycle_jw


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    random.seed()
    global _pid_counter
    _pid_counter = 0

    day_peds = 0
    day_jw   = 0
    summaries = []

    os.system('cls' if os.name == 'nt' else 'clear')
    print("  Starting Campus Jaywalking Simulation…")
    time.sleep(1.2)

    for cyc in TIME_CYCLES:
        day_peds, day_jw, cpeds, cjw = run_cycle(cyc, day_peds, day_jw)
        summaries.append((cyc, cpeds, cjw))

        # Brief between-cycle summary
        os.system('cls' if os.name == 'nt' else 'clear')
        bw = 59
        print("╔" + "═" * bw + "╗")
        print("║" + f"  {cyc['name']} complete  ·  {cyc['time']}  ·  {cyc['label']}".ljust(bw) + "║")
        print("╠" + "═" * bw + "╣")
        pct = cjw / cpeds * 100 if cpeds else 0
        print("║" + f"  Pedestrians this period : {cpeds}".ljust(bw) + "║")
        print("║" + f"  Jaywalkers              : {cjw}  ({pct:.1f}%)".ljust(bw) + "║")
        print("║" + f"  Social contagion base   : {cyc['base_jaywalk']*100:.1f}%  "
                    f"burst chance: {cyc['burst_chance']*100:.0f}%".ljust(bw - 2) + "  ║")
        print("╚" + "═" * bw + "╝")
        time.sleep(1.8)

    # ── Full-day summary ─────────────────────────────────────────────
    os.system('cls' if os.name == 'nt' else 'clear')
    bw = 59
    print()
    print("╔" + "═" * bw + "╗")
    print("║" + "  FULL DAY SUMMARY".ljust(bw) + "║")
    print("╠" + "═" * bw + "╣")
    for cyc, cpeds, cjw in summaries:
        pct = cjw / cpeds * 100 if cpeds else 0
        row = f"  {cyc['time']}  {cyc['label']:<28}  {cpeds:>3} peds  {pct:4.1f}% jw"
        print("║" + row.ljust(bw) + "║")
    print("╠" + "═" * bw + "╣")
    final_pct = day_jw / day_peds * 100 if day_peds else 0
    print("║" + f"  Total pedestrians  : {day_peds}".ljust(bw) + "║")
    print("║" + f"  Total jaywalkers   : {day_jw}".ljust(bw) + "║")
    print("║" + f"  Overall jaywalk %  : {final_pct:.2f}%".ljust(bw) + "║")
    print("╚" + "═" * bw + "╝")
    print()


if __name__ == "__main__":
    main()