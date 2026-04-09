"""
Rich terminal dashboard for Wolfson.

Displays a live-updating full-screen view of the system's state:
  - Arc progress bar with stage boundaries colour-coded by section
  - Current stage, harmonic mode, scale tracking, contour, velocity
  - Last-phrase note names
  - Rolling statistics (last STATS_WINDOW phrases)

Usage::

    dash = WolfsonDashboard()
    dash.start()                              # call after startup prints

    # on each phrase completion:
    dash.update(params, notes, bpm, elapsed, triggered_by)

    dash.stop()                               # call on exit
"""

from collections import Counter, deque

from rich.console import Console
from rich.layout  import Layout
from rich.live    import Live
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text

from data.chords import chord_index_to_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATS_WINDOW  = 8     # rolling window size
ARC_MAX_WIDTH = 80    # max arc-bar character width — keeps the dashboard
                      # usable in a narrower terminal window; the info and
                      # stats panels below naturally fit within this budget

ARC_TOTAL  = 300.0   # seconds

# (name, start_sec, end_sec, rich_colour)
ARC_STAGES = [
    ("sparse",          0,   60, "white"),
    ("building",       60,  150, "cyan"),
    ("peak",          150,  210, "red"),
    ("recapitulation", 210, 270, "green"),
    ("resolution",    270,  300, "yellow"),
]

_STAGE_COLOR = {name: color for name, _, _, color in ARC_STAGES}

# Shortened labels for the arc bar — full names used everywhere else
_STAGE_LABEL = {"building": "build", "recapitulation": "recap", "resolution": "res"}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

_CONTOUR_ARROW = {"ascending": "↑", "descending": "↓", "neutral": "→"}
_SCALE_STYLE   = {"bass": "bold green", "blend": "yellow", "arc": "white"}

# Shared styles
_DIM   = "bright_black"   # secondary labels — visible but not competing
_BODY  = "white"          # normal body text
_HEAD  = "bold white"     # emphasis / headers
_PANEL = "white"          # panel borders


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class WolfsonDashboard:
    """Full-screen rich terminal dashboard, updated on each phrase."""

    def __init__(self, stats_window: int = STATS_WINDOW):
        self._console      = Console(style="on black")
        self._live         = Live(
            console            = self._console,
            screen             = True,
            refresh_per_second = 4,
        )
        self._phrase_count = 0
        self._elapsed      = 0.0
        self._bpm          = 120.0
        self._params       = {}
        self._notes: list  = []
        self._triggered_by = ""

        self._harm    = deque(maxlen=stats_window)
        self._scale   = deque(maxlen=stats_window)
        self._arc     = deque(maxlen=stats_window)
        self._contour = deque(maxlen=stats_window)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        self._live.start()
        self._live.update(self._render())

    def stop(self):
        self._live.stop()

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    def update(
        self,
        params:       dict,
        notes:        list,
        bpm:          float,
        elapsed:      float,
        triggered_by: str,
    ):
        self._phrase_count += 1
        self._params        = params
        self._notes         = notes
        self._bpm           = bpm
        self._elapsed       = elapsed
        self._triggered_by  = triggered_by

        self._harm.append(params.get("harmonic_mode", "?"))
        self._scale.append(params.get("scale_source",  "arc"))
        self._arc.append(params.get("phrase_energy_arc", "flat"))
        self._contour.append(params.get("contour_target", "?"))

        self._live.update(self._render())

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="arc",    size=5),
            Layout(name="info",   size=6),
            Layout(name="stats",  size=5),
        )
        layout["header"].update(self._header())
        layout["arc"].update(self._arc_panel())
        layout["info"].update(self._info_panel())
        layout["stats"].update(self._stats_panel())
        return layout

    # -- Header --------------------------------------------------------------

    def _header(self) -> Text:
        mins = int(self._elapsed // 60)
        secs = int(self._elapsed % 60)
        t = Text(style="on black")
        t.append("  WOLFSON", style="bold cyan on black")
        t.append(
            f"   t={mins}:{secs:02d}"
            f"  phrase #{self._phrase_count}"
            f"  {self._bpm:.1f} bpm",
            style=f"{_BODY} on black",
        )
        return t

    # -- Arc progress --------------------------------------------------------

    def _arc_panel(self) -> Panel:
        width = min(ARC_MAX_WIDTH, max(20, (self._console.width or 80) - 4))

        bar    = Text(overflow="crop", no_wrap=True)
        labels = Text(overflow="crop", no_wrap=True)

        for name, start, end, color in ARC_STAGES:
            seg = max(1, round(width * (end - start) / ARC_TOTAL))

            if self._elapsed <= start:
                bar.append("░" * seg, style=f"bright_black")
            elif self._elapsed >= end:
                bar.append("█" * seg, style=f"bold {color}")
            else:
                filled = max(0, round(seg * (self._elapsed - start) / (end - start)))
                bar.append("█" * filled,        style=f"bold {color}")
                bar.append("░" * (seg - filled), style=f"bright_black")

            label       = _STAGE_LABEL.get(name, name)[:seg].center(seg)
            label_style = f"bold {color}" if self._elapsed >= start else "bright_black"
            labels.append(label, style=label_style)

        # Stack bar and labels in a borderless grid
        grid = Table.grid()
        grid.add_column()
        grid.add_row(bar)
        grid.add_row(labels)

        return Panel(grid, border_style=_PANEL, style="on black")

    # -- Info panel ----------------------------------------------------------

    def _info_panel(self) -> Panel:
        stage     = self._params.get("stage",          "—")
        harm      = self._params.get("harmonic_mode",   "—")
        src       = self._params.get("scale_source",    "arc")
        contour   = self._params.get("contour_target",  "—")
        vel       = self._params.get("velocity",         80)
        lead      = self._params.get("leadership",      "—")
        mode      = self._params.get("mode",            "—")
        chord_idx = self._params.get("chord_idx")
        chord_name = (chord_index_to_name(chord_idx)
                      if chord_idx is not None else "—")
        n         = len(self._notes)

        note_str = " ".join(
            _NOTE_NAMES[note["pitch"] % 12] for note in self._notes
        )

        arrow   = _CONTOUR_ARROW.get(contour, "?")
        s_col   = _STAGE_COLOR.get(stage, "white")
        src_col = _SCALE_STYLE.get(src, "white")

        grid = Table.grid(padding=(0, 3))
        grid.add_column(min_width=16)
        grid.add_column(min_width=16)
        grid.add_column(min_width=18)
        grid.add_column()

        grid.add_row(
            Text(stage.upper(),              style=f"bold {s_col}"),
            Text(harm,                       style="bold cyan"),
            Text(f"scale  {src.upper()}",    style=src_col),
            Text(f"{arrow}  {contour}",      style=_BODY),
        )
        grid.add_row(
            Text(f"lead   {lead}",           style=_DIM),
            Text(f"chord  {chord_name}",     style="bold yellow"),
            Text(f"vel {vel}   n {n}",       style=_BODY),
            Text(note_str,                   style="cyan"),
        )

        trigger_label = (
            "⟵ bass" if self._triggered_by == "bass"
            else "◎ proactive" if self._triggered_by == "sax"
            else ""
        )
        return Panel(
            grid,
            title=f"last phrase   {trigger_label}",
            border_style=_PANEL,
            style="on black",
        )

    # -- Stats panel ---------------------------------------------------------

    def _stats_panel(self) -> Panel:
        n = len(self._harm)

        def fmt(d: deque) -> Text:
            if not d:
                return Text("—", style=_DIM)
            t = Text()
            for i, (k, v) in enumerate(Counter(d).most_common()):
                if i:
                    t.append("  ", style=_DIM)
                t.append(k,       style="bold cyan")
                t.append(f" {v}", style=_BODY)
            return t

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style=_DIM, min_width=9)
        grid.add_column()

        grid.add_row("harm",    fmt(self._harm))
        grid.add_row("scale",   fmt(self._scale))
        grid.add_row("arc",     fmt(self._arc))
        grid.add_row("contour", fmt(self._contour))

        return Panel(grid, title=f"last {n} phrases", border_style=_PANEL, style="on black")
