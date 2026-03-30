"""
Audience web display for Wolfson.

Serves a mobile-optimised page showing live performance state to any device
on the same WiFi network.  No app required — just a browser URL.

Architecture
------------
A Flask server runs in a daemon background thread inside the Wolfson process.
On each phrase, WebAudienceDisplay.update() pushes a JSON state snapshot into
a per-subscriber queue.  Browser clients connect to /stream (Server-Sent Events)
and receive updates in real time; JS rewrites the DOM without a page reload.

Usage::

    display = WebAudienceDisplay(port=5000)
    display.start()                   # local URL only
    display.start(tunnel=True)        # also opens a cloudflared public tunnel
    display.update(params, notes, bpm, elapsed, triggered_by)
    display.stop()                    # graceful shutdown + tunnel termination

Network — local
---------------
Binds to 0.0.0.0 so any device on the same LAN can connect.
The startup print shows the machine's LAN IP so the performer can share the URL.

Network — public tunnel (--tunnel)
-----------------------------------
Shells out to ``cloudflared tunnel --url http://localhost:<port>``.
cloudflared makes an outbound HTTPS connection to Cloudflare's edge, which
issues a public URL (e.g. https://curious-fox-amazing.trycloudflare.com).
This works on eduroam and most institutional networks because it requires no
inbound ports — only outbound HTTPS.

Requires cloudflared to be installed:
    brew install cloudflare/cloudflare/cloudflared

No account or authentication needed for quick tunnels.
All assets are embedded — no CDN requests, works offline once the page is loaded.
"""

import json
import logging
import queue
import re
import socket
import subprocess
import threading

from flask import Flask, Response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

_ARC_TOTAL  = 300.0   # seconds


# ---------------------------------------------------------------------------
# Embedded HTML page (single self-contained file, no external dependencies)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>Wolfson</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  color-scheme:dark;
  --c-sparse:#aaaaaa;
  --c-building:#00ffff;
  --c-peak:#ff4444;
  --c-recap:#44ff44;
  --c-res:#ffff44;
  --bg:#0a0a0a;
  --card:#161616;
  --dim:#444;
  --text:#ffffff;
}
body{
  background:#0a0a0a;
  color:#ffffff;
  font-family:'Courier New',Courier,monospace;
  padding:14px 14px env(safe-area-inset-bottom,14px);
  min-height:100svh;
  transition:box-shadow .05s;
}
body.pulse{box-shadow:inset 0 0 0 2px var(--pulse-color,#00ffff)}

/* ---- header ---- */
.hdr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}
.hdr-title{color:#00ffff;font-size:1.15em;font-weight:bold;letter-spacing:3px}
.hdr-info{color:#555;font-size:.8em}

/* ---- arc bar ---- */
.arc-wrap{margin-bottom:5px}
.arc-bar{
  display:flex;height:22px;border-radius:3px;overflow:hidden;
  border:1px solid #222;
}
.arc-seg{height:100%;position:relative;transition:opacity .4s}
.arc-labels{display:flex;margin-top:4px;margin-bottom:16px}
.arc-lbl{
  font-size:.55em;text-transform:uppercase;letter-spacing:.8px;
  text-align:center;overflow:hidden;white-space:nowrap;
  color:#333;padding:0 2px;
}
.arc-lbl.active{font-weight:bold;opacity:1}

/* ---- stage ---- */
.stage-row{
  display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:14px;
}
.stage-name{font-size:2em;font-weight:bold;text-transform:uppercase;letter-spacing:4px}
.stage-rem{color:#555;font-size:.85em}

/* ---- big numbers ---- */
.nums-row{display:flex;gap:8px;margin-bottom:10px}
.num-card{
  flex:1;background:var(--card);border-radius:6px;
  padding:10px 8px 8px;text-align:center;
}
.num-val{font-size:2em;font-weight:bold;color:#00ffff;line-height:1}
.num-val.white{color:#fff}
.num-unit{font-size:.6em;color:#444;margin-top:3px;letter-spacing:1px}

/* ---- info cards ---- */
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.card{background:var(--card);border-radius:6px;padding:8px 10px}
.card-lbl{color:#444;font-size:.6em;text-transform:uppercase;letter-spacing:1px}
.card-val{font-size:.95em;margin-top:3px}
.card-val.cyan{color:#00ffff}

/* ---- notes ---- */
.notes-wrap{min-height:38px;display:flex;flex-wrap:wrap;gap:6px;
  align-items:center;margin-bottom:10px}
.note{
  border-radius:4px;padding:5px 9px;
  font-size:.85em;font-weight:bold;color:#000;
  transition:background .3s;
}

/* ---- trigger ---- */
.trigger{text-align:center;font-size:.8em;color:#333;padding:6px;letter-spacing:1px}
.trigger.bass{color:#00ffff}
.trigger.sax{color:#ffff00}

/* ---- offline ---- */
.offline-msg{
  position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
  background:#1a1a1a;border:1px solid #333;border-radius:6px;
  padding:8px 16px;font-size:.75em;color:#555;
  display:none;
}
.offline-msg.show{display:block}
</style>
</head>
<body style="background:#0a0a0a;color:#ffffff;font-family:'Courier New',Courier,monospace">

<div class="hdr">
  <span class="hdr-title" style="color:#00ffff">WOLFSON</span>
  <span class="hdr-info" id="hdr-info" style="color:#888">&mdash;</span>
</div>

<div class="arc-wrap">
  <div class="arc-bar"   id="arc-bar"></div>
  <div class="arc-labels" id="arc-labels"></div>
</div>

<div class="stage-row">
  <span class="stage-name" id="stage-name">&mdash;</span>
  <span class="stage-rem"  id="stage-rem">&mdash;</span>
</div>

<div class="nums-row">
  <div class="num-card">
    <div class="num-val"       id="bpm">&mdash;</div>
    <div class="num-unit">BPM</div>
  </div>
  <div class="num-card">
    <div class="num-val white" id="phrase-n">&mdash;</div>
    <div class="num-unit">PHRASES</div>
  </div>
</div>

<div class="cards">
  <div class="card"><div class="card-lbl">Harmony</div>
    <div class="card-val cyan" id="harm">&mdash;</div></div>
  <div class="card"><div class="card-lbl">Scale</div>
    <div class="card-val"      id="scale">&mdash;</div></div>
  <div class="card"><div class="card-lbl">Contour</div>
    <div class="card-val"      id="contour">&mdash;</div></div>
  <div class="card"><div class="card-lbl">Velocity</div>
    <div class="card-val"      id="vel">&mdash;</div></div>
</div>

<div class="notes-wrap" id="notes"></div>
<div class="trigger"    id="trigger">waiting&hellip;</div>

<div class="offline-msg" id="offline">reconnecting&hellip;</div>

<script>
const STAGES = [
  {name:"sparse",         short:"sparse",  start:0,   end:60,  pct:20, color:"#aaaaaa"},
  {name:"building",       short:"building",start:60,  end:150, pct:30, color:"#00ffff"},
  {name:"peak",           short:"peak",    start:150, end:210, pct:20, color:"#ff4444"},
  {name:"recapitulation", short:"recap",   start:210, end:270, pct:20, color:"#44ff44"},
  {name:"resolution",     short:"res",     start:270, end:300, pct:10, color:"#ffff44"},
];
const STAGE_COLOR = {};
STAGES.forEach(s => STAGE_COLOR[s.name] = s.color);

const ARROWS = {ascending:"↑", descending:"↓", neutral:"→"};

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2,"0");
}

function buildArc(elapsed) {
  const bar = document.getElementById("arc-bar");
  const lbls = document.getElementById("arc-labels");
  bar.innerHTML = ""; lbls.innerHTML = "";
  STAGES.forEach(seg => {
    // bar segment
    const d = document.createElement("div");
    d.className = "arc-seg";
    d.style.width = seg.pct + "%";
    if (elapsed <= seg.start) {
      d.style.background = seg.color + "1e";   // very dim future
    } else if (elapsed >= seg.end) {
      d.style.background = seg.color;          // fully done
    } else {
      const f = (elapsed - seg.start) / (seg.end - seg.start);
      const pct = (f * 100).toFixed(1);
      d.style.background =
        "linear-gradient(to right," + seg.color + " " + pct + "%," +
        seg.color + "22 " + pct + "%)";
    }
    bar.appendChild(d);

    // label
    const l = document.createElement("div");
    l.className = "arc-lbl" +
      (elapsed >= seg.start && elapsed < seg.end ? " active" : "");
    l.style.width = seg.pct + "%";
    l.style.color = elapsed >= seg.start ? seg.color : "#333";
    l.textContent = seg.short;
    lbls.appendChild(l);
  });
}

function update(state) {
  const elapsed   = state.elapsed || 0;
  const remaining = Math.max(0, 300 - elapsed);
  const stage     = state.stage || "";
  const stageCol  = STAGE_COLOR[stage] || "#fff";

  document.getElementById("hdr-info").textContent =
    "t=" + fmtTime(elapsed) + "  #" + (state.phrase_count || 0);

  buildArc(elapsed);

  const sn = document.getElementById("stage-name");
  sn.textContent   = stage.toUpperCase().replace("RECAPITULATION","RECAP");
  sn.style.color   = stageCol;
  document.getElementById("stage-rem").textContent = fmtTime(remaining) + " left";

  document.getElementById("bpm").textContent =
    state.bpm ? state.bpm.toFixed(1) : "—";
  document.getElementById("phrase-n").textContent = state.phrase_count || "—";

  document.getElementById("harm").textContent    = state.harmonic_mode || "—";
  document.getElementById("scale").textContent   = (state.scale_source  || "—").toUpperCase();
  const ct = state.contour_target || "";
  document.getElementById("contour").textContent =
    (ARROWS[ct] || "") + " " + ct;
  document.getElementById("vel").textContent     = state.velocity || "—";

  const notesEl = document.getElementById("notes");
  notesEl.innerHTML = "";
  (state.note_names || []).forEach(n => {
    const chip = document.createElement("span");
    chip.className     = "note";
    chip.style.background = stageCol;
    chip.textContent   = n;
    notesEl.appendChild(chip);
  });

  const trig = document.getElementById("trigger");
  const tb   = state.triggered_by || "";
  if (tb === "bass") {
    trig.textContent  = "⟵  bass phrase";
    trig.className    = "trigger bass";
  } else if (tb === "sax") {
    trig.textContent  = "◎  sax initiates";
    trig.className    = "trigger sax";
  } else {
    trig.textContent  = "…";
    trig.className    = "trigger";
  }

  // pulse border in stage colour
  document.body.style.setProperty("--pulse-color", stageCol);
  document.body.classList.remove("pulse");
  void document.body.offsetWidth;   // force reflow
  document.body.classList.add("pulse");
  setTimeout(() => document.body.classList.remove("pulse"), 600);
}

function connect() {
  const src = new EventSource("/stream");
  const off = document.getElementById("offline");
  src.onopen    = () => off.classList.remove("show");
  src.onmessage = e => { try { update(JSON.parse(e.data)); } catch(_){} };
  src.onerror   = () => {
    src.close();
    off.classList.add("show");
    setTimeout(connect, 3000);
  };
}

buildArc(0);
connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# WebAudienceDisplay
# ---------------------------------------------------------------------------

class WebAudienceDisplay:
    """
    Mobile-optimised audience display served over HTTP/SSE.

    Parameters
    ----------
    port : int
        TCP port to bind (default 5000).  Must be reachable from the audience
        devices — ensure your firewall allows inbound connections on this port.
    """

    def __init__(self, port: int = 5000):
        self._port         = port
        self._state        = {}
        self._state_lock   = threading.Lock()
        self._subs         = []    # list[queue.Queue]
        self._subs_lock    = threading.Lock()
        self._phrase_count = 0
        self._tunnel_proc  = None  # cloudflared subprocess, if started

        # Suppress Flask/Werkzeug startup noise
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        self._app = Flask(__name__)
        self._setup_routes()

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    def _setup_routes(self):
        app = self._app

        @app.route("/")
        def index():
            return _HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/headers")
        def show_headers():
            from flask import request
            lines = "\n".join(f"{k}: {v}" for k, v in sorted(request.headers))
            return lines, 200, {"Content-Type": "text/plain; charset=utf-8"}

        @app.route("/stream")
        def stream():
            q = queue.Queue(maxsize=16)
            with self._subs_lock:
                self._subs.append(q)

            def generate():
                # Send current state immediately so the page loads populated
                with self._state_lock:
                    snapshot = dict(self._state)
                if snapshot:
                    yield f"data: {json.dumps(snapshot)}\n\n"
                try:
                    while True:
                        try:
                            data = q.get(timeout=20)
                            yield f"data: {json.dumps(data)}\n\n"
                        except queue.Empty:
                            yield ": keepalive\n\n"   # prevent proxy timeouts
                except GeneratorExit:
                    pass
                finally:
                    with self._subs_lock:
                        try:
                            self._subs.remove(q)
                        except ValueError:
                            pass

            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":    "no-cache",
                    "X-Accel-Buffering":"no",   # disable nginx buffering
                },
            )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self, tunnel: bool = False):
        """Start the Flask server in a background daemon thread.

        Parameters
        ----------
        tunnel : bool
            If True, also open a cloudflared quick tunnel and print the
            public URL.  Requires ``cloudflared`` to be installed:
            ``brew install cloudflare/cloudflare/cloudflared``
        """
        t = threading.Thread(target=self._serve, daemon=True)
        t.start()
        ip  = _local_ip()
        url = f"http://{ip}:{self._port}"
        print(f"Audience display (local):   {url}")
        print(f"  Share with audience on the same WiFi network.")
        if tunnel:
            self._start_tunnel(self._port)

    def _start_tunnel(self, port: int):
        """Launch cloudflared and print the public trycloudflare.com URL."""
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.PIPE,
            )
            self._tunnel_proc = proc
        except FileNotFoundError:
            print("  cloudflared not found — install with:")
            print("    brew install cloudflare/cloudflare/cloudflared")
            return

        url_found = threading.Event()
        pattern   = re.compile(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')

        def _read_stderr():
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="replace")
                m = pattern.search(line)
                if m:
                    print(f"Audience display (public):  {m.group()}")
                    print(f"  Share this URL with the audience anywhere.")
                    url_found.set()
                    break
            # Drain remaining stderr so the pipe buffer never fills
            for _ in proc.stderr:
                pass

        threading.Thread(target=_read_stderr, daemon=True).start()

        if not url_found.wait(timeout=20):
            print("  cloudflared tunnel URL not found within 20 s "
                  "(is cloudflared installed and reachable?)")

    def stop(self):
        """Signal all connected clients to disconnect and terminate the tunnel."""
        if self._tunnel_proc is not None:
            try:
                self._tunnel_proc.terminate()
            except Exception:
                pass
            self._tunnel_proc = None
        with self._subs_lock:
            for q in self._subs:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

    def _serve(self):
        self._app.run(
            host      = "0.0.0.0",
            port      = self._port,
            threaded  = True,
            use_reloader = False,
            debug     = False,
        )

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
        """
        Push updated state to all connected browsers.
        Called once per phrase, same signature as WolfsonDashboard.update().
        """
        self._phrase_count += 1

        note_names = [
            _NOTE_NAMES[n["pitch"] % 12]
            for n in notes
            if n.get("pitch", -1) >= 0
        ]

        state = {
            "elapsed":        elapsed,
            "phrase_count":   self._phrase_count,
            "bpm":            round(bpm, 1),
            "stage":          params.get("stage",          ""),
            "harmonic_mode":  params.get("harmonic_mode",  ""),
            "scale_source":   params.get("scale_source",   "arc"),
            "contour_target": params.get("contour_target", "neutral"),
            "velocity":       params.get("velocity",        80),
            "triggered_by":   triggered_by,
            "note_names":     note_names,
        }

        with self._state_lock:
            self._state = state

        with self._subs_lock:
            dead = []
            for q in self._subs:
                try:
                    q.put_nowait(state)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subs.remove(q)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    """Return the machine's LAN IP address (best-effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
