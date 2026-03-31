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
import time

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
</head>
<body style="box-sizing:border-box;margin:0;padding:14px;background:#0a0a0a;color:#ffffff;font-family:'Courier New',Courier,monospace;min-height:100vh">

<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px">
  <span style="color:#00ffff;font-size:1.15em;font-weight:bold;letter-spacing:3px">WOLFSON</span>
  <span id="hdr-info" style="color:#888;font-size:.8em">&mdash;</span>
</div>

<div style="margin-bottom:5px">
  <div id="arc-bar"    style="display:flex;height:22px;border-radius:3px;overflow:hidden;border:1px solid #222"></div>
  <div id="arc-labels" style="display:flex;margin-top:4px;margin-bottom:16px"></div>
</div>

<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px">
  <span id="stage-name" style="font-size:2em;font-weight:bold;text-transform:uppercase;letter-spacing:4px;color:#ffffff">&mdash;</span>
  <span id="stage-rem"  style="color:#888;font-size:.85em">&mdash;</span>
</div>

<div style="display:flex;gap:8px;margin-bottom:10px">
  <div style="flex:1;background:#161616;border-radius:6px;padding:10px 8px 8px;text-align:center">
    <div id="bpm"      style="font-size:2em;font-weight:bold;color:#00ffff;line-height:1">&mdash;</div>
    <div style="font-size:.6em;color:#666;margin-top:3px;letter-spacing:1px">BPM</div>
  </div>
  <div style="flex:1;background:#161616;border-radius:6px;padding:10px 8px 8px;text-align:center">
    <div id="phrase-n" style="font-size:2em;font-weight:bold;color:#ffffff;line-height:1">&mdash;</div>
    <div style="font-size:.6em;color:#666;margin-top:3px;letter-spacing:1px">PHRASES</div>
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
  <div style="background:#161616;border-radius:6px;padding:8px 10px">
    <div style="color:#666;font-size:.6em;text-transform:uppercase;letter-spacing:1px">Harmony</div>
    <div id="harm"    style="font-size:.95em;margin-top:3px;color:#00ffff">&mdash;</div>
  </div>
  <div style="background:#161616;border-radius:6px;padding:8px 10px">
    <div style="color:#666;font-size:.6em;text-transform:uppercase;letter-spacing:1px">Scale</div>
    <div id="scale"   style="font-size:.95em;margin-top:3px;color:#ffffff">&mdash;</div>
  </div>
  <div style="background:#161616;border-radius:6px;padding:8px 10px">
    <div style="color:#666;font-size:.6em;text-transform:uppercase;letter-spacing:1px">Contour</div>
    <div id="contour" style="font-size:.95em;margin-top:3px;color:#ffffff">&mdash;</div>
  </div>
  <div style="background:#161616;border-radius:6px;padding:8px 10px">
    <div style="color:#666;font-size:.6em;text-transform:uppercase;letter-spacing:1px">Velocity</div>
    <div id="vel"     style="font-size:.95em;margin-top:3px;color:#ffffff">&mdash;</div>
  </div>
</div>

<div id="notes"   style="min-height:38px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:10px"></div>
<div id="trigger" style="text-align:center;font-size:.8em;color:#555;padding:6px;letter-spacing:1px">waiting&hellip;</div>

<div id="offline" style="display:none;position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:8px 16px;font-size:.75em;color:#888">reconnecting&hellip;</div>

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
  const bar  = document.getElementById("arc-bar");
  const lbls = document.getElementById("arc-labels");
  bar.innerHTML = ""; lbls.innerHTML = "";
  STAGES.forEach(seg => {
    const d = document.createElement("div");
    d.style.cssText = "height:100%;position:relative;width:" + seg.pct + "%";
    if (elapsed <= seg.start) {
      d.style.background = seg.color + "1e";
    } else if (elapsed >= seg.end) {
      d.style.background = seg.color;
    } else {
      const f   = (elapsed - seg.start) / (seg.end - seg.start);
      const pct = (f * 100).toFixed(1);
      d.style.background =
        "linear-gradient(to right," + seg.color + " " + pct + "%," +
        seg.color + "22 " + pct + "%)";
    }
    bar.appendChild(d);

    const l = document.createElement("div");
    const isActive = elapsed >= seg.start && elapsed < seg.end;
    l.style.cssText = "font-size:.55em;text-transform:uppercase;letter-spacing:.8px;" +
      "text-align:center;overflow:hidden;white-space:nowrap;padding:0 2px;" +
      "width:" + seg.pct + "%;font-weight:" + (isActive ? "bold" : "normal") + ";";
    l.style.color = elapsed >= seg.start ? seg.color : "#444";
    l.textContent = seg.short;
    lbls.appendChild(l);
  });
}

var _pulseTimer = null;
function pulse(color) {
  if (_pulseTimer) clearTimeout(_pulseTimer);
  document.body.style.boxShadow = "inset 0 0 0 2px " + color;
  _pulseTimer = setTimeout(function(){ document.body.style.boxShadow = "none"; }, 600);
}

function update(state) {
  const elapsed   = state.elapsed || 0;
  const remaining = Math.max(0, 300 - elapsed);
  const stage     = state.stage || "";
  const stageCol  = STAGE_COLOR[stage] || "#ffffff";

  document.getElementById("hdr-info").textContent =
    "t=" + fmtTime(elapsed) + "  #" + (state.phrase_count || 0);

  buildArc(elapsed);

  const sn = document.getElementById("stage-name");
  sn.textContent = stage.toUpperCase().replace("RECAPITULATION","RECAP");
  sn.style.color = stageCol;
  document.getElementById("stage-rem").textContent = fmtTime(remaining) + " left";

  document.getElementById("bpm").textContent =
    state.bpm ? state.bpm.toFixed(1) : "—";
  document.getElementById("phrase-n").textContent = state.phrase_count || "—";

  document.getElementById("harm").textContent    = state.harmonic_mode || "—";
  document.getElementById("scale").textContent   = (state.scale_source || "—").toUpperCase();
  const ct = state.contour_target || "";
  document.getElementById("contour").textContent = (ARROWS[ct] || "") + " " + ct;
  document.getElementById("vel").textContent     = state.velocity || "—";

  const notesEl = document.getElementById("notes");
  notesEl.innerHTML = "";
  (state.note_names || []).forEach(function(n) {
    const chip = document.createElement("span");
    chip.style.cssText = "border-radius:4px;padding:5px 9px;font-size:.85em;" +
      "font-weight:bold;color:#000000;background:" + stageCol;
    chip.textContent = n;
    notesEl.appendChild(chip);
  });

  const trig = document.getElementById("trigger");
  const tb   = state.triggered_by || "";
  if (tb === "bass") {
    trig.textContent  = "⟵  bass phrase";
    trig.style.color  = "#00ffff";
  } else if (tb === "sax") {
    trig.textContent  = "◎  sax initiates";
    trig.style.color  = "#ffff00";
  } else {
    trig.textContent  = "…";
    trig.style.color  = "#555";
  }

  pulse(stageCol);
}

// Polling: fetch /poll every 2 s, update only when phrase_count changes.
// More robust than SSE through proxies and Cloudflare tunnels.
var _lastPhrase = -1;
function poll() {
  fetch("/poll")
    .then(function(r) { return r.json(); })
    .then(function(state) {
      document.getElementById("offline").style.display = "none";
      if ((state.phrase_count || 0) !== _lastPhrase) {
        _lastPhrase = state.phrase_count || 0;
        update(state);
      }
    })
    .catch(function() {
      document.getElementById("offline").style.display = "block";
    });
  setTimeout(poll, 2000);
}

buildArc(0);
poll();
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
            from flask import request, make_response
            lines = "=== REQUEST HEADERS ===\n"
            lines += "\n".join(f"{k}: {v}" for k, v in sorted(request.headers))
            r = make_response(lines, 200)
            r.headers["Content-Type"] = "text/plain; charset=utf-8"
            return r

        @app.route("/test")
        def style_test():
            # Minimal page: NO <style> block, only inline style= attributes.
            # If this renders correctly over the tunnel, the Worker is
            # stripping/overriding our <style> block.
            # If this is also broken, the Worker modifies style= attributes too.
            html = (
                "<!DOCTYPE html>"
                "<html><head><meta charset='utf-8'>"
                "<meta name='color-scheme' content='dark'>"
                "</head>"
                "<body style='background:#0a0a0a;color:#ffffff;"
                "font-family:monospace;padding:20px'>"
                "<p style='color:#00ffff;font-size:2em'>CYAN — inline style</p>"
                "<p style='color:#ff4444;font-size:1.5em'>RED — inline style</p>"
                "<p style='color:#ffffff'>WHITE — inline style</p>"
                "<p>DEFAULT — no inline style (should be white if body color inherited)</p>"
                "<p style='background:#ffffff;color:#000000;padding:4px'>"
                "BLACK ON WHITE — explicit</p>"
                "</body></html>"
            )
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/nosse")
        def nosse():
            """Main display using 2-second polling instead of SSE (diagnostic).
            If this renders correctly but / does not, the SSE connection is
            causing the rendering failure through the Cloudflare tunnel."""
            poll_override = (
                "<script>"
                "function connect(){}"   # no-op: disable SSE
                ";(function poll(){"
                "fetch('/poll').then(function(r){return r.json();})"
                ".then(function(s){update(s);}).catch(function(){});"
                "setTimeout(poll,2000);"
                "})();"
                "</script>"
            )
            html = _HTML.replace("</body>", poll_override + "</body>")
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}

        @app.route("/poll")
        def poll():
            """JSON snapshot for polling — no SSE, works through any proxy."""
            from flask import jsonify
            with self._state_lock:
                return jsonify(dict(self._state))

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

    def start(self, tunnel: bool = False,
              tinyurl_token: str = "", tinyurl_alias: str = ""):
        """Start the Flask server in a background daemon thread.

        Parameters
        ----------
        tunnel : bool
            If True, also open a cloudflared quick tunnel and print the
            public URL.  Requires ``cloudflared`` to be installed:
            ``brew install cloudflare/cloudflare/cloudflared``
        tinyurl_token : str
            TinyURL API bearer token.  When provided alongside tinyurl_alias,
            the alias is automatically updated to point to the tunnel URL so
            the audience can always use the same stable link.
        tinyurl_alias : str
            TinyURL alias to update, e.g. ``"wolfson-live"`` for
            ``tinyurl.com/wolfson-live``.
        """
        t = threading.Thread(target=self._serve, daemon=True)
        t.start()
        ip  = _local_ip()
        url = f"http://{ip}:{self._port}"
        print(f"Audience display (local):   {url}")
        print(f"  Share with audience on the same WiFi network.")
        if tunnel:
            self._start_tunnel(self._port, tinyurl_token, tinyurl_alias)

    def _wait_for_flask(self, port: int, timeout: float = 10.0):
        """Block until Flask is accepting connections, or timeout expires."""
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(
                    f"http://localhost:{port}/poll", timeout=1
                )
                return True          # Flask is up
            except Exception:
                time.sleep(0.2)
        return False                 # timed out

    def _start_tunnel(self, port: int,
                      tinyurl_token: str = "", tinyurl_alias: str = ""):
        """Launch cloudflared and print the public trycloudflare.com URL."""
        # Wait for Flask to be ready before cloudflared tries to connect,
        # to avoid Cloudflare Error 1033 (origin unreachable on first request).
        if not self._wait_for_flask(port):
            print("  Warning: Flask did not become ready — tunnel may show Error 1033")
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
                    tunnel_url = m.group()
                    print(f"Audience display (public):  {tunnel_url}")
                    print(f"  Share this URL with the audience anywhere.")
                    url_found.set()
                    if tinyurl_token and tinyurl_alias:
                        self._update_tinyurl(tunnel_url, tinyurl_token, tinyurl_alias)
                    break
            # Drain remaining stderr so the pipe buffer never fills
            for _ in proc.stderr:
                pass

        threading.Thread(target=_read_stderr, daemon=True).start()

        if not url_found.wait(timeout=20):
            print("  cloudflared tunnel URL not found within 20 s "
                  "(is cloudflared installed and reachable?)")

    def _update_tinyurl(self, tunnel_url: str, token: str, alias: str):
        """Update a TinyURL alias to point to tunnel_url."""
        import urllib.request as _ur
        payload = json.dumps({"url": tunnel_url}).encode()
        req = _ur.Request(
            f"https://api.tinyurl.com/alias/tinyurl.com/{alias}",
            data    = payload,
            method  = "PATCH",
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with _ur.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print(f"Audience display (stable):  https://tinyurl.com/{alias}")
                    print(f"  This link always points to the current tunnel.")
                else:
                    print(f"  TinyURL update returned HTTP {resp.status}")
        except Exception as exc:
            print(f"  TinyURL update failed: {exc}")

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
