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

<div id="summary-overlay" style="display:none;position:fixed;inset:0;background:#0a0a0a;overflow-y:auto;padding:20px 16px;z-index:90">
  <div style="color:#00ffff;font-size:1.1em;font-weight:bold;letter-spacing:4px;margin-bottom:2px">WOLFSON</div>
  <div style="color:#444;font-size:.65em;letter-spacing:3px;margin-bottom:20px;text-transform:uppercase">performance complete</div>
  <div id="sum-grid" style="display:grid;grid-template-columns:auto 1fr 1fr;gap:0;margin-bottom:16px;font-size:.85em"></div>
  <div id="sum-obs"></div>
</div>

<div id="waiting" style="display:flex;position:fixed;inset:0;background:#0a0a0a;flex-direction:column;align-items:center;justify-content:center;z-index:100">
  <div style="color:#00ffff;font-size:1.6em;font-weight:bold;letter-spacing:6px;margin-bottom:20px">WOLFSON</div>
  <div style="color:#444;font-size:.75em;letter-spacing:2px">performance will begin shortly</div>
</div>

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
  var sm = state.scale_mode || "";
  var ss = (state.scale_source || "").toUpperCase();
  document.getElementById("scale").textContent   = sm ? (sm + "  ·  " + ss) : (ss || "—");
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

function showSummary(data) {
  var bass = data.bass || {};
  var sax  = data.sax  || {};
  var grid = document.getElementById("sum-grid");
  var obs  = document.getElementById("sum-obs");
  grid.innerHTML = ""; obs.innerHTML = "";

  function cell(text, color, bold, align) {
    var d = document.createElement("div");
    d.style.cssText = "padding:7px 6px;border-bottom:1px solid #1a1a1a;" +
      "color:" + (color || "#ffffff") + ";" +
      (bold ? "font-weight:bold;" : "") +
      (align === "right" ? "text-align:right;" : "");
    d.textContent = text || "—";
    return d;
  }
  function row(label, bval, sval) {
    grid.appendChild(cell(label, "#999",    false));
    grid.appendChild(cell(bval,  "#ffffff", false, "right"));
    grid.appendChild(cell(sval,  "#00ffff", false, "right"));
  }

  // Header
  grid.appendChild(cell("",            "#555",    false));
  grid.appendChild(cell("BASS (you)", "#ffffff", true,  "right"));
  grid.appendChild(cell("SAX",         "#00ffff", true,  "right"));

  row("Phrases",    bass.phrases, sax.phrases);
  row("Notes",      bass.notes,   sax.notes);
  if (bass.mean_dur !== undefined || sax.mean_dur !== undefined)
    row("Note length",
        bass.mean_dur !== undefined ? bass.mean_dur.toFixed(2) + "b" : "—",
        sax.mean_dur  !== undefined ? sax.mean_dur.toFixed(2)  + "b" : "—");
  if (bass.short_pct !== undefined || sax.short_pct !== undefined)
    row("Short notes",
        bass.short_pct !== undefined ? bass.short_pct + "%" : "—",
        sax.short_pct  !== undefined ? sax.short_pct  + "%" : "—");
  if (bass.pitch_lo || sax.pitch_lo)
    row("Pitch range",
        bass.pitch_lo ? bass.pitch_lo + " \u2013 " + bass.pitch_hi : "—",
        sax.pitch_lo  ? sax.pitch_lo  + " \u2013 " + sax.pitch_hi  : "—");
  if (bass.vel_lo !== undefined || sax.vel_lo !== undefined)
    row("Dynamics",
        bass.vel_lo !== undefined ? bass.vel_lo + " \u2013 " + bass.vel_hi : "—",
        sax.vel_lo  !== undefined ? sax.vel_lo  + " \u2013 " + sax.vel_hi  : "—");

  // Observations (addressed to the human player)
  var obsList = data.observations || [];
  if (obsList.length > 0) {
    var hdr = document.createElement("div");
    hdr.style.cssText = "color:#888;font-size:.6em;letter-spacing:2px;" +
      "text-transform:uppercase;margin-bottom:10px;margin-top:4px";
    hdr.textContent = "observations";
    obs.appendChild(hdr);
    obsList.forEach(function(o) {
      var d = document.createElement("div");
      d.style.cssText = "color:#aaa;font-size:.8em;margin-bottom:8px;line-height:1.5";
      d.textContent = "\u2014 " + o;
      obs.appendChild(d);
    });
  }

  document.getElementById("summary-overlay").style.display = "block";
}

// Polling: fetch /poll every 2 s, update only when phrase_count changes.
// More robust than SSE through proxies and Cloudflare tunnels.
// _lastStarted tracks whether the performance has begun; the waiting
// overlay is shown whenever started=false, so stale browsers from a
// previous run automatically reset to the pre-show screen on the first
// poll after the script is restarted.
// _summaryShown ensures the summary overlay fires once and stays visible.
var _lastPhrase   = -1;
var _lastStarted  = false;
var _summaryShown = false;
function poll() {
  fetch("/poll")
    .then(function(r) { return r.json(); })
    .then(function(state) {
      document.getElementById("offline").style.display = "none";
      var started = !!state.started;
      if (started !== _lastStarted) {
        _lastStarted = started;
        document.getElementById("waiting").style.display = started ? "none" : "flex";
        // New script run detected — reset summary state so the overlay
        // fires again when the new arc's summary arrives.
        if (!started) {
          _summaryShown = false;
          document.getElementById("summary-overlay").style.display = "none";
        }
      }
      if (!_summaryShown && state.summary) {
        _summaryShown = true;
        showSummary(state.summary);
      }
      if (started && !_summaryShown && (state.phrase_count || 0) !== _lastPhrase) {
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
        self._state        = {"started": False}
        self._state_lock   = threading.Lock()
        self._subs         = []    # list[queue.Queue]
        self._subs_lock    = threading.Lock()
        self._phrase_count = 0
        self._tunnel_proc  = None  # cloudflared subprocess, if started
        self._summary      = None  # sticky end-of-arc summary; never cleared by update()

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
                state = dict(self._state)
            # Always inject the sticky summary if available so it is never
            # missed regardless of update() call ordering.
            if self._summary is not None:
                state["summary"] = self._summary
            return jsonify(state)

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
              tinyurl_token: str = "", tinyurl_alias: str = "",
              tunnel_name: str = "", tunnel_host: str = ""):
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
        tunnel_name : str
            Cloudflare named tunnel name (e.g. ``"wolfson"``).  When set,
            runs ``cloudflared tunnel run <name>`` for a stable URL instead
            of a random trycloudflare.com quick tunnel.
        tunnel_host : str
            Stable hostname for the named tunnel, e.g.
            ``"wolfson.numbersintonotes.net"``.
        """
        t = threading.Thread(target=self._serve, daemon=True)
        t.start()
        ip  = _local_ip()
        url = f"http://{ip}:{self._port}"
        print(f"Audience display (local):   {url}")
        print(f"  Share with audience on the same WiFi network.")
        if tunnel:
            self._start_tunnel(self._port, tinyurl_token, tinyurl_alias,
                               tunnel_name, tunnel_host)

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
                      tinyurl_token: str = "", tinyurl_alias: str = "",
                      tunnel_name: str = "", tunnel_host: str = ""):
        """Launch cloudflared tunnel (named or quick) in a background thread."""
        if not self._wait_for_flask(port):
            print("  Warning: Flask did not become ready — tunnel may show Error 1033")

        if tunnel_name:
            self._start_named_tunnel(tunnel_name, tunnel_host, port)
        else:
            self._start_quick_tunnel(port, tinyurl_token, tinyurl_alias)

    def _start_named_tunnel(self, name: str, host: str, port: int = 0):
        """Run a pre-configured named tunnel (stable URL).

        Uses the tunnel name only — ingress is read from
        ~/.cloudflared/config.yml which cloudflared tunnel create/login set up.
        All cloudflared output is printed so problems are immediately visible.
        """
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--protocol", "http2",
                 "run", name],
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,   # merge stderr into stdout
            )
            self._tunnel_proc = proc
        except FileNotFoundError:
            print("  cloudflared not found — install with:")
            print("    brew install cloudflare/cloudflare/cloudflared")
            return

        url = f"https://{host}" if host else f"(tunnel: {name})"
        print(f"Audience display (stable):  {url}")
        print(f"  This URL is permanent — share it before the performance.")

        def _log_output():
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"  [cloudflared] {line}")
        threading.Thread(target=_log_output, daemon=True).start()

    def _start_quick_tunnel(self, port: int,
                            tinyurl_token: str = "", tinyurl_alias: str = ""):
        """Start a trycloudflare.com quick tunnel (random URL each run)."""
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--protocol", "http2",
                 "--url", f"http://localhost:{port}"],
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
            for _ in proc.stderr:   # drain to prevent buffer fill
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

    def show_summary(self, summary: dict):
        """
        Push the end-of-arc performance summary to all connected browsers.

        *summary* is the dict produced by ``_compute_performance_summary()``
        in main.py — keys ``bass``, ``sax``, and ``observations``.  The JS
        layer renders it as a full-screen overlay once and keeps it visible
        for the rest of the session.

        Stored in both ``_summary`` (sticky; injected by every /poll response)
        and ``_state`` (for immediate consistency) so the summary is never
        lost due to a concurrent update() call.
        """
        self._summary = summary
        with self._state_lock:
            self._state = dict(self._state)
            self._state["summary"] = summary

    def reset_summary(self):
        """Remove the summary from state so browsers return to the live view.
        Called at the start of a new loop iteration before the arc restarts.
        """
        self._summary = None
        with self._state_lock:
            self._state = {k: v for k, v in self._state.items() if k != "summary"}
            self._phrase_count = 0

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
            "started":        True,
            "elapsed":        elapsed,
            "phrase_count":   self._phrase_count,
            "bpm":            round(bpm, 1),
            "stage":          params.get("stage",          ""),
            "harmonic_mode":  params.get("harmonic_mode",  ""),
            "scale_source":   params.get("scale_source",   "arc"),
            "scale_mode":     params.get("scale_mode",     ""),
            "contour_target": params.get("contour_target", "neutral"),
            "velocity":       params.get("velocity",        80),
            "triggered_by":   triggered_by,
            "note_names":     note_names,
        }

        with self._state_lock:
            # Preserve summary if already set — update() may be called after
            # show_summary() if a self-play feedback phrase fires at arc end,
            # and we must not lose the summary from the state before the
            # browser has had a chance to poll for it.
            if "summary" in self._state:
                state["summary"] = self._state["summary"]
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
