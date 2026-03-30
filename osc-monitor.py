"""
Simple OSC monitor for testing Wolfson OSC output.

Prints every incoming OSC message with a timestamp.

Usage:
    python osc-monitor.py           # listen on port 9000
    python osc-monitor.py --port 8000
"""

import argparse
import time
from pythonosc import dispatcher, osc_server

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9000)
args = parser.parse_args()

def handle(address, *args):
    ts = time.strftime("%H:%M:%S")
    vals = "  ".join(str(a) for a in args)
    print(f"{ts}  {address:<40s}  {vals}")

d = dispatcher.Dispatcher()
d.set_default_handler(handle)

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.port), d)
print(f"Listening for OSC on port {args.port}  (Ctrl-C to stop)\n")
server.serve_forever()
