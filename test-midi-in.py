import rtmidi, time

mid = rtmidi.MidiIn()
mid.open_port(2)  # adjust index if needed
print("Listening for MIDI... (Ctrl-C to stop)")

def cb(event, _):
    msg, _ = event
    status = msg[0] & 0xF0
    if status == 0x90:
        print(f"  Note ON  pitch={msg[1]} vel={msg[2]}")
    elif status == 0x80:
        print(f"  Note OFF pitch={msg[1]}")

mid.set_callback(cb)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
mid.close_port()
