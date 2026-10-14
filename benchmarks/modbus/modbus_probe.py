"""
Modbus TCP probe / diagnostic tool.

Usage:
    python modbus_probe.py [readonly|coil0|hr9]

Modes:
  readonly  - Read registers, wait 3s, show changes (heartbeat check).
  coil0     - Write coil 0, wait 3s, show trigger effect.
  hr9       - Write HR9=1, wait 3s, show trigger effect.
  hr0       - (Legacy alias for hr9) Same as hr9.
"""
import sys, time
from pymodbus.client import ModbusTcpClient

HOST, PORT = "127.0.0.1", 5020
mode = sys.argv[1] if len(sys.argv) > 1 else "readonly"

c = ModbusTcpClient(HOST, port=PORT)
if not c.connect():
    sys.exit("Cannot connect to %s:%d - is the server running with Modbus enabled?" % (HOST, PORT))

def snap():
    hr = c.read_holding_registers(0, count=30)
    co = c.read_coils(0, count=8)
    if hr.isError() or co.isError():
        sys.exit("Read error: %s / %s" % (hr, co))
    return list(hr.registers), list(co.bits[:8])

def changed(a, b, label):
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print("  %s changed: %s" % (label, diffs if diffs else "nothing"))

s0 = snap()
print("HR0-29:", s0[0]); print("coils0-7:", s0[1])
time.sleep(3)
s1 = snap()
print("after 3 s idle:"); changed(s0[0], s1[0], "HR"); changed(s0[1], s1[1], "coils")

if mode in ("coil0", "hr0", "hr9"):
    print("firing trigger via", mode)
    if mode == "coil0":
        c.write_coil(0, True)
    elif mode in ("hr0", "hr9"):
        c.write_register(9, 1)
    time.sleep(3)
    s2 = snap()
    print("after trigger:"); changed(s1[0], s2[0], "HR"); changed(s1[1], s2[1], "coils")
    print("HR0-29:", s2[0]); print("coils0-7:", s2[1])
c.close()
