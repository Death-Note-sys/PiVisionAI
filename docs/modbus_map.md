# PiVisionAI Modbus Map

The system runs a `ModbusTcpServer` on **port 5020** by default (configurable via
`modbus_port` / `PI_VISION_MODBUS_PORT`).

## Network / Security

| Setting | Default | Env var |
| :--- | :--- | :--- |
| Bind address (`modbus_host`) | `0.0.0.0` (all interfaces) | `PI_VISION_MODBUS_HOST` |
| Port (`modbus_port`) | `5020` | `PI_VISION_MODBUS_PORT` |

> **WARNING — firewall required.**  
> When `modbus_host` is `0.0.0.0` the server is reachable from every network
> interface. **Firewall TCP port 5020 to the PLC-facing NIC only.** Set
> `modbus_host` to the specific IP of the PLC-facing interface (e.g.
> `192.168.1.10`) to restrict binding. The server logs a WARNING on startup
> when bound to all interfaces.

---

## Word Order

32-bit values are stored in **Big-Endian / Network Byte Order**:

- Register N: High word (Most Significant 16 bits)
- Register N+1: Low word (Least Significant 16 bits)

---

## Holding Registers (4x)

| Register | Address | Name | Type | R/W | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 40001 | 0 | Active Module ID | UInt16 | R | 1=Object Detection, 2=Measurement, 3=AI Identify, 4=OCR, 5=Color Analysis. 0=no module active. |
| 40002 | 1 | Overall Status | UInt16 | R | Per-module last judgment. **0=Idle** (no triggered inspection yet for this module since server start), **1=OK**, **2=NG**, **3=Uncertain (treat as reject — subset of NG)**. |
| 40003–40004 | 2–3 | Total Count | UInt32 | R | Total triggered inspections (OK + NG). Big-endian: HR2=high word, HR3=low word. |
| 40005–40006 | 4–5 | OK Count | UInt32 | R | Passed inspections. |
| 40007–40008 | 6–7 | NG Count | UInt32 | R | Failed/Uncertain inspections. |
| 40009 | 8 | Heartbeat | UInt16 | R | Increments once per second to indicate system alive. |
| 40010 | 9 | Trigger | UInt16 | **R/W** | **PLC-owned.** Write `1` to trigger an inspection (Modbus/External trigger mode). The server reads and auto-resets to `0` after consuming. **The sync loop never writes HR9.** |
| 40011 | 10 | Error State | UInt16 | R | System error code. 0 = Normal, 1 = Camera Error. |
| 40012 | 11 | Last Trigger ID | UInt16 | R | Monotonic ID of the most recent triggered inspection. **Wraps from 65535 → 0.** PLC should check *changed* (not `>`) to handle wrap. |
| 40013 | 12 | Status Age ms | UInt16 | R | Milliseconds since HR1 (Overall Status) was last updated by a triggered inspection. **65535 = HR1 has never been set** (server just started or no triggered inspection yet). Use to detect stale statuses: if Status Age ms is large, HR1 carries the result of an old inspection. |

---

## Coils (0x)

| Coil | Address | Name | Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| 00001 | 0 | Trigger | Bool | **Alternative trigger channel.** Write `1` to trigger an inspection. Auto-resets to `0` after consuming. Equivalent to writing `1` to HR9. |

Both **Coil 0** and **HR9** are valid trigger inputs. The server consumes whichever fires first within each 100 ms poll cycle.

---

## Module Codes

| Code | Module |
| :--- | :--- |
| 1 | Object Detection |
| 2 | Measurement |
| 3 | AI Identify |
| 4 | OCR |
| 5 | Color Analysis |

---

## Status Codes

| Code | Meaning | PLC Action |
| :--- | :--- | :--- |
| 0 | Idle | No triggered inspection yet for this module since server start |
| 1 | OK (Pass) | Accept part |
| 2 | NG (Fail) | **Reject part** |
| 3 | Uncertain | **Treat as reject** (subset of NG, counted in NG Count) |

---

## Trigger Acknowledgment (Recommended Handshake)

A PLC should:

1. Write `1` to **HR9** (or **Coil 0**).
2. Poll until **both** conditions are true:
   - **Last Trigger ID (HR11) has changed** (compare to stored previous value; use *changed*, not `>`, because HR11 wraps at 65535).
   - **Total Count (HR2–HR3) has incremented** by at least 1.
3. Read **Overall Status (HR1)** for the judgment.
4. Optionally check **Status Age ms (HR12)** — a value near zero confirms the status is fresh.

> Two consecutive identical judgments are distinguishable because Last Trigger ID is monotonic (wraps at 65535).

---

## Counter Reset

Counters are **not** reset over Modbus by default. Counter reset is **opt-in** and
must be explicitly enabled via the application configuration (`counter_reset_enabled`,
default `false`). This prevents accidental resets from a PLC write collision.
To reset counters, use the web UI or the REST API (`POST /api/v1/inspection/reset`).

