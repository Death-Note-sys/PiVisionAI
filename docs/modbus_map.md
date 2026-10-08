# PiVisionAI Modbus Map

The system runs a `ModbusTcpServer` (default port: 5020, configurable).
Counters are represented as 32-bit unsigned integers, mapped to two 16-bit holding registers.

## Word Order
32-bit values are stored in **Big-Endian / Network Byte Order**:
- Register N: High word (Most Significant 16 bits)
- Register N+1: Low word (Least Significant 16 bits)

## Holding Registers (4x)

| Register | Name | Type | R/W | Description |
| :--- | :--- | :--- | :--- | :--- |
| 40001 (0) | Heartbeat | UInt16 | R | Toggles / increments every second to indicate system alive. |
| 40002 (1) | Trigger | UInt16 | R/W | Write `1` to trigger an inspection (in Modbus/External trigger mode). Auto-resets to `0`. |
| 40003-40004 (2-3) | Total Count | UInt32 | R | Total inspections performed (Total = OK + NG). |
| 40005-40006 (4-5) | OK Count | UInt32 | R | Count of passed inspections. |
| 40007-40008 (6-7) | NG Count | UInt32 | R | Count of failed/uncertain inspections. Uncertain is a subset of NG. |
| 40009 (8) | Error State | UInt16 | R | System error code (0 = Normal, 1 = Camera Error, etc.). |
