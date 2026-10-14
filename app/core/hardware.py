import threading
import time
import logging
from typing import Optional
from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from app.core.config_service import ConfigService
from app.core.event_bus import EventBus
from app.core.state import StateStore

logger = logging.getLogger(__name__)

# Plan module codes: 1=Object Detection, 2=Measurement, 3=AI Identify, 4=OCR, 5=Color Analysis
MODULE_CODE_MAP = {
    "core-object-detection": 1,
    "core-measurement": 2,
    "core-ai-identify": 3,
    "core-ocr": 4,
    "core-color-analysis": 5,
}

class HardwareManager:
    """Manages Modbus TCP Server and synchronizes with StateStore."""
    def __init__(self, config: ConfigService, event_bus: EventBus, state_store: StateStore):
        self.config = config
        self.event_bus = event_bus
        self.state_store = state_store
        
        settings = self.config.get_settings()
        self.host = getattr(settings, "modbus_host", "0.0.0.0")
        self.port = settings.modbus_port
        self.enabled = settings.modbus_enabled
        
        self._modbus_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False

        # Track per-module last status: remains 0 (Idle) until an inspection
        # fires for the currently-active module after server start.
        self._module_last_status: dict = {}  # module_id -> int (0/1/2/3)

        # Last trigger ID written by Pipeline via FrameProcessed event.
        self._last_trigger_id = 0
        self._trigger_id_lock = threading.Lock()

        # Timestamp of last status update for HR12 (status age in ms).
        self._last_status_ts: float = 0.0  # 0 = never updated
        
        # Register map (32 words allocated):
        # HR0:  Active Module ID
        # HR1:  Overall Status  (per-module, 0=Idle until first inspection)
        # HR2-3: Total Count (big-endian UInt32)
        # HR4-5: OK Count
        # HR6-7: NG Count
        # HR8:  Heartbeat
        # HR9:  Trigger (PLC-owned, R/W — NEVER written by sync loop)
        # HR10: Error State
        # HR11: Last Trigger ID (UInt16, monotonic, wraps at 65535)
        # HR12: Status Age ms (UInt16, ms since HR1 last updated; 65535=never)
        self.store = ModbusSequentialDataBlock(0, [0] * 32)
        self.coil_store = ModbusSequentialDataBlock(0, [0] * 32)
        self.context = ModbusServerContext(slaves=ModbusSlaveContext(
            di=self.coil_store, co=self.coil_store, hr=self.store, ir=self.store,
            zero_mode=True
        ), single=True)

        # Subscribe to FrameProcessed to capture per-module status and trigger_id
        if self.event_bus:
            self.event_bus.subscribe("FrameProcessed", self._on_frame_processed)
        
    def _on_frame_processed(self, payload: dict):
        """Update per-module status and last trigger ID from pipeline events."""
        if not payload:
            return
        module_id = payload.get("module_id", "")
        triggered = payload.get("triggered", False)
        trigger_id = payload.get("trigger_id", 0)
        result = payload.get("result")

        if triggered and result is not None:
            # Derive status from result (same logic as coordinator.record)
            status = 0
            if hasattr(result, "match"):
                if result.match is not None:
                    status = 1 if result.match else 2
            elif hasattr(result, "classification"):
                if result.classification == "Good":
                    status = 1
                elif result.classification == "Bad":
                    status = 2
                elif result.classification == "Uncertain":
                    status = 3
            elif hasattr(result, "is_ok"):
                if result.is_ok is not None:
                    status = 1 if result.is_ok else 2

            if status > 0:
                self._module_last_status[module_id] = status
                self._last_status_ts = time.time()

            with self._trigger_id_lock:
                self._last_trigger_id = trigger_id & 0xFFFF

    def initialize(self) -> bool:
        if not self.enabled:
            logger.info("Modbus is disabled in configuration.")
            return True

        if self.host in ("", "0.0.0.0"):
            logger.warning(
                "Modbus server is binding to ALL interfaces (%s:%d). "
                "Restrict modbus_host (or PI_VISION_MODBUS_HOST) to the PLC-facing "
                "network interface in production to avoid unintended exposure.",
                self.host, self.port
            )
            
        self._running = True
        
        # Start the Modbus sync thread
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        
        # Start the Modbus TCP server thread
        self._modbus_thread = threading.Thread(target=self._run_server, daemon=True)
        self._modbus_thread.start()
        logger.info(f"HardwareManager initialized Modbus on {self.host}:{self.port}")
        return True
        
    def _run_server(self):
        identity = ModbusDeviceIdentification()
        identity.VendorName = 'PiVisionAI'
        identity.ProductCode = 'PV'
        identity.ProductName = 'PiVisionAI'
        identity.ModelName = 'PiVisionAI Node'
        identity.MajorMinorRevision = '1.0'
        
        try:
            # We use the sync server since it's running in its own daemon thread
            StartTcpServer(context=self.context, identity=identity, address=(self.host, self.port))
        except Exception as e:
            logger.error(f"Modbus server crashed: {e}")
            
    def _sync_loop(self):
        heartbeat = 0
        loop_count = 0
        while self._running:
            try:
                # Fetch Active Module first so commands know target module
                active_str = ""
                from app.core.container import Container
                container = Container.get_instance()
                if container and hasattr(container, "module_controller") and container.module_controller and container.module_controller.active_metadata:
                    active_str = container.module_controller.active_metadata.id

                # 1. Check Coil 0 (Coil trigger)
                if self.coil_store.getValues(0, 1)[0]:
                    logger.info("Modbus trigger received via Coil 0.")
                    self.event_bus.publish("HardwareTrigger", {})
                    self.coil_store.setValues(0, [0])

                # 2. Check Register 9 (Holding Register dedicated trigger)
                trigger_val = self.store.getValues(9, 1)[0]
                if trigger_val == 1:
                    logger.info("Modbus trigger received via HR 9.")
                    self.event_bus.publish("HardwareTrigger", {})
                    self.store.setValues(9, [0])


                # Update Heartbeat every 10 iterations (approx 1 second)
                loop_count += 1
                if loop_count >= 10:
                    heartbeat = (heartbeat + 1) % 65535
                    loop_count = 0
                
                active_numeric = MODULE_CODE_MAP.get(active_str, 0)
                
                # Per-module status: 0 (Idle) until the active module has had
                # at least one triggered inspection since server start.
                status_numeric = self._module_last_status.get(active_str, 0)

                # Fetch StateStore counts for the active module
                counters = self.state_store.get_counters(active_str) if active_str else {}
                total = counters.get("total_count", 0)
                ok = counters.get("ok_count", 0)
                ng = counters.get("ng_count", 0)
                error_state = self.state_store.error_state

                with self._trigger_id_lock:
                    last_tid = self._last_trigger_id

                # HR12: Status Age in milliseconds (capped at 65535).
                # 65535 means HR1 has never been updated since server start.
                if self._last_status_ts == 0.0:
                    status_age_ms = 65535
                else:
                    status_age_ms = min(int((time.time() - self._last_status_ts) * 1000), 65535)
                
                # Build values for read-only registers only.
                # HR9 is PLC-owned (trigger), so we NEVER write it here.
                ro_0_to_8 = [
                    active_numeric,
                    status_numeric,
                    (total >> 16) & 0xFFFF, total & 0xFFFF,
                    (ok >> 16) & 0xFFFF, ok & 0xFFFF,
                    (ng >> 16) & 0xFFFF, ng & 0xFFFF,
                    heartbeat,
                ]
                self.store.setValues(0, ro_0_to_8)         # HR0..HR8

                # HR10 (Error State), HR11 (Last Trigger ID), HR12 (Status Age) — skip HR9
                self.store.setValues(10, [error_state, last_tid, status_age_ms])
                
            except Exception as e:
                logger.error(f"Modbus sync loop error: {e}")
                
            time.sleep(0.1) # 100ms poll

    def stop(self):
        """Stop modbus synchronization loop."""
        self._running = False
