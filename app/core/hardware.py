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

class HardwareManager:
    """Manages Modbus TCP Server and synchronizes with StateStore."""
    def __init__(self, config: ConfigService, event_bus: EventBus, state_store: StateStore):
        self.config = config
        self.event_bus = event_bus
        self.state_store = state_store
        
        settings = self.config.get_settings()
        self.port = settings.modbus_port
        self.enabled = settings.modbus_enabled
        
        self._modbus_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 0: Active Module ID, 1: Overall Status, 2-3: Total, 4-5: OK, 6-7: NG, 8: Heartbeat, 9: Trigger, 10: Error
        self.store = ModbusSequentialDataBlock(0, [0] * 32)
        self.coil_store = ModbusSequentialDataBlock(0, [0] * 32)
        self.context = ModbusServerContext(slaves=ModbusSlaveContext(
            di=self.coil_store, co=self.coil_store, hr=self.store, ir=self.store,
            zero_mode=True
        ), single=True)
        
    def initialize(self) -> bool:
        if not self.enabled:
            logger.info("Modbus is disabled in configuration.")
            return True
            
        self._running = True
        
        # Start the Modbus sync thread
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        
        # Start the Modbus TCP server thread
        self._modbus_thread = threading.Thread(target=self._run_server, daemon=True)
        self._modbus_thread.start()
        logger.info(f"HardwareManager initialized Modbus on port {self.port}")
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
            StartTcpServer(context=self.context, identity=identity, address=("0.0.0.0", self.port))
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
                
                module_map = {
                    "core-ocr": 1,
                    "core-color-analysis": 2,
                    "core-ai-identify": 3,
                    "core-object-detection": 4,
                    "core-measurement": 5
                }
                active_numeric = module_map.get(active_str, 0)
                
                # Fetch StateStore counts for the active module
                state = self.state_store.get_global_state()
                status_numeric = state.get("latest_status", 0)
                counters = self.state_store.get_counters(active_str) if hasattr(self.state_store, "get_counters") else state
                total = counters.get("total_count", 0)
                ok = counters.get("ok_count", 0)
                ng = counters.get("ng_count", 0)
                error_state = state.get("error_state", 0)
                
                # Big-Endian mapping
                values = [
                    active_numeric,
                    status_numeric,
                    (total >> 16) & 0xFFFF, total & 0xFFFF,
                    (ok >> 16) & 0xFFFF, ok & 0xFFFF,
                    (ng >> 16) & 0xFFFF, ng & 0xFFFF,
                    heartbeat,
                    0, # trigger auto-resets to 0
                    error_state
                ]
                
                self.store.setValues(0, values)
                
            except Exception as e:
                logger.error(f"Modbus sync loop error: {e}")
                
            time.sleep(0.1) # 100ms poll

    def stop(self):
        """Stop modbus synchronization loop."""
        self._running = False
