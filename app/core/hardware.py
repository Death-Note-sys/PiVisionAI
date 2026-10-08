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
        
        # 0: Heartbeat, 1: Trigger, 2-3: Total, 4-5: OK, 6-7: NG, 8: Error
        self.store = ModbusSequentialDataBlock(0, [0] * 10)
        self.context = ModbusServerContext(slaves=ModbusSlaveContext(
            di=self.store, co=self.store, hr=self.store, ir=self.store
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
                # Read Trigger Register (Address 1 in 0-indexed block)
                # fx=3 is holding registers, but directly on DataBlock we don't pass fx
                trigger_val = self.store.getValues(1, 1)[0]
                if trigger_val == 1:
                    logger.info("Modbus trigger received.")
                    self.event_bus.publish("HardwareTrigger", {})
                    # Reset trigger to 0
                    self.store.setValues(1, [0])
                    
                # Update Heartbeat every 10 iterations (approx 1 second)
                loop_count += 1
                if loop_count >= 10:
                    heartbeat = (heartbeat + 1) % 65535
                    loop_count = 0
                
                # Fetch StateStore counts
                state = self.state_store.get_global_state()
                total = state.get("total_count", 0)
                ok = state.get("ok_count", 0)
                ng = state.get("ng_count", 0)
                error_state = state.get("error_state", 0)
                
                # Big-Endian mapping
                values = [
                    heartbeat,
                    self.store.getValues(1, 1)[0], # Keep trigger intact if not 1
                    (total >> 16) & 0xFFFF, total & 0xFFFF,
                    (ok >> 16) & 0xFFFF, ok & 0xFFFF,
                    (ng >> 16) & 0xFFFF, ng & 0xFFFF,
                    error_state
                ]
                
                self.store.setValues(0, values)
                
            except Exception as e:
                logger.error(f"Modbus sync loop error: {e}")
                
            time.sleep(0.1) # 100ms poll
