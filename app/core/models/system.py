from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class CameraInfo(BaseModel):
    """Metadata about a connected camera."""
    index: int
    name: str
    resolution: str
    fps: float
    is_connected: bool
    backend: str = Field(default="OpenCV")


class PerformanceMetrics(BaseModel):
    """System-level performance tracking."""
    fps: float = 0.0
    cpu_usage_percent: float = 0.0
    ram_usage_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_usage_percent: float = 0.0
    vram_usage_gb: float = 0.0
    disk_usage_percent: float = 0.0
    net_rx_mbps: float = 0.0
    net_tx_mbps: float = 0.0
    uptime_seconds: float = 0.0
    capture_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0


class AIBackend(BaseModel):
    """Information about an available AI Execution Provider."""
    name: str
    provider: str
    is_available: bool
    device_name: Optional[str] = None
    priority: int


class ModuleMetadata(BaseModel):
    """Manifest metadata for a plugin/module."""
    id: str
    name: str
    version: str
    author: str
    description: str
    category: str = "Vision"
    task: str = "General"
    entry_point: str
    renderer: str = "default"
    api_version: str = "v1"
    supported_camera_types: List[str] = Field(default_factory=list)
    supported_results: List[str] = Field(default_factory=list)
    required_models: List[str] = Field(default_factory=list)
    required_backends: List[str] = Field(default_factory=list)
    optional_backends: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    settings_schema: Dict[str, Any] = Field(default_factory=dict)
    minimum_framework_version: str = "1.0.0"
    license: str = "MIT"
    source_dir: str = ""

class ModelMetadata(BaseModel):
    """Metadata for an AI model."""
    id: str
    name: str
    version: str
    framework: str = Field(..., description="ONNX, PyTorch, OpenVINO, ultralytics")
    task: str = Field(..., description="Object Detection, Segmentation, OCR")
    format: str = ""
    input_size: List[int] = Field(default_factory=list)
    output_format: str = ""
    classes: List[str] = Field(default_factory=list)
    backend: str = ""
    precision: str = "FP32"
    license: str = "MIT"
    author: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    parameters: int = 0
    framework_version: str = ""
    supported_backends: List[str] = Field(default_factory=list)
    recommended_backend: str = ""
    minimum_framework_version: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict) # mAP50, f1_score, etc
    hardware_requirements: Dict[str, str] = Field(default_factory=dict)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionMetadata(BaseModel):
    """Metadata for an application session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    loaded_modules: List[str] = Field(default_factory=list)
    performance_summary: Dict[str, Any] = Field(default_factory=dict)

class SystemStatus(BaseModel):
    """Generic status response for the /api/v1/system/status endpoint."""
    status: str = "running"
    camera: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    backend: Optional[Dict[str, Any]] = None
    session: Dict[str, Any] = Field(default_factory=dict)
    
    is_recording: bool = False
    trigger_mode: str = "continuous"
    trigger_interval_seconds: float = 5.0
    pending_trigger: bool = False
    last_trigger_time: float = 0.0
    
    total_count: int = 0
    ok_count: int = 0
    ng_count: int = 0
    error_state: int = 0
    latest_ng_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    modbus_enabled: bool = False
    modbus_port: int = 5020
    
    active_module_id: Optional[str] = None
    module_state: Dict[str, Any] = Field(default_factory=dict)
