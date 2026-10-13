import pytest
from unittest.mock import MagicMock
from app.core.pipeline import Pipeline
from app.core.event_bus import EventBus
from app.core.renderer_manager import RendererManager
from app.core.output_manager import OutputManager
from app.core.contracts import IModule

class DummyModule(IModule):
    id = "dummy-module"

    def initialize(self) -> bool:
        return True

    def process(self, context):
        class DummyResult:
            match = True
            is_ok = True
        return DummyResult()

    def render(self, result):
        return result

    def configure(self, settings) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def health_check(self) -> bool:
        return True

def test_pipeline_continuous_mode_preview_only_until_triggered():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("continuous")

    # Frame 1: continuous preview -> process=True, triggered=False
    pipeline.process_frame("frame_1")
    coordinator.record.assert_not_called()

    # Frame 2: still continuous preview -> process=True, triggered=False
    pipeline.process_frame("frame_2")
    coordinator.record.assert_not_called()

    # Fire trigger
    pipeline.fire_trigger()

    # Frame 3: triggered frame -> process=True, triggered=True
    pipeline.process_frame("frame_3")
    assert coordinator.record.call_count == 1
    args, kwargs = coordinator.record.call_args
    # signature: record(result, frame, module_id, trigger_id)
    assert args[2] == "dummy-module"
    assert args[3] == 1  # trigger_id = 1

    # Frame 4: subsequent frame reverts to untriggered preview
    pipeline.process_frame("frame_4")
    assert coordinator.record.call_count == 1

def test_pipeline_single_mode_processes_only_when_triggered():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("single")

    # Without trigger, single mode does not process inference
    pipeline.process_frame("frame_1")
    coordinator.record.assert_not_called()

    # Fire trigger
    pipeline.fire_trigger()
    pipeline.process_frame("frame_2")
    assert coordinator.record.call_count == 1
    assert coordinator.record.call_args[0][3] == 1  # trigger_id = 1

    # Next frame without trigger does not process
    pipeline.process_frame("frame_3")
    assert coordinator.record.call_count == 1

def test_pipeline_monotonic_trigger_id_increments():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("single")

    pipeline.fire_trigger()
    pipeline.process_frame("f1")
    assert coordinator.record.call_args[0][3] == 1

    pipeline.fire_trigger()
    pipeline.process_frame("f2")
    assert coordinator.record.call_args[0][3] == 2

    pipeline.fire_trigger()
    pipeline.process_frame("f3")
    assert coordinator.record.call_args[0][3] == 3

def test_pipeline_freeze_until_next_trigger_in_single_and_interval():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    renderer_manager.render.return_value = "rendered_frozen_output"
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    class FreezeModule(DummyModule):
        def render(self, result):
            return "rendered_frozen_output"

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(FreezeModule())
    pipeline.set_trigger_mode("single")

    # Before any trigger, raw incoming frame is broadcast
    pipeline.process_frame("frame_initial")
    assert output_manager.broadcast.call_args[0][0] == "frame_initial"

    # Trigger fired -> processes and freezes output
    pipeline.fire_trigger()
    pipeline.process_frame("frame_trigger_1")
    assert output_manager.broadcast.call_args[0][0] == "rendered_frozen_output"

    # Subsequent frames without trigger -> broadcast the frozen rendered output
    pipeline.process_frame("frame_subsequent_1")
    output_manager.broadcast.assert_called_with("rendered_frozen_output", {})

    pipeline.process_frame("frame_subsequent_2")
    output_manager.broadcast.assert_called_with("rendered_frozen_output", {})

def test_pipeline_on_raw_frame_hook_called_regardless_of_gating():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    raw_frames_seen = []
    class HookModule(DummyModule):
        def on_raw_frame(self, frame):
            raw_frames_seen.append(frame)

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(HookModule())
    pipeline.set_trigger_mode("single")

    # Single mode without trigger: inference is gated off, but on_raw_frame receives every frame
    pipeline.process_frame("raw_frame_1")
    pipeline.process_frame("raw_frame_2")
    pipeline.process_frame("raw_frame_3")

    assert raw_frames_seen == ["raw_frame_1", "raw_frame_2", "raw_frame_3"]
    coordinator.record.assert_not_called()

def test_pipeline_trigger_mode_preserved_across_module_switch():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("interval")
    pipeline.set_trigger_interval(2.5)

    assert pipeline.trigger_mode == "interval"
    assert pipeline.trigger_interval_seconds == 2.5

    # Switch to another module
    class AnotherModule(DummyModule):
        id = "another-module"

    pipeline.set_module(AnotherModule())
    assert pipeline.trigger_mode == "interval"
    assert pipeline.trigger_interval_seconds == 2.5

def test_pipeline_fire_in_continuous_mode_records_and_reverts():
    event_bus = MagicMock(spec=EventBus)
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("continuous")

    # Frames without trigger do not record
    pipeline.process_frame("f1")
    pipeline.process_frame("f2")
    assert coordinator.record.call_count == 0

    # Trigger fired in continuous mode -> records exactly once
    pipeline.fire_trigger()
    pipeline.process_frame("f3")
    assert coordinator.record.call_count == 1

    # Following frames revert to untriggered preview
    pipeline.process_frame("f4")
    pipeline.process_frame("f5")
    assert coordinator.record.call_count == 1

def test_pipeline_exactly_one_row_and_counter_per_fire_in_all_modes():
    for mode in ["single", "interval", "continuous"]:
        event_bus = MagicMock(spec=EventBus)
        renderer_manager = MagicMock(spec=RendererManager)
        output_manager = MagicMock(spec=OutputManager)
        coordinator = MagicMock()

        pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
        pipeline.set_module(DummyModule())
        pipeline.set_trigger_mode(mode)
        if mode == "interval":
            pipeline.set_trigger_interval(60.0) # Long interval so timer doesn't fire naturally

        # Untriggered frames
        pipeline.process_frame("u1")
        pipeline.process_frame("u2")
        assert coordinator.record.call_count == 0, f"Mode {mode} recorded without trigger"

        # Explicit fire
        pipeline.fire_trigger()
        pipeline.process_frame("t1")
        assert coordinator.record.call_count == 1, f"Mode {mode} did not record exactly 1 on fire"

        # Subsequent frames
        pipeline.process_frame("u3")
        pipeline.process_frame("u4")
        assert coordinator.record.call_count == 1, f"Mode {mode} recorded extra rows after fire"

def test_pipeline_hardware_trigger_event_fires_trigger():
    event_bus = EventBus()
    renderer_manager = MagicMock(spec=RendererManager)
    output_manager = MagicMock(spec=OutputManager)
    coordinator = MagicMock()

    pipeline = Pipeline(event_bus, renderer_manager, output_manager, coordinator=coordinator)
    pipeline.set_module(DummyModule())
    pipeline.set_trigger_mode("single")

    assert pipeline._pending_trigger is False
    event_bus.publish("HardwareTrigger", {})
    assert pipeline._pending_trigger is True

