import pytest
from unittest.mock import MagicMock
from app.core.module_controller import ModuleController


class FakeModule:
    def __init__(self, event_bus=None, ai_runtime=None, model_registry=None):
        self.initialized = False
        self.configured_with = None

    def initialize(self):
        self.initialized = True
        return True

    def configure(self, settings):
        self.configured_with = settings
        return True

    def cleanup(self):
        self.cleaned_up = True


class FakeService:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True
        return True


class FakePlugin:
    """Stand-in for an IPlugin factory, mirroring ObjectDetectionPlugin's shape."""
    def __init__(self, event_bus, ai_runtime, model_registry):
        self.event_bus = event_bus
        self.ai_runtime = ai_runtime
        self.model_registry = model_registry
        self._module = None

    def create_module(self):
        self._module = FakeModule()
        return self._module

    def create_service(self):
        return FakeService()


@pytest.fixture
def controller():
    plugin_manager = MagicMock()
    pipeline = MagicMock()
    event_bus = MagicMock()
    ai_runtime = MagicMock()
    model_registry = MagicMock()
    return ModuleController(plugin_manager, pipeline, event_bus, ai_runtime, model_registry)


def test_switch_module_unknown_id_returns_false(controller):
    controller.plugin_manager.registry.get.return_value = None

    result = controller.switch_module("does-not-exist")

    assert result is False
    assert controller.active_module_instance is None


def test_switch_module_success_uses_plugin_factory(controller):
    fake_meta = MagicMock()
    fake_meta.id = "core-object-detection"
    fake_meta.name = "Object Detection"
    controller.plugin_manager.registry.get.return_value = fake_meta
    controller.plugin_manager.load_plugin_class.return_value = FakePlugin

    result = controller.switch_module("core-object-detection")

    assert result is True
    assert isinstance(controller.active_module_instance, FakeModule)
    assert controller.active_module_instance.initialized is True
    assert isinstance(controller.active_service_instance, FakeService)
    controller.pipeline.set_module.assert_called_once_with(controller.active_module_instance)
    controller.event_bus.publish.assert_called_with(
        "ModuleLoaded", {"module_id": "core-object-detection", "name": "Object Detection"}
    )


def test_switch_module_instantiation_failure_returns_false(controller):
    fake_meta = MagicMock()
    controller.plugin_manager.registry.get.return_value = fake_meta

    class BrokenPlugin:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    controller.plugin_manager.load_plugin_class.return_value = BrokenPlugin

    result = controller.switch_module("core-object-detection")

    assert result is False
    assert controller.active_module_instance is None


def test_get_active_service_returns_none_before_any_switch(controller):
    assert controller.get_active_service() is None


def test_update_settings_calls_configure_not_update_settings(controller):
    fake_meta = MagicMock()
    controller.plugin_manager.registry.get.return_value = fake_meta
    controller.plugin_manager.load_plugin_class.return_value = FakePlugin
    controller.switch_module("core-object-detection")

    result = controller.update_settings({"confidence": 0.9})

    assert result is True
    assert controller.active_module_instance.configured_with == {"confidence": 0.9}


def test_switch_module_unloads_previous_module_first(controller):
    fake_meta = MagicMock()
    controller.plugin_manager.registry.get.return_value = fake_meta
    controller.plugin_manager.load_plugin_class.return_value = FakePlugin

    controller.switch_module("core-object-detection")
    first_module = controller.active_module_instance

    controller.switch_module("core-object-detection")

    assert hasattr(first_module, "cleaned_up")
    assert controller.active_module_instance is not first_module
