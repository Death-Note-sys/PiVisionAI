git reset

# Phase 1: Project Restructuring and Core Hardware Abstractions
git add .gitignore app/api/ app/core/base_module.py app/core/camera_manager.py app/core/config_service.py app/core/hardware.py app/core/models/ app/core/module_controller.py app/core/performance_monitor.py app/core/pipeline.py app/core/session_manager.py app/services/ app.py config.py core/ modules/ test_api.py test_thread.py boot.log
git commit -m "Phase 1: Project Restructuring and Core Hardware Abstractions"

# Phase 2: Generic Persistence Layer (SQLite)
git add app/core/state_store.py app/core/state.py app/core/container.py
git commit -m "Phase 2: Generic Persistence Layer (SQLite)"

# Phase 3: AI Identify overhaul (references routes, persistence integration)
git add app/api/v1/ai_identify.py app/modules/ai_identify/
git commit -m "Phase 3: AI Identify overhaul (references routes, persistence integration)"

# Phase 4: Per-module judgment rules (Measurement ROI, minAreaRect, Object Detection count_min, OCR regex)
git add app/modules/measurement/ app/modules/object_detection/ app/modules/ocr/
git commit -m "Phase 4: Per-module judgment rules (Measurement ROI and tolerance, Object Detection counts, OCR regex)"

# Phase 5: Modbus Polling Updates and Map adjustments
git add docs/modbus_map.md app/core/coordinator.py app/core/output_manager.py
git commit -m "Phase 5: Modbus Polling Updates and Map adjustments"

# Phase 6: Color Analysis CIELAB float32 fix
git add app/modules/color_analysis/
git commit -m "Phase 6: Color Analysis CIELAB float32 fix"

# Phase 7: Frontend UI updates (counters)
git add templates/ static/
git commit -m "Phase 7: Frontend UI updates (counters)"

# Phase 8: Cleanup and final test fixes
git add .
git commit -m "Phase 8: Cleanup and final test fixes"
