document.addEventListener('DOMContentLoaded', () => {
    
    // --- Elements ---
    const cameraFeed = document.getElementById('main-camera-feed');
    const offlinePlaceholder = document.getElementById('offline-placeholder');
    const globalStatus = document.getElementById('global-status');
    const camStatusBadge = document.getElementById('cam-status-badge');
    
    // Metrics
    const metricFps = document.getElementById('metric-fps');
    const metricCpu = document.getElementById('metric-cpu');
    const barCpu = document.getElementById('bar-cpu');
    const metricMem = document.getElementById('metric-mem');
    const barMem = document.getElementById('bar-mem');
    
    // Modules
    const moduleLinks = document.querySelectorAll('.module-item .nav-link');
    const activeModulesContainer = document.getElementById('active-modules-container');
    const activeModulesMap = new Map();

    // Buttons
    const btnScreenshot = document.getElementById('btn-screenshot');
    const btnRecord = document.getElementById('btn-record');
    const btnStopRecord = document.getElementById('btn-stop-record');
    const btnDownload = document.getElementById('btn-download');

    // Toast
    const toastEl = document.getElementById('liveToast');
    const toastMessage = document.getElementById('toast-message');
    // Settings Panel
    const settingsPanel = document.getElementById('module-settings-panel');
    const settingsContainer = document.getElementById('module-settings-container');
    
    // Camera Dropdown
    const cameraSelector = document.getElementById('camera-selector');
    const cameraTitleText = document.getElementById('camera-title-text');
    
    // --- State ---
    let isRecording = false;
    let availableModulesData = []; // Store fetched metadata

    // --- Helper Functions ---
    const toastContainer = document.querySelector('.toast-container');
    const showToast = (message, type='primary') => {
        const toastHtml = `
        <div class="toast align-items-center text-bg-${type} border-0 show" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>`;
        const wrap = document.createElement('div');
        wrap.innerHTML = toastHtml;
        const toastEl = wrap.firstElementChild;
        if(toastContainer) toastContainer.appendChild(toastEl);
        
        setTimeout(() => {
            toastEl.classList.remove('show');
            setTimeout(() => toastEl.remove(), 300);
        }, 3000);
    };

    // --- Initial Fetch of Modules ---
    async function fetchModules() {
        try {
            const res = await fetch('/api/modules');
            availableModulesData = await res.json();
            
            // Highlight active if any
            const activeMod = availableModulesData.find(m => m.is_active);
            if (activeMod) {
                activateUIForModule(activeMod.id, activeMod);
            }
        } catch (e) {
            console.error("Failed to fetch modules", e);
        }
    }
    // --- Initial Fetch of Cameras ---
    async function fetchCameras() {
        try {
            const res = await fetch('/api/cameras');
            const data = await res.json();
            
            if (data.available && data.available.length > 0) {
                cameraSelector.innerHTML = '';
                data.available.forEach(cam => {
                    const option = document.createElement('option');
                    option.value = cam.index;
                    option.textContent = cam.name;
                    // Auto select the current one
                    if (data.current && data.current.index == cam.index) {
                        option.selected = true;
                    }
                    cameraSelector.appendChild(option);
                });
                cameraSelector.classList.remove('d-none');
            }
        } catch (e) {
            console.error("Failed to fetch cameras", e);
        }
    }

    cameraSelector.addEventListener('change', async (e) => {
        const index = e.target.value;
        try {
            await fetch('/api/cameras/switch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ index: parseInt(index) })
            });
            showToast(`Switching to Camera ${index}...`, 'info');
        } catch(err) {
            console.error("Failed to switch camera", err);
        }
    });

    // --- View Navigation (SPA) ---
    const viewTabs = document.querySelectorAll('.view-tab');
    const viewSections = document.querySelectorAll('.view-section');
    viewTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            viewTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const targetId = tab.getAttribute('data-target');
            viewSections.forEach(sec => {
                if(sec.id === targetId) {
                    sec.classList.remove('d-none');
                    sec.classList.add('d-flex'); // or block if not flex
                } else {
                    sec.classList.add('d-none');
                    sec.classList.remove('d-flex');
                }
            });
            
            if (targetId === 'analytics-view') {
                initCharts();
            }
        });
    });

    // --- Settings & Theme ---
    const selTheme = document.getElementById('setTheme');
    if (selTheme) {
        selTheme.addEventListener('change', (e) => {
            document.body.className = e.target.value;
        });
    }

    fetchModules();
    fetchCameras();

    // --- Module Toggling ---
    moduleLinks.forEach(link => {
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            const moduleName = link.getAttribute('data-module');
            
            // Backend Activate API Call
            let targetModule = moduleName;
            if (activeModulesMap.has(moduleName)) {
                targetModule = ""; // Deactivate
            }

            try {
                const res = await fetch('/api/modules/activate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ module: targetModule })
                });
                if(res.ok) {
                    if (targetModule === "") {
                        deactivateAllUI();
                    } else {
                        // Find meta
                        const meta = availableModulesData.find(m => m.id === moduleName) || { name: moduleName, settings: {} };
                        activateUIForModule(moduleName, meta, link);
                    }
                }
            } catch (err) {
                console.error("Failed to switch module", err);
            }
        });
    });

    function deactivateAllUI() {
        activeModulesMap.clear();
        moduleLinks.forEach(l => {
            l.classList.remove('active');
            const badge = l.querySelector('.badge');
            if(badge) {
                badge.textContent = 'Off';
                badge.classList.remove('bg-primary', 'on');
                badge.classList.add('bg-secondary');
            }
        });
        updateActiveModulesList();
        renderSettings(null);
    }

    function activateUIForModule(modId, meta, linkElement=null) {
        deactivateAllUI(); // Ensure single active
        
        // Try to find link if not provided
        if (!linkElement) {
            linkElement = Array.from(moduleLinks).find(l => l.getAttribute('data-module') === modId);
        }

        let moduleText = meta.name || modId;
        let iconHtml = '<i class="fa-solid fa-cube"></i>';
        
        if (linkElement) {
            moduleText = linkElement.textContent.trim().replace('Off', '').replace('On', '').trim();
            iconHtml = linkElement.querySelector('i').outerHTML;
            linkElement.classList.add('active');
            const badge = linkElement.querySelector('.badge');
            if(badge) {
                badge.textContent = 'On';
                badge.classList.remove('bg-secondary');
                badge.classList.add('bg-primary', 'on');
            }
        }
        
        activeModulesMap.set(modId, { text: moduleText, icon: iconHtml });
        updateActiveModulesList();
        renderSettings(meta);
    }

    function renderSettings(meta) {
        if (!meta || !meta.settings || Object.keys(meta.settings).length === 0) {
            settingsPanel.classList.add('d-none');
            settingsPanel.classList.remove('d-flex');
            settingsContainer.innerHTML = '';
            return;
        }

        settingsPanel.classList.remove('d-none');
        settingsPanel.classList.add('d-flex');
        settingsContainer.innerHTML = '';

        for (const [key, config] of Object.entries(meta.settings)) {
            if (config.type === 'slider') {
                const wrapper = document.createElement('div');
                wrapper.className = 'mb-3';
                wrapper.innerHTML = `
                    <label class="form-label d-flex justify-content-between text-muted small">
                        <span class="text-capitalize">${key}</span>
                        <span id="val-${key}" class="text-accent fw-bold">${config.default}</span>
                    </label>
                    <input type="range" class="form-range" id="setting-${key}" 
                           min="${config.min}" max="${config.max}" step="${config.step}" value="${config.default}">
                `;
                settingsContainer.appendChild(wrapper);

                const slider = wrapper.querySelector(`#setting-${key}`);
                const valDisplay = wrapper.querySelector(`#val-${key}`);
                
                slider.addEventListener('input', (e) => {
                    valDisplay.textContent = e.target.value;
                });

                slider.addEventListener('change', async (e) => {
                    const payload = {};
                    payload[key] = e.target.value;
                    try {
                        await fetch('/api/modules/settings', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(payload)
                        });
                    } catch(err) {
                        console.error("Setting update failed", err);
                    }
                });
            } else if (config.type === 'select') {
                const wrapper = document.createElement('div');
                wrapper.className = 'mb-3';
                let optionsHtml = '';
                config.options.forEach(opt => {
                    const selected = opt === config.default ? 'selected' : '';
                    optionsHtml += `<option value="${opt}" ${selected}>${opt}</option>`;
                });
                
                wrapper.innerHTML = `
                    <label class="form-label text-muted small text-capitalize">${key.replace('_', ' ')}</label>
                    <select class="form-select form-select-sm bg-dark text-light border-secondary" id="setting-${key}">
                        ${optionsHtml}
                    </select>
                `;
                settingsContainer.appendChild(wrapper);

                const selectEl = wrapper.querySelector(`#setting-${key}`);
                selectEl.addEventListener('change', async (e) => {
                    const payload = {};
                    payload[key] = e.target.value;
                    try {
                        await fetch('/api/modules/settings', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(payload)
                        });
                    } catch(err) {
                        console.error("Setting update failed", err);
                    }
                });
            }
        }
    }

    function updateActiveModulesList() {
        activeModulesContainer.innerHTML = '';
        if (activeModulesMap.size === 0) {
            activeModulesContainer.innerHTML = '<li class="text-muted text-center py-4 empty-state">No modules active</li>';
            return;
        }

        activeModulesMap.forEach((data, id) => {
            const li = document.createElement('li');
            li.innerHTML = `${data.icon} <span>${data.text}</span>`;
            activeModulesContainer.appendChild(li);
        });
    }

    // --- Actions ---
    btnScreenshot.addEventListener('click', async () => {
        const fmt = document.getElementById('screenshotFormat')?.value || 'PNG';
        const feed = document.getElementById('mediaFeed')?.value || 'Processed Feed';
        
        try {
            const res = await fetch('/api/media/screenshot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ format: fmt, quality: 100, feed: feed })
            });
            const data = await res.json();
            if(data.success) {
                showToast(`Screenshot saved as ${data.filename}`, 'success');
            } else {
                showToast('Screenshot failed: ' + data.error, 'danger');
            }
        } catch(err) {
            console.error(err);
            showToast('Screenshot request failed', 'danger');
        }
    });

    btnRecord.addEventListener('click', async () => {
        const codec = document.getElementById('videoCodec')?.value || 'H264';
        const fmt = (codec === 'XVID' || codec === 'MJPEG') ? 'AVI' : 'MP4';
        const feed = document.getElementById('mediaFeed')?.value || 'Processed Feed';
        
        try {
            const res = await fetch('/api/media/record/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ format: fmt, codec: codec, feed: feed })
            });
            const data = await res.json();
            if(data.success) {
                isRecording = true;
                btnRecord.classList.add('d-none');
                btnStopRecord.classList.remove('d-none');
                showToast('Recording started...', 'danger');
            } else {
                showToast('Failed to start recording: ' + data.error, 'danger');
            }
        } catch(err) {
            console.error(err);
            showToast('Recording request failed', 'danger');
        }
    });

    btnStopRecord.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/media/record/stop', { method: 'POST' });
            const data = await res.json();
            if(data.success) {
                isRecording = false;
                btnStopRecord.classList.add('d-none');
                btnRecord.classList.remove('d-none');
                showToast(`Recording saved. Duration: ${data.duration}s`, 'success');
            } else {
                showToast('Failed to stop recording', 'danger');
            }
        } catch(err) {
            console.error(err);
        }
    });

    btnDownload.addEventListener('click', async () => {
        showToast('Exporting data...', 'info');
        const fmt = document.getElementById('exportFormat')?.value || 'CSV';
        
        try {
            let res = await fetch('/api/export', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ format: fmt })
            });
            let data = await res.json();
            
            if (data.success) {
                showToast(`Data exported as ${data.filename}! Check Gallery.`, 'success');
                return;
            }
            
            // Fallback for older modules
            res = await fetch('/api/modules/interact', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: 'export' })
            });
            data = await res.json();
            if (data.type === 'download') {
                const blob = new Blob([data.data], { type: data.mimetype || 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = data.filename;
                a.click();
                window.URL.revokeObjectURL(url);
                showToast('Data exported successfully!', 'success');
            } else {
                showToast('No export data available for this module.', 'warning');
            }
        } catch(err) {
            console.error("Export failed", err);
            showToast('Export failed.', 'danger');
        }
    });
    
    // --- Keyboard Shortcuts ---
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        
        if (e.key.toLowerCase() === 's' && !e.ctrlKey) {
            btnScreenshot.click();
        } else if (e.key.toLowerCase() === 'r' && !e.ctrlKey) {
            if (!isRecording) btnRecord.click();
        } else if (e.key === 'Escape') {
            if (isRecording) btnStopRecord.click();
        } else if (e.ctrlKey && e.key >= '1' && e.key <= '5') {
            e.preventDefault();
            const mods = [null, 'object-detection', 'measurement', 'ocr', 'color-detector', 'face-detection'];
            const modId = mods[parseInt(e.key)];
            if (modId) {
                const link = Array.from(moduleLinks).find(l => l.getAttribute('data-module') === modId);
                if (link) link.click();
            }
        }
    });

    // --- Camera Feed Interactions ---
    cameraFeed.addEventListener('click', async (e) => {
        // Calculate relative coordinates (0.0 to 1.0)
        const rect = cameraFeed.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        
        try {
            await fetch('/api/modules/interact', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: 'click', x: x, y: y })
            });
        } catch(err) {
            console.error("Interaction failed", err);
        }
    });

    // --- Status and Real-time Telemetry Polling ---
    async function checkSystemStatus() {
        try {
            // Fetch real backend status
            const response = await fetch('/api/status');
            const data = await response.json();

            if (data.status === 'running') {
                globalStatus.classList.remove('offline');
                globalStatus.classList.add('online');
                globalStatus.querySelector('.status-text').textContent = 'System Online';
                
                if (data.camera_active) {
                    offlinePlaceholder.style.display = 'none';
                    if (cameraFeed.style.display === 'none' || !cameraFeed.src.includes('?')) {
                        cameraFeed.src = "/video_feed?t=" + new Date().getTime();
                    }
                    cameraFeed.style.display = 'block';
                    camStatusBadge.textContent = 'Live';
                    camStatusBadge.className = 'badge bg-success';
                    
                    // Update Camera Header Text with Resolution
                    cameraTitleText.textContent = `Cam ${data.camera_index} (${data.resolution})`;
                    
                    // Update Real FPS
                    metricFps.textContent = data.fps;
                    
                    // Handle dynamic module data like color history
                    if (data.module_metadata && data.module_metadata.module_data) {
                        const mdata = data.module_metadata.module_data;
                        if (mdata.type === 'color_history') {
                            renderColorHistory(mdata);
                        }
                    }
                } else {
                    offlinePlaceholder.style.display = 'flex';
                    cameraFeed.style.display = 'none';
                    camStatusBadge.textContent = 'Offline';
                    camStatusBadge.className = 'badge bg-danger';
                    cameraTitleText.textContent = `Primary Feed`;
                    metricFps.textContent = "0.0";
                }
                
                // Real Telemetry
                if (data.telemetry) {
                    const tel = data.telemetry;
                    metricCpu.textContent = `${tel.cpu_usage}%`;
                    barCpu.style.width = `${tel.cpu_usage}%`;
                    if(tel.cpu_usage > 80) barCpu.className = 'progress-bar bg-danger';
                    else if(tel.cpu_usage > 50) barCpu.className = 'progress-bar bg-warning';
                    else barCpu.className = 'progress-bar bg-accent';
                    
                    const memPct = Math.round((tel.ram_usage / (tel.ram_total || 1)) * 100);
                    metricMem.textContent = `${tel.ram_usage} GB / ${tel.ram_total} GB`;
                    barMem.style.width = `${memPct}%`;
                    
                    document.getElementById('stat-net-rx').textContent = `${tel.net_speed_rx} Mbps`;
                    document.getElementById('stat-net-tx').textContent = `${tel.net_speed_tx} Mbps`;
                    
                    const diskPct = Math.round((tel.disk_usage / (tel.disk_total || 1)) * 100);
                    document.getElementById('stat-disk').textContent = `${diskPct}%`;
                    
                    // Format Uptime
                    let upStr = `${tel.uptime}s`;
                    if(tel.uptime > 60) upStr = `${Math.floor(tel.uptime/60)}m ${tel.uptime%60}s`;
                    if(tel.uptime > 3600) upStr = `${Math.floor(tel.uptime/3600)}h ${Math.floor((tel.uptime%3600)/60)}m`;
                    document.getElementById('stat-uptime').textContent = upStr;
                    
                    updateCharts(tel, data.analytics);
                }
                
                // Real Analytics
                if (data.analytics) {
                    const a = data.analytics;
                    document.getElementById('stat-objects').textContent = a.objects_detected || 0;
                    document.getElementById('stat-ocr').textContent = a.ocr_reads || 0;
                    document.getElementById('stat-barcode').textContent = a.barcode_scans || 0;
                    document.getElementById('stat-inf').textContent = `${a.avg_inference_time || 0}ms`;
                }

            }
        } catch (error) {
            console.error('Error fetching status:', error);
            globalStatus.classList.remove('online');
            globalStatus.classList.add('offline');
            globalStatus.querySelector('.status-text').textContent = 'System Offline';
            
            offlinePlaceholder.style.display = 'flex';
            cameraFeed.style.display = 'none';
            camStatusBadge.textContent = 'Offline';
            camStatusBadge.className = 'badge bg-danger';
            cameraTitleText.textContent = `Primary Feed`;
            metricFps.textContent = "0.0";
        }
    }

    // --- Charts ---
    let telemetryChart = null;
    let moduleChart = null;
    let chartTimeLabel = 0;
    
    function initCharts() {
        const ctxTel = document.getElementById('telemetryChart');
        if (ctxTel && !telemetryChart) {
            telemetryChart = new Chart(ctxTel, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'CPU %', data: [], borderColor: '#ef4444', tension: 0.4 },
                        { label: 'RAM (GB)', data: [], borderColor: '#3b82f6', tension: 0.4 },
                        { label: 'GPU %', data: [], borderColor: '#10b981', tension: 0.4 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 0 } } }
            });
        }
        
        const ctxMod = document.getElementById('moduleChart');
        if (ctxMod && !moduleChart) {
            moduleChart = new Chart(ctxMod, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{ data: [], backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9'] }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
    }

    function updateCharts(tel, analytics) {
        if (!telemetryChart) return;
        
        chartTimeLabel++;
        telemetryChart.data.labels.push(chartTimeLabel);
        telemetryChart.data.datasets[0].data.push(tel.cpu_usage);
        telemetryChart.data.datasets[1].data.push(tel.ram_usage);
        telemetryChart.data.datasets[2].data.push(tel.gpu_usage);
        
        if (telemetryChart.data.labels.length > 30) {
            telemetryChart.data.labels.shift();
            telemetryChart.data.datasets.forEach(ds => ds.data.shift());
        }
        telemetryChart.update();
        
        if (moduleChart && analytics && analytics.module_usage) {
            const labels = Object.keys(analytics.module_usage);
            const data = Object.values(analytics.module_usage);
            moduleChart.data.labels = labels;
            moduleChart.data.datasets[0].data = data;
            moduleChart.update();
        }
    }

    function renderColorHistory(mdata) {
        settingsPanel.classList.remove('d-none');
        settingsPanel.classList.add('d-flex');
        
        let html = `<div class="mb-3">
            <label class="form-label text-muted small">Dominant Color</label>
            <div class="d-flex align-items-center gap-2">
                <div style="width: 24px; height: 24px; border-radius: 4px; background-color: ${mdata.dominant_hex}"></div>
                <span class="text-light">${mdata.dominant_hex}</span>
            </div>
        </div>
        <hr class="border-secondary">
        <label class="form-label text-muted small mb-2 d-flex justify-content-between">
            <span>Color History</span>
            <span class="badge bg-secondary">${mdata.history.length}</span>
        </label>
        <div class="color-history-list d-flex flex-column gap-2" style="max-height: 250px; overflow-y: auto;">`;
        
        if (mdata.history.length === 0) {
            html += `<div class="text-muted small">Click the video feed to sample colors.</div>`;
        } else {
            mdata.history.forEach(c => {
                html += `
                <div class="glass-panel p-2 d-flex align-items-center gap-3">
                    <div style="width: 32px; height: 32px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.2); background-color: ${c.hex}"></div>
                    <div class="small w-100">
                        <div class="d-flex justify-content-between mb-1">
                            <span class="fw-bold">${c.hex}</span>
                            <span class="text-muted">${c.css}</span>
                        </div>
                        <div class="text-muted" style="font-size: 0.75rem;">${c.rgb} &bull; ${c.hsv}</div>
                    </div>
                </div>`;
            });
        }
        html += `</div>`;
        settingsContainer.innerHTML = html;
    }

    // Initialize loops
    setInterval(checkSystemStatus, 1000); 
    
    // Initial calls
    checkSystemStatus();
});
