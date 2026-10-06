document.addEventListener('DOMContentLoaded', () => {
    const videoUpload = document.getElementById('video-upload');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const errorMsg = document.getElementById('error-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsSection = document.getElementById('results-section');

    let selectedFile = null;

    videoUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        errorMsg.classList.add('hidden');
        resultsSection.classList.add('hidden');
        
        if (!file) return;

        // Validation
        if (file.type !== 'video/mp4') {
            showError("Only MP4 video files are supported.");
            videoUpload.value = '';
            selectedFile = null;
            updateFileUI();
            return;
        }

        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > 100) {
            showError(`File exceeds the 100 MB maximum size. (Current: ${sizeMB.toFixed(1)} MB)`);
            videoUpload.value = '';
            selectedFile = null;
            updateFileUI();
            return;
        }

        selectedFile = file;
        updateFileUI();
    });

    function updateFileUI() {
        if (selectedFile) {
            fileName.textContent = selectedFile.name;
            fileSize.textContent = (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB';
            fileInfo.classList.remove('hidden');
            analyzeBtn.disabled = false;
        } else {
            fileInfo.classList.add('hidden');
            analyzeBtn.disabled = true;
        }
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI state
        analyzeBtn.disabled = true;
        loadingIndicator.classList.remove('hidden');
        errorMsg.classList.add('hidden');
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('video', selectedFile);

        try {
            const response = await fetch('/api/module4/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Analysis failed');
            }

            populateResults(data);
            resultsSection.classList.remove('hidden');
        } catch (err) {
            showError(err.message);
        } finally {
            analyzeBtn.disabled = false;
            loadingIndicator.classList.add('hidden');
        }
    });

    function populateResults(data) {
        // Video Summary
        document.getElementById('res-filename').textContent = data.summary.filename;
        document.getElementById('res-processing-time').textContent = `${data.summary.processing_time.toFixed(2)}s`;

        // AI Detection
        setDetection('person', data.summary.person_detected, data.summary.person_count);
        setDetection('lighton', data.summary.light_on_detected, data.summary.light_on_count);
        setDetection('lightoff', data.summary.light_off_detected, data.summary.light_off_count);

        // Occupancy
        document.getElementById('res-person-tracks').textContent = data.summary.person_tracks;
        const occStatus = document.getElementById('res-occupancy-status');
        if (data.summary.person_detected) {
            occStatus.textContent = "Occupied";
            occStatus.style.color = "var(--accent-success)";
        } else {
            occStatus.textContent = "Unoccupied";
            occStatus.style.color = "var(--text-secondary)";
        }

        // Final Event
        const eventTitle = document.getElementById('res-event-title');
        const eventDetails = document.getElementById('event-details');
        const noEventDetails = document.getElementById('no-event-details');
        
        const incidentContent = document.getElementById('incident-content');
        const reviewContent = document.getElementById('review-content');

        if (data.event) {
            eventTitle.textContent = "POTENTIAL ENERGY WASTAGE DETECTED";
            eventTitle.classList.remove('no-event');
            eventDetails.classList.remove('hidden');
            noEventDetails.classList.add('hidden');

            document.getElementById('res-event-type').textContent = data.event.event_type || 'POTENTIAL_ENERGY_WASTAGE';
            document.getElementById('res-event-timestamp').textContent = data.event.timestamp;
            document.getElementById('res-event-confidence').textContent = data.event.confidence.toFixed(2);
            document.getElementById('res-event-rule').textContent = data.event.rule ? data.event.rule.name : 'Temporal wastage rule';

            // Incident UI
            incidentContent.innerHTML = `
                <div class="detail-row"><span class="label">Status:</span><span class="value" style="color:var(--accent-warning)">Pending Human Review</span></div>
                <div class="detail-row"><span class="label">Event ID:</span><span class="value">${data.event.event_id}</span></div>
                <div class="detail-row"><span class="label">Camera:</span><span class="value">${data.event.camera_id}</span></div>
            `;

            // Review UI
            reviewContent.innerHTML = `
                <p><strong>Review Status:</strong> PENDING</p>
                <div class="review-actions">
                    <button class="btn-sm btn-confirm" onclick="alert('Confirmed! (Local UI state only)')">Confirm</button>
                    <button class="btn-sm btn-dismiss" onclick="alert('Dismissed! (Local UI state only)')">Dismiss</button>
                    <button class="btn-sm btn-uncertain" onclick="alert('Marked as Uncertain (Local UI state only)')">Uncertain</button>
                </div>
            `;
        } else {
            eventTitle.textContent = "NO ENERGY WASTAGE EVENT DETECTED";
            eventTitle.classList.add('no-event');
            eventDetails.classList.add('hidden');
            noEventDetails.classList.remove('hidden');

            incidentContent.innerHTML = `<p>No incident generated from this video.</p>`;
            reviewContent.innerHTML = `<p class="label">Not applicable - no incident to review.</p>`;
        }

        // Evidence
        const evidenceContent = document.getElementById('evidence-content');
        let evHtml = '';
        if (data.annotated_video) {
            // Add a timestamp query param to bypass cache
            const ts = new Date().getTime();
            evHtml += `<h4>Annotated Output Video</h4>
                       <video controls src="${data.annotated_video}?t=${ts}"></video>`;
        }
        if (data.evidence_image) {
            const ts = new Date().getTime();
            evHtml += `<h4>Evidence Frame</h4>
                       <img src="${data.evidence_image}?t=${ts}" alt="Evidence Frame">`;
        }
        if (!evHtml) {
            evHtml = "<p>No evidence file was generated for this analysis.</p>";
        }
        evidenceContent.innerHTML = evHtml;

        // Raw Output
        document.getElementById('raw-json').textContent = JSON.stringify(data.raw_output, null, 2);
    }

    function setDetection(idPrefix, detected, count) {
        const badge = document.getElementById(`res-${idPrefix}-detected`);
        const countEl = document.getElementById(`res-${idPrefix}-count`);
        
        if (detected) {
            badge.textContent = "YES";
            badge.className = "badge yes";
        } else {
            badge.textContent = "NO";
            badge.className = "badge no";
        }
        countEl.textContent = count;
    }
});

// Expose globally for the onclick handler
window.toggleRaw = function() {
    const raw = document.getElementById('raw-json');
    const icon = document.getElementById('raw-icon');
    if (raw.classList.contains('hidden')) {
        raw.classList.remove('hidden');
        icon.textContent = '▲';
    } else {
        raw.classList.add('hidden');
        icon.textContent = '▼';
    }
}
