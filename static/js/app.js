// Multi-Stage Video Compilation Frontend JavaScript

let currentProject = null;
let stageInfoInterval = null;
let stageTimers = {}; // Track elapsed time for each stage
let stageStartTimes = {}; // Store start times for each stage

// Sync inputs between pipeline and stage tabs
function syncInput(type, value) {
    if (type === 'url') {
        document.getElementById('pipeline-url').value = value;
        document.getElementById('stage1-url').value = value;
    } else if (type === 'words') {
        document.getElementById('pipeline-words').value = value;
        document.getElementById('stage4-words').value = value;
    } else if (type === 'padding-before') {
        document.getElementById('pipeline-padding-before').value = value;
        document.getElementById('stage4-padding-before').value = value;
    } else if (type === 'padding-after') {
        document.getElementById('pipeline-padding-after').value = value;
        document.getElementById('stage4-padding-after').value = value;
    } else if (type === 'merge-gap') {
        document.getElementById('pipeline-merge-gap').value = value;
        document.getElementById('stage4-merge-gap').value = value;
    }
}

// Tab Switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;

        // Update active tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update active content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(tabName).classList.add('active');

        // Load stage info for newly active tab
        const stageNumber = parseInt(tabName.replace('stage', ''));
        if (currentProject && stageNumber >= 1 && stageNumber <= 6) {
            loadStageInfo(stageNumber);
        }
    });
});

// Load projects on page load
window.addEventListener('load', () => {
    loadProjects();
});

// Project Management
async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();

        const select = document.getElementById('project-select');
        select.innerHTML = '<option value="">Select Project</option>';

        data.projects.forEach(projectId => {
            const option = document.createElement('option');
            option.value = projectId;
            option.textContent = projectId;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading projects:', error);
    }
}

let selectClickCount = 0;
let clickTimer = null;

document.getElementById('project-select').addEventListener('change', (e) => {
    currentProject = e.target.value;
    selectClickCount = 0;
    if (currentProject) {
        loadProjectStatus();
        // Load stage info for current tab
        const activeTab = document.querySelector('.tab.active');
        if (activeTab) {
            const tabName = activeTab.dataset.tab;
            const stageNumber = parseInt(tabName.replace('stage', ''));
            loadStageInfo(stageNumber);
        }
    }
});

document.getElementById('project-select').addEventListener('mousedown', (e) => {
    selectClickCount++;

    if (clickTimer) {
        clearTimeout(clickTimer);
    }

    if (selectClickCount === 2 && currentProject) {
        e.preventDefault();
        enterEditMode();
        selectClickCount = 0;
    }

    clickTimer = setTimeout(() => {
        selectClickCount = 0;
    }, 500);
});

function enterEditMode() {
    const select = document.getElementById('project-select');
    const currentValue = select.value;

    if (!currentValue) return;

    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentValue;
    input.className = 'edit-mode';
    input.id = 'project-title-edit';

    select.style.display = 'none';
    select.parentNode.insertBefore(input, select);
    input.focus();
    input.select();

    input.addEventListener('blur', () => {
        exitEditMode(input, select);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            exitEditMode(input, select);
        } else if (e.key === 'Escape') {
            input.value = currentValue;
            exitEditMode(input, select);
        }
    });
}

async function exitEditMode(input, select) {
    const newTitle = input.value.trim();
    const oldTitle = currentProject;

    if (newTitle && newTitle !== oldTitle) {
        try {
            const response = await fetch(`/api/projects/${oldTitle}/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_title: newTitle })
            });

            if (response.ok) {
                currentProject = newTitle;
                await loadProjects();
                select.value = newTitle;
            } else {
                alert('Failed to rename project');
            }
        } catch (error) {
            alert('Error renaming project: ' + error.message);
        }
    }

    input.remove();
    select.style.display = '';
}

async function createNewProject() {
    try {
        const response = await fetch('/api/projects/create', { method: 'POST' });
        const data = await response.json();

        currentProject = data.project_id;
        await loadProjects();
        document.getElementById('project-select').value = currentProject;

        alert('New project created: ' + currentProject);
        loadProjectStatus();
    } catch (error) {
        alert('Error creating project: ' + error.message);
    }
}

async function loadProjectStatus() {
    if (!currentProject) return;

    try {
        const response = await fetch(`/api/projects/${currentProject}`);
        const data = await response.json();

        // Update stage status badges
        Object.keys(data.metadata.stages).forEach(stage => {
            const status = data.metadata.stages[stage].status;
            const badge = document.getElementById(`status-${stage.replace('_', '-')}`);
            if (badge) {
                badge.className = 'status-badge ' + status;
            }
        });
    } catch (error) {
        console.error('Error loading project status:', error);
    }
}

async function loadStageInfo(stageNumber) {
    if (!currentProject) return;

    try {
        const response = await fetch(`/api/projects/${currentProject}/stage${stageNumber}/info`);
        const data = await response.json();

        // Update stage info display
        document.getElementById(`stage${stageNumber}-file-count`).textContent = data.file_count;
        document.getElementById(`stage${stageNumber}-size`).textContent = `${data.size_gb.toFixed(3)} GB`;
    } catch (error) {
        console.error(`Error loading stage ${stageNumber} info:`, error);
        document.getElementById(`stage${stageNumber}-file-count`).textContent = '-';
        document.getElementById(`stage${stageNumber}-size`).textContent = '-';
    }
}

async function clearStageOutput(stageNumber) {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const confirmed = confirm(`Are you sure you want to delete all files from Stage ${stageNumber} output?\n\nThis cannot be undone.`);
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/projects/${currentProject}/stage${stageNumber}/clear`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            alert(`Cleared Stage ${stageNumber} output:\n- Deleted ${data.deleted_files} files\n- Freed ${data.freed_gb.toFixed(2)} GB`);
            await loadStageInfo(stageNumber);
            await loadProjectStatus();
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error clearing stage output: ' + error.message);
    }
}

// Stage 1: Video Analysis
async function runStage1() {
    if (!currentProject) {
        alert('Please select or create a project first');
        return;
    }

    const url = document.getElementById('stage1-url').value;

    if (!url) {
        alert('Please enter a YouTube URL');
        return;
    }

    try {
        showProgress('stage1');
        startStageInfoPolling(1);
        startStageTimer(1);

        const response = await fetch(`/api/projects/${currentProject}/stage1/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (data.success) {
            displayResults('stage1', [
                { label: 'Total Videos', value: data.result.total_videos },
                { label: 'Total Duration', value: `${(data.result.total_duration / 60).toFixed(1)} min (${(data.result.total_duration / 3600).toFixed(1)} hrs)` }
            ]);
            await loadProjectStatus();
            await loadStageInfo(1);
            switchToTab(2);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(1);
        hideProgress('stage1');
    }
}


// Start polling stage info updates
function startStageInfoPolling(stageNumber) {
    if (stageInfoInterval) {
        clearInterval(stageInfoInterval);
    }

    stageInfoInterval = setInterval(async () => {
        if (currentProject) {
            await loadStageInfo(stageNumber);
        }
    }, 1000); // Poll every 1 second
}

// Stop polling stage info
function stopStageInfoPolling() {
    if (stageInfoInterval) {
        clearInterval(stageInfoInterval);
        stageInfoInterval = null;
    }
}

// Stage 2: Audio Download
async function runStage2() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const gbLimit = parseFloat(document.getElementById('stage2-limit').value);

    try {
        showProgress('stage2');
        startStageInfoPolling(2);
        startStageTimer(2);

        const response = await fetch(`/api/projects/${currentProject}/stage2/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gb_limit: gbLimit })
        });

        const data = await response.json();

        if (data.success) {
            displayResults('stage2', [
                { label: 'Downloaded', value: data.result.downloaded },
                { label: 'Skipped', value: data.result.skipped },
                { label: 'Failed', value: data.result.failed },
                { label: 'Total Size', value: `${data.result.total_size_gb.toFixed(2)} GB` },
                { label: 'Pending', value: data.result.pending_count }
            ]);

            if (data.result.limit_reached) {
                alert('GB limit reached. Run again to download more.');
            }

            await loadProjectStatus();
            await loadStageInfo(2);
            switchToTab(3);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(2);
        hideProgress('stage2');
    }
}

// Stage 3: Transcription
async function runStage3() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const language = document.getElementById('stage3-language').value;

    try {
        showProgress('stage3');
        startStageInfoPolling(3);
        startStageTimer(3);

        const response = await fetch(`/api/projects/${currentProject}/stage3/transcribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language })
        });

        const data = await response.json();

        if (data.success) {
            displayResults('stage3', [
                { label: 'Transcribed', value: data.result.transcribed },
                { label: 'Skipped', value: data.result.skipped },
                { label: 'Failed', value: data.result.failed },
                { label: 'Pending', value: data.result.pending_count }
            ]);
            await loadProjectStatus();
            await loadStageInfo(3);
            switchToTab(4);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(3);
        hideProgress('stage3');
    }
}

// Stage 4: Clip Planning
async function runStage4() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const words = document.getElementById('stage4-words').value
        .split(',')
        .map(w => w.trim())
        .filter(w => w.length > 0);
    const paddingBefore = parseFloat(document.getElementById('stage4-padding-before').value);
    const paddingAfter = parseFloat(document.getElementById('stage4-padding-after').value);
    const mergeGap = parseFloat(document.getElementById('stage4-merge-gap').value);

    if (words.length === 0) {
        alert('Please enter target words');
        return;
    }

    try {
        showProgress('stage4');
        startStageInfoPolling(4);
        startStageTimer(4);

        const response = await fetch(`/api/projects/${currentProject}/stage4/plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_words: words,
                padding_before: paddingBefore,
                padding_after: paddingAfter,
                merge_gap: mergeGap
            })
        });

        const data = await response.json();

        if (data.success) {
            const resultItems = [
                { label: 'Total Clips', value: data.result.total_clips },
                { label: 'Total Duration', value: `${(data.result.total_duration / 60).toFixed(1)} min` }
            ];

            // Add word counts
            Object.entries(data.result.word_counts).forEach(([word, count]) => {
                resultItems.push({ label: `"${word}"`, value: `${count} clips` });
            });

            displayResults('stage4', resultItems);
            await loadProjectStatus();
            await loadStageInfo(4);
            switchToTab(5);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(4);
        hideProgress('stage4');
    }
}

// Stage 5: Clip Download
async function runStage5() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const gbLimit = parseFloat(document.getElementById('stage5-limit').value);
    const workers = parseInt(document.getElementById('stage5-workers').value);

    try {
        showProgress('stage5');
        startStageInfoPolling(5);
        startStageTimer(5);

        const response = await fetch(`/api/projects/${currentProject}/stage5/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gb_limit: gbLimit, max_workers: workers })
        });

        const data = await response.json();

        if (data.success) {
            displayResults('stage5', [
                { label: 'Downloaded', value: data.result.downloaded },
                { label: 'Skipped', value: data.result.skipped },
                { label: 'Failed', value: data.result.failed },
                { label: 'Total Size', value: `${data.result.total_size_gb.toFixed(2)} GB` },
                { label: 'Pending', value: data.result.pending_count }
            ]);

            if (data.result.limit_reached) {
                alert('GB limit reached. Run again to download more.');
            }

            await loadProjectStatus();
            await loadStageInfo(5);
            switchToTab(6);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(5);
        hideProgress('stage5');
    }
}

// Stage 6: Compilation
async function runStage6() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const maxLengthInput = document.getElementById('stage6-max-length').value;
    const maxLength = maxLengthInput ? parseFloat(maxLengthInput) : null;

    try {
        showProgress('stage6');
        startStageInfoPolling(6);
        startStageTimer(6);

        const response = await fetch(`/api/projects/${currentProject}/stage6/compile`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_length: maxLength })
        });

        const data = await response.json();

        if (data.success && data.result.success) {
            displayResults('stage6', [
                { label: 'Total Clips', value: data.result.total_clips },
                { label: 'Final Duration', value: `${(data.result.final_duration / 60).toFixed(1)} min` },
                { label: 'File Size', value: `${data.result.file_size_gb.toFixed(2)} GB` }
            ]);

            // Show download button
            document.getElementById('stage6-download').style.display = 'block';
            document.getElementById('download-link').href = `/api/projects/${currentProject}/download`;

            await loadProjectStatus();
            await loadStageInfo(6);
        } else {
            alert('Error: ' + (data.result?.error || data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        stopStageInfoPolling();
        stopStageTimer(6);
        hideProgress('stage6');
    }
}

// Pipeline Processing
async function runPipeline() {
    if (!currentProject) {
        alert('Please select a project first');
        return;
    }

    const stages = [];

    // Check all stages 1-6
    for (let i = 1; i <= 6; i++) {
        if (document.getElementById(`pipeline-stage${i}`).checked) {
            stages.push(i);
        }
    }

    if (stages.length === 0) {
        alert('Please select at least one stage to process');
        return;
    }

    const startTime = Date.now();

    for (const stage of stages) {
        try {
            console.log(`Running stage ${stage}...`);
            switch (stage) {
                case 1:
                    await runStage1();
                    break;
                case 2:
                    await runStage2();
                    break;
                case 3:
                    await runStage3();
                    break;
                case 4:
                    await runStage4();
                    break;
                case 5:
                    await runStage5();
                    break;
                case 6:
                    await runStage6();
                    break;
            }
        } catch (error) {
            console.error(`Stage ${stage} failed:`, error);
        }
    }

    const elapsedSeconds = (Date.now() - startTime) / 1000;

    try {
        await fetch('/api/log-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                elapsed_seconds: elapsedSeconds,
                stages_run: stages
            })
        });
    } catch (error) {
        console.error('Failed to log pipeline completion:', error);
    }

    alert('Pipeline processing complete');
}

// Helper Functions
function switchToTab(stageNumber) {
    const tabSelector = `stage${stageNumber}`;
    const tabButton = document.querySelector(`.tab[data-tab="${tabSelector}"]`);
    if (tabButton) {
        tabButton.click();
    }
}

function showProgress(stage) {
    document.getElementById(`${stage}-progress`).classList.add('active');
}

function hideProgress(stage) {
    document.getElementById(`${stage}-progress`).classList.remove('active');
}

// Time Elapsed Functions
function startStageTimer(stageNumber) {
    stopStageTimer(stageNumber); // Clear any existing timer
    stageStartTimes[stageNumber] = Date.now();

    stageTimers[stageNumber] = setInterval(() => {
        const elapsed = (Date.now() - stageStartTimes[stageNumber]) / 1000;
        updateElapsedDisplay(stageNumber, elapsed);
    }, 250); // Update every 250ms
}

function stopStageTimer(stageNumber) {
    if (stageTimers[stageNumber]) {
        clearInterval(stageTimers[stageNumber]);
        delete stageTimers[stageNumber];
    }
    // Keep final elapsed time displayed
}

function updateElapsedDisplay(stageNumber, seconds) {
    const elementId = `stage${stageNumber}-elapsed`;
    const element = document.getElementById(elementId);
    if (element) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        element.textContent = `${minutes}m ${secs}s`;
    }
}

function resetElapsedDisplay(stageNumber) {
    const elementId = `stage${stageNumber}-elapsed`;
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = '-';
    }
    if (stageStartTimes[stageNumber]) {
        delete stageStartTimes[stageNumber];
    }
}

function displayResults(stage, items) {
    const resultsDiv = document.getElementById(`${stage}-results`);
    const contentDiv = document.getElementById(`${stage}-results-content`);

    contentDiv.innerHTML = '';

    items.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'result-item';
        itemDiv.innerHTML = `
            <label>${item.label}</label>
            <value>${item.value}</value>
        `;
        contentDiv.appendChild(itemDiv);
    });

    resultsDiv.classList.add('active');
}

// Function to show tab content
function showTabContent(tabId) {
    // Hide all tab content
    const allTabContent = document.querySelectorAll('.tab-content');
    allTabContent.forEach(content => {
        content.classList.remove('active');
    });

    // Show the selected tab content
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) {
        selectedContent.classList.add('active');
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeTabSwitching();
});

// Initialize tab switching functionality
function initializeTabSwitching() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            this.classList.add('active');
            // Show corresponding tab content
            showTabContent(this.dataset.tab);
        });
    });
}