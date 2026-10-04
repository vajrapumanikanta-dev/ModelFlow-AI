/**
 * ModelFlow AI - Pure Vanilla JavaScript Client
 * Manages UI State, API communication, and interactive visualizations.
 */

// Application State
const appState = {
  activeView: 'dashboard',
  datasetLoaded: false,
  datasetName: 'No dataset loaded',
  columns: [],
  targetCol: null,
  featureCols: [],
  problemType: 'binary_classification',
  summary: null,
  cleaningLog: [],
  trainingResults: null,
  bestModelName: null,
  featureSchema: null,
  standaloneCode: null,
};

// DOM Elements
const views = [
  'dashboard',
  'dataset',
  'cleaning',
  'training',
  'diagnostics',
  'xai',
  'prediction',
  'reports',
  'tracker',
];

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initEventListeners();
  checkServerStatus();
});

// ==============================================================================
// 1. NAVIGATION SYSTEM
// ==============================================================================

function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetView = item.getAttribute('data-view');
      navigateTo(targetView);
    });
  });

  // Highlighted footer buttons
  document.querySelectorAll('.btn-nav').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = btn.getAttribute('data-target');
      if (target) navigateTo(target);
    });
  });
}

function navigateTo(viewId) {
  if (!views.includes(viewId)) return;
  
  appState.activeView = viewId;

  // Update Nav Items
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('data-view') === viewId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Update View Panels
  document.querySelectorAll('.view-panel').forEach(panel => {
    if (panel.id === `view-${viewId}`) {
      panel.classList.add('active');
    } else {
      panel.classList.remove('active');
    }
  });

  // Scroll to top
  const mainWrapper = document.querySelector('.main-wrapper');
  if (mainWrapper) mainWrapper.scrollTop = 0;

  // On-demand view refresh
  if (viewId === 'diagnostics') loadDiagnostics();
  if (viewId === 'xai') loadXAI();
  if (viewId === 'tracker') loadExperiments();
  if (viewId === 'prediction') setupPredictionForm();
}

// ==============================================================================
// 2. TOAST NOTIFICATIONS & STATUS
// ==============================================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  const icon = type === 'success' ? '✓' : type === 'error' ? '⚠️' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}

async function checkServerStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.dataset_loaded) {
      appState.datasetLoaded = true;
      appState.datasetName = data.dataset_name;
      appState.targetCol = data.target_col;
      appState.problemType = data.problem_type;
      appState.bestModelName = data.best_model_name;
      updateHeaderBadges();
      loadProfileData();
    }
  } catch (err) {
    console.error('Server status check error:', err);
  }
}

function updateHeaderBadges() {
  const dsBadge = document.getElementById('header-dataset-badge');
  const targetBadge = document.getElementById('header-target-badge');
  const typeBadge = document.getElementById('header-type-badge');

  if (dsBadge) {
    dsBadge.innerHTML = `📁 Dataset: <strong>${appState.datasetName}</strong>`;
    dsBadge.classList.add('active');
  }
  if (targetBadge && appState.targetCol) {
    targetBadge.innerHTML = `🎯 Target: <strong>${appState.targetCol}</strong>`;
    targetBadge.classList.add('active');
  }
  if (typeBadge && appState.problemType) {
    const ptypeTitle = appState.problemType === 'regression' ? 'Regression' : 
      appState.problemType === 'binary_classification' ? 'Binary Classification' : 'Multi-Class';
    typeBadge.innerHTML = `⚙️ Type: <strong>${ptypeTitle}</strong>`;
    typeBadge.classList.add('active');
  }
}

// ==============================================================================
// 3. DATASET LOADING & PROFILING
// ==============================================================================

function initEventListeners() {
  // Sample Data Buttons
  document.querySelectorAll('.btn-sample-load').forEach(btn => {
    btn.addEventListener('click', () => {
      const sample = btn.getAttribute('data-sample');
      loadSampleDataset(sample);
    });
  });

  // File Upload on Dataset Page
  const fileInput = document.getElementById('dataset-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadCustomDataset(e.target.files[0]);
      }
    });
  }

  // File Upload & Drag-and-Drop on Dashboard Page
  const dashFileInput = document.getElementById('dashboard-file-input');
  if (dashFileInput) {
    dashFileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadCustomDataset(e.target.files[0]);
      }
    });
  }

  const dropzone = document.getElementById('dashboard-dropzone');
  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        uploadCustomDataset(e.dataTransfer.files[0]);
      }
    });
  }

  // Target Selector Change
  const targetSelect = document.getElementById('target-column-select');
  if (targetSelect) {
    targetSelect.addEventListener('change', (e) => {
      configureTargetColumn(e.target.value);
    });
  }

  // Cleaning Actions
  const btnCleanMissing = document.getElementById('btn-clean-missing');
  if (btnCleanMissing) {
    btnCleanMissing.addEventListener('click', applyMissingCleaning);
  }

  const btnCleanDuplicates = document.getElementById('btn-clean-duplicates');
  if (btnCleanDuplicates) {
    btnCleanDuplicates.addEventListener('click', () => applyCleaningAction('duplicates'));
  }

  const btnCleanReset = document.getElementById('btn-clean-reset');
  if (btnCleanReset) {
    btnCleanReset.addEventListener('click', () => applyCleaningAction('reset'));
  }

  // Training Action
  const btnStartTraining = document.getElementById('btn-start-training');
  if (btnStartTraining) {
    btnStartTraining.addEventListener('click', runModelTraining);
  }

  // Standalone Code Copy & Download
  const btnCopyCode = document.getElementById('btn-copy-code');
  if (btnCopyCode) {
    btnCopyCode.addEventListener('click', copyStandaloneCode);
  }

  const btnDownloadCode = document.getElementById('btn-download-code');
  if (btnDownloadCode) {
    btnDownloadCode.addEventListener('click', downloadStandaloneCode);
  }

  // Predict Single Action
  const btnPredictSingle = document.getElementById('btn-predict-single');
  if (btnPredictSingle) {
    btnPredictSingle.addEventListener('click', generateSinglePrediction);
  }

  // HTML Report Download
  const btnGenerateReport = document.getElementById('btn-generate-report');
  if (btnGenerateReport) {
    btnGenerateReport.addEventListener('click', downloadHTMLReport);
  }
}

async function loadSampleDataset(sampleKey) {
  showToast(`Loading ${sampleKey} dataset...`, 'info');
  try {
    const res = await fetch('/api/load-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: sampleKey })
    });
    const data = await res.json();
    if (data.success) {
      applyDatasetResponse(data);
      showToast(`Loaded ${data.dataset_name} successfully!`, 'success');
      navigateTo('dataset');
    } else {
      showToast(data.error, 'error');
    }
  } catch (err) {
    showToast('Failed to load sample dataset', 'error');
  }
}

async function uploadCustomDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  showToast(`Uploading ${file.name}...`, 'info');

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (data.success) {
      applyDatasetResponse(data);
      showToast(`Uploaded ${data.dataset_name} successfully!`, 'success');
      navigateTo('dataset');
    } else {
      showToast(data.error, 'error');
    }
  } catch (err) {
    showToast('Upload failed', 'error');
  }
}

function applyDatasetResponse(data) {
  appState.datasetLoaded = true;
  appState.datasetName = data.dataset_name;
  appState.targetCol = data.target_col;
  appState.featureCols = data.feature_cols;
  appState.problemType = data.problem_type;
  appState.columns = data.columns;
  appState.summary = data.summary;

  updateHeaderBadges();
  renderProfileStats(data.summary);
  renderPreviewTable(data.preview, 'dataset-preview-table');
  populateTargetDropdown(data.columns, data.target_col);
  populateFeatureChecklist(data.columns, data.target_col);
  populateCleaningColumns(data.columns);
}

async function loadProfileData() {
  try {
    const res = await fetch('/api/dataset-profile');
    const data = await res.json();
    if (data.success) {
      appState.summary = data.summary;
      appState.columns = data.columns;
      renderProfileStats(data.summary);
      renderPreviewTable(data.preview, 'dataset-preview-table');
      if (data.col_summary_table) {
        renderColSummaryTable(data.col_summary_table);
      }
      populateTargetDropdown(data.columns, data.target_col);
      populateFeatureChecklist(data.columns, data.target_col);
      populateCleaningColumns(data.columns);
    }
  } catch (err) {
    console.error(err);
  }
}

function renderProfileStats(sum) {
  if (!sum) return;
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.innerText = val;
  };
  setVal('stat-rows', Number(sum.total_rows).toLocaleString());
  setVal('stat-cols', sum.total_cols);
  setVal('stat-memory', sum.memory_usage_str);
  setVal('stat-missing', `${sum.total_missing_cells} (${sum.total_missing_pct}%)`);
  setVal('stat-duplicates', sum.duplicate_rows);
}

function renderPreviewTable(records, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !records || records.length === 0) return;

  const cols = Object.keys(records[0]);
  let html = `<table class="data-table"><thead><tr>`;
  cols.forEach(c => { html += `<th>${c}</th>`; });
  html += `</tr></thead><tbody>`;

  records.forEach(row => {
    html += `<tr>`;
    cols.forEach(c => {
      const val = row[c] === null ? '<span class="badge badge-amber">NaN</span>' : row[c];
      html += `<td>${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderColSummaryTable(records) {
  const container = document.getElementById('col-summary-table-container');
  if (!container || !records) return;
  renderPreviewTable(records, 'col-summary-table-container');
}

function populateTargetDropdown(cols, selectedTarget) {
  const select = document.getElementById('target-column-select');
  if (!select) return;
  select.innerHTML = '';
  cols.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.innerText = c;
    if (c === selectedTarget) opt.selected = true;
    select.appendChild(opt);
  });
}

function populateFeatureChecklist(cols, targetCol) {
  const container = document.getElementById('feature-checklist-container');
  if (!container) return;
  container.innerHTML = '';

  cols.forEach(c => {
    if (c === targetCol) return;
    const div = document.createElement('div');
    div.style.display = 'flex';
    div.style.alignItems = 'center';
    div.style.gap = '0.4rem';
    div.style.padding = '0.35rem 0.5rem';
    div.style.background = '#f8fafc';
    div.style.borderRadius = '6px';
    div.style.border = '1px solid #e2e8f0';

    div.innerHTML = `
      <input type="checkbox" id="feat-chk-${c}" value="${c}" checked class="feature-checkbox" />
      <label for="feat-chk-${c}" style="font-size:0.85rem; font-weight:500; cursor:pointer;">${c}</label>
    `;
    container.appendChild(div);
  });
}

function populateCleaningColumns(cols) {
  const select = document.getElementById('clean-col-select');
  if (!select) return;
  select.innerHTML = '';
  cols.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.innerText = c;
    select.appendChild(opt);
  });
}

async function configureTargetColumn(targetCol) {
  const chks = document.querySelectorAll('.feature-checkbox:checked');
  const selectedFeats = Array.from(chks).map(el => el.value);

  try {
    const res = await fetch('/api/configure-target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_col: targetCol, feature_cols: selectedFeats })
    });
    const data = await res.json();
    if (data.success) {
      appState.targetCol = data.target_col;
      appState.problemType = data.problem_type;
      updateHeaderBadges();
      populateFeatureChecklist(appState.columns, data.target_col);
      showToast(`Target configured to ${targetCol} (${data.problem_type})`, 'success');
    }
  } catch (err) {
    showToast('Failed to update target', 'error');
  }
}

// ==============================================================================
// 4. DATA CLEANING
// ==============================================================================

async function applyMissingCleaning() {
  const col = document.getElementById('clean-col-select')?.value;
  const strategy = document.getElementById('clean-missing-strategy')?.value || 'median';
  if (!col) return;

  await applyCleaningAction('missing', { columns: [col], strategy: strategy });
}

async function applyCleaningAction(actionType, extraParams = {}) {
  showToast(`Applying cleaning action...`, 'info');
  try {
    const payload = { action: actionType, ...extraParams };
    const res = await fetch('/api/clean', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      renderProfileStats(data.summary);
      renderPreviewTable(data.preview, 'dataset-preview-table');
      renderCleaningLog(data.cleaning_log);
      showToast(`Applied ${actionType} successfully!`, 'success');
    } else {
      showToast(data.error, 'error');
    }
  } catch (err) {
    showToast('Cleaning action failed', 'error');
  }
}

function renderCleaningLog(logList) {
  const container = document.getElementById('cleaning-log-container');
  if (!container) return;
  if (!logList || logList.length === 0) {
    container.innerHTML = '<p style="font-size:0.85rem; color:#64748b;">No cleaning actions executed yet.</p>';
    return;
  }
  let html = '<ul style="list-style:none; display:flex; flex-direction:column; gap:0.4rem;">';
  logList.forEach(entry => {
    html += `<li style="font-size:0.84rem; padding:0.4rem 0.6rem; background:#f8fafc; border-left:3px solid #10b981; border-radius:4px;">
      <strong>${entry.timestamp}</strong> — <code>${entry.action}</code>: ${JSON.stringify(entry.params)}
    </li>`;
  });
  html += '</ul>';
  container.innerHTML = html;
}

// ==============================================================================
// 5. MODEL TRAINING & COMPARISON
// ==============================================================================

async function runModelTraining() {
  const chks = document.querySelectorAll('.feature-checkbox:checked');
  const selectedFeats = Array.from(chks).map(el => el.value);

  const payload = {
    target_col: appState.targetCol,
    feature_cols: selectedFeats,
    test_size: parseFloat(document.getElementById('train-test-size')?.value || 0.2),
    num_scaler: document.getElementById('train-scaler')?.value || 'standard',
    num_imputer: 'median',
    cat_encoder: 'onehot',
    tune_best: document.getElementById('train-tune-toggle')?.checked || false,
    cv_folds: 3,
  };

  const btn = document.getElementById('btn-start-training');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '⚡ Training Models in Parallel...';
  }

  showToast('Starting model training pipeline...', 'info');

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      appState.trainingResults = data.leaderboard;
      appState.bestModelName = data.best_model_name;
      appState.featureSchema = data.feature_schema;
      appState.standaloneCode = data.standalone_code;

      renderBestModelBanner(data.best_model_name, data.leaderboard[0]);
      renderLeaderboard(data.leaderboard);
      renderStandaloneCode(data.standalone_code);
      showToast(`Training complete! Best Model: ${data.best_model_name}`, 'success');
      setupPredictionForm();
    } else {
      showToast(`Training error: ${data.error}`, 'error');
    }
  } catch (err) {
    showToast('Failed to complete training', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '🚀 Train & Compare Models';
    }
  }
}

function renderBestModelBanner(bestName, topRecord) {
  const container = document.getElementById('best-model-banner-container');
  if (!container) return;
  container.innerHTML = `
    <div class="best-model-banner">
      <h3>🏆 Recommended Overall Best Model: ${bestName}</h3>
      <p>
        Selected as the highest-performing model on cross-validation and test evaluation. 
        Ready for immediate one-click inference and code export.
      </p>
    </div>
  `;
}

function renderLeaderboard(board) {
  const container = document.getElementById('leaderboard-table-container');
  if (!container || !board || board.length === 0) return;

  const cols = Object.keys(board[0]);
  let html = `<table class="data-table"><thead><tr><th>Rank</th>`;
  cols.forEach(c => { html += `<th>${c}</th>`; });
  html += `</tr></thead><tbody>`;

  board.forEach((row, idx) => {
    const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx+1}`;
    const rowClass = idx === 0 ? 'style="background-color:#f0fdf4; font-weight:600;"' : '';
    html += `<tr ${rowClass}><td>${medal}</td>`;
    cols.forEach(c => {
      let val = row[c];
      if (typeof val === 'number') val = val.toFixed(4);
      html += `<td>${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderStandaloneCode(codeText) {
  const el = document.getElementById('standalone-code-view');
  if (el) el.innerText = codeText;
  const section = document.getElementById('standalone-code-section');
  if (section) section.style.display = 'block';
}

function copyStandaloneCode() {
  if (!appState.standaloneCode) return;
  navigator.clipboard.writeText(appState.standaloneCode);
  showToast('Standalone Python code copied to clipboard!', 'success');
}

function downloadStandaloneCode() {
  if (!appState.standaloneCode) return;
  const blob = new Blob([appState.standaloneCode], { type: 'text/x-python' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `train_${appState.bestModelName.toLowerCase().replace(/ /g, '_')}_pipeline.py`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==============================================================================
// 6. VISUAL DIAGNOSTICS & XAI
// ==============================================================================

async function loadDiagnostics() {
  const container = document.getElementById('diagnostics-charts-container');
  if (!container) return;
  container.innerHTML = '<p>Loading diagnostic plots...</p>';

  try {
    const res = await fetch('/api/diagnostics');
    const data = await res.json();
    if (data.success && data.charts) {
      let html = '<div class="grid-2">';
      for (const [key, base64] of Object.entries(data.charts)) {
        const title = key.replace(/_/g, ' ').toUpperCase();
        html += `
          <div class="card">
            <div class="card-header"><h3 class="card-title">📊 ${title}</h3></div>
            <img class="chart-img" src="data:image/png;base64,${base64}" alt="${title}" />
          </div>
        `;
      }
      html += '</div>';
      container.innerHTML = html;
    } else {
      container.innerHTML = '<p class="text-muted">Train a model first to view diagnostic plots.</p>';
    }
  } catch (err) {
    container.innerHTML = '<p class="text-muted">Diagnostics not available.</p>';
  }
}

async function loadXAI() {
  const container = document.getElementById('xai-content-container');
  if (!container) return;
  container.innerHTML = '<p>Computing feature attributions...</p>';

  try {
    const res = await fetch('/api/xai');
    const data = await res.json();
    if (data.success) {
      let html = `
        <div class="card">
          <div class="card-header"><h3 class="card-title">🔬 Feature Importance for ${data.model_name}</h3></div>
          <div class="table-responsive">
            <table class="data-table">
              <thead><tr><th>Feature</th><th>Importance Weight</th></tr></thead>
              <tbody>
      `;
      (data.feature_importance || []).forEach(row => {
        html += `<tr><td><strong>${row.Feature}</strong></td><td><code>${Number(row.Importance).toFixed(4)}</code></td></tr>`;
      });
      html += `</tbody></table></div></div>`;

      if (data.shap_available && data.shap_summary.length > 0) {
        html += `
          <div class="card">
            <div class="card-header"><h3 class="card-title">📈 Global SHAP Impact Analysis</h3></div>
            <div class="table-responsive">
              <table class="data-table">
                <thead><tr><th>Feature</th><th>Mean |SHAP Value|</th></tr></thead>
                <tbody>
        `;
        data.shap_summary.forEach(row => {
          html += `<tr><td><strong>${row.Feature}</strong></td><td><code>${Number(row['Mean |SHAP Value|']).toFixed(4)}</code></td></tr>`;
        });
        html += `</tbody></table></div></div>`;
      }
      container.innerHTML = html;
    } else {
      container.innerHTML = '<p class="text-muted">Train a model first to compute explainability insights.</p>';
    }
  } catch (err) {
    container.innerHTML = '<p class="text-muted">Explainability insights unavailable.</p>';
  }
}

// ==============================================================================
// 7. PREDICTION HUB (HUMAN-READABLE & PLAIN-ENGLISH)
// ==============================================================================

function setupPredictionForm() {
  const container = document.getElementById('prediction-inputs-form');
  if (!container) return;

  const schema = appState.featureSchema;
  if (!schema) {
    container.innerHTML = '<p style="color:#64748b;">Train a model first to generate the input form.</p>';
    return;
  }

  // Determine Context Header
  const targetLower = (appState.targetCol || '').toLowerCase();
  let title = '📋 Input Feature Values';
  if (targetLower.includes('medhouseval') || targetLower.includes('housing') || targetLower.includes('house')) {
    title = '🏡 Property Information';
  } else if (targetLower.includes('churn') || targetLower.includes('customer')) {
    title = '👤 Customer Profile Information';
  } else if (targetLower.includes('survived') || targetLower.includes('passenger')) {
    title = '🚢 Passenger Details';
  }

  const titleEl = document.getElementById('prediction-section-title');
  if (titleEl) titleEl.innerText = title;

  container.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'grid-3';

  // Strictly filter out target column from inputs
  for (const [colName, meta] of Object.entries(schema)) {
    if (colName === appState.targetCol) continue;

    const grp = document.createElement('div');
    grp.className = 'form-group';

    if (meta.type === 'numeric') {
      grp.innerHTML = `
        <label class="form-label">${colName}:</label>
        <input type="number" step="any" class="form-control pred-input" data-col="${colName}" value="${meta.default || 0.0}" />
      `;
    } else {
      let optsHtml = '';
      (meta.options || []).forEach(opt => {
        const isSel = opt === meta.default ? 'selected' : '';
        optsHtml += `<option value="${opt}" ${isSel}>${opt}</option>`;
      });
      grp.innerHTML = `
        <label class="form-label">${colName}:</label>
        <select class="form-select pred-input" data-col="${colName}">
          ${optsHtml}
        </select>
      `;
    }
    grid.appendChild(grp);
  }

  container.appendChild(grid);
}

async function generateSinglePrediction() {
  const inputs = {};
  document.querySelectorAll('.pred-input').forEach(el => {
    const col = el.getAttribute('data-col');
    const val = el.tagName === 'INPUT' && el.type === 'number' ? parseFloat(el.value) : el.value;
    inputs[col] = val;
  });

  const btn = document.getElementById('btn-predict-single');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/predict/single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: inputs })
    });
    const data = await res.json();
    if (data.success) {
      renderPredictionHero(data);
      showToast('Prediction generated!', 'success');
    } else {
      showToast(`Prediction error: ${data.error}`, 'error');
    }
  } catch (err) {
    showToast('Inference request failed', 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderPredictionHero(pred) {
  const container = document.getElementById('prediction-output-card-container');
  if (!container) return;

  const isReg = pred.problem_type === 'regression';
  const humanVal = pred.human_readable_prediction;
  const title = pred.display_title || `Predicted ${pred.target_name}`;
  const themeClass = isReg ? '' : (humanVal.toLowerCase().includes('churn') || humanVal.toLowerCase().includes('not survive')) ? 'danger-theme' : 'blue-theme';

  let html = `
    <div class="prediction-hero-card ${themeClass}">
      <div class="hero-target-label">${title}</div>
      <div class="hero-prediction-val">${humanVal}</div>
      <div class="hero-explanation-box">
        <strong>💡 Plain-English Meaning:</strong> "${pred.explanation}"
      </div>
    </div>

    <!-- 4-Column Metadata Summary -->
    <div class="card" style="margin-top: 1rem;">
      <div class="card-header"><h3 class="card-title">🔍 Prediction & Model Breakdown</h3></div>
      <div class="grid-4">
        <div class="stat-box">
          <div class="stat-label">🎯 Target Variable</div>
          <div class="stat-value" style="font-size:1.2rem;">${pred.target_name}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">⚙️ Prediction Type</div>
          <div class="stat-value" style="font-size:1.2rem;">${pred.problem_type_display}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">🤖 Active Model</div>
          <div class="stat-value" style="font-size:1.2rem;">${pred.model_name}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">${isReg ? '📊 Raw Model Prediction' : '📊 Confidence / Probability'}</div>
          <div class="stat-value" style="font-size:1.2rem;">${isReg ? pred.raw_prediction_str : (pred.confidence_pct || 'N/A')}</div>
        </div>
      </div>
    </div>
  `;

  // Probabilities Table if Classification
  if (pred.probabilities && Object.keys(pred.probabilities).length > 0) {
    html += `
      <div class="card">
        <div class="card-header"><h3 class="card-title">📊 Classification Probability Distribution</h3></div>
        <div class="table-responsive">
          <table class="data-table">
            <thead><tr><th>Class / Outcome</th><th>Probability</th><th>Confidence %</th></tr></thead>
            <tbody>
    `;
    for (const [cls, prob] of Object.entries(pred.probabilities)) {
      html += `<tr><td><strong>${cls}</strong></td><td><code>${prob}</code></td><td>${(prob * 100).toFixed(2)}%</td></tr>`;
    }
    html += `</tbody></table></div></div>`;
  }

  container.innerHTML = html;
}

// ==============================================================================
// 8. REPORTS & EXPERIMENT TRACKER
// ==============================================================================

async function downloadHTMLReport() {
  showToast('Generating comprehensive HTML ML Report...', 'info');
  window.open('/api/report/html', '_blank');
}

async function loadExperiments() {
  const container = document.getElementById('tracker-table-container');
  if (!container) return;
  container.innerHTML = '<p>Loading logged experiments...</p>';

  try {
    const res = await fetch('/api/experiments');
    const data = await res.json();
    if (data.success && data.runs) {
      if (data.runs.length === 0) {
        container.innerHTML = '<p class="text-muted">No runs logged yet. Train a model to log experiment artifacts.</p>';
        return;
      }
      let html = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Timestamp</th>
              <th>Model</th>
              <th>Problem Type</th>
              <th>Dataset</th>
              <th>Primary Metric</th>
              <th>CV Score</th>
            </tr>
          </thead>
          <tbody>
      `;
      data.runs.forEach(run => {
        const metricSummary = Object.entries(run.test_metrics || {}).map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`).join(', ');
        html += `
          <tr>
            <td><code>${run.run_id}</code></td>
            <td>${run.timestamp}</td>
            <td><strong>${run.model_name}</strong></td>
            <td><span class="badge badge-blue">${run.problem_type}</span></td>
            <td>${run.dataset_name}</td>
            <td>${metricSummary}</td>
            <td>${run.cv_score ? run.cv_score.mean?.toFixed(4) : 'N/A'}</td>
          </tr>
        `;
      });
      html += `</tbody></table>`;
      container.innerHTML = html;
    }
  } catch (err) {
    container.innerHTML = '<p class="text-muted">Failed to load experiments.</p>';
  }
}
