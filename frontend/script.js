// Initialize Lucide icons
lucide.createIcons();

// State variables
let modelsData = null;
let chart = null;

// Simulated band names and centers
const bandNames = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B5 (RE1)", "B6 (RE2)", "B7 (RE3)", "B8 (NIRb)", "B8A (NIRn)"];
const bandKeys = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a"];
const bandWavelengths = [490, 560, 665, 705, 740, 783, 842, 865];

// Preset definitions (Reflectance profiles)
const presets = {
  "loam": { b2: 0.120, b3: 0.160, b4: 0.220, b5: 0.250, b6: 0.280, b7: 0.310, b8: 0.330, b8a: 0.340 },
  "calcareous": { b2: 0.280, b3: 0.340, b4: 0.400, b5: 0.430, b6: 0.460, b7: 0.490, b8: 0.520, b8a: 0.530 },
  "sandy": { b2: 0.180, b3: 0.220, b4: 0.280, b5: 0.310, b6: 0.340, b7: 0.370, b8: 0.390, b8a: 0.400 },
  "peat": { b2: 0.040, b3: 0.060, b4: 0.080, b5: 0.100, b6: 0.120, b7: 0.140, b8: 0.160, b8a: 0.170 }
};

// ==========================================================================
// ML INFERENCE ENGINES (CLIENT-SIDE)
// ==========================================================================

// Standard Scaler logic
function scaleFeatures(x, mean, scale) {
  return x.map((val, idx) => (val - mean[idx]) / scale[idx]);
}

// Partial Least Squares Regression
function predictPLS(xScaled, plsCoeffs, plsIntercept) {
  let pred = plsIntercept[0];
  for (let i = 0; i < xScaled.length; i++) {
    pred += xScaled[i] * plsCoeffs[i][0];
  }
  return pred;
}

// Gaussian Process Regression with Matern 3/2 Kernel
function predictGPR(xScaled, gprData) {
  const X_fit = gprData.X_fit;
  const alpha = gprData.alpha;
  const K_inv = gprData.K_inv;
  const constant = gprData.hyperparameters.constant_value;
  const lengthScale = gprData.hyperparameters.length_scale;
  const noiseLevel = gprData.hyperparameters.noise_level;
  
  const nSamples = X_fit.length;
  
  // 1. Calculate k_star (covariance vector between new x and all training samples)
  const k_star = new Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    let sumSq = 0;
    for (let k = 0; k < xScaled.length; k++) {
      sumSq += Math.pow(xScaled[k] - X_fit[i][k], 2);
    }
    const d = Math.sqrt(sumSq);
    
    if (d === 0) {
      k_star[i] = constant;
    } else {
      // Matern 3/2: constant * (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
      const val = constant * (1.0 + Math.sqrt(3.0) * d / lengthScale) * Math.exp(-Math.sqrt(3.0) * d / lengthScale);
      k_star[i] = val;
    }
  }
  
  // 2. Predict mean: y_pred = k_star . alpha
  let meanVal = 0;
  for (let i = 0; i < nSamples; i++) {
    meanVal += k_star[i] * alpha[i];
  }
  
  // 3. Predict variance: var_val = constant - k_star * K_inv * k_star_T + noise_level
  // Compute k_star * K_inv
  const k_star_K_inv = new Array(nSamples).fill(0);
  for (let col = 0; col < nSamples; col++) {
    for (let row = 0; row < nSamples; row++) {
      k_star_K_inv[col] += k_star[row] * K_inv[row][col];
    }
  }
  
  // Compute (k_star * K_inv) * k_star_T
  let k_star_K_inv_k_star = 0;
  for (let i = 0; i < nSamples; i++) {
    k_star_K_inv_k_star += k_star_K_inv[i] * k_star[i];
  }
  
  const variance = constant - k_star_K_inv_k_star + noiseLevel;
  const std = Math.sqrt(Math.max(1e-10, variance));
  
  return {
    mean: meanVal,
    std: std
  };
}

// ==========================================================================
// CORE APPLICATION INFERENCE
// ==========================================================================
function runInference() {
  if (!modelsData) return;

  // 1. Gather slider inputs
  const inputs = bandKeys.map(key => parseFloat(document.getElementById(`${key}-slider`).value));

  // Update slider numeric readouts
  bandKeys.forEach((key, idx) => {
    document.getElementById(`${key}-val`).textContent = inputs[idx].toFixed(3);
  });

  // 2. Standardize features
  const mean = modelsData.scaler.mean;
  const scale = modelsData.scaler.scale;
  const xScaled = scaleFeatures(inputs, mean, scale);

  // 3. Compute Predictions
  const plsVal = predictPLS(xScaled, modelsData.plsr.coefficients, modelsData.plsr.intercept);
  const gprResult = predictGPR(xScaled, modelsData.gpr);
  
  // GPR Prediction Interval: 90% confidence bounds = 1.645 * std
  const gprVal = gprResult.mean;
  const gprStd = gprResult.std;
  const margin = 1.645 * gprStd;
  const lowerPI = Math.max(0, gprVal - margin);
  const upperPI = gprVal + margin;
  const piWidth = 2 * margin;

  // 4. Update UI Displays
  document.getElementById("pls-pred-val").textContent = plsVal.toFixed(2);
  document.getElementById("gpr-pred-val").textContent = gprVal.toFixed(2);
  document.getElementById("gpr-pi-range").textContent = `${lowerPI.toFixed(2)} – ${upperPI.toFixed(2)}`;
  document.getElementById("gpr-pi-width").textContent = `${piWidth.toFixed(2)} pH`;
  document.getElementById("pls-gpr-diff").textContent = `${Math.abs(plsVal - gprVal).toFixed(2)} pH`;

  // Update graphical interval slider for GP
  // Mapping pH 4.0 - 9.0 to 0% - 100%
  const pHMin = 4.0;
  const pHMax = 9.0;
  const getPercent = (val) => Math.min(100, Math.max(0, ((val - pHMin) / (pHMax - pHMin)) * 100));

  const gprPercent = getPercent(gprVal);
  const lowerPercent = getPercent(lowerPI);
  const upperPercent = getPercent(upperPI);
  
  const spanElement = document.getElementById("gpr-pi-span");
  spanElement.style.left = `${lowerPercent}%`;
  spanElement.style.width = `${upperPercent - lowerPercent}%`;
  
  document.getElementById("gpr-pi-marker").style.left = `${gprPercent}%`;
  document.getElementById("pls-marker").style.left = `${getPercent(plsVal)}%`;

  // Update pH Spectrum Classification
  const activePH = Math.round(gprVal);
  document.querySelectorAll(".ph-color").forEach(el => {
    el.classList.remove("active");
    if (parseInt(el.getAttribute("data-ph")) === activePH) {
      el.classList.add("active");
    }
  });

  // Describe classification status
  let phClass = "Neutral";
  let phColor = "var(--accent-green)";
  if (gprVal < 5.5) {
    phClass = "Highly Acidic";
    phColor = "#FF5A5A";
  } else if (gprVal < 6.5) {
    phClass = "Moderately Acidic";
    phColor = "#FFAE5A";
  } else if (gprVal > 8.0) {
    phClass = "Highly Alkaline";
    phColor = "#5AAEEB";
  } else if (gprVal > 7.2) {
    phClass = "Slightly Alkaline";
    phColor = "#5AEBBA";
  } else {
    phClass = "Neutral Soil";
    phColor = "var(--accent-green)";
  }
  const classTextSpan = document.getElementById("ph-class-text");
  classTextSpan.textContent = phClass;
  classTextSpan.style.color = phColor;

  // 5. Update Reflectance chart
  updateChart(inputs);
}

// ==========================================================================
// CHART VISUALIZATION
// ==========================================================================
function initChart(initialData) {
  const ctx = document.getElementById('spectra-chart').getContext('2d');
  
  // Custom design gradients
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, 'rgba(168, 85, 247, 0.25)');
  gradient.addColorStop(1, 'rgba(168, 85, 247, 0.0)');

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: bandNames,
      datasets: [{
        label: 'Reflectance Profile',
        data: initialData,
        borderColor: '#a855f7',
        borderWidth: 3,
        pointBackgroundColor: '#06b6d4',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8,
        backgroundColor: gradient,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#a39eb9', font: { family: 'Inter' } }
        },
        y: {
          min: 0.0,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#a39eb9', font: { family: 'Inter' } }
        }
      }
    }
  });
}

function updateChart(data) {
  if (chart) {
    chart.data.datasets[0].data = data;
    chart.update();
  }
}

// ==========================================================================
// SOIL PROFILE PRESETS
// ==========================================================================
function applyPreset(presetName) {
  const values = presets[presetName];
  if (!values) return;

  // Set sliders
  bandKeys.forEach(key => {
    document.getElementById(`${key}-slider`).value = values[key];
  });

  // Set active state on buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.remove("active");
  });
  document.getElementById(`preset-${presetName}`).classList.add("active");

  runInference();
}

// ==========================================================================
// CSV BATCH INFERENCE SYSTEM
// ==========================================================================
let batchResults = [];

function handleCSVString(csvText) {
  if (!modelsData) return;

  // Split lines
  const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== "");
  if (lines.length < 2) {
    alert("CSV must contain a header row and at least one data row.");
    return;
  }

  // Parse headers
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  
  // Find column indexes for each band B2 - B8A
  const bandCols = bandKeys.map(key => {
    // Look for exact key (e.g. b2) or band descriptions
    let idx = headers.indexOf(key);
    if (idx === -1) idx = headers.indexOf(`scan_visnir.${key}_ref`); // common OSSL format
    if (idx === -1) {
      // Fallback: look for index starting with the key
      idx = headers.findIndex(h => h.startsWith(key));
    }
    return idx;
  });

  // If we couldn't resolve headers, assume column 0-7 are B2-B8A
  const resolvedAll = bandCols.every(idx => idx !== -1);
  if (!resolvedAll) {
    console.warn("Could not match all band headers in CSV. Defaulting to columns 0-7 in order: B2, B3, B4, B5, B6, B7, B8, B8A.");
  }

  const tableBody = document.getElementById("results-table-body");
  tableBody.innerHTML = "";
  batchResults = [];

  const mean = modelsData.scaler.mean;
  const scale = modelsData.scaler.scale;
  const plsCoeffs = modelsData.plsr.coefficients;
  const plsIntercept = modelsData.plsr.intercept;

  let rowCount = 0;
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    if (cols.length < 8) continue;

    const sampleId = cols[0] || `Sample_${i}`;
    
    // Extract band inputs
    let inputs = [];
    if (resolvedAll) {
      inputs = bandCols.map(colIdx => parseFloat(cols[colIdx]));
    } else {
      // Parse first 8 numerical columns (ignoring non-numeric or sample ID)
      let numCols = cols.map(c => parseFloat(c)).filter(n => !isNaN(n));
      if (numCols.length >= 8) {
        inputs = numCols.slice(0, 8);
      } else {
        // Fallback directly to index mapping
        inputs = bandKeys.map((_, idx) => parseFloat(cols[idx + 1]) || parseFloat(cols[idx]) || 0);
      }
    }

    // Skip if any band is invalid
    if (inputs.some(val => isNaN(val))) continue;

    // Run inference
    const xScaled = scaleFeatures(inputs, mean, scale);
    const plsVal = predictPLS(xScaled, plsCoeffs, plsIntercept);
    const gprResult = predictGPR(xScaled, modelsData.gpr);
    
    const gprVal = gprResult.mean;
    const gprStd = gprResult.std;
    const margin = 1.645 * gprStd;
    const lowerPI = Math.max(0, gprVal - margin);
    const upperPI = gprVal + margin;
    const piWidth = 2 * margin;

    batchResults.push({
      sampleId,
      inputs,
      plsPH: plsVal,
      gprPH: gprVal,
      lowerPI,
      upperPI,
      std: gprStd
    });

    // Create Table Row
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${sampleId}</strong></td>
      <td class="pls-text">${plsVal.toFixed(2)}</td>
      <td class="gpr-text">${gprVal.toFixed(2)}</td>
      <td><span class="range-badge">${lowerPI.toFixed(2)} – ${upperPI.toFixed(2)}</span></td>
      <td><span class="std-badge">±${margin.toFixed(2)} (90%)</span></td>
    `;
    tableBody.appendChild(tr);
    rowCount++;
  }

  if (rowCount === 0) {
    tableBody.innerHTML = `<tr class="empty-state"><td colspan="5">Could not parse any valid rows. Please check CSV format.</td></tr>`;
    document.getElementById("download-results-btn").disabled = true;
  } else {
    document.getElementById("download-results-btn").disabled = false;
  }
}

function downloadCSVResults() {
  if (batchResults.length === 0) return;

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Sample_ID,B2,B3,B4,B5,B6,B7,B8,B8A,PLS_pH,GPR_pH,90_Percent_PI_Lower,90_Percent_PI_Upper,Uncertainty_Std\n";

  batchResults.forEach(r => {
    const inputsStr = r.inputs.map(val => val.toFixed(4)).join(",");
    const row = `${r.sampleId},${inputsStr},${r.plsPH.toFixed(4)},${r.gprPH.toFixed(4)},${r.lowerPI.toFixed(4)},${r.upperPI.toFixed(4)},${r.std.toFixed(4)}`;
    csvContent += row + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "soil_ph_predictions_report.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// APP INITIALIZATION & BINDINGS
// ==========================================================================
async function initApp() {
  try {
    // 1. Fetch trained model parameters JSON
    console.log("Loading models.json...");
    const response = await fetch("./models.json");
    if (!response.ok) {
      throw new Error(`Failed to fetch models.json: Status ${response.status}`);
    }
    modelsData = await response.json();
    console.log("Models loaded successfully!", modelsData);

    // 2. Populate diagnostics panel from training metrics
    const meta = modelsData.metadata;
    document.getElementById("diag-pls-rmse").textContent = meta.pls_rmse.toFixed(3);
    document.getElementById("diag-pls-r2").textContent = `${(meta.pls_r2 * 100).toFixed(1)}%`;
    document.getElementById("diag-gpr-rmse").textContent = meta.gpr_rmse.toFixed(3);
    document.getElementById("diag-gpr-r2").textContent = `${(meta.gpr_r2 * 100).toFixed(1)}%`;
    document.getElementById("diag-gpr-pi").textContent = `${meta.gpr_mean_pi_width.toFixed(2)} pH`;

    // 3. Setup sliders event listeners
    bandKeys.forEach(key => {
      const slider = document.getElementById(`${key}-slider`);
      slider.addEventListener("input", runInference);
    });

    // 4. Setup preset buttons
    Object.keys(presets).forEach(presetName => {
      const btn = document.getElementById(`preset-${presetName}`);
      btn.addEventListener("click", () => applyPreset(presetName));
    });

    // 5. Setup Drag-and-Drop file listeners
    const dropZone = document.getElementById("csv-drop-zone");
    const fileInput = document.getElementById("csv-file-input");

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (evt) => handleCSVString(evt.target.result);
        reader.readAsText(file);
      }
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const reader = new FileReader();
        reader.onload = (evt) => handleCSVString(evt.target.result);
        reader.readAsText(file);
      }
    });

    document.getElementById("download-results-btn").addEventListener("click", downloadCSVResults);

    // 6. Initialize UI with the default Organic Loam preset
    const defaultPreset = presets.loam;
    const initialInputs = bandKeys.map(key => defaultPreset[key]);
    
    initChart(initialInputs);
    applyPreset("loam");

  } catch (error) {
    console.error("App initialization failed:", error);
    // Display error modal/text in diagnostic panel
    document.querySelectorAll(".stat-val").forEach(el => el.textContent = "Err");
  }
}

// Start app
window.addEventListener("DOMContentLoaded", initApp);
