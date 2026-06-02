// Import the pre-compiled models directly as an ES module (prevents all fetch errors!)
import { modelsData } from "./models_data.js";

// Initialize Lucide icons
lucide.createIcons();

// State variables
let chart = null;

// Simulated band names and centers
const bandNames = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B5 (RE1)", "B6 (RE2)", "B7 (RE3)", "B8 (NIRb)", "B8A (NIRn)"];
const bandKeys = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a"];
const bandWavelengths = [490, 560, 665, 705, 740, 783, 842, 865];

// Preset definitions (Reflectance profiles)
const presets = {
  "loam": { b2: 0.100, b3: 0.140, b4: 0.200, b5: 0.230, b6: 0.260, b7: 0.290, b8: 0.310, b8a: 0.320 },
  "calcareous": { b2: 0.280, b3: 0.340, b4: 0.400, b5: 0.430, b6: 0.460, b7: 0.490, b8: 0.520, b8a: 0.530 },
  "sandy": { b2: 0.180, b3: 0.220, b4: 0.280, b5: 0.310, b6: 0.340, b7: 0.370, b8: 0.390, b8a: 0.400 },
  "peat": { b2: 0.030, b3: 0.040, b4: 0.050, b5: 0.060, b6: 0.070, b7: 0.080, b8: 0.100, b8a: 0.110 }
};

// Curated reference soil spectral database representing real global soil classes
const soilDatabase = [
  { id: "US-IA-01", desc: "Mollisol (Midwest Grassland Organic Loam) • Iowa, USA", actualPH: 6.8, bands: { b2: 0.100, b3: 0.140, b4: 0.200, b5: 0.230, b6: 0.260, b7: 0.290, b8: 0.310, b8a: 0.320 } },
  { id: "BR-AM-03", desc: "Ultisol (Tropical Acidic Rainforest Clay) • Amazonas, Brazil", actualPH: 4.8, bands: { b2: 0.080, b3: 0.120, b4: 0.180, b5: 0.210, b6: 0.240, b7: 0.270, b8: 0.290, b8a: 0.300 } },
  { id: "IN-KA-05", desc: "Vertisol (Cracking Alkaline Black Clay) • Karnataka, India", actualPH: 8.2, bands: { b2: 0.260, b3: 0.320, b4: 0.380, b5: 0.410, b6: 0.440, b7: 0.470, b8: 0.500, b8a: 0.510 } },
  { id: "US-AZ-02", desc: "Aridisol (Desert Silt & Calcium Carbonate) • Arizona, USA", actualPH: 7.9, bands: { b2: 0.240, b3: 0.300, b4: 0.360, b5: 0.390, b6: 0.420, b7: 0.450, b8: 0.480, b8a: 0.490 } },
  { id: "UK-SC-04", desc: "Histosol (Highly Acidic Organic Sphagnum Peat) • Highlands, Scotland", actualPH: 3.8, bands: { b2: 0.030, b3: 0.040, b4: 0.050, b5: 0.060, b6: 0.070, b7: 0.080, b8: 0.100, b8a: 0.110 } },
  { id: "SE-NL-01", desc: "Spodosol (Coniferous Boreal Forest Podzol) • Norrland, Sweden", actualPH: 4.5, bands: { b2: 0.060, b3: 0.090, b4: 0.130, b5: 0.160, b6: 0.190, b7: 0.220, b8: 0.240, b8a: 0.250 } },
  { id: "JP-KY-02", desc: "Andisol (Allophane Volcanic Ash) • Kyushu, Japan", actualPH: 5.6, bands: { b2: 0.050, b3: 0.070, b4: 0.100, b5: 0.120, b6: 0.140, b7: 0.160, b8: 0.180, b8a: 0.190 } },
  { id: "KE-RV-03", desc: "Oxisol (Highly Weathered Siderite & Clay) • Rift Valley, Kenya", actualPH: 5.2, bands: { b2: 0.120, b3: 0.160, b4: 0.220, b5: 0.250, b6: 0.280, b7: 0.310, b8: 0.330, b8a: 0.340 } },
  { id: "FR-LV-05", desc: "Alfisol (Temperate Deciduous Silt-Loam) • Loire Valley, France", actualPH: 6.2, bands: { b2: 0.140, b3: 0.180, b4: 0.240, b5: 0.270, b6: 0.300, b7: 0.330, b8: 0.350, b8a: 0.360 } },
  { id: "ES-AN-01", desc: "Calcisol (Highly Calcareous Gravelly Silt) • Andalusia, Spain", actualPH: 8.5, bands: { b2: 0.300, b3: 0.360, b4: 0.420, b5: 0.450, b6: 0.480, b7: 0.510, b8: 0.540, b8a: 0.550 } }
];

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
// SOIL REFERENCE DATABASE POPULATION
// ==========================================================================
function loadSoilSample(sampleIndex) {
  const sample = soilDatabase[sampleIndex];
  if (!sample) return;

  // Set sliders
  bandKeys.forEach(key => {
    document.getElementById(`${key}-slider`).value = sample.bands[key];
  });

  // Clear presets active state since we loaded a custom sample
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.remove("active");
  });

  runInference();
  
  // Highlight active row in table
  document.querySelectorAll("#database-table-body tr").forEach((row, idx) => {
    row.style.background = idx === sampleIndex ? "rgba(168, 85, 247, 0.15)" : "";
    row.style.borderColor = idx === sampleIndex ? "var(--accent-purple)" : "";
  });
}

function populateDatabaseTable() {
  if (!modelsData) return;

  const tableBody = document.getElementById("database-table-body");
  tableBody.innerHTML = "";

  const mean = modelsData.scaler.mean;
  const scale = modelsData.scaler.scale;
  const plsCoeffs = modelsData.plsr.coefficients;
  const plsIntercept = modelsData.plsr.intercept;

  soilDatabase.forEach((sample, index) => {
    // Extract band inputs
    const inputs = bandKeys.map(key => sample.bands[key]);

    // Run inference live in browser for database items
    const xScaled = scaleFeatures(inputs, mean, scale);
    const plsVal = predictPLS(xScaled, plsCoeffs, plsIntercept);
    const gprResult = predictGPR(xScaled, modelsData.gpr);
    
    const gprVal = gprResult.mean;
    const gprStd = gprResult.std;
    const margin = 1.645 * gprStd;
    const lowerPI = Math.max(0, gprVal - margin);
    const upperPI = gprVal + margin;

    // Create Table Row
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => loadSoilSample(index));
    
    tr.innerHTML = `
      <td><strong>${sample.id}</strong></td>
      <td>${sample.desc}</td>
      <td style="font-weight:600;">${sample.actualPH.toFixed(1)}</td>
      <td class="pls-text">${plsVal.toFixed(2)}</td>
      <td class="gpr-text" style="color:var(--accent-cyan); font-weight:600;">${gprVal.toFixed(2)}</td>
      <td><span class="range-badge">${lowerPI.toFixed(2)} – ${upperPI.toFixed(2)}</span></td>
      <td>
        <button class="action-btn" style="padding: 0.35rem 0.75rem; font-size: 0.75rem;" type="button">
          Load &amp; Predict
        </button>
      </td>
    `;
    
    tableBody.appendChild(tr);
  });
}

// ==========================================================================
// APP INITIALIZATION & BINDINGS
// ==========================================================================
function initApp() {
  if (!modelsData) {
    console.error("Models data could not be imported!");
    return;
  }
  
  console.log("ES Module modelsData loaded successfully!", modelsData);

  // 1. Populate diagnostics panel from training metrics
  const meta = modelsData.metadata;
  document.getElementById("diag-pls-rmse").textContent = meta.pls_rmse.toFixed(3);
  document.getElementById("diag-pls-r2").textContent = `${(meta.pls_r2 * 100).toFixed(1)}%`;
  document.getElementById("diag-gpr-rmse").textContent = meta.gpr_rmse.toFixed(3);
  document.getElementById("diag-gpr-r2").textContent = `${(meta.gpr_r2 * 100).toFixed(1)}%`;
  document.getElementById("diag-gpr-pi").textContent = `${meta.gpr_mean_pi_width.toFixed(2)} pH`;

  // 2. Setup sliders event listeners
  bandKeys.forEach(key => {
    const slider = document.getElementById(`${key}-slider`);
    slider.addEventListener("input", runInference);
  });

  // 3. Setup preset buttons
  Object.keys(presets).forEach(presetName => {
    const btn = document.getElementById(`preset-${presetName}`);
    btn.addEventListener("click", () => applyPreset(presetName));
  });

  // 4. Populate interactive database table
  populateDatabaseTable();

  // 5. Initialize UI with the default Organic Loam preset
  const defaultPreset = presets.loam;
  const initialInputs = bandKeys.map(key => defaultPreset[key]);
  
  initChart(initialInputs);
  applyPreset("loam");
}

// Start app
window.addEventListener("DOMContentLoaded", initApp);
