const PRESETS = {
  low: {
    pregnancies: 1,
    glucose: 89,
    blood_pressure: 66,
    skin_thickness: 23,
    insulin: 94,
    bmi: 22.8,
    diabetes_pedigree: 0.167,
    age: 24,
  },
  moderate: {
    pregnancies: 3,
    glucose: 126,
    blood_pressure: 78,
    skin_thickness: 0,
    insulin: 0,
    bmi: 30.2,
    diabetes_pedigree: 0.452,
    age: 38,
  },
  high: {
    pregnancies: 6,
    glucose: 168,
    blood_pressure: 84,
    skin_thickness: 35,
    insulin: 185,
    bmi: 36.4,
    diabetes_pedigree: 0.785,
    age: 51,
  },
};

let probabilityChart = null;

function applyPreset(presetKey) {
  const values = PRESETS[presetKey];
  if (!values) return;

  Object.entries(values).forEach(([field, val]) => {
    const input = document.getElementById(field);
    if (input) input.value = val;
  });
}

function getFormPayload() {
  return {
    pregnancies: Number(document.getElementById("pregnancies").value),
    glucose: Number(document.getElementById("glucose").value),
    blood_pressure: Number(document.getElementById("blood_pressure").value),
    skin_thickness: Number(document.getElementById("skin_thickness").value),
    insulin: Number(document.getElementById("insulin").value),
    bmi: Number(document.getElementById("bmi").value),
    diabetes_pedigree: Number(document.getElementById("diabetes_pedigree").value),
    age: Number(document.getElementById("age").value),
  };
}

function showErrors(errors) {
  const banner = document.getElementById("error-banner");
  const list = document.getElementById("error-list");
  list.innerHTML = "";

  errors.forEach((msg) => {
    const li = document.createElement("li");
    li.textContent = msg;
    list.appendChild(li);
  });

  banner.classList.remove("hidden");
}

function hideErrors() {
  document.getElementById("error-banner").classList.add("hidden");
}

function updateConsensusUI(consensus) {
  const badge = document.getElementById("consensus-badge");
  const avgProb = document.getElementById("avg-probability");
  const voteTally = document.getElementById("vote-tally");

  badge.textContent = consensus.risk_tier;
  badge.className = "inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ";

  if (consensus.risk_tier === "High Risk") {
    badge.className += "bg-rose-500/15 text-rose-300 border border-rose-500/30";
  } else if (consensus.risk_tier === "Moderate Risk") {
    badge.className += "bg-amber-500/15 text-amber-300 border border-amber-500/30";
  } else {
    badge.className += "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
  }

  avgProb.textContent = `${consensus.average_probability.toFixed(1)}%`;
  voteTally.textContent = `${consensus.positive_votes} / ${consensus.total_models}`;
}

function updateChart(models) {
  const ctx = document.getElementById("probability-chart").getContext("2d");
  const labels = models.map((m) => m.name);
  const values = models.map((m) => m.probability);
  const barColors = models.map((m) =>
    m.prediction === 1 ? "rgba(244, 63, 94, 0.85)" : "rgba(16, 185, 129, 0.85)"
  );
  const borderColors = models.map((m) =>
    m.prediction === 1 ? "rgb(244, 63, 94)" : "rgb(16, 185, 129)"
  );

  if (probabilityChart) {
    probabilityChart.data.labels = labels;
    probabilityChart.data.datasets[0].data = values;
    probabilityChart.data.datasets[0].backgroundColor = barColors;
    probabilityChart.data.datasets[0].borderColor = borderColors;
    probabilityChart.update();
    return;
  }

  probabilityChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Diabetic Risk Probability (%)",
          data: values,
          backgroundColor: barColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 6,
          barThickness: 22,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` Risk Probability: ${context.parsed.x}%`,
          },
        },
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: "rgba(148, 163, 184, 0.12)" },
          ticks: {
            color: "#94a3b8",
            callback: (val) => `${val}%`,
          },
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#e2e8f0",
            font: { size: 12, weight: "500" },
          },
        },
      },
    },
  });
}

function updateTable(models) {
  const tbody = document.getElementById("models-table-body");
  tbody.innerHTML = "";

  models.forEach((m) => {
    const isPositive = m.prediction === 1;
    const badgeClass = isPositive
      ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
      : "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";

    const row = document.createElement("tr");
    row.className = "hover:bg-slate-800/40 transition";
    row.innerHTML = `
      <td class="py-3.5 px-6 font-medium text-slate-200">${m.name}</td>
      <td class="py-3.5 px-4">
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${badgeClass}">
          ${m.label}
        </span>
      </td>
      <td class="py-3.5 px-4 text-right font-mono font-semibold text-white">${m.probability.toFixed(1)}%</td>
      <td class="py-3.5 px-4 text-right font-mono text-slate-400">${m.accuracy.toFixed(1)}%</td>
      <td class="py-3.5 px-6 text-right font-mono text-indigo-300">${m.roc_auc.toFixed(4)}</td>
    `;
    tbody.appendChild(row);
  });
}

async function runPrediction() {
  hideErrors();
  const submitBtn = document.getElementById("submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getFormPayload()),
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      const errorMessages = result.errors || [result.error || "Prediction request failed."];
      showErrors(errorMessages);
      return;
    }

    updateConsensusUI(result.data.consensus);
    updateChart(result.data.models);
    updateTable(result.data.models);
  } catch (err) {
    showErrors([`Network error: ${err.message}`]);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Multi-Model Inference";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyPreset(btn.dataset.preset);
      runPrediction();
    });
  });

  document.getElementById("prediction-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runPrediction();
  });

  // Load initial healthy sample so dashboard renders populated on first view
  applyPreset("low");
  runPrediction();
});