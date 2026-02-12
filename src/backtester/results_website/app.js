// app.js - Enhanced with signal aggregation and improved UI
const RESULTS_ROOT = "assets/results";
let ACTIVE_RUN_ID = null;

function runRoot() {
  if (!ACTIVE_RUN_ID) throw new Error("ACTIVE_RUN_ID is null");
  return `${RESULTS_ROOT}/runs/${ACTIVE_RUN_ID}`;
}

function stockRoot(ticker) {
  return `${runRoot()}/stocks/${ticker}`;
}

// -----------------------------
// Fetch helpers
// -----------------------------
async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to load ${url}: ${r.status}`);
  return await r.json();
}

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to load ${url}: ${r.status}`);
  return await r.text();
}

// -----------------------------
// Ticker helpers
// -----------------------------
function normalizeTicker(t) {
  return String(t ?? "").trim().toUpperCase();
}

function pickTicker(s) {
  const t =
    s?.ticker ??
    s?.Ticker ??
    s?.symbol ??
    s?.Symbol ??
    s?.code ??
    s?.Code ??
    "";
  return String(t).trim();
}

// -----------------------------
// Glossary links
// -----------------------------
const GLOSSARY_LINKS = {
  "cagr": "glossary.html#cagr",
  "total return": "glossary.html#total-return",
  "max drawdown": "glossary.html#max-drawdown",
  "maxdd": "glossary.html#max-drawdown",
  "drawdown": "glossary.html#max-drawdown",
  "sharpe": "glossary.html#sharpe",
  "signal": "glossary.html#signal",
  "pnl": "glossary.html#pnl",
  "# trades": "glossary.html#trades",
  "win %": "glossary.html#win-rate",
};

// -----------------------------
// Number formatting
// -----------------------------
function formatNumber2(n) {
  if (n === null || n === undefined || n === "") return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function formatPercentage(n) {
  if (n === null || n === undefined || n === "") return "—";
  const val = Number(n) * 100; // Convert decimal to percentage
  return val.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }) + "%";
}

function tryNumber(x) {
  const v = Number(String(x).replaceAll("%", "").replaceAll(" ", "").replaceAll(",", ""));
  return Number.isFinite(v) ? v : null;
}

// -----------------------------
// CSV parsing
// -----------------------------
function parseCSV(csvText) {
  const text = (csvText ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.trim()) return { headers: [], rows: [] };

  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];

    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else {
      if (c === '"') {
        inQuotes = true;
      } else if (c === ",") {
        row.push(field);
        field = "";
      } else if (c === "\n") {
        row.push(field);
        field = "";
        if (row.some(x => x.trim() !== "")) rows.push(row);
        row = [];
      } else {
        field += c;
      }
    }
  }

  row.push(field);
  if (row.some(x => x.trim() !== "")) rows.push(row);

  const headers = (rows[0] ?? []).map(s => (s ?? "").trim());
  const dataRows = rows.slice(1);

  const outRows = dataRows.map(cols => {
    const obj = {};
    headers.forEach((h, idx) => (obj[h] = (cols[idx] ?? "").trim()));
    return obj;
  });

  return { headers, rows: outRows };
}

// -----------------------------
// DOM helpers
// -----------------------------
function $(sel) { return document.querySelector(sel); }
function el(tag, cls) {
  const x = document.createElement(tag);
  if (cls) x.className = cls;
  return x;
}

function setVisible(selector, on) {
  const node = $(selector);
  if (!node) return;
  node.style.display = on ? "" : "none";
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatGeneratedAt(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

// -----------------------------
// Global state
// -----------------------------
let SITE = null;
let STOCKS = [];
let ACTIVE_TICKER = null;

let LB_HEADERS = [];
let LB_ROWS = [];
let LB_CSV_URL = null;

let PLOTS = [];
let ACTIVE_PLOT = null;

// Store aggregated signals per stock
let STOCK_SIGNALS = {};

// -----------------------------
// Signal helpers
// -----------------------------
function getSignalClass(signal) {
  const s = String(signal).toUpperCase();
  if (s === "BUY" || s === "+1" || s === "1") return "signal--buy";
  if (s === "SELL" || s === "-1") return "signal--sell";
  return "signal--hold";
}

function getSignalText(signal) {
  const s = String(signal).toUpperCase();
  if (s === "BUY" || s === "+1" || s === "1") return "BUY";
  if (s === "SELL" || s === "-1") return "SELL";
  return "HOLD";
}

function normalizeSignalValue(signal) {
  const s = String(signal).trim();
  if (s === "1" || s === "+1" || s.toUpperCase() === "BUY") return 1;
  if (s === "-1" || s.toUpperCase() === "SELL") return -1;
  return 0;
}

// -----------------------------
// Signal Aggregation
// -----------------------------
function computeSignalAggregation(headers, rows) {
  // Find signal column
  const signalNumCol = findHeaderCaseInsensitive(headers, "signal (-1/0/+1)") ||
    findHeaderCaseInsensitive(headers, "signal");

  if (!signalNumCol) {
    return { strategies: [], majorityVote: "HOLD", weightedVote: "HOLD" };
  }

  // Extract strategies and their signals
  const strategies = rows.map(row => {
    const strategyName = row["Strategy"] || "Unknown";
    const signalValue = normalizeSignalValue(row[signalNumCol]);
    const signalText = getSignalText(signalValue);

    return {
      name: strategyName.replace(" (best)", "").trim(),
      signalValue: signalValue,
      signalText: signalText,
      cagr: tryNumber(row["CAGR"]) || 0
    };
  });

  // Compute majority vote
  const buys = strategies.filter(s => s.signalValue === 1).length;
  const sells = strategies.filter(s => s.signalValue === -1).length;
  const holds = strategies.filter(s => s.signalValue === 0).length;

  let majorityVote = "HOLD";
  if (buys > sells && buys > holds) majorityVote = "BUY";
  else if (sells > buys && sells > holds) majorityVote = "SELL";
  else if (buys === sells && buys > holds) majorityVote = "HOLD"; // tie defaults to hold

  // Compute weighted vote (using CAGR as default weights)
  let weightedSum = 0;
  let totalWeight = 0;

  strategies.forEach(s => {
    const weight = Math.max(0, s.cagr); // Use CAGR as default weight
    weightedSum += s.signalValue * weight;
    totalWeight += weight;
  });

  let weightedVote = "HOLD";
  if (totalWeight > 0) {
    const avgSignal = weightedSum / totalWeight;
    if (avgSignal > 0.1) weightedVote = "BUY";
    else if (avgSignal < -0.1) weightedVote = "SELL";
  }

  return {
    strategies: strategies,
    majorityVote: majorityVote,
    majorityBreakdown: { buys, sells, holds },
    weightedVote: weightedVote,
    weightedSum: weightedSum,
    totalWeight: totalWeight
  };
}

function renderSignalAggregation(aggregation) {
  const panel = $("#signalsPanel");
  const strategiesContainer = $("#signalStrategies");
  const majorityResult = $("#majorityResult");
  const majorityBreakdown = $("#majorityBreakdown");
  const weightedInputs = $("#weightedInputs");
  const weightedResult = $("#weightedResult");
  const weightedBreakdown = $("#weightedBreakdown");

  if (!panel || !aggregation.strategies.length) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "";

  // Render individual strategy signals
  strategiesContainer.innerHTML = "";
  aggregation.strategies.forEach(strategy => {
    const card = el("div", "strategy-signal-card");

    const name = el("div", "strategy-signal-card__name");
    name.textContent = strategy.name;

    const signal = el("div", `strategy-signal-card__signal signal ${getSignalClass(strategy.signalText)}`);
    signal.textContent = strategy.signalText;

    card.appendChild(name);
    card.appendChild(signal);
    strategiesContainer.appendChild(card);
  });

  // Render majority vote
  majorityResult.textContent = aggregation.majorityVote;
  majorityResult.className = `aggregation-card__result signal ${getSignalClass(aggregation.majorityVote)}`;
  majorityBreakdown.textContent = `Buy: ${aggregation.majorityBreakdown.buys} | Sell: ${aggregation.majorityBreakdown.sells} | Hold: ${aggregation.majorityBreakdown.holds}`;

  // Render weighted vote inputs
  weightedInputs.innerHTML = "";
  aggregation.strategies.forEach((strategy, idx) => {
    const row = el("div", "weight-input-row");

    const label = el("div", "weight-input-row__label");
    label.textContent = strategy.name;

    const input = el("input", "weight-input-row__input");
    input.type = "number";
    input.value = Math.max(0, strategy.cagr).toFixed(4);
    input.step = "0.01";
    input.min = "0";
    input.dataset.strategyIndex = idx;

    input.addEventListener("input", () => updateWeightedVote(aggregation));

    row.appendChild(label);
    row.appendChild(input);
    weightedInputs.appendChild(row);
  });

  // Initial weighted vote
  updateWeightedVote(aggregation);
}

function updateWeightedVote(aggregation) {
  const weightedResult = $("#weightedResult");
  const weightedBreakdown = $("#weightedBreakdown");
  const inputs = document.querySelectorAll(".weight-input-row__input");

  let weightedSum = 0;
  let totalWeight = 0;

  inputs.forEach((input, idx) => {
    const weight = Math.max(0, parseFloat(input.value) || 0);
    const strategy = aggregation.strategies[idx];
    weightedSum += strategy.signalValue * weight;
    totalWeight += weight;
  });

  let vote = "HOLD";
  if (totalWeight > 0) {
    const avgSignal = weightedSum / totalWeight;
    if (avgSignal > 0.1) vote = "BUY";
    else if (avgSignal < -0.1) vote = "SELL";
  }

  weightedResult.textContent = vote;
  weightedResult.className = `aggregation-card__result signal ${getSignalClass(vote)}`;

  if (totalWeight > 0) {
    const avgSignal = weightedSum / totalWeight;
    weightedBreakdown.textContent = `Weighted average: ${avgSignal.toFixed(3)} (Total weight: ${totalWeight.toFixed(2)})`;
  } else {
    weightedBreakdown.textContent = "Set weights above zero to calculate";
  }
}

// -----------------------------
// Stock list rendering
// -----------------------------
function renderStockList(stocks) {
  const list = $("#stockList");
  list.innerHTML = "";

  stocks.forEach(s => {
    const item = el("div", "stockitem");
    item.dataset.ticker = s.ticker;

    const left = el("div", "stockitem__left");

    const t = el("div", "stockitem__ticker");
    t.textContent = s.ticker_display || s.ticker;

    const n = el("div", "stockitem__name");
    n.textContent = s.name || s.ticker_display || s.ticker;

    left.appendChild(t);
    left.appendChild(n);

    const right = el("div", "stockitem__right");

    // Add signal badge if available
    const signalData = STOCK_SIGNALS[s.ticker];
    if (signalData) {
      const signalBadge = el("div", `stockitem__signal stockitem__signal--${signalData.vote.toLowerCase()}`);
      signalBadge.textContent = signalData.vote;
      right.appendChild(signalBadge);
    }

    const chev = el("div", "stockitem__chev");
    chev.textContent = "›";
    right.appendChild(chev);

    item.appendChild(left);
    item.appendChild(right);

    item.addEventListener("click", () => {
      loadStock(s.ticker).catch(showError);
    });

    list.appendChild(item);
  });

  updateActiveStockInList();
}

function updateActiveStockInList() {
  document.querySelectorAll(".stockitem").forEach(node => {
    const isActive = node.dataset.ticker === ACTIVE_TICKER;
    node.classList.toggle("stockitem--active", isActive);
  });
}

function setupSearch() {
  const input = $("#stockSearch");
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    const filtered = STOCKS.filter(s => {
      const a = String(s.ticker_display || s.ticker || "").toLowerCase();
      const b = String(s.name || "").toLowerCase();
      return a.includes(q) || b.includes(q);
    });
    renderStockList(filtered);
  });
}

// -----------------------------
// About section
// -----------------------------
function renderAbout(site) {
  $("#siteTitle").textContent = site.site_title || "Backtest Results";
  $("#siteSubtitle").textContent = site.site_subtitle || "Technical analysis strategy optimization";
  $("#generatedAt").textContent = formatGeneratedAt(site.generated_at);

  const aboutSub = $("#aboutSubtitle");
  aboutSub.textContent = site.about_subtitle || "";

  const bullets = $("#aboutBullets");
  bullets.innerHTML = "";
  (site.about_bullets || []).forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    bullets.appendChild(li);
  });
}

// -----------------------------
// Profile rendering
// -----------------------------
function renderProfile(profile) {
  $("#stockTicker").textContent = profile.ticker || ACTIVE_TICKER;
  $("#stockName").textContent = profile.name || profile.ticker || ACTIVE_TICKER;
  $("#stockMeta").textContent = profile.meta || "";
  $("#stockSummary").textContent = profile.summary || "";

  const sector = $("#stockSector");
  if (profile.sector) {
    sector.textContent = profile.sector;
    sector.style.display = "";
  } else {
    sector.style.display = "none";
  }

  const notes = $("#stockNotes");
  if (profile.notes) {
    notes.innerHTML = `<div class="notes__title">Notes</div>${profile.notes}`;
    notes.style.display = "";
  } else {
    notes.style.display = "none";
  }
}

// -----------------------------
// Leaderboard rendering
// -----------------------------
function findHeaderCaseInsensitive(headers, wantLower) {
  const map = new Map(headers.map(h => [String(h).trim().toLowerCase(), h]));
  return map.get(wantLower.toLowerCase()) || null;
}

function extractParamsFromBestParams(bestParamsJson) {
  try {
    const params = JSON.parse(bestParamsJson);

    // Extract strategy params
    const strategyParams = params.strategy || {};

    // Format as readable string
    const entries = Object.entries(strategyParams)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");

    return entries || "—";
  } catch (e) {
    return "—";
  }
}

function isPercentColumn(header) {
  const h = String(header).toLowerCase();
  return h.includes("cagr") ||
    h.includes("return") ||
    h.includes("win") ||
    h.includes("drawdown");
}

function formatCellValue(header, value) {
  const h = String(header).toLowerCase();
  const num = tryNumber(value);

  if (num === null) return String(value || "—");

  // Special formatting for percentage columns
  if (h.includes("cagr") || h.includes("total return")) {
    return formatPercentage(num);
  }

  if (h.includes("max drawdown")) {
    // These are already in percentage format
    return formatNumber2(num * 100) + "%";
  }
  if (h.includes("win") && h.includes("%")) {
    return formatNumber2(num) + "%";
  }

  // Default: 2 decimal places
  return formatNumber2(num);
}

function renderLeaderboard(headers, rows) {
  const thead = $("#leaderboardThead");
  const tbody = $("#leaderboardTbody");

  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (!headers.length) {
    tbody.innerHTML = '<tr><td colspan="100%">No data</td></tr>';
    return;
  }

  // Separate signal columns from metric columns
  const signalColumns = ["Signal Date", "Signal", "Signal (-1/0/+1)"];
  const excludeColumns = ["Best Params"]; // We'll integrate this into Strategy

  const metricColumns = headers.filter(h =>
    !signalColumns.includes(h) && !excludeColumns.includes(h)
  );

  const displayHeaders = [...metricColumns, ...signalColumns.filter(h => headers.includes(h))];

  // Render headers
  const trh = document.createElement("tr");
  displayHeaders.forEach(h => {
    const th = document.createElement("th");

    // Add glossary link if available
    const key = String(h).trim().toLowerCase();
    const href = GLOSSARY_LINKS[key];

    if (href) {
      const a = document.createElement("a");
      a.href = href;
      a.textContent = h;
      a.className = "glosslink";
      a.target = "_blank";
      th.appendChild(a);
    } else {
      th.textContent = h;
    }

    // Mark signal columns
    if (signalColumns.includes(h)) {
      th.classList.add("col-group-signal");
    }

    trh.appendChild(th);
  });
  thead.appendChild(trh);

  // Render rows
  rows.forEach(r => {
    const tr = document.createElement("tr");

    displayHeaders.forEach(h => {
      const td = document.createElement("td");
      let raw = r[h] ?? "";

      // Special handling for Strategy column - integrate params
      if (h === "Strategy" && r["Best Params"]) {
        const strategyName = String(raw).replace(" (best)", "").trim();
        const params = extractParamsFromBestParams(r["Best Params"]);

        const nameSpan = document.createElement("div");
        nameSpan.style.fontWeight = "700";
        nameSpan.textContent = strategyName;

        const paramsSpan = document.createElement("div");
        paramsSpan.style.fontSize = "0.85em";
        paramsSpan.style.color = "var(--muted)";
        paramsSpan.style.fontFamily = "var(--mono)";
        paramsSpan.style.marginTop = "0.25rem";
        paramsSpan.textContent = params;

        td.appendChild(nameSpan);
        td.appendChild(paramsSpan);
      } else {
        const formatted = formatCellValue(h, raw);
        td.textContent = formatted;
      }

      // Right align numeric columns
      if (tryNumber(raw) !== null || isPercentColumn(h)) {
        td.classList.add("right");
      }

      // Mark signal columns
      if (signalColumns.includes(h)) {
        td.classList.add("col-group-signal");
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  $("#leaderboardHint").textContent = rows.length
    ? `Showing ${rows.length} ${rows.length === 1 ? 'strategy' : 'strategies'}`
    : "No strategies found.";
}

function sortLeaderboardBy(metric) {
  if (!LB_ROWS || !LB_ROWS.length) return;

  const rows = [...LB_ROWS];
  rows.sort((a, b) => {
    const av = a[metric];
    const bv = b[metric];
    const an = tryNumber(av);
    const bn = tryNumber(bv);

    if (an !== null && bn !== null) {
      // For drawdown (negative values), lower is better
      if (metric.toLowerCase().includes("drawdown")) {
        return an - bn; // Ascending for drawdown
      }
      return bn - an; // Descending for everything else
    }
    return String(av ?? "").localeCompare(String(bv ?? ""));
  });

  renderLeaderboard(LB_HEADERS, rows);
}

// -----------------------------
// Plot rendering
// -----------------------------
async function listPlotFiles(ticker) {
  const url = `${stockRoot(ticker)}/plots/plots.json`;
  return await fetchJSON(url);
}

function normalizePlots(plotsManifest) {
  const p = plotsManifest?.plots || [];
  return p.map(x => {
    if (typeof x === "string") {
      return {
        file: x,
        label: x.replaceAll("_", " ").replace(".html", "").toUpperCase()
      };
    }
    if (x && typeof x === "object") {
      return {
        file: x.file || x.path || "",
        label: (x.label || x.title || (x.file || "").replaceAll("_", " ").replace(".html", "")).toUpperCase()
      };
    }
    return { file: "", label: "Plot" };
  }).filter(x => x.file);
}

function renderPlotButtons(plots) {
  const container = $("#plotButtons");
  container.innerHTML = "";

  plots.forEach(p => {
    const btn = el("button", "plot-btn");
    btn.textContent = p.label || p.file;
    btn.dataset.file = p.file;

    btn.addEventListener("click", () => setActivePlot(p.file));

    container.appendChild(btn);
  });

  updateActivePlotButton();
}

function updateActivePlotButton() {
  document.querySelectorAll(".plot-btn").forEach(btn => {
    const isActive = btn.dataset.file === ACTIVE_PLOT;
    btn.classList.toggle("plot-btn--active", isActive);
  });
}

function setActivePlot(file) {
  ACTIVE_PLOT = file || null;
  const frame = $("#plotFrame");
  const href = $("#openPlotNewTab");

  if (!ACTIVE_PLOT) {
    frame.removeAttribute("src");
    href.setAttribute("href", "#");
    setVisible("#paramsPanel", false);
    setVisible("#ledgerPanel", false);
    return;
  }

  const src = `${stockRoot(ACTIVE_TICKER)}/plots/${ACTIVE_PLOT}`;
  console.log(`Loading plot: ${src}`); // Debug log

  frame.src = src;
  href.href = src;

  updateActivePlotButton();

  // Render best params for this plot
  renderBestParams(ACTIVE_PLOT);

  // Load ledger for this strategy
  const strategyKey = inferStrategyFromPlotFile(ACTIVE_PLOT);
  console.log(`Extracted strategy key from "${ACTIVE_PLOT}": "${strategyKey}"`); // Debug log

  loadAndRenderLedger(strategyKey).catch(err => {
    console.warn(`Could not load ledger for ${strategyKey}:`, err);
  });
}

function inferStrategyFromPlotFile(filename) {
  // Example: "bollinger_price_panel.html" -> "bollinger"
  // Example: "rsi_equity_curve.html" -> "rsi"
  // Example: "macd_something_else.html" -> "macd"

  // Remove .html extension
  let name = filename.replace(".html", "");

  // Split by underscore and take first part (the strategy name)
  name = name.split("_")[0];

  return name.toLowerCase();
}

function renderBestParams(plotFile) {
  const paramsPanel = $("#paramsPanel");
  const paramsDisplay = $("#paramsDisplay");
  const paramsMeta = $("#paramsMeta");

  if (!plotFile) {
    setVisible("#paramsPanel", false);
    return;
  }

  // Find the strategy matching this plot
  const strategyKey = inferStrategyFromPlotFile(plotFile);
  console.log(`Looking for strategy "${strategyKey}" in leaderboard...`); // Debug

  const strategy = LB_ROWS.find(r => {
    const name = (r["Strategy"] || "").toLowerCase().replace(" (best)", "").trim();
    const matches = name === strategyKey.toLowerCase() || name.startsWith(strategyKey.toLowerCase());
    if (matches) {
      console.log(`Found matching strategy: "${r["Strategy"]}" for key "${strategyKey}"`); // Debug
    }
    return matches;
  });

  if (!strategy || !strategy["Best Params"]) {
    console.warn(`No parameters found for strategy "${strategyKey}"`); // Debug
    paramsDisplay.innerHTML = '<div style="text-align: center; color: var(--muted); padding: 2rem;">No parameters found for this strategy</div>';
    setVisible("#paramsPanel", false);
    return;
  }

  try {
    const params = JSON.parse(strategy["Best Params"]);
    const strategyParams = params.strategy || {};
    const portfolioParams = params.portfolio || {};

    let html = '<div class="params-grid">';

    // Strategy parameters
    if (Object.keys(strategyParams).length > 0) {
      html += '<div class="param-section">';
      html += '<div class="param-section__title">Strategy Parameters</div>';
      Object.entries(strategyParams).forEach(([key, value]) => {
        html += `
          <div class="param-row">
            <div class="param-row__key">${escapeHtml(key)}</div>
            <div class="param-row__value">${escapeHtml(String(value))}</div>
          </div>
        `;
      });
      html += '</div>';
    }

    // Portfolio parameters
    if (Object.keys(portfolioParams).length > 0) {
      html += '<div class="param-section">';
      html += '<div class="param-section__title">Portfolio Parameters</div>';
      Object.entries(portfolioParams).forEach(([key, value]) => {
        html += `
          <div class="param-row">
            <div class="param-row__key">${escapeHtml(key)}</div>
            <div class="param-row__value">${escapeHtml(String(value))}</div>
          </div>
        `;
      });
      html += '</div>';
    }

    html += '</div>';

    paramsDisplay.innerHTML = html;
    paramsMeta.textContent = `${strategyKey.toUpperCase()} Strategy`;
    setVisible("#paramsPanel", true);

  } catch (e) {
    console.error("Failed to parse params:", e);
    paramsDisplay.innerHTML = '<div style="text-align: center; color: var(--danger); padding: 2rem;">Error parsing parameters</div>';
    setVisible("#paramsPanel", false);
  }
}

// -----------------------------
// Ledger rendering
// -----------------------------
async function loadAndRenderLedger(strategyKey) {
  const panel = $("#ledgerPanel");
  const meta = $("#ledgerMeta");
  const thead = $("#plotLedgerThead");
  const tbody = $("#plotLedgerTbody");

  if (!strategyKey) {
    setVisible("#ledgerPanel", false);
    return;
  }

  // Construct ledger path: assets/results/stocks/{ticker}/ledgers/{strategy}_trade_ledger.csv
  const url = `assets/results/stocks/${ACTIVE_TICKER}/ledgers/${strategyKey}_trade_ledger.csv`;

  console.log(`Attempting to load ledger from: ${url}`); // Debug log

  try {
    const csv = await fetchText(url);
    const { headers, rows } = parseCSV(csv);

    if (!headers.length || !rows.length) {
      console.log(`Ledger file is empty: ${url}`);
      setVisible("#ledgerPanel", false);
      return;
    }

    // Render table
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const trh = document.createElement("tr");
    headers.forEach(h => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    rows.forEach(r => {
      const tr = document.createElement("tr");
      headers.forEach(h => {
        const td = document.createElement("td");
        const raw = r[h] ?? "";
        const num = tryNumber(raw);

        if (num !== null) {
          td.textContent = formatNumber2(num);
          td.classList.add("right");
        } else {
          td.textContent = raw;
        }

        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    meta.textContent = `${strategyKey.toUpperCase()} • ${rows.length} ${rows.length === 1 ? 'trade' : 'trades'}`;
    setVisible("#ledgerPanel", true);

  } catch (err) {
    // If 404, ledger doesn't exist - that's ok, just hide the panel
    if (String(err).includes("404") || String(err).includes("Failed to load")) {
      console.log(`Ledger not found (this is OK): ${url}`);
      setVisible("#ledgerPanel", false);
      return;
    }
    // Other errors should be reported
    console.error(`Error loading ledger: ${err}`);
    throw err;
  }
}

// -----------------------------
// Download leaderboard
// -----------------------------
function bindDownloadLeaderboard() {
  const btn = $("#downloadLeaderboardBtn");
  btn.onclick = () => {
    if (!LB_CSV_URL) return;
    const a = document.createElement("a");
    a.href = LB_CSV_URL;
    a.download = `${ACTIVE_TICKER || "leaderboard"}_leaderboard.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
}

// -----------------------------
// Error handling
// -----------------------------
function clearError() {
  setVisible("#errorPanel", false);
  $("#errorDetails").textContent = "";
}

function showError(err) {
  console.error(err);
  setVisible("#errorPanel", true);
  $("#errorDetails").textContent = String(err?.stack || err);
}

function setLoadingState(isLoading) {
  const r = $("#refreshBtn");
  if (r) r.style.pointerEvents = isLoading ? "none" : "auto";
  if (r) r.style.opacity = isLoading ? "0.6" : "1";
}

function showContentPanels(hasStock) {
  setVisible("#aboutPanel", true);
  setVisible("#stockPanel", hasStock);
  setVisible("#signalsPanel", hasStock);
  setVisible("#leaderboardPanel", hasStock);
  setVisible("#plotsPanel", hasStock);
  setVisible("#emptyState", !hasStock);
}

// -----------------------------
// Load stock
// -----------------------------
async function loadStock(ticker) {
  if (!ticker) {
    throw new Error("loadStock() called with empty ticker");
  }

  clearError();
  setLoadingState(true);

  ACTIVE_TICKER = ticker;
  updateActiveStockInList();
  showContentPanels(true);

  // Load profile
  const profileUrl = `${stockRoot(ticker)}/profile.json`;
  const profile = await fetchJSON(profileUrl);
  renderProfile(profile);

  // Load leaderboard
  LB_CSV_URL = `${stockRoot(ticker)}/leaderboard.csv`;
  const csv = await fetchText(LB_CSV_URL);
  const { headers, rows } = parseCSV(csv);
  LB_HEADERS = headers;
  LB_ROWS = rows;

  // Compute and render signal aggregation
  const aggregation = computeSignalAggregation(LB_HEADERS, LB_ROWS);
  renderSignalAggregation(aggregation);

  // Store the majority vote for this stock
  STOCK_SIGNALS[ticker] = { vote: aggregation.majorityVote };

  // Sort and render leaderboard
  const metric = $("#rankBy").value || "CAGR";
  sortLeaderboardBy(metric);

  // Load plots
  const plotsManifest = await listPlotFiles(ticker);
  PLOTS = normalizePlots(plotsManifest);
  renderPlotButtons(PLOTS);

  if (PLOTS.length) {
    setActivePlot(PLOTS[0].file);
  } else {
    setActivePlot(null);
  }

  bindDownloadLeaderboard();
  setLoadingState(false);
}

// -----------------------------
// Init
// -----------------------------
async function init() {
  clearError();
  setLoadingState(true);

  const idx = await fetchJSON(`${RESULTS_ROOT}/index.json`);
  ACTIVE_RUN_ID = idx.latest;
  if (!ACTIVE_RUN_ID) throw new Error("No latest run found in assets/results/index.json");

  SITE = await fetchJSON(`${runRoot()}/manifest.json`);

  STOCKS = (SITE.stocks || [])
    .map(s => {
      const ticker = pickTicker(s);
      return {
        ...s,
        ticker,
        ticker_display: normalizeTicker(ticker),
      };
    })
    .filter(s => s.ticker);

  renderAbout(SITE);

  // Pre-load signals for all stocks
  for (const stock of STOCKS) {
    try {
      const csvUrl = `assets/results/stocks/${stock.ticker}/leaderboard.csv`;
      const csv = await fetchText(csvUrl);
      const { headers, rows } = parseCSV(csv);
      const aggregation = computeSignalAggregation(headers, rows);
      STOCK_SIGNALS[stock.ticker] = { vote: aggregation.majorityVote };
    } catch (e) {
      // If we can't load, just skip
      console.warn(`Could not load signals for ${stock.ticker}`);
    }
  }

  renderStockList(STOCKS);
  setupSearch();

  $("#rankBy").onchange = () => {
    const metric = $("#rankBy").value;
    sortLeaderboardBy(metric);
  };

  $("#refreshBtn").onclick = (e) => {
    e.preventDefault();
    init().catch(showError);
  };

  if (SITE.repo_url) {
    const b = $("#openRepoBtn");
    b.href = SITE.repo_url;
    b.style.display = "";
  }

  if (STOCKS.length > 0) {
    await loadStock(STOCKS[0].ticker);
  } else {
    showContentPanels(false);
  }

  setLoadingState(false);
}

init().catch(showError);
