// startup
(function () {
  'use strict';

  // dom
  const $ = (id) => document.getElementById(id);
  const f = $('risk-form'), e = $('form-error'), d = $('artery-detail');
  const r = Array.from(document.querySelectorAll('[data-artery]'));

  // descriptions
  const m = {
    LAD: ['Left anterior descending', 'Runs down the front of the heart and feeds the front wall and the tip of the left ventricle.'],
    LCX: ['Left circumflex', 'Wraps around the left side and feeds the side and back wall of the left ventricle.'],
    RCA: ['Right coronary artery', 'Runs along the right side and feeds the right ventricle and the bottom of the heart.'],
  };

  // state
  let rs = null, s = null, tm = null;

  // percentage
  const pc = (v) => Math.round(v * 100);

  // payload
  function pl() {
    const o = {};
    f.querySelectorAll('[data-field]').forEach((x) => { o[x.dataset.field] = x.value; });
    return o;
  }

  // request
  async function pd() {
    e.textContent = '';
    const b = f.querySelector('button[type="submit"]');
    if (b) b.disabled = true;

    try {
      const ac = new AbortController();
      const ti = setTimeout(() => ac.abort(), 8000);

      const q = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pl()),
        signal: ac.signal
      });

      clearTimeout(ti);
      const dt = await q.json().catch(() => ({ error: 'Invalid response from server.' }));

      if (!q.ok) {
        throw new Error(dt.error || 'Prediction request failed with status ' + q.status);
      }

      rs = dt;
      rn();
    } catch (er) {
      if (er.name === 'AbortError') {
        e.textContent = 'Server response timed out. Please check server status.';
      } else {
        e.textContent = er.message || 'Unable to connect to prediction service.';
      }
    } finally {
      if (b) b.disabled = false;
    }
  }

  // high-contrast clinical color palette for white card backgrounds
  function clinicalColorForPanel(p) {
    if (p < 0.25) return '#26734D'; // dark medical green (>4.5:1 contrast)
    if (p < 0.55) return '#A86500'; // dark amber (>4.5:1 contrast)
    return '#B3263E';              // clinical red
  }

  // renderer
  function rn() {
    const ov = rs.overall;
    const bg = $('overall-pct');
    bg.textContent = pc(ov.risk) + '%';
    bg.style.color = clinicalColorForPanel(ov.risk);
    $('overall-level').textContent = ov.level + ' chance of coronary artery disease';

    const mp = { overall: ov.risk };
    r.forEach((rw) => {
      const k = rw.dataset.artery, a = rs.arteries[k], lw = a.confidence === 'low';
      mp[k] = { p: a.risk, low: lw };
    });

    window.Heart3D.setRisks(mp);
    const hlMap = window.Heart3D.getDamageHighlights ? window.Heart3D.getDamageHighlights() : {};

    r.forEach((rw) => {
      const k = rw.dataset.artery, a = rs.arteries[k], lw = a.confidence === 'low';
      const fl = rw.querySelector('.fill');
      const bar = rw.querySelector('.bar');
      const dmg = a.risk >= 0.55;
      const hl = hlMap[k];

      fl.style.width = pc(a.risk) + '%';
      if (bar) {
        bar.setAttribute('aria-valuenow', String(pc(a.risk)));
      }

      if (dmg) {
        // Red color for high predicted risk in bottom card:
        fl.style.background = '#E63946';
        fl.style.boxShadow = '0 0 12px rgba(230, 57, 70, 0.75)';
        fl.style.border = 'none';
      } else {
        fl.style.background = window.Heart3D.colorFor(a.risk);
        fl.style.boxShadow = 'none';
        fl.style.border = 'none';
      }

      rw.querySelector('.val').textContent = pc(a.risk) + '%';
      rw.querySelector('.val').style.color = dmg ? '#B3263E' : '';

      if (dmg) {
        const colorName = hl ? hl.name : 'White';
        const colorHex = hl ? hl.hex : '#FFFFFF';
        rw.querySelector('.conf').innerHTML =
          '<span class="dmg-pill">HIGH RISK</span> <span class="hl-pill" style="--hlc:' + colorHex + '">3D: ' + colorName + '</span>';
      } else {
        rw.querySelector('.conf').textContent = (lw ? 'Low confidence' : 'Moderate');
      }

      rw.classList.toggle('is-low', lw && !dmg);
      rw.classList.toggle('is-damaged', dmg);
    });
    rd();
  }

  // details
  function rd() {
    if (!s) {
      d.innerHTML = '<span class="detail-hint">Select an artery on the heart, or from the list above, to read what it supplies and its validation metrics.</span>';
      renderShap();
      return;
    }
    let html = '<div class="detail-header"><strong class="detail-name">' + m[s][0] + ' (' + s + ')</strong>: ' + m[s][1] + '</div>';
    if (rs) {
      const at = rs.arteries[s];
      const isDmg = at.risk >= 0.55;
      html += '<div class="detail-badges">' +
        '<span class="detail-badge ' + (isDmg ? 'badge-danger' : 'badge-safe') + '">Predicted Risk: ' + pc(at.risk) + '% (' + at.level + ')</span>' +
        '<span class="detail-metric">Accuracy: <strong>' + pc(at.accuracy) + '%</strong></span>' +
        '<span class="detail-metric">ROC-AUC: <strong>' + at.auc.toFixed(2) + '</strong></span>' +
        (at.recall ? '<span class="detail-metric">Sensitivity: <strong>' + pc(at.recall) + '%</strong></span>' : '') +
        (at.f1 ? '<span class="detail-metric">F1: <strong>' + pc(at.f1) + '%</strong></span>' : '') +
      '</div>';
      if (at.confidence === 'low') {
        html += '<div class="detail-callout callout-low"><strong>Low Confidence Notice:</strong> High class imbalance in clinical records brings validation metrics close to baseline. Illustrated with dashed border for academic honesty.</div>';
      } else {
        html += '<div class="detail-callout callout-good"><strong>Moderate Confidence:</strong> Evaluated via cross-validation on a single-center cohort (N=303); lacks external multi-center validation.</div>';
      }
    }
    d.innerHTML = html;
    renderShap();
  }

  // SHAP feature breakdown renderer
  function renderShap() {
    const listEl = $('shap-list');
    const badgeEl = $('shap-target-badge');
    if (!listEl || !rs || !rs.explanations) return;

    const targetKey = s || 'CAD';
    const explanation = rs.explanations[targetKey];
    if (!explanation || !explanation.features) return;

    const targetNames = {
      CAD: 'Overall CAD',
      LAD: 'LAD Artery',
      LCX: 'LCX Artery',
      RCA: 'RCA Artery'
    };
    if (badgeEl) {
      badgeEl.textContent = targetNames[targetKey] || targetKey;
    }

    const maxAbs = Math.max(...explanation.features.map((item) => Math.abs(item.shap_value)), 0.04);

    let html = '';
    explanation.features.forEach((item) => {
      const sv = item.shap_value;
      const isPos = sv > 0;
      const isNeg = sv < 0;
      const pct = Math.min(Math.round((Math.abs(sv) / maxAbs) * 100), 100);
      const sign = isPos ? '+' : (isNeg ? '-' : '');
      const formattedVal = sign + Math.abs(sv).toFixed(3);

      html += '<div class="shap-row">' +
        '<div class="shap-info">' +
          '<span class="shap-fname" title="' + item.label + '">' + item.label + '</span>' +
          '<span class="shap-fval">' + item.patient_value + '</span>' +
        '</div>' +
        '<div class="shap-bar-zone">' +
          '<div class="shap-divergent-track">' +
            '<div class="shap-track-neg">' +
              (isNeg ? '<span class="shap-bar bar-neg" style="width:' + pct + '%;"></span>' : '') +
            '</div>' +
            '<div class="shap-axis"></div>' +
            '<div class="shap-track-pos">' +
              (isPos ? '<span class="shap-bar bar-pos" style="width:' + pct + '%;"></span>' : '') +
            '</div>' +
          '</div>' +
          '<span class="shap-num ' + (isPos ? 'num-pos' : (isNeg ? 'num-neg' : 'num-zero')) + '">' + formattedVal + '</span>' +
        '</div>' +
      '</div>';
    });

    listEl.innerHTML = html;
  }

  // selection coordinator
  function updateSelection(nm) {
    s = (s === nm) ? null : nm;
    r.forEach((rw) => rw.setAttribute('aria-pressed', String(rw.dataset.artery === s)));
    if (window.Heart3D && window.Heart3D.select) {
      window.Heart3D.select(s);
    }
    rd();
  }

  // selection from 3D scene
  window.Heart3D.onSelect((nm) => {
    s = nm;
    r.forEach((rw) => rw.setAttribute('aria-pressed', String(rw.dataset.artery === nm)));
    rd();
  });

  // selection from list clicks
  r.forEach((rw) => rw.addEventListener('click', () => {
    const ky = rw.dataset.artery;
    updateSelection(ky);
  }));

  // submission
  f.addEventListener('submit', (ev) => { ev.preventDefault(); pd(); });

  // autosync
  function triggerAutoSync() {
    if (!rs) return;
    clearTimeout(tm);
    tm = setTimeout(pd, 300);
  }
  f.addEventListener('input', triggerAutoSync);
  f.addEventListener('change', triggerAutoSync);

  // Initialize predictions and 3D visualization immediately on load
  pd();
})();
