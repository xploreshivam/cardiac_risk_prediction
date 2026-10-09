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

  // renderer
  function rn() {
    const ov = rs.overall;
    const bg = $('overall-pct');
    bg.textContent = pc(ov.risk) + '%';
    bg.style.color = window.Heart3D.colorFor(ov.risk);
    $('overall-level').textContent = ov.level + ' chance of coronary artery disease';

    const mp = { overall: ov.risk };
    r.forEach((rw) => {
      const k = rw.dataset.artery, a = rs.arteries[k], lw = a.confidence === 'low';
      const fl = rw.querySelector('.fill');
      const dmg = a.risk >= 0.55;
      fl.style.width = pc(a.risk) + '%';
      if (dmg) {
        fl.style.background = '#FFFFFF';
        fl.style.boxShadow = '0 0 10px #FFFFFF';
        fl.style.border = '1px solid #CAD5D9';
      } else {
        fl.style.background = window.Heart3D.colorFor(a.risk);
        fl.style.boxShadow = 'none';
        fl.style.border = 'none';
      }
      rw.querySelector('.val').textContent = pc(a.risk) + '%';
      rw.querySelector('.val').style.color = dmg ? '#B3263E' : '';
      rw.querySelector('.conf').textContent = dmg ? 'DAMAGED / HIGH RISK' : (lw ? 'Low confidence' : 'Reliable');
      rw.classList.toggle('is-low', lw && !dmg);
      rw.classList.toggle('is-damaged', dmg);
      mp[k] = { p: a.risk, low: lw };
    });
    window.Heart3D.setRisks(mp);
    rd();
  }

  // details
  function rd() {
    if (!s) {
      d.textContent = 'Select an artery on the heart, or from this list, to read what it supplies and its validation reliability.';
      return;
    }
    let tx = m[s][0] + '. ' + m[s][1];
    if (rs) {
      const at = rs.arteries[s];
      tx += ' Predicted chance of stenosis: ' + pc(at.risk) + '% (' + at.level.toLowerCase() +
        '). Model accuracy: ' + pc(at.accuracy) + '% (ROC-AUC: ' + at.auc.toFixed(2) + ').';
      if (at.confidence === 'low') {
        tx += ' [Low Confidence Notice]: In tabular cardiac datasets, LCX & RCA show high class imbalance and overlapping clinical features, making cross-validated accuracy closer to baseline. Shown with dashed/muted indicator for academic integrity.';
      } else {
        tx += ' [Reliable Model]: Strong cross-validated accuracy and discriminative power for this artery.';
      }
    }
    d.textContent = tx;
  }

  // selection
  window.Heart3D.onSelect((nm) => {
    s = nm;
    r.forEach((rw) => rw.setAttribute('aria-pressed', String(rw.dataset.artery === nm)));
    rd();
  });

  // clicks
  r.forEach((rw) => rw.addEventListener('click', () => {
    const ky = rw.dataset.artery;
    window.Heart3D.select(s === ky ? null : ky);
  }));

  // submission
  f.addEventListener('submit', (ev) => { ev.preventDefault(); pd(); });

  // autosync
  f.addEventListener('input', () => {
    if (!rs) return;
    clearTimeout(tm);
    tm = setTimeout(pd, 300);
  });
})();
