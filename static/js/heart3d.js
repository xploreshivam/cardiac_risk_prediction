// heart
(function () {
  'use strict';

  // stage
  const st = document.getElementById('stage');
  const ap = { setRisks() {}, select() {}, onSelect() {}, colorFor() { return '#1DD1A1'; } };
  window.Heart3D = ap;
  if (!st) return;

  // webgl
  if (typeof THREE === 'undefined') {
    st.insertAdjacentHTML('beforeend', '<p class="stage-note">WebGL loading error.</p>');
    return;
  }

  // math
  const rm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const cp = (v, l, h) => Math.min(h, Math.max(l, v));

  // palette
  const cl = {
    g: new THREE.Color(0x1DD1A1),
    c: new THREE.Color(0x00D2D3),
    o: new THREE.Color(0xFF9F43),
    w: new THREE.Color(0xFFFFFF),
    k: new THREE.Color(0x3B6978)
  };

  // colormap
  function rc(p) {
    const col = new THREE.Color();
    if (p < 0.25) return col.copy(cl.g);
    if (p < 0.45) return col.copy(cl.g).lerp(cl.c, (p - 0.25) / 0.20);
    if (p < 0.55) return col.copy(cl.c).lerp(cl.o, (p - 0.45) / 0.10);
    return col.copy(cl.w);
  }
  ap.colorFor = (v) => '#' + rc(v).getHexString();

  // scene
  const sc = new THREE.Scene();
  const cm = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  cm.position.set(0, 0, 22);

  // renderer
  const rd = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  rd.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  rd.outputEncoding = THREE.sRGBEncoding;
  const cv = rd.domElement;
  st.insertBefore(cv, st.firstChild);

  // lighting
  sc.add(new THREE.AmbientLight(0xffffff, 1.15));
  const l1 = new THREE.DirectionalLight(0xffffff, 1.5);
  l1.position.set(10, 15, 18);
  sc.add(l1);
  const l2 = new THREE.DirectionalLight(0xa5cde0, 0.8);
  l2.position.set(-14, -6, 12);
  sc.add(l2);
  const l3 = new THREE.DirectionalLight(0xffffff, 0.65);
  l3.position.set(0, 12, -14);
  sc.add(l3);

  // hierarchy
  const gp = new THREE.Group();
  sc.add(gp);

  // elements
  let hm = null, mx = null, tg = null, ca = null;
  let damageColorAttr = null, damageWeightAttr = null;
  let currentDamageHighlights = {};
  ap.getDamageHighlights = function () { return currentDamageHighlights; };
  const lb = document.getElementById('stage-labels');

  // badge
  const bg = document.createElement('div');
  bg.className = 'stage-note';
  bg.textContent = 'Loading 3D beating heart model...';
  st.appendChild(bg);

  // vessels
  const ad = {
    LAD: { tag: null, p: null, mid: new THREE.Vector3(-1.0, -1.5, 3.2) },
    LCX: { tag: null, p: null, mid: new THREE.Vector3(-2.8, 1.2, 1.8) },
    RCA: { tag: null, p: null, mid: new THREE.Vector3(2.8, -0.8, 2.2) }
  };
  const of = { LAD: [30, 6], LCX: [-34, 0], RCA: [34, 0] };

  // markers
  Object.keys(ad).forEach((k) => {
    const el = document.createElement('div');
    el.className = 'tag';
    el.textContent = k;
    el.hidden = true;
    lb.appendChild(el);
    ad[k].tag = el;
  });

  // data
  const tp = fetch('/static/data/vertex_anatomy_tags.json')
    .then((r) => r.ok ? r.json() : null)
    .catch(() => null);

  // gltf
  const ld = new THREE.GLTFLoader();
  ld.load(
    '/static/models/beating-heart.glb',
    async function (gf) {
      if (bg.parentNode) bg.parentNode.removeChild(bg);
      const md = gf.scene;
      const bx = new THREE.Box3().setFromObject(md);
      const ct = bx.getCenter(new THREE.Vector3());
      const sz = bx.getSize(new THREE.Vector3());
      const sc = 13.5 / (Math.max(sz.x, sz.y, sz.z) || 1);
      md.scale.setScalar(sc);
      md.position.set(-ct.x * sc, -ct.y * sc, -ct.z * sc);

      md.traverse((ch) => {
        if (ch.isMesh && !hm) hm = ch;
      });

      tg = await tp;
      if (hm && tg) {
        const ge = hm.geometry;
        const vc = ge.attributes.position.count;

        // Custom damage color & weight attributes for direct pixel overwrite
        const colArr = new Float32Array(vc * 3);
        const wgtArr = new Float32Array(vc);
        damageColorAttr = new THREE.BufferAttribute(colArr, 3);
        damageWeightAttr = new THREE.BufferAttribute(wgtArr, 1);
        ge.setAttribute('aDamageColor', damageColorAttr);
        ge.setAttribute('aDamageWeight', damageWeightAttr);

        // Fallback vertex color attribute
        const ar = new Float32Array(vc * 3);
        ar.fill(1.0);
        ca = new THREE.BufferAttribute(ar, 3);
        ge.setAttribute('color', ca);

        hm.material = hm.material.clone();
        hm.material.vertexColors = true;
        hm.material.roughness = 0.35;
        hm.material.metalness = 0.08;
        hm.material.userData.uPulse = { value: 1.0 };
        hm.material.customProgramCacheKey = function () { return 'cardiac_damage_shader_v4'; };

        hm.material.onBeforeCompile = function (shader) {
          shader.uniforms.uPulse = hm.material.userData.uPulse;
          shader.vertexShader = shader.vertexShader.replace(
            'void main() {',
            `attribute vec3 aDamageColor;
             attribute float aDamageWeight;
             varying vec3 vDamageColor;
             varying float vDamageWeight;
             void main() {
               vDamageColor = aDamageColor;
               vDamageWeight = aDamageWeight;`
          );
          shader.fragmentShader = shader.fragmentShader.replace(
            'void main() {',
            `uniform float uPulse;
             varying vec3 vDamageColor;
             varying float vDamageWeight;
             void main() {`
          );
          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <dithering_fragment>',
            `#include <dithering_fragment>
             if (vDamageWeight > 0.01) {
               float w = min(1.0, vDamageWeight);
               // Overwrite diffuse color with pure luminous highlight
               gl_FragColor.rgb = mix(gl_FragColor.rgb, vDamageColor, w * 0.95);
               // Add vibrant alert emission
               gl_FragColor.rgb += vDamageColor * (w * 0.32 * uPulse);
             }`
          );
        };
        hm.material.needsUpdate = true;
      }

      gp.add(md);
      if (gf.animations && gf.animations.length > 0) {
        mx = new THREE.AnimationMixer(md);
        mx.clipAction(gf.animations[0]).play();
      }
      if (lr) dc();
    },
    undefined,
    function () {
      bg.textContent = 'Could not load 3D model.';
    }
  );

  // events
  const ls = [];
  let sl = null, lr = null;

  // coloring
  function dc() {
    if (!lr) return;
    const r = lr;
    const p1 = r.LAD ? r.LAD.p : 0, p2 = r.LCX ? r.LCX.p : 0, p3 = r.RCA ? r.RCA.p : 0;
    const d1 = p1 >= 0.55, d2 = p2 >= 0.55, d3 = p3 >= 0.55;

    const damagedList = [];
    if (d1) damagedList.push('LAD');
    if (d2) damagedList.push('LCX');
    if (d3) damagedList.push('RCA');

    // Distinct palette
    const COLOR_WHITE = { r: 1.0, g: 1.0, b: 1.0, hex: '#FFFFFF', name: 'White' };
    const COLOR_CYAN  = { r: 0.0, g: 0.92, b: 1.0, hex: '#00F5D4', name: 'Cyan' };
    const COLOR_GOLD  = { r: 1.0, g: 0.72, b: 0.05, hex: '#FFB703', name: 'Gold' };

    const highlightConfig = {};
    if (damagedList.length === 1) {
      // Single damaged area -> pure white highlight
      highlightConfig[damagedList[0]] = COLOR_WHITE;
    } else if (damagedList.length >= 2) {
      // Multiple damaged areas -> each gets a distinct color
      if (d1) highlightConfig['LAD'] = COLOR_WHITE;
      if (d2) highlightConfig['LCX'] = (d1 ? COLOR_CYAN : COLOR_WHITE);
      if (d3) highlightConfig['RCA'] = (d1 && d2 ? COLOR_GOLD : (d1 || d2 ? COLOR_CYAN : COLOR_WHITE));
    }
    currentDamageHighlights = highlightConfig;

    if (hm && tg) {
      if (damageColorAttr && damageWeightAttr) {
        const cArr = damageColorAttr.array;
        const wArr = damageWeightAttr.array;

        for (let i = 0; i < tg.length; i++) {
          const t = tg[i] || 0;
          let hl = null;
          let weight = 0.0;

          if (t === 1 && highlightConfig['LAD']) { hl = highlightConfig['LAD']; weight = 1.0; }
          else if (t === 4 && highlightConfig['LAD']) { hl = highlightConfig['LAD']; weight = 0.92; }
          else if (t === 2 && highlightConfig['LCX']) { hl = highlightConfig['LCX']; weight = 1.0; }
          else if (t === 5 && highlightConfig['LCX']) { hl = highlightConfig['LCX']; weight = 0.92; }
          else if (t === 3 && highlightConfig['RCA']) { hl = highlightConfig['RCA']; weight = 1.0; }
          else if (t === 6 && highlightConfig['RCA']) { hl = highlightConfig['RCA']; weight = 0.92; }
          else if (sl) {
            if ((t === 1 && sl === 'LAD') || (t === 2 && sl === 'LCX') || (t === 3 && sl === 'RCA')) {
              hl = COLOR_WHITE; weight = 0.75;
            } else if ((t === 4 && sl === 'LAD') || (t === 5 && sl === 'LCX') || (t === 6 && sl === 'RCA')) {
              hl = COLOR_WHITE; weight = 0.45;
            }
          }

          if (hl && weight > 0) {
            cArr[i * 3] = hl.r;
            cArr[i * 3 + 1] = hl.g;
            cArr[i * 3 + 2] = hl.b;
            wArr[i] = weight;
          } else {
            cArr[i * 3] = 0;
            cArr[i * 3 + 1] = 0;
            cArr[i * 3 + 2] = 0;
            wArr[i] = 0.0;
          }
        }
        damageColorAttr.needsUpdate = true;
        damageWeightAttr.needsUpdate = true;
      }

      if (ca) {
        const arr = ca.array;
        const tc = [
          new THREE.Color(1, 1, 1),
          highlightConfig['LAD'] ? new THREE.Color(highlightConfig['LAD'].hex) : (sl === 'LAD' ? new THREE.Color(1.5, 1.5, 1.5) : rc(p1)),
          highlightConfig['LCX'] ? new THREE.Color(highlightConfig['LCX'].hex) : (sl === 'LCX' ? new THREE.Color(1.5, 1.5, 1.5) : rc(p2)),
          highlightConfig['RCA'] ? new THREE.Color(highlightConfig['RCA'].hex) : (sl === 'RCA' ? new THREE.Color(1.5, 1.5, 1.5) : rc(p3)),
          highlightConfig['LAD'] ? new THREE.Color(2.2, 2.2, 2.2) : new THREE.Color(1, 1, 1),
          highlightConfig['LCX'] ? new THREE.Color(highlightConfig['LCX'].hex) : new THREE.Color(1, 1, 1),
          highlightConfig['RCA'] ? new THREE.Color(highlightConfig['RCA'].hex) : new THREE.Color(1, 1, 1)
        ];

        for (let i = 0; i < tg.length; i++) {
          const c = tc[tg[i] || 0];
          arr[i * 3] = c.r;
          arr[i * 3 + 1] = c.g;
          arr[i * 3 + 2] = c.b;
        }
        ca.needsUpdate = true;
      }
    }

    ['LAD', 'LCX', 'RCA'].forEach((k) => {
      const it = ad[k];
      const is = (k === sl);
      const pr = r[k] ? r[k].p : 0;
      const dg = highlightConfig[k];

      if (it.tag) {
        it.tag.classList.toggle('is-selected', is);
        it.tag.classList.toggle('is-damaged', Boolean(dg));
        if (dg) {
          it.tag.textContent = k + ' ' + Math.round(pr * 100) + '% [DAMAGED - ' + dg.name.toUpperCase() + ']';
          it.tag.style.setProperty('--c', dg.hex);
          it.tag.style.background = dg.hex;
          it.tag.style.color = '#0E2530';
          it.tag.style.fontWeight = '700';
          it.tag.style.boxShadow = '0 0 16px ' + dg.hex;
        } else if (k === 'LCX' || k === 'RCA') {
          it.tag.textContent = k + ' ' + Math.round(pr * 100) + '% (Low conf)';
          it.tag.style.setProperty('--c', '#FF9F43');
          it.tag.style.background = 'rgba(14, 37, 48, 0.88)';
          it.tag.style.color = '#FFFFFF';
          it.tag.style.boxShadow = '';
        } else {
          it.tag.textContent = k + ' ' + Math.round(pr * 100) + '% (Normal)';
          it.tag.style.setProperty('--c', '#1DD1A1');
          it.tag.style.background = 'rgba(14, 37, 48, 0.88)';
          it.tag.style.color = '#FFFFFF';
          it.tag.style.boxShadow = '';
        }
      }
    });
  }

  // selection
  ap.select = (k) => {
    sl = k || null;
    dc();
    ls.forEach((f) => f(sl));
  };
  ap.onSelect = (f) => ls.push(f);
  ap.setRisks = (r) => {
    lr = r;
    dc();
  };

  // raycaster
  const ry = new THREE.Raycaster();
  let dn = null, inx = false;

  function pt(e) {
    const rc = cv.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - rc.left) / rc.width) * 2 - 1, -((e.clientY - rc.top) / rc.height) * 2 + 1);
  }

  function ht(e) {
    if (!hm || !tg) return null;
    ry.setFromCamera(pt(e), cm);
    const it = ry.intersectObject(hm, false);
    if (it.length > 0 && it[0].face) {
      const v = tg[it[0].face.a];
      if (v === 1 || v === 4) return 'LAD';
      if (v === 2 || v === 5) return 'LCX';
      if (v === 3 || v === 6) return 'RCA';
    }
    return null;
  }

  // controls
  cv.addEventListener('pointerdown', (e) => {
    dn = { x: e.clientX, y: e.clientY, rx: gp.rotation.x, ry: gp.rotation.y, m: false };
    cv.setPointerCapture(e.pointerId);
  });

  cv.addEventListener('pointermove', (e) => {
    if (!dn) { cv.style.cursor = ht(e) ? 'pointer' : ''; return; }
    const dx = e.clientX - dn.x, dy = e.clientY - dn.y;
    if (Math.hypot(dx, dy) > 4) { dn.m = true; inx = true; }
    if (dn.m) {
      gp.rotation.y = dn.ry + dx * 0.008;
      gp.rotation.x = cp(dn.rx + dy * 0.006, -0.85, 0.85);
    }
  });

  cv.addEventListener('pointerup', (e) => {
    if (dn && !dn.m) {
      const k = ht(e);
      if (k) ap.select(k);
    }
    dn = null;
  });

  cv.addEventListener('wheel', (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    cm.position.z = cp(cm.position.z + e.deltaY * 0.03, 14, 38);
  }, { passive: false });

  // actions
  const zm = (d) => { cm.position.z = cp(cm.position.z + d, 14, 38); };
  document.getElementById('zoom-in').addEventListener('click', () => zm(-3));
  document.getElementById('zoom-out').addEventListener('click', () => zm(3));
  document.getElementById('reset-view').addEventListener('click', () => {
    gp.rotation.set(0, 0, 0);
    cm.position.z = 22;
    inx = false;
  });

  // resize
  function rs() {
    const w = st.clientWidth, h = st.clientHeight;
    if (!w || !h) return;
    rd.setSize(w, h, false);
    cm.aspect = w / h;
    cm.updateProjectionMatrix();
  }
  new ResizeObserver(rs).observe(st);
  rs();

  // projection
  const fc = new THREE.Vector3(), tp2 = new THREE.Vector3();
  function pl() {
    const w = st.clientWidth, h = st.clientHeight;
    fc.set(0, 0, 1).applyQuaternion(gp.quaternion);
    ['LAD', 'LCX', 'RCA'].forEach((k) => {
      const it = ad[k];
      if (!it.mid) return;
      const sh = it.tag && !it.tag.hidden && fc.z > 0.15;
      it.tag.hidden = !sh;
      if (!sh) return;
      tp2.copy(it.mid);
      gp.localToWorld(tp2);
      tp2.project(cm);
      const x = (tp2.x * 0.5 + 0.5) * w + of[k][0];
      const y = (-tp2.y * 0.5 + 0.5) * h + of[k][1];
      it.tag.style.transform = 'translate(' + x + 'px,' + y + 'px) translate(-50%,-50%)';
    });
  }

  // ticker
  const ck = new THREE.Clock();
  (function tk() {
    const dt = ck.getDelta();
    const t = ck.getElapsedTime();
    if (!rm) {
      if (mx) mx.update(dt);
      if (!inx) gp.rotation.y = Math.sin(t * 0.4) * 0.25;
      if (hm && hm.material && hm.material.userData && hm.material.userData.uPulse) {
        hm.material.userData.uPulse.value = 0.82 + 0.28 * Math.sin(t * 3.8);
      }
    }
    rd.render(sc, cm);
    pl();
    requestAnimationFrame(tk);
  })();
})();
