# How I Built This Project - Cardiac Risk Prediction from Scratch to Deployment

---

## 1. Project Implementation Flowchart

```mermaid
graph TD
    %% PHASE 1: DATA
    subgraph P1 ["Phase 1: Dataset Acquisition & Type-Safe Cleaning"]
        A1["Raw Clinical Records<br/>(Z-Alizadeh Sani .xlsx)"] --> A2["Type-Safe Coercion & Imputation<br/>(pd.to_numeric & Categorical Encoding)"]
        A2 --> A3["Feature-Target Partitioning<br/>(12 Non-Invasive Features vs CAD, LAD, LCX, RCA)"]
    end

    %% PHASE 2: ML
    subgraph P2 ["Phase 2: ML Pipeline & Stratified Validation"]
        A3 --> B1["Pipeline Design<br/>(StandardScaler + Balanced Random Forest)"]
        B1 --> B2["5-Fold Stratified Cross-Validation<br/>(Accuracy, ROC-AUC, Recall Optimization)"]
        B2 --> B3["Model Serialization<br/>(Export .pkl Models & metrics.json)"]
    end

    %% PHASE 3: 3D ASSETS
    subgraph P3 ["Phase 3: 3D Asset Rigging & Spatial Vertex Tagging"]
        C1["Cardiac 3D Mesh Asset<br/>(beating-heart.glb with Skeletal Rig)"] --> C2["Spatial Perfusion Partitioning<br/>(Map 20,139 Vertices to LAD, LCX, RCA Territories)"]
        C2 --> C3["Export Geometric Mapping Table<br/>(vertex_anatomy_tags.json)"]
    end

    %% PHASE 4: BACKEND
    subgraph P4 ["Phase 4: Defensive Flask Backend & Error Handling"]
        B3 --> D1["Flask Application Architecture<br/>(Endpoints: /, /api/predict, /health, /flowchart)"]
        D1 --> D2["Defensive Input Validation Layer<br/>(isinstance dict, math.isfinite, Bounds Check)"]
        D2 --> D3["Standardized JSON Error Handlers<br/>(HTTP 400, 405, 415, 422, 500)"]
    end

    %% PHASE 5: FRONTEND
    subgraph P5 ["Phase 5: Interactive WebGL Frontend & UI Synchronization"]
        D3 --> E1["Responsive Clinical UI<br/>(index.html, style.css, ARIA Progressbars)"]
        C3 --> E2["Three.js WebGL Engine<br/>(OrbitControls, AnimationMixer, Raycaster, Billboard Tags)"]
        E1 --> E3["Dual-Feedback Diagnostic System"]
        E2 --> E3
        E3 --> E4["3D GPU Shading: White (Single) / Multi-Color (Multi)<br/>UI Readout: Medical Red #E63946 Alert Bars & 3D Color Pills"]
    end

    %% PHASE 6: DEPLOYMENT
    subgraph P6 ["Phase 6: Verification, Testing & Cloud Deployment"]
        E4 --> F1["Automated Test Verification<br/>(Client validation, NaN/Inf tests, 400/405/422 checks)"]
        F1 --> F2["Production Containerization<br/>(Procfile + Gunicorn WSGI Server)"]
        F2 --> F3["Live Cloud Hosting<br/>(Render / Heroku / AWS Production Deploy)"]
    end
```

---

## 2. Step-by-Step Implementation Roadmap

### Phase 1: Data Acquisition & Preprocessing
1. **Source Dataset**: Store clinical data in `data/raw/extention_of_Z-Alizadeh_sani_dataset.xlsx`.
2. **Type-Safe Cleaning (`src/data.py`)**: Use explicit `pd.to_numeric` parsing to avoid zeroing out numeric fields.
3. **Feature Selection (`src/config.py`)**: Select 12 key non-invasive features (Age, Sex, BMI, Blood Pressure, Smoking, Diabetes, ECG findings, Echo EF).
4. **Target Definition**: 
   - `CAD` (Overall disease)
   - `LAD` (Left Anterior Descending stenosis)
   - `LCX` (Left Circumflex stenosis)
   - `RCA` (Right Coronary Artery stenosis)

### Phase 2: ML Model Training (`src/train.py`)
1. **Model Architecture**: Use `RandomForestClassifier(class_weight='balanced')` inside a `Pipeline([('scaler', StandardScaler()), ('clf', ...)])`.
2. **Cross-Validation**: Run 5-fold stratified cross-validation to assess true generalization.
3. **Artifact Generation**: Save trained models to `models/` as `cad_model.pkl`, `lad_model.pkl`, `lcx_model.pkl`, `rca_model.pkl`, along with validation stats in `metrics.json`.

### Phase 3: 3D Asset & Vertex Mapping
1. **GLB Model**: Place the animated cardiac model in `static/models/beating-heart.glb`.
2. **Anatomical Partitioning**: Map each vertex in the mesh to vascular coronary supply zones:
   - `LAD territory`: Anterior wall and apex.
   - `LCX territory`: Lateral and posterior left wall.
   - `RCA territory`: Inferior wall and right ventricle.
3. **Mapping Asset**: Save 20,139 integer vertex tags to `static/data/vertex_anatomy_tags.json`.

### Phase 4: Backend API & Defensive Validation (`app.py`, `src/predict.py`)
1. Create `/api/predict` accepting JSON clinical metrics.
2. Validate incoming types and numerical boundaries (return `422` with clear error messages if out of range, `NaN`, or `Infinity`).
3. Defensive JSON error handlers for `400` (non-dict payload), `405` (wrong method), `415` (unsupported media type), and `500` (server errors).
4. Return predicted risks (`0.0` to `1.0`), categorical levels, and validation metrics.

### Phase 5: Interactive WebGL Frontend & Dual Diagnostic Feedback
1. **Three.js Scene (`static/js/heart3d.js`)**:
   - Initialize Perspective Camera, Studio Directional Lights, and WebGLRenderer.
   - Load `beating-heart.glb` and activate beating animation via `THREE.AnimationMixer`.
   - Bind `THREE.BufferAttribute(colors, 3)` with `material.vertexColors = true`.
2. **Damage Shading & Multi-Area Highlights**:
   - Single artery damaged: Highlights territory in **Stark White (`#FFFFFF`, HDR overdrive multiplier 6.5)**.
   - Multiple arteries damaged: Highlights each territory in a distinct contrast color (**LAD: White, LCX: Electric Cyan, RCA: Vivid Gold**).
   - Clinical Readout Card: Highlights damaged stenosis risk in bold **Medical Red (`#E63946`)** with ARIA progressbars.
   - Healthy/mild: Retains natural anatomical gradient (Green/Cyan).

### Phase 6: Testing & Deployment
1. **Automated Verification**: Run test client checks for boundaries, `NaN`/`Inf`, and HTTP error codes.
2. **Containerization**: Configure `Procfile` containing `web: gunicorn app:app`.
3. **Deployment**: Deploy to Render, Heroku, or AWS with Python 3.11+ runtime.
