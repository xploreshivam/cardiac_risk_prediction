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
    subgraph P2 ["Phase 2: ML Pipeline, Benchmarking & Explainable AI"]
        A3 --> B1["Pipeline Design<br/>(StandardScaler + Balanced Random Forest)"]
        B1 --> B2["5-Fold Stratified Cross-Validation<br/>(Accuracy, ROC-AUC, Recall Optimization)"]
        B2 --> B2B["Model Benchmarking & Calibration<br/>(RF vs. XGBoost / GradientBoosting / Calibrated)"]
        B2B --> B3["Model & Explainer Serialization<br/>(Export .pkl Models, SHAP & model_metrics.json)"]
    end

    %% PHASE 3: 3D ASSETS
    subgraph P3 ["Phase 3: 3D Asset Rigging & Spatial Vertex Tagging"]
        C1["Cardiac 3D Mesh Asset<br/>(beating-heart.glb with Skeletal Rig)"] --> C2["Spatial Perfusion Partitioning<br/>(Map 20,139 Vertices to LAD, LCX, RCA Territories)"]
        C2 --> C3["Export Geometric Mapping Table<br/>(vertex_anatomy_tags.json)"]
    end

    %% PHASE 4: BACKEND
    subgraph P4 ["Phase 4: Defensive Flask Backend & Error Handling"]
        B3 --> D1["Flask Application Architecture<br/>(Endpoints: /, /api/predict, /api/metrics, /health)"]
        D1 --> D2["Defensive Input Validation Layer<br/>(isinstance dict, math.isfinite, Bounds Check)"]
        D2 --> D3["Standardized JSON Error Handlers<br/>(HTTP 400, 405, 415, 422, 500)"]
    end

    %% PHASE 5: FRONTEND
    subgraph P5 ["Phase 5: Interactive WebGL Frontend & Multi-Modal Diagnostics"]
        D3 --> E1["Responsive Clinical UI & 1-Click Presets<br/>(STEMI LAD | Multi-Vessel | Healthy Baseline)"]
        C3 --> E2["Three.js WebGL Engine<br/>(AHA Perfusion HUD, 3D Risk Legend, Raycaster)"]
        E1 --> E3["Multi-Modal Diagnostic Feedback"]
        E2 --> E3
        E3 --> E4["3D GPU Shading (White/Multi-Color) + Wall Illumination<br/>UI: Medical Red Alert Bars & SHAP Divergent Bar Chart"]
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

### Phase 2: ML Training, Benchmarking & Explainable AI (`src/train.py`, `src/benchmark.py`)
1. **Model Architecture**: Use `RandomForestClassifier(class_weight='balanced')` inside a `Pipeline([('scaler', StandardScaler()), ('clf', ...)])`.
2. **5-Fold Stratified Cross-Validation**: Assess true generalization on small cohort ($N=303$).
3. **Benchmarking & Calibration**: Compare Random Forest vs. Gradient Boosting, XGBoost, and CalibratedClassifierCV across Brier scores and recall (`docs/BENCHMARK_REPORT.md`).
4. **Explainable AI**: Integrate SHAP TreeExplainer for patient-specific feature attributions.
5. **Artifact Generation**: Save trained models to `models/` as `cad_model.pkl`, `lad_model.pkl`, `lcx_model.pkl`, `rca_model.pkl`, along with validation stats in `model_metrics.json`.

### Phase 3: 3D Asset & Vertex Mapping
1. **GLB Model**: Place the animated cardiac model in `static/models/beating-heart.glb`.
2. **Anatomical Partitioning**: Map each vertex in the mesh to vascular coronary supply zones via automated 3D geometric coordinate bounding rules (AHA 17-segment inspired illustrative mapping):
   - `LAD territory`: Anterior wall and apex (Tag 1 for artery trace, Tag 4 for supplied myocardium).
   - `LCX territory`: Lateral and posterior left wall (Tag 2 for artery trace, Tag 5 for supplied myocardium).
   - `RCA territory`: Inferior wall and right ventricle (Tag 3 for artery trace, Tag 6 for supplied myocardium).
   - Baseline structures: Default Tag 0.
3. **Mapping Asset**: Save 20,139 integer vertex tags to `static/data/vertex_anatomy_tags.json`.

### Phase 4: Backend API & Defensive Gateway (`app.py`, `src/predict.py`)
1. Create `/api/predict` accepting JSON clinical metrics and returning both multi-target risk percentages and SHAP feature attributions.
2. Validate incoming types and numerical boundaries (return `422` with clear error messages if out of range, `NaN`, or `Infinity`).
3. Defensive JSON error handlers for `400` (non-dict payload), `405` (wrong method), `415` (unsupported media type), and `500` (server errors).
4. Return predicted risks (`0.0` to `1.0`), categorical levels, validation metrics, and SHAP feature breakdown.

### Phase 5: Interactive WebGL Frontend & Multi-Modal Diagnostics
1. **1-Click Demo Presets**: Add quick presets for judges (`High-Risk Anterior STEMI (LAD)`, `Multi-Vessel CAD`, `Healthy Baseline`) for instant 3D color transitions.
2. **Three.js Scene (`static/js/heart3d.js`)**:
   - Initialize Perspective Camera, Studio Directional Lights, and WebGLRenderer.
   - Load `beating-heart.glb` and activate beating animation via `THREE.AnimationMixer`.
   - Bind `THREE.BufferAttribute(colors, 3)` with `material.vertexColors = true`.
3. **AHA Perfusion HUD & Color Scale Legend**:
   - Floating AHA tooltip displaying exact supplied myocardial walls (Anterior/Septal, Lateral/Posterior, Inferior/RV) on hover/click.
   - 3D color scale legend overlay with interactive territory selection pills.
4. **High Predicted Risk Shading & Multi-Area Highlights**:
   - Single artery high predicted risk ($\ge 55\%$): Highlights territory in **Stark White (`#FFFFFF`, HDR overdrive multiplier 6.5)**.
   - Multiple arteries high predicted risk ($\ge 55\%$): Highlights each territory in a distinct contrast color (**LAD: White, LCX: Electric Cyan, RCA: Vivid Gold**).
   - Hover / Click Myocardial Illumination: Illuminates the exact supplied wall vertices.
5. **Clinical Readout & SHAP Panel**:
   - Medical Red (`#E63946`) alert bars with ARIA progressbars.
   - Bi-directional divergent bar charts (Red for risk-raising, Blue for risk-lowering).

### Phase 6: Testing & Deployment
1. **Automated Verification**: Run test client checks for boundaries, `NaN`/`Inf`, and HTTP error codes.
2. **Containerization**: Configure `Procfile` containing `web: gunicorn app:app`.
3. **Deployment**: Deploy to Render, Heroku, or AWS with Python 3.11+ runtime.
