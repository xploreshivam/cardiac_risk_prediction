# Project Development Lifecycle: Scratch to Deployment

A step-by-step end-to-end flowchart and development guide showing how to build this entire **Cardiac 3D Risk** platform from zero to full deployment.

---

## 1. Project Implementation Flowchart

```mermaid
graph TD
    %% PHASE 1: DATA
    subgraph P1 ["Phase 1: Dataset Acquisition & Preprocessing"]
        A["1. Collect Dataset<br/>(Z-Alizadeh Sani .xlsx)"] --> B["2. Clean & Preprocess Data<br/>(Handle missing, encode categories)"]
        B --> C["3. Split Feature Matrix & Targets<br/>(12 Features vs CAD, LAD, LCX, RCA)"]
    end

    %% PHASE 2: ML
    subgraph P2 ["Phase 2: Machine Learning Pipeline"]
        C --> D["4. Design ML Pipelines<br/>(StandardScaler + RandomForest)"]
        D --> E["5. 5-Fold Stratified Cross-Validation<br/>(Calculate Accuracy, ROC-AUC, Recall)"]
        E --> F["6. Serialize Models<br/>(Export .pkl files + metrics.json)"]
    end

    %% PHASE 3: 3D ASSETS
    subgraph P3 ["Phase 3: 3D Heart Modeling & GPU Mapping"]
        G["7. Obtain 3D Heart Model<br/>(beating-heart.glb with skeletal animation)"] --> H["8. Spatial & Vertex Mapping<br/>(Map 20,139 vertices to LAD, LCX, RCA territories)"]
        H --> I["9. Export Vertex Geometry Tags<br/>(vertex_anatomy_tags.json)"]
    end

    %% PHASE 4: BACKEND
    subgraph P4 ["Phase 4: Backend API & Error Handling"]
        F --> J["10. Build Flask REST API<br/>(POST /api/predict, GET /health)"]
        J --> K["11. Robust Error Handling<br/>(Input bounds, type checking, HTTP 422/500)"]
    end

    %% PHASE 5: FRONTEND
    subgraph P5 ["Phase 5: Frontend Interface & Three.js Engine"]
        K --> L["12. Create UI Form & Readout<br/>(index.html & responsive style.css)"]
        I --> M["13. Build Three.js 3D Engine<br/>(GLTFLoader, OrbitControls, AnimationMixer)"]
        L --> N["14. Dynamic GPU Shading Hook<br/>(Set BufferAttribute color to White on damage >= 55%)"]
        M --> N
    end

    %% PHASE 6: PRODUCTION
    subgraph P6 ["Phase 6: Verification & Cloud Deployment"]
        N --> O["15. End-to-End Testing<br/>(Test Client, payload ranges, 3D render)"]
        O --> P["16. Containerize / Deploy<br/>(Procfile + gunicorn on Render/Heroku)"]
    end
```

---

## 2. Step-by-Step Implementation Roadmap

### Phase 1: Data Preparation
1. **Source Dataset**: Place clinical data in `data/raw/extention_of_Z-Alizadeh_sani_dataset.xlsx`.
2. **Feature Selection (`src/config.py`)**: Select 12 key non-invasive features (Age, Sex, BMI, Blood Pressure, Smoking, Diabetes, ECG findings, Echo EF).
3. **Target Definition**: 
   - `CAD` (Overall disease)
   - `LAD` (Left Anterior Descending stenosis)
   - `LCX` (Left Circumflex stenosis)
   - `RCA` (Right Coronary Artery stenosis)

### Phase 2: ML Model Training (`src/train.py`)
1. **Model Architecture**: Use `RandomForestClassifier(class_weight='balanced')` inside a `Pipeline([('scaler', StandardScaler()), ('clf', ...)])`.
2. **Cross-Validation**: Run 5-fold stratified cross-validation to assess true generalization.
3. **Artifact Generation**: Save trained models to `models/` as `cad_model.pkl`, `lad_model.pkl`, `lcx_model.pkl`, `rca_model.pkl`.

### Phase 3: 3D Asset & Vertex Mapping
1. **GLB Model**: Place the animated cardiac model in `static/models/beating-heart.glb`.
2. **Anatomical Partitioning**: Map each vertex in the mesh to vascular coronary supply zones:
   - `LAD territory`: Anterior wall and apex.
   - `LCX territory`: Lateral and posterior left wall.
   - `RCA territory`: Inferior wall and right ventricle.
3. **Mapping Asset**: Save integer vertex tags to `static/data/vertex_anatomy_tags.json`.

### Phase 4: Backend API (`app.py`)
1. Create `/api/predict` accepting JSON clinical metrics.
2. Validate incoming types and numerical boundaries (return `422` with clear error messages if out of range).
3. Return predicted risks (`0.0` to `1.0`), categorical levels, and validation metrics.

### Phase 5: Interactive WebGL Frontend
1. **Three.js Scene (`static/js/heart3d.js`)**:
   - Initialize Perspective Camera, Studio Directional Lights, and WebGLRenderer.
   - Load `beating-heart.glb` and activate beating animation via `THREE.AnimationMixer`.
   - Bind `THREE.BufferAttribute(colors, 3)` with `material.vertexColors = true`.
2. **Damage Shader**:
   - If predicted stenosis $\ge 55\%$, set territory vertex colors to **Stark White (`#FFFFFF`)**.
   - If healthy/mild, set to natural anatomical gradient (Green/Cyan).

### Phase 6: Deployment
1. Add `Procfile` containing `web: gunicorn app:app`.
2. Configure `requirements.txt` with Flask, scikit-learn, pandas, numpy, and gunicorn.
3. Deploy to Render, Heroku, or AWS.
