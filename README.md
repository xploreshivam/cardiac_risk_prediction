# Cardiac Risk 3D - ML-based Coronary Artery Disease & Territory Mapping

An interactive clinical ML and WebGL platform that predicts stenosis risk across the primary coronary arteries (**LAD**, **LCX**, **RCA**) and overall Coronary Artery Disease (**CAD**), projecting real-time risk scores onto an anatomically partitioned 3D beating heart model with dynamic GPU vertex shading and an accessible clinical readout.

> **Clinical Disclaimer:** This application is an educational and research prototype trained on 303 patient records from the Z-Alizadeh Sani clinical dataset. It is not an FDA/CE-cleared medical device and should not be used as the sole basis for clinical diagnosis.

---

## Key Features & Highlights

- **Multi-Target Ensemble Machine Learning:**
  - 4 specialized `RandomForestClassifier` pipelines predicting:
    1. Overall Coronary Artery Disease (`CAD`)
    2. Left Anterior Descending Artery (`LAD`) Stenosis
    3. Left Circumflex Artery (`LCX`) Stenosis
    4. Right Coronary Artery (`RCA`) Stenosis
  - 5-Fold Stratified Cross-Validation with balanced class weights.
- **Real-Time 3D WebGL Heart Rendering (Three.js):**
  - Skeletal cardiac cycle deformation (`beating-heart.glb`) via WebGL rendering.
  - Studio lighting, responsive camera orbit controls, and camera-facing anatomical billboard tags.
  - Interactive anatomical raycasting to inspect local myocardial territories and vessel supply zones.
- **Dual Visual Diagnostic Feedback System:**
  - **3D Heart GPU Shading (Direct `BufferAttribute` Overdrive):**
    - **Single High Predicted Risk Artery ($\ge 55\%$):** Stark Glowing White (`#FFFFFF`, HDR multiplier `6.5`).
    - **Multiple High Predicted Risk Arteries ($\ge 55\%$):** High-contrast multi-color palette:
      - **LAD Territory:** Stark White (`#FFFFFF`)
      - **LCX Territory:** Electric Cyan (`#00E5FF`)
      - **RCA Territory:** Vivid Amber/Gold (`#FFB800`)
    - **Normal / Baseline Territory ($< 55\%$):** Clinical soft teal/green gradient.
    - *(Note on cutoff & calibration: The $\ge 55\%$ threshold is an illustrative heuristic cutoff rather than a definitive clinical diagnostic threshold; model probabilities are uncalibrated and intended for relative risk stratification.)*
  - **Clinical Readout Panel:**
    - High-contrast alert progress bars with bold **Medical Red (`#E63946`)** fill for high predicted risk arteries.
    - Matching 3D color pill tags (`3D: White`, `3D: Cyan`, `3D: Gold`) for instantaneous cross-referencing.
    - Full WCAG 2.1 AA compliant contrast ratios and ARIA accessibility (`role="progressbar"`, `aria-valuenow`, `aria-live="polite"`).
- **Defensive Production Gateway (Flask REST API):**
  - Strict payload validation rejecting malformed bodies (`400 Bad Request`), unsupported content types (`415 Unsupported Media Type`), invalid HTTP methods (`405 Method Not Allowed`), and out-of-range / non-finite inputs like `NaN` and `Infinity` (`422 Unprocessable Entity`).

---

---

## System Runtime Architecture

For the complete technical runtime data flow diagram detailing how patient clinical variables pass through defensive validation gateways, the 4-target Random Forest inference engine, and dual visual dispatch (3D GPU vertex color overdrive + Medical Red UI alert bars), please refer to the documentation:
👉 **[docs/architecture_flow.md](docs/architecture_flow.md)**

---

## How I Built This Project: Development Lifecycle (Scratch to Deployment)

The complete 6-phase engineering lifecycle used to design, train, rig, build, and deploy this project from scratch:

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
        B3 --> D1["Flask Application Architecture<br/>(Endpoints: /, /api/predict, /health)"]
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

### Phase 3 Deep-Dive: How 20,139 Vertices Were Assigned to LAD / LCX / RCA Territories

A common question in cardiac 3D mesh modeling is how individual vertices are linked to coronary blood supply. Here is the exact methodology:

- **Assignment Method:** Programmatic spatial / geometric position rule script.
  - The vertices were **not** hand-painted manually vertex-by-vertex, nor extracted from patient-specific voxel CT/MRI segmentation.
  - Instead, an automated geometry script evaluated the local 3D Cartesian coordinates $(x, y, z)$ and surface normal vectors of the heart mesh (`beating-heart.glb`) against spatial bounding rules aligned with anatomical coronary landmarks:
    1. **LAD Territory (Anterior Interventricular Sulcus & Apex):**
       - Evaluated anterior surface coordinates and apical tip.
       - **Tag 1 (LAD Artery Trace):** 563 vertices.
       - **Tag 4 (LAD Supplied Myocardium):** 1,006 vertices.
    2. **LCX Territory (Left Lateral & Posterior Free Wall):**
       - Evaluated left atrioventricular groove and posterolateral coordinates.
       - **Tag 2 (LCX Artery Trace):** 1,020 vertices.
       - **Tag 5 (LCX Supplied Myocardium):** 3,124 vertices.
    3. **RCA Territory (Right Ventricle & Inferior / Diaphragmatic Wall):**
       - Evaluated right coronary sulcus and inferior diaphragmatic surface coordinates.
       - **Tag 3 (RCA Artery Trace):** 577 vertices.
       - **Tag 6 (RCA Supplied Myocardium):** 2,871 vertices.
    4. **Normal / Baseline Myocardium & Great Vessels:**
       - Basal structures, valves, aorta/pulmonary trunks, and non-coronary segments defaulted to **Tag 0** (10,978 vertices).
- **Resulting Mapping Table:**
  All 20,139 vertex integer tags were exported to `static/data/vertex_anatomy_tags.json`, which Three.js streams into a `THREE.BufferAttribute` color array for real-time vertex shader coloring.
- **Academic Disclaimer:**
  This spatial partitioning is an **illustrative mapping inspired by the AHA 17-segment model** designed for interactive WebGL visualization, rather than a clinically validated patient segmentation.

---

## Project Structure

```
cardiac-3d-risk/
├── app.py                      # Production Flask application & REST endpoints
├── Procfile                    # Gunicorn production entrypoint
├── requirements.txt            # Python dependencies
├── docs/
│   ├── architecture_flow.md    # System Runtime Architecture Flowchart (How project works)
│   └── PROJECT_BUILD_GUIDE.md  # Step-by-step scratch-to-deployment roadmap
├── data/
│   └── raw/                    # Extension of Z-Alizadeh Sani clinical dataset (.xlsx)
├── models/                     # Serialized scikit-learn pipelines & validation metrics
│   ├── cad_model.pkl           # Overall CAD model
│   ├── lad_model.pkl           # Left Anterior Descending model
│   ├── lcx_model.pkl           # Left Circumflex model
│   ├── rca_model.pkl           # Right Coronary Artery model
│   └── model_metrics.json      # Cross-validation statistics
├── src/
│   ├── config.py               # Feature definitions, normal ranges, and file paths
│   ├── data.py                 # Type-safe parsing, imputation, and feature extraction
│   ├── train.py                # 5-fold cross-validation & model training script
│   └── predict.py              # Strict schema/range validation & inference engine
├── templates/
│   └── index.html              # Main interactive 3D clinical diagnostic workspace
└── static/
    ├── css/
    │   └── style.css           # Modern clinical UI stylesheet & WCAG contrast system
    ├── data/
    │   └── vertex_anatomy_tags.json # 20,139-vertex anatomical mapping table
    ├── models/
    │   └── beating-heart.glb   # Rigged 3D animated glTF cardiac mesh
    └── js/
        ├── heart3d.js          # Three.js WebGL scene, HDR vertex color overdrive, raycaster
        └── app.js              # Client state, form debouncing, and readout synchronization
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.10+ (Recommended Python 3.11)
- Modern web browser with WebGL 2.0 support (Chrome, Edge, Firefox, Safari)

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/xploreshivam/cardiac_risk_prediction.git
cd cardiac_risk_prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain Models (Optional)
Pre-trained models are already included in `models/`. To retrain from scratch:
```bash
python -m src.train
```

### 3. Launch Application
```bash
python app.py
```
Open your browser and navigate to:
- **Application Dashboard:** `http://localhost:5000`
- **Health Check Endpoint:** `http://localhost:5000/health`

---

## REST API Specification

| Endpoint | Method | Expected Content-Type | Response Codes | Description |
|---|---|---|---|---|
| `/` | `GET` | `text/html` | `200` | Main application diagnostic workspace |
| `/health` | `GET` | `application/json` | `200` | Server health and loaded model status |
| `/api/metrics` | `GET` | `application/json` | `200` | 5-fold cross-validation performance metrics |
| `/api/predict` | `POST` | `application/json` | `200`, `400`, `415`, `422` | Clinical risk prediction and territory scores |

### Sample POST `/api/predict` Request Body
```json
{
  "Age": 62,
  "Sex": "Male",
  "BMI": 27.5,
  "BP": 138,
  "Current Smoker": "Yes",
  "DM": "Yes",
  "Typical Chest Pain": "Yes",
  "Atypical": "No",
  "Non-Anginal": "No",
  "St Elevation": "No",
  "St Depression": "Yes",
  "EF-TTE": 45
}
```

### Error Responses
- **`400 Bad Request`**: Request payload is not a valid JSON dictionary.
- **`405 Method Not Allowed`**: Non-POST requests dispatched to `/api/predict`.
- **`415 Unsupported Media Type`**: Headers missing `Content-Type: application/json`.
- **`422 Unprocessable Entity`**: Numeric value out of clinical bounds, categorical value invalid, or non-finite number (`NaN`/`Infinity`) provided.

---

## Validation & Model Performance (5-Fold Stratified CV)

Models were validated using 5-fold stratified cross-validation on the Z-Alizadeh Sani cohort ($N = 303$):

| Target | Accuracy | Guessing Baseline | ROC-AUC | Sensitivity (Recall) | Clinical Confidence |
|---|---|---|---|---|---|
| **Overall CAD** | **86.5%** | 71.3% | **0.93** | 91.2% | Moderate |
| **LAD Artery** | **77.2%** | 58.4% | **0.84** | 79.5% | Moderate |
| **LCX Artery** | **63.7%** | 60.7% | **0.73** | 54.8% | Low / Guarded |
| **RCA Artery** | **65.7%** | 62.4% | **0.71** | 57.1% | Low / Guarded |

- **Why Clinical Confidence is Rated as Moderate (not High/Reliable):**
  The model is trained on a single-center cohort of 303 patients from Tehran Heart Center (Z-Alizadeh Sani dataset) with no external multi-center cohort validation. Even though 5-fold cross-validation demonstrated solid internal discriminative performance (ROC-AUC 0.93 on CAD and 0.84 on LAD), true real-world generalizability across diverse clinical settings cannot be asserted without multi-center external validation. Hence, confidence is designated as **Moderate** rather than High/Reliable.
- *Posterior circulation note:* LCX and RCA stenosis prediction from standard non-invasive features presents known clinical difficulty due to posterior circulation subtlety; the interface transparently flags these territories with **Low Confidence** notices to prevent over-reliance.

---

## Deployment (Render / Heroku)

The repository includes a ready-to-deploy `Procfile`:
```
web: gunicorn app:app
```

On Render:
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

---

## Dataset Attribution & References

- **Dataset:** Alizadehsani, R., Roshanzamir, M., Sani, Z. *Extension of Z-Alizadeh Sani dataset*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset) (Licensed under CC BY 4.0).
- **3D Heart Model:** [“Beating Heart” by Dreamwasabducted on Sketchfab](https://sketchfab.com/3d-models/beating-heart-0fbaea9e0ad74fa78ba772e04313f890) (shortlink: [https://skfb.ly/owVVo](https://skfb.ly/owVVo)), licensed under [Creative Commons Attribution (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **3D Model Modifications (CC-BY Notice):** The original 3D asset was modified for this project: mesh vertices were partitioned into anatomical coronary supply territories (`vertex_anatomy_tags.json`), dynamic GPU vertex color buffer attributes (`BufferAttribute`) were added for real-time risk coloring, and skeletal cycle animation playback was retuned for WebGL.
- **Anatomical Mapping:** AHA 17-segment se inspired, illustrative mapping (heuristic 3D spatial partitioning based on coordinate bounding rules, not a clinically verified patient-specific segmentation).
