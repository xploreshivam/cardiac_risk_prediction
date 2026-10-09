# Cardiac Risk Prediction Using ML & 3D Visualization - System Architecture Flow

This document outlines the end-to-end architecture and data flow of the **Cardiac Risk Prediction** platform, from patient clinical parameter collection and defensive API validation to 4-target machine learning inference and dual visual feedback (3D GPU vertex color overdrive & accessible medical red UI indicators).

---

## Architecture Flow Diagram

```mermaid
graph TD
    subgraph UI ["1. User Interface & Interactive Diagnostic Stage"]
        A["Patient Clinical Form<br/>(12 Variables: Demographics, ECG, Echo EF)"] -->|Input / Change Event| B["Client Payload Generator & Debounce"]
        B -->|JSON Request| C["HTTP POST /api/predict"]
        
        M["Three.js 3D WebGL Canvas<br/>(OrbitControls & AnimationMixer)"] --> N["Raycaster & Camera-Facing<br/>Billboard Anatomy Labels"]
        O["Accessible Clinical Readout Panel<br/>(ARIA Progressbars & WCAG Contrast)"]
        O2["High Risk Status Badges<br/>(Bold Medical Red #E63946 Alert)"]
    end

    subgraph API ["2. Flask REST Backend & Validation Gateway"]
        C --> D{"Defensive Request<br/>Validation Gateway"}
        D -->|Invalid Method| E1["HTTP 405 Method Not Allowed"]
        D -->|Non-JSON Header| E2["HTTP 415 Unsupported Media Type"]
        D -->|Malformed / Non-Dict| E3["HTTP 400 Bad Request"]
        D -->|Out of Range / NaN / Inf| E4["HTTP 422 Unprocessable Entity"]
        D -->|Validated Clean Payload| F["StandardScaler Matrix Transformer"]
        
        E1 --> A
        E2 --> A
        E3 --> A
        E4 --> A
    end

    subgraph ML ["3. Multi-Target ML Inference Pipeline"]
        F --> G1["Overall CAD Classifier<br/>(Balanced Random Forest)"]
        F --> G2["LAD Artery Classifier<br/>(Left Anterior Descending)"]
        F --> G3["LCX Artery Classifier<br/>(Left Circumflex)"]
        F --> G4["RCA Artery Classifier<br/>(Right Coronary Artery)"]
        
        G1 --> H["Risk Probability Aggregator & Threshold Engine"]
        G2 --> H
        G3 --> H
        G4 --> H
        H --> I["HTTP 200 JSON Prediction Response<br/>(Risk %, Confidence, Levels)"]
    end

    subgraph DUAL_DISPATCH ["4. Dual Visual Diagnostic Feedback System"]
        I -->|Update UI Readout| O
        O --> O2
        
        I -->|Update 3D Heart| J["Territory Risk & Color Dispatcher"]
        J --> K{"Risk Threshold Check<br/>(Illustrative Cutoff >= 55%)"}
        
        K -->|0 High Risk Territories| L0["Normal / Baseline Gradient<br/>(Clinical Soft Cyan/Green)"]
        K -->|1 High Risk Territory| L1["Single High-Risk Mode<br/>Stark Glowing White #FFFFFF"]
        K -->|2+ High Risk Territories| L2["Multi-Territory Palette Mode<br/>LAD: White | LCX: Cyan | RCA: Gold"]
        
        L0 --> P["Direct GPU BufferAttribute Color Overdrive<br/>(HDR Float32Array on 20,139 Vertices)"]
        L1 --> P
        L2 --> P
        
        P --> Q["beating-heart.glb Skinned Mesh<br/>(Dynamic Territory Shading)"]
        Q --> M
    end
```

---

## Component Breakdown

### 1. User Interface Layer
- **Clinical Parameter Form**: Captures 12 patient variables (demographics, symptoms, ECG findings, and echocardiography metrics) with synchronized input/change listeners.
- **Interactive 3D Stage**: WebGL-powered 3D stage featuring a realistic beating heart model (`beating-heart.glb`) with full orbit controls and continuous skeletal animations.
- **Raycasting & Billboard Labels**: Allows clicking on myocardial walls and coronary vessels to inspect local risk scores and supply regions with depth-tested camera-facing labels.
- **Accessible Clinical Readout Panel**: Displays numerical stenosis percentages, WCAG 4.5:1 compliant contrast cards, accessible ARIA progressbars (`aria-valuenow`, `aria-live="polite"`), and bold medical red (`#E63946`) high predicted risk alert bars.

### 2. Backend Engine & Defensive Gateway (Flask API)
- **Defensive Multi-Tier Validation**:
  - `405 Method Not Allowed`: Graceful JSON error if non-POST methods are routed to `/api/predict`.
  - `415 Unsupported Media Type`: Guard if client sends raw text or non-`application/json` payload.
  - `400 Bad Request`: Rejection if JSON body is not an object/dictionary.
  - `422 Unprocessable Entity`: Strict numeric boundary checks and `math.isfinite()` validation blocking `NaN` and `Infinity`.
- **Precomputed Scaler**: Applies `StandardScaler` transformation on 12 features before inference.

### 3. Machine Learning Inference Pipeline
- **Ensemble Random Forest Classifiers**: 4 trained models predict independent probabilities:
  1. Overall Coronary Artery Disease (`CAD`)
  2. Left Anterior Descending (`LAD`) stenosis
  3. Left Circumflex (`LCX`) stenosis
  4. Right Coronary Artery (`RCA`) stenosis
- **Risk Aggregator**: Evaluates severe stenosis cutoffs (threshold $\ge 55\%$) and dispatches clinical confidence alerts.

### 4. Three.js Dual Diagnostic Feedback System
- **Real-Time GPU Vertex Shading**: Overdrives Float32 vertex colors directly into `THREE.BufferAttribute` across 20,139 tagged vertices:
  - **Single stenosed territory**: Stark Glowing White (`#FFFFFF`, HDR overdrive multiplier `6.5`).
  - **Multiple stenosed territories**: Distinct contrasting color coding:
    - **LAD**: Stark White (`#FFFFFF`)
    - **LCX**: Electric Cyan (`#00E5FF`)
    - **RCA**: Vivid Amber/Gold (`#FFB800`)
  - **Normal / Low Risk**: Clinical soft baseline gradient (`#1DD1A1` to `#00E5FF`).
- **UI Readout Synchronization**: High predicted risk territories are flagged with bold Medical Red (`#E63946`) status bars alongside 3D color pill tags for instant cross-referencing.
