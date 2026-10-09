# Cardiac 3D Risk - System Architecture Flow

This document outlines the end-to-end architecture and data flow of the **Cardiac 3D Risk** platform, from patient clinical parameter collection to 3D GPU vertex color rendering.

---

## Architecture Flow Diagram

```mermaid
graph TD
    subgraph UI ["User Interface Layer"]
        A["Patient Clinical Form"] --> B["Payload Generator"]
        B --> C["HTTP POST /api/predict"]
        M["Interactive 3D Stage"] --> N["Anatomical Raycasting"]
        O["Result Readout Panel"]
    end

    subgraph API ["Flask Backend Engine"]
        C --> D{"Schema & Range Validation"}
        D -->|Validation Failed| E["HTTP 422 JSON Error"]
        D -->|Valid| F["Matrix Transformer"]
        E --> A
    end

    subgraph ML ["Machine Learning Inference Pipeline"]
        F --> G1["Overall CAD Classifier"]
        F --> G2["LAD Artery Classifier"]
        F --> G3["LCX Artery Classifier"]
        F --> G4["RCA Artery Classifier"]
        G1 --> H["Risk Probability Aggregator"]
        G2 --> H
        G3 --> H
        G4 --> H
    end

    subgraph GPU ["Three.js 3D Heart Renderer"]
        H --> I["HTTP 200 Prediction Result"]
        I --> O
        I --> J["Territory Risk Dispatcher"]
        J --> K{"Stenosis Risk Check"}
        K -->|Risk >= 55%| L1["Stark White Shading #FFFFFF"]
        K -->|Risk < 55%| L2["Clinical Gradient Normal/Mild"]
        L1 --> P["Direct GPU Buffer Attribute Update"]
        L2 --> P
        P --> Q["beating-heart.glb Skinned Mesh"]
        Q --> M
    end
```

---

## Component Breakdown

### 1. User Interface Layer
- **Clinical Parameter Form**: Captures 12 patient variables (demographics, symptoms, ECG findings, and echocardiography metrics).
- **Interactive 3D Stage**: WebGL-powered 3D stage featuring a realistic beating heart model with full orbit controls.
- **Raycasting**: Allows clicking on myocardial walls and coronary vessels to inspect local risk scores and supply regions.

### 2. Backend Engine (Flask API)
- **Validation**: Ensures all numeric ranges and categorical values are strictly verified before ML execution.
- **Error Handling**: Gracefully returns `415` for invalid media types, `422` for schema violations, and `500` for server exceptions.

### 3. Machine Learning Inference Pipeline
- **Ensemble Random Forest Models**: Four trained classifiers predict:
  1. Overall Coronary Artery Disease (CAD)
  2. Left Anterior Descending (LAD) stenosis
  3. Left Circumflex (LCX) stenosis
  4. Right Coronary Artery (RCA) stenosis

### 4. Three.js 3D Heart Renderer
- **Anatomical Vertex Shading**: Direct GPU BufferAttribute updating paints damaged myocardium and stenosed vessels in stark glowing white (`#FFFFFF`).
- **Real-Time Beating Animation**: Synced with Three.js AnimationMixer for natural cardiac cycles.
