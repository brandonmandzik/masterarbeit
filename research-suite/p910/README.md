# 📊 P.910 Video Quality Assessment

## 🎯 Overview

Web-based implementation of ITU-T P.910 standard for subjective video quality assessment. Enables systematic collection of Absolute Category Rating (ACR) data using a 5-point scale, following internationally recognized methodologies.

## ✨ Features

### 🎬 Assessment
- 5-point ACR scale (Bad → Excellent)
- 🎲 Fisher-Yates randomization
- ⬜ Grey screen intervals (50% grey, 2s before/after)
- 🔇 Muted autoplay, no user controls

### 📈 Data
- 👤 Participant tracking
- ⏱️ Response time measurement
- 🕒 Timestamp logging
- 📄 CSV export: `ParticipantID, VideoIndex, Filename, Rating, Timestamp, ResponseTime`

### 🖥️ Interface
- 📊 Progress indicator
- ⏳ Loading feedback
- 📋 Session summary
- 📱 Responsive layout

### ⚙️ Config
- 🔍 Auto-scan videos (mp4, webm, ogg, mov)
- ⏲️ Adjustable grey screen duration
- 📝 JSON configuration

## 🚀 Quick Start

```bash
cd video-player
ln -s ../../data/source_videos videos
python3 -m http.server 8000
```

**→** Visit `http://localhost:8000` → Complete assessment → CSV saved to `/results`

## 🏗️ Architecture

Client-side state machine with four phases: **welcome** → **video** → **rating** → **completion**

### 🔄 Flow
1. Load config → scan videos → validate
2. Initialize VideoAssessment (state + events)
3. Participant ID → Fisher-Yates shuffle
4. Loop: `grey(2s) → play → grey(2s) → rate → store`
5. Generate CSV → download

### 🧩 Components
| File | Role |
|------|------|
| `index.html` | DOM structure, 4 screen states |
| `app.js` | VideoAssessment class, state management |
| `config.json` | Study parameters (grey duration, paths) |
| `styles.css` | ITU-compliant styling (50% grey, dark UI) |

### 📐 Flowcharts

**High-Level Flow:**
```
User → Config Load → Validate
  ↓
Welcome (ID Input) → Fisher-Yates Shuffle
  ↓
┌─────────────────────────────────┐
│ Loop (per video):               │
│  Grey(2s) → Play → Grey(2s)     │
│  → Rate → Store [ID, idx, file, │
│    rating, time, response]      │
└─────────────────────────────────┘
  ↓
Completion → CSV Download
```

**Component Interaction:**
```
index.html ──┐
             ↓
config.json ──→ app.js (VideoAssessment) ←── styles.css
                  │
                  ├─→ State: [participantId, videoSequence[], ratings[]]
                  ├─→ Events: [start, load, submit, generate, download]
                  └─→ /results/*.csv
```

## 🛠️ Technologies

| Technology | Purpose | Concept |
|------------|---------|---------|
| **HTML5 Video API** | Video playback control | `<video>` events (loadeddata, ended, error) for state tracking |
| **Vanilla JS (ES6+)** | Client logic | async/await, classes, DOM manipulation without frameworks |
| **Fetch API** | Config/directory loading | Promise-based async HTTP requests |
| **DOMParser** | Video auto-detection | Parse HTML directory listings to extract file links |
| **Fisher-Yates** | Randomization | Unbiased shuffle preventing order bias (P.910 requirement) |
| **Blob API** | CSV export | In-memory file creation + download without server |
| **CSS3 Transitions** | Visual timing | Hardware-accelerated opacity/visibility for grey screens |

## 📚 Foundation Knowledge

### 💻 Technical Prerequisites
- HTTP protocol (request/response, localhost servers)
- JavaScript async programming (promises, async/await)
- DOM events and HTML5 media lifecycle
- CSV encoding and browser security (CORS)

### 🧠 Theoretical Prerequisites
- **Subjective vs Objective Quality:** Human perception vs algorithmic metrics
- **P.910 ACR Methodology:** 5-point scale, grey screens, randomization
- **Psychophysics:** Visual stabilization, order bias prevention
- **Human Factors:** Response time, fatigue effects in quality assessment

## 📖 References

**Primary Standard:**
- 📜 [ITU-T P.910 (10/2023)](https://www.itu.int/rec/T-REC-P.910-202310-I/en) - Subjective video quality assessment methods for multimedia applications
  - Defines ACR methodology, 5-point scales, grey screen requirements, test procedures
