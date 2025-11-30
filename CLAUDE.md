# CLAUDE.md

Developer guide for Video Quality Assessment Research. Evaluates AI videos via 6 objective metrics (COVER, CLIP, SI-TI, LPIPS, SSIM, TV-L1), P.910 subjective testing, P.1401 validation.

---

## 📁 Repository Structure

```
project/
├─ infrastructure/                # 🎬 Video generation
│  ├─ wan22/                      # Wan2.2 TIV2V (14B, Docker)
│  ├─ hunyuanvideo/               # HunyuanVideo T2V/I2V
│  └─ terraform/                  # AWS + L40S(48gb) / H100(80gb) GPU Compute Server provisioning
│
└─ research-suite/
   ├─ data/
   │  ├─ result_videos/           # source{i}_{j}.mp4 (10)
   │  ├─ input_videos/            # original{i}.mp4 (5) (https://database.mmsp-kn.de/konvid-1k-database.html)
   │  ├─ input_videos_24fps/      # original{i}.mp4 (5)
   │  ├─ input_images/            # original{i}.mp4 (5) style reference images (https://www.kaggle.com/datasets/skjha69/artistic-images-for-neural-style-transfer)
   │  └─ test_videos/             # 
   │
   ├─ qualifier/                  # 🔬 6 metric suites (venv/, src/, results/)
   │  ├─ cover_assessment/        
   │  ├─ clip-score_assessment/
   │  ├─ si-ti_assessment/
   │  ├─ tv-l1_assessment/
   │  └─ lpips-ssim_assessment/ 
   │
   ├─ p910/video-player/          # 👥 Web UI
   └─ p1401/src/                  # 📊 ci95.py, mapping.py
```

---

## ⚡ Quick Commands

### 🔬 Objective Metrics
```bash
# COVER (NR, multi-dimensional neural)
cd research-suite/qualifier/cover_assessment && source venv/bin/activate
python src/cover_assessment.py --input-dir ../../data/result_videos && deactivate

# CLIP (NR, semantic alignment)
cd ../clip-score_assessment && source venv/bin/activate
python src/clip_score_assessment.py --input ../../data/result_videos && deactivate

# SI-TI (NR, spatial/temporal complexity)
cd ../si-ti_assessment && source venv/bin/activate
python src/main.py --input ../../data/result_videos --output ./results && deactivate

# TV-L1 (NR, optical flow consistency)
cd ../tv-l1_assessment && source venv/bin/activate
python src/tv-l1_assessment.py --input ../../data/result_videos && deactivate

# LPIPS + SSIM (FR, perceptual similarity)
cd ../ls_assessment && source venv/bin/activate
python main.py --input-dir ../../data/result_videos --reference-dir ../../data/Input_videos && deactivate
```

### 👥 Subjective Testing
```bash
cd ../../p910/video-player && ln -s ../../data/result_videos videos
python3 -m http.server 8000  # → http://localhost:8000
```

### 📊 Validation
```bash
cd ../../p1401 && source venv/bin/activate
python src/ci95.py -i ../p910/results/ -o results/mos/mos_results.csv
python src/mapping.py --mos results/mos/mos_results.csv \
  --metrics ../qualifier/*/results/*.csv --output results/p1401/
deactivate
```

---

## 🎨 Qualifier Modules

| Metric | Type | Output | Use Case |
|--------|------|--------|----------|
| **COVER** | NR | semantic, technical, aesthetic, overall | Multi-dimensional quality |
| **CLIP** | NR | mean/median semantic scores | Prompt adherence |
| **SI-TI** | NR | SI/TI mean/std/max | Scene complexity |
| **LPIPS** | FR | mean/median/std/min/max perceptual distance | Similarity to reference |
| **SSIM** | FR | mean/median/std/min/max structural similarity | Similarity to reference |
| **TV-L1** | NR | 11 temporal metrics (fb_error, warp_error, Q_*) | Temporal coherence |

*NR=No-Reference, FR=Full-Reference*

### COVER
3-branch neural (CLIP ViT-L/14 + Swin3D + ConvNeXt). Unbounded scores (negative normal). GPU ~30s/video.
```csv
Filename,semantic_score,technical_score,aesthetic_score,overall_score
source1_1.mp4,-1.234,0.567,-0.789,-0.452
```

### CLIP
Text-video embedding similarity. `cosine_similarity(CLIP_text(prompt), CLIP_image(frame))`. GPU ~10s/video.
```csv
Filename,clip_mean,clip_median
source1_1.mp4,0.76,0.78
```

### SI-TI
Classical spatial/temporal metrics. `SI = stddev(Sobel(Y))`, `TI = stddev(Y_n - Y_{n-1})`. CPU ~5s/video.
```json
{"source1_1.mp4": {"SI_mean": 45.67, "TI_mean": 23.45, ...}}
```

### LPIPS
AlexNet deep features + learned weights. Range [0,∞), lower = similar. GPU ~20s/video.
```csv
Filename,LPIPS_mean,LPIPS_median,LPIPS_std,LPIPS_min,LPIPS_max
source1_1.mp4,0.234,0.231,0.045,0.189,0.312
```

### SSIM
Structural similarity (luminance × contrast × structure). Range [0,1], 1 = identical. CPU ~3s/video.
```csv
Filename,SSIM_mean,SSIM_median,SSIM_std,SSIM_min,SSIM_max
source1_1.mp4,0.876,0.882,0.034,0.812,0.934
```

### TV-L1
DualTVL1 optical flow. 11 metrics: fb_error, warp_error, motion_magnitude, Q_fb, Q_warp, etc. GPU ~15s/video.
```csv
Filename,fb_error,warp_error,motion_magnitude,Q_fb,Q_warp,...
source1_1.mp4,2.34,1.56,12.3,0.876,0.912,...
```

**LPIPS+SSIM pairing**: `source{i}_{j} → original{j}` (10 pairs)
**Optional args**: `--image-size 256`, `--frame-stride 2`, `--max-frames 100`, `--cpu`

---

## 🎭 P.910 Subjective Testing

**Setup**:
```bash
cd research-suite/p910/video-player
ln -s ../../data/result_videos videos
python3 -m http.server 8000
```

**Compliance**:
- ACR 5-point: 1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent
- Randomization: Fisher-Yates (app.js:226-233)
- Grey screens: 50% grey, 2s (#808080)
- Test video: Mandatory training

**Config** (`config.json`):
```json
{
  "greyScreenDuration": 2000,
  "testVideo": {"enabled": true, "filename": "test.mp4"},
  "export": {"filenamePattern": "p910_assessment_{participantId}.csv"}
}
```

**Output**:
```csv
ParticipantID,VideoIndex,Filename,Rating,Timestamp,ResponseTime_seconds
alice,0,source1_3.mp4,4,2024-11-30T10:15:23Z,3.2
```

---

## 📊 P.1401 Statistical Validation

### MOS Computation (ci95.py)
```bash
python src/ci95.py -i ../p910/results/ -o results/mos/mos_results.csv
```
`MOS = mean(ratings)`, `CI95 = t(0.975, N-1) × STD / √N`

**Output**:
```csv
Filename,MOS,STD,N,CI95
source1_1.mp4,3.45,0.87,13,0.53
```

### Metric Validation (mapping.py)
```bash
python src/mapping.py --mos mos_results.csv --metrics *.csv --output results/p1401/
```

**Process** (per metric):
1. Load MOS + CI95, merge metrics on `Filename`
2. Fit 3rd-order polynomial: `MOS_pred = a₀ + a₁x + a₂x² + a₃x³`
3. Pearson (r), Spearman (ρ)
4. RMSE, RMSE* (CI95-discounted)
5. LOOCV → RMSE_CV
6. Gap = RMSE_CV - RMSE
7. Categorize (Excellent/Good/Fair/Poor)
8. Generate plots

**Outputs**:
- `p1401_summary_enhanced.csv`: All metrics × validation stats
- `p1401_ranking_table.csv`: Ranked by performance
- `{metric}_p1401.png`: 50+ scatter plots

**Performance Categories**:

| Category | Criteria | Interpretation |
|----------|----------|----------------|
| Excellent | RMSE_CV < 0.3 AND \|r\| > 0.7 | Outstanding |
| Good | RMSE_CV < 0.5 AND \|r\| > 0.5 | Reliable |
| Fair | RMSE_CV < 0.7 AND \|r\| > 0.3 | Moderate |
| Poor | Otherwise | Weak |

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Videos not loading (P910) | Missing symlink | `ln -s ../../data/result_videos videos` |
| RMSE_CV >> RMSE | Overfitting (N < 20) | Collect more data |
| COVER negative scores | Normal (unbounded) | Use relative ranking |
| "No module 'X'" | Wrong venv | `source venv/bin/activate` in tool dir |
| CUDA OOM | Insufficient VRAM | Use `--cpu` flag |
| LPIPS/SSIM mismatch | Wrong pairing | Verify `source{i}_{j} → original{j}` |
| mapping.py merge error | Filename mismatch | Ensure all CSVs have `Filename` column |

---

## 📐 Implementation Notes

**Virtual Environments**: Isolated per tool (Python 3.9). Always activate: `cd {tool} && source venv/bin/activate`

**Data Locations**:
- Generated: `research-suite/data/result_videos/` (10 videos)
- References: `research-suite/data/Input_videos/` (5 originals)

**P.1401 LOOCV**: Leave-one-out cross-validation. Train N-1, test 1, repeat N times. Tests generalization.

**RMSE\* Denominator**: (N-4) accounts for 4 polynomial coefficients. Degrees of freedom = N - k.

**Metric Standardization**: All CSVs must have `Filename` column + numeric metric columns + header row.

---

## 📚 Key Formulas

### Statistical Validation
```
MOS_pred = a₀ + a₁x + a₂x² + a₃x³
RMSE = √(Σ(MOS - MOS_pred)² / (N-1))
RMSE* = √(Σ max(|MOS - MOS_pred| - CI95, 0)² / (N-4))
CI95 = t(0.975, N-1) × STD / √N
```

### Quality Metrics
```
COVER: Multi-branch regression (CLIP + Swin3D + ConvNeXt)
CLIP: cosine_similarity(text_emb, frame_emb)
SI: stddev(Sobel(Y)) where Y = 0.299R + 0.587G + 0.114B
TI: stddev(Y_n - Y_{n-1})
LPIPS: weighted L2(AlexNet(x), AlexNet(y))
SSIM: l(x,y) × c(x,y) × s(x,y)
TV-L1: Q = exp(-α × error)
```

---

## 🚀 Quick Reference

### Venv Activation
```bash
cd research-suite/qualifier/cover_assessment && source venv/bin/activate
cd research-suite/qualifier/clip-score_assessment && source venv/bin/activate
cd research-suite/qualifier/si-ti_assessment && source venv/bin/activate
cd research-suite/qualifier/tv-l1_assessment && source venv/bin/activate
cd research-suite/qualifier/ls_assessment && source venv/bin/activate
cd research-suite/p1401 && source venv/bin/activate
```

### Output Locations
```
research-suite/qualifier/cover_assessment/results/cover_results.csv
research-suite/qualifier/clip-score_assessment/results/clip_score_results.csv
research-suite/qualifier/si-ti_assessment/results/si_ti_results.csv
research-suite/qualifier/tv-l1_assessment/results/tv_l1_results.csv
research-suite/qualifier/ls_assessment/lpips_ssim_results.csv
research-suite/p910/results/p910_assessment_{ID}.csv
research-suite/p1401/results/mos/mos_results.csv
research-suite/p1401/results/p1401_summary_enhanced.csv
research-suite/p1401/results/p1401_ranking_table.csv
research-suite/p1401/results/{metric}_p1401.png
```

---

**Research Guide**: See `README.md` for theory, background, references
