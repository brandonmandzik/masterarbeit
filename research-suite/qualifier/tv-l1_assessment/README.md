# TV-L1 Video Quality Assessment

A no-reference video quality assessment tool based on TV-L1 optical flow analysis. This tool evaluates temporal consistency in videos by computing optical flow-based metrics that correlate with perceived video quality.

## Overview

This tool analyzes video quality without requiring a reference video by measuring:
- **Forward-backward consistency**: How well optical flow estimates agree in both temporal directions
- **Temporal warp error**: How accurately consecutive frames can be predicted using optical flow
- **Motion magnitude variance**: Temporal stability of motion patterns

These metrics are combined to produce quality scores that can detect temporal artifacts, flickering, warping, and motion inconsistencies commonly found in AI-generated or heavily processed videos.

## Features

- No-reference quality assessment (no ground truth needed)
- Dense optical flow using TV-L1 algorithm
- Multiple complementary metrics (L1 and L2 norms)
- Configurable frame sampling for performance tuning
- JSON output for easy integration into pipelines
- Alternative Farneback implementation available (`farneback.py`)

## Installation

### Requirements

- Python 3.7+
- OpenCV with contrib modules (for TV-L1 optical flow)
- NumPy

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-contrib-python numpy
```

**Note**: The standard `opencv-python` package does NOT include the TV-L1 optical flow implementation. You must install `opencv-contrib-python` instead.

## Usage

### Basic Usage

```bash
python assess_videos.py --video path/to/video.mp4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | *required* | Path to input video file |
| `--max-frames` | int | None | Limit number of frames processed (for faster evaluation) |
| `--step` | int | 1 | Process every N-th frame pair (e.g., 2 = every other pair) |
| `--pretty` | flag | False | Pretty-print JSON output |

### Examples

**Evaluate full video:**
```bash
python assess_videos.py --video sample.mp4 --pretty
```

**Quick evaluation (first 100 frames):**
```bash
python assess_videos.py --video sample.mp4 --max-frames 100
```

**Faster evaluation (every 3rd frame pair):**
```bash
python assess_videos.py --video sample.mp4 --step 3 --pretty
```

### Output Format

The tool outputs JSON with the following structure:

```json
{
  "fb_mean_l1": 0.234,
  "fb_mean_l2": 0.156,
  "warp_mean_l1": 0.012,
  "warp_mean_l2": 0.008,
  "motion_mag_mean": 2.45,
  "motion_mag_var": 0.123,
  "Q_fb_mean_l1": 0.791,
  "Q_fb_mean_l2": 0.856,
  "Q_warp_mean_l1": 0.988,
  "Q_warp_mean_l2": 0.992,
  "Q_motion_mag_var": 0.884,
  "num_pairs": 299
}
```

## Metrics Explained

### Raw Error Metrics (Lower is Better)

#### Forward-Backward Consistency Error
- `fb_mean_l1`: Mean L1 norm of flow inconsistency (|u| + |v|)
- `fb_mean_l2`: Mean L2 norm of flow inconsistency (√(u² + v²))

For each pixel, we compute forward flow (t → t+1) and backward flow (t+1 → t). In a perfectly consistent scene, these should be inverses: `flow_forward + flow_backward ≈ 0`. Deviations indicate temporal inconsistencies, artifacts, or unreliable motion estimates.

**Interpretation:**
- Low values (< 0.5): Good temporal consistency
- Medium values (0.5 - 2.0): Moderate artifacts or complex motion
- High values (> 2.0): Significant temporal inconsistencies or artifacts

#### Temporal Warp Error
- `warp_mean_l1`: Mean L1 photometric error after warping
- `warp_mean_l2`: RMSE after warping (Root Mean Square Error)

We warp frame t+1 back to frame t using the forward flow and measure the pixel-wise difference. Lower error means the optical flow accurately predicts frame-to-frame changes.

**Interpretation:**
- Low values (< 0.02): Excellent temporal prediction
- Medium values (0.02 - 0.05): Acceptable quality
- High values (> 0.05): Poor temporal coherence or challenging motion

#### Motion Magnitude Statistics
- `motion_mag_mean`: Average optical flow magnitude across all frames
- `motion_mag_var`: Variance of motion magnitude over time

These metrics characterize the motion content and temporal stability:
- **Mean**: Indicates overall amount of motion (higher = more motion)
- **Variance**: Indicates temporal stability (lower = more stable motion patterns)

High variance suggests flickering, jitter, or temporally inconsistent motion artifacts.

### Quality Scores (Higher is Better)

Quality scores transform errors into 0-1 scale scores using Q = exp(-α × Error):

- `Q_fb_mean_l1`: Forward-backward consistency quality (L1)
- `Q_fb_mean_l2`: Forward-backward consistency quality (L2)
- `Q_warp_mean_l1`: Temporal warp quality (L1)
- `Q_warp_mean_l2`: Temporal warp quality (L2)
- `Q_motion_mag_var`: Motion stability quality

**Interpretation:**
- 0.9 - 1.0: Excellent quality
- 0.7 - 0.9: Good quality
- 0.5 - 0.7: Fair quality
- < 0.5: Poor quality

## Technical Details

### TV-L1 Optical Flow

The TV-L1 algorithm (Total Variation - L1) is a variational optical flow method that:
- Minimizes a combination of data term (L1 photometric consistency) and regularization term (Total Variation)
- Provides robust, dense flow estimates
- Handles large displacements and motion discontinuities well
- More accurate than classical methods (e.g., Farneback) for complex scenes

**OpenCV Convention:**
```
flow[y, x] = (u, v)
```
where pixel at (x, y) in frame t maps to (x+u, y+v) in frame t+1.

### Algorithm Overview

For each consecutive frame pair:

1. **Compute Bidirectional Flow:**
   - Forward flow: frame_t → frame_{t+1}
   - Backward flow: frame_{t+1} → frame_t

2. **Forward-Backward Consistency Check:**
   - For each pixel p in frame_t:
     - Warp to p' = p + flow_forward(p) in frame_{t+1}
     - Sample flow_backward at p'
     - Compute error: ||flow_forward(p) + flow_backward(p')||

3. **Temporal Warp Error:**
   - Warp frame_{t+1} back to frame_t using flow_forward
   - Compute photometric difference with frame_t

4. **Motion Analysis:**
   - Compute flow magnitude at each pixel
   - Track statistics across temporal sequence

5. **Aggregation:**
   - Average metrics across all frame pairs
   - Transform errors to quality scores

### Implementation Notes

- **Grayscale conversion**: Optical flow computed on grayscale frames (see `assess_videos.py:13`)
- **Float32 precision**: Frames normalized to [0, 1] range
- **Border handling**: Reflects pixels at boundaries during warping
- **Invalid pixel handling**: NaN values for pixels that map outside frame boundaries
- **Reusable TV-L1 object**: Single instance created for efficiency (`assess_videos.py:57`)

## TV-L1 vs Farneback

This directory contains two implementations:

| Feature | `assess_videos.py` (TV-L1) | `farneback.py` (Farneback) |
|---------|---------------------------|----------------------------|
| Algorithm | TV-L1 variational method | Polynomial expansion |
| Accuracy | Higher (especially large motions) | Good for moderate motion |
| Speed | Slower | Faster |
| Large displacement | Excellent | Limited |
| Motion boundaries | Sharp, well-preserved | May blur discontinuities |
| Dependencies | Requires opencv-contrib | Standard opencv package |

**Recommendation:**
- Use **TV-L1** (`assess_videos.py`) for highest accuracy and research applications
- Use **Farneback** (`farneback.py`) for faster evaluation or when opencv-contrib is unavailable

## Use Cases

- **AI-generated video quality assessment**: Detect temporal artifacts in diffusion models, frame interpolation, or video synthesis
- **Video compression analysis**: Measure temporal quality degradation
- **Video processing pipeline validation**: Ensure temporal consistency after editing, stabilization, or effects
- **Temporal artifact detection**: Identify flickering, warping, or motion inconsistencies
- **Video restoration quality**: Assess denoising, super-resolution, or colorization results

## Performance Considerations

### Speed Optimization

1. **Limit frames**: Use `--max-frames` for quick evaluations
   ```bash
   python assess_videos.py --video input.mp4 --max-frames 50
   ```

2. **Subsample frame pairs**: Use `--step` to skip frames
   ```bash
   python assess_videos.py --video input.mp4 --step 5
   ```

3. **Consider Farneback**: Use `farneback.py` for 2-3x speedup
   ```bash
   python farneback.py --video input.mp4
   ```

### Memory Considerations

The tool loads frames into memory. For very long videos:
- Use `--max-frames` to cap memory usage
- Process video in chunks and average results
- Consider downsampling video resolution beforehand

### Typical Performance

On a modern CPU (example: Apple M1):
- 720p video: ~1-2 seconds per frame pair (TV-L1)
- 1080p video: ~3-5 seconds per frame pair (TV-L1)
- Farneback: 2-3x faster than TV-L1

## Example Output Interpretation

```json
{
  "fb_mean_l1": 0.156,        // Low error → good consistency
  "fb_mean_l2": 0.098,        // Low error → good consistency
  "warp_mean_l1": 0.018,      // Very low → excellent prediction
  "warp_mean_l2": 0.012,      // Very low → excellent prediction
  "motion_mag_mean": 3.42,    // Moderate motion content
  "motion_mag_var": 0.087,    // Low variance → stable motion
  "Q_fb_mean_l1": 0.856,      // Good quality (> 0.8)
  "Q_fb_mean_l2": 0.907,      // Excellent quality (> 0.9)
  "Q_warp_mean_l1": 0.982,    // Excellent quality (> 0.9)
  "Q_warp_mean_l2": 0.988,    // Excellent quality (> 0.9)
  "Q_motion_mag_var": 0.917,  // Excellent stability (> 0.9)
  "num_pairs": 149            // 150 frame video
}
```

**Analysis:** This video shows excellent temporal quality with good flow consistency, accurate motion prediction, and stable motion patterns. Quality scores are all above 0.85, indicating high-quality temporal characteristics.

## Troubleshooting

### Import Error: `cv2.optflow`

**Error:**
```
AttributeError: module 'cv2' has no attribute 'optflow'
```

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Video Not Opening

**Error:**
```
RuntimeError: Could not open video: path/to/video.mp4
```

**Solutions:**
- Verify the file path is correct
- Ensure OpenCV has proper codec support
- Try converting video to a common format (H.264/MP4)

### Memory Issues

**Error:**
```
MemoryError or killed process
```

**Solutions:**
```bash
# Limit frames
python assess_videos.py --video large.mp4 --max-frames 100

# Reduce video resolution beforehand
ffmpeg -i input.mp4 -vf scale=640:-1 input_small.mp4
```

## License

This tool is part of the research suite for video quality assessment.

## Related Tools

In the parent `research-suite` directory:
- `cover_assessment/`: COVER video quality metric
- `dover_assessment/`: DOVER video quality metric
- `tlpips_assessment/`: LPIPS-based temporal assessment

# Sources
- https://docs.opencv.org/3.4/dc/d47/classcv_1_1DualTVL1OpticalFlow.html
- https://www.ipol.im/pub/art/2013/26/article.pdf