# SI/TI Assessment Tool

A Python tool for calculating **Spatial Information (SI)** and **Temporal Information (TI)** metrics for video quality assessment according to **ITU-T Recommendation P.910**.

## Reference
ITU-T P.910: Subjective video quality assessment methods for multimedia applications
https://www.itu.int/rec/T-REC-P.910-202310-I/en

## Features

- **ITU-T P.910 Compliant**: Implements the official SI/TI calculation formulas
- **Memory Efficient**: Processes frames incrementally - handles videos of any length
- **Multi-format Support**: Processes MP4, AVI, MOV, MKV, FLV, WMV, and WebM videos
- **Comprehensive Analysis**: Provides per-frame values and aggregate statistics
- **Rich Visualizations**: Generates multiple plot types using Matplotlib
  - Time series plots (SI/TI over time)
  - Per-frame SI-TI scatter plot (detailed temporal analysis)
  - Standard P.910 classification plot (content complexity quadrants)
  - Distribution histograms
  - Summary statistics bar charts
- **Multiple Output Formats**: CSV, JSON, and human-readable text reports

## SI/TI Calculations

### Spatial Information (SI)
Measures spatial complexity using Sobel edge detection:
```
SI = stddev(Sobel(Frame))
```

- Converts frame to grayscale using ITU-R BT.601 luma formula: `Y = 0.299*R + 0.587*G + 0.114*B`
- Applies Sobel edge detection filters (horizontal and vertical)
- Calculates standard deviation of edge-filtered pixels
- Higher SI values indicate more spatial detail/complexity

### Temporal Information (TI)
Measures temporal complexity based on frame differences:
```
TI = stddev(Frame_n - Frame_{n-1})
```

- Computes pixel-wise differences between consecutive frames
- Calculates standard deviation of temporal changes
- Higher TI values indicate more motion/temporal activity

## Installation

### 1. Clone or navigate to the project directory
```bash
cd si-ti-assessment
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python src/main.py --input /path/to/videos --output ./output
```

### Command-Line Arguments

- `-i, --input`: **(Required)** Input directory containing video files
- `-o, --output`: Output directory for results (default: `./output`)
- `--no-csv`: Skip CSV export of per-frame data
- `--no-json`: Skip JSON export of statistics
- `--no-plots`: Skip plot generation

### Examples

**Process all videos in a folder:**
```bash
python src/main.py -i ./videos -o ./results
```

**Generate only plots (skip CSV/JSON):**
```bash
python src/main.py -i ./videos -o ./results --no-csv --no-json
```

**Generate only data files (skip plots):**
```bash
python src/main.py -i ./videos -o ./results --no-plots
```

## Output Files

The tool generates the following outputs in the specified output directory:

### 1. CSV Files (per video)
`<video_name>_si_ti.csv` - Frame-by-frame SI and TI values
```
Frame,SI,TI
0,45.23,N/A
1,46.12,12.34
2,44.87,11.98
...
```

### 2. JSON File
`si_ti_results.json` - Aggregate statistics for all videos
```json
{
  "video.mp4": {
    "video_info": {...},
    "statistics": {
      "si_max": 78.45,
      "si_mean": 45.67,
      "ti_max": 23.45,
      "ti_mean": 12.34
    }
  }
}
```

### 3. Summary Report
`si_ti_summary.txt` - Human-readable text report

### 4. Plots (PNG format, 300 DPI)
- `si_ti_timeseries.png` - SI/TI values over time for each video
- `si_ti_scatter.png` - Per-frame SI-TI distribution with mean markers
- `si_ti_p910_classification.png` - Standard P.910 content classification (one point per video with quadrants)
- `si_ti_histograms.png` - Distribution histograms
- `si_ti_summary.png` - Summary statistics bar charts

## Project Structure

```
si-ti-assessment/
├── venv/                   # Virtual environment (created during setup)
├── src/
│   ├── __init__.py
│   ├── main.py            # CLI entry point
│   ├── video_processor.py # Frame extraction from videos
│   ├── si_ti_calculator.py # SI/TI calculation (ITU-T P.910)
│   └── visualizer.py      # Matplotlib plotting
├── output/                # Default output directory
├── requirements.txt       # Python dependencies
├── .gitignore
└── readme.md             # This file
```

## Dependencies

- **opencv-python** (4.10.0.84): Video processing and frame extraction
- **numpy** (1.26.4): Numerical computations
- **scipy** (1.13.1): Sobel filter implementation
- **matplotlib** (3.9.0): Plot generation
- **tqdm** (4.66.4): Progress bars

## Interpreting Results

### SI (Spatial Information)
- **Low SI (< 30)**: Simple scenes with little spatial detail (e.g., talking heads, static backgrounds)
- **Medium SI (30-60)**: Moderate complexity (e.g., indoor scenes with some texture)
- **High SI (> 60)**: Complex scenes with high detail (e.g., nature, crowds, detailed textures)

### TI (Temporal Information)
- **Low TI (< 10)**: Minimal motion (e.g., static camera, slow movements)
- **Medium TI (10-25)**: Moderate motion (e.g., normal camera movements, walking)
- **High TI (> 25)**: High motion (e.g., sports, action scenes, fast camera pans)

### SI-TI Plots

The tool generates **two types of SI-TI visualizations**:

#### 1. Per-Frame Scatter Plot (`si_ti_scatter.png`)
Shows detailed temporal analysis with all frame-by-frame SI/TI values:
- Each point represents one frame pair's SI and TI values
- Star markers indicate mean SI/TI for each video
- Useful for analyzing temporal variation within videos

#### 2. P.910 Classification Plot (`si_ti_p910_classification.png`)
Standard ITU-T P.910 content classification (one point per video):
- Each video represented by a single point at (mean SI, mean TI)
- Four classification quadrants (default thresholds: SI=35, TI=18):
  - **Simple/Static** (Low SI < 35, Low TI < 18): Talking heads, simple backgrounds
  - **Complex/Static** (High SI ≥ 35, Low TI < 18): Detailed scenes with minimal motion
  - **Simple/Dynamic** (Low SI < 35, High TI ≥ 18): Simple scenes with high motion
  - **Complex/Dynamic** (High SI ≥ 35, High TI ≥ 18): Action scenes, sports, complex motion
- Useful for comparing content complexity across multiple videos

## Troubleshooting

**No video files found:**
- Ensure your video files have supported extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`
- Check that the input directory path is correct

**Unable to open video file:**
- Verify the video file is not corrupted
- Ensure OpenCV can decode the video codec

**Matplotlib style warnings:**
- The tool uses seaborn style if available, but gracefully falls back to default style
- No action needed - plots will still be generated correctly

## License

This tool is provided as-is for research and educational purposes.

## Citation

If you use this tool in your research, please cite the ITU-T P.910 recommendation:

```
ITU-T Recommendation P.910 (2023), Subjective video quality assessment methods
for multimedia applications. International Telecommunication Union, Geneva.
```