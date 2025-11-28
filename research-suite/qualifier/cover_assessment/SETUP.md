# COVER Assessment - Setup Instructions

Complete installation guide for the COVER video quality assessment tool.

## Prerequisites

- **Python**: 3.9 or higher
- **pip**: Package manager (comes with Python)
- **Git**: Version control system
- **Storage**: ~300MB (250MB model + dependencies)
- **RAM**: ~4GB minimum for inference

## Installation

### Step 1: Navigate to Directory

```bash
cd /Users/b.mandzik/Documents/Workspace/MA/codebase/research-suite/cover_assessment
```

### Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 3: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9+

# Check installed packages
pip list | grep cover
pip list | grep torch

# Test the script
python assess_videos.py --help
```

### Step 4: Run Assessment

```bash
# Process all videos in ../source_videos/
python assess_videos.py
```

### Step 5: Deactivate When Done

```bash
deactivate
```

## Installation Already Complete

This project has been pre-installed with:

1. **Virtual environment** created at `venv/`
2. **COVER repository** cloned to `cover_repo/`
3. **Pretrained model** downloaded to `pretrained_weights/COVER.pth`
4. **Dependencies** installed (see `requirements-frozen.txt`)
5. **ARM64 compatibility** configured using PyAV instead of decord

## Manual Reinstallation (if needed)

If you need to recreate the environment from scratch:

### 1. Remove Old Environment

```bash
cd /Users/b.mandzik/Documents/Workspace/MA/codebase/research-suite/cover_assessment
rm -rf venv
rm -rf cover_repo
rm -rf pretrained_weights
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Clone COVER Repository

```bash
git clone https://github.com/taco-group/COVER.git cover_repo
```

### 5. Modify Requirements for ARM64

Edit `cover_repo/requirements.txt` to replace `decord` with `av`:

```bash
sed -i '' 's/decord/av/g' cover_repo/requirements.txt
```

Or manually change line 4 from `decord` to `av`.

### 6. Install COVER Package

```bash
cd cover_repo
pip install -e .
cd ..
```

### 7. Install Additional Requirements

```bash
pip install -r requirements.txt
```

### 8. Freeze Requirements

```bash
pip freeze > requirements-frozen.txt
```

### 9. Download Pretrained Model

```bash
mkdir -p pretrained_weights
curl -L "https://github.com/vztu/COVER/raw/release/Model/COVER.pth" \
     -o pretrained_weights/COVER.pth
```

### 10. Verify Model File

```bash
ls -lh pretrained_weights/COVER.pth
# Should show ~250MB file
```

## Troubleshooting

### Issue: "No module named 'cover'"

**Solution**: Ensure virtual environment is activated and COVER is installed:

```bash
source venv/bin/activate
cd cover_repo
pip install -e .
cd ..
```

### Issue: "decord not found"

**Solution**: This is expected on ARM64 systems. The script uses PyAV as a fallback. Verify `av` is installed:

```bash
pip list | grep av
```

### Issue: "COVER.pth not found"

**Solution**: Re-download the model:

```bash
curl -L "https://github.com/vztu/COVER/raw/release/Model/COVER.pth" \
     -o pretrained_weights/COVER.pth
```

### Issue: "No MP4 files found"

**Solution**: Ensure videos exist in the correct location:

```bash
ls -l ../source_videos/*.mp4
```

### Issue: Out of memory

**Solution**: Process videos one at a time, or use a machine with more RAM. The script already processes sequentially to minimize memory usage.

### Issue: CUDA errors

**Solution**: The script automatically falls back to CPU if CUDA is unavailable. This is expected behavior on systems without NVIDIA GPUs (like macOS).

## System-Specific Notes

### macOS ARM64 (Apple Silicon)

- **decord** library is not available for ARM64
- **PyAV** is used as a drop-in replacement
- This is handled automatically by `assess_videos.py`
- No additional configuration needed

### Linux

- Should work out of the box
- If decord is available, the mock will still work
- Consider using native decord for potentially better performance

### Windows

- Not officially tested
- Virtual environment activation: `venv\Scripts\activate`
- Path separators use backslash: `\`
- PyAV should work on Windows as well

## Performance Notes

- **CPU inference**: ~5-10 seconds per video
- **GPU inference**: ~1-2 seconds per video (with CUDA)
- **Memory usage**: ~2-4GB per video
- **Model loading**: ~3-5 seconds (one-time)

## Verifying Successful Installation

Run this command to test everything:

```bash
source venv/bin/activate
python -c "
import torch
import sys
sys.path.insert(0, 'cover_repo')
from cover.models import COVER
print('✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

Expected output:
```
✓ All imports successful
✓ PyTorch version: 1.13.1
✓ CUDA available: False  # or True if GPU is available
```

## Next Steps

After successful installation:

1. Read `README.md` for usage examples
2. Review `IMPLEMENTATION_NOTES.md` for technical details
3. Run `python assess_videos.py` to process videos
4. Check terminal output for quality scores

## Getting Help

If you encounter issues not covered here:

1. Check `IMPLEMENTATION_NOTES.md` for technical details
2. Review the official COVER repository: https://github.com/taco-group/COVER
3. Ensure all prerequisites are met
4. Verify file paths and permissions

## Updating

To update COVER to the latest version:

```bash
source venv/bin/activate
cd cover_repo
git pull
pip install -e . --upgrade
cd ..
```

Note: This may require updating the pretrained model as well.
