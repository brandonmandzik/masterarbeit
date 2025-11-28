# Data Directory

This directory contains video files used for subjective quality assessment testing following ITU-T P.910 and P.1401 standards.

## Directory Structure

### `/source_videos`
Contains reference source videos for quality evaluation.

**Purpose:** These are the original, high-quality videos that will be used as the basis for evaluation in the ITU-T P.910 and P.1401 Framework.

**Usage:** Place your source videos in this directory before running quality assessments.

### `/tests_videos`
Contains test videos for verification and sanity checks.

**Purpose:** Use these videos to verify your qualifier implementation and perform sanity checks before running full evaluations.

**Current contents:**
- `test_real.mp4` - Real-world test sample
- `mov_circle.mp4` - Moving circle pattern
- `mov_circle+noise.mp4` - Moving circle with noise
- `static_scale.mp4` - Static scale pattern
- `static_solidgrey.mp4` - Solid grey static pattern

**Usage:** Reference these test videos during development and validation phases.