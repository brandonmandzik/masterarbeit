# Install dependencies as needed:
# pip install kagglehub
import kagglehub
import os

# Download the UCF101 video dataset
# This returns the path to the downloaded dataset
dataset_path = kagglehub.dataset_download("abdallahwagih/ucf101-videos")

print(f"Dataset downloaded to: {dataset_path}")

# List the contents of the dataset
print("\nDataset contents:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files in each directory
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")

# Example: List all video files
video_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.avi', '.mp4', '.mov')):
            video_files.append(os.path.join(root, file))

print(f"\nTotal video files found: {len(video_files)}")
if video_files:
    print(f"First video: {video_files[0]}")