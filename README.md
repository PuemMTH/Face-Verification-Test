# Face Verification Test

A multi-threaded face verification system that processes images to check various face quality metrics.

## Features

- **Multi-threading**: Process multiple images concurrently for faster execution
- **Progress tracking**: Real-time progress bars for both folders and individual images
- **Comprehensive face analysis**: Checks face size, eye status, lighting, blur, head pose, and head completeness
- **Detailed timing**: Tracks processing time for each analysis function
- **CSV output**: Generates detailed reports in CSV format

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to process all folders in the `test` directory:

```bash
python run.py
```

### Silent Mode

Run without verbose output for testing:

```bash
python run_silent.py
```

## Configuration

Edit `config.yml` to adjust thresholds and parameters:

```yaml
threshold:
  face_size: 150
  blur: 90
  dark_threshold: 35
  bright_threshold: 200
  diff_threshold: 20
  margin: 0.1
  head_fully_th: 10
  EAR_THRESHOLD: 0.37
  left_th: -0.3
  right_th: 0.3
  down_th: -10
  up_th: 15
  til_left_th: -0.10
  til_right_th: 0.10
```

## Multi-threading Configuration

The system uses `ThreadPoolExecutor` with configurable worker count:

- Default: 4 workers
- Adjustable via `max_workers` parameter in `process_images()`
- Each worker processes one image at a time
- Progress bars show real-time completion status

## Output Files

For each processed folder, the following files are generated in `output/{folder_name}/`:

1. **results.csv**: Main analysis results for each image
2. **timing_per_image.csv**: Processing time for each function per image
3. **summary.csv**: Total processing time summary by function

## Analysis Functions

1. **Face Size Check**: Ensures face is large enough for analysis
2. **Eye Status Check**: Detects if eyes are open/closed
3. **Lighting Analysis**: Checks for proper illumination
4. **Blur Detection**: Identifies image blur
5. **Head Pose Analysis**: Detects head orientation
6. **Head Completeness**: Ensures full head is visible

## Performance

- Multi-threading significantly improves processing speed
- Progress tracking provides real-time feedback
- Error handling ensures robust processing
- Memory efficient processing of large image sets

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- PyYAML
- tqdm (for progress bars)

## Directory Structure

```
Face-Verification-Test/
├── run.py              # Main script with multi-threading
├── run_silent.py       # Silent version for testing
├── config.yml          # Configuration file
├── requirements.txt    # Dependencies
├── test/              # Input folders
│   ├── folder1/
│   ├── folder2/
│   └── ...
├── output/            # Results output
│   ├── folder1/
│   ├── folder2/
│   └── ...
└── func/              # Analysis functions
    ├── check_eye.py
    ├── check_face_blur.py
    ├── check_face_size.py
    ├── check_head_fully.py
    ├── check_head_pose.py
    ├── check_light_pollution.py
    └── get_landmarks.py
``` 