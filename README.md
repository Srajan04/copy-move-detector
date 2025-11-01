# Visual Forensics and Copy-Move Forgery Detection

This project implements a comprehensive copy-move forgery detection system using SIFT (Scale-Invariant Feature Transform) features. The system can detect and localize regions in digital images that have been duplicated from one part to another, which is a common type of image tampering.

## Features

- **SIFT-based Feature Detection**: Uses robust SIFT keypoints and descriptors
- **Lowe's Ratio Test**: Filters matches using the proven ratio test
- **RANSAC Geometric Verification**: Robust estimation of geometric transformations
- **Forgery Localization**: Visual highlighting of source and target regions
- **Batch Processing**: Process multiple images efficiently
- **Comprehensive Evaluation**: Precision, recall, F1-score metrics

## Detection Pipeline

The system follows a 6-stage pipeline with early-exit mechanisms for efficiency:

```
INPUT IMAGE
    ↓
[Stage 1: Preprocessing]
    ImagePreprocessor → Grayscale + Optional Blur
    ↓
[Stage 2: Feature Extraction]
    SIFTExtractor → Keypoints & 128-D Descriptors
    ↓ (exit if < 4 keypoints)
[Stage 3: Matching]
    KeypointMatcher → FLANN + Lowe's Ratio Test
    ↓ (exit if < 4 matches)
[Stage 4: Geometric Analysis]
    GeometricAnalyzer → RANSAC Homography
    ↓
[Stage 5: Validation]
    7 Validation Checks:
      • Minimum inliers
      • Minimum region area
      • Minimum translation distance
      • Acceptable overlap ratio
      • Valid inlier ratio
      • Significant transformation
      • Reasonable inlier percentage
    ↓
[Stage 6: Output]
    ✓ FORGERY DETECTED / ❌ NO FORGERY
    Result Dictionary + Visualization
```

> **📊 Detailed Flow Diagram**: See [`FLOW.md`](FLOW.md) for an interactive Mermaid flowchart showing all decision points and data flow.

## Dataset

The project uses the MICC-F220 dataset, which contains:
- Original images
- Copy-move tampered versions with various transformations
- Ground truth annotations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Image Detection
```python
from src.forgery_detector import CopyMoveForgeryDetector

detector = CopyMoveForgeryDetector()
result = detector.detect_forgery("path/to/image.jpg")

if result['is_forgery']:
    print(f"Forgery detected with {result['num_inliers']} inlier keypoints")
    print(f"Transformation type: {result['transformation_type']}")
    detector.visualize_detection(result)
```

### Batch Processing
```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor()
results = processor.process_dataset("MICC-F220/")
metrics = processor.evaluate_performance(results)
processor.generate_report(results, "results/experiment_01")
```

## Project Structure

```
updated-project/
├── src/
│   ├── __init__.py
│   ├── image_preprocessor.py    # Stage 1: Image preprocessing (grayscale, denoising)
│   ├── sift_extractor.py        # Stage 2: SIFT feature detection and description
│   ├── keypoint_matcher.py      # Stage 3: Keypoint matching with Lowe's ratio test
│   ├── geometric_analyzer.py    # Stage 4: RANSAC-based geometric verification
│   ├── forgery_detector.py      # Pipeline orchestrator
│   ├── visualizer.py            # Visualization utilities
│   └── batch_processor.py       # Batch processing and evaluation
├── notebooks/
│   ├── demo.ipynb               # Interactive demonstration
│   └── evaluation.ipynb         # Performance evaluation
├── MICC-F220/                   # Dataset directory
├── results/                     # Output directory
├── FLOW.md                      # Detailed pipeline flowchart
├── requirements.txt
└── README.md
```

## Algorithm Pipeline Details

### Stage 1: Image Preprocessing
- Convert to grayscale (required for SIFT)
- Optional Gaussian blur for noise reduction
- **Module**: [`image_preprocessor.py`](src/image_preprocessor.py)

### Stage 2: SIFT Feature Extraction
- Detect scale-invariant keypoints across multiple octaves
- Compute 128-dimensional descriptors
- **Early Exit**: If keypoints < 4, return "insufficient keypoints"
- **Module**: [`sift_extractor.py`](src/sift_extractor.py)

### Stage 3: Keypoint Matching
- FLANN-based k-NN matching (k=2)
- Lowe's ratio test filtering (default threshold: 0.8)
- Self-match removal (distance == 0)
- **Early Exit**: If matches < 4, return "insufficient matches"
- **Module**: [`keypoint_matcher.py`](src/keypoint_matcher.py)

### Stage 4: Geometric Analysis
- RANSAC homography estimation
- Inlier detection via reprojection error
- Transformation type classification (translation, rotation, scaling, etc.)
- Region extraction (source & destination bounding boxes)
- **Module**: [`geometric_analyzer.py`](src/geometric_analyzer.py)

### Stage 5: Validation (7 Checks)
1. **Inlier Count**: ≥ `min_cluster_pts` (default: 10)
2. **Region Area**: ≥ `min_region_area` (default: 100 px²)
3. **Translation**: ≥ `min_translation` (default: 1.0 px)
4. **Overlap Ratio**: ≤ `max_overlap_ratio` (default: 0.95)
5. **Inlier Ratio**: ≥ `min_inlier_ratio` (default: 0.01)
6. **Transformation**: Not identity matrix
7. **Inlier Percentage**: ≥ 1% of total matches

### Stage 6: Output
Returns a comprehensive result dictionary containing:
- Detection status (`is_forgery`, `soft_detection`, `validation_passed`)
- Keypoints, descriptors, matches, inliers
- Homography matrix and transformation type
- Region information (bounding boxes, overlap ratio)
- Original and preprocessed images
- Processing time and statistics

## Parameters

Key parameters that can be tuned via `CopyMoveForgeryDetector.update_parameters()`:

| Parameter | Default | Description | Sensitivity |
|-----------|---------|-------------|-------------|
| `ratio_threshold` | 0.8 | Lowe's ratio test threshold | Lower = stricter matches |
| `min_cluster_pts` | 10 | Minimum inliers for detection | Primary detection lever |
| `reproj_threshold` | 3.0 | RANSAC reprojection threshold (pixels) | Geometric tolerance |
| `max_iterations` | 1000 | RANSAC iterations | Accuracy vs. speed trade-off |
| `min_inlier_ratio` | 0.01 | Minimum inlier-to-match ratio | Prevents weak detections |
| `min_translation` | 1.0 | Minimum translation distance (pixels) | Avoids self-overlap |
| `min_region_area` | 100.0 | Minimum forged region area (px²) | Filters tiny artifacts |
| `max_overlap_ratio` | 0.95 | Maximum source-destination overlap | Prevents full overlaps |

**Tuning Tips**:
- Increase `ratio_threshold` (0.8 → 0.85) to detect more subtle tampering (more false positives)
- Decrease `min_cluster_pts` (10 → 8) to detect smaller copied regions
- Increase `reproj_threshold` (3.0 → 5.0) for images with geometric distortion

## Performance

The system achieves high accuracy on the MICC-F220 dataset with proper parameter tuning. Results include:
- Detection accuracy
- Precision, recall, F1-score
- False positive/negative rates
- Processing time per image (~2-5 seconds on typical hardware)

## Interactive Exploration

Use the Jupyter notebooks for hands-on experimentation:

1. **[`notebooks/demo.ipynb`](notebooks/demo.ipynb)**: Step-by-step pipeline walkthrough
   - Visualize each processing stage
   - Tune parameters interactively
   - Test on individual images

2. **[`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb)**: Dataset-wide evaluation
   - Batch processing
   - Performance metrics
   - Statistical analysis

## References

- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints" (2004)
- MICC-F220 Dataset: Media Integration and Communication Center, University of Florence
- RANSAC: Fischler & Bolles "Random Sample Consensus" (1981)
- Bay, H., et al. "SURF: Speeded Up Robust Features" (2006)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The software is provided "as is", without warranty of any kind. Feel free to use, modify, and distribute this code for educational, research, or commercial purposes.
