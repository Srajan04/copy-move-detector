"""
Batch Processor Module

This module provides batch processing capabilities for evaluating the copy-move
forgery detection system on datasets like MICC-F220.
"""

import os
import glob
import json
import csv
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from .forgery_detector import CopyMoveForgeryDetector
from .visualizer import Visualizer


class BatchProcessor:
    """
    Batch processor for copy-move forgery detection evaluation.

    Provides functionality to:
    - Process entire datasets
    - Generate performance reports
    - Compute evaluation metrics
    - Create summary visualizations
    """

    def __init__(self, detector: Optional[CopyMoveForgeryDetector] = None):
        """
        Initialize batch processor.

        Args:
            detector: Optional pre-configured detector instance
        """
        self.detector = detector if detector else CopyMoveForgeryDetector()
        self.visualizer = Visualizer()
        self.results = []

    def process_dataset(
        self,
        dataset_path: str,
        image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    ) -> List[Dict]:
        """
        Process an entire dataset of images.

        Args:
            dataset_path: Path to dataset directory
            image_extensions: List of valid image extensions

        Returns:
            List of detection results
        """
        print(f"Processing dataset: {dataset_path}")

        # Find all image files
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(dataset_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(dataset_path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))

        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        print(f"Found {len(image_files)} images")

        results = []
        for i, image_path in enumerate(image_files):
            print(
                f"Processing {i + 1}/{len(image_files)}: {os.path.basename(image_path)}"
            )

            try:
                result = self.detector.detect_forgery(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(
                    {"image_path": image_path, "is_forgery": False, "error": str(e)}
                )

        self.results = results
        return results

    def evaluate_performance(
        self, results: List[Dict], ground_truth: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate detection performance with metrics.

        Args:
            results: List of detection results
            ground_truth: Optional ground truth annotations

        Returns:
            Performance metrics dictionary
        """
        if ground_truth is None:
            # Use filename heuristics for MICC-F220 dataset
            ground_truth = self._infer_ground_truth_micc(results)

        # Calculate confusion matrix
        tp = fp = tn = fn = 0

        for result in results:
            image_name = os.path.basename(result["image_path"])
            is_predicted_forgery = result["is_forgery"]
            is_actual_forgery = ground_truth.get(image_name, False)

            if is_predicted_forgery and is_actual_forgery:
                tp += 1
            elif is_predicted_forgery and not is_actual_forgery:
                fp += 1
            elif not is_predicted_forgery and not is_actual_forgery:
                tn += 1
            else:  # not predicted but actual forgery
                fn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        metrics = {
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "total_images": len(results),
            "forgeries_detected": sum(1 for r in results if r["is_forgery"]),
            "actual_forgeries": sum(ground_truth.values()),
        }

        return metrics

    def _infer_ground_truth_micc(self, results: List[Dict]) -> Dict:
        """
        Infer ground truth for MICC-F220 dataset based on filename patterns.

        Args:
            results: Detection results

        Returns:
            Ground truth dictionary
        """
        ground_truth = {}

        for result in results:
            filename = os.path.basename(result["image_path"])

            # MICC-F220 naming convention:
            # - Original images: CRW_xxxx_scale.jpg, DSC_xxxx_scale.jpg
            # - Tampered images: contain 'tamp' in filename

            is_forgery = "tamp" in filename.lower()
            ground_truth[filename] = is_forgery

        return ground_truth

    def generate_report(
        self, results: List[Dict], output_dir: str, include_visualizations: bool = True
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Detection results
            output_dir: Output directory for report
            include_visualizations: Whether to generate visualization images

        Returns:
            Path to generated report file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Evaluate performance
        metrics = self.evaluate_performance(results)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        # Generate report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "total_images": int(len(results)),
                "processing_time": float(sum(r.get("processing_time", 0) for r in results)),
                "average_processing_time": float(np.mean(
                    [r.get("processing_time", 0) for r in results]
                )),
            },
            "detector_parameters": self.detector.parameters,
            "performance_metrics": convert_to_native(metrics),
            "detailed_results": [],
        }

        # Add detailed results
        for result in results:
            if "error" not in result:
                detail = {
                    "image_path": result["image_path"],
                    "filename": os.path.basename(result["image_path"]),
                    "is_forgery": bool(result["is_forgery"]),
                    "num_keypoints": int(result["num_keypoints"]),
                    "num_matches": int(result["num_matches"]),
                    "num_inliers": int(result["num_inliers"]),
                    "inlier_ratio": float(result["inlier_ratio"]),
                    "transformation_type": str(result["transformation_type"]),
                    "processing_time": float(result["processing_time"]),
                }
                report_data["detailed_results"].append(detail)

        # Save JSON report
        json_path = os.path.join(output_dir, "detection_report.json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)

        # Save CSV summary
        csv_path = os.path.join(output_dir, "detection_summary.csv")
        self._save_csv_summary(report_data["detailed_results"], csv_path)

        # Generate visualizations
        if include_visualizations:
            self._generate_visualization_report(results, metrics, output_dir)

        # Generate text report
        txt_path = os.path.join(output_dir, "detection_report.txt")
        self._generate_text_report(report_data, txt_path)

        print(f"Report generated in: {output_dir}")
        return txt_path

    def _save_csv_summary(self, detailed_results: List[Dict], csv_path: str):
        """Save detailed results as CSV."""
        if not detailed_results:
            return

        fieldnames = detailed_results[0].keys()
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)

    def _generate_visualization_report(
        self, results: List[Dict], metrics: Dict, output_dir: str
    ):
        """Generate visualization plots for the report."""
        # Statistics plot
        stats_fig = self.visualizer.create_statistics_plot(results)
        stats_path = os.path.join(output_dir, "statistics.png")
        stats_fig.savefig(stats_path, dpi=150, bbox_inches="tight")
        plt.close(stats_fig)

        # Performance metrics plot
        self._create_performance_plot(metrics, output_dir)

        # Sample detection visualizations
        self._create_sample_visualizations(results, output_dir)

    def _create_performance_plot(self, metrics: Dict, output_dir: str):
        """Create performance metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Confusion matrix
        cm = metrics["confusion_matrix"]
        cm_matrix = np.array(
            [
                [cm["true_positives"], cm["false_negatives"]],
                [cm["false_positives"], cm["true_negatives"]],
            ]
        )

        im = axes[0, 0].imshow(cm_matrix, cmap="Blues")
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_xticklabels(["Forgery", "Authentic"])
        axes[0, 0].set_yticklabels(["Forgery", "Authentic"])

        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[0, 0].text(j, i, cm_matrix[i, j], ha="center", va="center")

        # Metrics bar chart
        metric_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
        metric_values = [
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["accuracy"],
        ]

        bars = axes[0, 1].bar(
            metric_names,
            metric_values,
            color=["skyblue", "lightgreen", "orange", "pink"],
        )
        axes[0, 1].set_title("Performance Metrics")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # Detection summary pie chart
        labels = [
            "True Positives",
            "False Positives",
            "True Negatives",
            "False Negatives",
        ]
        sizes = [
            cm["true_positives"],
            cm["false_positives"],
            cm["true_negatives"],
            cm["false_negatives"],
        ]
        colors = ["lightgreen", "lightcoral", "lightblue", "lightyellow"]

        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
        axes[1, 0].set_title("Detection Results Distribution")

        # Summary text
        summary_text = f"""Total Images: {metrics["total_images"]}
Actual Forgeries: {metrics["actual_forgeries"]}
Detected Forgeries: {metrics["forgeries_detected"]}

Precision: {metrics["precision"]:.3f}
Recall: {metrics["recall"]:.3f}
F1-Score: {metrics["f1_score"]:.3f}
Accuracy: {metrics["accuracy"]:.3f}"""

        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Performance Summary")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "performance_metrics.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _create_sample_visualizations(
        self, results: List[Dict], output_dir: str, max_samples: int = 5
    ):
        """Create sample detection visualizations."""
        vis_dir = os.path.join(output_dir, "sample_detections")
        os.makedirs(vis_dir, exist_ok=True)

        # Select representative samples
        forgery_results = [r for r in results if r["is_forgery"] and "error" not in r]
        authentic_results = [
            r for r in results if not r["is_forgery"] and "error" not in r
        ]

        # Save forgery examples
        for i, result in enumerate(forgery_results[:max_samples]):
            if "original_image" in result:
                vis_image = self.detector.visualize_detection(result)
                filename = (
                    f"forgery_example_{i + 1}_{os.path.basename(result['image_path'])}"
                )
                save_path = os.path.join(vis_dir, filename)
                import cv2

                cv2.imwrite(save_path, vis_image)

    def _generate_text_report(self, report_data: Dict, txt_path: str):
        """Generate human-readable text report."""
        with open(txt_path, "w") as f:
            f.write("COPY-MOVE FORGERY DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Report Generated: {report_data['timestamp']}\n\n")

            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            dataset_info = report_data["dataset_info"]
            f.write(f"Total Images: {dataset_info['total_images']}\n")
            f.write(
                f"Total Processing Time: {dataset_info['processing_time']:.2f} seconds\n"
            )
            f.write(
                f"Average Processing Time: {dataset_info['average_processing_time']:.2f} seconds/image\n\n"
            )

            # Detection parameters
            f.write("DETECTION PARAMETERS\n")
            f.write("-" * 20 + "\n")
            params = report_data["detector_parameters"]
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            metrics = report_data["performance_metrics"]
            cm = metrics["confusion_matrix"]

            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives:  {cm['true_positives']}\n")
            f.write(f"  False Positives: {cm['false_positives']}\n")
            f.write(f"  True Negatives:  {cm['true_negatives']}\n")
            f.write(f"  False Negatives: {cm['false_negatives']}\n\n")

            f.write(f"Precision: {metrics['precision']:.3f}\n")
            f.write(f"Recall:    {metrics['recall']:.3f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.3f}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.3f}\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Images Processed: {metrics['total_images']}\n")
            f.write(f"Actual Forgeries: {metrics['actual_forgeries']}\n")
            f.write(f"Detected Forgeries: {metrics['forgeries_detected']}\n")
            f.write(
                f"Detection Rate: {metrics['forgeries_detected'] / metrics['total_images'] * 100:.1f}%\n"
            )

    def compare_parameters(self, parameter_sets: List[Dict], dataset_path: str) -> Dict:
        """
        Compare different parameter configurations.

        Args:
            parameter_sets: List of parameter dictionaries to test
            dataset_path: Path to test dataset

        Returns:
            Comparison results
        """
        comparison_results = []

        for i, params in enumerate(parameter_sets):
            print(f"Testing parameter set {i + 1}/{len(parameter_sets)}")

            # Create detector with current parameters
            detector = CopyMoveForgeryDetector(**params)
            processor = BatchProcessor(detector)

            # Process dataset
            results = processor.process_dataset(dataset_path)
            metrics = processor.evaluate_performance(results)

            comparison_results.append(
                {"parameters": params, "metrics": metrics, "results": results}
            )

        return {
            "parameter_sets": parameter_sets,
            "comparison_results": comparison_results,
            "best_f1": max(comparison_results, key=lambda x: x["metrics"]["f1_score"]),
        }
