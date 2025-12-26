#!/usr/bin/env python3
"""
Comprehensive Dataset Quality Analysis for Clothing Recognition Model
Analyzes image quality, class balance, duplicates, and potential issues.
"""

import os
import sys
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path("/home/user/Clothes_recognition_Model/dataset")
OUTPUT_DIR = Path("/home/user/Clothes_recognition_Model/analysis_output")
SPLITS = ["train", "val", "Test"]
CLASSES = ["Romper", "Tee", "Tank", "Sweatpants", "Cardigan", "Jacket",
           "Jumpsuit", "Blouse", "Skirt", "Jeans", "Sweater", "Blazer",
           "Dress", "Top", "Shorts"]

# Thresholds
MIN_IMAGE_SIZE = 100  # pixels
MAX_IMAGE_SIZE = 2000  # pixels
MIN_SAMPLES_WARNING = 50
DUPLICATE_HASH_SIMILARITY = 0.95

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)


class DatasetAnalyzer:
    def __init__(self, dataset_path, splits, classes):
        self.dataset_path = Path(dataset_path)
        self.splits = splits
        self.classes = classes
        self.results = {
            "structure": {},
            "image_quality": {},
            "duplicates": {},
            "issues": {},
            "recommendations": [],
            "health_score": "Unknown"
        }
        self.all_images_data = []
        self.image_hashes = defaultdict(list)  # hash -> list of (path, split, class)

    def analyze_all(self):
        """Run all analyses"""
        print("=" * 70)
        print("DATASET QUALITY ANALYSIS FOR CLOTHING RECOGNITION")
        print("=" * 70)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # 1. Structure and Balance
        print("\n[1/7] Analyzing dataset structure and balance...")
        self.analyze_structure()

        # 2. Image Quality
        print("\n[2/7] Checking image quality (this may take a while)...")
        self.analyze_image_quality()

        # 3. Duplicate Detection
        print("\n[3/7] Detecting duplicates...")
        self.detect_duplicates()

        # 4. Potential Issues
        print("\n[4/7] Detecting potential issues...")
        self.detect_issues()

        # 5. Class Confusion Analysis
        print("\n[5/7] Analyzing class confusion potential...")
        self.analyze_class_confusion()

        # 6. Generate Visualizations
        print("\n[6/7] Generating visualizations...")
        self.generate_visualizations()

        # 7. Calculate Health Score and Generate Report
        print("\n[7/7] Generating final report...")
        self.calculate_health_score()
        self.generate_report()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

    def analyze_structure(self):
        """Analyze dataset structure and class balance"""
        structure = {
            "total_images": 0,
            "splits": {},
            "class_distribution": {},
            "imbalance_ratio": 0,
            "split_ratios": {},
            "low_sample_classes": []
        }

        all_counts = []

        for split in self.splits:
            split_path = self.dataset_path / split
            split_data = {"total": 0, "classes": {}}

            for cls in self.classes:
                cls_path = split_path / cls
                if cls_path.exists():
                    images = list(cls_path.glob("*"))
                    images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']]
                    count = len(images)
                    split_data["classes"][cls] = count
                    split_data["total"] += count

                    # Track for class distribution
                    if cls not in structure["class_distribution"]:
                        structure["class_distribution"][cls] = {"total": 0, "splits": {}}
                    structure["class_distribution"][cls]["total"] += count
                    structure["class_distribution"][cls]["splits"][split] = count

                    all_counts.append(count)
                else:
                    split_data["classes"][cls] = 0

            structure["splits"][split] = split_data
            structure["total_images"] += split_data["total"]

        # Calculate imbalance ratio
        if all_counts:
            min_count = min(c for c in all_counts if c > 0) if any(c > 0 for c in all_counts) else 1
            max_count = max(all_counts) if all_counts else 1
            structure["imbalance_ratio"] = max_count / min_count if min_count > 0 else float('inf')

        # Calculate split ratios
        total = structure["total_images"]
        if total > 0:
            for split in self.splits:
                structure["split_ratios"][split] = structure["splits"][split]["total"] / total

        # Find low sample classes
        for cls, data in structure["class_distribution"].items():
            for split, count in data["splits"].items():
                if count < MIN_SAMPLES_WARNING:
                    structure["low_sample_classes"].append({
                        "class": cls,
                        "split": split,
                        "count": count
                    })

        self.results["structure"] = structure

        # Print summary
        print(f"\n  Total images: {structure['total_images']}")
        print(f"  Split distribution:")
        for split in self.splits:
            count = structure["splits"][split]["total"]
            ratio = structure["split_ratios"].get(split, 0)
            print(f"    - {split}: {count} images ({ratio*100:.1f}%)")
        print(f"  Class imbalance ratio: {structure['imbalance_ratio']:.2f}")
        if structure["low_sample_classes"]:
            print(f"  WARNING: {len(structure['low_sample_classes'])} class/split combinations have < {MIN_SAMPLES_WARNING} samples")

    def analyze_image_quality(self):
        """Analyze image quality metrics"""
        quality = {
            "resolutions": {"widths": [], "heights": [], "aspects": []},
            "formats": defaultdict(int),
            "color_modes": defaultdict(int),
            "too_small": [],
            "too_large": [],
            "corrupted": [],
            "unusual_aspect": [],
            "statistics": {}
        }

        # Iterate through all images
        total_images = self.results["structure"]["total_images"]

        for split in self.splits:
            split_path = self.dataset_path / split
            for cls in self.classes:
                cls_path = split_path / cls
                if not cls_path.exists():
                    continue

                images = list(cls_path.glob("*"))
                for img_path in tqdm(images, desc=f"  {split}/{cls}", leave=False):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                        continue

                    img_data = {
                        "path": str(img_path),
                        "split": split,
                        "class": cls,
                        "filename": img_path.name
                    }

                    try:
                        with Image.open(img_path) as img:
                            # Basic properties
                            width, height = img.size
                            img_data["width"] = width
                            img_data["height"] = height
                            img_data["format"] = img.format or img_path.suffix.upper().replace(".", "")
                            img_data["mode"] = img.mode

                            quality["resolutions"]["widths"].append(width)
                            quality["resolutions"]["heights"].append(height)

                            aspect = width / height if height > 0 else 0
                            quality["resolutions"]["aspects"].append(aspect)
                            img_data["aspect_ratio"] = aspect

                            quality["formats"][img_data["format"]] += 1
                            quality["color_modes"][img.mode] += 1

                            # Size checks
                            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                                quality["too_small"].append(img_data.copy())
                            if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
                                quality["too_large"].append(img_data.copy())

                            # Unusual aspect ratio (very wide or very tall)
                            if aspect < 0.3 or aspect > 3.0:
                                quality["unusual_aspect"].append(img_data.copy())

                            # Calculate hash for duplicate detection
                            img_resized = img.convert('RGB').resize((8, 8), Image.Resampling.LANCZOS)
                            pixels = list(img_resized.getdata())
                            avg = sum(sum(p) for p in pixels) / (len(pixels) * 3)
                            bits = ''.join('1' if sum(p)/3 > avg else '0' for p in pixels)
                            img_hash = hashlib.md5(bits.encode()).hexdigest()
                            img_data["phash"] = img_hash

                            # Also store exact hash
                            img.seek(0) if hasattr(img, 'seek') else None
                            exact_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                            img_data["exact_hash"] = exact_hash

                            self.image_hashes[exact_hash].append((str(img_path), split, cls))

                            # Get image statistics for issue detection
                            if img.mode in ['RGB', 'RGBA']:
                                stat = ImageStat.Stat(img)
                                img_data["mean"] = stat.mean[:3]
                                img_data["stddev"] = stat.stddev[:3]
                            elif img.mode == 'L':
                                stat = ImageStat.Stat(img)
                                img_data["mean"] = [stat.mean[0]] * 3
                                img_data["stddev"] = [stat.stddev[0]] * 3
                            else:
                                img_rgb = img.convert('RGB')
                                stat = ImageStat.Stat(img_rgb)
                                img_data["mean"] = stat.mean[:3]
                                img_data["stddev"] = stat.stddev[:3]

                            img_data["corrupted"] = False

                    except Exception as e:
                        img_data["corrupted"] = True
                        img_data["error"] = str(e)
                        quality["corrupted"].append(img_data.copy())

                    self.all_images_data.append(img_data)

        # Calculate statistics
        if quality["resolutions"]["widths"]:
            widths = np.array(quality["resolutions"]["widths"])
            heights = np.array(quality["resolutions"]["heights"])
            aspects = np.array(quality["resolutions"]["aspects"])

            quality["statistics"] = {
                "width": {
                    "min": int(np.min(widths)),
                    "max": int(np.max(widths)),
                    "mean": float(np.mean(widths)),
                    "std": float(np.std(widths))
                },
                "height": {
                    "min": int(np.min(heights)),
                    "max": int(np.max(heights)),
                    "mean": float(np.mean(heights)),
                    "std": float(np.std(heights))
                },
                "aspect_ratio": {
                    "min": float(np.min(aspects)),
                    "max": float(np.max(aspects)),
                    "mean": float(np.mean(aspects)),
                    "std": float(np.std(aspects))
                }
            }

        self.results["image_quality"] = quality

        # Print summary
        print(f"\n  Resolution statistics:")
        if quality["statistics"]:
            stats = quality["statistics"]
            print(f"    Width:  min={stats['width']['min']}, max={stats['width']['max']}, avg={stats['width']['mean']:.1f}, std={stats['width']['std']:.1f}")
            print(f"    Height: min={stats['height']['min']}, max={stats['height']['max']}, avg={stats['height']['mean']:.1f}, std={stats['height']['std']:.1f}")
        print(f"  Image formats: {dict(quality['formats'])}")
        print(f"  Color modes: {dict(quality['color_modes'])}")
        print(f"  Too small (<{MIN_IMAGE_SIZE}px): {len(quality['too_small'])}")
        print(f"  Too large (>{MAX_IMAGE_SIZE}px): {len(quality['too_large'])}")
        print(f"  Corrupted: {len(quality['corrupted'])}")
        print(f"  Unusual aspect ratio: {len(quality['unusual_aspect'])}")

    def detect_duplicates(self):
        """Detect duplicate images"""
        duplicates = {
            "exact_duplicates": [],
            "cross_split_duplicates": [],
            "within_class_duplicates": [],
            "total_exact_duplicates": 0,
            "total_cross_split": 0
        }

        # Find exact duplicates
        for hash_val, locations in self.image_hashes.items():
            if len(locations) > 1:
                duplicates["exact_duplicates"].append({
                    "hash": hash_val,
                    "count": len(locations),
                    "locations": locations
                })
                duplicates["total_exact_duplicates"] += len(locations) - 1

                # Check for cross-split duplicates (data leakage!)
                splits_involved = set(loc[1] for loc in locations)
                if len(splits_involved) > 1:
                    duplicates["cross_split_duplicates"].append({
                        "hash": hash_val,
                        "splits": list(splits_involved),
                        "locations": locations
                    })
                    duplicates["total_cross_split"] += 1

                # Check for within-class duplicates
                classes_involved = set(loc[2] for loc in locations)
                if len(classes_involved) == 1:
                    duplicates["within_class_duplicates"].append({
                        "hash": hash_val,
                        "class": list(classes_involved)[0],
                        "count": len(locations),
                        "locations": locations
                    })

        self.results["duplicates"] = duplicates

        # Print summary
        print(f"\n  Exact duplicates found: {duplicates['total_exact_duplicates']}")
        print(f"  Duplicate groups: {len(duplicates['exact_duplicates'])}")
        print(f"  CRITICAL - Cross-split duplicates (data leakage): {duplicates['total_cross_split']}")
        if duplicates["cross_split_duplicates"]:
            print("  WARNING: Found images that appear in multiple splits!")
            for dup in duplicates["cross_split_duplicates"][:5]:
                print(f"    - Appears in {dup['splits']}: {dup['locations'][0][0]}")

    def detect_issues(self):
        """Detect potential issues with images"""
        issues = {
            "blank_images": [],
            "nearly_blank": [],
            "very_dark": [],
            "very_bright": [],
            "low_contrast": [],
            "mostly_one_color": [],
            "noisy_images": []
        }

        for img_data in tqdm(self.all_images_data, desc="  Checking issues"):
            if img_data.get("corrupted", False):
                continue

            mean = img_data.get("mean", [128, 128, 128])
            stddev = img_data.get("stddev", [50, 50, 50])

            avg_mean = np.mean(mean)
            avg_std = np.mean(stddev)

            # Blank or nearly blank (very low variance)
            if avg_std < 5:
                issues["blank_images"].append(img_data)
            elif avg_std < 15:
                issues["nearly_blank"].append(img_data)

            # Very dark images
            if avg_mean < 30:
                issues["very_dark"].append(img_data)

            # Very bright images
            if avg_mean > 240:
                issues["very_bright"].append(img_data)

            # Low contrast
            if avg_std < 25 and 30 <= avg_mean <= 240:
                issues["low_contrast"].append(img_data)

            # Mostly one color (low std in all channels)
            if all(s < 20 for s in stddev):
                issues["mostly_one_color"].append(img_data)

        self.results["issues"] = issues

        # Print summary
        total_issues = sum(len(v) for v in issues.values())
        print(f"\n  Total potential issues found: {total_issues}")
        print(f"    - Blank images: {len(issues['blank_images'])}")
        print(f"    - Nearly blank: {len(issues['nearly_blank'])}")
        print(f"    - Very dark: {len(issues['very_dark'])}")
        print(f"    - Very bright: {len(issues['very_bright'])}")
        print(f"    - Low contrast: {len(issues['low_contrast'])}")
        print(f"    - Mostly one color: {len(issues['mostly_one_color'])}")

    def analyze_class_confusion(self):
        """Analyze potential class confusion"""
        confusion_analysis = {
            "similar_classes": [
                {"classes": ["Top", "Tee", "Blouse", "Tank"],
                 "reason": "All are upper body garments with similar silhouettes"},
                {"classes": ["Sweater", "Cardigan", "Sweatpants"],
                 "reason": "Similar textures and casual style"},
                {"classes": ["Jeans", "Sweatpants", "Shorts"],
                 "reason": "All are bottom wear, may have similar patterns"},
                {"classes": ["Dress", "Jumpsuit", "Romper"],
                 "reason": "Full-body garments that may look similar"},
                {"classes": ["Jacket", "Blazer", "Cardigan"],
                 "reason": "All are outerwear/layering pieces"}
            ],
            "recommendations": [
                "Consider using attention mechanisms to focus on distinguishing features",
                "Data augmentation should preserve class-specific features",
                "May need more samples for visually similar classes",
                "Consider hierarchical classification (upper/lower/full body first)"
            ]
        }

        self.results["class_confusion"] = confusion_analysis

        print("\n  Potentially confusing class groups identified:")
        for group in confusion_analysis["similar_classes"]:
            print(f"    - {', '.join(group['classes'])}")
            print(f"      Reason: {group['reason']}")

    def generate_visualizations(self):
        """Generate all visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Class distribution bar chart
        self._plot_class_distribution()

        # 2. Split distribution pie chart
        self._plot_split_distribution()

        # 3. Resolution distribution histogram
        self._plot_resolution_distribution()

        # 4. Aspect ratio distribution
        self._plot_aspect_ratio_distribution()

        # 5. Sample images from each class
        self._plot_sample_images()

        # 6. Problematic images
        self._plot_problematic_images()

        print(f"\n  Visualizations saved to: {OUTPUT_DIR}")

    def _plot_class_distribution(self):
        """Plot class distribution across splits"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, split in enumerate(self.splits):
            split_data = self.results["structure"]["splits"][split]["classes"]
            classes = list(split_data.keys())
            counts = list(split_data.values())

            colors = sns.color_palette("husl", len(classes))
            bars = axes[idx].bar(range(len(classes)), counts, color=colors)
            axes[idx].set_xticks(range(len(classes)))
            axes[idx].set_xticklabels(classes, rotation=45, ha='right')
            axes[idx].set_title(f'{split.capitalize()} Split Distribution')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Number of Images')

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{count}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_split_distribution(self):
        """Plot train/val/test split ratios"""
        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [self.results["structure"]["splits"][s]["total"] for s in self.splits]
        labels = [f'{s}\n({n} images)\n{n/sum(sizes)*100:.1f}%' for s, n in zip(self.splits, sizes)]
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90)
        ax.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')

        plt.savefig(OUTPUT_DIR / 'split_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_resolution_distribution(self):
        """Plot resolution distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        widths = self.results["image_quality"]["resolutions"]["widths"]
        heights = self.results["image_quality"]["resolutions"]["heights"]

        # Width histogram
        axes[0].hist(widths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
        axes[0].axvline(MIN_IMAGE_SIZE, color='orange', linestyle=':', label=f'Min threshold: {MIN_IMAGE_SIZE}')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Image Width Distribution')
        axes[0].legend()

        # Height histogram
        axes[1].hist(heights, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
        axes[1].axvline(MIN_IMAGE_SIZE, color='orange', linestyle=':', label=f'Min threshold: {MIN_IMAGE_SIZE}')
        axes[1].set_xlabel('Height (pixels)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Image Height Distribution')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'resolution_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_aspect_ratio_distribution(self):
        """Plot aspect ratio distribution"""
        fig, ax = plt.subplots(figsize=(10, 5))

        aspects = self.results["image_quality"]["resolutions"]["aspects"]

        ax.hist(aspects, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.axvline(1.0, color='green', linestyle='--', label='Square (1:1)')
        ax.axvline(0.75, color='blue', linestyle=':', label='Portrait (3:4)')
        ax.axvline(1.33, color='orange', linestyle=':', label='Landscape (4:3)')
        ax.set_xlabel('Aspect Ratio (Width/Height)')
        ax.set_ylabel('Count')
        ax.set_title('Image Aspect Ratio Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'aspect_ratio_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_sample_images(self):
        """Plot sample images from each class"""
        n_samples = 5
        n_classes = len(self.classes)

        fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 2.5, n_classes * 2.5))

        for cls_idx, cls in enumerate(self.classes):
            # Get images for this class from training set
            cls_images = [img for img in self.all_images_data
                         if img.get("class") == cls and img.get("split") == "train"
                         and not img.get("corrupted", False)]

            # Sample random images
            np.random.seed(42)
            if len(cls_images) > n_samples:
                sampled = np.random.choice(len(cls_images), n_samples, replace=False)
                cls_images = [cls_images[i] for i in sampled]

            for img_idx in range(n_samples):
                ax = axes[cls_idx, img_idx]

                if img_idx < len(cls_images):
                    try:
                        img = Image.open(cls_images[img_idx]["path"])
                        ax.imshow(img)
                        ax.set_title(f'{cls_images[img_idx]["width"]}x{cls_images[img_idx]["height"]}', fontsize=8)
                    except:
                        ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

                ax.axis('off')

                if img_idx == 0:
                    ax.set_ylabel(cls, fontsize=10, rotation=0, ha='right', va='center')

        plt.suptitle('Random Samples from Each Class (Training Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'sample_images.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_problematic_images(self):
        """Plot problematic images found"""
        issues = self.results["issues"]
        quality = self.results["image_quality"]

        # Collect problematic images
        problem_images = []

        # Add too small images
        for img in quality["too_small"][:3]:
            problem_images.append(("Too Small", img))

        # Add corrupted images
        for img in quality["corrupted"][:3]:
            problem_images.append(("Corrupted", img))

        # Add blank/nearly blank
        for img in issues["blank_images"][:2]:
            problem_images.append(("Blank", img))
        for img in issues["nearly_blank"][:2]:
            problem_images.append(("Nearly Blank", img))

        # Add dark/bright
        for img in issues["very_dark"][:2]:
            problem_images.append(("Very Dark", img))
        for img in issues["very_bright"][:2]:
            problem_images.append(("Very Bright", img))

        if not problem_images:
            print("  No problematic images to display")
            return

        n_images = min(len(problem_images), 12)
        cols = 4
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < len(problem_images):
                issue_type, img_data = problem_images[idx]
                try:
                    img = Image.open(img_data["path"])
                    ax.imshow(img)
                    ax.set_title(f'{issue_type}\n{img_data.get("width", "?")}x{img_data.get("height", "?")}', fontsize=8)
                except:
                    ax.text(0.5, 0.5, f'{issue_type}\n(Cannot load)', ha='center', va='center')
            ax.axis('off')

        plt.suptitle('Problematic Images Found', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'problematic_images.png', dpi=150, bbox_inches='tight')
        plt.close()

    def calculate_health_score(self):
        """Calculate overall dataset health score"""
        issues_count = 0
        critical_issues = 0
        warnings = 0

        # Check for critical issues
        if self.results["duplicates"]["total_cross_split"] > 0:
            critical_issues += 1
            self.results["recommendations"].append(
                f"CRITICAL: Remove {self.results['duplicates']['total_cross_split']} cross-split duplicates (data leakage!)"
            )

        if len(self.results["image_quality"]["corrupted"]) > 0:
            critical_issues += 1
            self.results["recommendations"].append(
                f"CRITICAL: Fix or remove {len(self.results['image_quality']['corrupted'])} corrupted images"
            )

        # Check for warnings
        if self.results["structure"]["imbalance_ratio"] > 10:
            warnings += 1
            self.results["recommendations"].append(
                f"WARNING: High class imbalance (ratio: {self.results['structure']['imbalance_ratio']:.1f}). Consider oversampling minority classes or using weighted loss."
            )

        if len(self.results["structure"]["low_sample_classes"]) > 0:
            warnings += 1
            self.results["recommendations"].append(
                f"WARNING: {len(self.results['structure']['low_sample_classes'])} class/split combinations have < {MIN_SAMPLES_WARNING} samples"
            )

        if len(self.results["image_quality"]["too_small"]) > 0:
            warnings += 1
            self.results["recommendations"].append(
                f"WARNING: {len(self.results['image_quality']['too_small'])} images are smaller than {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE} pixels"
            )

        if self.results["duplicates"]["total_exact_duplicates"] > 10:
            warnings += 1
            self.results["recommendations"].append(
                f"WARNING: {self.results['duplicates']['total_exact_duplicates']} duplicate images found (excluding cross-split)"
            )

        total_issues_found = sum(len(v) for v in self.results["issues"].values())
        if total_issues_found > self.results["structure"]["total_images"] * 0.05:
            warnings += 1
            self.results["recommendations"].append(
                f"WARNING: {total_issues_found} potential quality issues found (>{5}% of dataset)"
            )

        # Calculate score
        if critical_issues > 0:
            self.results["health_score"] = "CRITICAL"
        elif warnings >= 3:
            self.results["health_score"] = "WARNING"
        elif warnings >= 1:
            self.results["health_score"] = "FAIR"
        else:
            self.results["health_score"] = "GOOD"

        # Add general recommendations
        self.results["recommendations"].extend([
            "Consider resizing all images to a consistent size (e.g., 224x224 or 256x256)",
            "Apply data augmentation (rotation, flip, color jitter) to increase dataset diversity",
            "Verify that class labels are accurate by manual inspection of edge cases",
            "Consider using pretrained models (transfer learning) given the dataset size"
        ])

    def generate_report(self):
        """Generate the final text report"""
        report_lines = []

        report_lines.append("=" * 70)
        report_lines.append("DATASET QUALITY ANALYSIS REPORT")
        report_lines.append("Clothing Recognition Model Dataset")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)

        # Health Score
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"OVERALL HEALTH SCORE: {self.results['health_score']}")
        report_lines.append("=" * 70)

        # Dataset Structure
        report_lines.append("\n" + "-" * 70)
        report_lines.append("1. DATASET STRUCTURE & BALANCE")
        report_lines.append("-" * 70)

        structure = self.results["structure"]
        report_lines.append(f"\nTotal Images: {structure['total_images']}")
        report_lines.append("\nSplit Distribution:")
        for split in self.splits:
            count = structure["splits"][split]["total"]
            ratio = structure["split_ratios"].get(split, 0)
            report_lines.append(f"  - {split}: {count} images ({ratio*100:.1f}%)")

        report_lines.append(f"\nClass Imbalance Ratio: {structure['imbalance_ratio']:.2f}")

        report_lines.append("\nImages per Class (Train/Val/Test):")
        for cls in self.classes:
            cls_data = structure["class_distribution"].get(cls, {})
            splits = cls_data.get("splits", {})
            train_c = splits.get("train", 0)
            val_c = splits.get("val", 0)
            test_c = splits.get("Test", 0)
            total_c = cls_data.get("total", 0)
            report_lines.append(f"  {cls}: {train_c}/{val_c}/{test_c} (Total: {total_c})")

        if structure["low_sample_classes"]:
            report_lines.append(f"\nLow Sample Warning (< {MIN_SAMPLES_WARNING} images):")
            for item in structure["low_sample_classes"]:
                report_lines.append(f"  - {item['class']} in {item['split']}: {item['count']} images")

        # Image Quality
        report_lines.append("\n" + "-" * 70)
        report_lines.append("2. IMAGE QUALITY ANALYSIS")
        report_lines.append("-" * 70)

        quality = self.results["image_quality"]
        if quality["statistics"]:
            stats = quality["statistics"]
            report_lines.append("\nResolution Statistics:")
            report_lines.append(f"  Width:  min={stats['width']['min']}, max={stats['width']['max']}, mean={stats['width']['mean']:.1f}, std={stats['width']['std']:.1f}")
            report_lines.append(f"  Height: min={stats['height']['min']}, max={stats['height']['max']}, mean={stats['height']['mean']:.1f}, std={stats['height']['std']:.1f}")
            report_lines.append(f"  Aspect: min={stats['aspect_ratio']['min']:.2f}, max={stats['aspect_ratio']['max']:.2f}, mean={stats['aspect_ratio']['mean']:.2f}")

        report_lines.append(f"\nImage Formats: {dict(quality['formats'])}")
        report_lines.append(f"Color Modes: {dict(quality['color_modes'])}")
        report_lines.append(f"\nSize Issues:")
        report_lines.append(f"  - Too small (<{MIN_IMAGE_SIZE}px): {len(quality['too_small'])}")
        report_lines.append(f"  - Too large (>{MAX_IMAGE_SIZE}px): {len(quality['too_large'])}")
        report_lines.append(f"  - Unusual aspect ratio: {len(quality['unusual_aspect'])}")
        report_lines.append(f"  - Corrupted/unreadable: {len(quality['corrupted'])}")

        if quality["corrupted"]:
            report_lines.append("\nCorrupted Images:")
            for img in quality["corrupted"][:10]:
                report_lines.append(f"  - {img['path']}")
                report_lines.append(f"    Error: {img.get('error', 'Unknown')}")

        # Duplicates
        report_lines.append("\n" + "-" * 70)
        report_lines.append("3. DUPLICATE DETECTION")
        report_lines.append("-" * 70)

        dups = self.results["duplicates"]
        report_lines.append(f"\nExact Duplicates: {dups['total_exact_duplicates']}")
        report_lines.append(f"Duplicate Groups: {len(dups['exact_duplicates'])}")
        report_lines.append(f"Cross-Split Duplicates (DATA LEAKAGE!): {dups['total_cross_split']}")

        if dups["cross_split_duplicates"]:
            report_lines.append("\nCross-Split Duplicate Details:")
            for dup in dups["cross_split_duplicates"][:10]:
                report_lines.append(f"  - Found in splits: {dup['splits']}")
                for loc in dup["locations"]:
                    report_lines.append(f"    {loc[0]}")

        # Issues
        report_lines.append("\n" + "-" * 70)
        report_lines.append("4. POTENTIAL ISSUES")
        report_lines.append("-" * 70)

        issues = self.results["issues"]
        report_lines.append(f"\nBlank images: {len(issues['blank_images'])}")
        report_lines.append(f"Nearly blank: {len(issues['nearly_blank'])}")
        report_lines.append(f"Very dark: {len(issues['very_dark'])}")
        report_lines.append(f"Very bright: {len(issues['very_bright'])}")
        report_lines.append(f"Low contrast: {len(issues['low_contrast'])}")
        report_lines.append(f"Mostly one color: {len(issues['mostly_one_color'])}")

        # List problematic files
        all_problem_files = []
        for issue_type, issue_list in issues.items():
            for img in issue_list[:5]:
                all_problem_files.append((issue_type, img.get("path", "Unknown")))

        if all_problem_files:
            report_lines.append("\nSample Problematic Files:")
            for issue_type, path in all_problem_files[:20]:
                report_lines.append(f"  [{issue_type}] {path}")

        # Class Confusion
        report_lines.append("\n" + "-" * 70)
        report_lines.append("5. CLASS CONFUSION ANALYSIS")
        report_lines.append("-" * 70)

        confusion = self.results.get("class_confusion", {})
        if confusion.get("similar_classes"):
            report_lines.append("\nPotentially Confusing Class Groups:")
            for group in confusion["similar_classes"]:
                report_lines.append(f"  - {', '.join(group['classes'])}")
                report_lines.append(f"    Reason: {group['reason']}")

        # Recommendations
        report_lines.append("\n" + "-" * 70)
        report_lines.append("6. RECOMMENDATIONS")
        report_lines.append("-" * 70)

        for i, rec in enumerate(self.results["recommendations"], 1):
            report_lines.append(f"\n{i}. {rec}")

        # Summary
        report_lines.append("\n" + "=" * 70)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append(f"\nDataset Health Score: {self.results['health_score']}")
        report_lines.append(f"Total Images: {structure['total_images']}")
        report_lines.append(f"Classes: {len(self.classes)}")
        report_lines.append(f"Splits: {', '.join(self.splits)}")
        report_lines.append(f"\nKey Metrics:")
        report_lines.append(f"  - Train/Val/Test ratio: {structure['split_ratios'].get('train', 0)*100:.1f}% / {structure['split_ratios'].get('val', 0)*100:.1f}% / {structure['split_ratios'].get('Test', 0)*100:.1f}%")
        report_lines.append(f"  - Class imbalance ratio: {structure['imbalance_ratio']:.2f}")
        report_lines.append(f"  - Corrupted images: {len(quality['corrupted'])}")
        report_lines.append(f"  - Cross-split duplicates: {dups['total_cross_split']}")
        report_lines.append(f"  - Total quality issues: {sum(len(v) for v in issues.values())}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        # Write report to file
        report_text = "\n".join(report_lines)
        report_path = Path("/home/user/Clothes_recognition_Model/dataset_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"\n  Report saved to: {report_path}")

        # Also print the report
        print("\n" + report_text)

        # Save detailed JSON results
        json_report_path = OUTPUT_DIR / "analysis_results.json"

        # Convert defaultdicts to regular dicts for JSON serialization
        json_results = {
            "structure": self.results["structure"],
            "image_quality": {
                "statistics": self.results["image_quality"]["statistics"],
                "formats": dict(self.results["image_quality"]["formats"]),
                "color_modes": dict(self.results["image_quality"]["color_modes"]),
                "too_small_count": len(self.results["image_quality"]["too_small"]),
                "too_large_count": len(self.results["image_quality"]["too_large"]),
                "corrupted_count": len(self.results["image_quality"]["corrupted"]),
                "unusual_aspect_count": len(self.results["image_quality"]["unusual_aspect"])
            },
            "duplicates": {
                "total_exact": self.results["duplicates"]["total_exact_duplicates"],
                "total_cross_split": self.results["duplicates"]["total_cross_split"],
                "groups": len(self.results["duplicates"]["exact_duplicates"])
            },
            "issues": {k: len(v) for k, v in self.results["issues"].items()},
            "health_score": self.results["health_score"],
            "recommendations": self.results["recommendations"]
        }

        with open(json_report_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"  JSON results saved to: {json_report_path}")


def main():
    analyzer = DatasetAnalyzer(DATASET_PATH, SPLITS, CLASSES)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()
