"""Page type classifier using image statistics.

Classifies scanned document pages into:
- typed: clean printed text (→ Marker)
- handwritten: readable cursive/print handwriting (→ Qwen2.5-VL)
- hardest: composited, rotated, very messy (→ Chandra + rotation)
- diagram: org charts, flowcharts, tables (→ Qwen2.5-VL)
- blank: empty or near-empty pages (→ skip)

Uses OpenCV image analysis — no ML model needed.
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PageClassification:
    page_class: str       # typed/handwritten/hardest/diagram/blank
    confidence: float     # 0-1
    rotation_hint: int    # 0, 90, 180, 270
    features: dict        # raw feature values for debugging


def classify_page(image_path: str) -> PageClassification:
    """Classify a page image by type."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return PageClassification("blank", 1.0, 0, {"error": "could not read"})

    h, w = img.shape
    total_pixels = h * w

    # --- Feature extraction ---

    # 1. Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Ink density
    ink_pixels = np.count_nonzero(binary)
    ink_density = ink_pixels / total_pixels

    # Blank detection
    if ink_density < 0.005:
        return PageClassification("blank", 0.95, 0, {"ink_density": ink_density})

    # 3. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    if len(areas) == 0:
        return PageClassification("blank", 0.9, 0, {"ink_density": ink_density, "components": 0})

    num_components = len(areas)
    avg_area = np.mean(areas)
    area_std = np.std(areas)
    area_cv = area_std / avg_area if avg_area > 0 else 0  # coefficient of variation

    # Filter meaningful components (> 20 pixels)
    meaningful = areas > 20
    num_meaningful = np.sum(meaningful)
    meaningful_areas = areas[meaningful]

    # Large component ratio (components > 1% of page area)
    large_components = np.sum(areas > total_pixels * 0.01)

    # 4. Horizontal projection profile
    proj = binary.sum(axis=1) / 255
    # Find peaks and valleys
    proj_normalized = proj / max(proj.max(), 1)
    # Count line-like peaks (rows with > 10% of max ink)
    text_rows = proj_normalized > 0.1
    # Count transitions (text → gap → text = one line)
    transitions = np.diff(text_rows.astype(int))
    num_lines = np.sum(transitions == 1)

    # Line spacing regularity
    line_starts = np.where(transitions == 1)[0]
    if len(line_starts) > 2:
        spacings = np.diff(line_starts)
        spacing_cv = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 1
    else:
        spacing_cv = 1.0  # unknown = assume irregular

    # 5. Aspect ratio of components
    if len(widths[meaningful]) > 0:
        aspect_ratios = widths[meaningful].astype(float) / np.maximum(heights[meaningful], 1)
        avg_aspect = np.mean(aspect_ratios)
        aspect_std = np.std(aspect_ratios)
    else:
        avg_aspect = 1.0
        aspect_std = 0.0

    # 6. Edge density (Canny)
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.count_nonzero(edges) / total_pixels

    # 7. Rotation check — compare projection regularity at 0° vs 180°
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    _, bin_180 = cv2.threshold(img_180, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj_180 = bin_180.sum(axis=1) / 255
    # The "right" orientation usually has sharper projection peaks
    proj_sharpness_0 = np.std(proj)
    proj_sharpness_180 = np.std(proj_180)

    rotation_hint = 0
    if proj_sharpness_180 > proj_sharpness_0 * 1.2:
        rotation_hint = 180

    # Check 90° too (for landscape pages scanned portrait)
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    _, bin_90 = cv2.threshold(img_90, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj_90 = bin_90.sum(axis=1) / 255
    proj_sharpness_90 = np.std(proj_90)
    if proj_sharpness_90 > max(proj_sharpness_0, proj_sharpness_180) * 1.2:
        rotation_hint = 90

    features = {
        "ink_density": round(ink_density, 4),
        "num_components": num_components,
        "num_meaningful": int(num_meaningful),
        "avg_area": round(avg_area, 1),
        "area_cv": round(area_cv, 2),
        "large_components": int(large_components),
        "num_lines": int(num_lines),
        "spacing_cv": round(spacing_cv, 3),
        "avg_aspect": round(avg_aspect, 2),
        "edge_density": round(edge_density, 4),
        "proj_sharpness_0": round(proj_sharpness_0, 1),
        "proj_sharpness_180": round(proj_sharpness_180, 1),
        "proj_sharpness_90": round(proj_sharpness_90, 1),
        "rotation_hint": rotation_hint,
    }

    # --- Classification rules ---
    # Tuned on 13 test pages with known ground truth.
    # Key discriminator: area_cv (component size variance)
    #   typed:       area_cv < 1.0 (uniform letter sizes)
    #   diagram:     area_cv > 5.0 with many lines/components
    #   handwritten: area_cv 3-8, fewer components
    #   hardest:     rotation needed, or very messy features

    # Hardest: needs rotation — always flag
    if rotation_hint != 0:
        return PageClassification("hardest", 0.8, rotation_hint, features)

    # Typed: uniform component sizes, many components, many lines
    if area_cv < 1.0 and num_meaningful > 500 and num_lines > 20:
        return PageClassification("typed", 0.85, rotation_hint, features)

    # Also typed: moderate uniformity with lots of content
    if area_cv < 1.7 and num_meaningful > 800 and num_lines > 30:
        return PageClassification("typed", 0.75, rotation_hint, features)

    # Diagram/table: high area variance + many lines + many components
    if area_cv > 5.0 and num_lines > 30 and num_meaningful > 1000:
        return PageClassification("diagram", 0.75, rotation_hint, features)

    # Diagram: very high area variance with structured layout
    if area_cv > 10.0 and num_meaningful > 500:
        return PageClassification("diagram", 0.7, rotation_hint, features)

    # Hardest: few meaningful components + high variance (messy cursive)
    if num_meaningful < 500 and area_cv > 5.0:
        return PageClassification("hardest", 0.7, rotation_hint, features)

    # Hardest: very few lines detected (composited/illegible)
    if num_lines < 5 and num_meaningful < 500:
        return PageClassification("hardest", 0.65, rotation_hint, features)

    # Handwritten: moderate components, higher area variance than typed
    if num_meaningful < 600 and area_cv > 2.0:
        return PageClassification("handwritten", 0.7, rotation_hint, features)

    # Default: handwritten (safe default — Qwen handles both readable HW and moderate typed)
    return PageClassification("handwritten", 0.6, rotation_hint, features)


def classify_batch(image_paths: list) -> list:
    """Classify multiple pages."""
    return [classify_page(p) for p in image_paths]


if __name__ == "__main__":
    import sys, os, glob, json

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        # Default: test on our known pages
        paths = sorted(glob.glob("/home/omen/Documents/Project/Richiebot/scans/test_pages/*.png"))

    # Ground truth from WORKLOG
    GROUND_TRUTH = {
        "B_-_The_Keys_to_Demand_Generation_p1": "typed",
        "B_-_Notes1_p1": "hardest",
        "B_-_Notes1_p2": "hardest",
        "R_-_Org_Charts_p1": "diagram",
        "R_-_Org_Charts_p2": "hardest",
        "R_-_Presenataions_p9": "diagram",
        "R_-_Theory_of_the_Business_-_Underlying_Philosophy_p1": "typed",
        "R_-_Theory_of_the_Business_-_Underlying_Philosophy_p3": "typed",
        "R_-_Written_Notes_p1": "hardest",
        "R_-_Written_Notes_p3": "handwritten",
        "R_-_Written_Notes_p4": "handwritten",
        "RandB_-_Low_Hanging_Fruit_p1": "typed",
        "RandB_-_Low_Hanging_Fruit_p5": "diagram",
    }

    correct = 0
    total = 0
    for path in paths:
        result = classify_page(path)
        page_id = os.path.basename(path).replace(".png", "")
        gt = GROUND_TRUTH.get(page_id, "?")
        match = "OK" if result.page_class == gt else "MISS"
        if gt != "?":
            total += 1
            if result.page_class == gt:
                correct += 1

        print(f"  {match:4s}  {result.page_class:12s} (conf={result.confidence:.2f}, rot={result.rotation_hint:3d})  "
              f"gt={gt:12s}  {page_id}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.0f}%")
