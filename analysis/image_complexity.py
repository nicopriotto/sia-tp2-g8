import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from skimage.measure import regionprops, label as sklabel


_HIST_BINS = 32  # bins per channel for color histogram entropy (C_color)
_K = 16          # color clusters for connected-component segmentation (C_forma)

# Calibration: practical maximum for each component.
_C_COLOR_MAX = 0.55
_C_FORMA_MAX = 0.80


def image_complexity(path: str, alpha: float = 0.5, sigma: float = 2.0,
                     min_area: int = 50) -> dict:
    """
    Compute image complexity C ∈ [0, 1] based on color diversity and shape irregularity.

    C = alpha * C_color + (1 - alpha) * C_forma

    C_color: entropy of the RGB color histogram.
    C_forma: area-weighted average of (1 - solidity) across connected regions.
             solidity = area / convex_hull_area — robust to JPEG artifacts.
    """
    # 1. Load and preprocess
    img = Image.open(path).convert("RGB").resize((256, 256))
    pixels = np.array(img, dtype=float)
    pixels_smooth = gaussian_filter(pixels, sigma=sigma)
    flat = pixels_smooth.reshape(-1, 3)

    # 2. C_color: normalized color histogram entropy
    hist, _ = np.histogramdd(flat / 255.0, bins=_HIST_BINS,
                             range=[[0, 1], [0, 1], [0, 1]])
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    H = float(-np.sum(probs * np.log(probs)))
    C_color = min(H / np.log(_HIST_BINS ** 3) / _C_COLOR_MAX, 1.0)

    # 3. C_forma: area-weighted (1 - solidity) over connected regions
    labels = KMeans(n_clusters=_K, n_init=5, random_state=42).fit_predict(flat)
    quantized = labels.reshape(256, 256)

    irregularidades, areas = [], []
    for color_id in range(_K):
        mask = (quantized == color_id).astype(np.uint8)
        for prop in regionprops(sklabel(mask)):
            if prop.area >= min_area:
                irregularidades.append(1.0 - prop.solidity)
                areas.append(prop.area)

    C_forma = min(
        float(np.average(irregularidades, weights=areas)) / _C_FORMA_MAX if areas else 0.0,
        1.0,
    )

    # 4. Combine
    C = alpha * C_color + (1 - alpha) * C_forma

    return {
        "C": round(C, 4),
        "C_color": round(C_color, 4),
        "C_forma": round(C_forma, 4),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m analysis.image_complexity <image_path> [alpha]")
        sys.exit(1)

    path = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    result = image_complexity(path, alpha=alpha)
    print(f"Image: {path}")
    print(f"  C        = {result['C']}")
    print(f"  C_color  = {result['C_color']}")
    print(f"  C_forma  = {result['C_forma']}")
