import cv2, numpy as np, re, glob, os
from pathlib import Path

# ======= Configurable Hyperparameters =======
SIZE        = 320          # Unified output size
MARGIN      = 0.15         # ROI bounding box expansion ratio (0.10~0.25)
PAD_VAL     = 255          # Padding color (white)
MIN_AREA_FR = 0.0005       # Minimum candidate area ratio, skip if too small (prevent treating noise as main object)
SCORE_BETA  = 0.4          # Contour scoring: beta in area_norm * (aspect^beta)
                            # Higher beta prefers elongated shapes; 0.3~0.6 commonly used
# =======================

def natural_key(p):  # Slide1, Slide2, ...
    m = re.findall(r'(\d+)', Path(p).stem)
    return int(m[-1]) if m else 0

def letterbox(img, size=SIZE, pad_value=PAD_VAL):
    H, W = img.shape[:2]
    scale = min(size/W, size/H)
    nw, nh = int(W*scale), int(H*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    x0 = (size - nw)//2; y0 = (size - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _binary_robust(gray):
    """
    Returns two binary images: adaptive threshold & Otsu; union will be used for contours.
    This ensures stability even with uneven background lighting.
    """
    g = cv2.GaussianBlur(gray, (3,3), 0)

    # Otsu: foreground in black
    _, th_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive: blockSize 51 adjustable, C set between 2~8
    th_ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 51, 4)

    # Morphological cleanup
    k = np.ones((3,3), np.uint8)
    th_otsu = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, k, iterations=2)
    th_ad   = cv2.morphologyEx(th_ad,   cv2.MORPH_CLOSE, k, iterations=2)
    return th_ad, th_otsu

def _score_contour(cnt, H, W):
    area = cv2.contourArea(cnt)
    if area <= 1: return -1, (0,0,0,0)
    x,y,w,h = cv2.boundingRect(cnt)
    # Area ratio + elongation (aspect ratio)
    area_norm = area / float(H*W)
    aspect = max(w, h) / (min(w, h) + 1e-6)
    score = area_norm * (aspect ** SCORE_BETA)
    return score, (x,y,w,h)

def find_roi_bbox(gray):
    """
    Returns (x0,y0,x1,y1). Returns full image on failure.
    """
    H, W = gray.shape
    th1, th2 = _binary_robust(gray)
    th = cv2.bitwise_or(th1, th2)

    # Find contours
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0,0,W,H), {'used_mask': th, 'reason': 'no_cnt'}

    # Filter out tiny candidates
    min_area = MIN_AREA_FR * H * W
    best = (-1, (0,0,0,0))
    for c in cnts:
        if cv2.contourArea(c) < min_area: 
            continue
        score, rect = _score_contour(c, H, W)
        if score > best[0]:
            best = (score, rect)

    if best[0] < 0:
        return (0,0,W,H), {'used_mask': th, 'reason': 'no_valid_cnt'}

    x,y,w,h = best[1]
    # Expand borders
    mx = int(w * MARGIN); my = int(h * MARGIN)
    x0 = max(0, x - mx); y0 = max(0, y - my)
    x1 = min(W, x + w + mx); y1 = min(H, y + h + my)
    return (x0,y0,x1,y1), {'used_mask': th, 'reason': 'ok'}

def crop(img, box):
    x0,y0,x1,y1 = box
    return img[y0:y1, x0:x1]

def preprocess_one(in_path, out_path):
    bgr = cv2.imread(str(in_path))
    if bgr is None:
        raise RuntimeError(f"Cannot read: {in_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Find ROI
    box, info = find_roi_bbox(gray)
    roi = crop(bgr, box)

    # Unified size: proportional scaling + padding
    out = letterbox(roi, size=SIZE, pad_value=PAD_VAL)
    if np.all(out == out[0, 0]):
        raise RuntimeError(f"Preprocessing produced a uniform image: {in_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out):
        raise RuntimeError(f"Cannot write: {out_path}")

def process_dir(in_dir, out_dir, pattern="*.PNG", debug=False):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    paths = sorted(glob.glob(str(in_dir/pattern)), key=natural_key)
    if not paths:
        print(f"No files found: {in_dir/pattern}")
        return
    for i, p in enumerate(paths, 1):
        fname = f"slide_{i:04d}.PNG"
        preprocess_one(p, out_dir/fname)
    print(f"Processed {len(paths)} images -> {out_dir}")

if __name__ == "__main__":
    process_dir("slides_png", "dataset/raw/images")
