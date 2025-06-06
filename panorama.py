import cv2
import glob
import os
import numpy as np

# Konfiguration
IMAGE_FOLDER = 'images_2'
PATTERN      = '*.png'   # Ändra om du använder jpg

# Läs in bilder
image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, PATTERN)))
if not image_paths:
    raise RuntimeError(f"Inga bilder hittade i {IMAGE_FOLDER} med mönster {PATTERN}")
imgs = [cv2.imread(p) for p in image_paths]

# Panoramasy ihop bilderna
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch(imgs)
if status != cv2.Stitcher_OK:
    raise RuntimeError(f"Stitching misslyckades (kod {status})")

# Spara rå panoramabild
raw_path = os.path.join(IMAGE_FOLDER, 'mosaic_raw.jpg')
cv2.imwrite(raw_path, pano)
print(f"Rå mosaik sparad som {raw_path}")

# Beskär bort svarta kanter
gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
ys, xs = np.where(mask > 0)
x1, x2 = xs.min(), xs.max()
y1, y2 = ys.min(), ys.max()
crop = pano[y1:y2+1, x1:x2+1]

# Gör fyrkantig med padding 
h, w = crop.shape[:2]
side = max(h, w)
pad_vert = side - h
pad_top = pad_vert // 2
pad_bottom = pad_vert - pad_top
pad_horiz = side - w
pad_left = pad_horiz // 2
pad_right = pad_horiz - pad_left

# Vit bakgrund (ändra till [0,0,0] för svart)
square = cv2.copyMakeBorder(
    crop,
    pad_top, pad_bottom,
    pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT,
    value=[255,255,255]
)

# Spara slutlig fyrkantig mosaik
square_path = os.path.join(IMAGE_FOLDER, 'mosaic_square.jpg')
cv2.imwrite(square_path, square)
print(f"Fyrkantig mosaik sparad som {square_path}")
