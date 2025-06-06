import cv2, math, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter

# -------------------------------------------------
# 1. Modell & bild
# -------------------------------------------------
MODEL_PATH = "best_2.pt"
IMG_PATH   = "mosaic_raw.jpg"

model  = YOLO(MODEL_PATH)
res    = model(IMG_PATH, task="segment")[0]
orig   = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
h, w   = orig.shape[:2]

# -------------------------------------------------
# 2. Gemensam hinder‑mask (boxar + segment + dilation)
# -------------------------------------------------
obstacles = np.zeros((h, w), np.uint8)
MARGIN = 0

for i, box in enumerate(res.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(obstacles,
                  (max(0, x1-MARGIN), max(0, y1-MARGIN)),
                  (min(w-1, x2+MARGIN), min(h-1, y2+MARGIN)),
                  255, -1)
    if res.masks is not None and len(res.masks.data) > i:
        obstacles |= res.masks.data[i].cpu().numpy().astype(np.uint8)*255

obstacles = cv2.dilate(obstacles, np.ones((25, 25), np.uint8), 2)

# -------------------------------------------------
# 3. Expanderad rektangel + säkra hörn
# -------------------------------------------------
cnts,_ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt_union = np.vstack(cnts)
x,y,wc,hc = cv2.boundingRect(cnt_union)

OFF, CLEAR = 20, 10
rect = [
    [max(0,x-OFF),           max(0,y-OFF)],
    [min(w-1,x+wc+OFF),      max(0,y-OFF)],
    [min(w-1,x+wc+OFF),      min(h-1,y+hc+OFF)],
    [max(0,x-OFF),           min(h-1,y+hc+OFF)]
]

dt = distance_transform_edt(obstacles==0)
cx, cy = x+wc/2, y+hc/2
for pt in rect:
    if dt[pt[1], pt[0]] >= CLEAR: continue
    vx, vy = pt[0]-cx, pt[1]-cy
    if vx==vy==0: vx=1
    ux, uy = np.array([vx,vy])/math.hypot(vx,vy)
    for step in range(200):
        qx, qy = int(pt[0]+ux*step), int(pt[1]+uy*step)
        if 0<=qx<w and 0<=qy<h and dt[qy,qx] >= CLEAR:
            pt[0], pt[1] = qx, qy; break

# -------------------------------------------------
# 4. Kostnadskarta
# -------------------------------------------------
free  = obstacles==0
dist  = distance_transform_edt(free).astype(np.float32)
cost  = 1.0 + 50.0/(dist+1e-3)
cost[~free] = 1e6

def astar(a,b):
    idx,_ = route_through_array(cost, a[::-1], b[::-1],
                                fully_connected=True, geometric=True)
    return np.array(idx)[:, ::-1]

# -------------------------------------------------
# 5. Planera slinga
# -------------------------------------------------
start = (0, h//2)
wps   = [start] + rect + [start]
segs  = []
for A,B in zip(wps[:-1], wps[1:]):
    segs.append(astar(A,B) if not segs else astar(A,B)[1:])
path = np.vstack(segs)

if len(path) >= 31:
    path = np.vstack([
        savgol_filter(path[:,0],31,3),
        savgol_filter(path[:,1],31,3)
    ]).T.astype(int)

# -------------------------------------------------
# 6. Rita CANVAS 1 (mask + box)  → spara
# -------------------------------------------------
canvas1 = orig.copy()
overlay = canvas1.copy(); overlay[obstacles==255] = (255,0,0)
canvas1 = cv2.addWeighted(overlay,0.25,canvas1,0.75,0)

for box in res.boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    cv2.rectangle(canvas1,(x1,y1),(x2,y2),(0,0,0),2)

cv2.imwrite("objects_mask_only.jpg", cv2.cvtColor(canvas1, cv2.COLOR_RGB2BGR))
print("Sparad som objects_mask_only.jpg")

# -------------------------------------------------
# 7. Kopiera → rita väg + waypoints  → spara CANVAS 2
# -------------------------------------------------
canvas2 = canvas1.copy()
for p1,p2 in zip(path[:-1], path[1:]):
    cv2.line(canvas2, tuple(p1), tuple(p2), (0,255,0), 3)
cv2.circle(canvas2, start, 8, (0,255,255), -1)
for pt in rect:
    cv2.circle(canvas2, tuple(pt), 6, (255,0,255), -1)

cv2.imwrite("robot_path_overlay.jpg",
            cv2.cvtColor(canvas2, cv2.COLOR_RGB2BGR))
print("Sparad som robot_path_overlay.jpg")

# -------------------------------------------------
# 8. Visa båda (valfritt)
# -------------------------------------------------
plt.figure(figsize=(14,6))
plt.subplot(1,2,1); plt.imshow(canvas1); plt.axis("off"); plt.title("Endast mask + box")
plt.subplot(1,2,2); plt.imshow(canvas2); plt.axis("off"); plt.title("Mask + box + väg")
plt.show()
