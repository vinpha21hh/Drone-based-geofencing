# compare_green_blue.py

import cv2 as cv
import numpy as np

def path_length_simple(image_path, lower_hsv, upper_hsv,
                       min_length=100, morph_kernel=3, closed=False):
    """
    Trösklar på ett HSV‐intervall, rensar små fläckar och returnerar
    längden på den längsta konturen (arcLength).
    """
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Kan inte läsa in: {image_path}")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Maska ut färgen
    mask = cv.inRange(hsv,
                      np.array(lower_hsv, dtype=np.uint8),
                      np.array(upper_hsv, dtype=np.uint8))

    # Morfologisk öppning för brus
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # Hitta konturer
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError(f"Inga konturer hittades i {image_path}")

    # Filtrera bort för korta
    valid = [c for c in contours
             if cv.arcLength(c, closed) >= min_length]
    if not valid:
        raise RuntimeError(f"Inga konturer ≥ {min_length}px i {image_path}")

    # Välj längsta konturen och returnera dess längd
    best = max(valid, key=lambda c: cv.arcLength(c, closed))
    return cv.arcLength(best, closed)


def main():
    # --- Grönt (robotens bana) ----
    green_lower = [40,  50,  50]   # H:40–80, S/V höga nog
    green_upper = [80, 255, 255]

    # --- Blått (människans bana) ---
    # För RGB (0,0,255) blir HSV ≈ (Hue=240° → H≈120 i OpenCV)
    blue_lower = [100, 50,  50]    # prova H från 100 till 130
    blue_upper = [130, 255, 255]

    robot_img = "robot_path_overlay.jpg"
    human_img = "human.jpg"

    L_robot = path_length_simple(robot_img, green_lower, green_upper,
                                 min_length=200, morph_kernel=5, closed=True)
    L_human = path_length_simple(human_img,  blue_lower,  blue_upper,
                                 min_length=200, morph_kernel=5, closed=False)

    print(f"Robotens bana:  {L_robot:.1f} px")
    print(f"Människans bana: {L_human:.1f} px")
    if L_robot>0:
        print(f"Relativ skillnad: {(L_human/L_robot-1)*100:.1f}%")

if __name__ == "__main__":
    main()
