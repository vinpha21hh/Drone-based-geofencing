import time
import os
import cv2
from djitellopy import Tello

# Flygsekvens: täck area 2m längd x 1m bredd med 50 cm steg
# Drönaren tar en bild vid varje 50 cm-stopp.

def simple_mission():
    # Initiera drönare
    drone = Tello()
    drone.connect()
    drone.streamoff()
    drone.streamon()

    # Takeoff och stig till 2 meter
    drone.takeoff()
    time.sleep(2)
    drone.move_up(100)
    time.sleep(2)

    # Skapa mapp för bilder
    folder = 'images'
    os.makedirs(folder, exist_ok=True)

    # Grundsteg i cm
    STEP = 50

    # Funktion för att spara ram med RGB-konvertering
    def save_frame(idx):
        frame = drone.get_frame_read().frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        path = os.path.join(folder, f"bild_{idx:02d}.png")
        cv2.imwrite(path, frame_rgb)
        print(f"Sparade {path}")

    # Börja med första bilden
    img_idx = 1
    save_frame(img_idx)
    img_idx += 1
    time.sleep(1)

    # Rörelsesekvens: 2m framåt, 1m höger, 2m bakåt
    sequence = [
        (drone.move_forward, 4),  # 2 m framåt (4×50cm)
        (drone.move_right,   2),  # 1 m åt höger (2×50cm)
        (drone.move_back,    4),  # 2 m bakåt  (4×50cm)  
    ]

    # Utför rörelser och ta bild vid varje stopp
    for action, steps in sequence:
        for _ in range(steps):
            action(STEP)
            time.sleep(1)
            save_frame(img_idx)
            img_idx += 1
            time.sleep(0.5)

    # Landning
    drone.land()
    drone.end()

if __name__ == '__main__':
    simple_mission()
