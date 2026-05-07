import cv2
import numpy as np
import pytesseract
import mss
import re
import time
import csv
import os
from datetime import datetime

# ====== ROIs ======
HR_ROI   = (60, 220, 150, 290)
RESP_ROI = (190, 460, 350, 530)
SPO2_ROI = (240, 230, 330, 280)

# ====== FILE PATH ======
OUTPUT_FOLDER = "data"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "vitals_clean.csv")

# Create folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ====== HELPER ======

def clean_number(text):
    nums = re.findall(r'\d+', text)
    if nums:
        return int(nums[0])
    return None


def preprocess_common(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def preprocess_spo2(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return mask


# ====== OCR FUNCTIONS ======

def extract_hr(frame):
    x1,y1,x2,y2 = HR_ROI
    roi = frame[y1:y2, x1:x2]
    proc = preprocess_common(roi)

    text = pytesseract.image_to_string(
        proc,
        config='--psm 7 -c tessedit_char_whitelist=0123456789'
    )

    value = clean_number(text)
    if value is not None and 30 <= value <= 220:
        return value
    return None


def extract_resp(frame):
    x1,y1,x2,y2 = RESP_ROI
    roi = frame[y1:y2, x1:x2]
    proc = preprocess_common(roi)

    text = pytesseract.image_to_string(
        proc,
        config='--psm 7 -c tessedit_char_whitelist=0123456789'
    )

    value = clean_number(text)
    if value is not None and 5 <= value <= 60:
        return value
    return None


def extract_spo2(frame):
    x1,y1,x2,y2 = SPO2_ROI
    roi = frame[y1:y2, x1:x2]

    proc = preprocess_spo2(roi)

    text = pytesseract.image_to_string(
        proc,
        config='--psm 8 -c tessedit_char_whitelist=0123456789'
    )

    value = clean_number(text)
    if value is not None and 50 <= value <= 100:
        return value

    return None


# ====== SAVE TO CSV ======

def save_to_csv(hr, resp, spo2):

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "heart_rate", "respiration", "spo2"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hr,
            resp,
            spo2
        ])


# ====== MAIN LOOP ======

last_print = 0

with mss.mss() as sct:

    monitor = sct.monitors[1]

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        hr = extract_hr(frame)
        resp = extract_resp(frame)
        spo2 = extract_spo2(frame)

        # Only print + save every 3 seconds
        if time.time() - last_print >= 3:

            print("HR:", hr, "RESP:", resp, "SpO2:", spo2)

            # Save only if all values valid
            if hr is not None and resp is not None and spo2 is not None:
                save_to_csv(hr, resp, spo2)

            last_print = time.time()

        # Draw ROI boxes
        cv2.rectangle(frame, (HR_ROI[0], HR_ROI[1]), (HR_ROI[2], HR_ROI[3]), (0,255,0), 2)
        cv2.rectangle(frame, (RESP_ROI[0], RESP_ROI[1]), (RESP_ROI[2], RESP_ROI[3]), (255,0,0), 2)
        cv2.rectangle(frame, (SPO2_ROI[0], SPO2_ROI[1]), (SPO2_ROI[2], SPO2_ROI[3]), (0,0,255), 2)

        cv2.imshow("Screen OCR", frame)

        # Press Q to quit (CLICK window first)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
