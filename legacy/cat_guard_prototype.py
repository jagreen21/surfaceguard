#!/usr/bin/env python3
import os, time, threading, signal, math, random

# ===== MOCK SWITCH =====
MOCK = os.getenv("CATGUARD_MOCK", "0") == "1"

# Try optional deps; in MOCK we safely stub them.
if not MOCK:
    import cv2
else:
    cv2 = None

try:
    from mocks import pigpio_mock as pigpio
except Exception:
    if MOCK:
        from mocks.pigpio_mock import pi as pigpio_pi
        class _Pig:
            def __init__(self): self._p = pigpio_pi()
            def __getattr__(self, k): return getattr(self._p, k)
        pigpio = _Pig()  # bare shim with .pi() below
        def pi(): return pigpio
    else:
        raise

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    tflite = None

from collections import deque

# ========= USER CONFIG =========
# (Same interface as the Pi version; harmless on desktop in MOCK)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ssd_mobilenet_v2.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "models", "coco_labels.txt")

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CONF_THRESH = 0.5
CAT_CLASS_NAME = "cat"

DETER_DURATION_S = 20
RECHECK_PAUSE_S = 4
NO_CAT_CLEAR_S = 10

ULTRA_FREQ_HZ = 25000
ULTRA_DUTIES = [200_000, 400_000, 700_000]
ULTRA_PIN = 18

CLICK_PIN = 12
CLICK_FREQ_HZ = 3000
CLICK_DUTY = 500_000
CLICK_PATTERN = [(0.12, 0.08), (0.12, 0.25)]

IR_PIN = 24
DARK_MEAN_THRESH = 40
BRIGHT_MEAN_THRESH = 60

COLLAR_PIN = 23
COLLAR_PRESS_S = 1.0

SMOOTH_WINDOW = 15

stop_flag = False

def cleanup(*_a):
    global stop_flag
    stop_flag = True
    try:
        if not MOCK:
            pi.hardware_PWM(ULTRA_PIN, 0, 0)
            pi.hardware_PWM(CLICK_PIN, 0, 0)
    except Exception:
        pass
    print("[cat_guard] Clean exit.")
    raise SystemExit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ----- Labels -----
def load_labels(path):
    labels = {}
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    labels[i] = name
    except FileNotFoundError:
        pass
    return labels

labels = load_labels(LABELS_PATH)
cat_label_ids = [i for i, n in labels.items() if n == CAT_CLASS_NAME]
if not cat_label_ids:
    cat_label_ids = [17]  # common COCO id

# ----- Camera / Model init -----
if MOCK:
    cap = None   # no camera required
    interpreter = None
else:
    import cv2
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if tflite is None:
        raise RuntimeError("tflite_runtime not available on Pi; install it before deployment.")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# ----- pigpio -----
if MOCK:
    class _PiMock:
        connected = True
        def set_mode(self, *a, **k): pass
        def write(self, *a, **k): pass
        def hardware_PWM(self, *a, **k): pass
        def stop(self): pass
    pi = _PiMock()
else:
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running on Pi.")

def play_ultrasonic(duration_s, duty):
    if MOCK:
        print(f"[MOCK] ultrasonic {ULTRA_FREQ_HZ/1000:.1f} kHz duty={duty/10000:.1f}% for {duration_s}s")
        time.sleep(min(duration_s, 0.5))  # don't stall your desktop for 20s while testing
    else:
        pi.hardware_PWM(ULTRA_PIN, ULTRA_FREQ_HZ, duty)
        time.sleep(duration_s)
        pi.hardware_PWM(ULTRA_PIN, 0, 0)

def clicker_reward():
    if MOCK:
        print("[MOCK] click-click (reinforcement)")
        return
    for on_s, off_s in CLICK_PATTERN:
        pi.hardware_PWM(CLICK_PIN, CLICK_FREQ_HZ, CLICK_DUTY)
        time.sleep(on_s)
        pi.hardware_PWM(CLICK_PIN, 0, 0)
        time.sleep(off_s)

def press_collar():
    if MOCK:
        print("[MOCK] collar vibrate (relay press 1s)")
        return
    pi.write(COLLAR_PIN, 1)
    time.sleep(COLLAR_PRESS_S)
    pi.write(COLLAR_PIN, 0)

def control_ir(mean_luma, ir_on):
    if MOCK:
        return ir_on
    if IR_PIN is None:
        return ir_on
    if not ir_on and mean_luma < DARK_MEAN_THRESH:
        pi.write(IR_PIN, 1); return True
    if ir_on and mean_luma > BRIGHT_MEAN_THRESH:
        pi.write(IR_PIN, 0); return False
    return ir_on

def preprocess(frame):
    ih, iw = input_details[0]['shape'][1], input_details[0]['shape'][2]
    import cv2
    img = cv2.resize(frame, (iw, ih))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img_rgb, axis=0).astype(input_details[0]['dtype'])

def infer_real(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    for i in range(count):
        if scores[i] >= CONF_THRESH and classes[i] in cat_label_ids:
            return True, float(scores[i])
    return False, 0.0

def infer_mock():
    # Simulate "cat present" in bursts, so you can test the logic and output
    # 6s present -> 4s absent pattern
    period = 10.0
    t = time.time() % period
    present = t < 6.0
    conf = 0.7 if present else 0.1
    return present, conf

def frame_luma_mock():
    # Alternate bright/dark to exercise IR logic
    return 30 if int(time.time()) % 8 < 4 else 80

try:
    import numpy as np
except Exception:
    # We'll avoid numpy path in mock if not installed
    np = None

presence = deque(maxlen=15)
last_seen_ts = 0
state_cat_present = False
in_deterrence = False
ir_on = False

print("[cat_guard] Running", "(MOCK)" if MOCK else "")
print(" Press Ctrl+C to stop.")

def deterrence_cycle():
    global in_deterrence, state_cat_present
    in_deterrence = True
    try:
        for duty in ULTRA_DUTIES:
            play_ultrasonic(DETER_DURATION_S, duty)
            time.sleep(RECHECK_PAUSE_S)
            # quick recheck
            still_here = False
            for _ in range(6):
                if MOCK:
                    d2, _ = infer_mock()
                else:
                    ok2, fr2 = cap.read()
                    if not ok2: continue
                    d2, _ = infer_real(fr2)
                if d2:
                    still_here = True; break
                time.sleep(0.08)
            if not still_here:
                clicker_reward()
                press_collar()
                state_cat_present = False
                break
        else:
            # finished all escalations
            time.sleep(1)
    finally:
        in_deterrence = False

while not stop_flag:
    if MOCK:
        detected, conf = infer_mock()
        mean_luma = frame_luma_mock()
    else:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        import cv2
        mean_luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        detected, conf = infer_real(frame)

    presence.append(1 if detected else 0)
    smooth = sum(presence) > (len(presence)//2)
    now = time.time()

    if not MOCK:
        ir_on = control_ir(mean_luma, ir_on)

    if smooth:
        last_seen_ts = now
        if not state_cat_present and not in_deterrence:
            state_cat_present = True
            print("[cat_guard] Cat detected. Starting deterrence cycle.")
            threading.Thread(target=deterrence_cycle, daemon=True).start()
    else:
        if state_cat_present and (now - last_seen_ts) >= NO_CAT_CLEAR_S and not in_deterrence:
            print("[cat_guard] Cat left. Reward click + collar.")
            clicker_reward()
            press_collar()
            state_cat_present = False

    time.sleep(0.02)
