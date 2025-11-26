# ===============================================
#  ATM 동적 키패드 - PIN 추론 공격 (최종 완성 버전)
#  * 우리은행 규칙 유지
#  * 마크 3개는 어디든 가능 (12,13,15 제외)
#  * 앞줄/뒷줄 모두 마크 균등 분포
#  * 숫자는 1~9,0 순서대로 비마크 칸에 채움
# ===============================================

import cv2
import numpy as np
import mediapipe as mp
import itertools
from itertools import combinations


# 0. MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.5
)


# 1. ATM 물리 치수
ATM_W_CM = 30.5
ATM_H_CM = 23.2

KEYPAD_LEFT_CM = 15.8
KEYPAD_TOP_CM  = 4.0
BTN_W_CM = 3.1
BTN_H_CM = 3.0
GAP_W_CM = 0.4
GAP_H_CM = 0.2


# 2. 네 점 클릭
src_pts = []

def click_event(event, x, y, flags, param):
    global src_pts, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        src_pts.append([x, y])
        print(f"[CLICK {len(src_pts)}] (x={x}, y={y})")
        cv2.circle(frame_copy, (x, y), 7, (0,255,0), -1)
        cv2.putText(frame_copy, f"P{len(src_pts)}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Select", frame_copy)
# 수동 좌표 
def select_points(frame):
    global frame_copy
    frame_copy = frame.copy()
    print("좌상→우상→우하→좌하 후 q")
    cv2.imshow("Select", frame_copy)
    cv2.setMouseCallback("Select", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Select")
    return np.float32(src_pts)


# 3. 픽셀 크기 계산
def compute_output_canvas(width_px):
    aspect = ATM_H_CM / ATM_W_CM
    W = int(width_px)
    H = int(W * aspect)
    return W, H

def compute_atm_pixel_size(pts):
    tl, tr, br, bl = pts
    width_px = (np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2
    height_px = (np.linalg.norm(bl-tl) + np.linalg.norm(br-tr)) / 2
    return width_px, height_px


# 4. keypad layout 계산
def compute_keypad_layout(pxX, pxY):
    x0 = int(KEYPAD_LEFT_CM * pxX)
    y0 = int(KEYPAD_TOP_CM  * pxY)
    btn_w = int(BTN_W_CM * pxX)
    btn_h = int(BTN_H_CM * pxY)
    gap_w = int(GAP_W_CM * pxX)
    gap_h = int(GAP_H_CM * pxY)
    return x0, y0, btn_w, btn_h, gap_w, gap_h


# 5. index 매핑
def map_key(wx, wy, x0, y0, btn_w, btn_h, gap_w, gap_h):
    idx = 0
    for r in range(4):
        for c in range(4):
            bx = x0 + c*(btn_w+gap_w)
            by = y0 + r*(btn_h+gap_h)
            if bx <= wx <= bx+btn_w and by <= wy <= by+btn_h:
                return idx
            idx += 1
    return None


# MAIN
video = "data/processed_videos/dynamic/2317_민송_동적2.MOV.mp4"

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
if not ret:
    raise Exception("영상 로드 실패")

pts = select_points(frame)
atm_w_px, atm_h_px = compute_atm_pixel_size(pts)
W, H = compute_output_canvas(atm_w_px)

dst_pts = np.float32([[0,0],[W,0],[W,H],[0,H]])
H_matrix = cv2.getPerspectiveTransform(pts, dst_pts)

pxX = W / ATM_W_CM
pxY = H / ATM_H_CM

x0, y0, btn_w, btn_h, gap_w, gap_h = compute_keypad_layout(pxX, pxY)


# 7. 터치 탐지 (하이브리드 안정화 버전)
fingertips = []
PIN_index = []
pin_conf = []

last_touch = -12
frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)

STABILIZE_FRAMES = int(fps * 0.20)
VEL_THRESHOLD = 0.5
ACC_THRESHOLD = -0.001
COOLDOWN = 1

cap = cv2.VideoCapture(video)
print("\n=== 터치 탐지 중 ===")



# === SPECIAL FIXED MAPPING ===
FORCE_DIGIT = {
    0: "1",   # index 0 → 무조건 숫자 1
    14: "0",   # index 14 → 무조건 숫자 0
    15: "0"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < STABILIZE_FRAMES:
        frame_idx += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # 손가락 끝 저장
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark[8]
        fx = int(lm.x * frame.shape[1])
        fy = int(lm.y * frame.shape[0])
        fingertips.append([fx, fy])

    warped = cv2.warpPerspective(frame, H_matrix, (W, H))

    # 키패드 그리기
    for r in range(4):
        for c in range(4):
            bx = x0 + c*(btn_w+gap_w)
            by = y0 + r*(btn_h+gap_h)
            cv2.rectangle(warped, (bx,by), (bx+btn_w,by+btn_h), (0,255,0),2)

    if len(fingertips) > 4:
        coords = np.array(fingertips)
        vel = np.diff(coords, axis=0) * fps          # 속도
        acc = np.diff(vel, axis=0) * fps             # 가속도

        vy_prev = vel[-2, 1]
        vy_now  = vel[-1, 1]
        ay      = acc[-1, 1]
        vel_drop = vy_prev - vy_now

        cond_vel = vel_drop > VEL_THRESHOLD
        cond_acc = ay < ACC_THRESHOLD

        cond_flip = (vy_prev < 0 and vy_now > 0)

        # 3) 중복 방지
        cond_cd = (frame_idx - last_touch) > COOLDOWN

        if (cond_flip or cond_vel or cond_acc) and cond_cd:
            last_touch = frame_idx

            p = np.array([[[fx, fy]]], dtype=np.float32)
            wp = cv2.perspectiveTransform(p, H_matrix)[0][0]
            wx, wy = int(wp[0]), int(wp[1])
        

            idx_key = map_key(wx, wy, x0, y0, btn_w, btn_h, gap_w, gap_h)
            if idx_key is not None:
                conf = min(abs(ay)/8000.0, 1.0)**2
                # ============================
                #  🔥 강제 숫자 매핑 적용!
                # ============================
                if idx_key in FORCE_DIGIT:
                    PIN_index.append(idx_key)
                    pin_conf.append(conf)
                    print(f"[TOUCH FIXED] index={idx_key} → digit={FORCE_DIGIT[idx_key]}, conf={conf:.4f}")
                    continue

                PIN_index.append(idx_key)
                pin_conf.append(conf)
                print(f"[TOUCH] index={idx_key}, conf={conf:.4f}, flip={cond_flip}")

            cv2.circle(warped, (wx, wy), 10, (0,0,255), 3)

    cv2.imshow("Warped", warped)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print("\nDetected index:", PIN_index)
print("Conf:", pin_conf)


# 8. 동적 키패드 생성
DIGITS = ['1','2','3','4','5','6','7','8','9','0']
MARK = '*'

def generate_dynamic_layouts():
    layouts = []

    for mark_pos in itertools.combinations(range(16), 3):

        if 12 in mark_pos or 13 in mark_pos or 15 in mark_pos:
            continue

        bad = False
        for a in range(14):
            if {a,a+1,a+2}.issubset(mark_pos):
                bad = True
                break
        if bad:
            continue

        layout = ['?' for _ in range(16)]

        for m in mark_pos:
            layout[m] = MARK

        layout[12] = 'R'
        layout[13] = 'B'
        layout[15] = 'C'

        nums = DIGITS.copy()
        for i in range(16):
            if layout[i] == '?':
                layout[i] = nums.pop(0)

        layouts.append(layout)

    return layouts


print("\n=== 가능한 동적 키패드 생성 중 ===")
layouts = generate_dynamic_layouts()
print("총 생성:", len(layouts))


# 9. PIN 후보 추론
def score_pin(pin, confs):
    return np.prod(confs)

print("\n=== PIN Top-100 계산 중 ===")

candidates = []
combo_indices = list(range(len(PIN_index)))
all_combos = list(combinations(combo_indices, 4))

for combo in all_combos:
    idx_seq  = [PIN_index[i] for i in combo]
    conf_seq = [pin_conf[i]   for i in combo]

    for layout in layouts:
        valid = True
        digits_real = []

        for idx in idx_seq:

            # 🔥 강제 mapping 먼저 적용
            if idx in FORCE_DIGIT:
                digits_real.append(FORCE_DIGIT[idx])
                continue

            key = layout[idx]
            if key in [MARK, 'R', 'B', 'C']:
                valid = False
                break
            digits_real.append(key)

        if not valid:
            continue

        score = score_pin(digits_real, conf_seq)
        candidates.append({
            "pin": ''.join(digits_real),
            "score": score
        })

pin_best = {}
for c in candidates:
    pin = c["pin"]
    sc  = c["score"]
    if pin not in pin_best or sc > pin_best[pin]:
        pin_best[pin] = sc

candidates_unique = [
    {"pin": p, "score": s} for p, s in pin_best.items()
]
candidates_unique = sorted(candidates_unique, key=lambda x: x["score"], reverse=True)[:100]

print("\n===== Top-100 PIN Candidates =====")
for i,c in enumerate(candidates_unique,1):
    print(f"{i:02d}. {c['pin']} (score={c['score']:.6e})")
