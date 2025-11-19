# homography 및 MediaPipe를 이용한 ATM 키패드 터치 감지 및 PIN 추출
import cv2
import numpy as np
import mediapipe as mp

###################################################
# 0. MediaPipe Hands 초기화
###################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

###################################################
# 1. ATM 실제 물리 치수
###################################################
ATM_W_CM = 30.5
ATM_H_CM = 23.2

KEYPAD_W_CM = 13.7
KEYPAD_H_CM = 12.5
KEYPAD_LEFT_CM = 15.8
KEYPAD_TOP_CM  = 4.0

BTN_W_CM = 3.1
BTN_H_CM = 3.0
GAP_W_CM = 0.4
GAP_H_CM = 0.2

###################################################
# 2. 영상에서 ATM 화면 네 점 클릭받기
###################################################
src_pts = []

def click_event(event, x, y, flags, param):
    global src_pts, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        src_pts.append([x, y])
        print(f"[CLICK {len(src_pts)}] (x={x}, y={y})")

        cv2.circle(frame_copy, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(frame_copy, f"P{len(src_pts)}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if len(src_pts) > 1:
            cv2.line(frame_copy, tuple(src_pts[-2]), tuple(src_pts[-1]), (0,255,0), 2)

        cv2.imshow("Select ATM Corners", frame_copy)

def select_points(frame):
    global frame_copy
    frame_copy = frame.copy()

    print("좌상 → 우상 → 우하 → 좌하 클릭 후 q")

    cv2.imshow("Select ATM Corners", frame_copy)
    cv2.setMouseCallback("Select ATM Corners", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyWindow("Select ATM Corners")
    return np.float32(src_pts)

###################################################
# 3. ATM 화면 픽셀 크기 계산
###################################################
def compute_atm_pixel_size(pts):
    tl, tr, br, bl = pts
    width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return width_px, height_px

###################################################
# 4. 실제 ATM 화면 비율 기반 warp 크기
###################################################
def compute_output_canvas(width_px):
    aspect = ATM_H_CM / ATM_W_CM
    W = int(width_px)
    H = int(W * aspect)
    return W, H

###################################################
# 5. 키패드 layout 계산
###################################################
def compute_keypad_layout(pxX, pxY):
    x0 = int(KEYPAD_LEFT_CM * pxX)
    y0 = int(KEYPAD_TOP_CM  * pxY)

    btn_w = int(BTN_W_CM * pxX)
    btn_h = int(BTN_H_CM * pxY)
    gap_w = int(GAP_W_CM * pxX)
    gap_h = int(GAP_H_CM * pxY)

    return x0, y0, btn_w, btn_h, gap_w, gap_h

###################################################
# 6-A. 어떤 키를 눌렀는지 매핑
###################################################
def map_key(wx, wy, x0, y0, btn_w, btn_h, gap_w, gap_h):
    idx = 0
    for r in range(4):
        for c in range(4):
            bx = x0 + c*(btn_w + gap_w)
            by = y0 + r*(btn_h + gap_h)

            if bx <= wx <= bx + btn_w and by <= wy <= by + btn_h:
                return idx
            idx += 1
    return None

###################################################
# ---------------- MAIN PIPELINE ------------------
###################################################

video = "data/processed_videos/static/검지2.MOV.mp4"
cap = cv2.VideoCapture(video)

ret, frame = cap.read()
if not ret:
    raise Exception("영상 로드 실패!")

# ① 네 점 선택
pts = select_points(frame)
atm_w_px, atm_h_px = compute_atm_pixel_size(pts)

# ② warp 크기 결정
W, H = compute_output_canvas(atm_w_px)
dst_pts = np.float32([[0,0],[W,0],[W,H],[0,H]])
H_matrix = cv2.getPerspectiveTransform(pts, dst_pts)

# ③ 스케일 계산
pxX = W / ATM_W_CM
pxY = H / ATM_H_CM

# ④ 키패드 layout
x0, y0, btn_w, btn_h, gap_w, gap_h = compute_keypad_layout(pxX, pxY)

###################################################
# 7. 손가락 추적 + 터치 감지 + PIN 추출
###################################################
fingertips = []
PIN = []
last_touch = -10
MIN_PEAK = -0.02

frame_idx = 0

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- MediaPipe로 검지 추적 ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark[8]
        fx = int(lm.x * frame.shape[1])
        fy = int(lm.y * frame.shape[0])

        fingertips.append([fx, fy])

    # --- warp ---
    warped = cv2.warpPerspective(frame, H_matrix, (W, H))

    # 3프레임 이후부터 가속도 계산 가능
    if len(fingertips) > 2:
        coords = np.array(fingertips)
        vel = np.diff(coords, axis=0) * fps
        acc = np.diff(vel, axis=0) * fps

        ay = acc[-1,1]

        # --- 터치 감지 ---
        if ay < MIN_PEAK and (frame_idx - last_touch) > 2:

            last_touch = frame_idx

            # 원본 검지 좌표 → warp 좌표
            p = np.array([[[fx, fy]]], dtype=np.float32)
            wp = cv2.perspectiveTransform(p, H_matrix)[0][0]
            wx, wy = int(wp[0]), int(wp[1])

            # 키 매핑
            key = map_key(wx, wy, x0, y0, btn_w, btn_h, gap_w, gap_h)
            if key is not None:
                PIN.append(key)
                print(f"Touch detected! Key = {key}")

            cv2.circle(warped, (wx, wy), 10, (0,255,0), 3)

    # --- 키패드 그리기 ---
    idx = 0
    for r in range(4):
        for c in range(4):
            bx = x0 + c*(btn_w+gap_w)
            by = y0 + r*(btn_h+gap_h)
            cv2.rectangle(warped, (bx,by), (bx+btn_w,by+btn_h), (0,255,0),2)
            idx+=1

    cv2.imshow("Warped with Keys + Touch Detection", warped)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print("Detected PIN:", PIN)
