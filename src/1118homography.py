# 실제 ATM 화면 비율 기반 호모그래피 + 키패드 오버레이
import cv2
import numpy as np

###################################################
# 1. ATM 실제 물리 치수
###################################################
ATM_W_CM = 30.5   # ATM 화면 가로(cm)
ATM_H_CM = 23.2   # ATM 화면 세로(cm)

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

        # 저장
        src_pts.append([x, y])
        print(f"[CLICK {len(src_pts)}] (x={x}, y={y})")

        # 시각화: 클릭한 점 표시
        cv2.circle(frame_copy, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(frame_copy, f"P{len(src_pts)}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 점 개수가 2 이상이면 선으로 연결
        if len(src_pts) > 1:
            cv2.line(frame_copy, tuple(src_pts[-2]), tuple(src_pts[-1]), (0,255,0), 2)

        cv2.imshow("Select ATM Corners", frame_copy)

def select_points(frame):
    global frame_copy
    frame_copy = frame.copy()

    print("🟩 좌상(TL) → 우상(TR) → 우하(BR) → 좌하(BL) 순으로 네 점 클릭하고")
    print("🟦 완료 후 'q' 누르기")

    cv2.imshow("Select ATM Corners", frame_copy)
    cv2.setMouseCallback("Select ATM Corners", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyWindow("Select ATM Corners")

    return np.float32(src_pts)


###################################################
# 3. ATM 화면 픽셀 크기 자동 계산
###################################################
def compute_atm_pixel_size(pts):
    tl, tr, br, bl = pts

    # 가로 평균 픽셀 길이
    width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2

    # 세로 평균 픽셀 길이
    height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    return width_px, height_px

###################################################
# 4. 실제 비율 기반으로 warped 화면 크기 결정
###################################################
def compute_output_canvas(width_px, height_px):
    aspect = ATM_H_CM / ATM_W_CM  # 실제 ATM 화면 비율

    W = int(width_px)     # 가로 픽셀 = 그대로
    H = int(W * aspect)   # 비율 유지하여 세로 계산

    return W, H

###################################################
# 5. 키패드 좌표 계산 (실제 치수 기반)
###################################################
def compute_keypad_layout(px_per_cm_x, px_per_cm_y):

    # 키패드 시작점 (px)
    x0 = int(KEYPAD_LEFT_CM * px_per_cm_x)
    y0 = int(KEYPAD_TOP_CM  * px_per_cm_y)

    keypad_w = int(KEYPAD_W_CM * px_per_cm_x)
    keypad_h = int(KEYPAD_H_CM * px_per_cm_y)

    btn_w = int(BTN_W_CM * px_per_cm_x)
    btn_h = int(BTN_H_CM * px_per_cm_y)

    gap_w = int(GAP_W_CM * px_per_cm_x)
    gap_h = int(GAP_H_CM * px_per_cm_y)

    return x0, y0, keypad_w, keypad_h, btn_w, btn_h, gap_w, gap_h

###################################################
# ---------------- MAIN PIPELINE ------------------
###################################################

video = "data/processed_videos/static/7045_정적2_민송2.MOV.mp4"
cap = cv2.VideoCapture(video)

ret, frame = cap.read()
if not ret:
    raise Exception("영상 로드 실패!")

# ① 네 점 클릭
pts = select_points(frame)
if len(pts) != 4:
    raise Exception("네 점을 정확히 클릭하세요!")

# ② ATM 화면 실제 픽셀 크기 자동 계산
atm_w_px, atm_h_px = compute_atm_pixel_size(pts)
print("ATM Pixel Width:", atm_w_px, "ATM Pixel Height:", atm_h_px)

# ③ 실제 비율대로 warped 크기 결정
W, H = compute_output_canvas(atm_w_px, atm_h_px)
print("Warped Size:", W, H)

# ④ 호모그래피 행렬 계산
dst_pts = np.float32([[0,0],[W,0],[W,H],[0,H]])
H_matrix = cv2.getPerspectiveTransform(pts, dst_pts)

# ⑤ px/cm 스케일 결정
px_per_cm_x = W / ATM_W_CM
px_per_cm_y = H / ATM_H_CM

# ⑥ 키패드 레이아웃 자동 계산
x0, y0, keypad_w, keypad_h, btn_w, btn_h, gap_w, gap_h = compute_keypad_layout(px_per_cm_x, px_per_cm_y)

###################################################
# 6. 영상 재생하며 호모그래피 + 키패드 오버레이 출력
###################################################
while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped = cv2.warpPerspective(frame, H_matrix, (W, H))

    # 키패드 그리기
    idx = 0
    for r in range(4):
        for c in range(4):

            x = x0 + c * (btn_w + gap_w)
            y = y0 + r * (btn_h + gap_h)

            cv2.rectangle(warped, (x, y), (x+btn_w, y+btn_h), (0,255,0), 2)
            cv2.putText(warped, str(idx), (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
            idx += 1

    cv2.imshow("Warped ATM with Keypad Overlay", warped)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
