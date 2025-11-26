import itertools
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict

# ============================================================
# 0) 동적 키패드 환경 정의
# ============================================================

# 고정 버튼 슬롯 인덱스 (4x4 그리드, 왼쪽 위부터 0, 1, 2, 3...)
FIX_SHUFFLE = 12   # 재배열
FIX_BACK    = 13   # ←
FIX_CLEAR   = 15   # 정정

fixed_slots = {FIX_SHUFFLE, FIX_BACK, FIX_CLEAR}


# ============================================================
# 1) PIN 후보 추론 로직 (동적 키패드 286가지 경우의 수 고려)
# ============================================================

def possible_pins_from_slots(slot_sequence, max_candidates=10):
    """
    슬롯 시퀀스를 받아 모든 가능한 동적 키패드 배치 (286가지)를 생성하여 
    가장 자주 등장하는 PIN 후보 Top10을 반환.
    """
    pin_counter = {}
    free_slots = [s for s in range(16) if s not in fixed_slots]
    from itertools import combinations

    # 가능한 워터마크 위치 조합 286가지 탐색
    for wm_positions_tuple in combinations(free_slots, 3):
        keypad = {}
        
        for s in free_slots:
            keypad[s] = None

        wm_positions = set(wm_positions_tuple)
        for s in wm_positions:
            keypad[s] = "WM"

        # 남은 슬롯에 숫자 순서대로 배치 (1~9, 0)
        numbers = ["1","2","3","4","5","6","7","8","9","0"]
        idx = 0
        for s in free_slots:
            if keypad[s] is None:
                keypad[s] = numbers[idx]
                idx += 1

        keypad[FIX_SHUFFLE] = "SHUF"
        keypad[FIX_BACK]    = "BACK"
        keypad[FIX_CLEAR]   = "CLEAR"

        # slot 시퀀스를 PIN으로 변환
        pin = ""
        for s in slot_sequence:
            v = keypad.get(s, None)
            if v in ["BACK", "CLEAR", "SHUF", "WM", None]:
                pin += "?"      # 모호하거나 기능 버튼인 경우
            else:
                pin += v

        if pin:
            pin_counter[pin] = pin_counter.get(pin, 0) + 1

    ranked = sorted(pin_counter.items(), key=lambda x: -x[1])
    return ranked[:max_candidates]


# ============================================================
# 2-A. ATM 물리 치수 정의 (CM 단위)
# ============================================================

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

# ============================================================
# 2-B. MediaPipe Hands 초기화
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# 3) Homography 및 키패드 매핑 함수
# ============================================================

# 3-A. 영상에서 ATM 화면 네 점 클릭받기
src_pts = []
frame_copy = None 

def click_event(event, x, y, flags, param):
    """ 마우스 클릭 이벤트 핸들러 """
    global src_pts, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_pts) < 4:
            src_pts.append([x, y])
            print(f"[CLICK {len(src_pts)}] (x={x}, y={y})")

            cv2.circle(frame_copy, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame_copy, f"P{len(src_pts)}", (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if len(src_pts) > 1:
                cv2.line(frame_copy, tuple(src_pts[-2]), tuple(src_pts[-1]), (0,255,0), 2)

            cv2.imshow("Select ATM Corners", frame_copy)
        else:
             print("경고: 이미 4개의 점이 선택되었습니다. 'q'를 눌러 완료하세요.")


def select_points(frame):
    """ Homography를 위한 4개의 코너를 선택 """
    global frame_copy
    frame_copy = frame.copy()
    src_pts.clear()
    
    print("좌상 → 우상 → 우하 → 좌하 순서로 클릭 후 'q'를 누르세요.")

    cv2.imshow("Select ATM Corners", frame_copy)
    cv2.setMouseCallback("Select ATM Corners", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cv2.destroyWindow("Select ATM Corners")
    return np.float32(src_pts)

# 3-B. ATM 화면 픽셀 크기 계산
def compute_atm_pixel_size(pts):
    tl, tr, br, bl = pts
    width_px = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height_px = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return width_px, height_px

# 3-C. 실제 ATM 화면 비율 기반 warp 크기
def compute_output_canvas(width_px):
    aspect = ATM_H_CM / ATM_W_CM
    W = int(width_px)
    H = int(W * aspect)
    return W, H

# 3-D. 키패드 layout 계산
def compute_keypad_layout(pxX, pxY):
    x0 = int(KEYPAD_LEFT_CM * pxX)
    y0 = int(KEYPAD_TOP_CM  * pxY)

    btn_w = int(BTN_W_CM * pxX)
    btn_h = int(BTN_H_CM * pxY)
    gap_w = int(GAP_W_CM * pxX)
    gap_h = int(GAP_H_CM * pxY)

    return x0, y0, btn_w, btn_h, gap_w, gap_h

# 3-E. 어떤 키를 눌렀는지 매핑 (슬롯 인덱스 반환)
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

# ============================================================
# 4) End-to-End 메인 파이프라인 (터치 필터링 로직 추가)
# ============================================================

def process_video_and_rank_pins(video_path="data/processed_videos/dynamic/7045_은아_동적1.MOV.mp4"):
    
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("영상 로드 실패!")
        return

    # ① 네 점 선택 (Homography 입력)
    pts = select_points(frame)
    if len(pts) != 4:
        print("경고: 4개의 점이 선택되지 않았습니다. 종료합니다.")
        cap.release()
        return

    # ② Homography 및 키패드 계산
    atm_w_px, atm_h_px = compute_atm_pixel_size(pts)
    W, H = compute_output_canvas(atm_w_px)
    dst_pts = np.float32([[0,0],[W,0],[W,H],[0,H]])
    H_matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    pxX = W / ATM_W_CM
    pxY = H / ATM_H_CM
    x0, y0, btn_w, btn_h, gap_w, gap_h = compute_keypad_layout(pxX, pxY)

    # ③ 손가락 추적 + 터치 감지 및 필터링
    fingertips = []
    slot_sequence = []
    last_touch = -10
    
    # 💡 임계값 및 필터 강화
    MIN_PEAK = -1.8      # Y축 가속도 임계값 강화
    MIN_FRAME_GAP = 13       # 최소 10 프레임 간격 (약 0.33초)
    last_key = None          # 마지막으로 감지된 슬롯 인덱스

    frame_idx = 0
    cap.release()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- MediaPipe로 검지 추적 ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        fx, fy = -1, -1

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark[8]
            fx = int(lm.x * frame.shape[1])
            fy = int(lm.y * frame.shape[0])
            fingertips.append([fx, fy])

        # --- Warp 변환 및 시각화 ---
        warped = cv2.warpPerspective(frame, H_matrix, (W, H))

        # 가속도 계산
        if len(fingertips) > 2 and fx != -1:
            coords = np.array(fingertips)
            vel = np.diff(coords, axis=0) * fps
            acc = np.diff(vel, axis=0) * fps
            ay = acc[-1,1]

            # --- 터치 감지 조건 (임계값 + 시간 필터) ---
            if ay < MIN_PEAK and (frame_idx - last_touch) > MIN_FRAME_GAP:
                
                # 원본 좌표를 Warp 좌표로 변환
                p = np.array([[[fx, fy]]], dtype=np.float32)
                wp = cv2.perspectiveTransform(p, H_matrix)[0][0]
                wx, wy = int(wp[0]), int(wp[1])

                # 키 매핑: 슬롯 인덱스 추출
                key = map_key(wx, wy, x0, y0, btn_w, btn_h, gap_w, gap_h)
                
                if key is not None:
                    # 💡 잔떨림 제거 로직: 1초(30프레임) 이내 동일 키 중복 무시
                    if key == last_key and (frame_idx - last_touch) < 30: 
                        pass # 중복 터치 무시
                    else:
                        last_touch = frame_idx
                        slot_sequence.append(key)
                        last_key = key          # 마지막 키 업데이트
                        print(f"Touch detected! Key = {key}, ay={ay:.1f}")

                    cv2.circle(warped, (wx, wy), 10, (0,255,0), 3)

        # --- 키패드 윤곽 그리기 ---
        idx = 0
        for r in range(4):
            for c in range(4):
                bx = x0 + c*(btn_w+gap_w)
                by = y0 + r*(btn_h+gap_h)
                cv2.rectangle(warped, (bx,by), (bx+btn_w,by+btn_h), (0,255,0),2)
                idx+=1

        cv2.imshow("Warped with Keys + Touch Detection", warped)

        if cv2.waitKey(1) & 0xFF == 27: # ESC 키
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n====================================")
    print("Detected slot sequence:", slot_sequence)
    print("====================================")


    # ④ PIN 후보 생성 및 순위화
    if slot_sequence:
        candidates = possible_pins_from_slots(slot_sequence, max_candidates=10)
        print("\n=== Top10 PIN 후보 (동적 키패드 고려) ===")
        for pin, score in candidates:
            print(f"{pin}   (count={score})")
    else:
        print("감지된 터치 슬롯이 없습니다.")

    return slot_sequence

# ============================================================
# 5) 실행 예시
# ============================================================

# ★★★ 중요: 이 경로를 사용하시는 영상 파일 경로로 변경해야 합니다 ★★★
process_video_and_rank_pins(video_path="data/processed_videos/dynamic/7045_은아_동적1.MOV.mp4")