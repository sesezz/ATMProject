# -*- coding: utf-8 -*-
import cv2
import glob
import os
import mediapipe as mp
import numpy as np
from homography_detector import select_homography_points

### 정적키패드(종이) 탐지 可
# ===============================
# Homography 적용 함수
# ===============================
def apply_homography(H, points):
    pts = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    mapped = (H @ pts.T).T
    mapped /= mapped[:, 2:3]
    return mapped[:, :2]

# ===============================
# 버튼 탐지 함수 (4x4 키패드)
# ===============================
def create_keypads(dst_size):
    W, H = dst_size
    keypads = {}
    rows, cols = 4, 4
    btn_w = W / cols
    btn_h = H / rows
    labels = [
        "1","2","3","4",
        "5","6","7","8",
        "9","0","*","*",
        "*","*","*","*"
    ]
    for i in range(rows):
        for j in range(cols):
            keypads[labels[i*cols + j]] = (j*btn_w, i*btn_h, (j+1)*btn_w, (i+1)*btn_h)
    return keypads

def detect_key(mapped_point, keypads):
    x, y = mapped_point
    for key, (x1, y1, x2, y2) in keypads.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None

# ===============================
# Mediapipe Hands 설정
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ===============================
# 영상 폴더 설정
# ===============================
video_folder = 'data/processed_videos/static/'
video_files = glob.glob(os.path.join(video_folder, '6382_민송_정적*.MOV.mp4'))

if not video_files:
    print("오류: 폴더 안에 mp4 파일이 없습니다.")
    exit()

# ===============================
# 영상 처리 시작
# ===============================
for video_path in video_files:
    H, dst_size = select_homography_points(video_path)
    if H is None:
        print("Homography 설정 실패. 넘어갑니다.")
        continue

    print("Homography matrix from detector:\n", H)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다 → {video_path}")
        continue

    keypads = create_keypads(dst_size)
    pin_sequence = []

    prev_points = None
    mapped_prev_points = None
    click_count = 0
    min_move_threshold = 5
    display_click_text = ""
    display_counter = 0
    display_duration = 30

    print(f"영상 처리중: {video_path}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("비디오 스트림 종료.")
            break

        adjusted = cv2.convertScaleAbs(image, alpha=1.3, beta=-130)
        rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        current_mapped_points = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                )

                h, w, _ = image.shape
                current_points = np.array([
                    [w * hand_landmarks.landmark[0].x, h * hand_landmarks.landmark[0].y],   # wrist
                    [w * hand_landmarks.landmark[5].x, h * hand_landmarks.landmark[5].y],   # index_mcp
                    [w * hand_landmarks.landmark[17].x, h * hand_landmarks.landmark[17].y]  # pinky_mcp
                ], dtype=np.float32)

                current_mapped_points = apply_homography(H, current_points)

                # -------------------------
                # Click 감지
                # -------------------------
                if prev_points is not None and mapped_prev_points is not None:
                    delta = current_mapped_points - mapped_prev_points
                    negative_mask = (delta[:, 0] < -min_move_threshold) | (delta[:, 1] < -min_move_threshold)
                    neg_points = current_mapped_points[negative_mask]

                    if len(neg_points) >= 1:
                        click_count += 1
                        display_click_text = f"Click {click_count}!"
                        display_counter = display_duration

                        # -------------------------
                        # PIN 입력 추적
                        # -------------------------
                        key = detect_key(current_mapped_points[1], keypads)  # 검지 기준
                        if key:
                            pin_sequence.append(key)
                            print(f"[Click {click_count}] 입력된 키: {key}")
                            if len(pin_sequence) == 4:
                                print("4자리 PIN 완성:", "".join(pin_sequence))
                                pin_sequence = []

                prev_points = current_points
                mapped_prev_points = current_mapped_points

        # -------------------------
        # 화면 표시
        # -------------------------
        if display_counter > 0:
            cv2.putText(
                image, display_click_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA
            )
            display_counter -= 1

        warped_frame = cv2.warpPerspective(image, H, (dst_size[0], dst_size[1]))

        # 키패드 버튼 영역 표시
        for key, (x1, y1, x2, y2) in keypads.items():
            cv2.rectangle(warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 2)
            cv2.putText(warped_frame, key, (int(x1)+10, int(y1)+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 손 좌표 표시
        if current_mapped_points is not None:
            for p in current_mapped_points:
                px, py = int(p[0]), int(p[1])
                cv2.circle(warped_frame, (px, py), 6, (0, 0, 255), -1)

        cv2.imshow("Homography View", warped_frame)
        cv2.imshow("MediaPipe Hand Tracking (ESC to skip)", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

hands.close()
print("모든 영상 처리 완료.")
