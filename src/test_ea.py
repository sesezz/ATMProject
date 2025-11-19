# -*- coding: utf-8 -*-
import cv2
import glob
import os
import mediapipe as mp
import numpy as np
from homography_detector import select_homography_points

def apply_homography(H, points):
    pts = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    mapped = (H @ pts.T).T
    mapped /= mapped[:, 2:3]
    return mapped[:, :2]

def create_keypads(dst_size):
    W, H = dst_size
    rows, cols = 4, 4
    btn_w = W / cols
    btn_h = H / rows
    labels = [
        "1","2","3","4",
        "5","6","7","8",
        "9","0","*","*",
        "*","*","*","*"
    ]
    keypads = {}
    for i in range(rows):
        for j in range(cols):
            keypads[labels[i*cols + j]] = (
                j*btn_w, i*btn_h, (j+1)*btn_w, (i+1)*btn_h
            )
    return keypads, labels

def detect_key(mapped_point, keypads):
    x, y = mapped_point
    for key, (x1, y1, x2, y2) in keypads.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

video_folder = 'data/processed_videos/static/'
video_files = glob.glob(os.path.join(video_folder, '검지2.MOV.mp4'))
if not video_files:
    print("오류: 비디오 없음")
    exit()

PIXEL_TOUCH_THRESHOLD = 60.0
SPEED_THRESHOLD = 5
ACC_THRESHOLD = 2
HOLD_FRAMES_REQUIRED = 2
SHOW_FRAMES = 25

for video_path in video_files:
    H, dst_size = select_homography_points(video_path)
    if H is None:
        print("Homography 실패:", video_path)
        continue

    # 역행렬 (dst→src)
    try:
        H_inv = np.linalg.inv(H)
    except Exception as e:
        print("H 역행렬 계산 실패:", e)
        continue

    print(f"\n=== Processing: {video_path}  dst_size={dst_size} ===")
    print("Homography matrix H:\n", H)

    # 키패드 정보 + 레이블
    keypads, labels = create_keypads(dst_size)

    # Warped 버튼 중심들 (dst coords)
    Wdst, Hdst = dst_size
    rows, cols = 4, 4
    btn_w = Wdst / cols
    btn_h = Hdst / rows

    dst_centers = []
    for i in range(rows):
        for j in range(cols):
            cx = j*btn_w + btn_w/2
            cy = i*btn_h + btn_h/2
            dst_centers.append((cx, cy))

    dst_centers_np = np.array(dst_centers, dtype=np.float32)

    # dst → src 변환
    centers_src = apply_homography(H_inv, dst_centers_np)  # shape (16,2)

    # 실제 숫자/문자 라벨 기반 매핑
    key_index_to_src = {}  # label → (x_src, y_src)
    for idx, label in enumerate(labels):
        key_index_to_src[label] = (
            float(centers_src[idx][0]),
            float(centers_src[idx][1])
        )

    print("Label-based key centers (SOURCE image):")
    for label in labels:
        print(f"  {label}: {key_index_to_src[label]}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 열기 실패:", video_path)
        continue

    pin_sequence = []
    prev_wrist_y = None
    prev_speed = 0
    hold_counter = 0
    is_clicking = False
    display_text = ""
    display_cnt = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        adjusted = cv2.convertScaleAbs(frame, alpha=1.3, beta=-130)
        h, w, _ = adjusted.shape

        rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        vis_frame = adjusted.copy()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(vis_frame, hand, mp_hands.HAND_CONNECTIONS)

            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            wrist_x = wrist.x * w
            wrist_y = wrist.y * h
            tip_x = tip.x * w
            tip_y = tip.y * h

            # 손목이 가장 가까운 label 찾기
            nearest_label_wrist = None
            min_dist_wrist = float('inf')

            for label, (kx, ky) in key_index_to_src.items():
                d = np.hypot(wrist_x - kx, wrist_y - ky)
                if d < min_dist_wrist:
                    min_dist_wrist = d
                    nearest_label_wrist = label

            # 손가락 끝이 가장 가까운 label 찾기
            nearest_label_tip = None
            min_dist_tip = float('inf')

            for label, (kx, ky) in key_index_to_src.items():
                d = np.hypot(tip_x - kx, tip_y - ky)
                if d < min_dist_tip:
                    min_dist_tip = d
                    nearest_label_tip = label

            print(f"[DBG] frame={frame_num} wrist=({int(wrist_x)},{int(wrist_y)}) "
                  f"tip=({int(tip_x)},{int(tip_y)}) "
                  f"nearest_tip={nearest_label_tip} dist={min_dist_tip:.1f}")

            # 시각화용 중심점
            for label, (kx, ky) in key_index_to_src.items():
                cv2.circle(vis_frame, (int(kx), int(ky)), 8, (0,255,255), 2)
                cv2.putText(vis_frame, label, (int(kx)-10, int(ky)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.circle(vis_frame, (int(wrist_x), int(wrist_y)), 6, (0,0,255), -1)
            cv2.circle(vis_frame, (int(tip_x), int(tip_y)), 6, (255,0,0), -1)

            # 클릭 판정
            click_by_tip = min_dist_tip < PIXEL_TOUCH_THRESHOLD

            click_by_speed = False
            if prev_wrist_y is not None:
                speed = wrist_y - prev_wrist_y
                acc = speed - prev_speed
                if speed > SPEED_THRESHOLD and acc > ACC_THRESHOLD:
                    hold_counter += 1
                else:
                    hold_counter = 0
                if hold_counter >= HOLD_FRAMES_REQUIRED:
                    click_by_speed = True
                prev_speed = speed
            prev_wrist_y = wrist_y

            click_detected = False
            chosen_key = None
            mapped_choice = None

            if click_by_tip:
                chosen_key = nearest_label_tip
                mapped_choice = key_index_to_src[chosen_key]
                click_detected = True

            elif click_by_speed:
                chosen_key = nearest_label_wrist
                mapped_choice = key_index_to_src[chosen_key]
                if min_dist_wrist < PIXEL_TOUCH_THRESHOLD * 2:
                    click_detected = True

            if click_detected and not is_clicking and chosen_key is not None:
                pin_sequence.append(chosen_key)
                display_text = f"CLICK! → {chosen_key}"
                display_cnt = SHOW_FRAMES
                print(f"[CLICK] f={frame_num} key={chosen_key} dist={min_dist_tip:.1f}")

                if len(pin_sequence) == 4:
                    print("★★ PIN 완성:", "".join(pin_sequence))

                is_clicking = True

            if not click_detected:
                is_clicking = False

        # Warped view (키패드 + 매핑된 점)
        warped = cv2.warpPerspective(adjusted, H, dst_size)

        # 키패드 번호 대신 실제 라벨 출력
        for i, (cx, cy) in enumerate(dst_centers):
            label = labels[i]
            cv2.circle(warped, (int(cx), int(cy)), 18, (200,200,200), 2)
            cv2.putText(warped, label, (int(cx)-10, int(cy)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # 클릭 위치 warping
        if 'mapped_choice' in locals() and mapped_choice is not None:
            mapped_in_warp = apply_homography(
                H, np.array([[mapped_choice[0], mapped_choice[1]]], dtype=np.float32)
            )[0]
            mapped_in_warp[0] = float(np.clip(mapped_in_warp[0], 0, dst_size[0]-1))
            mapped_in_warp[1] = float(np.clip(mapped_in_warp[1], 0, dst_size[1]-1))
            cv2.circle(warped, (int(mapped_in_warp[0]), int(mapped_in_warp[1])),
                       12, (0,255,255), 3)

        if display_cnt > 0:
            cv2.putText(warped, display_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            display_cnt -= 1

        cv2.putText(warped, "PIN: " + "".join(pin_sequence),
                    (10, dst_size[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Original (with centers & hand)", vis_frame)
        cv2.imshow("Warped View (keys & mapped points)", warped)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

hands.close()
print("=== Done ===")
