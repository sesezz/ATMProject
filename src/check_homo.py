# -*- coding: utf-8 -*-
import cv2
import glob
import os
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

video_folder = 'data/processed_videos/static/'
#'*.mp4' 부분을 바꾸면 적용 가능합니다.
video_files = glob.glob(os.path.join(video_folder, '6382_민송_정적*.MOV.mp4'))

if not video_files:
    print("오류: 폴더 안에 mp4 파일이 없습니다.")
else:
    for video_path in video_files:
        print(f"영상 처리중: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"오류: 비디오 파일을 열 수 없습니다 → {video_path}")
            continue

        prev_points = None
        tracked_points = []
        click_count = 0
        min_move_threshold = 5  # 픽셀 기준
        min_points_for_h = 4
        display_click_text = ""  # 화면에 보여줄 텍스트
        display_counter = 0      # 텍스트 유지 프레임 카운트
        display_duration = 30    # 텍스트 표시 프레임 수

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("비디오 스트림 종료.")
                break

            alpha = 1.3
            beta = -130
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            image_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                    )

                    wrist = hand_landmarks.landmark[0]
                    index_mcp = hand_landmarks.landmark[5]
                    pinky_mcp = hand_landmarks.landmark[17]

                    h, w, _ = image.shape
                    current_points = np.array([
                        [w * wrist.x, h * wrist.y],
                        [w * index_mcp.x, h * index_mcp.y],
                        [w * pinky_mcp.x, h * pinky_mcp.y]
                    ], dtype=np.float32)

                    if prev_points is not None:
                        delta = current_points - prev_points
                        negative_mask = (delta[:, 0] < -min_move_threshold) | (delta[:, 1] < -min_move_threshold)
                        neg_points = current_points[negative_mask]

                        if len(neg_points) >= 1:
                            click_count += 1
                            display_click_text = f"Click {click_count}!"
                            display_counter = display_duration

                            # Homography 계산
                            if len(tracked_points + [neg_points]) >= min_points_for_h:
                                pts_src = np.vstack(tracked_points + [neg_points])[:min_points_for_h]
                                pts_dst = np.array([[0,0],[100,0],[100,100],[0,100]], dtype=np.float32)
                                H, status = cv2.findHomography(pts_src, pts_dst)
                                print(f"Click {click_count} Homography matrix:\n", H)
                            else:
                                print(f"Click {click_count} Homography 계산에 충분한 포인트 없음.")

                            tracked_points = []
                        else:
                            tracked_points.append(current_points)
                    else:
                        tracked_points.append(current_points)

                    prev_points = current_points

            # 화면에 클릭 텍스트 표시
            if display_counter > 0:
                cv2.putText(
                    image, display_click_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA
                )
                display_counter -= 1

            cv2.imshow('MediaPipe Hand Tracking (ESC to skip)', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

hands.close()
print("모든 영상 처리 완료.")
