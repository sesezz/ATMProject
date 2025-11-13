# 영상 내 손가락 인식 확인
import cv2
import mediapipe as mp
from homography_detector import select_homography_points
import numpy as np

# Homography 점 4개 선택
video_path = '../data/processed_videos/static/6382_민송_정적1.MOV.mp4'
H, dst_size = select_homography_points(video_path)
if H is None:
    print("Homography 선택 실패")
    exit()

# MediaPipe 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# --- Homography 좌표 준비 ---
dst_pts = np.array([[0,0],[dst_size[0],0],[dst_size[0],dst_size[1]],[0,dst_size[1]]], dtype=np.float32)
src_pts = cv2.perspectiveTransform(dst_pts[None, :, :], np.linalg.inv(H))[0]

# 영상 처리
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("영상 스트림 종료.")
        break

    # 대비/밝기 조정
    alpha, beta = 1.3, -130
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # MediaPipe 처리
    rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    mp_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # MediaPipe 랜드마크 시각화
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                mp_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)
            )

    # Homography 네 점 표시 (원본 영상에)
    for pt in src_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(mp_frame, (x, y), 5, (0,255,0), -1)

    # --- Warp 영상 생성 (실시간) ---
    warped = cv2.warpPerspective(frame, H, (int(dst_size[0]), int(dst_size[1])))
    for pt in dst_pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(warped, (x, y), 5, (0,255,0), -1)

    # --- 화면 출력 ---
    cv2.imshow('MediaPipe + Homography Points', mp_frame)
    cv2.imshow('Warped View', warped)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 종료
        break

# 루프 종료 후 warp 영상 정지해서 확인
cv2.imshow("Warped Video (Last Frame)", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()
hands.close()
