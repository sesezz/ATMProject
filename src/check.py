# 영상 내 손가락 인식 확인
import cv2
import mediapipe as mp

# MediaPipe 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 랜드마크 시각화를 위한 Drawing Utility
mp_drawing = mp.solutions.drawing_utils

# 파일 경로 설정: ATMProject 디렉토리 기준으로 작성!
video_path = 'data/processed_videos/static/'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# 영상 재생 및 시각화 
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("비디오 스트림 종료.")
        break


    # 대비, 밝기 조정
    alpha = 1.0  # 대비 (1.0 = 원본, 높을수록 강함)
    beta = 0    # 밝기 (+값은 밝게, -값은 어둡게)

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 1. MediaPipe 처리를 위해 BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    
    # 성능 최적화를 위해 이미지를 쓰기 불가 상태로 설정 (옵션)
    image_rgb.flags.writeable = False 
    
    # 2. 랜드마크 추론
    results = hands.process(image_rgb)
    
    # 3. 결과를 다시 BGR로 변환하여 시각화 준비
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 4. 랜드마크 시각화
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크(점)와 연결선(뼈대)을 이미지에 그립니다.
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), # 랜드마크 색상: 파랑
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)  # 연결선 색상: 빨강
            )
            
            # (추가 분석) 검지 끝 (Landmark 8) 위치를 확인하는 코드
            # index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # print(f"검지 끝 XY: ({index_finger_tip.x:.2f}, {index_finger_tip.y:.2f})")

    # 5. 화면에 영상 출력
    cv2.imshow('MediaPipe Hand Tracking (Press ESC to exit)', image)
    
    # 'ESC' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 종료 및 정리
cap.release()
hands.close()
cv2.destroyAllWindows()