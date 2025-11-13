import cv2
import numpy as np

def select_homography_points(video_path, dst_size=(500, 300)):
    points = []

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = param
            if len(pts) < 4:
                pts.append((x, y))
                print(f"Point {len(pts)} selected: ({x}, {y})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return None, None

    ret, first_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return None, None

    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", select_points, points)

    while True:
        display_frame = first_frame.copy()
        for p in points:
            cv2.circle(display_frame, p, 5, (0, 255, 0), -1)

        cv2.imshow("Select 4 Points", display_frame)
        key = cv2.waitKey(1)
        if key == 27 or len(points) == 4:
            break

    cv2.destroyWindow("Select 4 Points")

    if len(points) != 4:
        print("4점을 선택하지 않았습니다.")
        return None, None

    src_pts = np.float32(points)
    dst_pts = np.float32([[0,0],[dst_size[0],0],[dst_size[0],dst_size[1]],[0,dst_size[1]]])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    cap.release()
    return H, dst_size
