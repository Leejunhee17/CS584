import cv2
import imutils

# OpenCV의 GOTURN 트래커 초기화
tracker = cv2.TrackerGOTURN_create()

# 카메라 장치를 열고 프레임을 가져옵니다.
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# GOTURN 트래커 초기화를 위한 바운딩 박스를 선택합니다.
bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 읽기 오류")
        break

    # GOTURN 트래커를 업데이트합니다.
    (success, bbox) = tracker.update(frame)

    if success:
        # 트래커가 성공적으로 업데이트되면 박스를 그립니다.
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# 카메라 장치를 닫습니다.
cap.release()
cv2.destroyAllWindows()
