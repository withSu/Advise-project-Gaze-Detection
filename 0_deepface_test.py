import cv2
from deepface import DeepFace
import time

# 웹캠 초기화 (0은 기본 카메라)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 해상도를 320x240으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창을 일반 모드로 설정하고 크기 조정 (640x480)
cv2.namedWindow('Real-time Age and Gender Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Age and Gender Detection', 640, 480)

last_analysis_time = 0  # 마지막 분석 시간
analysis_interval = 1   # 분석 주기 (1초)
last_analysis = []      # 마지막 분석 결과 저장

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # 1초마다 얼굴 분석 수행
    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        try:
            analysis = DeepFace.analyze(img_path=frame, actions=['age', 'gender'], 
                                       enforce_detection=False, detector_backend='opencv')
            last_analysis = analysis  # 성공한 분석 결과 저장
            last_analysis_time = current_time
        except Exception as e:
            print(f"분석 중 오류: {e}")
            analysis = []

    # 마지막 분석 결과를 화면에 표시
    if last_analysis:
        for face in last_analysis:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            age = face['age']
            gender = face['gender']
            # 얼굴 영역에 사각형과 텍스트 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {age}, Gender: {gender}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No face detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 결과 화면에 표시
    cv2.imshow('Real-time Age and Gender Detection', frame)
    if cv2.waitKey(1) == 27:  # Esc 키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()