import cv2
from ultralytics import YOLO
from deepface import DeepFace
import time

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 해상도 낮춰 성능 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age and Gender', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age and Gender', 640, 480)

last_analysis = {}      # 추적 ID별 분석 결과 저장
analyzed_ids = set()    # 이미 분석된 ID 저장

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # YOLO로 사람 추적
    results = model.track(frame, classes=[0], persist=True)  # classes=[0]은 '사람' 클래스

    for result in results:
        for box in result.boxes:
            if box.id is not None:  # ID가 있는 경우에만 처리
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                track_id = int(box.id.item())  # 추적 ID

                # 아직 분석되지 않은 ID에 대해서만 얼굴 분석 수행
                if track_id not in analyzed_ids:
                    person_roi = frame[y1:y2, x1:x2]
                    try:
                        analysis = DeepFace.analyze(img_path=person_roi, actions=['age', 'gender'], 
                                                    enforce_detection=False, detector_backend='opencv')
                        if analysis:
                            last_analysis[track_id] = analysis[0]  # 첫 번째 얼굴 결과 저장
                            analyzed_ids.add(track_id)  # 분석 완료로 기록
                    except Exception as e:
                        print(f"분석 중 오류 (ID {track_id}): {e}")

                # 분석 결과 표시
                if track_id in last_analysis:
                    age = last_analysis[track_id]['age']
                    gender = last_analysis[track_id]['gender']
                    label = f'ID: {track_id}, Age: {age}, Gender: {gender}'
                else:
                    label = f'ID: {track_id} (Analyzing...)'

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("ID가 없는 객체가 감지되었습니다. 추적이 실패했거나 새로운 객체입니다.")

    # 결과 화면에 표시
    cv2.imshow('Real-time Tracking with Age and Gender', frame)
    if cv2.waitKey(1) == 27:  # Esc 키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()