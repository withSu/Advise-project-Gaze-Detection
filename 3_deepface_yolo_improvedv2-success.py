import cv2
from ultralytics import YOLO
from deepface import DeepFace
import time
from collections import Counter

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 해상도 낮춰 성능 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age and Gender', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age and Gender', 640, 480)

# 분석 관련 변수
last_analysis_time = 0          # 마지막 분석 시간
analysis_interval = 5           # 분석 주기 (5초)
analysis_results = {}           # ID별 분석 결과 누적

# 평균 나이 계산 함수
def get_average_age(result_list):
    if not result_list:
        return None
    ages = [result['age'] for result in result_list]
    return int(sum(ages) / len(ages))

# 가장 빈번한 값 선택 함수 (성별 등)
def get_most_common(result_list, key):
    if not result_list:
        return None
    if key == 'gender':
        # gender는 딕셔너리이므로, 가장 높은 확률의 성별을 추출
        values = [max(result[key], key=result[key].get) for result in result_list]
    else:
        values = [result[key] for result in result_list]
    most_common = Counter(values).most_common(1)[0][0]
    return most_common

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # YOLO로 사람 추적
    results = model.track(frame, classes=[0], persist=True)  # classes=[0]은 '사람' 클래스

    # 5초마다 분석 수행
    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    if track_id not in analysis_results:
                        analysis_results[track_id] = []  # 새로운 ID 초기화
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_roi = frame[y1:y2, x1:x2]
                    try:
                        analysis = DeepFace.analyze(img_path=person_roi, actions=['age', 'gender'], 
                                                    enforce_detection=False, detector_backend='opencv')
                        if analysis:
                            analysis_results[track_id].append(analysis[0])  # 결과 누적
                    except Exception as e:
                        print(f"분석 중 오류 (ID {track_id}): {e}")
        last_analysis_time = current_time  # 분석 시간 갱신

    # 결과 표시
    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                if track_id in analysis_results and analysis_results[track_id]:
                    age = get_average_age(analysis_results[track_id])
                    gender = get_most_common(analysis_results[track_id], 'gender')
                    label = f'ID: {track_id}, Age: {age}, Gender: {gender}'
                else:
                    label = f'ID: {track_id} (Analyzing...)'
                
                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("ID가 없는 객체가 감지되었습니다. 추적이 실패했거나 새로운 객체입니다.")

    # 화면에 표시
    cv2.imshow('Real-time Tracking with Age and Gender', frame)
    if cv2.waitKey(1) == 27:  # Esc 키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()