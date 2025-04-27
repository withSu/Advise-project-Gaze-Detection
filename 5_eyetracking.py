import cv2
from ultralytics import YOLO
from deepface import DeepFace
from gaze_tracking import GazeTracking
import time
from collections import Counter

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# GazeTracking 초기화
gaze = GazeTracking()

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 해상도 낮춰 성능 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age, Gender, and Gaze', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age, Gender, and Gaze', 640, 480)

# 분석 관련 변수
last_analysis_time = 0          # 마지막 분석 시간
analysis_interval = 5           # 분석 주기 (5초)
analysis_results = {}           # ID별 분석 결과 누적
gaze_times = {}                 # ID별 응시 시간 누적 (프레임 단위)

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

    # 시선 추적 및 응시 시간 누적
    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                person_roi = frame[y1:y2, x1:x2]
                
                # GazeTracking으로 시선 추적
                gaze.refresh(person_roi)
                if gaze.is_center():  # 시선이 중앙(카메라 방향)을 향하고 있는 경우
                    if track_id not in gaze_times:
                        gaze_times[track_id] = 0
                    gaze_times[track_id] += 1  # 응시 시간 누적 (프레임 단위)

    # 결과 표시
    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                if track_id in analysis_results and analysis_results[track_id]:
                    age = get_average_age(analysis_results[track_id])
                    gender = get_most_common(analysis_results[track_id], 'gender')
                    gaze_time = gaze_times.get(track_id, 0)  # 응시 시간 (프레임 단위)
                    label = f'ID: {track_id}, Age: {age}, Gender: {gender}'
                    gaze_label = f'Gaze Time: {gaze_time} frames'
                    
                    # DeepFace가 감지한 얼굴 바운딩 박스 그리기 (빨간색)
                    for face in analysis_results[track_id]:
                        fx, fy, fw, fh = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                        face_x1 = x1 + fx
                        face_y1 = y1 + fy
                        face_x2 = face_x1 + fw
                        face_y2 = face_y1 + fh
                        cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                    
                    # YOLO 바운딩 박스 및 분석 정보 (녹색)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 응시 시간 표시 (파란색)
                    cv2.putText(frame, gaze_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    label = f'ID: {track_id} (Analyzing...)'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("ID가 없는 객체가 감지되었습니다. 추적이 실패했거나 새로운 객체입니다.")

    # 화면에 표시
    cv2.imshow('Real-time Tracking with Age, Gender, and Gaze', frame)
    if cv2.waitKey(1) == 27:  # Esc 키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()