import cv2
from ultralytics import YOLO
from deepface import DeepFace
import mediapipe as mp
import time
from collections import Counter

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# MediaPipe FaceMesh (Iris 포함) 초기화
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,         # 홍채(iris) 추적 활성화
    max_num_faces=1,               # 동시에 추적할 얼굴 개수 (필요시 늘릴 수 있음)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 해상도 낮춰 성능 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age, Gender, and Iris', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age, Gender, and Iris', 640, 480)

# 분석 관련 변수
last_analysis_time = 0
analysis_interval = 5
analysis_results = {}    # ID별 분석 결과
iris_times = {}          # ID별 '카메라 응시' 시간 (프레임 단위)

def get_average_age(result_list):
    if not result_list:
        return None
    ages = [r['age'] for r in result_list]
    return int(sum(ages) / len(ages))

def get_most_common(result_list, key):
    if not result_list:
        return None
    if key == 'gender':
        values = [max(r[key], key=r[key].get) for r in result_list]
    else:
        values = [r[key] for r in result_list]
    return Counter(values).most_common(1)[0][0]

# [도움 함수] 0~1로 정규화된 랜드마크를 실제 좌표로 변환
def denormalize_landmark(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

# [핵심 함수] MediaPipe Iris로 'face_roi' 분석 -> 홍채/눈 좌표 얻기
def analyze_iris(face_roi):
    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_face)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]  # 한 얼굴만 가정

# [예시] 단순히 '정면 응시'를 판단하는 로직 (매우 간단화)
def is_center_looking(face_landmarks, width, height):
    # 왼쪽 홍채 중심(468) ~ 오른쪽 홍채 중심(473~476) 등
    left_iris_idx = 468
    left_corner_idx = 33
    right_corner_idx = 133

    px, py = denormalize_landmark(face_landmarks.landmark[left_iris_idx], width, height)
    lx, ly = denormalize_landmark(face_landmarks.landmark[left_corner_idx], width, height)
    rx, ry = denormalize_landmark(face_landmarks.landmark[right_corner_idx], width, height)

    if rx != lx:  # 나눗셈 방지
        ratio = (px - lx) / float(rx - lx)  # 0.0 ~ 1.0
        if 0.4 < ratio < 0.6:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # YOLO로 사람 추적
    results = model.track(frame, classes=[0], persist=True)

    # 5초마다 DeepFace로 나이/성별 분석
    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    if track_id not in analysis_results:
                        analysis_results[track_id] = []
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_roi = frame[y1:y2, x1:x2]
                    try:
                        analysis = DeepFace.analyze(
                            img_path=person_roi,
                            actions=['age', 'gender'],
                            enforce_detection=False,
                            detector_backend='opencv'
                        )
                        if analysis:
                            analysis_results[track_id].append(analysis[0])
                    except Exception as e:
                        print(f"분석 중 오류 (ID {track_id}): {e}")
        last_analysis_time = current_time

    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 결과 표시
    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                if track_id in analysis_results and analysis_results[track_id]:
                    # DeepFace 최신 얼굴 정보
                    face_info = analysis_results[track_id][-1]
                    fx, fy, fw, fh = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']

                    # 얼굴 ROI 좌표 보정
                    height, width, _ = frame.shape
                    face_x1 = max(0, x1 + fx)
                    face_y1 = max(0, y1 + fy)
                    face_x2 = min(width, face_x1 + fw)
                    face_y2 = min(height, face_y1 + fh)

                    if face_x2 > face_x1 and face_y2 > face_y1:
                        face_roi = frame[face_y1:face_y2, face_x1:face_x2]

                        if face_roi.size > 0:
                            # 1) MediaPipe Iris 분석
                            face_landmarks = analyze_iris(face_roi)
                            if face_landmarks:
                                # 2) 정면 응시 판단
                                h_roi, w_roi, _ = face_roi.shape
                                if is_center_looking(face_landmarks, w_roi, h_roi):
                                    iris_times[track_id] = iris_times.get(track_id, 0) + 1
                                    print(f"ID {track_id}: Gaze center. Total frames: {iris_times[track_id]}")
                                else:
                                    print(f"ID {track_id}: Gaze not center")

                                # +++++++++++++++++++++++++++++++++++++++
                                #   [눈(홍채) 시각화 로직 추가 부분]
                                # +++++++++++++++++++++++++++++++++++++++
                                # 왼눈 홍채(468~471), 오른눈 홍채(473~476)
                                left_iris_indices = [468, 469, 470, 471]
                                right_iris_indices = [473, 474, 475, 476]

                                for idx in left_iris_indices + right_iris_indices:
                                    lm = face_landmarks.landmark[idx]
                                    px = int(lm.x * w_roi)
                                    py = int(lm.y * h_roi)
                                    # 메인 프레임 좌표로 변환
                                    main_x = face_x1 + px
                                    main_y = face_y1 + py
                                    # 초록색 점 그리기
                                    cv2.circle(frame, (main_x, main_y), 2, (0, 255, 0), -1)
                                # +++++++++++++++++++++++++++++++++++++++

                            else:
                                print(f"ID {track_id}: MediaPipe 얼굴 검출 실패.")
                        else:
                            print(f"ID {track_id}: face_roi가 비어 있음.")
                    else:
                        print(f"ID {track_id}: 잘못된 얼굴 좌표.")

                    # 분석 결과 표시
                    age = get_average_age(analysis_results[track_id])
                    gender = get_most_common(analysis_results[track_id], 'gender')
                    gaze_time_sec = iris_times.get(track_id, 0) / fps

                    label = f'ID: {track_id}, Age: {age}, Gender: {gender}'
                    gaze_label = f'Gaze Time: {gaze_time_sec:.1f} sec'

                    # DeepFace 얼굴 바운딩 박스 (빨간색)
                    for face in analysis_results[track_id]:
                        fxx, fyy, fww, fhh = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                        fx1 = max(0, x1 + fxx)
                        fy1 = max(0, y1 + fyy)
                        fx2 = min(width, fx1 + fww)
                        fy2 = min(height, fy1 + fhh)
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

                    # YOLO 바운딩 박스 (녹색)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, gaze_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                else:
                    label = f'ID: {track_id} (Analyzing...)'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                print("ID가 없는 객체 감지됨")

    cv2.imshow('Real-time Tracking with Age, Gender, and Iris', frame)
    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
