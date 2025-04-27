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
    max_num_faces=1,               # 동시에 추적할 얼굴 개수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age, Gender, and Iris', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age, Gender, and Iris', 640, 480)

# 분석 관련 변수
last_analysis_time = 0
analysis_interval = 5
analysis_results = {}   # ID별 DeepFace 분석 결과 누적
iris_times = {}         # ID별 '카메라 응시' 시간 (프레임 단위)

def get_average_age(result_list):
    """분석 결과 중 나이(age)의 평균 반환."""
    if not result_list:
        return None
    ages = [r['age'] for r in result_list]
    return int(sum(ages) / len(ages))

def get_most_common(result_list, key):
    """
    DeepFace 분석 결과의 누적 리스트에서
    gender/emotion 등을 가장 확률 높은 값으로 추출해 반환한다.
    """
    if not result_list:
        return None

    if key in ('gender', 'emotion'):
        # gender, emotion은 딕셔너리 형태 -> 가장 확률 높은 값 뽑음
        values = [max(r[key], key=r[key].get) for r in result_list]
    else:
        # 나머지는 단순 값
        values = [r[key] for r in result_list]

    return Counter(values).most_common(1)[0][0]

def denormalize_landmark(landmark, width, height):
    """0~1 정규화된 랜드마크를 실제 픽셀 좌표로 변환."""
    return int(landmark.x * width), int(landmark.y * height)

def analyze_iris(face_roi):
    """MediaPipe Iris로 face_roi 분석 -> 홍채(눈) 랜드마크."""
    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_face)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]

def is_center_looking(face_landmarks, width, height):
    """
    왼쪽 눈 홍채 중심(468)만 확인해 (0.4~0.6) 범위면 '정면 응시'로 간주 (단순화).
    """
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

def draw_iris_points(face_landmarks, face_x1, face_y1, w_roi, h_roi, frame):
    """
    왼눈 홍채(468~471), 오른눈 홍채(473~476) 인덱스 위치에 빨간색 점을 찍는다.
    (BGR: (0,0,255) → 빨간색)
    """
    left_iris_indices = [468, 469, 470, 471]
    right_iris_indices = [473, 474, 475, 476]

    for idx in left_iris_indices + right_iris_indices:
        px, py = denormalize_landmark(face_landmarks.landmark[idx], w_roi, h_roi)
        main_x = face_x1 + px
        main_y = face_y1 + py
        # 빨간색 점
        cv2.circle(frame, (main_x, main_y), 2, (0, 0, 255), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # YOLO로 사람 추적
    results = model.track(frame, classes=[0], persist=True)

    # 5초마다 DeepFace로 나이/성별/감정 분석
    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    if track_id not in analysis_results:
                        analysis_results[track_id] = []

                    # YOLO 박스 → person_roi
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_roi = frame[y1:y2, x1:x2]

                    try:
                        # 감정(emotion)도 분석
                        analysis = DeepFace.analyze(
                            img_path=person_roi,
                            actions=['age', 'gender', 'emotion'],
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
            if box.id is None:
                print("ID가 없는 객체 감지됨")
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # YOLO 바운딩 박스 (초록색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if track_id in analysis_results and analysis_results[track_id]:
                # DeepFace 최신 결과만 사용 -> [-1]
                face_info = analysis_results[track_id][-1]
                fx, fy, fw, fh = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']

                # ROI 좌표 보정
                height, width, _ = frame.shape
                face_x1 = max(0, x1 + fx)
                face_y1 = max(0, y1 + fy)
                face_x2 = min(width, face_x1 + fw)
                face_y2 = min(height, face_y1 + fh)

                if face_x2 > face_x1 and face_y2 > face_y1:
                    face_roi = frame[face_y1:face_y2, face_x1:face_x2]
                    if face_roi.size > 0:
                        face_landmarks = analyze_iris(face_roi)
                        if face_landmarks:
                            # 정면 응시 판단
                            h_roi, w_roi, _ = face_roi.shape
                            if is_center_looking(face_landmarks, w_roi, h_roi):
                                iris_times[track_id] = iris_times.get(track_id, 0) + 1
                                print(f"ID {track_id}: Gaze center -> frames: {iris_times[track_id]}")
                            else:
                                print(f"ID {track_id}: Gaze not center")

                            # MediaPipe Iris (홍채) 점(빨간색)
                            draw_iris_points(face_landmarks, face_x1, face_y1, w_roi, h_roi, frame)

                        else:
                            print(f"ID {track_id}: MediaPipe 얼굴 검출 실패.")
                    else:
                        print(f"ID {track_id}: face_roi가 비어 있음.")
                else:
                    print(f"ID {track_id}: 잘못된 얼굴 좌표.")

                # ====== DeepFace 분석 결과 (나이, 성별, 감정) ======
                age = get_average_age(analysis_results[track_id])
                gender = get_most_common(analysis_results[track_id], 'gender')
                emotion = get_most_common(analysis_results[track_id], 'emotion')
                gaze_time_sec = iris_times.get(track_id, 0) / fps

                # ID 표시 (기존: 초록색 그대로 사용)
                id_label = f"ID: {track_id}"
                cv2.putText(frame, id_label,
                            (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # 나이, 성별, 감정 → 여기서는 파란색으로 변경
                info_label = f"Age: {age}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, info_label,
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)  # 파란색 텍스트

                # 응시 시간 → 빨간색으로 변경 (원한다면)
                gaze_label = f"Gaze Time: {gaze_time_sec:.1f} sec"
                cv2.putText(frame, gaze_label,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

                # DeepFace 얼굴 바운딩박스 (파란색)
                fx1 = max(0, x1 + fx)
                fy1 = max(0, y1 + fy)
                fx2 = min(width, fx1 + fw)
                fy2 = min(height, fy1 + fh)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255,0,0), 2)

            else:
                # 아직 DeepFace 분석이 없을 때
                label = f'ID: {track_id} (Analyzing...)'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Real-time Tracking with Age, Gender, and Iris', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
