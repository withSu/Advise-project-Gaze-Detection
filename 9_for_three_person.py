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
    max_num_faces=1,               # ROI 하나당 한 명 처리
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
    gender/emotion 등 최빈값(가장 높은 확률)을 추출.
    """
    if not result_list:
        return None

    if key in ('gender', 'emotion'):
        # gender, emotion은 딕셔너리 -> 가장 확률 높은 항목만 추출
        values = [max(r[key], key=r[key].get) for r in result_list]
    else:
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
    왼쪽 눈 홍채 중심(468)만 확인해 0.4~0.6 사이면 '정면 응시'로 간주.
    """
    left_iris_idx = 468
    left_corner_idx = 33
    right_corner_idx = 133

    px, py = denormalize_landmark(face_landmarks.landmark[left_iris_idx], width, height)
    lx, ly = denormalize_landmark(face_landmarks.landmark[left_corner_idx], width, height)
    rx, ry = denormalize_landmark(face_landmarks.landmark[right_corner_idx], width, height)

    if rx != lx:  # 0 나눗셈 방지
        ratio = (px - lx) / float(rx - lx)  # 0.0 ~ 1.0
        if 0.4 < ratio < 0.6:
            return True
    return False

def draw_iris_points(face_landmarks, face_x1, face_y1, w_roi, h_roi, frame):
    """
    왼눈 홍채(468~471), 오른눈 홍채(473~476) 인덱스 위치에 빨간색 점 표시.
    (BGR: (0,0,255))
    """
    left_iris_indices = [468, 469, 470, 471]
    right_iris_indices = [473, 474, 475, 476]

    for idx in left_iris_indices + right_iris_indices:
        px, py = denormalize_landmark(face_landmarks.landmark[idx], w_roi, h_roi)
        main_x = face_x1 + px
        main_y = face_y1 + py
        cv2.circle(frame, (main_x, main_y), 2, (0, 0, 255), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 스트림을 읽을 수 없습니다.")
        break

    # YOLO로 사람 추적
    results = model.track(frame, classes=[0], persist=True)

    # 보통 한 프레임당 results는 1개일 가능성이 큼(ultralytics 구조상)
    # 여러 프레임이 동시에 들어오면 여러 result가 생길 수 있으므로 루프
    for result in results:
        # 1) 현재 프레임에서 감지된 사람 박스들을 수집
        boxes_data = []
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w = x2 - x1
                h = y2 - y1
                area = w * h
                boxes_data.append((track_id, area, x1, y1, x2, y2))

        # 2) 면적 기준으로 내림차순 정렬 -> 상위 3명만 추출
        boxes_data.sort(key=lambda x: x[1], reverse=True)
        top_3 = boxes_data[:3]

        # 3) 5초마다 DeepFace 분석 (나이/성별/감정)
        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval:
            for (track_id, area, x1, y1, x2, y2) in top_3:
                if track_id not in analysis_results:
                    analysis_results[track_id] = []

                person_roi = frame[y1:y2, x1:x2]
                try:
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

        # 4) 매 프레임마다 MediaPipe Iris + 시선 추적
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        for (track_id, area, x1, y1, x2, y2) in top_3:
            # YOLO 바운딩 박스 (초록색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # DeepFace 분석 결과 있는지 확인
            if track_id in analysis_results and analysis_results[track_id]:
                # DeepFace 최신 결과
                face_info = analysis_results[track_id][-1]
                fx, fy, fw, fh = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']

                # ROI 좌표
                H, W, _ = frame.shape
                face_x1 = max(0, x1 + fx)
                face_y1 = max(0, y1 + fy)
                face_x2 = min(W, face_x1 + fw)
                face_y2 = min(H, face_y1 + fh)

                face_roi = None
                if face_x2>face_x1 and face_y2>face_y1:
                    face_roi = frame[face_y1:face_y2, face_x1:face_x2]

                if face_roi is not None and face_roi.size > 0:
                    # MediaPipe Iris
                    face_landmarks = analyze_iris(face_roi)
                    if face_landmarks:
                        h_roi, w_roi, _ = face_roi.shape
                        # 시선 추적
                        if is_center_looking(face_landmarks, w_roi, h_roi):
                            iris_times[track_id] = iris_times.get(track_id, 0) + 1
                            print(f"ID {track_id}: Gaze center -> frames: {iris_times[track_id]}")
                        else:
                            print(f"ID {track_id}: Gaze not center")

                        # 홍채(동공) 점 찍기
                        draw_iris_points(face_landmarks, face_x1, face_y1, w_roi, h_roi, frame)
                    else:
                        print(f"ID {track_id}: MediaPipe 얼굴 검출 실패.")
                else:
                    print(f"ID {track_id}: face_roi가 비어있거나 잘못된 좌표.")

                # 나이·성별·감정·응시시간 표시
                age = get_average_age(analysis_results[track_id])
                gender = get_most_common(analysis_results[track_id], 'gender')
                emotion = get_most_common(analysis_results[track_id], 'emotion')
                gaze_time_sec = iris_times.get(track_id, 0) / fps

                # ID (초록)
                id_label = f"ID: {track_id}"
                cv2.putText(frame, id_label,
                            (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # 나이, 성별, 감정 (파란)
                info_label = f"Age: {age}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, info_label,
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)

                # 응시시간 (빨간)
                gaze_label = f"Gaze Time: {gaze_time_sec:.1f} sec"
                cv2.putText(frame, gaze_label,
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

                # DeepFace 얼굴 박스 (파란)
                fx1 = max(0, x1 + fx)
                fy1 = max(0, y1 + fy)
                fx2 = min(W, fx1 + fw)
                fy2 = min(H, fy1 + fh)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255,0,0), 2)
            else:
                # 아직 DeepFace 분석 결과가 없음
                label = f'ID: {track_id} (Analyzing...)'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Real-time Tracking with Age, Gender, and Iris', frame)
    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
