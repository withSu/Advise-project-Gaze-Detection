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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 해상도 낮춰 성능 개선
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 창 설정
cv2.namedWindow('Real-time Tracking with Age, Gender, and Gaze', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Tracking with Age, Gender, and Gaze', 640, 480)

# 분석 관련 변수
last_analysis_time = 0           # 마지막 분석 시간
analysis_interval = 5            # 분석 주기(초)
analysis_results = {}            # ID별 분석 결과 누적
gaze_times = {}                  # ID별 응시 시간 누적 (프레임 단위)

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

    # 현재 프레임의 높이와 너비 (좌표 보정용)
    height, width, _ = frame.shape

    # YOLO로 사람(클래스 0) 추적
    results = model.track(frame, classes=[0], persist=True)

    # 일정 시간(analysis_interval)마다 DeepFace로 나이/성별 분석
    current_time = time.time()
    if current_time - last_analysis_time >= analysis_interval:
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id.item())

                    # 새로운 ID라면 초기화
                    if track_id not in analysis_results:
                        analysis_results[track_id] = []

                    # YOLO 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_roi = frame[y1:y2, x1:x2]

                    try:
                        analysis = DeepFace.analyze(
                            img_path=person_roi,
                            actions=['age', 'gender'],
                            enforce_detection=False,   # 얼굴이 작아도 예외 발생 방지
                            detector_backend='opencv'  # OpenCV 백엔드 사용
                        )
                        if analysis:
                            # 결과를 누적
                            analysis_results[track_id].append(analysis[0])
                    except Exception as e:
                        print(f"분석 중 오류 (ID {track_id}): {e}")

        last_analysis_time = current_time

    # FPS 가져오기 (기본값 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 시선 추적 및 결과 표시
    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id.item())

                # YOLO 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # 분석 결과가 있는 경우에만 시선 추적 시도
                if track_id in analysis_results and analysis_results[track_id]:
                    # DeepFace의 최신 얼굴 정보
                    face_info = analysis_results[track_id][-1]
                    fx, fy, fw, fh = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']

                    # DeepFace의 얼굴 바운딩 박스를 YOLO 바운딩 박스 좌표에 합산
                    face_x1 = x1 + fx
                    face_y1 = y1 + fy
                    face_x2 = face_x1 + fw
                    face_y2 = face_y1 + fh

                    # 프레임 범위를 벗어나지 않도록 보정
                    face_x1 = max(0, face_x1)
                    face_y1 = max(0, face_y1)
                    face_x2 = min(width, face_x2)
                    face_y2 = min(height, face_y2)

                    # 유효한 얼굴 영역인지 확인
                    if face_x2 > face_x1 and face_y2 > face_y1:
                        face_roi = frame[face_y1:face_y2, face_x1:face_x2]

                        # face_roi가 비어있거나 너무 작으면 시선 추적 불가
                        if face_roi.size > 0:
                            h_roi, w_roi, _ = face_roi.shape
                            if h_roi > 10 and w_roi > 10:
                                # 시선 추적 중 OpenCV 오류가 발생할 수 있으므로 예외 처리
                                try:
                                    gaze.refresh(face_roi)

                                    # -------------------------
                                    # 1) GazeTracking 결과 시각화 (눈 주변 바운딩 박스)
                                    #    gaze.eye_left, gaze.eye_right는 각각 Eye 객체로,
                                    #    origin과 frame.shape를 통해 실제 좌표를 구할 수 있다.
                                    if gaze.eye_left is not None:
                                        lx, ly = gaze.eye_left.origin  # 눈 영역의 좌상단 (ROI 내부 좌표)
                                        lw = gaze.eye_left.frame.shape[1]
                                        lh = gaze.eye_left.frame.shape[0]
                                        # 메인 프레임 좌표로 변환
                                        left_eye_x1 = face_x1 + lx
                                        left_eye_y1 = face_y1 + ly
                                        left_eye_x2 = left_eye_x1 + lw
                                        left_eye_y2 = left_eye_y1 + lh
                                        # 눈 영역 사각형(하늘색) 표시
                                        cv2.rectangle(frame, (left_eye_x1, left_eye_y1),
                                                      (left_eye_x2, left_eye_y2),
                                                      (255, 255, 0), 2)

                                    if gaze.eye_right is not None:
                                        rx, ry = gaze.eye_right.origin
                                        rw = gaze.eye_right.frame.shape[1]
                                        rh = gaze.eye_right.frame.shape[0]
                                        right_eye_x1 = face_x1 + rx
                                        right_eye_y1 = face_y1 + ry
                                        right_eye_x2 = right_eye_x1 + rw
                                        right_eye_y2 = right_eye_y1 + rh
                                        cv2.rectangle(frame, (right_eye_x1, right_eye_y1),
                                                      (right_eye_x2, right_eye_y2),
                                                      (255, 255, 0), 2)
                                    # -------------------------

                                    # 시선 방향 감지
                                    if gaze.is_center():
                                        gaze_times[track_id] = gaze_times.get(track_id, 0) + 1
                                        print(f"ID {track_id}: Gaze detected, Total frames: {gaze_times[track_id]}")
                                    else:
                                        print(f"ID {track_id}: Gaze not detected")
                                except cv2.error as e:
                                    print(f"OpenCV error in gaze.refresh(): {e}")
                                except Exception as ex:
                                    print(f"Other error in gaze.refresh(): {ex}")
                            else:
                                print(f"ID {track_id}: face_roi가 너무 작아 시선 추적 불가")
                        else:
                            print(f"ID {track_id}: face_roi가 비어 있음, 시선 추적 스킵")
                    else:
                        print(f"ID {track_id}: 잘못된 얼굴 좌표, 시선 추적 스킵")

                # 결과 표시
                if track_id in analysis_results and analysis_results[track_id]:
                    age = get_average_age(analysis_results[track_id])
                    gender = get_most_common(analysis_results[track_id], 'gender')
                    gaze_time_sec = gaze_times.get(track_id, 0) / fps

                    label = f'ID: {track_id}, Age: {age}, Gender: {gender}'
                    gaze_label = f'Gaze Time: {gaze_time_sec:.1f} sec'

                    # DeepFace가 감지한 얼굴 바운딩 박스 (빨간색)
                    for face in analysis_results[track_id]:
                        fx, fy, fw, fh = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                        fx1 = x1 + fx
                        fy1 = y1 + fy
                        fx2 = fx1 + fw
                        fy2 = fy1 + fh

                        # 바운딩 박스도 프레임 범위 내로 보정
                        fx1 = max(0, fx1)
                        fy1 = max(0, fy1)
                        fx2 = min(width, fx2)
                        fy2 = min(height, fy2)

                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

                    # YOLO 바운딩 박스 및 텍스트 (녹색)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, gaze_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    # DeepFace 분석이 아직 끝나지 않은 상태
                    label = f'ID: {track_id} (Analyzing...)'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("ID가 없는 객체가 감지되었습니다. (새로운 객체이거나 추적 실패)")

    # 화면에 표시
    cv2.imshow('Real-time Tracking with Age, Gender, and Gaze', frame)
    if cv2.waitKey(1) == 27:  # Esc 키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
