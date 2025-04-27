#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 기반 사용자 특성 추적 및 시선 분석 시스템 (AdVise 프로젝트)

이 모듈은 실시간 카메라 영상에서 사용자를 감지하고, 해당 사용자의 특성(나이, 성별, 감정 등)과
시선 응시 시간을 분석합니다. 분석 결과는 JSON 파일로 저장되어 광고 추천 시스템에서 활용됩니다.

Authors: [Your Names]
Version: 1.0.0
"""

import cv2
import time
import os
import json
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
from deepface import DeepFace
import mediapipe as mp
import argparse
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CameraTracker")

# 결과 저장 경로 설정
PROJECT_ROOT = "/home/a/A_2025/AdVise-ML/graduate_project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_PATH = os.path.join(DATA_DIR, "user_features.json")
os.makedirs(DATA_DIR, exist_ok=True)

# 사용자 속성 변환 매핑
AGE_TO_CATEGORY = {
    (0, 20): "20세미만",
    (20, 30): "20-30세",
    (31, 40): "31-40세",
    (41, 50): "41-50세",
    (51, 60): "51-60세",
    (61, 100): "51-60세"  # 60세 이상도 51-60세로 매핑
}

class CameraTracker:
    """
    카메라 기반 사용자 추적 및 특성 분석 클래스
    
    이 클래스는 YOLO로 객체 감지, DeepFace로 얼굴 특성 분석, MediaPipe로 시선 추적을
    수행하여 사용자의 특성과 응시 시간을 측정합니다.
    """
    def __init__(self, yolo_model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        CameraTracker 초기화
        
        Args:
            yolo_model_path (str): YOLO 모델 파일 경로
            confidence_threshold (float): 객체 감지 신뢰도 임계값
        """
        logger.info("CameraTracker 초기화 중...")
        
        # YOLO 모델 로드
        try:
            self.model = YOLO(yolo_model_path)
            logger.info(f"YOLO 모델 로드 성공: {yolo_model_path}")
        except Exception as e:
            logger.error(f"YOLO 모델 로드 실패: {e}")
            raise
        
        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,         # 홍채(iris) 추적 활성화
            max_num_faces=1,               # ROI 하나당 한 명 처리
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # 웹캠 초기화
        self.cap = None
        
        # 분석 관련 변수
        self.last_analysis_time = 0
        self.analysis_interval = 3  # 3초마다 DeepFace 분석
        self.analysis_results = {}  # ID별 DeepFace 분석 결과 누적
        self.iris_times = {}        # ID별 '카메라 응시' 시간 (프레임 단위)
        self.confidence_threshold = confidence_threshold
        
        # 사용자 특성 결과 저장 
        self.user_data = []
        self.max_history = 100  # 최대 저장 데이터 수
        self.result_queue = deque(maxlen=self.max_history)
        
        # 이전 결과 로드 (있을 경우)
        self.load_previous_results()
        
        logger.info("CameraTracker 초기화 완료")
    
    def load_previous_results(self):
        """이전에 저장된 결과 로드"""
        if os.path.exists(RESULT_PATH):
            try:
                with open(RESULT_PATH, 'r', encoding='utf-8') as f:
                    self.user_data = json.load(f)
                    # 최신 데이터만 큐에 추가
                    recent_data = self.user_data[-self.max_history:] if len(self.user_data) > self.max_history else self.user_data
                    self.result_queue.extend(recent_data)
                logger.info(f"{len(self.user_data)}개의 이전 데이터를 로드했습니다.")
            except Exception as e:
                logger.error(f"이전 데이터 로드 중 오류: {e}")
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        try:
            # 새로운 결과와 이전 결과 통합
            with open(RESULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.user_data, f, ensure_ascii=False, indent=2)
            logger.info(f"{len(self.user_data)}개의 데이터가 {RESULT_PATH}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
    
    def initialize_camera(self, camera_id=0, width=800, height=600):
        """카메라 초기화"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            logger.error("카메라를 열 수 없습니다!")
            raise ValueError("카메라를 열 수 없습니다!")
        
        logger.info(f"카메라 초기화 완료 (ID: {camera_id}, {width}x{height})")
        return self.cap.isOpened()
    
    def release_camera(self):
        """카메라 자원 해제"""
        if self.cap is not None:
            self.cap.release()
            logger.info("카메라 자원 해제 완료")
    
    def get_average_age(self, result_list):
        """분석 결과 중 나이(age)의 평균 반환"""
        if not result_list:
            return None
        ages = [r['age'] for r in result_list]
        return int(sum(ages) / len(ages))

    def get_most_common(self, result_list, key):
        """DeepFace 분석 결과의 누적 리스트에서 gender/emotion 등 최빈값 추출"""
        if not result_list:
            return None

        if key in ('gender', 'emotion'):
            # gender, emotion은 딕셔너리 -> 가장 확률 높은 항목만 추출
            values = [max(r[key], key=r[key].get) for r in result_list]
        else:
            values = [r[key] for r in result_list]

        return Counter(values).most_common(1)[0][0]

    def denormalize_landmark(self, landmark, width, height):
        """0~1 정규화된 랜드마크를 실제 픽셀 좌표로 변환"""
        return int(landmark.x * width), int(landmark.y * height)

    def analyze_iris(self, face_roi):
        """MediaPipe Iris로 face_roi 분석 -> 홍채(눈) 랜드마크"""
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_face)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def is_center_looking(self, face_landmarks, width, height):
        """
        시선이 중앙을 향하고 있는지 확인
        
        왼쪽 눈 홍채 중심(468)을 확인해 0.4~0.6 사이면 '정면 응시'로 간주
        
        Args:
            face_landmarks: MediaPipe FaceMesh 랜드마크
            width, height: 얼굴 ROI의 너비와 높이
            
        Returns:
            bool: 정면 응시 여부
        """
        # 눈 랜드마크 인덱스
        left_iris_idx = 468  # 왼쪽 눈 홍채 중심
        left_corner_idx = 33  # 왼쪽 눈 왼쪽 코너
        right_corner_idx = 133  # 왼쪽 눈 오른쪽 코너

        # 좌표 계산
        px, py = self.denormalize_landmark(face_landmarks.landmark[left_iris_idx], width, height)
        lx, ly = self.denormalize_landmark(face_landmarks.landmark[left_corner_idx], width, height)
        rx, ry = self.denormalize_landmark(face_landmarks.landmark[right_corner_idx], width, height)

        if rx != lx:  # 0 나눗셈 방지
            ratio = (px - lx) / float(rx - lx)  # 0.0 ~ 1.0
            # 0.4 ~ 0.6 범위인 경우 정면 응시로 판단
            if 0.4 < ratio < 0.6:
                return True
        return False

    def draw_iris_points(self, face_landmarks, face_x1, face_y1, w_roi, h_roi, frame):
        """왼눈 홍채, 오른눈 홍채 인덱스 위치에 점 표시"""
        # 홍채 랜드마크 인덱스
        left_iris_indices = [468, 469, 470, 471]  # 왼쪽 눈 홍채
        right_iris_indices = [473, 474, 475, 476]  # 오른쪽 눈 홍채

        # 각 랜드마크에 빨간색 점 그리기
        for idx in left_iris_indices + right_iris_indices:
            px, py = self.denormalize_landmark(face_landmarks.landmark[idx], w_roi, h_roi)
            main_x = face_x1 + px
            main_y = face_y1 + py
            cv2.circle(frame, (main_x, main_y), 2, (0, 0, 255), -1)
    
    def age_to_category_map(self, age):
        """나이를 카테고리로 변환"""
        for age_range, category in AGE_TO_CATEGORY.items():
            if age_range[0] <= age <= age_range[1]:
                return category
        return "20세미만"  # 기본값
    
    def process_frame(self, frame=None, display_results=True):
        """
        프레임을 처리하고 사용자 특성 추출
        
        Args:
            frame: 처리할 비디오 프레임 (None이면 카메라에서 획득)
            display_results: 결과를 화면에 표시할지 여부
            
        Returns:
            tuple: (프레임, 사용자 특성 리스트, 응시 시간)
        """
        # 카메라에서 프레임 읽기
        if frame is None:
            if self.cap is None:
                logger.error("카메라가 초기화되지 않았습니다!")
                raise ValueError("카메라가 초기화되지 않았습니다!")
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("카메라에서 프레임을 읽을 수 없습니다.")
                return None, [], {}
        
        # YOLO로 사람 추적
        try:
            results = self.model.track(frame, classes=[0], persist=True)
        except Exception as e:
            logger.error(f"YOLO 추적 중 오류: {e}")
            return frame, [], {}
        
        # 현재 시간
        current_time = time.time()
        
        user_features = []
        gaze_times = {}
        
        # 결과 처리
        for result in results:
            # 현재 프레임에서 감지된 사람 박스들을 수집
            boxes_data = []
            for box in result.boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    boxes_data.append((track_id, area, x1, y1, x2, y2))
            
            # 면적 기준으로 내림차순 정렬 -> 상위 3명만 추출
            boxes_data.sort(key=lambda x: x[1], reverse=True)
            top_3 = boxes_data[:3]
            
            # 3초마다 DeepFace 분석 (나이/성별/감정)
            if current_time - self.last_analysis_time >= self.analysis_interval:
                for (track_id, area, x1, y1, x2, y2) in top_3:
                    if track_id not in self.analysis_results:
                        self.analysis_results[track_id] = []
                    
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size == 0:
                        continue
                        
                    try:
                        analysis = DeepFace.analyze(
                            img_path=person_roi,
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False,
                            detector_backend='opencv'
                        )
                        if analysis:
                            self.analysis_results[track_id].append(analysis[0])
                            logger.debug(f"ID {track_id}: DeepFace 분석 성공")
                    except Exception as e:
                        logger.debug(f"분석 중 오류 (ID {track_id}): {e}")
                
                self.last_analysis_time = current_time
            
            # 매 프레임마다 MediaPipe Iris + 시선 추적
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            for (track_id, area, x1, y1, x2, y2) in top_3:
                # YOLO 바운딩 박스 (초록색)
                if display_results:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                
                # DeepFace 분석 결과 있는지 확인
                if track_id in self.analysis_results and self.analysis_results[track_id]:
                    # DeepFace 최신 결과
                    face_info = self.analysis_results[track_id][-1]
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
                    
                    gaze_detected = False
                    if face_roi is not None and face_roi.size > 0:
                        # MediaPipe Iris
                        face_landmarks = self.analyze_iris(face_roi)
                        if face_landmarks:
                            h_roi, w_roi, _ = face_roi.shape
                            # 시선 추적
                            if self.is_center_looking(face_landmarks, w_roi, h_roi):
                                self.iris_times[track_id] = self.iris_times.get(track_id, 0) + 1
                                gaze_detected = True
                                if display_results:
                                    logger.debug(f"ID {track_id}: Gaze center -> frames: {self.iris_times[track_id]}")
                            
                            # 홍채(동공) 점 찍기
                            if display_results:
                                self.draw_iris_points(face_landmarks, face_x1, face_y1, w_roi, h_roi, frame)
                    
                    # 나이·성별·감정·응시시간 추출
                    age = self.get_average_age(self.analysis_results[track_id])
                    gender = self.get_most_common(self.analysis_results[track_id], 'gender')
                    emotion = self.get_most_common(self.analysis_results[track_id], 'emotion')
                    gaze_time_sec = self.iris_times.get(track_id, 0) / fps
                    
                    # 성별 매핑
                    if gender == "Man" or gender == "man":
                        gender_ko = "남성"
                    else:
                        gender_ko = "여성"
                    
                    # 현재 시간 결정
                    current_hour = datetime.now().hour
                    time_period = "오전" if 6 <= current_hour < 12 else "오후"
                    
                    # 계절 결정 (현재 월 기준)
                    current_month = datetime.now().month
                    if 3 <= current_month <= 5:
                        season = "봄"
                    elif 6 <= current_month <= 8:
                        season = "여름"
                    elif 9 <= current_month <= 11:
                        season = "가을"
                    else:
                        season = "겨울"
                    
                    # 사용자 특성 저장
                    user_feature = {
                        "id": track_id,
                        "age": self.age_to_category_map(age),
                        "gender": gender_ko,
                        "emotion": emotion.lower(),  # 소문자로 통일
                        "time": time_period,
                        "weather": season,
                        "gaze_time": gaze_time_sec,
                        "timestamp": current_time
                    }
                    user_features.append(user_feature)
                    
                    # 응시 시간 저장
                    gaze_times[track_id] = gaze_time_sec
                    
                    # 유의미한 응시 시간이 있으면 결과 큐에 추가 (0.5초 이상)
                    if gaze_time_sec >= 0.5:
                        self.result_queue.append(user_feature)
                        # 전체 데이터 목록에도 추가
                        self.user_data.append(user_feature)
                        logger.info(f"ID {track_id}: 사용자 데이터 추가됨 (응시 시간: {gaze_time_sec:.2f}초)")
                    
                    if display_results:
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
                    if display_results:
                        label = f'ID: {track_id} (Analyzing...)'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        return frame, user_features, gaze_times
    
    def get_latest_features(self, max_entries=10):
        """
        최신 사용자 특성 데이터 반환
        
        Args:
            max_entries: 반환할 최대 항목 수
            
        Returns:
            list: 최신 사용자 특성 데이터 목록
        """
        return list(self.result_queue)[-max_entries:]

def main():
    """카메라 추적 시스템 메인 함수"""
    parser = argparse.ArgumentParser(description="카메라 기반 사용자 특성 및 시선 추적 시스템")
    parser.add_argument('--camera', type=int, default=0, help='카메라 ID (기본값: 0)')
    parser.add_argument('--width', type=int, default=800, help='화면 너비 (기본값: 800)')
    parser.add_argument('--height', type=int, default=600, help='화면 높이 (기본값: 600)')
    parser.add_argument('--save_interval', type=int, default=50, help='결과 저장 간격 (프레임 단위, 기본값: 50)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO 모델 경로 (기본값: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5, help='감지 신뢰도 임계값 (기본값: 0.5)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    # 디버그 모드일 경우 로깅 레벨 변경
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    tracker = CameraTracker(yolo_model_path=args.model, confidence_threshold=args.confidence)
    tracker.initialize_camera(camera_id=args.camera, width=args.width, height=args.height)
    
    # 창 설정
    window_name = 'AdVise: User Feature Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_to_display = 0
    
    try:
        logger.info("카메라 추적 시스템 시작")
        while True:
            # FPS 계산
            if time.time() - start_time >= 1:
                fps_to_display = fps_counter
                fps_counter = 0
                start_time = time.time()
            
            frame, user_features, gaze_times = tracker.process_frame(display_results=True)
            fps_counter += 1
            
            if frame is None:
                logger.error("카메라 프레임을 읽을 수 없습니다.")
                break
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {fps_to_display}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow(window_name, frame)
            
            # 주기적으로 결과 저장
            frame_count += 1
            if frame_count % args.save_interval == 0:
                tracker.save_results()
                logger.debug(f"{frame_count}번째 프레임에서 결과 저장")
            
            # ESC 키를 누르면 종료
            if cv2.waitKey(1) == 27:
                logger.info("ESC 키 입력으로 프로그램 종료")
                break
    
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트로 프로그램 중단")
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    
    finally:
        # 최종 결과 저장
        tracker.save_results()
        
        # 자원 해제
        tracker.release_camera()
        cv2.destroyAllWindows()
        
        logger.info("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()