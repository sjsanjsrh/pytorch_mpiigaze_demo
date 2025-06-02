import datetime
import logging
import pathlib
from typing import Optional
import pyautogui
import time

import cv2
import numpy as np
from omegaconf import DictConfig

from .common import Face, FacePartsName, Visualizer
from .gaze_estimator import GazeEstimator
from .utils import get_3d_face_model
from .point.calibration import Calibration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model
        self.show_calibration = False

        self.calibration = None
        self.fullscimg = None

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)
            self._display_calibration(face)


        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        elif key == ord('c'):
            self.show_calibration = not self.show_calibration
            pass
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

    def _display_calibration(self, face: Face) -> None:
        if not self.show_calibration:
            return

        if self.calibration is None:
            self.calibration = Calibration(
                self.gaze_estimator,
                pyautogui.size().width,
                pyautogui.size().height,
                process_noise=1e-9
            )
            self.calibration_start_time = time.time()
            self.current_calibration_index = -1
            self.collected_points = [[],[],[],[]]
            self.fullscimg = np.zeros((self.calibration.screen_height, self.calibration.screen_width, 3), dtype=np.uint8)

        # 캘리브레이션 점들 정의
        screen_width = self.calibration.screen_width
        screen_height = self.calibration.screen_height
        calibration_points = [
            np.array((screen_width / 10, screen_height / 10), dtype=np.float32),
            np.array((screen_width * 9 / 10, screen_height / 10), dtype=np.float32),
            np.array((screen_width * 9 / 10, screen_height * 9 / 10), dtype=np.float32),
            np.array((screen_width / 10, screen_height * 9 / 10), dtype=np.float32)
        ]
        
        # if True: # for debugging
        #     img = np.zeros((self.calibration.screen_height, self.calibration.screen_width, 3), dtype=np.uint8)
        #     cx = int(self.calibration.screen_width / 2)
        #     cy = int(self.calibration.screen_height / 2)
        #     eye_vector = self.calibration.calc_eye_2d_vector(face) * 1000
        #     cv2.line(img, (cx, cy), (cx+int(eye_vector[0]), cy+int(eye_vector[1])), (255, 0, 0), 2)
        #     cv2.circle(img, (cx+int(eye_vector[0]), cy+int(eye_vector[1])), 5, (0, 255, 0), -1)
        #     cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
        #     cv2.setWindowProperty("calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        #     cv2.imshow('calibration', img)
        #     return
        
        if self.current_calibration_index >= len(calibration_points):
            points = []
            for e in self.collected_points:
                points.append(self.calibration.calc_filtered_centers(e))
            self.calibration.calc_trs_matrix(calibration_points, points)
            self.calibration.calc_noize(points, self.collected_points, 5)

            # 계산된 점 표시
            # 점점 연하게 하기 위해서 self.fullscimg에서 빼기
            fade_factor = 0.02
            self.fullscimg = cv2.addWeighted(self.fullscimg, 1 - fade_factor, np.zeros_like(self.fullscimg), fade_factor, 0)
            
            centerd_point = self.calibration.calc_eye_2d_vector(face)
            point = self.calibration.calc_trs_transform(centerd_point)

            k_point = self.calibration.calc_filtered_point(centerd_point)
            k_point = self.calibration.calc_trs_transform(k_point)

            cv2.circle(self.fullscimg, (int(k_point[0]), int(k_point[1])), 10, (255, 0, 0), -1)
            cv2.circle(self.fullscimg, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            
            cv2.putText(self.fullscimg, "Noize: "+str(self.calibration.filter.R), (0, int(self.calibration.screen_height) - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(self.fullscimg, str(self.calibration.trs_matrix), (0,int(self.calibration.screen_height)-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow('calibration', self.fullscimg)

        elif self.current_calibration_index == -1:
            # n초 대기 후 첫 번째 점으로 이동
            n = 3
            dt = time.time() - self.calibration_start_time
            img = np.zeros((self.calibration.screen_height, self.calibration.screen_width, 3), dtype=np.uint8)
            cv2.putText(img, "Wait "+str(int(n - dt + .5))+"sec", (int(self.calibration.screen_width / 2) - 50, int(self.calibration.screen_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow('calibration', img)

            if dt >= n:
                self.current_calibration_index = 0
                self.calibration_start_time = time.time()

        else:
            # 현재 점 표시
            current_point = calibration_points[self.current_calibration_index]
            img = np.zeros((self.calibration.screen_height, self.calibration.screen_width, 3), dtype=np.uint8)
            
            # 점을 그리기
            if time.time() - self.calibration_start_time >= 0.5:
                cv2.circle(img, (int(current_point[0]), int(current_point[1])), 20, (0, 255, 255), -1)
            if time.time() - self.calibration_start_time >= 1.5:
                cv2.circle(img, (int(current_point[0]), int(current_point[1])), 20, (0, 255, 0), -1)
            for i in range(len(calibration_points)):
                cv2.putText(img, f"{i+1}", 
                            (int(calibration_points[i][0]) - 10, int(calibration_points[i][1] + 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            
            cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('calibration', img)

            if time.time() - self.calibration_start_time >= 2:
                # 현재 얼굴의 중앙점 추가
                eye_vector = self.calibration.calc_eye_2d_vector(face)
                self.collected_points[self.current_calibration_index].append(eye_vector)
            
            # 다음 점으로 이동
            if time.time() - self.calibration_start_time >= 4:

                # 다음 점으로 이동
                self.current_calibration_index += 1
                self.calibration_start_time = time.time()