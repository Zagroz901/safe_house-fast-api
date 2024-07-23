import numpy as np
import cv2
from typing import Tuple, Optional, List, Sequence
import os

Point = Tuple[float, float]

class Outlier(Exception):
    pass

class FeatureMatching:
    
    def __init__(self, template_folder: str):
        self.f_extractor = cv2.SIFT_create()
        self.templates = self.load_templates(template_folder)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=25)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.last_hinv = np.zeros((3, 3))
        self.max_error_hinv = 500.
        self.num_frames_no_success = 0
        self.max_frames_no_success = 20

    def load_templates(self, folder: str) -> List[Tuple[np.ndarray, List[cv2.KeyPoint], np.ndarray]]:
        templates = []
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            keypoints, descriptors = self.f_extractor.detectAndCompute(img, None)
            if keypoints is None or descriptors is None:
                print("No keypoints or descriptors found.")
            templates.append((img, keypoints, descriptors))
        return templates

    def match(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        img_query = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh_query = img_query.shape

        key_query = self.f_extractor.detect(img_query)
        key_query, desc_query = self.f_extractor.compute(img_query, key_query)

        best_match = None
        best_warped = None
        best_flann = None

        for template, key_train, desc_train in self.templates:
            good_matches = self.match_features(desc_query, desc_train)
            print(f"Number of good matches: {len(good_matches)}")

            train_points = [key_train[match.queryIdx].pt for match in good_matches]
            query_points = [key_query[match.trainIdx].pt for match in good_matches]

            try:
                if len(good_matches) < 4:
                    raise Outlier("Too few matches")

                dst_corners = detect_corner_points(train_points, query_points, template.shape[:2])

                if np.any((dst_corners < -20) | (dst_corners > np.array(sh_query) + 20)):
                    raise Outlier("Out of image")

                area = calculate_quadrilateral_area(dst_corners)
                if not np.prod(sh_query) / 16. < area < np.prod(sh_query) / 2.:
                    raise Outlier("Area is unreasonably small or large")

                train_points_scaled = self.scale_and_offset(train_points, template.shape[:2], sh_query)
                Hinv, _ = cv2.findHomography(np.array(query_points), np.array(train_points_scaled), cv2.RANSAC)
                similar = np.linalg.norm(Hinv - self.last_hinv) < self.max_error_hinv
                recent = self.num_frames_no_success < self.max_frames_no_success
                if recent and not similar:
                    raise Outlier("Not similar transformation")

            except Outlier:
                continue

            else:
                self.num_frames_no_success = 0
                self.last_hinv = Hinv

                img_warped = cv2.warpPerspective(img_query, Hinv, (sh_query[1], sh_query[0]))
                img_flann = draw_good_matches(template, key_train, img_query, key_query, good_matches)

                dst_corners[:, 0] += template.shape[1]
                cv2.polylines(img_flann, [dst_corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)
                
                if best_match is None or len(good_matches) > len(best_match):
                    best_match = good_matches
                    best_warped = img_warped
                    best_flann = img_flann

        if best_match is not None:
            return True, best_warped, best_flann
        else:
            self.num_frames_no_success += 1
            return False, None, None

    def match_features(self, desc_frame: np.ndarray, desc_train: np.ndarray) -> List[cv2.DMatch]:
        matches = self.flann.knnMatch(desc_train, desc_frame, k=2)
        good_matches = [x[0] for x in matches if x[0].distance < 0.90 * x[1].distance]
        return good_matches

    @staticmethod
    def scale_and_offset(points: Sequence[Point], source_size: Tuple[int, int], dst_size: Tuple[int, int], factor: float = 0.5) -> List[Point]:
        dst_size = np.array(dst_size)
        scale = 1 / np.array(source_size) * dst_size * factor
        bias = dst_size * (1 - factor) / 2
        return [tuple(np.array(pt) * scale + bias) for pt in points]

def detect_corner_points(src_points: Sequence[Point], dst_points: Sequence[Point], sh_src: Tuple[int, int]) -> np.ndarray:
    H, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
    if H is None:
        raise Outlier("Homography not found")
    height, width = sh_src
    src_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)
    return cv2.perspectiveTransform(src_corners[None, :, :], H)[0]

def calculate_quadrilateral_area(corners: np.ndarray) -> float:
    area = 0
    for prev, nxt in zip(corners, np.roll(corners, -1, axis=0)):
        area += (prev[0] * nxt[1] - prev[1] * nxt[0]) / 2.
    return area

def draw_good_matches(img1: np.ndarray, kp1: Sequence[cv2.KeyPoint], img2: np.ndarray, kp2: Sequence[cv2.KeyPoint], matches: Sequence[cv2.DMatch]) -> np.ndarray:
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1[..., None]
    out[:rows2, cols1:cols1 + cols2, :] = img2[..., None]

    for match in matches:
        c1 = tuple(map(int, kp1[match.queryIdx].pt))
        c2 = tuple(map(int, kp2[match.trainIdx].pt))
        c2 = c2[0] + cols1, c2[1]

        radius = 4
        BLUE = (255, 0, 0)
        thickness = 1
        cv2.circle(out, c1, radius, BLUE, thickness)
        cv2.circle(out, c2, radius, BLUE, thickness)
        cv2.line(out, c1, c2, BLUE, thickness)

    return out

