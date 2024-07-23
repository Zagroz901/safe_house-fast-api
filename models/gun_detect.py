import os
import cv2
import numpy as np

class GunDetector:
    def __init__(self, template_folder, ratio_test_threshold=0.75, min_match_count=20):
        self.template_folder = template_folder
        self.ratio_test_threshold = ratio_test_threshold
        self.min_match_count = min_match_count

    def preprocess_image(self, image):
        if len(image.shape) > 2 and image.shape[2] == 3:  # Check if the image is in color
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        else:
            gray_image = image  # It's already a grayscale image

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(blurred_image)
        return enhanced_image


    def configure_flann_matcher(self):
        index_params = dict(algorithm=1, trees=20)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann

    def detect_guns(self, image):
        sift = cv2.SIFT_create()
        keypoints_image, descriptors_image = sift.detectAndCompute(image, None)
        descriptors_image = np.float32(descriptors_image)
        flann = self.configure_flann_matcher()

        best_match = None
        max_inliers = 0

        for template_name in os.listdir(self.template_folder):
            template_path = os.path.join(self.template_folder, template_name)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            template = self.preprocess_image(template)

            keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
            descriptors_template = np.float32(descriptors_template)
            
            matches = flann.knnMatch(descriptors_image, descriptors_template, k=2)
            good = []
            for m, n in matches:
                if m.distance < self.ratio_test_threshold * n.distance:
                    good.append(m)

            if len(good) > self.min_match_count:
                src_pts = np.float32([keypoints_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_template[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    num_inliers = np.sum(mask)
                    if num_inliers > max_inliers:
                        max_inliers = num_inliers
                        best_match = (good, mask, keypoints_image, keypoints_template, template_name, M)

        return best_match

# # Usage
# if __name__ == "__main__":
#     detector = GunDetector(template_folder='gun_template')
#     main_image = cv2.imread('gun_photo/download (3).jpeg', cv2.IMREAD_GRAYSCALE)
#     main_image_preprocessed = detector.preprocess_image(main_image)

#     best_match = detector.detect_guns(main_image_preprocessed)
#     if best_match:
#         good_matches, matchesMask, keypoints_main, keypoints_template, template_name, M = best_match
#         print(f"Best match found with template: {template_name} having {np.sum(matchesMask)} inliers")
#         draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask.ravel().tolist(), flags=2)
#         result_image = cv2.drawMatches(main_image, keypoints_main, cv2.imread(os.path.join(detector.template_folder, template_name), cv2.IMREAD_GRAYSCALE), keypoints_template, good_matches, None, **draw_params)
#         cv2.imshow("Best Match", result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("No sufficient matches found.")
