from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 100, tm_threshold: float = 0.0, aligned_image_size: int = 224
    ) -> None:
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference: Optional[FaceDetectionResult] = None # Can be None or can be a Face. Depending on when we ask

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        # Template matching window size: search region around reference face position
        self.tm_window_size = tm_window_size # Depending on cv2.TemplateMatching method, we can choose the window size.
        # Template matching similarity threshold: minimum score to consider tracking successful
        self.tm_threshold = tm_threshold if tm_threshold > 0.0 else 0.4 # Depends on the cv2.TemplateMatching method, we can choose the threshold.
        # Template matching method: normalized correlation coefficient (robust to illumination)
        self.tm_method = cv2.TM_CCOEFF_NORMED # we try multiple methods to see which one works best.

    # TODO: Track a face in a new image using template matching.
    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # If no reference exists, detect face and store as reference
        if self.reference is None:
            result = self.detect_face(image)
            if result:
                self.reference = result
            return result

        # Extract template from reference image using reference bounding box
        ref_rect = self.reference.rect
        template = self.crop_face(self.reference.image, ref_rect)

        # Check if template is valid (not empty)
        if template.size == 0:
            # Re-initialize if template is invalid
            result = self.detect_face(image)
            if result:
                self.reference = result
            return result

        # Define search window: reference bounding box +/- window_size pixels
        search_x1 = max(0, ref_rect[0] - self.tm_window_size)
        search_y1 = max(0, ref_rect[1] - self.tm_window_size)
        search_x2 = min(image.shape[1], ref_rect[0] + ref_rect[2] + self.tm_window_size)
        search_y2 = min(image.shape[0], ref_rect[1] + ref_rect[3] + self.tm_window_size)

        # Extract search region from current image
        search_region = image[search_y1:search_y2, search_x1:search_x2]

        # Check if search region is large enough for template
        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            # Search region too small, re-initialize
            result = self.detect_face(image)
            if result:
                self.reference = result
            return result

        # Perform template matching
        match_result = cv2.matchTemplate(search_region, template, self.tm_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

        # Get similarity score based on matching method
        if self.tm_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            # For squared difference methods, lower is better (invert for threshold comparison)
            similarity = 1 - min_val
            best_loc = min_loc
        else:
            # For correlation methods, higher is better
            similarity = max_val
            best_loc = max_loc

        # Check if similarity meets threshold
        if similarity >= self.tm_threshold:
            # Tracking successful: update bounding box position
            new_x = search_x1 + best_loc[0]
            new_y = search_y1 + best_loc[1]
            new_rect = (new_x, new_y, ref_rect[2], ref_rect[3])
            # Align the tracked face
            aligned = self.align_face(image, new_rect)
            return FaceDetectionResult(rect=new_rect, image=image, aligned=aligned)
        else:
            # Tracking failed: re-initialize using MTCNN
            result = self.detect_face(image)
            if result:
                # Update reference with new detection
                self.reference = result
            return result

    # Face detection in a new image.
    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # Validate input image
        if image is None or image.size == 0:
            self.reference = None
            return None
        
        # Check minimum image size
        if image.shape[0] < 48 or image.shape[1] < 48:
            self.reference = None
            return None
        
        # Retrieve all detectable faces in the given image.
        try:
            detections = self.detector.detect_faces(image, threshold_pnet=0.85, threshold_rnet=0.9)
        except (ValueError, Exception) as e:
            # Skip problematic frames
            self.reference = None
            return None
        
        if not detections:
            self.reference = None
            return None

        # Select face with the largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
