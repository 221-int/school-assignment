# #안녕 :)
# 0.92
#
#
# import cv2
# import numpy as np
#
#
# def remove_punch(image):
#     """
#     Flood Fill로 배경 테두리와 펀쳐를 구분하여
#     펀쳐만 인페인팅으로 복원한다.
#     Perspective Transform 전에 적용해야 정확하게 탐지됨.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     dark = (gray < 15).astype(np.uint8) * 255
#     bg_mask = dark.copy()
#     for corner in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]:
#         cv2.floodFill(bg_mask, None, corner, 128)
#     punch_only = (bg_mask == 255).astype(np.uint8) * 255
#     if punch_only.sum() == 0:
#         return image
#     kernel = np.ones((5, 5), np.uint8)
#     punch_mask = cv2.dilate(punch_only, kernel)
#     return cv2.inpaint(image, punch_mask, 10, cv2.INPAINT_TELEA)
#
#
# # def remove_background_noise(image, left_ratio=0.15, debug=False):
# #     h, w = image.shape[:2]
# #     lx = int(w * left_ratio)
# #
# #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #
# #     # ── 1. 파란 배경 마스크 ──
# #     blue_mask = cv2.inRange(
# #         hsv,
# #         np.array([85, 60, 50]),
# #         np.array([135, 255, 255])
# #     )
# #
# #     # ── 2. 탐지용: 침식 후 CLOSE (안정적인 위치 탐지) ──
# #     kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# #     blue_eroded = cv2.erode(blue_mask, kernel_erode, iterations=1)
# #     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
# #     blue_closed_strict = cv2.morphologyEx(blue_eroded, cv2.MORPH_CLOSE, kernel_close)
# #
# #     # ── 3. 제거용: CLOSE만 (더 넓게 잡아서 마커 잘 지워짐) ──
# #     blue_closed_wide = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
# #
# #     # ── 4. 좌측 ROI → 탐지용으로 search_zone 구성 ──
# #     roi_mask = np.zeros((h, w), dtype=np.uint8)
# #     roi_mask[:, :lx] = 255
# #     search_zone = cv2.bitwise_and(blue_closed_strict, roi_mask)
# #
# #     if debug:
# #         cv2.imwrite("debug_blue_mask.png", blue_mask)
# #         cv2.imwrite("debug_blue_eroded.png", blue_eroded)
# #         cv2.imwrite("debug_search_zone.png", search_zone)
# #
# #     if cv2.countNonZero(search_zone) < 300:
# #         if debug:
# #             print("[debug] search_zone 픽셀 부족 → 스킵")
# #         return image
# #
# #     # ── 5. 파란 배경과 색상이 다른 픽셀 탐지 ──
# #     H = hsv[:, :, 0]
# #     S = hsv[:, :, 1]
# #
# #     not_blue = ((H < 75) | (H > 145)).astype(np.uint8) * 255
# #     has_saturation = (S > 15).astype(np.uint8) * 255
# #     color_diff_mask = cv2.bitwise_and(not_blue, has_saturation)
# #     candidate_mask = cv2.bitwise_and(color_diff_mask, search_zone)
# #
# #     if debug:
# #         cv2.imwrite("debug_candidate_mask.png", candidate_mask)
# #
# #     # ── 6. 연결 컴포넌트 크기 필터 ──
# #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
# #         candidate_mask, connectivity=8
# #     )
# #
# #     final_mask = np.zeros((h, w), dtype=np.uint8)
# #     for i in range(1, num_labels):
# #         area = stats[i, cv2.CC_STAT_AREA]
# #         cw_  = stats[i, cv2.CC_STAT_WIDTH]
# #         ch_  = stats[i, cv2.CC_STAT_HEIGHT]
# #
# #         if debug:
# #             print(f"  [component {i}] area={area}, w={cw_}, h={ch_}")
# #
# #         if 3 < area < 500 and cw_ < lx * 0.5 and ch_ < h * 0.15:
# #             final_mask[labels == i] = 255
# #
# #     if debug:
# #         cv2.imwrite("debug_final_mask_before_dilate.png", final_mask)
# #         print(f"[debug] final_mask 픽셀 수 (팽창 전): {cv2.countNonZero(final_mask)}")
# #
# #     if cv2.countNonZero(final_mask) == 0:
# #         if debug:
# #             print("[debug] final_mask 비어있음 → 스킵")
# #         return image
# #
# #     # ── 7. 팽창 후 제거용(wide) blue_closed로 제한 ──
# #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# #     final_mask = cv2.dilate(final_mask, kernel, iterations=2)
# #     final_mask = cv2.bitwise_and(final_mask, blue_closed_wide)  # 넓은 마스크로 제거
# #
# #     if debug:
# #         cv2.imwrite("debug_final_mask.png", final_mask)
# #         print(f"[debug] final_mask 픽셀 수 (최종): {cv2.countNonZero(final_mask)}")
# #
# #     if cv2.countNonZero(final_mask) == 0:
# #         if debug:
# #             print("[debug] 차단 후 final_mask 비어있음 → 스킵")
# #         return image
# #
# #     # ── 8. 주변 검정 픽셀 미리 파랑으로 채운 후 inpainting ──
# #     kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# #     ref_zone = cv2.dilate(final_mask, kernel_ref, iterations=1)
# #
# #     gray_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     dark_pixels = (gray_tmp < 20).astype(np.uint8) * 255
# #     dark_in_ref = cv2.bitwise_and(dark_pixels, ref_zone)
# #
# #     temp_image = image.copy()
# #     blue_pixels = image[search_zone == 255]
# #     if len(blue_pixels) > 0:
# #         mean_blue = blue_pixels.mean(axis=0).astype(np.uint8)
# #         temp_image[dark_in_ref == 255] = mean_blue
# #
# #     return cv2.inpaint(temp_image, final_mask, inpaintRadius=7,
# #                        flags=cv2.INPAINT_TELEA)
#
# def remove_background_noise(image, left_ratio=0.15, debug=False):
#     h, w = image.shape[:2]
#     lx = int(w * left_ratio)
#
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     # ── 1. 파란 배경 마스크 ──
#     blue_mask = cv2.inRange(
#         hsv,
#         np.array([85, 60, 50]),
#         np.array([135, 255, 255])
#     )
#
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#
#     # ── 2. 탐지용: 침식 후 CLOSE ──
#     kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     blue_eroded = cv2.erode(blue_mask, kernel_erode, iterations=1)
#     blue_closed_strict = cv2.morphologyEx(blue_eroded, cv2.MORPH_CLOSE, kernel_close)
#
#     # ── 3. 제거용: CLOSE만 ──
#     blue_closed_wide = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
#
#     # ── 4. 좌측 ROI → search_zone ──
#     roi_mask = np.zeros((h, w), dtype=np.uint8)
#     roi_mask[:, :lx] = 255
#     search_zone = cv2.bitwise_and(blue_closed_strict, roi_mask)
#
#     # ── 4-1. search_zone 부족 시 wide fallback ──
#     if cv2.countNonZero(search_zone) < 300:
#         if debug:
#             print("[debug] strict search_zone 부족 → wide fallback")
#         search_zone = cv2.bitwise_and(blue_closed_wide, roi_mask)
#
#     if debug:
#         cv2.imwrite("debug_blue_mask.png", blue_mask)
#         cv2.imwrite("debug_blue_eroded.png", blue_eroded)
#         cv2.imwrite("debug_search_zone.png", search_zone)
#
#     if cv2.countNonZero(search_zone) < 50:
#         if debug:
#             print("[debug] search_zone 픽셀 부족 → 스킵")
#         return image
#
#     # ── 5. 파란 배경과 색상이 다른 픽셀 탐지 ──
#     H = hsv[:, :, 0]
#     S = hsv[:, :, 1]
#
#     not_blue = ((H < 75) | (H > 145)).astype(np.uint8) * 255
#     has_saturation = (S > 8).astype(np.uint8) * 255
#     color_diff_mask = cv2.bitwise_and(not_blue, has_saturation)
#     candidate_mask = cv2.bitwise_and(color_diff_mask, search_zone)
#
#     if debug:
#         cv2.imwrite("debug_candidate_mask.png", candidate_mask)
#
#     # ── 6. 연결 컴포넌트 크기 필터 + 주변 파란 비율 검증 ──
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         candidate_mask, connectivity=8
#     )
#
#     kernel_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     final_mask = np.zeros((h, w), dtype=np.uint8)
#
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         cw_ = stats[i, cv2.CC_STAT_WIDTH]
#         ch_ = stats[i, cv2.CC_STAT_HEIGHT]
#
#         if debug:
#             print(f"  [component {i}] area={area}, w={cw_}, h={ch_}")
#
#         if not (3 < area < 2000 and cw_ < lx * 0.7 and ch_ < h * 0.3):
#             continue
#
#         # 주변 파란 배경 비율 검증
#         comp_mask = (labels == i).astype(np.uint8) * 255
#         surrounding = cv2.dilate(comp_mask, kernel_check)
#         surrounding = cv2.bitwise_and(surrounding, cv2.bitwise_not(comp_mask))
#
#         blue_around = cv2.countNonZero(cv2.bitwise_and(blue_mask, surrounding))
#         total_around = cv2.countNonZero(surrounding)
#         blue_ratio = blue_around / total_around if total_around > 0 else 0
#
#         if debug:
#             print(f"    → blue_ratio={blue_ratio:.2f}")
#
#         if blue_ratio > 0.4:
#             final_mask[labels == i] = 255
#
#     if debug:
#         cv2.imwrite("debug_final_mask_before_dilate.png", final_mask)
#         print(f"[debug] final_mask 픽셀 수 (팽창 전): {cv2.countNonZero(final_mask)}")
#
#     if cv2.countNonZero(final_mask) == 0:
#         if debug:
#             print("[debug] final_mask 비어있음 → 스킵")
#         return image
#
#     # ── 7. 팽창 후 wide로 제한 ──
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     final_mask = cv2.dilate(final_mask, kernel, iterations=3)
#     final_mask = cv2.bitwise_and(final_mask, blue_closed_wide)
#
#     if debug:
#         cv2.imwrite("debug_final_mask.png", final_mask)
#         print(f"[debug] final_mask 픽셀 수 (최종): {cv2.countNonZero(final_mask)}")
#
#     if cv2.countNonZero(final_mask) == 0:
#         if debug:
#             print("[debug] 차단 후 final_mask 비어있음 → 스킵")
#         return image
#
#     # ── 8. 주변 검정 픽셀 미리 파랑으로 채운 후 inpainting ──
#     kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#     ref_zone = cv2.dilate(final_mask, kernel_ref, iterations=1)
#
#     gray_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     dark_pixels = (gray_tmp < 20).astype(np.uint8) * 255
#     dark_in_ref = cv2.bitwise_and(dark_pixels, ref_zone)
#
#     temp_image = image.copy()
#     blue_pixels = image[search_zone == 255]
#     if len(blue_pixels) > 0:
#         mean_blue = blue_pixels.mean(axis=0).astype(np.uint8)
#         temp_image[dark_in_ref == 255] = mean_blue
#
#     return cv2.inpaint(temp_image, final_mask, inpaintRadius=7,
#                        flags=cv2.INPAINT_TELEA)
#
# # def correct_perspective(image):
# #     """
# #     X-ray 내부 사각형의 꼭짓점 4개를 찾아
# #     Perspective Transform으로 정사각형으로 보정한다.
# #     """
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
# #     h, w = gray.shape
# #     interior = (gray > 20).astype(np.uint8) * 255
# #     contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not contours:
# #         return image
# #     largest = max(contours, key=cv2.contourArea)
# #     epsilon = 0.02 * cv2.arcLength(largest, True)
# #     approx = cv2.approxPolyDP(largest, epsilon, True)
# #     if len(approx) != 4:
# #         return image
# #     pts = approx.reshape(4, 2).astype(np.float32)
# #     rect = np.zeros((4, 2), dtype=np.float32)
# #     s = pts.sum(axis=1)
# #     rect[0] = pts[np.argmin(s)]
# #     rect[2] = pts[np.argmax(s)]
# #     diff = np.diff(pts, axis=1)
# #     rect[1] = pts[np.argmin(diff)]
# #     rect[3] = pts[np.argmax(diff)]
# #     dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
# #     M = cv2.getPerspectiveTransform(rect, dst)
# #     return cv2.warpPerspective(image, M, (w, h),
# #                                flags=cv2.INTER_CUBIC,
# #                                borderMode=cv2.BORDER_REFLECT)
#
# # def correct_perspective(image):
# #     """
# #     X-ray 내부 사각형의 꼭짓점 4개를 찾아
# #     Perspective Transform으로 정사각형으로 보정한다.
# #     - 4각형 검출 실패 시 원본 반환
# #     - 보정 전후 비율 변화가 너무 크면 원본 반환
# #     - 꼭짓점이 이미지 경계에 너무 가까우면 원본 반환
# #     - 변환 후 검은 영역이 너무 많으면 원본 반환
# #     """
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
# #     h, w = gray.shape
# #
# #     interior = (gray > 20).astype(np.uint8) * 255
# #     contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not contours:
# #         return image
# #
# #     largest = max(contours, key=cv2.contourArea)
# #     epsilon = 0.02 * cv2.arcLength(largest, True)
# #     approx = cv2.approxPolyDP(largest, epsilon, True)
# #
# #     # 4각형 못 찾으면 원본 반환
# #     if len(approx) != 4:
# #         return image
# #
# #     pts = approx.reshape(4, 2).astype(np.float32)
# #
# #     # ── 안전장치 1: 꼭짓점이 이미지 경계에서 너무 멀면 원본 반환 ──
# #     # 즉, 검출된 사각형이 이미지 전체 대비 너무 작으면 스킵
# #     x_coords = pts[:, 0]
# #     y_coords = pts[:, 1]
# #     box_w = x_coords.max() - x_coords.min()
# #     box_h = y_coords.max() - y_coords.min()
# #     if box_w < w * 0.5 or box_h < h * 0.5:
# #         return image
# #
# #     # ── 안전장치 2: 보정 전 비율 저장 ──
# #     before_ratio = h / w
# #
# #     rect = np.zeros((4, 2), dtype=np.float32)
# #     s = pts.sum(axis=1)
# #     rect[0] = pts[np.argmin(s)]
# #     rect[2] = pts[np.argmax(s)]
# #     diff = np.diff(pts, axis=1)
# #     rect[1] = pts[np.argmin(diff)]
# #     rect[3] = pts[np.argmax(diff)]
# #
# #     # ── 안전장치 3: 꼭짓점 기울기가 너무 작으면 (거의 반듯하면) 스킵 ──
# #     # 좌상단과 우상단의 y 차이가 너무 작으면 보정 불필요
# #     tl, tr = rect[0], rect[1]
# #     bl, br = rect[3], rect[2]
# #     top_skew = abs(tl[1] - tr[1])
# #     side_skew = abs(tl[0] - bl[0])
# #     if top_skew < h * 0.03 and side_skew < w * 0.03:
# #         return image
# #
# #     dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
# #     M = cv2.getPerspectiveTransform(rect, dst)
# #     result = cv2.warpPerspective(image, M, (w, h),
# #                                  flags=cv2.INTER_CUBIC,
# #                                  borderMode=cv2.BORDER_REFLECT)
# #
# #     # ── 안전장치 4: 보정 후 비율 변화가 너무 크면 원본 반환 ──
# #     after_ratio = result.shape[0] / result.shape[1]
# #     if abs(before_ratio - after_ratio) > 0.2:
# #         return image
# #
# #     # ── 안전장치 5: 보정 후 검은 영역이 너무 많으면 원본 반환 ──
# #     result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# #     black_ratio = (result_gray < 20).sum() / (h * w)
# #     if black_ratio > 0.15:
# #         return image
# #
# #     return result
#
# def correct_perspective(image):
#     """
#     X-ray 내부 사각형의 꼭짓점 4개를 찾아
#     Perspective Transform으로 정사각형으로 보정한다.
#     안전장치 포함 - 실패 시 원본 반환
#     """
#     def blue_ratio(img):
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         blue = cv2.inRange(hsv, np.array([85, 60, 50]), np.array([135, 255, 255]))
#         return cv2.countNonZero(blue) / (img.shape[0] * img.shape[1])
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
#     h, w = gray.shape
#
#     interior = (gray > 20).astype(np.uint8) * 255
#     contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return image
#
#     largest = max(contours, key=cv2.contourArea)
#     epsilon = 0.02 * cv2.arcLength(largest, True)
#     approx = cv2.approxPolyDP(largest, epsilon, True)
#
#     if len(approx) != 4:
#         return image
#
#     pts = approx.reshape(4, 2).astype(np.float32)
#
#     # 안전장치 1: 사각형이 너무 작으면 스킵
#     x_coords = pts[:, 0]
#     y_coords = pts[:, 1]
#     box_w = x_coords.max() - x_coords.min()
#     box_h = y_coords.max() - y_coords.min()
#     if box_w < w * 0.5 or box_h < h * 0.5:
#         return image
#
#     before_ratio = h / w
#
#     rect = np.zeros((4, 2), dtype=np.float32)
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#
#     # 안전장치 2: 기울기가 거의 없으면 스킵
#     tl, tr = rect[0], rect[1]
#     bl = rect[3]
#     top_skew = abs(tl[1] - tr[1])
#     side_skew = abs(tl[0] - bl[0])
#     if top_skew < h * 0.03 and side_skew < w * 0.03:
#         return image
#
#     dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(rect, dst)
#     result = cv2.warpPerspective(image, M, (w, h),
#                                  flags=cv2.INTER_CUBIC,
#                                  borderMode=cv2.BORDER_REFLECT)
#
#     # 안전장치 3: 비율 변화가 너무 크면 원본 반환
#     after_ratio = result.shape[0] / result.shape[1]
#     if abs(before_ratio - after_ratio) > 0.2:
#         return image
#
#     # 안전장치 4: 검은 영역이 너무 많으면 원본 반환
#     result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
#     black_ratio = (result_gray < 20).sum() / (h * w)
#     if black_ratio > 0.15:
#         return image
#
#     # 안전장치 5: 보정 전후 파란 영역 비율 변화가 크면 원본 반환
#     before_blue = blue_ratio(image)
#     after_blue = blue_ratio(result)
#     if abs(before_blue - after_blue) > 0.15:
#         return image
#
#     return result
#
#
# def remove_noise(image):
#     """
#     Bilateral Filter로 엣지를 보존하면서 노이즈를 제거한다.
#     """
#
#     return cv2.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=50)
#
#
# def enhance_color(image, saturation_scale=1.2, contrast_clip=1.0):
#     """
#     처리 후 색상 복원: 채도 강화 + 선택적 contrast 조정
#     """
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
#     result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#     return result
#
# def apply_gamma(image, gamma=1.2):
#     """
#     Gamma 보정으로 밝기 곡선을 조정한다.
#     gamma > 1 : 밝은 영역 억제 (Healthy 오분류 방지)
#     gamma < 1 : 어두운 영역 강조
#     """
#     inv_gamma = 1.0 / gamma
#     table = np.array([
#         ((i / 255.0) ** inv_gamma) * 255
#         for i in range(256)
#     ]).astype(np.uint8)
#     return cv2.LUT(image, table)
#
#
# def process_image(img_path, failed_files, debug=False):
#     try:
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError(f"이미지 로드 실패: {img_path}")
#
#         if debug:
#             import os
#             base_name = os.path.splitext(os.path.basename(img_path))[0]
#             debug_dir = os.path.join(os.path.dirname(img_path), f"debug_{base_name}")
#             os.makedirs(debug_dir, exist_ok=True)
#             step = 0
#
#             def save_step(img, name):
#                 nonlocal step
#                 step += 1
#                 save_path = os.path.join(debug_dir, f"step{step:02d}_{name}.jpg")
#                 cv2.imwrite(save_path, img)
#                 print(f"  [debug] saved: {save_path}")
#
#             save_step(image, "00_original")  # ← 여기만 있고
#
#         # ① 펀쳐 복원
#         image = remove_punch(image)
#         if debug: save_step(image, "01_remove_punch")        # ← 이게 없었음
#         #
#         # image = remove_background_noise(image, debug=True)
#         # if debug: save_step(image, "02_remove_background_noise")
#
#         # ③ Perspective Transform 기울기 보정
#         image = correct_perspective(image)
#         if debug: save_step(image, "03_correct_perspective")  # ← 이게 없었음
#
#         # ④ Bilateral Filter 노이즈 제거
#         image = remove_noise(image)
#         if debug: save_step(image, "04_remove_noise")         # ← 이게 없었음
#
#         # # ⑤ Gamma 보정
#         # image = apply_gamma(image, gamma=1.03)
#
#         return image
#
#     except Exception as e:
#         print(f"처리 실패: {img_path}, 오류: {e}")
#         failed_files.append(img_path)
#         return None
#
# # def process_image(img_path, failed_files, debug=False):
# #     try:
# #         image = cv2.imread(img_path)
# #         if image is None:
# #             raise ValueError(f"이미지 로드 실패: {img_path}")
# #
# #         # 디버그 저장 경로 설정
# #         if debug:
# #             import os
# #             base_name = os.path.splitext(os.path.basename(img_path))[0]
# #             debug_dir = os.path.join(os.path.dirname(img_path), f"debug_{base_name}")
# #             os.makedirs(debug_dir, exist_ok=True)
# #             step = 0
# #
# #             def save_step(img, name):
# #                 nonlocal step
# #                 step += 1
# #                 save_path = os.path.join(debug_dir, f"step{step:02d}_{name}.jpg")
# #                 cv2.imwrite(save_path, img)
# #                 print(f"  [debug] saved: {save_path}")
# #
# #             save_step(image, "00_original")
# #
# #
# #
# #         # ① 펀쳐 복원
# #         image = remove_punch(image)
# #
# #         # # ② R 마커 제거 ← perspective 전에 실행해야 파란 배경이 남아있음
# #         # image = remove_background_noise(image, debug=True)
# #
# #         # ③ Perspective Transform 기울기 보정
# #         image = correct_perspective(image)
# #
# #         # ④ Bilateral Filter 노이즈 제거
# #         image = remove_noise(image)
# #
# #         return image
# #
# #     except Exception as e:
# #         print(f"처리 실패: {img_path}, 오류: {e}")
# #         failed_files.append(img_path)
# #         return None
