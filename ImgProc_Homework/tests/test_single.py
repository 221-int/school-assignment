import cv2
import numpy as np
from ImgProc_Homework.functions_2 import remove_punch, correct_perspective, remove_noise


def remove_background_noise_test(image, left_ratio=0.15, debug=False):
    h, w = image.shape[:2]
    lx = int(w * left_ratio)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── 1. 파란 배경 마스크 ──
    blue_mask = cv2.inRange(
        hsv,
        np.array([85, 60, 50]),
        np.array([135, 255, 255])
    )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # ── 2. 탐지용: 침식 후 CLOSE ──
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_eroded = cv2.erode(blue_mask, kernel_erode, iterations=1)
    blue_closed_strict = cv2.morphologyEx(blue_eroded, cv2.MORPH_CLOSE, kernel_close)

    # ── 3. 제거용: CLOSE만 ──
    blue_closed_wide = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)

    # ── 4. 좌측 ROI → search_zone ──
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[:, :lx] = 255
    search_zone = cv2.bitwise_and(blue_closed_strict, roi_mask)

    # ── 4-1. search_zone 부족 시 wide fallback ──
    if cv2.countNonZero(search_zone) < 300:
        if debug:
            print("[debug] strict search_zone 부족 → wide fallback")
        search_zone = cv2.bitwise_and(blue_closed_wide, roi_mask)

    if debug:
        cv2.imwrite("test_image/debug_blue_mask.png", blue_mask)
        cv2.imwrite("test_image/debug_blue_eroded.png", blue_eroded)
        cv2.imwrite("test_image/debug_search_zone.png", search_zone)

    if cv2.countNonZero(search_zone) < 50:
        if debug:
            print("[debug] search_zone 픽셀 부족 → 스킵")
        return image

    # ── 5. 파란 배경과 색상이 다른 픽셀 탐지 ──
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]

    not_blue = ((H < 75) | (H > 145)).astype(np.uint8) * 255
    has_saturation = (S > 15).astype(np.uint8) * 255
    color_diff_mask = cv2.bitwise_and(not_blue, has_saturation)
    candidate_mask = cv2.bitwise_and(color_diff_mask, search_zone)

    if debug:
        cv2.imwrite("test_image/debug_candidate_mask.png", candidate_mask)

    # ── 6. 연결 컴포넌트 크기 필터 + 주변 파란 비율 검증 ──
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    kernel_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw_  = stats[i, cv2.CC_STAT_WIDTH]
        ch_  = stats[i, cv2.CC_STAT_HEIGHT]

        if debug:
            print(f"  [component {i}] area={area}, w={cw_}, h={ch_}")

        if not (3 < area < 2000 and cw_ < lx * 0.7 and ch_ < h * 0.3):
            continue

        # 주변 파란 배경 비율 검증
        comp_mask = (labels == i).astype(np.uint8) * 255
        surrounding = cv2.dilate(comp_mask, kernel_check)
        surrounding = cv2.bitwise_and(surrounding, cv2.bitwise_not(comp_mask))

        blue_around = cv2.countNonZero(cv2.bitwise_and(blue_mask, surrounding))
        total_around = cv2.countNonZero(surrounding)
        blue_ratio = blue_around / total_around if total_around > 0 else 0

        if debug:
            print(f"    → blue_ratio={blue_ratio:.2f}")

        if blue_ratio > 0.4:
            final_mask[labels == i] = 255

    if debug:
        cv2.imwrite("test_image/debug_final_mask_before_dilate.png", final_mask)
        print(f"[debug] final_mask 픽셀 수 (팽창 전): {cv2.countNonZero(final_mask)}")

    if cv2.countNonZero(final_mask) == 0:
        if debug:
            print("[debug] final_mask 비어있음 → 스킵")
        return image

    # ── 7. 팽창 후 wide로 제한 ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, kernel, iterations=3)
    final_mask = cv2.bitwise_and(final_mask, blue_closed_wide)

    if debug:
        cv2.imwrite("test_image/debug_final_mask.png", final_mask)
        print(f"[debug] final_mask 픽셀 수 (최종): {cv2.countNonZero(final_mask)}")

    if cv2.countNonZero(final_mask) == 0:
        if debug:
            print("[debug] 차단 후 final_mask 비어있음 → 스킵")
        return image

    # ── 8. 주변 검정 픽셀 미리 파랑으로 채운 후 inpainting ──
    kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ref_zone = cv2.dilate(final_mask, kernel_ref, iterations=1)

    gray_tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels = (gray_tmp < 20).astype(np.uint8) * 255
    dark_in_ref = cv2.bitwise_and(dark_pixels, ref_zone)

    temp_image = image.copy()
    blue_pixels = image[search_zone == 255]
    if len(blue_pixels) > 0:
        mean_blue = blue_pixels.mean(axis=0).astype(np.uint8)
        temp_image[dark_in_ref == 255] = mean_blue

    return cv2.inpaint(temp_image, final_mask, inpaintRadius=7,
                       flags=cv2.INPAINT_TELEA)


img_path = "test_image/im078-pneumonia.jpg"
image = cv2.imread(img_path)

image = remove_punch(image)
cv2.imwrite("test_image/step1_punch.jpg", image)

image = remove_background_noise_test(image, debug=True)
cv2.imwrite("test_image/step2_noise.jpg", image)

image = correct_perspective(image)
cv2.imwrite("test_image/step3_perspective.jpg", image)

image = remove_noise(image)
cv2.imwrite("test_image/step4_final.jpg", image)

print("완료!")