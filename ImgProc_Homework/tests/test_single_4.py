import cv2
import numpy as np
from ImgProc_Homework.functions_2 import remove_punch, correct_perspective, remove_noise


def remove_background_noise(image, left_ratio=0.15, right_ratio=0.15, debug=False):
    h, w = image.shape[:2]
    lx = int(w * left_ratio)
    rx = int(w * (1 - right_ratio))  # 오른쪽 경계 x좌표

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── 1. 파란 배경 마스크 ──
    blue_mask = cv2.inRange(
        hsv,
        np.array([85, 60, 50]),
        np.array([135, 255, 255])
    )

    # ── 2. 탐지용: 침식 후 CLOSE ──
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_eroded = cv2.erode(blue_mask, kernel_erode, iterations=1)
    kernel_close_strict = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blue_closed_strict = cv2.morphologyEx(blue_eroded, cv2.MORPH_CLOSE, kernel_close_strict)

    # ── 3. 제거용: CLOSE만 ──
    kernel_close_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blue_closed_wide = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close_wide)

    # ── 4. 양쪽 가장자리 ROI → search_zone ──
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[:, :lx] = 255        # 왼쪽
    roi_mask[:, rx:] = 255        # 오른쪽

    search_zone = cv2.bitwise_and(blue_closed_strict, roi_mask)

    if cv2.countNonZero(search_zone) < 300:
        if debug:
            print("[debug] strict search_zone 부족 → wide fallback")
        search_zone = cv2.bitwise_and(blue_closed_wide, roi_mask)

    if debug:
        cv2.imwrite("test_image/debug_blue_mask.png", blue_mask)
        cv2.imwrite("test_image/debug_search_zone.png", search_zone)

    if cv2.countNonZero(search_zone) < 50:
        if debug:
            print("[debug] search_zone 픽셀 부족 → 스킵")
        return image

    # ── 5. 노이즈 후보 탐지: 색상 다른 픽셀 OR 흰색/밝은 픽셀 ──
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    not_blue = ((H < 75) | (H > 145)).astype(np.uint8) * 255
    has_saturation = (S > 8).astype(np.uint8) * 255
    color_diff_mask = cv2.bitwise_and(not_blue, has_saturation)

    # 흰색/밝은 노이즈
    bright_mask = ((S < 40) & (V > 160)).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_or(color_diff_mask, bright_mask)
    candidate_mask = cv2.bitwise_and(combined_mask, search_zone)

    if debug:
        cv2.imwrite("test_image/debug_bright_mask.png", bright_mask)
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
        cx_  = stats[i, cv2.CC_STAT_LEFT]

        # 왼쪽/오른쪽 각각 크기 기준 적용
        side_w = lx if cx_ < w // 2 else (w - rx)
        if not (3 < area < 2000 and cw_ < side_w * 0.7 and ch_ < h * 0.3):
            if debug:
                print(f"  [component {i}] area={area}, w={cw_}, h={ch_} → 크기 필터 탈락")
            continue

        # 주변 파란 배경 비율 검증
        comp_mask = (labels == i).astype(np.uint8) * 255
        surrounding = cv2.dilate(comp_mask, kernel_check)
        surrounding = cv2.bitwise_and(surrounding, cv2.bitwise_not(comp_mask))

        blue_around = cv2.countNonZero(cv2.bitwise_and(blue_mask, surrounding))
        total_around = cv2.countNonZero(surrounding)
        blue_ratio = blue_around / total_around if total_around > 0 else 0

        if debug:
            print(f"  [component {i}] area={area}, w={cw_}, h={ch_}, blue_ratio={blue_ratio:.2f}")

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

    # ── 8. 각 컴포넌트 주변 로컬 파란 평균색으로 직접 덮기 ──
    result = image.copy()

    num_final, final_labels, _, _ = cv2.connectedComponentsWithStats(
        final_mask, connectivity=8
    )

    kernel_local = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    for i in range(1, num_final):
        comp_mask = (final_labels == i).astype(np.uint8) * 255

        local_zone = cv2.dilate(comp_mask, kernel_local, iterations=2)
        local_zone = cv2.bitwise_and(local_zone, cv2.bitwise_not(comp_mask))
        local_blue = cv2.bitwise_and(local_zone, blue_mask)

        local_blue_pixels = image[local_blue == 255]

        if len(local_blue_pixels) > 0:
            fill_color = local_blue_pixels.mean(axis=0).astype(np.uint8)
        else:
            all_blue_pixels = image[search_zone == 255]
            fill_color = all_blue_pixels.mean(axis=0).astype(np.uint8) if len(all_blue_pixels) > 0 else np.array([180, 100, 50], dtype=np.uint8)

        result[comp_mask == 255] = fill_color

    if debug:
        cv2.imwrite("test_image/debug_result.png", result)

    return result


# ── 메인 파이프라인 ──
img_path = "test_image/im100-pneumonia.jpg"
image = cv2.imread(img_path)
if image is None:
    print("이미지 로드 실패")
    exit()

image = remove_punch(image)
cv2.imwrite("test_image/step1_punch.jpg", image)
print("step1 완료")

image = remove_background_noise(image, debug=True)
cv2.imwrite("test_image/step2_noise.jpg", image)
print("step2 완료")

image = correct_perspective(image)
cv2.imwrite("test_image/step3_perspective.jpg", image)
print("step3 완료")

image = remove_noise(image)
cv2.imwrite("test_image/step4_final.jpg", image)
print("step4 완료")

print("완료!")