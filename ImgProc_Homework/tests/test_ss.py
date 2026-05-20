import cv2
import numpy as np
import os

def remove_punch(image, debug=False, out_dir="test_image_2"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # V채널 기반으로 극단적으로 어두운 영역만 탐지
    dark = (hsv[:, :, 2] < 15).astype(np.uint8) * 255

    # 펀치홀은 오른쪽 위 탐색
    roi = np.zeros_like(dark)
    roi[:int(h * 0.40), int(w * 0.50):] = 255
    dark_roi = cv2.bitwise_and(dark, roi)

    # 노이즈 정리
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_roi = cv2.morphologyEx(dark_roi, cv2.MORPH_OPEN, kernel_small, iterations=1)
    dark_roi = cv2.morphologyEx(dark_roi, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    contours, _ = cv2.findContours(dark_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    punch_mask = np.zeros_like(dark)

    # 이미지 크기 대비 펀치홀 반지름 상한
    MAX_PUNCH_RADIUS = int(min(h, w) * 0.06)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 50:
            continue
        if area > h * w * 0.02:  # 0.035 → 0.02로 더 엄격하게
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0

        # 원형도 기준 강화: 0.45 → 0.70
        if circularity < 0.70:
            continue
        if not (0.65 < aspect < 1.35):  # 더 타이트하게
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if not (cx > w * 0.50 and cy < h * 0.40):
            continue

        # 면적 기반 반지름 + bbox 기반 반지름 중 작은 값 채택
        radius_from_area = int(np.sqrt(area / np.pi))
        radius_from_bbox = int(min(bw, bh) / 2)
        radius = min(radius_from_area, radius_from_bbox)

        # 상한 초과 시 스킵
        if radius > MAX_PUNCH_RADIUS:
            if debug:
                print(f"  [punch] 반지름 상한 초과 스킵: radius={radius}, max={MAX_PUNCH_RADIUS}")
            continue

        pad = 2

        if debug:
            print(
                f"  [punch] cx={cx}, cy={cy}, "
                f"area={area:.1f}, radius={radius}, "
                f"circularity={circularity:.2f}, aspect={aspect:.2f}"
            )

        cv2.circle(punch_mask, (cx, cy), radius + pad, 255, -1)

    if debug:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/debug_punch_mask.png", punch_mask)

    if cv2.countNonZero(punch_mask) == 0:
        if debug:
            print("  [punch] 펀치홀 후보 없음")
        return image

    # 항상 inpainting 사용 (단색 채우기 제거)
    # 파란 배경이라도 inpainting이 더 자연스러운 결과를 냄
    result = image.copy()

    # 마스크를 살짝 dilate해서 경계 픽셀까지 커버
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    punch_mask_dilated = cv2.dilate(punch_mask, kernel_dilate, iterations=1)

    result = cv2.inpaint(result, punch_mask_dilated, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

    if debug:
        print("  [punch] → inpainting 적용 (반경 15)")
        cv2.imwrite(f"{out_dir}/debug_punch_result.png", result)

    return result


def remove_background_noise(image, left_ratio=0.15, debug=False, out_dir="test_image_2"):
    h, w = image.shape[:2]
    lx = int(w * left_ratio)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 60, 50]), np.array([135, 255, 255]))
    kernel_bg = np.ones((7, 7), np.uint8)
    bg_mask = cv2.dilate(blue_mask, kernel_bg, iterations=2)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_bg)
    body_mask_raw = cv2.bitwise_not(bg_mask)
    contours, _ = cv2.findContours(body_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_mask = np.zeros_like(body_mask_raw)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(body_mask, [largest], -1, 255, thickness=cv2.FILLED)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blue_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[:, :lx] = 255
    search_zone = cv2.bitwise_and(blue_closed, roi_mask)
    if cv2.countNonZero(search_zone) < 50:
        return image
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    not_blue = ((H < 75) | (H > 145)).astype(np.uint8) * 255
    has_saturation = (S > 15).astype(np.uint8) * 255
    color_diff_mask = cv2.bitwise_and(not_blue, has_saturation)
    candidate_mask = cv2.bitwise_and(color_diff_mask, search_zone)

    if debug:
        cv2.imwrite(f"{out_dir}/debug_blue_mask.png", blue_mask)
        cv2.imwrite(f"{out_dir}/debug_search_zone.png", search_zone)
        cv2.imwrite(f"{out_dir}/debug_candidate_mask.png", candidate_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw_ = stats[i, cv2.CC_STAT_WIDTH]
        ch_ = stats[i, cv2.CC_STAT_HEIGHT]
        if 3 < area < 100 and cw_ < lx * 0.3 and ch_ < h * 0.08:
            final_mask[labels == i] = 255
    if cv2.countNonZero(final_mask) == 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(body_mask))
    if cv2.countNonZero(final_mask) == 0:
        return image
    blue_pixels = image[search_zone == 255]
    if len(blue_pixels) > 0:
        mean_color = blue_pixels.mean(axis=0).astype(np.uint8)
    else:
        mean_color = np.array([255, 0, 0], dtype=np.uint8)
    result = image.copy()
    result[final_mask == 255] = mean_color
    return result


def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape
    interior = (gray > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) != 4:
        return image
    pts = approx.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def remove_noise(image):
    return cv2.bilateralFilter(image, d=6, sigmaColor=50, sigmaSpace=50)


def apply_clahe_conditional(image):
    img_i = image[20:236, 20:236]
    gray = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)

    white_ratio = (gray > 200).sum() / gray.size
    red = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([25, 255, 255]))
    red_ratio = cv2.countNonZero(red) / gray.size
    bright_score = white_ratio / (red_ratio + 0.03)

    print(f"  bright_score={bright_score:.2f} (임계값 7.0) → {'CLAHE 적용' if bright_score > 7.0 else '스킵'}")

    if bright_score > 7.0:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return image


# ── 테스트할 이미지 경로 ──
IMG_PATH = r"C:\Users\sch\PycharmProjects\school-assignment\ImgProc_Homework\xray_images\im013-healthy.jpg"  # ← 여기만 바꾸면 됨
OUT_DIR  = r"C:\Users\sch\PycharmProjects\school-assignment\ImgProc_Homework\test_image_2"

# ──────────────────────────────
image = cv2.imread(IMG_PATH)
if image is None:
    print("이미지 로드 실패")
    exit()

print(f"[테스트] {IMG_PATH}")

image = remove_punch(image, debug=True, out_dir=OUT_DIR)
cv2.imwrite(f"{OUT_DIR}/step1_punch.jpg", image)
print("step1 remove_punch 완료")

image = remove_background_noise(image, debug=True, out_dir=OUT_DIR)
cv2.imwrite(f"{OUT_DIR}/step2_noise.jpg", image)
print("step2 remove_background_noise 완료")

image = correct_perspective(image)
cv2.imwrite(f"{OUT_DIR}/step3_perspective.jpg", image)
print("step3 correct_perspective 완료")

image = remove_noise(image)
cv2.imwrite(f"{OUT_DIR}/step4_remove_noise.jpg", image)
print("step4 remove_noise 완료")

image = apply_clahe_conditional(image)
cv2.imwrite(f"{OUT_DIR}/step5_clahe.jpg", image)
print("step5 apply_clahe_conditional 완료")

print(f"\n저장 완료 → {OUT_DIR}/step1~5 + debug_*.png")