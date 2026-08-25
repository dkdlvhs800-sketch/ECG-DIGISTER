from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 경로
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_IMAGE = (
    PROJECT_ROOT
    / "finetune_data"
    / "images"
    / "EKG.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "finetune_data"
    / "mask_debug"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# ECG 행 검출
# ==========================================================

def detect_ecg_rows(dark_mask):

    height, width = dark_mask.shape

    x1 = int(width * 0.07)
    x2 = int(width * 0.98)

    analysis_mask = dark_mask[:, x1:x2]

    row_score = (
        analysis_mask > 0
    ).sum(axis=1).astype(float)

    kernel_size = 21

    kernel = (
        np.ones(kernel_size)
        / kernel_size
    )

    smoothed_score = np.convolve(
        row_score,
        kernel,
        mode="same"
    )

    search_top = int(
        height * 0.10
    )

    search_bottom = int(
        height * 0.88
    )

    candidate_indices = np.argsort(
        smoothed_score[
            search_top:search_bottom
        ]
    )[::-1]

    candidate_indices += search_top

    min_distance = int(
        height * 0.12
    )

    detected_rows = []

    for y in candidate_indices:

        y = int(y)

        if all(
            abs(y - old_y) >= min_distance
            for old_y in detected_rows
        ):

            detected_rows.append(y)

        if len(detected_rows) == 4:
            break

    detected_rows.sort()

    return detected_rows


# ==========================================================
# component 분석
# ==========================================================

def remove_text_and_vertical_noise(binary_region):

    h, w = binary_region.shape

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_region,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        binary_region
    )

    removed = np.zeros_like(
        binary_region
    )

    print()
    print(
        "===== COMPONENT ANALYSIS ====="
    )

    for label_id in range(
        1,
        num_labels
    ):

        x, y, cw, ch, area = (
            stats[label_id]
        )

        # ----------------------------------------------
        # 기본 특징
        # ----------------------------------------------

        aspect_ratio = (
            cw / max(ch, 1)
        )

        # 매우 긴 수직선
        very_vertical = (
            ch >= 18
            and cw <= 4
        )

        # 작은 국소 component
        small_local = (
            area <= 35
            and cw <= 18
            and ch <= 18
        )

        # 글자처럼 비교적 좁은 영역
        text_like = (
            area <= 70
            and cw <= 25
            and ch <= 25
        )

        # ----------------------------------------------
        # ECG 파형 component를 함부로 지우지 않기 위한 조건
        #
        # x 방향으로 충분히 길면 보존
        # ----------------------------------------------

        horizontally_extended = (
            cw >= 30
        )

        # ----------------------------------------------
        # 제거 여부
        # ----------------------------------------------

        should_remove = False

        reason = []

        if very_vertical:

            should_remove = True
            reason.append(
                "vertical"
            )

        if small_local:

            should_remove = True
            reason.append(
                "small"
            )

        if (
            text_like
            and not horizontally_extended
        ):

            should_remove = True
            reason.append(
                "text-like"
            )

        component_mask = (
            labels == label_id
        )

        if should_remove:

            removed[
                component_mask
            ] = 255

        else:

            cleaned[
                component_mask
            ] = 255

        if area >= 5:

            print(
                f"ID={label_id:3d} "
                f"x={x:3d} "
                f"y={y:3d} "
                f"w={cw:3d} "
                f"h={ch:3d} "
                f"area={area:4d} "
                f"remove={should_remove} "
                f"{','.join(reason)}"
            )

    return cleaned, removed


# ==========================================================
# 실행
# ==========================================================

def main():

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:

        raise FileNotFoundError(
            INPUT_IMAGE
        )

    height, width = (
        image.shape[:2]
    )

    print(
        "IMAGE:",
        width,
        "x",
        height
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = np.where(
        gray < 100,
        255,
        0
    ).astype(
        np.uint8
    )

    # ======================================================
    # ECG 행 찾기
    # ======================================================

    detected_rows = (
        detect_ecg_rows(
            dark_mask
        )
    )

    print(
        "Detected rows:",
        detected_rows
    )

    if len(detected_rows) != 4:

        raise RuntimeError(
            "ECG 4개 행 검출 실패"
        )

    # ======================================================
    # V3 위치
    #
    # 표준 3x4 구조:
    #
    # row 1: I   aVR  V1  V4
    # row 2: II  aVL  V2  V5
    # row 3: III aVF  V3  V6
    #
    # 따라서 V3 = 세 번째 행 / 세 번째 column
    # ======================================================

    row_index = 2
    column_index = 2

    row_gaps = np.diff(
        detected_rows
    )

    median_gap = float(
        np.median(
            row_gaps
        )
    )

    half_height = int(
        median_gap * 0.35
    )

    y_center = (
        detected_rows[
            row_index
        ]
    )

    y1 = max(
        0,
        y_center - half_height
    )

    y2 = min(
        height,
        y_center + half_height
    )

    # ECG 실제 영역
    ecg_x1 = int(
        width * 0.07
    )

    ecg_x2 = int(
        width * 0.98
    )

    ecg_width = (
        ecg_x2 - ecg_x1
    )

    column_width = (
        ecg_width / 4
    )

    x1 = int(
        ecg_x1
        + column_index
        * column_width
    )

    x2 = int(
        ecg_x1
        + (
            column_index + 1
        )
        * column_width
    )

    # 약간 안쪽으로 crop
    margin = int(
        column_width * 0.02
    )

    x1 += margin
    x2 -= margin

    print()
    print(
        "V3 crop:"
    )

    print(
        "x:",
        x1,
        "-",
        x2
    )

    print(
        "y:",
        y1,
        "-",
        y2
    )

    # ======================================================
    # V3 crop
    # ======================================================

    v3_binary = dark_mask[
        y1:y2,
        x1:x2
    ]

    cleaned, removed = (
        remove_text_and_vertical_noise(
            v3_binary
        )
    )

    # ======================================================
    # 결과 저장
    # ======================================================

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "10_v3_original_binary.png"
        ),
        v3_binary
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "10_v3_cleaned.png"
        ),
        cleaned
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "10_v3_removed.png"
        ),
        removed
    )

    # ======================================================
    # 비교 그림
    # ======================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 9)
    )

    axes[0].imshow(
        v3_binary,
        cmap="gray"
    )

    axes[0].set_title(
        "1. Original V3 binary"
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        cleaned,
        cmap="gray"
    )

    axes[1].set_title(
        "2. Kept components"
    )

    axes[1].axis(
        "off"
    )

    axes[2].imshow(
        removed,
        cmap="gray"
    )

    axes[2].set_title(
        "3. Removed components"
    )

    axes[2].axis(
        "off"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "10_v3_text_removal_test.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "완료:"
    )

    print(
        output_path
    )


if __name__ == "__main__":

    main()
