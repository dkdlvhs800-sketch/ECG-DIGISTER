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
# ECG 4개 행 검출
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

    search_top = int(height * 0.10)
    search_bottom = int(height * 0.88)

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
# Baseline 찾기
# ==========================================================

def estimate_baseline(binary_region):

    h, w = binary_region.shape

    # ------------------------------------------------------
    # 좌우 가장자리는 다른 lead/경계가 섞일 수 있으므로 제외
    # ------------------------------------------------------

    margin_x = int(w * 0.08)

    work = binary_region[
        :,
        margin_x:w - margin_x
    ]

    # ------------------------------------------------------
    # 각 y 위치에서 어두운 픽셀이
    # 얼마나 넓은 x 영역에 존재하는지 계산
    #
    # ECG baseline 근처는 여러 x 위치에 반복적으로
    # 파형 픽셀이 존재한다.
    # ------------------------------------------------------

    row_presence = (
        work > 0
    ).sum(axis=1).astype(float)

    # ------------------------------------------------------
    # 세로 방향 smoothing
    # ------------------------------------------------------

    kernel_size = 9

    kernel = (
        np.ones(kernel_size)
        / kernel_size
    )

    score = np.convolve(
        row_presence,
        kernel,
        mode="same"
    )

    # ------------------------------------------------------
    # 글씨가 주로 아래쪽에 있으므로
    # crop 전체를 무작정 검색하지 않고
    # 중앙 영역을 우선 검색
    # ------------------------------------------------------

    search_top = int(h * 0.25)
    search_bottom = int(h * 0.75)

    local_score = score[
        search_top:search_bottom
    ]

    baseline_y = (
        int(np.argmax(local_score))
        + search_top
    )

    return (
        baseline_y,
        row_presence,
        score
    )


# ==========================================================
# Baseline 주변 후보 표시
# ==========================================================

def make_baseline_corridor(
    binary_region,
    baseline_y
):

    h, w = binary_region.shape

    # 현재는 테스트용으로 넉넉하게 설정
    corridor_half_height = max(
        12,
        int(h * 0.18)
    )

    y1 = max(
        0,
        baseline_y - corridor_half_height
    )

    y2 = min(
        h,
        baseline_y + corridor_half_height + 1
    )

    corridor = np.zeros_like(
        binary_region
    )

    corridor[
        y1:y2,
        :
    ] = binary_region[
        y1:y2,
        :
    ]

    return (
        corridor,
        y1,
        y2
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:
        raise FileNotFoundError(
            INPUT_IMAGE
        )

    height, width = image.shape[:2]

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
    # ECG 4개 행
    # ======================================================

    detected_rows = detect_ecg_rows(
        dark_mask
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
    # V3 crop
    #
    # row 3:
    # III | aVF | V3 | V6
    # ======================================================

    row_index = 2
    column_index = 2

    row_gaps = np.diff(
        detected_rows
    )

    median_gap = float(
        np.median(row_gaps)
    )

    half_height = int(
        median_gap * 0.35
    )

    y_center = detected_rows[
        row_index
    ]

    y1 = max(
        0,
        y_center - half_height
    )

    y2 = min(
        height,
        y_center + half_height
    )

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
        + column_index * column_width
    )

    x2 = int(
        ecg_x1
        + (column_index + 1)
        * column_width
    )

    margin = int(
        column_width * 0.02
    )

    x1 += margin
    x2 -= margin

    v3_binary = dark_mask[
        y1:y2,
        x1:x2
    ]

    print(
        "V3 shape:",
        v3_binary.shape
    )

    # ======================================================
    # Baseline 계산
    # ======================================================

    (
        baseline_y,
        row_presence,
        baseline_score
    ) = estimate_baseline(
        v3_binary
    )

    print()
    print(
        "Estimated V3 baseline y:",
        baseline_y
    )

    # ======================================================
    # Corridor
    # ======================================================

    (
        corridor,
        corridor_y1,
        corridor_y2
    ) = make_baseline_corridor(
        v3_binary,
        baseline_y
    )

    print(
        "Corridor:",
        corridor_y1,
        "-",
        corridor_y2
    )

    # ======================================================
    # 저장
    # ======================================================

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "11_v3_baseline_corridor.png"
        ),
        corridor
    )

    # ======================================================
    # 그림
    # ======================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 11)
    )

    # ------------------------------------------------------
    # 원본 + baseline
    # ------------------------------------------------------

    axes[0].imshow(
        v3_binary,
        cmap="gray",
        origin="upper"
    )

    axes[0].axhline(
        baseline_y,
        linewidth=2
    )

    axes[0].axhline(
        corridor_y1,
        linestyle="--",
        linewidth=1
    )

    axes[0].axhline(
        corridor_y2 - 1,
        linestyle="--",
        linewidth=1
    )

    axes[0].set_title(
        "V3 - Estimated baseline and corridor"
    )

    axes[0].set_xlim(
        0,
        v3_binary.shape[1]
    )

    axes[0].set_ylim(
        v3_binary.shape[0],
        0
    )

    # ------------------------------------------------------
    # y별 pixel 수
    # ------------------------------------------------------

    y_axis = np.arange(
        len(row_presence)
    )

    axes[1].plot(
        row_presence,
        y_axis,
        label="Raw row presence"
    )

    axes[1].plot(
        baseline_score,
        y_axis,
        label="Smoothed score"
    )

    axes[1].axhline(
        baseline_y,
        linewidth=2
    )

    axes[1].invert_yaxis()

    axes[1].set_title(
        "Baseline score by Y position"
    )

    axes[1].set_xlabel(
        "Dark pixel count"
    )

    axes[1].set_ylabel(
        "Y"
    )

    axes[1].legend()

    axes[1].grid()

    # ------------------------------------------------------
    # corridor 안에 남은 픽셀
    # ------------------------------------------------------

    axes[2].imshow(
        corridor,
        cmap="gray",
        origin="upper"
    )

    axes[2].axhline(
        baseline_y,
        linewidth=2
    )

    axes[2].set_title(
        "Pixels inside baseline corridor"
    )

    axes[2].set_xlim(
        0,
        corridor.shape[1]
    )

    axes[2].set_ylim(
        corridor.shape[0],
        0
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "11_v3_baseline_test.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "완료:",
        output_path
    )


if __name__ == "__main__":
    main()
