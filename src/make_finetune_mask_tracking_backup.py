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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 파형 경로 추적 함수
# ==========================================================

def trace_waveform(binary_row):

    h, w = binary_row.shape

    # 각 x좌표에서 검출된 흰 픽셀의 y 위치
    candidates = []

    for x in range(w):

        ys = np.where(
            binary_row[:, x] > 0
        )[0]

        candidates.append(ys)

    # ------------------------------------------------------
    # 시작점 찾기
    # ------------------------------------------------------

    center_y = h // 2

    previous_y = center_y

    traced_y = np.full(
        w,
        np.nan,
        dtype=float
    )

    # ------------------------------------------------------
    # 왼쪽 -> 오른쪽 추적
    # ------------------------------------------------------

    max_jump = max(
        8,
        int(h * 0.20)
    )

    for x in range(w):

        ys = candidates[x]

        if len(ys) == 0:
            continue

        # 직전 위치와 가장 가까운 픽셀 선택
        distances = np.abs(
            ys - previous_y
        )

        best_index = np.argmin(
            distances
        )

        best_y = int(
            ys[best_index]
        )

        # 너무 멀리 떨어진 점은
        # 글자/잡음일 가능성이 있으므로 무시
        if abs(best_y - previous_y) <= max_jump:

            traced_y[x] = best_y
            previous_y = best_y

    # ------------------------------------------------------
    # 짧은 빈 구간 보간
    # ------------------------------------------------------

    valid = np.where(
        ~np.isnan(traced_y)
    )[0]

    if len(valid) >= 2:

        interpolated = np.interp(
            np.arange(w),
            valid,
            traced_y[valid]
        )

        # 실제 검출점에서 너무 멀리 떨어진
        # 큰 공백은 그대로 두기 위해
        # 짧은 gap만 채움
        max_gap = 12

        filled = traced_y.copy()

        for i in range(len(valid) - 1):

            left = valid[i]
            right = valid[i + 1]

            gap = right - left

            if gap <= max_gap:

                filled[left:right + 1] = (
                    interpolated[left:right + 1]
                )

        traced_y = filled

    return traced_y


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

    height, width = image.shape[:2]

    print(
        "이미지 크기:",
        width,
        "x",
        height
    )

    # ======================================================
    # 1. 어두운 픽셀 검출
    # ======================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = np.where(
        gray < 100,
        255,
        0
    ).astype(np.uint8)

    # ======================================================
    # 2. ECG 행 자동 검출
    # ======================================================

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

    print(
        "자동 검출된 ECG 행:",
        detected_rows
    )

    # ======================================================
    # 3. 행 crop 높이 결정
    # ======================================================

    row_gaps = np.diff(
        detected_rows
    )

    median_gap = float(
        np.median(row_gaps)
    )

    half_height = int(
        median_gap * 0.35
    )

    # ======================================================
    # 4. 각 행에서 파형 추적
    # ======================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10)
    )

    for index, y_center in enumerate(
        detected_rows
    ):

        y_top = max(
            0,
            y_center - half_height
        )

        y_bottom = min(
            height,
            y_center + half_height
        )

        row_mask = dark_mask[
            y_top:y_bottom,
            x1:x2
        ]

        traced = trace_waveform(
            row_mask
        )

        axes[index].imshow(
            row_mask,
            cmap="gray",
            origin="upper"
        )

        valid = ~np.isnan(
            traced
        )

        x_values = np.arange(
            row_mask.shape[1]
        )

        axes[index].plot(
            x_values[valid],
            traced[valid],
            linewidth=1
        )

        axes[index].set_title(
            f"Waveform Tracking Test - Row {index + 1}"
        )

        axes[index].set_xlim(
            0,
            row_mask.shape[1]
        )

        axes[index].set_ylim(
            row_mask.shape[0],
            0
        )

        axes[index].axis("off")

        print(
            f"ROW {index + 1}:",
            "추적 픽셀 =",
            int(valid.sum()),
            "/",
            row_mask.shape[1]
        )

    # ======================================================
    # 5. 저장
    # ======================================================

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "05_waveform_tracking_test.png"
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