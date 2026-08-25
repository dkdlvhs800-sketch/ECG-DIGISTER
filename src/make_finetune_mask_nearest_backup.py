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

MASK_DIR = (
    PROJECT_ROOT
    / "finetune_data"
    / "masks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Lead 정보
# ==========================================================

LEAD_LABELS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "aVR": 4,
    "aVL": 5,
    "aVF": 6,
    "V1": 7,
    "V2": 8,
    "V3": 9,
    "V4": 10,
    "V5": 11,
    "V6": 12,
}

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


# ==========================================================
# ECG 행 자동 검출
# ==========================================================

def detect_ecg_rows(dark_mask):

    height, width = dark_mask.shape

    x1 = int(width * 0.07)
    x2 = int(width * 0.98)

    analysis = dark_mask[:, x1:x2]

    row_score = (
        analysis > 0
    ).sum(axis=1).astype(float)

    kernel = np.ones(21) / 21

    smooth = np.convolve(
        row_score,
        kernel,
        mode="same"
    )

    search_top = int(
        height * 0.08
    )

    search_bottom = int(
        height * 0.88
    )

    candidates = np.argsort(
        smooth[
            search_top:search_bottom
        ]
    )[::-1]

    candidates += search_top

    min_distance = int(
        height * 0.12
    )

    rows = []

    for y in candidates:

        y = int(y)

        if all(
            abs(y - old_y) >= min_distance
            for old_y in rows
        ):

            rows.append(y)

        if len(rows) == 4:
            break

    rows.sort()

    return rows


# ==========================================================
# 한 lead 영역에서 파형 추적
# ==========================================================

def trace_lead_waveform(binary_region):

    h, w = binary_region.shape

    traced = np.full(
        w,
        np.nan,
        dtype=np.float32
    )

    # 시작 위치는 중앙
    previous_y = h // 2

    max_jump = max(
        6,
        int(h * 0.18)
    )

    for x in range(w):

        ys = np.where(
            binary_region[:, x] > 0
        )[0]

        if len(ys) == 0:
            continue

        # 이전 위치와 가장 가까운 점 우선
        distances = np.abs(
            ys - previous_y
        )

        order = np.argsort(
            distances
        )

        chosen = None

        for idx in order:

            candidate_y = int(
                ys[idx]
            )

            if (
                abs(
                    candidate_y
                    - previous_y
                )
                <= max_jump
            ):

                chosen = candidate_y
                break

        if chosen is not None:

            traced[x] = chosen
            previous_y = chosen

    # ======================================================
    # 짧은 빈 구간만 보간
    # ======================================================

    valid = np.where(
        np.isfinite(traced)
    )[0]

    if len(valid) < 2:
        return traced

    interpolated = np.interp(
        np.arange(w),
        valid,
        traced[valid]
    )

    result = traced.copy()

    max_gap = 10

    for i in range(
        len(valid) - 1
    ):

        left = valid[i]
        right = valid[i + 1]

        gap = (
            right - left
        )

        if gap <= max_gap:

            result[
                left:right + 1
            ] = interpolated[
                left:right + 1
            ]

    return result


# ==========================================================
# 추적 선 → mask 픽셀
# ==========================================================

def trace_to_mask(
    traced,
    height,
    thickness=2
):

    width = len(
        traced
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for x, y in enumerate(
        traced
    ):

        if not np.isfinite(
            y
        ):
            continue

        y = int(
            round(y)
        )

        y1 = max(
            0,
            y - thickness
        )

        y2 = min(
            height,
            y + thickness + 1
        )

        mask[
            y1:y2,
            x
        ] = 255

    return mask


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
    ).astype(
        np.uint8
    )

    # ======================================================
    # 2. ECG 행 검출
    # ======================================================

    detected_rows = detect_ecg_rows(
        dark_mask
    )

    print(
        "자동 검출된 ECG 행:",
        detected_rows
    )

    if len(detected_rows) != 4:

        raise RuntimeError(
            "ECG 행 4개를 찾지 못했습니다."
        )

    # ======================================================
    # 3. 3개 표준 행의 높이 계산
    # ======================================================

    row_gaps = np.diff(
        detected_rows
    )

    median_gap = float(
        np.median(
            row_gaps
        )
    )

    half_height = int(
        median_gap * 0.34
    )

    left = int(
        width * 0.055
    )

    right = int(
        width * 0.985
    )

    x_edges = np.linspace(
        left,
        right,
        5
    ).astype(
        int
    )

    # ======================================================
    # 4. 최종 12-class mask
    # ======================================================

    final_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # 시각화용
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 10)
    )

    # ======================================================
    # 5. 각 lead 독립 추적
    # ======================================================

    for row_idx in range(3):

        center_y = (
            detected_rows[
                row_idx
            ]
        )

        y1 = max(
            0,
            center_y - half_height
        )

        y2 = min(
            height,
            center_y + half_height
        )

        for col_idx in range(4):

            lead = (
                LEAD_LAYOUT[
                    row_idx
                ][
                    col_idx
                ]
            )

            label = (
                LEAD_LABELS[
                    lead
                ]
            )

            x1 = (
                x_edges[
                    col_idx
                ]
            )

            x2 = (
                x_edges[
                    col_idx + 1
                ]
            )

            region = dark_mask[
                y1:y2,
                x1:x2
            ]

            traced = trace_lead_waveform(
                region
            )

            traced_mask = trace_to_mask(
                traced,
                region.shape[0],
                thickness=1
            )

            target = final_mask[
                y1:y2,
                x1:x2
            ]

            target[
                traced_mask > 0
            ] = label

            # ==============================================
            # Debug plot
            # ==============================================

            ax = axes[
                row_idx,
                col_idx
            ]

            ax.imshow(
                region,
                cmap="gray"
            )

            valid = np.isfinite(
                traced
            )

            xs = np.arange(
                region.shape[1]
            )

            ax.plot(
                xs[valid],
                traced[valid],
                linewidth=1
            )

            ax.set_title(
                lead
            )

            ax.axis(
                "off"
            )

            print(
                f"{lead}: "
                f"추적 {int(valid.sum())}"
                f"/{region.shape[1]} columns"
            )

    # ======================================================
    # 6. mask 저장
    # ======================================================

    mask_path = (
        MASK_DIR
        / "EKG_individual_trace_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        final_mask
    )

    # ======================================================
    # 7. 12개 lead 추적 결과 저장
    # ======================================================

    plt.tight_layout()

    trace_path = (
        OUTPUT_DIR
        / "10_individual_lead_tracking.png"
    )

    plt.savefig(
        trace_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # 8. overlay
    # ======================================================

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(16, 12)
    )

    plt.imshow(
        rgb
    )

    masked = np.ma.masked_where(
        final_mask == 0,
        final_mask
    )

    plt.imshow(
        masked,
        cmap="tab20",
        vmin=0,
        vmax=12,
        alpha=0.85
    )

    plt.colorbar(
        ticks=range(1, 13)
    )

    plt.title(
        "Individual Lead Trace Mask"
    )

    plt.axis(
        "off"
    )

    overlay_path = (
        OUTPUT_DIR
        / "10_individual_trace_overlay.png"
    )

    plt.tight_layout()

    plt.savefig(
        overlay_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # 9. 확인
    # ======================================================

    print()
    print(
        "생성된 labels:",
        np.unique(
            final_mask
        )
    )

    print(
        "mask:",
        mask_path
    )

    print(
        "tracking:",
        trace_path
    )

    print(
        "overlay:",
        overlay_path
    )

    print()
    print(
        "※ 아직 학습에 사용하지 마세요."
    )


if __name__ == "__main__":
    main()