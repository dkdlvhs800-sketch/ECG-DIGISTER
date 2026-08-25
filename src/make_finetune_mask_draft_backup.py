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
# Lead label
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

    # 좌우 가장자리 제외
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
    # 1. 검은 ECG 후보 픽셀
    # ======================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 선명한 검정/회색 선 위주
    dark_mask = np.where(
        gray < 100,
        255,
        0
    ).astype(
        np.uint8
    )

    # ======================================================
    # 2. 4개 ECG 행 자동 찾기
    # ======================================================

    detected_rows = detect_ecg_rows(
        dark_mask
    )

    print(
        "자동 검출된 행:",
        detected_rows
    )

    if len(detected_rows) != 4:

        raise RuntimeError(
            "ECG 행 4개를 찾지 못했습니다."
        )

    # ======================================================
    # 3. 행별 영역 계산
    # ======================================================

    row_gaps = np.diff(
        detected_rows
    )

    median_gap = float(
        np.median(row_gaps)
    )

    half_height = int(
        median_gap * 0.34
    )

    # 좌우 분석 범위
    left = int(
        width * 0.055
    )

    right = int(
        width * 0.985
    )

    usable_width = (
        right - left
    )

    # 3x4 layout이므로
    # 현재 한 장의 annotation을 만들기 위해
    # 각 row 안을 4 lead 영역으로 나눔
    x_edges = np.linspace(
        left,
        right,
        5
    ).astype(int)

    # ======================================================
    # 4. label mask 생성
    #
    # 0 = background
    # 1~12 = lead
    # ======================================================

    label_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # ------------------------------------------------------
    # 표준 3개 행
    # ------------------------------------------------------

    for row_index in range(3):

        center_y = (
            detected_rows[
                row_index
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

        for col_index in range(4):

            lead = (
                LEAD_LAYOUT[
                    row_index
                ][
                    col_index
                ]
            )

            label_value = (
                LEAD_LABELS[
                    lead
                ]
            )

            x1 = (
                x_edges[
                    col_index
                ]
            )

            x2 = (
                x_edges[
                    col_index + 1
                ]
            )

            region = dark_mask[
                y1:y2,
                x1:x2
            ]

            # 검은 픽셀 위치에만
            # 해당 lead label 부여
            target = label_mask[
                y1:y2,
                x1:x2
            ]

            target[
                region > 0
            ] = label_value

            print(
                f"{lead}:",
                f"x={x1}:{x2}",
                f"y={y1}:{y2}",
                "label=",
                label_value
            )

    # ------------------------------------------------------
    # 아래 II rhythm strip
    # ------------------------------------------------------

    rhythm_y = (
        detected_rows[3]
    )

    rhythm_y1 = max(
        0,
        rhythm_y - half_height
    )

    rhythm_y2 = min(
        height,
        rhythm_y + half_height
    )

    rhythm_region = dark_mask[
        rhythm_y1:rhythm_y2,
        left:right
    ]

    rhythm_target = label_mask[
        rhythm_y1:rhythm_y2,
        left:right
    ]

    rhythm_target[
        rhythm_region > 0
    ] = LEAD_LABELS["II"]

    print(
        "II rhythm:",
        f"x={left}:{right}",
        f"y={rhythm_y1}:{rhythm_y2}"
    )

    # ======================================================
    # 5. 실제 mask 파일 저장
    # ======================================================

    mask_path = (
        MASK_DIR
        / "EKG_draft_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        label_mask
    )

    # ======================================================
    # 6. 색깔 overlay 생성
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
        label_mask == 0,
        label_mask
    )

    plt.imshow(
        masked,
        cmap="tab20",
        vmin=0,
        vmax=12,
        alpha=0.75
    )

    colorbar = plt.colorbar(
        ticks=range(1, 13)
    )

    colorbar.set_label(
        "Lead label"
    )

    plt.title(
        "Draft 12-lead Ground Truth Mask"
    )

    plt.axis(
        "off"
    )

    overlay_path = (
        OUTPUT_DIR
        / "06_draft_mask_overlay.png"
    )

    plt.tight_layout()

    plt.savefig(
        overlay_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # 7. label 확인
    # ======================================================

    unique_labels = np.unique(
        label_mask
    )

    print()
    print(
        "생성된 label:",
        unique_labels
    )

    print(
        "mask 초안:",
        mask_path
    )

    print(
        "확인용 overlay:",
        overlay_path
    )

    print()
    print(
        "※ 아직 학습에 사용하지 마세요."
    )

    print(
        "※ 글자/잡선이 mask에 포함됐는지 먼저 확인해야 합니다."
    )


if __name__ == "__main__":
    main()