


from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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
# Lead 이름
# ==========================================================

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


# ==========================================================
# 실행
# ==========================================================

def main():

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"이미지가 없습니다: {INPUT_IMAGE}"
        )

    image = Image.open(
        INPUT_IMAGE
    ).convert(
        "RGB"
    )

    rgb = np.array(
        image
    )

    height, width = (
        rgb.shape[:2]
    )

    print(
        "원본 크기:",
        (width, height)
    )

    # ------------------------------------------------------
    # 테스트용:
    # 상단 3x4 표준 lead 영역을 대략 시각화
    #
    # 아직 최종 정답 mask가 아님
    # ------------------------------------------------------

    preview = rgb.copy()

    # 실제 ECG 본문 기준 대략적인 영역
    top = int(
        height * 0.14
    )

    bottom = int(
        height * 0.55
    )

    left = int(
        width * 0.03
    )

    right = int(
        width * 0.98
    )

    region_width = (
        right - left
    )

    region_height = (
        bottom - top
    )

    cell_width = (
        region_width / 4
    )

    cell_height = (
        region_height / 3
    )

    # ------------------------------------------------------
    # 3 x 4 영역 표시
    # ------------------------------------------------------

    for row in range(3):

        for col in range(4):

            x1 = int(
                left
                + col * cell_width
            )

            x2 = int(
                left
                + (col + 1) * cell_width
            )

            y1 = int(
                top
                + row * cell_height
            )

            y2 = int(
                top
                + (row + 1) * cell_height
            )

            lead_name = (
                LEAD_LAYOUT[
                    row
                ][
                    col
                ]
            )

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 180, 0),
                2
            )

            cv2.putText(
                preview,
                lead_name,
                (
                    x1 + 5,
                    y1 + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 90, 255),
                2,
                cv2.LINE_AA
            )

    output_path = (
        OUTPUT_DIR
        / "01_lead_layout_preview.png"
    )

    Image.fromarray(
        preview
    ).save(
        output_path
    )

    print(
        "저장:",
        output_path
    )


if __name__ == "__main__":
    main()
