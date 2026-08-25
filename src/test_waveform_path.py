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

    analysis = dark_mask[:, x1:x2]

    score = (
        analysis > 0
    ).sum(axis=1).astype(float)

    kernel = np.ones(21) / 21

    score = np.convolve(
        score,
        kernel,
        mode="same"
    )

    top = int(height * 0.08)
    bottom = int(height * 0.88)

    candidates = np.argsort(
        score[top:bottom]
    )[::-1] + top

    min_distance = int(
        height * 0.12
    )

    rows = []

    for y in candidates:

        y = int(y)

        if all(
            abs(y - old) >= min_distance
            for old in rows
        ):
            rows.append(y)

        if len(rows) == 4:
            break

    rows.sort()

    return rows


# ==========================================================
# 각 column의 ECG 후보를 중심점으로 압축
# ==========================================================

def column_candidates(binary):

    h, w = binary.shape

    all_candidates = []

    for x in range(w):

        ys = np.where(
            binary[:, x] > 0
        )[0]

        if len(ys) == 0:
            all_candidates.append([])
            continue

        # 연속된 픽셀들을 하나의 그룹으로 묶음
        groups = np.split(
            ys,
            np.where(
                np.diff(ys) > 1
            )[0] + 1
        )

        centers = []

        for group in groups:

            if len(group) == 0:
                continue

            centers.append(
                float(
                    np.mean(group)
                )
            )

        all_candidates.append(
            centers
        )

    return all_candidates


# ==========================================================
# Dynamic Programming 기반 전체 경로 탐색
# ==========================================================

def find_best_path(binary):

    h, w = binary.shape

    candidates = column_candidates(
        binary
    )

    # ------------------------------------------------------
    # 후보가 없는 column 때문에 경로가 끊기지 않도록
    # 각 x에서 주변 ±2 column도 같이 확인
    # ------------------------------------------------------

    expanded = []

    for x in range(w):

        nearby = []

        for dx in range(-2, 3):

            xx = x + dx

            if 0 <= xx < w:

                nearby.extend(
                    candidates[xx]
                )

        if len(nearby) > 0:

            nearby = sorted(
                set(
                    round(v, 2)
                    for v in nearby
                )
            )

        expanded.append(
            nearby
        )

    # ------------------------------------------------------
    # DP
    #
    # cost가 작을수록 자연스러운 경로
    # ------------------------------------------------------

    costs = []
    parents = []

    center_y = h / 2

    first_x = None

    for x in range(w):

        if len(expanded[x]) > 0:
            first_x = x
            break

    if first_x is None:

        return np.full(
            w,
            np.nan,
            dtype=float
        )

    first_candidates = np.array(
        expanded[first_x],
        dtype=float
    )

    # 시작점은 중앙에 가까울수록 유리
    first_cost = (
        np.abs(
            first_candidates
            - center_y
        )
        * 0.15
    )

    costs.append(
        first_cost
    )

    parents.append(
        np.full(
            len(first_candidates),
            -1,
            dtype=int
        )
    )

    active_x = [
        first_x
    ]

    previous_candidates = (
        first_candidates
    )

    previous_cost = (
        first_cost
    )

    # ------------------------------------------------------
    # 좌 -> 우 전체 경로 계산
    # ------------------------------------------------------

    for x in range(
        first_x + 1,
        w
    ):

        current = expanded[x]

        if len(current) == 0:
            continue

        current = np.array(
            current,
            dtype=float
        )

        current_cost = np.full(
            len(current),
            np.inf
        )

        current_parent = np.full(
            len(current),
            -1,
            dtype=int
        )

        dx = (
            x
            - active_x[-1]
        )

        for j, y in enumerate(
            current
        ):

            dy = np.abs(
                previous_candidates
                - y
            )

            # ----------------------------------------------
            # 이동 비용
            #
            # 가까운 파형은 낮은 비용
            # 큰 점프는 매우 높은 비용
            # ----------------------------------------------

            transition = (
                dy * 1.0
                + (dy ** 2) * 0.035
            )

            # 여러 column을 건너뛰었다면
            # 어느 정도 이동 허용
            transition = (
                transition
                / max(dx, 1)
            )

            total = (
                previous_cost
                + transition
            )

            best_parent = int(
                np.argmin(total)
            )

            current_cost[j] = (
                total[
                    best_parent
                ]
            )

            current_parent[j] = (
                best_parent
            )

        costs.append(
            current_cost
        )

        parents.append(
            current_parent
        )

        active_x.append(
            x
        )

        previous_candidates = (
            current
        )

        previous_cost = (
            current_cost
        )

    # ------------------------------------------------------
    # Backtracking
    # ------------------------------------------------------

    path = np.full(
        w,
        np.nan,
        dtype=float
    )

    last_index = int(
        np.argmin(
            costs[-1]
        )
    )

    for step in range(
        len(active_x) - 1,
        -1,
        -1
    ):

        x = active_x[step]

        current = np.array(
            expanded[x],
            dtype=float
        )

        if (
            last_index < 0
            or
            last_index >= len(current)
        ):
            break

        path[x] = (
            current[
                last_index
            ]
        )

        last_index = (
            parents[step][
                last_index
            ]
        )

    return path


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

    height, width = (
        image.shape[:2]
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    binary = np.uint8(
        gray < 100
    ) * 255

    rows = detect_ecg_rows(
        binary
    )

    print(
        "Detected rows:",
        rows
    )

    if len(rows) != 4:

        raise RuntimeError(
            "ECG 4개 행 검출 실패"
        )

    # ======================================================
    # V3 위치
    #
    # 3번째 ECG row
    # 3번째 column
    # ======================================================

    row_gaps = np.diff(
        rows
    )

    median_gap = float(
        np.median(
            row_gaps
        )
    )

    half_height = int(
        median_gap * 0.34
    )

    center_y = rows[2]

    y1 = max(
        0,
        center_y - half_height
    )

    y2 = min(
        height,
        center_y + half_height
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
    ).astype(int)

    # V3 = column index 2
    x1 = x_edges[2]
    x2 = x_edges[3]

    crop = binary[
        y1:y2,
        x1:x2
    ]

    # ======================================================
    # 전체 경로 탐색
    # ======================================================

    path = find_best_path(
        crop
    )

    valid = np.isfinite(
        path
    )

    print(
        "V3 crop:",
        crop.shape
    )

    print(
        "Path pixels:",
        int(valid.sum()),
        "/",
        crop.shape[1]
    )

    # ======================================================
    # 결과 표시
    # ======================================================

    plt.figure(
        figsize=(16, 6)
    )

    plt.imshow(
        crop,
        cmap="gray",
        origin="upper"
    )

    xs = np.arange(
        crop.shape[1]
    )

    # IMPORTANT:
    # 연속된 실제 계산점만 표시하고
    # NaN gap을 가로질러 연결하지 않음
    plt.plot(
        xs,
        path,
        linewidth=1.5
    )

    plt.xlim(
        0,
        crop.shape[1]
    )

    plt.ylim(
        crop.shape[0],
        0
    )

    plt.title(
        "V3 - Dynamic Path Test"
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "12_v3_dynamic_path.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "저장:",
        output
    )


if __name__ == "__main__":
    main()