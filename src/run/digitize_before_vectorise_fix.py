# Run the digitization of ECG images.

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import sys

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import rotate

import wfdb

from config import (
    DATASET_NAME,
    IMAGE_TYPE,
    FREQUENCY,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
    SIGNAL_UNITS,
    LEAD_LABEL_MAPPING,
    FMT,
    ADC_GAIN,
    BASELINE,
)


# ==========================================================
# Parse arguments
# ==========================================================

def get_parser():

    description = "Run the trained Challenge models."

    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True,
        help="Folder containing the images to digitize.",
    )

    parser.add_argument(
        "-m",
        "--model_folder",
        type=str,
        required=False,
        default="models/M3/",
        help="Folder containing the nnUNet folder nnUNet_results.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the digitized images.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output.",
    )

    parser.add_argument(
        "--show_image",
        action="store_true",
        default=False,
        help="Show the image with the mask.",
    )

    parser.add_argument(
        "-f",
        "--allow_failures",
        action="store_true",
        default=False,
        help="Allow failures.",
    )

    return parser


# ==========================================================
# Rotation
# ==========================================================

def get_rotation_angle(np_image):

    """Get the rotation angle of the image."""

    lines = get_lines(
        np_image,
        threshold_HoughLines=1200
    )

    filtered_lines = filter_lines(
        lines,
        degree_window=30,
        parallelism_count=3,
        parallelism_window=2
    )

    if filtered_lines is None:

        rot_angle = np.nan

    else:

        rot_angle = get_median_degrees(
            filtered_lines
        )

    return rot_angle


def get_median_degrees(lines):

    """Get the median angle of the lines."""

    lines = lines[:, 0, :]

    line_angles = [
        -(90 - line[1] * 180 / np.pi)
        for line in lines
    ]

    return round(
        np.median(line_angles),
        4
    )


def is_within_x_degrees_of_horizontal(
    theta,
    degree_window
):

    """Check if line is near horizontal."""

    theta_degrees = (
        theta * 180 / np.pi
    )

    deviation_from_horizontal = abs(
        90 - theta_degrees
    )

    return (
        deviation_from_horizontal
        < degree_window
    )


def get_lines(
    np_image,
    threshold_HoughLines=1380,
    rho_resolution=1
):

    """Get lines in image."""

    image = cv2.cvtColor(
        np_image,
        cv2.COLOR_RGB2BGR
    )

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray_image,
        50,
        150,
        apertureSize=3
    )

    lines = cv2.HoughLines(
        edges,
        rho_resolution,
        np.pi / 180,
        threshold_HoughLines,
        None,
        0,
        0
    )

    return lines


def filter_lines(
    lines,
    degree_window=20,
    parallelism_count=0,
    parallelism_window=2
):

    """Filter lines to obtain rotation angle."""

    parallelism_radian = np.deg2rad(
        parallelism_window
    )

    filtered_lines = []
    line_angles = []

    if lines is not None:

        for line in lines:

            for rho, theta in line:

                if is_within_x_degrees_of_horizontal(
                    theta,
                    degree_window
                ):

                    filtered_lines.append(
                        (rho, theta)
                    )

                    line_angles.append(
                        theta
                    )

    parallel_lines = []

    if len(filtered_lines) > 0:

        for rho, theta in filtered_lines:

            count = 0

            for (
                comp_rho,
                comp_theta
            ) in filtered_lines:

                if (
                    abs(
                        theta - comp_theta
                    )
                    < parallelism_radian
                    or
                    abs(
                        (
                            theta
                            - comp_theta
                        )
                        - np.pi
                    )
                    < parallelism_radian
                ):

                    count += 1

            if count >= parallelism_count:

                parallel_lines.append(
                    (rho, theta)
                )

    if len(parallel_lines) == 0:

        parallel_lines = None

    else:

        parallel_lines = np.array(
            parallel_lines
        )[:, np.newaxis, :]

    return parallel_lines


# ==========================================================
# nnU-Net prediction
# ==========================================================

def predict_mask_nnunet(
    image,
    dataset_name,
    model_folder
):

    """Predict mask using nnUNet."""

    temp_folder_input = (
        "data/temp_nnUNet_input"
    )

    temp_folder_output = (
        "data/temp_nnUNet_output"
    )

    image_path_temp = os.path.join(
        temp_folder_input,
        "00000_temp_0000.png"
    )

    mask_path_temp = os.path.join(
        temp_folder_output,
        "00000_temp.png"
    )

    # ------------------------------------------------------
    # nnU-Net result location
    # ------------------------------------------------------

    os.environ[
        "nnUNet_results"
    ] = os.path.join(
        model_folder,
        "nnUNet_results"
    )

    # ------------------------------------------------------
    # M1 / M3 fold 자동 선택
    #
    # M1:
    #   fold_0/checkpoint_final.pth
    #
    # M3:
    #   fold_all/checkpoint_*.pth
    # ------------------------------------------------------

    model_folder_normalized = (
        model_folder
        .replace("\\", "/")
        .rstrip("/")
    )

    if model_folder_normalized.endswith(
        "/M1"
    ):

        fold = "0"

        print(
            "Using model M1 / fold 0"
        )

    else:

        fold = "all"

        print(
            "Using model M3 / fold all"
        )

    # ------------------------------------------------------
    # Temporary folders
    # ------------------------------------------------------

    shutil.rmtree(
        temp_folder_input,
        ignore_errors=True
    )

    shutil.rmtree(
        temp_folder_output,
        ignore_errors=True
    )

    os.makedirs(
        temp_folder_input,
        exist_ok=True
    )

    os.makedirs(
        temp_folder_output,
        exist_ok=True
    )

    write_png(
        image,
        image_path_temp
    )

    # ------------------------------------------------------
    # Run nnU-Net inference
    # ------------------------------------------------------

    if torch.cuda.is_available():

        command_run = (
            f"nnUNetv2_predict "
            f"-d {dataset_name} "
            f"-i {temp_folder_input} "
            f"-o {temp_folder_output} "
            f"-f {fold} "
            f"-tr nnUNetTrainer "
            f"-c 2d "
            f"-p nnUNetPlans"
        )

    else:

        print(
            "CUDA not available. "
            "Running on CPU."
        )

        command_run = (
            f"nnUNetv2_predict "
            f"-d {dataset_name} "
            f"-i {temp_folder_input} "
            f"-o {temp_folder_output} "
            f"-f {fold} "
            f"-tr nnUNetTrainer "
            f"-c 2d "
            f"-p nnUNetPlans "
            f"-device cpu "
            f"--verbose"
        )

    print(
        "nnU-Net command:"
    )

    print(
        command_run
    )

    result = subprocess.run(
        command_run,
        shell=True
    )

    # ------------------------------------------------------
    # nnU-Net 명령 자체가 실패한 경우
    # ------------------------------------------------------

    if result.returncode != 0:

        raise RuntimeError(
            "nnU-Net prediction command failed."
        )

    if not os.path.exists(
        mask_path_temp
    ):

        raise FileNotFoundError(
            "nnU-Net output mask was not created: "
            f"{mask_path_temp}"
        )

    # ------------------------------------------------------
    # Get predicted mask
    # ------------------------------------------------------

    mask = read_image(
        mask_path_temp
    )

    # ------------------------------------------------------
    # Delete temporary files
    # ------------------------------------------------------

    shutil.rmtree(
        temp_folder_input,
        ignore_errors=True
    )

    shutil.rmtree(
        temp_folder_output,
        ignore_errors=True
    )

    return mask


# ==========================================================
# Crop mask
# ==========================================================

def cut_to_mask(
    img,
    mask,
    return_y1=False
):

    """Cut image to non-zero mask."""

    coords = torch.where(
        mask[0] >= 1
    )

    y_min = coords[0].min().item()
    y_max = coords[0].max().item()

    x_min = coords[1].min().item()
    x_max = coords[1].max().item()

    img = img[
        :,
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    if return_y1:

        return (
            img,
            y_min,
            x_min
        )

    return img


# ==========================================================
# Split segmentation labels by lead
# ==========================================================

def cut_binary(
    mask_to_use,
    image_rotated
):

    """Cut binary mask into individual lead masks."""

    signal_masks = {}
    signal_images = {}
    signal_positions = {}

    possible_lead_names = (
        LEAD_LABEL_MAPPING
    )

    lead_names_in_mask = {
        k: v
        for k, v
        in possible_lead_names.items()
    }

    for (
        lead_name,
        lead_value
    ) in lead_names_in_mask.items():

        binary_mask = torch.where(
            mask_to_use == lead_value,
            1,
            0
        )

        if binary_mask.sum() > 0:

            (
                signal_img,
                y1,
                x1
            ) = cut_to_mask(
                image_rotated,
                binary_mask,
                True
            )

            signal_mask = cut_to_mask(
                binary_mask,
                binary_mask
            )

            signal_images[
                lead_name
            ] = signal_img

            signal_masks[
                lead_name
            ] = signal_mask

            signal_positions[
                lead_name
            ] = {
                "y1": y1,
                "x1": x1
            }

        else:

            signal_images[
                lead_name
            ] = None

            signal_masks[
                lead_name
            ] = None

            signal_positions[
                lead_name
            ] = None

    return (
        signal_masks,
        signal_positions,
        signal_images
    )


# ==========================================================
# Vectorisation
# ==========================================================

def vectorise(
    image_rotated,
    mask,
    signal_cropped,
    sec_per_pixel,
    mV_per_pixel,
    y_shift_ratio,
    lead
):
    """
    Vectorise ECG segmentation mask.

    개선점
    1. mask가 없는 x열은 NaN으로 처리
    2. 실제 mask가 존재하는 열만 y 중심값 계산
    3. 짧은/중간 끊김을 선형 보간
    4. 갑작스러운 단발성 이상값 완화
    5. 이후 기존 방식대로 시간축 resampling
    """

    # ======================================================
    # 1. 출력 길이 결정
    # ======================================================

    total_seconds_from_mask = round(
        float(sec_per_pixel) * mask.shape[2],
        1
    )

    if total_seconds_from_mask > (
        LONG_SIGNAL_LENGTH_SEC / 2
    ):
        total_seconds = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio["full"]

    else:
        total_seconds = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio[lead]

    values_needed = int(
        total_seconds * FREQUENCY
    )

    # ======================================================
    # 2. 각 x열에서 실제 mask 중심 y 계산
    # ======================================================

    mask_np = (
        mask[0]
        .detach()
        .cpu()
        .numpy()
    )

    height, width = mask_np.shape

    y_values = np.full(
        width,
        np.nan,
        dtype=np.float32
    )

    for x in range(width):

        ys = np.where(
            mask_np[:, x] > 0
        )[0]

        if len(ys) > 0:

            # 해당 x열에서 segmentation된
            # 픽셀들의 가운데 위치
            y_values[x] = float(
                np.median(ys)
            )

    # ======================================================
    # 3. 유효한 mask가 충분한지 확인
    # ======================================================

    valid = np.where(
        np.isfinite(y_values)
    )[0]

    if len(valid) < 2:

        print(
            f"[{lead}] vectorise 실패: "
            "유효한 mask column이 너무 적습니다."
        )

        return torch.zeros(
            values_needed,
            dtype=torch.float32
        )

    valid_ratio = (
        len(valid) / width
    )

    print(
        f"[{lead}] valid mask columns: "
        f"{len(valid)}/{width} "
        f"({valid_ratio:.1%})"
    )

    # ======================================================
    # 4. 빈 x열 보간
    # ======================================================

    all_x = np.arange(
        width
    )

    interpolated = np.interp(
        all_x,
        valid,
        y_values[valid]
    ).astype(
        np.float32
    )

    # ======================================================
    # 5. 아주 짧은 단발성 이상값 완화
    #
    # QRS 자체를 평평하게 만들면 안 되므로
    # median filter는 매우 약하게 적용
    # ======================================================

    if width >= 5:

        smooth_reference = cv2.medianBlur(
            interpolated.reshape(
                1,
                -1
            ),
            5
        ).reshape(
            -1
        )

        difference = np.abs(
            interpolated
            - smooth_reference
        )

        # 굉장히 튀는 단발점만 교체
        threshold = max(
            10.0,
            float(
                np.percentile(
                    difference,
                    99
                )
            )
        )

        bad = (
            difference > threshold
        )

        # 너무 많은 부분이 bad라면
        # 실제 QRS를 손상시킬 수 있으므로
        # 보정하지 않음
        if np.mean(bad) < 0.02:

            interpolated[
                bad
            ] = smooth_reference[
                bad
            ]

    # ======================================================
    # 6. 이미지 y좌표 → ECG amplitude
    # ======================================================

    signal_cropped_shifted = (
        (
            1
            - y_shift_ratio_
        )
        * image_rotated.shape[1]
        - signal_cropped
    )

    predicted_signal_np = (
        signal_cropped_shifted
        - interpolated
    ) * float(
        mV_per_pixel
    )

    # ======================================================
    # 7. Tensor 변환
    # ======================================================

    predicted_signal = torch.tensor(
        predicted_signal_np,
        dtype=torch.float32
    )

    # ======================================================
    # 8. 시간축 resampling
    # ======================================================

    n = predicted_signal.shape[0]

    if n < 2:

        return torch.zeros(
            values_needed,
            dtype=torch.float32
        )

    data_reshaped = (
        predicted_signal.view(
            1,
            1,
            n
        )
    )

    resampled_data = F.interpolate(
        data_reshaped,
        size=values_needed,
        mode="linear",
        align_corners=False
    )

    predicted_signal_sampled = (
        resampled_data.view(-1)
    )

    return predicted_signal_sampled
    
# ==========================================================
# Plot mask + signals
# ==========================================================

def save_plot_masks_and_signals(
    image,
    masks_cropped,
    mask_start_position,
    signals,
    sig_names,
    output_folder,
    filename="record.png"
):

    try:

        num_signals = (
            signals.shape[1]
        )

    except IndexError:

        print(
            "No signals to plot."
        )

        print(
            f"Signals: {signals}"
        )

        print(
            f"Image shape: {image.shape}"
        )

        return

    fig, axs = plt.subplots(
        1 + num_signals,
        1,
        figsize=(
            10,
            2.5
            * (
                1
                + num_signals
            )
        ),
        gridspec_kw={
            "height_ratios":
                [4]
                + [1] * num_signals
        }
    )

    if hasattr(
        image,
        "numpy"
    ):

        image = image.numpy()

    if (
        image.ndim == 3
        and image.shape[0] == 1
    ):

        image = image.squeeze(
            0
        )

    if (
        image.ndim == 3
        and image.shape[0]
        in [3, 4]
    ):

        image = image.transpose(
            1,
            2,
            0
        )

    if image.ndim == 2:

        mask_combined = np.zeros_like(
            image,
            dtype=np.uint8
        )

    else:

        mask_combined = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

    for (
        lead,
        mask_cropped
    ) in masks_cropped.items():

        if mask_cropped is None:

            continue

        if (
            mask_cropped.ndim == 3
            and mask_cropped.shape[0] == 1
        ):

            mask_cropped = (
                mask_cropped.squeeze(0)
            )

        start_row = (
            mask_start_position[
                lead
            ]["y1"]
        )

        start_col = (
            mask_start_position[
                lead
            ]["x1"]
        )

        (
            mask_height,
            mask_width
        ) = mask_cropped.shape

        current_region = (
            mask_combined[
                start_row:
                    start_row
                    + mask_height,
                start_col:
                    start_col
                    + mask_width
            ]
        )

        mask_combined[
            start_row:
                start_row
                + mask_height,
            start_col:
                start_col
                + mask_width
        ] = np.maximum(
            current_region,
            mask_cropped
        )

    axs[0].imshow(
        image,
        cmap=(
            "gray"
            if image.ndim == 2
            else None
        )
    )

    axs[0].imshow(
        mask_combined,
        cmap="jet",
        alpha=0.5
    )

    axs[0].set_title(
        "Masks overlayed on image"
    )

    axs[0].axis(
        "off"
    )

    time_axis = np.arange(
        signals.shape[0]
    )

    for (
        i,
        signal
    ) in enumerate(
        signals.T
    ):

        axs[i + 1].plot(
            time_axis,
            signal
        )

        axs[i + 1].set_title(
            sig_names[i]
        )

        axs[i + 1].set_xlabel(
            "Time"
        )

        axs[i + 1].set_ylabel(
            "Signal amplitude"
        )

        axs[i + 1].grid()

    plt.tight_layout()

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            output_folder,
            filename
        ),
        dpi=300
    )

    plt.close(
        fig
    )


# ==========================================================
# Main
# ==========================================================

def run(args):

    os.makedirs(
        args.output_folder,
        exist_ok=True
    )

    if args.verbose:

        print(
            "Running digitization model..."
        )

    image_files = [
        f
        for f
        in os.listdir(
            args.data_folder
        )
        if f.endswith(
            f".{IMAGE_TYPE}"
        )
    ]

    for (
        _,
        image_file
    ) in tqdm(
        enumerate(
            image_files
        ),
        total=len(
            image_files
        )
    ):

        image_file_path = (
            os.path.join(
                args.data_folder,
                image_file
            )
        )

        record = image_file.replace(
            f".{IMAGE_TYPE}",
            ""
        )

        os.makedirs(
            args.output_folder,
            exist_ok=True
        )

        image = read_image(
            image_file_path
        )

        # RGBA인 경우 RGB만 사용
        image = image[:3]

        # --------------------------------------------------
        # Rotate
        # --------------------------------------------------

        rot_angle = get_rotation_angle(
            image
            .permute(
                1,
                2,
                0
            )
            .numpy()
            .astype(
                np.uint8
            )
        )

        # Hough line이 없으면 회전하지 않음
        if np.isnan(
            rot_angle
        ):

            rot_angle = 0.0

        print(
            f"Rotation angle: {rot_angle}"
        )

        image_rotated = rotate(
            image,
            rot_angle
        )

        # --------------------------------------------------
        # Segment
        # --------------------------------------------------

        mask_to_use = (
            predict_mask_nnunet(
                image_rotated,
                DATASET_NAME,
                args.model_folder
            )
        )

        write_png(mask_to_use, os.path.join(args.output_folder, f"{record}_raw_mask.png"))

        # --------------------------------------------------
        # DEBUG
        # --------------------------------------------------

        print(
            "\n==================================="
        )

        print(
            f"Record: {record}"
        )

        print(
            "Model folder:",
            args.model_folder
        )

        print(
            "UNIQUE MASK LABELS:",
            torch.unique(
                mask_to_use
            )
        )

        print(
            "MASK SHAPE:",
            tuple(
                mask_to_use.shape
            )
        )

        print(
            "===================================\n"
        )

        # --------------------------------------------------
        # Split lead masks
        # --------------------------------------------------

        (
            signal_masks_cropped,
            signal_positions_cropped,
            _
        ) = cut_binary(
            mask_to_use,
            image_rotated
        )

        available_leads = [
            lead
            for (
                lead,
                mask
            )
            in signal_masks_cropped.items()
            if mask is not None
        ]

        print(
            "Detected leads:",
            available_leads
        )

        print(
            "Detected lead count:",
            len(
                available_leads
            )
        )

        # --------------------------------------------------
        # Vectorise
        # --------------------------------------------------

        x_pixel_list = [
            v.shape[2]
            for v
            in signal_masks_cropped.values()
            if v is not None
        ]

        if len(
            x_pixel_list
        ) == 0:

            print(
                f"=========== "
                f"No valid lead masks for record "
                f"{record}. "
                f"==========="
            )

            print(
                "nnU-Net segmentation 결과에서 "
                "LEAD_LABEL_MAPPING에 해당하는 "
                "lead가 검출되지 않았습니다."
            )

            if args.allow_failures:

                continue

            raise ValueError(
                f"Signal is empty for record "
                f"{record}."
            )

        x_pixel_list_median = (
            np.median(
                x_pixel_list
            )
        )

        valid_widths = [
            v
            for v
            in x_pixel_list
            if (
                v
                <
                2
                * x_pixel_list_median
            )
        ]

        if len(
            valid_widths
        ) == 0:

            raise ValueError(
                "Could not estimate ECG "
                "pixel scaling."
            )

        x_pixel_list_below_2x_median_mean = (
            np.mean(
                valid_widths
            )
        )

        sec_per_pixel = (
            2.5
            /
            x_pixel_list_below_2x_median_mean
        )

        mm_per_pixel = (
            25
            * sec_per_pixel
        )

        sec_per_pixel = (
            mm_per_pixel
            / 25
        )

        mV_per_pixel = (
            mm_per_pixel
            / 10
        )

        signals_predicted = {}

        for (
            lead,
            mask
        ) in signal_masks_cropped.items():

            if mask is not None:

                signals_predicted[
                    lead
                ] = vectorise(
                    image_rotated,
                    mask,
                    signal_positions_cropped[
                        lead
                    ]["y1"],
                    sec_per_pixel,
                    mV_per_pixel,
                    Y_SHIFT_RATIO,
                    lead,
                )

            else:

                signals_predicted[
                    lead
                ] = None

        # --------------------------------------------------
        # Collect signals
        # --------------------------------------------------

        signals = {
            signal_name:
                signals_predicted[
                    signal_name
                ].numpy()

            for signal_name
            in LEAD_LABEL_MAPPING.keys()

            if signals_predicted[
                signal_name
            ] is not None
        }

        num_samples = int(
            LONG_SIGNAL_LENGTH_SEC
            * FREQUENCY
        )

        signal_list = []

        for signal in (
            signals.values()
        ):

            if len(
                signal
            ) < num_samples:

                nan_signal = (
                    np.empty(
                        num_samples
                    )
                )

                nan_signal[:] = np.nan

                nan_signal[
                    :int(
                        len(
                            signal
                        )
                    )
                ] = signal

                signal_list.append(
                    nan_signal
                )

            else:

                signal_list.append(
                    signal
                )

        sig_names = list(
            signals.keys()
        )

        if len(
            signal_list
        ) == 0:

            print(
                f"=========== "
                f"Signal is empty for "
                f"record {record}. "
                f"==========="
            )

            if args.allow_failures:

                continue

            raise ValueError(
                f"Signal is empty for "
                f"record {record}."
            )

        signals = np.array(
            signal_list
        ).T

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        if args.show_image:

            print(
                "Storing image of shape "
                f"{image_rotated.shape}"
            )

            save_plot_masks_and_signals(
                image_rotated,
                signal_masks_cropped,
                signal_positions_cropped,
                signals,
                sig_names,
                args.output_folder,
                f"{record}.png",
            )

        # --------------------------------------------------
        # Save WFDB
        # --------------------------------------------------

        if args.verbose:

            print(
                f"Storing signals for "
                f"record {record} "
                f"with shape "
                f"{signals.shape}"
            )

        if (
            np.nanmax(
                signals
            ) > 10
            or
            np.nanmin(
                signals
            ) < -10
        ):

            print(
                f"Signal out of range "
                f"for record {record}, "
                "normalizing to range "
                "between 1 and -1"
            )

            max_val = np.nanmax(
                signals
            )

            min_val = np.nanmin(
                signals
            )

            if max_val != min_val:

                signals = (
                    (
                        signals
                        - min_val
                    )
                    /
                    (
                        max_val
                        - min_val
                    )
                    * 2
                    - 1
                )

        wfdb.wrsamp(
            record,
            fs=FREQUENCY,
            units=[
                SIGNAL_UNITS
            ] * signals.shape[1],
            sig_name=sig_names,
            p_signal=np.nan_to_num(
                signals
            ),
            write_dir=args.output_folder,
            fmt=[
                FMT
            ] * signals.shape[1],
            adc_gain=[
                ADC_GAIN
            ] * signals.shape[1],
            baseline=[
                BASELINE
            ] * signals.shape[1],
        )

    if args.verbose:

        print(
            "Done."
        )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    run(
        get_parser().parse_args(
            sys.argv[1:]
        )
    )