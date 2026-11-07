from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_KERNEL_SIZE = 31
DEFAULT_KERNEL = "0, -1, 0\n-1, 5, -1\n0, -1, 0"


def parse_kernel(text: str) -> np.ndarray:
    """Parse a square, odd-sized kernel from comma- or whitespace-separated rows."""
    rows = [
        [float(value) for value in re.split(r"[\s,]+", line.strip()) if value]
        for line in text.replace(";", "\n").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("請輸入卷積核數值。")
    if any(len(row) != len(rows) for row in rows):
        raise ValueError("卷積核必須是方形矩陣，每列的數值個數需相同。")
    if len(rows) % 2 == 0:
        raise ValueError("卷積核邊長必須是奇數（例如 1、3、5）。")
    if len(rows) > MAX_KERNEL_SIZE:
        raise ValueError(f"卷積核最大支援 {MAX_KERNEL_SIZE} × {MAX_KERNEL_SIZE}。")

    kernel = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(kernel).all():
        raise ValueError("卷積核只能包含有限數值，不能使用 NaN 或無限大。")
    if np.max(np.abs(kernel)) > 1_000_000:
        raise ValueError("單一權重絕對值不可超過 1,000,000。")
    return kernel


def filter_image(
    image: np.ndarray, kernel: np.ndarray, normalize: bool
) -> Tuple[np.ndarray, bool]:
    """Filter an RGB uint8 image; return the result and whether normalization was used."""
    applied_kernel = kernel.astype(np.float32)
    normalized = False
    if normalize:
        kernel_sum = float(kernel.sum())
        if abs(kernel_sum) < 1e-12:
            raise ValueError(
                "此卷積核權重總和為 0，無法正規化；請取消正規化後再套用。"
            )
        applied_kernel /= kernel_sum
        normalized = True

    filtered = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=applied_kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.clip(filtered, 0, 255).astype(np.uint8), normalized


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate RGB PSNR in dB using a peak intensity of 255."""
    difference = original.astype(np.float64) - processed.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0**2) / mse)


def get_display_channels(image: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    if mode == "Original":
        return {"R": image[:, :, 0], "G": image[:, :, 1], "B": image[:, :, 2]}
    if mode in ("R", "G", "B"):
        channel = {"R": 0, "G": 1, "B": 2}[mode]
        return {mode: image[:, :, channel]}
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    channel = {"Y": 0, "U": 1, "V": 2}[mode]
    return {mode: yuv[:, :, channel]}


def draw_histogram(channels: Dict[str, np.ndarray]) -> plt.Figure:
    colors = {"R": "#e74c3c", "G": "#27ae60", "B": "#3498db", "Y": "#555555",
              "U": "#8e44ad", "V": "#e67e22"}
    figure, axis = plt.subplots(figsize=(10, 3))
    for name, channel in channels.items():
        counts = np.bincount(channel.ravel(), minlength=256)
        axis.plot(np.arange(256), counts, color=colors[name], label=name, linewidth=1.4)
    axis.set_xlim(0, 255)
    axis.set_xlabel("Pixel intensity (0–255)")
    axis.set_ylabel("Pixel count")
    axis.set_title("Intensity histogram")
    axis.grid(alpha=0.2)
    if len(channels) > 1:
        axis.legend()
    figure.tight_layout()
    return figure


def detect_faces(image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"無法載入 OpenCV Haar 分類器：{cascade_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boxes, _, margins = detector.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        outputRejectLevels=True,
    )
    annotated = image.copy()
    detections: List[Tuple[int, int, int, int, float]] = []
    for (x, y, width, height), margin in zip(boxes, margins):
        # Haar decision margins are not calibrated probabilities; map them to a readable score.
        bounded_margin = float(np.clip(np.asarray(margin).item(), -50, 50))
        confidence = 1.0 / (1.0 + np.exp(-bounded_margin))
        x, y, width, height = int(x), int(y), int(width), int(height)
        detections.append((x, y, width, height, confidence))
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 220, 80), 2)
        cv2.putText(
            annotated,
            f"{confidence:.0%}*",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return annotated, detections


def load_uploaded_image(image_bytes: bytes) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source_format = source.format or "Unknown"
            source_mode = source.mode
            image = ImageOps.exif_transpose(source).convert("RGB")
            return np.asarray(image, dtype=np.uint8), source_mode, image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"影像無法讀取或檔案內容不完整：{error}") from error


def main() -> None:
    st.set_page_config(page_title="互動式影像處理與人臉分析", layout="wide")
    st.title("互動式影像處理與人臉分析")
    st.caption("上傳影像後可切換色彩通道、檢視直方圖、自訂空間濾波，並執行 Haar 人臉偵測。")

    uploaded_file = st.sidebar.file_uploader(
        "載入影像（PNG、JPG/JPEG、BMP）",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded_file is None:
        st.info("請從左側上傳 PNG、JPG/JPEG 或 BMP 影像以開始分析。")
        return

    image_bytes = uploaded_file.getvalue()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.error("檔案超過 20 MiB 上限，請縮小影像後重新上傳。")
        return
    try:
        original, source_mode, (width, height) = load_uploaded_image(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    st.sidebar.subheader("影像資訊")
    st.sidebar.write(f"檔名：{uploaded_file.name}")
    st.sidebar.write(f"解析度：{width} × {height} px")
    st.sidebar.write(f"原始色彩模式：{source_mode}；分析影像：RGB 三通道")
    st.sidebar.write(f"檔案大小：{len(image_bytes) / 1024:.1f} KiB")

    mode = st.sidebar.selectbox(
        "顯示模式",
        ["Original", "R", "G", "B", "Y", "U", "V"],
        format_func={
            "Original": "原始影像（Original）",
            "R": "R 通道",
            "G": "G 通道",
            "B": "B 通道",
            "Y": "Y 亮度分量",
            "U": "U 色度分量",
            "V": "V 色度分量",
        }.get,
    )
    view_tab, filter_tab, face_tab = st.tabs(
        ["影像檢視與直方圖", "自訂卷積與 PSNR", "人臉偵測"]
    )

    with view_tab:
        st.subheader("同步色彩通道檢視")
        if mode == "Original":
            st.image(original, caption="Original（RGB）", use_container_width=True)
        else:
            display_channel = get_display_channels(original, mode)[mode]
            st.image(display_channel, caption=f"{mode} 通道灰階檢視", use_container_width=True)
        st.subheader(f"{mode} 強度直方圖")
        st.pyplot(draw_histogram(get_display_channels(original, mode)), clear_figure=True)
        st.caption("橫軸：像素強度 0–255；縱軸：像素數量。")

    with filter_tab:
        st.subheader("自訂空間濾波")
        kernel_text = st.text_area(
            "輸入奇數邊長方形卷積核；每列以逗號或空白分隔，也可用分號分隔列",
            value=DEFAULT_KERNEL,
            height=130,
        )
        normalize = st.checkbox("以權重總和進行正規化", value=True)
        try:
            kernel = parse_kernel(kernel_text)
            filtered, normalized = filter_image(original, kernel, normalize)
        except ValueError as error:
            st.error(str(error))
        else:
            psnr = calculate_psnr(original, filtered)
            st.caption(
                f"Kernel：{kernel.shape[0]} × {kernel.shape[1]}；"
                f"{'已正規化' if normalized else '未正規化'}；"
                "邊界採反射填補。"
            )
            st.metric("PSNR", "∞ dB（影像完全相同）" if np.isinf(psnr) else f"{psnr:.2f} dB")
            before, after = st.columns(2)
            with before:
                st.image(original, caption="濾波前", use_container_width=True)
            with after:
                st.image(filtered, caption="濾波後", use_container_width=True)

    with face_tab:
        st.subheader("Haar Cascade 人臉偵測")
        st.caption(
            "分數由 Haar 分類器的決策邊際轉換為 0–100% 顯示，並非經校準的機率。"
        )
        if st.button("開始偵測", type="primary"):
            try:
                annotated, detections = detect_faces(original)
            except (cv2.error, RuntimeError) as error:
                st.error(f"人臉偵測失敗：{error}")
            else:
                st.session_state["face_result"] = {
                    "image": image_bytes,
                    "annotated": annotated,
                    "detections": detections,
                }

        result = st.session_state.get("face_result")
        if result is not None and result["image"] == image_bytes:
            st.metric("偵測到的人臉數量", len(result["detections"]))
            st.image(result["annotated"], caption="人臉邊界框與 Haar 邊際分數")
            if result["detections"]:
                for index, (x, y, box_width, box_height, confidence) in enumerate(
                    result["detections"], start=1
                ):
                    st.write(
                        f"人臉 {index}：x={x}, y={y}, "
                        f"寬={box_width}, 高={box_height}；邊際分數={confidence:.1%}*"
                    )
            else:
                st.info("此影像未偵測到人臉。")


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(Path(__file__).resolve()),
                ],
                cwd=Path(__file__).resolve().parent,
            )
        )
    main()