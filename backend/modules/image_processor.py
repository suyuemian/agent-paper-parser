"""
图像预处理模块
针对老旧扫描件和模糊拍照场景的预处理流水线

处理流程：
    原始图像 → 灰度归一化 → 自适应二值化 → 几何校正 → 去噪增强 → [超分辨率]
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


class ImagePreprocessor:
    """图像预处理流水线"""

    def __init__(self, enable_super_res: bool = False):
        """
        Args:
            enable_super_res: 是否启用超分辨率（计算密集，按需开启）
        """
        self.enable_super_res = enable_super_res

    def process(self, image_path: str) -> np.ndarray:
        """
        完整的预处理流水线

        Args:
            image_path: 图像文件路径

        Returns:
            预处理后的图像 (numpy array, BGR格式)
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # Step 1: 灰度归一化 - 消除光照不均
        img = self._normalize_lighting(img)

        # Step 2: 自适应二值化 - 分离文字与背景
        binary = self._adaptive_binarize(img)

        # Step 3: 几何校正 - 矫正倾斜
        img = self._correct_skew(img)

        # Step 4: 去噪增强
        img = self._denoise_and_sharpen(img)

        # Step 5: 超分辨率（按需）
        if self.enable_super_res:
            img = self._super_resolution(img)

        return img

    def _normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """
        灰度归一化：使用 CLAHE 均衡化，消除不均匀光照
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def _adaptive_binarize(self, img: np.ndarray) -> np.ndarray:
        """
        自适应二值化：Sauvola 算法思想，对古籍发黄纸张效果好
        这里使用 OpenCV 的 adaptiveThreshold 作为简化实现
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 大核高斯自适应阈值，适合不均匀背景
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,   # 大核适应缓慢变化的背景
            C=10            # 常数偏移，值越大文字越细
        )
        return binary

    def _correct_skew(self, img: np.ndarray) -> np.ndarray:
        """
        几何校正：霍夫变换检测文本行角度，旋转矫正
        对于严重倾斜的拍照扫描件有重要作用
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

        if lines is None:
            return img  # 无明显直线，不矫正

        # 计算所有直线的平均角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta) - 90  # 转为相对于垂直方向的角度
            if -45 < angle < 45:  # 只考虑合理范围内的倾斜
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)

        # 角度小于 0.5 度则不矫正
        if abs(median_angle) < 0.5:
            return img

        # 旋转矫正
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            img, rotation_matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return corrected

    def _denoise_and_sharpen(self, img: np.ndarray) -> np.ndarray:
        """
        去噪 + 锐化：中值滤波去噪，然后拉普拉斯锐化
        """
        # 中值滤波 - 去除椒盐噪声，同时保护边缘
        denoised = cv2.medianBlur(img, 3)

        # 拉普拉斯锐化 - 增强文字边缘
        kernel = np.array([
            [0, -1,  0],
            [-1,  5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def _super_resolution(self, img: np.ndarray) -> np.ndarray:
        """
        超分辨率重建（占位）
        实际使用时接入 Real-ESRGAN 模型

        当前简化实现：双三次插值上采样 + 锐化
        """
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # 额外锐化补偿插值模糊
        kernel = np.array([
            [0, -0.5,  0],
            [-0.5,  3, -0.5],
            [0, -0.5,  0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(upscaled, -1, kernel)

        return sharpened


def load_image_as_pil(image_path: str) -> Image.Image:
    """将图像加载为 PIL Image（用于 CLIP/Qwen-VL 输入）"""
    return Image.open(image_path).convert("RGB")


def save_processed_image(img: np.ndarray, output_path: str) -> str:
    """保存处理后的图像"""
    cv2.imwrite(output_path, img)
    return output_path
