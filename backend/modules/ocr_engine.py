"""
OCR 文本检测与识别模块
基于 PaddleOCR，封装检测-识别-后处理流水线
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from backend.config import OCR_CONFIDENCE_THRESHOLD, OCR_USE_ANGLE_CLS

logger = logging.getLogger(__name__)


@dataclass
class OCRBox:
    """单个文本框的识别结果"""
    text: str                       # 识别文本
    confidence: float               # 置信度 [0, 1]
    bbox: List[List[int]]          # 四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    is_low_confidence: bool = False # 是否为低置信度结果


@dataclass
class OCRResult:
    """OCR 完整识别结果"""
    full_text: str                  # 全文（按阅读顺序拼接）
    boxes: List[OCRBox] = field(default_factory=list)
    image_path: str = ""
    page_width: int = 0
    page_height: int = 0

    def get_low_confidence_boxes(self) -> List[OCRBox]:
        """获取所有低置信度的文本框"""
        return [b for b in self.boxes if b.is_low_confidence]

    def get_text_in_region(self, x1: int, y1: int, x2: int, y2: int) -> str:
        """获取指定区域内的文本"""
        texts = []
        for box in self.boxes:
            cx = sum(p[0] for p in box.bbox) / 4
            cy = sum(p[1] for p in box.bbox) / 4
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                texts.append(box.text)
        return "\n".join(texts)


class OCREngine:
    """PaddleOCR 封装引擎"""

    def __init__(self):
        self._ocr = None  # 延迟加载
        self.confidence_threshold = OCR_CONFIDENCE_THRESHOLD

    def _get_ocr(self):
        """延迟加载 PaddleOCR，避免导入时就初始化（慢）"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang="ch",
                    use_textline_orientation=True,   # 文本方向分类
                    text_det_limit_side_len=960,     # 检测边长限制
                    text_rec_score_thresh=0.5         # 识别置信度阈值
                )
                logger.info("PaddleOCR 初始化完成")
            except ImportError:
                raise ImportError(
                    "请安装 PaddleOCR: pip install paddleocr>=2.9.0\n"
                    "以及 PaddlePaddle: pip install paddlepaddle"
                )
        return self._ocr

    def recognize(self, image: np.ndarray, image_path: str = "") -> OCRResult:
        """
        对图像做 OCR 识别

        Args:
            image: OpenCV 图像 (BGR 格式)
            image_path: 图像路径（用于日志和结果标记）

        Returns:
            OCRResult: 包含全文和每个文本框的详细结果
        """
        ocr = self._get_ocr()
        h, w = image.shape[:2]

        # 调用 PaddleOCR
        raw_results = ocr.ocr(image)

        boxes = []
        texts = []

        # PaddleOCR 3.x 返回格式: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        if raw_results and isinstance(raw_results, list):
            for page_result in raw_results:
                if not isinstance(page_result, dict):
                    continue

                rec_texts = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])
                rec_polys = page_result.get("rec_polys", [])

                for i, (text, confidence) in enumerate(zip(rec_texts, rec_scores)):
                    is_low = confidence < self.confidence_threshold

                    # 获取四点坐标
                    bbox_points = rec_polys[i] if i < len(rec_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]

                    box = OCRBox(
                        text=text,
                        confidence=confidence,
                        bbox=[[int(p[0]), int(p[1])] for p in bbox_points],
                        is_low_confidence=is_low
                    )
                    boxes.append(box)
                    texts.append(text)

                    if is_low:
                        logger.debug(
                            f"低置信度文本: '{text}' (confidence={confidence:.2f})"
                        )

        full_text = "\n".join(texts)

        logger.info(
            f"OCR 完成: {len(boxes)} 个文本框, "
            f"其中 {len([b for b in boxes if b.is_low_confidence])} 个低置信度"
        )

        return OCRResult(
            full_text=full_text,
            boxes=boxes,
            image_path=image_path,
            page_width=w,
            page_height=h
        )

    def recognize_file(self, image_path: str) -> OCRResult:
        """直接从文件路径做 OCR"""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.recognize(img, image_path)


# 全局单例
ocr_engine = OCREngine()
