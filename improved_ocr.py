import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive thresholding improves readability for OCR
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return gray


def ocr_image(img):
    return pytesseract.image_to_string(img, lang="eng")


def process_file(path):
    path = Path(path)
    texts = []
    if path.suffix.lower() == ".pdf":
        images = convert_from_path(path)
        for idx, page in enumerate(images, 1):
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            pre = preprocess_image(img)
            text = ocr_image(pre)
            texts.append(text)
    else:
        img = cv2.imread(str(path))
        pre = preprocess_image(img)
        text = ocr_image(pre)
        texts.append(text)
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Simple OCR pipeline")
    parser.add_argument("input", help="Path to image or PDF")
    parser.add_argument("-o", "--output", help="File to write OCR text")
    args = parser.parse_args()

    text = process_file(args.input)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
