import os
import cv2
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Tesseractのパスを設定
tesseract_cmd = r'C:\2170114\djangoProject\tesseract\tesseract.exe'
pyocr.tesseract.TESSERACT_CMD = tesseract_cmd

def preprocess_image(image_path):
    try:
        # 画像を読み込む
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Error: Image not loaded.")
            return None

        # 画像をリサイズ
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

        # 画像をグレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # デバッグ: グレースケール画像を保存
        temp_gray_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_gray_file.name, gray)
        print(f"Grayscale image saved to: {temp_gray_file.name}")

        # 明るさとコントラストの調整
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # ノイズ除去
        gray = cv2.medianBlur(gray, 5)

        # 二値化
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # デバッグ: 二値化画像を保存
        temp_binary_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_binary_file.name, binary)
        print(f"Preprocessed (binary) image saved to: {temp_binary_file.name}")

        # グレースケール画像を表示
        gray_img = cv2.imread(temp_gray_file.name, cv2.IMREAD_GRAYSCALE)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Grayscale Image')
        plt.show()

        # 二値化画像を表示
        binary_img = cv2.imread(temp_binary_file.name, cv2.IMREAD_GRAYSCALE)
        plt.imshow(binary_img, cmap='gray')
        plt.title('Binary Image')
        plt.show()

        return temp_binary_file.name

    except Exception as e:
        print(f"Exception in preprocess_image: {e}")
        return None

def extract_text_from_image(image_path):
    try:
        # 画像を前処理する
        temp_image_path = preprocess_image(image_path)

        if temp_image_path is None:
            return "Error: Image preprocessing failed."

        # OCRツールを取得
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise RuntimeError("No OCR tool found. Please install Tesseract-OCR.")

        tool = tools[0]  # 最初のツールを選択
        lang = 'jpn'

        # 一時ファイルとして保存された画像をPIL画像に変換
        pil_img = Image.open(temp_image_path)

        # OCRを使用してテキストを抽出
        text = tool.image_to_string(
            pil_img,
            lang=lang,
            builder=pyocr.builders.TextBuilder()
        )

        # デバッグ: 抽出されたテキストを出力
        print(f"Extracted text: {text}")

        return text

    except Exception as e:
        print(f"Exception in extract_text_from_image: {e}")
        return "Error: OCR processing failed."
