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

        # ガウシアンブラーを適用してノイズを除去
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # コントラストと明るさの調整
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)

        # デバッグ: グレースケール画像を保存
        temp_gray_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_gray_file.name, gray)
        print(f"Grayscale image saved to: {temp_gray_file.name}")

        # 適応的な二値化を適用
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # モルフォロジー変換を適用して文字を強調
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # シャープ化フィルターを適用
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(morphed, -1, sharpen_kernel)

        # デバッグ: 最終処理画像を保存
        temp_final_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_final_file.name, sharpened)
        print(f"Final preprocessed image saved to: {temp_final_file.name}")

        # 処理結果の表示
        plt.imshow(sharpened, cmap='gray')
        plt.title('Final Preprocessed Image')
        plt.show()

        return temp_final_file.name

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
        lang = 'eng+jpn'  # 日本語と英数字を指定

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
