import cv2
import numpy as np
import pytesseract
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Tesseractの実行ファイルのパスを設定
tesseract_cmd = r'C:\2170114\djangoProject\tesseract\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def preprocess_image(image_path):
    try:
        # 画像をファイルから読み込む。np.fromfileを使用して、ファイルパスに日本語が含まれていても正しく読み込めるようにする
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Error: Image not loaded.")
            return None

        # 画像をリサイズして詳細をより明確にする
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)

        # 画像をグレースケールに変換して色情報を取り除く
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ガウシアンブラーを適用してノイズを除去
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # CLAHEを適用してコントラストを強化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # シャープ化フィルターを適用
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # 明るさとコントラストの調整
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

        # デバッグ用にグレースケール画像を一時ファイルに保存
        temp_gray_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_gray_file.name, gray)
        print(f"Grayscale image saved to: {temp_gray_file.name}")

        # 適応的な二値化を適用して、画像を白黒に変換
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 9)

        # モルフォロジー変換を適用して文字を強調
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # シャープ化フィルターをもう一度適用
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        binary = cv2.filter2D(binary, -1, kernel_sharpen)

        # デバッグ用に最終処理画像を一時ファイルに保存
        temp_final_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_final_file.name, binary)
        print(f"Final preprocessed image saved to: {temp_final_file.name}")

        # 処理結果の表示
        plt.imshow(binary, cmap='gray')
        plt.title('Final Preprocessed Image')
        plt.show()

        return temp_final_file.name

    except Exception as e:
        print(f"Exception in preprocess_image: {e}")
        return None

def extract_text_from_image(image_path):
    try:
        temp_image_path = preprocess_image(image_path)

        if temp_image_path is None:
            return "Error: Image preprocessing failed."

        pil_img = Image.open(temp_image_path)

        # OCR設定を詳細に指定
        custom_config = r'--oem 1 --psm 6'  # LSTM OCRエンジンを使用し、単一の均一なブロックのテキストとして処理

        # OCRを実行し、各文字の位置情報を取得
        text = pytesseract.image_to_string(pil_img, lang='jpn', config=custom_config)

        # デバッグ用に抽出されたテキストを出力
        print(f"Extracted text: {text}")

        return text

    except Exception as e:
        print(f"Exception in extract_text_from_image: {e}")
        return "Error: OCR processing failed."

# アップロードされた画像ファイルのパス
image_path = '/mnt/data/スクリーンショット 2024-07-18 101358.png'

# 画像からテキストを抽出
extracted_text = extract_text_from_image(image_path)
print(extracted_text)
