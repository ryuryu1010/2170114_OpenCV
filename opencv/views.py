from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .image_processing import extract_text_from_image

def index(request):
    return render(request, 'index.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        if image.content_type not in ['image/jpeg', 'image/png']:
            return render(request, 'index.html', {'error': 'Unsupported file format. Please upload a JPEG or PNG image.'})

        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_path = fs.path(filename)

        try:
            # デバッグ: 画像パスの確認
            print(f"Uploaded file path: {uploaded_file_path}")

            # 画像からテキストを抽出
            text = extract_text_from_image(uploaded_file_path)

            # デバッグ: 抽出されたテキストの確認
            print(f"Extracted text: {text}")

            # 抽出されたテキストを結果ページにリダイレクト
            return render(request, 'result.html', {'text': text})
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return render(request, 'index.html', {'error': str(e)})
        except Exception as e:
            print(f"Exception: {e}")
            return render(request, 'index.html', {'error': 'An error occurred during processing.'})

    return redirect('index')

def result(request):
    return render(request, 'result.html')
