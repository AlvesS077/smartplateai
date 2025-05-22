import os
import random
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import normalize
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Configuração do Flask ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
BASE_DIR = os.path.join(os.path.dirname(__file__), 'train')  # caminho relativo
EXCEL_PATH = os.path.join(os.path.dirname(__file__), 'Healthy_Food_Plates_100.xlsx')
EXCEL_COLUMNS = ['Plate Name', 'Protein (g)', 'Calories', 'Fat (g)', 'Carbs (g)', 'Fiber (g)']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Carregamento do modelo ResNet ---
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove a última camada
model.eval()

# --- Função para gerar embedding da imagem ---
def get_embedding(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return normalize(embedding, p=2, dim=1).squeeze()

# --- Rota principal para servir o index.html ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Rota da API para análise alimentar ---
@app.route('/analyze-food', methods=['POST'])
def analyze_food():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    try:
        user_embedding = get_embedding(image_path)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

    best_class = None
    best_distance = float('inf')
    for root, _, files in os.walk(BASE_DIR):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            base_img_path = os.path.join(root, fname)
            try:
                base_embedding = get_embedding(base_img_path)
                distance = 1 - torch.cosine_similarity(user_embedding, base_embedding, dim=0).item()
                class_name = os.path.basename(os.path.dirname(base_img_path))
                if distance < best_distance:
                    best_distance = distance
                    best_class = class_name
            except Exception:
                continue

    if best_class is None:
        return jsonify({'error': 'No images found in base'}), 404

    # Lê uma linha aleatória do Excel
    try:
        df = pd.read_excel(EXCEL_PATH)
        random_row = random.randint(0, min(99, len(df)-1))
        excel_data = df.loc[random_row, EXCEL_COLUMNS].to_dict()
    except Exception as e:
        excel_data = {'error': f'Excel read error: {str(e)}'}

    return jsonify({'class': best_class, 'excel': excel_data})

# --- Iniciar servidor e abrir browser automaticamente ---
if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)