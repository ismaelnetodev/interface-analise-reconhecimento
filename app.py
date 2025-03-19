from flask import Flask, request, jsonify, render_template
import cv2, numpy as np, base64, os, concurrent.futures, json
from datetime import datetime
from deepface import DeepFace
from src.database import init_db, select_alunos
from src.recognition import process_frame, MODEL_NAME, TARGET_SIZE

app = Flask(__name__)
sentiment_counts = {'happy': 0, 'neutral': 0, 'fear': 0, 'sad': 0, 'surprise': 0}

def load_registered_embeddings():
    alunos = select_alunos()
    registered_embeddings = []
    for aluno in alunos:
        try:
            local_path = os.path.join('imagens', os.path.basename(aluno[2]))
            img = cv2.imread(local_path)
            if img is None:
                continue
            img = cv2.resize(img, TARGET_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
            registered_embeddings.append({'id': aluno[0], 'name': aluno[1], 'embedding': np.array(embedding)})
        except Exception as e:
            print(f"Erro ao processar {aluno[1]}: {str(e)}")
    return registered_embeddings

registered_embeddings = load_registered_embeddings()

def do_recognition(frame):
    return process_frame(frame, registered_embeddings)

def analyze_sentiment(face_img):
    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            dominant = analysis[0]['dominant_emotion']
            print("Análise de sentimento:", dominant)
            return dominant
        else:
            return "none"
    except Exception as e:
        print("Error in analyze_sentiment:", e)
        return "error"

def save_sentiment_log(user_id, user_name, sentiment):
    global sentiment_counts
    if sentiment in sentiment_counts:
        sentiment_counts[sentiment] += 1
    else:
        sentiment_counts[sentiment] = 1
    print(f"Salvando log: ID: {user_id}, Nome: {user_name}, Sentimento: {sentiment}")
    record = {"user_id": user_id, "user_name": user_name, "sentiment": sentiment, "timestamp": datetime.now().isoformat()}
    try:
        with open("sentiment_logs.json", "r") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(record)
    with open("sentiment_logs.json", "w") as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    data = request.json
    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_recog = executor.submit(do_recognition, frame)
        recog_results = future_recog.result()
    
    faces_count = len(recog_results)
    
    for res in recog_results:
        if "box" in res:
            x, y, w, h = res["box"]
            face_img = frame[y:y+h, x:x+w]
            sentiment = analyze_sentiment(face_img)
            res["sentiment"] = sentiment  
            if res["recognized"] != "Desconhecido":
                save_sentiment_log(res.get("id", None), res["recognized"], sentiment)
    
    response = {"faces": faces_count, "results": recog_results}
    print("Process frame retornado:", response)
    return jsonify(response)

@app.route('/get_sentiment_stats', methods=['GET'])
def get_sentiment_stats():
    print("Retornando contagem de sentimentos:", sentiment_counts)
    return jsonify(sentiment_counts)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
