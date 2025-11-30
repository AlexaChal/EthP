import os
# 0. Прибираємо зайвий шум від TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import threading
import time
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ==========================================
# 1. ГЛОБАЛЬНІ НАЛАШТУВАННЯ
# ==========================================
class AppState:
    def __init__(self):
        self.config = {
            'coin_id': 'ethereum',
            'currency': 'usd',
            'days_data': 365,   
            'sequence_length': 30, 
            'epochs': 50
        }
        self.latest_result = {
            'status': 'starting',
            'timestamp': None,
            'prediction_prob': 0,
            'direction': 'WAITING',
            'accuracy': 0,
            'last_price': 0,
            'chart_data': []       
        }
        self.is_training = False
        self.lock = threading.Lock()

state = AppState()
app = Flask(__name__)
CORS(app)

# ==========================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ (ІНДИКАТОРИ + API)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def fetch_with_retry(url, params, retries=3):
    """Функція для безпечного запиту до API з повторними спробами та заголовками"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json'
    }
    
    for i in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"[УВАГА] Ліміт API (429). Чекаємо 65с...")
                time.sleep(65)
                continue
            elif response.status_code == 401:
                print(f"[УВАГА] Помилка авторизації (401).")
                time.sleep(10)
                continue
            else:
                time.sleep(5)
        except Exception as e:
            print(f"[ЗБІЙ МЕРЕЖІ] {e}")
            time.sleep(5)
            
    return None 

# ==========================================
# 3. РУЧНІ ШАРИ НЕЙРОМЕРЕЖІ
# ==========================================
class ManualLayer(tf.Module):
    pass

class ManualConv1D(ManualLayer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initializer(shape=(kernel_size, in_channels, out_channels)), name="conv_w")
        self.b = tf.Variable(tf.zeros((out_channels,)), name="conv_b")

    def __call__(self, x):
        padding_size = (self.kernel_size - 1) * self.dilation_rate
        x_padded = tf.pad(x, [[0, 0], [padding_size, 0], [0, 0]])
        res = tf.nn.convolution(
            input=x_padded, filters=self.w, strides=1, dilations=[self.dilation_rate],
            padding='VALID', data_format='NWC'
        )
        return tf.nn.relu(res + self.b)

class ManualLSTM(ManualLayer):
    def __init__(self, input_dim, units):
        super().__init__()
        self.units = units
        init = tf.initializers.GlorotUniform()
        self.Wx = tf.Variable(init(shape=(input_dim, 4 * units)), name="lstm_Wx")
        self.Wh = tf.Variable(init(shape=(units, 4 * units)), name="lstm_Wh")
        self.b = tf.Variable(tf.zeros((4 * units,)), name="lstm_b")

    def step(self, x_t, states):
        h_prev, c_prev = states
        gates = tf.matmul(x_t, self.Wx) + tf.matmul(h_prev, self.Wh) + self.b
        i_gate, f_gate, c_cand, o_gate = tf.split(gates, num_or_size_splits=4, axis=1)
        i = tf.sigmoid(i_gate)
        f = tf.sigmoid(f_gate + 1.0)
        c_tilde = tf.tanh(c_cand)
        o = tf.sigmoid(o_gate)
        c_new = f * c_prev + i * c_tilde
        h_new = o * tf.tanh(c_new)
        return h_new, c_new

    def __call__(self, x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))
        for t in range(time_steps):
            x_t = x[:, t, :]
            h, c = self.step(x_t, (h, c))
        return h

class ManualDense(ManualLayer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        init = tf.initializers.GlorotUniform()
        self.w = tf.Variable(init(shape=(input_dim, output_dim)), name="dense_w")
        self.b = tf.Variable(tf.zeros((output_dim,)), name="dense_b")
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

class HybridModel(tf.Module):
    def __init__(self, input_features):
        super().__init__()
        self.tcn1 = ManualConv1D(in_channels=input_features, out_channels=64, kernel_size=3, dilation_rate=1)
        self.tcn2 = ManualConv1D(in_channels=64, out_channels=64, kernel_size=3, dilation_rate=2)
        self.lstm = ManualLSTM(input_dim=64, units=50)
        self.dense = ManualDense(input_dim=50, output_dim=1)

    def __call__(self, x):
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.lstm(x)
        logits = self.dense(x)
        return tf.sigmoid(logits)

# ==========================================
# 4. ЛОГІКА ДАНИХ ТА НАВЧАННЯ
# ==========================================
def fetch_and_train():
    with state.lock:
        cfg = state.config.copy()
        state.is_training = True
        state.latest_result['status'] = 'training'
    
    print(f"\n--- [Сервер] Початок циклу. Історія={cfg['days_data']} дн., Епох={cfg['epochs']} ---")

    try:
        # 1. Завантаження Даних
        days_param = cfg['days_data']
        
        url_market = f"https://api.coingecko.com/api/v3/coins/{cfg['coin_id']}/market_chart"
        p_market = {'vs_currency': cfg['currency'], 'days': days_param}
        
        res_market = fetch_with_retry(url_market, p_market)
        
        if not res_market or 'prices' not in res_market:
            raise Exception("Не вдалося отримати дані. CoinGecko блокує запити.")

        df = pd.DataFrame(res_market['prices'], columns=['ts', 'price'])
        vols = pd.DataFrame(res_market['total_volumes'], columns=['ts', 'volume'])
        df['volume'] = vols['volume']
        df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.normalize()
        df = df.groupby('date').agg({'price': 'last', 'volume': 'last'})

        # Графік для фронтенду
        chart_data_export = []
        export_df = df.tail(365) 
        for date, row in export_df.iterrows():
            chart_data_export.append({
                'date': date.strftime('%d.%m.%Y'),
                'price': float(row['price'])
            })

        # --- SENTIMENT ---
        url_sent = "https://api.alternative.me/fng/"
        p_sent = {'limit': days_param, 'format': 'json'}
        headers_sent = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            res_sent = requests.get(url_sent, params=p_sent, headers=headers_sent, timeout=10).json()
            if 'data' not in res_sent: raise ValueError("No data")
            
            sent_data = []
            for item in res_sent['data']:
                sent_data.append({'date': pd.to_datetime(int(item['timestamp']), unit='s').normalize(), 'val': int(item['value'])})
            df_sent = pd.DataFrame(sent_data).set_index('date')
        except:
             print("[УВАГА] Сентимент недоступний, використовуємо заглушку.")
             sent_data = [{'date': df.index[i], 'val': 50} for i in range(len(df))]
             df_sent = pd.DataFrame(sent_data).set_index('date')
        
        # Об'єднання
        df.sort_index(inplace=True)
        df_sent.sort_index(inplace=True)
        full = df.join(df_sent, how='inner')
        full['val'] = full['val'].ffill().fillna(50)
        
        # --- ЛОГУВАННЯ ДАНИХ (Для наочності) ---
        print("\n[LOG] --- Останні 3 записи ринкових даних (Price/Volume/Sentiment) ---")
        print(full[['price', 'volume', 'val']].tail(3))
        print("----------------------------------------------------------------------")

        # Індикатори
        full['rsi'] = calculate_rsi(full['price'])
        full['macd'] = calculate_macd(full['price'])
        full.dropna(inplace=True)

        full['val'] = full['val'].shift(1)
        full.dropna(inplace=True)
        
        full['target'] = (full['price'].shift(-1) > full['price']).astype(int)
        full = full[:-1]

        if len(full) < 50:
            raise Exception("Замало даних після обробки")

        # 2. Підготовка
        scaler = MinMaxScaler()
        feature_cols = ['price', 'volume', 'val', 'rsi', 'macd']
        data_vals = full[feature_cols].values
        data_scaled = scaler.fit_transform(data_vals)
        targets = full['target'].values
        
        X, y = [], []
        seq_len = cfg['sequence_length']
        for i in range(len(data_scaled) - seq_len):
            X.append(data_scaled[i:i+seq_len])
            y.append(targets[i+seq_len])
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 3. Навчання
        model = HybridModel(input_features=len(feature_cols))
        optimizer = tf.optimizers.Adam(learning_rate=0.005)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(16)
        
        for epoch in range(cfg['epochs']):
            for x_b, y_b in train_ds:
                with tf.GradientTape() as tape:
                    pred = model(x_b)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_b, pred))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        val_pred = model(X_test)
        acc = tf.keras.metrics.BinaryAccuracy()(y_test, val_pred).numpy()

        # 4. Прогноз
        last_seq = data_scaled[-seq_len:]
        last_seq = np.expand_dims(last_seq, axis=0).astype(np.float32)
        final_prob = model(last_seq).numpy()[0][0]
        
        with state.lock:
            state.latest_result = {
                'status': 'ready',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction_prob': float(final_prob),
                'direction': 'UP 📈' if final_prob > 0.5 else 'DOWN 📉',
                'accuracy': float(acc),
                'last_price': float(full['price'].iloc[-1]),
                'chart_data': chart_data_export 
            }
            state.is_training = False
        
        print(f"--- [Сервер] Готово. Прогноз: {state.latest_result['direction']} ({final_prob:.2f}) ---")

    except Exception as e:
        print(f"[ПОМИЛКА] Цикл перервано: {e}")
        with state.lock:
            state.latest_result['status'] = f'error: {str(e)}'
            state.is_training = False

def background_loop():
    while True:
        if not state.is_training:
            t = threading.Thread(target=fetch_and_train)
            t.start()
        # Пауза
        time.sleep(30)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'config': state.config,
        'result': state.latest_result,
        'is_training': state.is_training
    })

@app.route('/config', methods=['POST'])
def update_config():
    data = request.json
    with state.lock:
        if 'days_data' in data:
            state.config['days_data'] = int(data['days_data'])
        if 'epochs' in data:
            state.config['epochs'] = int(data['epochs'])
            
    if not state.is_training:
        threading.Thread(target=fetch_and_train).start()
        
    return jsonify({'status': 'ok', 'new_config': state.config})

if __name__ == '__main__':
    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)