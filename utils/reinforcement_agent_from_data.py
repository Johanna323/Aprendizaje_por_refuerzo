import numpy as np
import pandas as pd
import os
import random
import joblib

from data_procesing import cargar_y_preparar_datos
from plot_rl_results import plot_rewards, plot_q_table_heatmap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class ChurnEnvFromData:
    def __init__(self, df):
        self.df = df.copy()
        self.customers = df.drop('Churn', axis=1).values
        self.labels = df['Churn'].values.flatten()  # Asegurar array 1D
        self.n_actions = 3
        self.current_index = None

    def reset(self):
        self.current_index = random.randint(0, len(self.customers) - 1)
        return self.customers[self.current_index]

    def step(self, action):
        churn_value = np.ravel(self.labels[self.current_index])[0]
        churn_value = int(churn_value)

        if action == 0:
            reward = -1 if churn_value == 1 else 1
        elif action == 1:
            reward = 1 if churn_value == 0 else 0
        else:
            reward = 2 if churn_value == 0 else -1

        return self.customers[self.current_index], reward, True

def train_q_learning_from_data(path_csv, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2, 
                               save_subdir="models", img_dir="static/trainedImages"):
    print("📢 Iniciando entrenamiento con Q-Learning...")

    df = pd.read_csv(path_csv)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Codificación
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Guardar objetos de preprocesamiento
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", save_subdir)
    os.makedirs(save_path, exist_ok=True)

    # Guardar nombres de columnas esperadas
    expected_columns = df.drop('Churn', axis=1).columns.tolist()    
    joblib.dump(expected_columns, os.path.join(save_path, "columns.pkl"))

    # Escalado
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Churn', axis=1)), columns=df.drop('Churn', axis=1).columns)

    joblib.dump(label_encoders, os.path.join(save_path, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    joblib.dump(df_scaled, os.path.join(save_path, "df_scaled.pkl"))

    # Entrenamiento del agente
    env = ChurnEnvFromData(df)
    n_states = len(env.customers)
    n_actions = env.n_actions

    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        state_index = random.randint(0, n_states - 1)
        state = env.customers[state_index]
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.argmax(Q[state_index])
            _, reward, done = env.step(action)
            Q[state_index, action] += alpha * (reward + gamma * np.max(Q[state_index]) - Q[state_index, action])
            total_reward += reward

        rewards_per_episode.append(total_reward)

    # Guardar resultados de entrenamiento
    np.save(os.path.join(save_path, "q_table.npy"), Q)
    np.save(os.path.join(save_path, "rewards.npy"), rewards_per_episode)
    print(f"✅ Q-table y recompensas guardadas en: {save_path}")

    # Guardar gráficas
    img_full_dir = os.path.join(script_dir, "..", img_dir)
    os.makedirs(img_full_dir, exist_ok=True)
    plot_rewards(rewards_per_episode, output_path=os.path.join(img_full_dir, "rewards_rl.png"))
    plot_q_table_heatmap(Q, output_path=os.path.join(img_full_dir, "q_heatmap.png"))
    print(f"📊 Gráficas generadas en: {img_full_dir}")

    return Q, rewards_per_episode, df

if __name__ == "__main__":
    train_q_learning_from_data("Telco-Customer-Churn.csv")
