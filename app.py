from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/reinforcement')
def reinforcement():
    q_table = np.load("models/q_table.npy")
    rewards = np.load("models/rewards.npy")

    return render_template("reinforcement.html", 
                           q_shape=q_table.shape, 
                           total_episodes=len(rewards),
                           total_reward=int(np.sum(rewards)),
                           rewards_plot="trainedImages/rewards_rl.png",
                           q_heatmap="trainedImages/q_heatmap.png")

@app.route("/ejecutar-modelo", methods=["GET", "POST"])
def formulario_refuerzo():
    import pandas as pd
    import numpy as np
    import joblib

    # Cargar objetos entrenados
    label_encoders = joblib.load("models/label_encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df_scaled = joblib.load("models/df_scaled.pkl")
    expected_columns = joblib.load("models/columns.pkl")
    q_table = np.load("models/q_table.npy")

    # Mapeo de acciones
    accion_map = {
        0: ("No hacer nada", "El agente considera que este cliente probablemente no abandonará, por lo que no es necesario intervenir."),
        1: ("Enviar encuesta", "El agente sugiere obtener retroalimentación para entender y anticiparse a posibles molestias."),
        2: ("Ofrecer descuento", "El cliente tiene un alto riesgo de cancelar, por lo tanto se recomienda aplicar una acción de retención directa.")
    }

    if request.method == "POST":
        # Recoger datos del formulario
        form_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': float(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form.get('MultipleLines', 'No'),
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form.get('OnlineSecurity', 'No'),
            'OnlineBackup': request.form.get('OnlineBackup', 'No'),
            'DeviceProtection': request.form.get('DeviceProtection', 'No'),
            'TechSupport': request.form.get('TechSupport', 'No'),
            'StreamingTV': request.form.get('StreamingTV', 'No'),
            'StreamingMovies': request.form.get('StreamingMovies', 'No'),
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form.get('PaperlessBilling', 'No'),
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
        }

        entry = pd.DataFrame([form_data])

        # Codificar variables categóricas
        for col in entry.columns:
            if col in label_encoders:
                entry[col] = label_encoders[col].transform(entry[col])

        # Añadir columnas faltantes
        for col in expected_columns:
            if col not in entry.columns:
                entry[col] = 0  # valor neutro o razonable

        # Reordenar columnas como en el entrenamiento
        entry = entry[expected_columns]

        # Escalar
        entry_scaled = pd.DataFrame(scaler.transform(entry), columns=entry.columns)

        # Buscar el cliente más similar
        diffs = np.linalg.norm(df_scaled.values - entry_scaled.values, axis=1)
        closest_index = np.argmin(diffs)

        mejor_accion = np.argmax(q_table[closest_index])
        accion_nombre, accion_explicacion = accion_map[mejor_accion]

        return render_template("formulario_refuerzo.html",
                               form_data=form_data,
                               resultado=accion_nombre,
                               explicacion=accion_explicacion)

    return render_template("formulario_refuerzo.html",
                           form_data=None,
                           resultado=None,
                           explicacion=None)

                           
if __name__ == '__main__':
    app.run(debug=True)