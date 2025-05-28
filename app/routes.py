from flask import Blueprint, render_template
import numpy as np

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template("index.html")

@main.route('/resultados')
def resultados():
    Q = np.load("modelo/modelo.npy")
    recompensas = np.load("modelo/recompensas.npy")

    acciones = ["No hacer nada", "Enviar promoción", "Llamar"]
    politica = [acciones[np.argmax(Q[s])] for s in range(Q.shape[0])]

    return render_template("resultados.html", q_table=Q, politica=politica, recompensas=recompensas.tolist())
