# Inicializar historial de recompensas
reward_history = []

# Entrenamiento
for ep in range(episodios):
    estado = entorno.reset()
    terminado = False
    recompensa_total = 0  # ← para acumular la recompensa del episodio

    while not terminado:
        if np.random.rand() < epsilon:
            accion = np.random.choice(entorno.acciones)
        else:
            accion = np.argmax(Q[estado])

        nuevo_estado, recompensa, terminado = entorno.step(accion)

        Q[estado, accion] += alpha * (recompensa + gamma * np.max(Q[nuevo_estado]) - Q[estado, accion])
        estado = nuevo_estado

        recompensa_total += recompensa  # ← suma la recompensa obtenida

    reward_history.append(recompensa_total)  # ← guarda la recompensa del episodio

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(reward_history)
plt.title("Recompensa Total por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.grid(True)
plt.savefig("app/static/recompensas.png")

