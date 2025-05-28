import numpy as np
import pandas as pd

class EntornoRetencion:
    def __init__(self, datos):
        self.datos = datos.reset_index(drop=True)
        self.num_estados = len(self.datos)
        self.estado_actual = 0
        self.acciones = [0, 1]  # 0: No retener, 1: Ofrecer retención

    def reset(self):
        self.estado_actual = np.random.randint(0, self.num_estados)
        return self.estado_actual

    def step(self, accion):
        fila = self.datos.iloc[self.estado_actual]
        se_va = fila['Churn']
        recompensa = 0

        if accion == 0:
            recompensa = -1 if se_va == 1 else 1
        elif accion == 1:
            costo = 0.2
            recompensa = (1 - costo) if se_va == 1 else -costo

        hecho = True
        nuevo_estado = self.reset()
        return nuevo_estado, recompensa, hecho
