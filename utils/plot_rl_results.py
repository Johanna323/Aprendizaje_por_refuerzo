import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(rewards, output_path="static/trainedImages/rewards_rl.png"):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensas")
    plt.title("Evolución de recompensa del agente")
    plt.savefig(output_path)
    plt.close()

def plot_q_table_heatmap(Q, output_path="static/trainedImages/q_heatmap.png"):
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.heatmap(Q, cmap="YlGnBu")
    plt.title("Q-table Heatmap")
    plt.xlabel("Acciones")
    plt.ylabel("Clientes")
    plt.savefig(output_path)
    plt.close()
