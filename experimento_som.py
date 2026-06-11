"""
=============================================================================
Experimento: Comparação de diferentes tamanhos de grade SOM
Testa grades de 3x3 até 15x15 e compara erro de quantização e acurácia
=============================================================================
"""

import os
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURAÇÕES
# ===========================================================================
DADOS_DIR = "dados"
ESPECIES = ["araponga", "bem_te_vi", "urutau"]
CORES = {"araponga": "#E63946", "bem_te_vi": "#457B9D", "urutau": "#2A9D8F"}
MARCADORES = {"araponga": "o", "bem_te_vi": "s", "urutau": "^"}

N_MFCC = 13
SR = 22050
SOM_EPOCAS = 10000

# Grades para testar
GRADES = [
    (3, 3),    #  9 neurônios
    (5, 5),    # 25 neurônios
    (7, 7),    # 49 neurônios
    (10, 10),  # 100 neurônios
    (15, 15),  # 225 neurônios
]


# ===========================================================================
# IMPLEMENTAÇÃO DO SOM
# ===========================================================================
class SOM:
    def __init__(self, largura, altura, n_features, seed=42):
        self.largura = largura
        self.altura = altura
        self.n_features = n_features
        np.random.seed(seed)
        self.pesos = np.random.rand(largura, altura, n_features)
        self.coordenadas = np.array(
            [[[i, j] for j in range(altura)] for i in range(largura)]
        )

    def _encontrar_bmu(self, x):
        diferencas = self.pesos - x
        distancias = np.sqrt(np.sum(diferencas ** 2, axis=2))
        idx = np.unravel_index(np.argmin(distancias), distancias.shape)
        return idx

    def _funcao_vizinhanca(self, bmu, sigma):
        bmu_coord = np.array([bmu[0], bmu[1]])
        dist_ao_bmu = np.sqrt(np.sum(
            (self.coordenadas - bmu_coord) ** 2, axis=2
        ))
        return np.exp(-(dist_ao_bmu ** 2) / (2 * sigma ** 2))

    def _decaimento(self, vi, vf, epoca, total):
        return vi * (vf / vi) ** (epoca / total)

    def treinar(self, X, epocas, sigma_i=4.0, sigma_f=0.5, lr_i=0.5, lr_f=0.01):
        # Ajusta sigma inicial ao tamanho da grade
        sigma_i = max(self.largura, self.altura) / 2
        self.historico_erro = []
        n = len(X)

        for epoca in range(epocas):
            sigma = self._decaimento(sigma_i, sigma_f, epoca, epocas)
            lr = self._decaimento(lr_i, lr_f, epoca, epocas)

            idx = np.random.randint(0, n)
            x = X[idx]

            bmu = self._encontrar_bmu(x)
            vizinhanca = self._funcao_vizinhanca(bmu, sigma)

            for i in range(self.largura):
                for j in range(self.altura):
                    influencia = lr * vizinhanca[i, j]
                    self.pesos[i, j] += influencia * (x - self.pesos[i, j])

            if (epoca + 1) % 500 == 0:
                erro = self._erro_quantizacao(X)
                self.historico_erro.append(erro)

        return self.historico_erro[-1] if self.historico_erro else 0

    def _erro_quantizacao(self, X):
        erro = 0
        for x in X:
            bmu = self._encontrar_bmu(x)
            erro += np.linalg.norm(x - self.pesos[bmu[0], bmu[1]])
        return erro / len(X)

    def vencedor(self, x):
        return self._encontrar_bmu(x)

    def u_matrix(self):
        umat = np.zeros((self.largura, self.altura))
        for i in range(self.largura):
            for j in range(self.altura):
                vizinhos = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.largura and 0 <= nj < self.altura:
                            dist = np.linalg.norm(self.pesos[i, j] - self.pesos[ni, nj])
                            vizinhos.append(dist)
                umat[i, j] = np.mean(vizinhos)
        return umat

    def rotular_neuronios(self, X, rotulos):
        contagem = {}
        for i in range(self.largura):
            for j in range(self.altura):
                contagem[(i, j)] = {}

        for x, rotulo in zip(X, rotulos):
            bmu = self._encontrar_bmu(x)
            if rotulo not in contagem[bmu]:
                contagem[bmu][rotulo] = 0
            contagem[bmu][rotulo] += 1

        self.mapa_rotulos = {}
        for pos, classes in contagem.items():
            if classes:
                self.mapa_rotulos[pos] = max(classes, key=classes.get)
            else:
                self.mapa_rotulos[pos] = None

    def classificar(self, x):
        bmu = self._encontrar_bmu(x)
        return self.mapa_rotulos.get(bmu, None)


# ===========================================================================
# CARREGAR DADOS
# ===========================================================================
print("=" * 60)
print("Carregando e processando áudios...")
print("=" * 60)


def carregar_audio(caminho, sr=SR):
    y, _ = librosa.load(caminho, sr=sr)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y


def extrair_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    media = np.mean(mfccs, axis=1)
    desvio = np.std(mfccs, axis=1)
    return np.concatenate([media, desvio])


features = []
rotulos = []

for especie in ESPECIES:
    pasta = os.path.join(DADOS_DIR, especie)
    arquivos = [f for f in os.listdir(pasta) if f.endswith(".mp3")]
    print(f"  {especie}: {len(arquivos)} arquivos")
    for arquivo in sorted(arquivos):
        caminho = os.path.join(pasta, arquivo)
        try:
            y = carregar_audio(caminho)
            mfcc = extrair_mfcc(y)
            features.append(mfcc)
            rotulos.append(especie)
        except Exception as e:
            print(f"    ERRO: {arquivo}: {e}")

X = np.array(features)
y_rotulos = np.array(rotulos)

# Normalização
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_range = X_max - X_min
X_range[X_range == 0] = 1
X_norm = (X - X_min) / X_range

# Split treino/teste
np.random.seed(42)
idx_treino, idx_teste = [], []
for especie in ESPECIES:
    indices = np.where(y_rotulos == especie)[0].tolist()
    np.random.shuffle(indices)
    corte = max(1, int(len(indices) * 0.7))
    idx_treino.extend(indices[:corte])
    idx_teste.extend(indices[corte:])

X_treino, y_treino = X_norm[idx_treino], y_rotulos[idx_treino]
X_teste, y_teste = X_norm[idx_teste], y_rotulos[idx_teste]

print(f"\n  Total: {len(X)} amostras | Treino: {len(X_treino)} | Teste: {len(X_teste)}")


# ===========================================================================
# EXPERIMENTO: TESTAR DIFERENTES GRADES
# ===========================================================================
print("\n" + "=" * 60)
print("EXPERIMENTO: Comparando tamanhos de grade")
print("=" * 60)

resultados = []

for (gx, gy) in GRADES:
    n_neuronios = gx * gy
    print(f"\n  --- Grade {gx}x{gy} ({n_neuronios} neurônios) ---")

    som = SOM(largura=gx, altura=gy, n_features=X_norm.shape[1], seed=42)
    erro_final = som.treinar(X_norm, epocas=SOM_EPOCAS)
    print(f"    Erro de quantização: {erro_final:.4f}")

    # Classificação
    som.rotular_neuronios(X_treino, y_treino)

    y_pred = []
    for x in X_teste:
        pred = som.classificar(x)
        y_pred.append(pred if pred is not None else "desconhecido")
    y_pred = np.array(y_pred)

    acuracia = np.sum(y_pred == y_teste) / len(y_teste)
    print(f"    Acurácia: {acuracia:.2%}")

    # Métricas por espécie
    metricas_especie = {}
    for especie in ESPECIES:
        mask_pred = y_pred == especie
        mask_real = y_teste == especie
        vp = np.sum(mask_pred & mask_real)
        total_pred = np.sum(mask_pred)
        total_real = np.sum(mask_real)
        precisao = vp / total_pred if total_pred > 0 else 0
        recall = vp / total_real if total_real > 0 else 0
        metricas_especie[especie] = {"precisao": precisao, "recall": recall}

    resultados.append({
        "grade": f"{gx}x{gy}",
        "neuronios": n_neuronios,
        "erro": erro_final,
        "acuracia": acuracia,
        "som": som,
        "historico": som.historico_erro,
        "metricas": metricas_especie,
    })


# ===========================================================================
# VISUALIZAÇÕES COMPARATIVAS
# ===========================================================================
print("\n" + "=" * 60)
print("Gerando gráficos comparativos...")
print("=" * 60)

# --- Gráfico 1: Erro e Acurácia por tamanho de grade ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

grades_label = [r["grade"] for r in resultados]
erros = [r["erro"] for r in resultados]
acuracias = [r["acuracia"] for r in resultados]

# Erro de quantização
axes[0].bar(grades_label, erros, color="#457B9D", alpha=0.8, edgecolor="white")
axes[0].set_title("Erro de Quantização por Grade", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Tamanho da Grade")
axes[0].set_ylabel("Erro Médio")
axes[0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(erros):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

# Acurácia
barras = axes[1].bar(grades_label, acuracias, color="#2A9D8F", alpha=0.8, edgecolor="white")
axes[1].set_title("Acurácia de Classificação por Grade", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Tamanho da Grade")
axes[1].set_ylabel("Acurácia")
axes[1].set_ylim(0, 1.1)
axes[1].axhline(y=1/3, color="red", linestyle="--", alpha=0.5, label="Acaso (33%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(acuracias):
    axes[1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Comparação de Tamanhos de Grade SOM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("exp_01_erro_acuracia.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: exp_01_erro_acuracia.png")

# --- Gráfico 2: Curvas de convergência sobrepostas ---
fig, ax = plt.subplots(figsize=(10, 5))
cores_curva = ["#E63946", "#F4A261", "#457B9D", "#2A9D8F", "#264653"]

for i, r in enumerate(resultados):
    epocas_plot = [(j + 1) * 500 for j in range(len(r["historico"]))]
    ax.plot(epocas_plot, r["historico"], "o-", color=cores_curva[i],
            linewidth=1.5, markersize=3, label=f"{r['grade']} ({r['neuronios']} neur.)")

ax.set_title("Convergência do SOM por Tamanho de Grade", fontsize=12, fontweight="bold")
ax.set_xlabel("Época")
ax.set_ylabel("Erro de Quantização")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("exp_02_convergencia.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: exp_02_convergencia.png")

# --- Gráfico 3: Mapas SOM lado a lado ---
fig, axes = plt.subplots(1, len(GRADES), figsize=(4 * len(GRADES), 4))

for idx, r in enumerate(resultados):
    som = r["som"]
    for i, x in enumerate(X_norm):
        w = som.vencedor(x)
        especie = y_rotulos[i]
        axes[idx].plot(
            w[0] + np.random.uniform(-0.2, 0.2),
            w[1] + np.random.uniform(-0.2, 0.2),
            MARCADORES[especie],
            color=CORES[especie],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    axes[idx].set_xlim(-0.5, som.largura - 0.5)
    axes[idx].set_ylim(-0.5, som.altura - 0.5)
    axes[idx].set_title(f"{r['grade']}\nAcc: {r['acuracia']:.0%}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("X")
    axes[idx].set_ylabel("Y" if idx == 0 else "")
    axes[idx].grid(True, alpha=0.2)

axes[-1].legend(
    handles=[Patch(color=CORES[e], label=e.replace("_", " ").title()) for e in ESPECIES],
    loc="upper right", fontsize=8,
)

plt.suptitle("Mapeamento das Amostras em Diferentes Grades SOM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("exp_03_mapas_comparados.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: exp_03_mapas_comparados.png")

# --- Gráfico 4: U-Matrix de cada grade ---
fig, axes = plt.subplots(1, len(GRADES), figsize=(4 * len(GRADES), 4))

for idx, r in enumerate(resultados):
    som = r["som"]
    umat = som.u_matrix()
    im = axes[idx].imshow(umat.T, cmap="bone_r", origin="lower")
    axes[idx].set_title(f"U-Matrix {r['grade']}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("X")
    axes[idx].set_ylabel("Y" if idx == 0 else "")

plt.suptitle("U-Matrix por Tamanho de Grade", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("exp_04_umatrix_comparada.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: exp_04_umatrix_comparada.png")


# ===========================================================================
# TABELA RESUMO
# ===========================================================================
print("\n" + "=" * 60)
print("TABELA COMPARATIVA")
print("=" * 60)
print(f"\n  {'Grade':<8} {'Neurônios':>10} {'Erro':>10} {'Acurácia':>10}")
print(f"  {'-' * 40}")
for r in resultados:
    print(f"  {r['grade']:<8} {r['neuronios']:>10} {r['erro']:>10.4f} {r['acuracia']:>10.2%}")

# Melhor configuração
melhor = max(resultados, key=lambda r: r["acuracia"])
print(f"\n  Melhor grade: {melhor['grade']} com {melhor['acuracia']:.2%} de acurácia")

# Detalhes da melhor
print(f"\n  Detalhes da melhor grade ({melhor['grade']}):")
print(f"  {'Espécie':<15} {'Precisão':>10} {'Recall':>10}")
print(f"  {'-' * 35}")
for especie in ESPECIES:
    m = melhor["metricas"][especie]
    nome = especie.replace("_", " ").title()
    print(f"  {nome:<15} {m['precisao']:>10.2%} {m['recall']:>10.2%}")

print(f"""
  Arquivos gerados:
    exp_01_erro_acuracia.png       — Erro e acurácia por grade
    exp_02_convergencia.png        — Curvas de convergência sobrepostas
    exp_03_mapas_comparados.png    — Mapas SOM lado a lado
    exp_04_umatrix_comparada.png   — U-Matrix de cada grade
""")
print("Experimento concluído!")