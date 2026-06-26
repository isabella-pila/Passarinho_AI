"""
=============================================================================
EXPERIMENTO COMPARATIVO: 26 vs 58 features
=============================================================================

Compara o desempenho do SOM com duas representações de features:
  - 26 features: MFCC média + desvio (baseline)
  - 58 features: MFCC + Delta MFCC + centróide + ZCR + rolloff

Roda cada configuração 5 vezes com sementes diferentes para obter
média ± desvio padrão (validação estatística).

Saídas:
  - exp_features_resultados.txt    Tabela com todos os resultados
  - exp_features_boxplot.png       Boxplot comparativo
  - exp_features_barras.png        Gráfico de barras com erro

Estrutura esperada:
    dados/
    ├── araponga/
    ├── bem_te_vi/
    └── urutau/

Dependências:
    pip install librosa numpy matplotlib
"""

import os
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURAÇÕES
# ===========================================================================
DADOS_DIR = "dados"
ESPECIES = ["araponga", "bem_te_vi", "urutau"]
CORES_FEATURES = {26: "#457B9D", 58: "#E63946"}

# Parâmetros áudio
N_MFCC = 13
SR = 22050

# Parâmetros SOM
SOM_X, SOM_Y = 5, 5
SOM_EPOCAS = 10000
SIGMA_INICIAL = 2.5
SIGMA_FINAL = 0.5
LR_INICIAL = 0.5
LR_FINAL = 0.01

# Experimento
SEMENTES = [42, 123, 456, 789, 1000]
SPLIT_TREINO = 0.7


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
        self.mapa_rotulos = {}

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

    def treinar(self, X, epocas):
        n = len(X)
        for epoca in range(epocas):
            sigma = self._decaimento(SIGMA_INICIAL, SIGMA_FINAL, epoca, epocas)
            lr = self._decaimento(LR_INICIAL, LR_FINAL, epoca, epocas)
            idx = np.random.randint(0, n)
            x = X[idx]
            bmu = self._encontrar_bmu(x)
            vizinhanca = self._funcao_vizinhanca(bmu, sigma)
            for i in range(self.largura):
                for j in range(self.altura):
                    self.pesos[i, j] += lr * vizinhanca[i, j] * (x - self.pesos[i, j])

    def rotular_neuronios(self, X, rotulos):
        contagem = {}
        for i in range(self.largura):
            for j in range(self.altura):
                contagem[(i, j)] = {}
        for x, rotulo in zip(X, rotulos):
            bmu = self._encontrar_bmu(x)
            contagem[bmu][rotulo] = contagem[bmu].get(rotulo, 0) + 1
        self.mapa_rotulos = {}
        for pos, classes in contagem.items():
            self.mapa_rotulos[pos] = max(classes, key=classes.get) if classes else None

    def classificar(self, x):
        bmu = self._encontrar_bmu(x)
        return self.mapa_rotulos.get(bmu, None)


# ===========================================================================
# EXTRAÇÃO DE FEATURES — DUAS VERSÕES
# ===========================================================================
def carregar_audio(caminho, sr=SR):
    y, _ = librosa.load(caminho, sr=sr)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y


def extrair_26_features(y, sr=SR, n_mfcc=N_MFCC):
    """Baseline: MFCC média + desvio padrão."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    media = np.mean(mfccs, axis=1)
    desvio = np.std(mfccs, axis=1)
    return np.concatenate([media, desvio])


def extrair_58_features(y, sr=SR, n_mfcc=N_MFCC):
    """Estendido: MFCC + Delta + Centróide + ZCR + Rolloff."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_media = np.mean(mfccs, axis=1)
    mfcc_desvio = np.std(mfccs, axis=1)
    delta = librosa.feature.delta(mfccs)
    delta_media = np.mean(delta, axis=1)
    delta_desvio = np.std(delta, axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = [np.mean(centroid), np.std(centroid)]
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feat = [np.mean(zcr), np.std(zcr)]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = [np.mean(rolloff), np.std(rolloff)]
    return np.concatenate([
        mfcc_media, mfcc_desvio,
        delta_media, delta_desvio,
        centroid_feat, zcr_feat, rolloff_feat
    ])


# ===========================================================================
# CARREGAMENTO DO DATASET (uma vez só, extraindo as duas versões)
# ===========================================================================
print("=" * 70)
print("CARREGAMENTO DO DATASET")
print("=" * 70)

features_26 = []
features_58 = []
rotulos = []

for especie in ESPECIES:
    pasta = os.path.join(DADOS_DIR, especie)
    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".mp3")])
    print(f"\n  {especie}: {len(arquivos)} arquivos")

    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)
        try:
            y = carregar_audio(caminho)
            features_26.append(extrair_26_features(y))
            features_58.append(extrair_58_features(y))
            rotulos.append(especie)
        except Exception as e:
            print(f"    ✗ {arquivo}: {e}")

X_26 = np.array(features_26)
X_58 = np.array(features_58)
y_rotulos = np.array(rotulos)

print(f"\n  Total de amostras: {len(X_26)}")
print(f"  Shape X_26: {X_26.shape}")
print(f"  Shape X_58: {X_58.shape}")


# ===========================================================================
# NORMALIZAÇÃO MIN-MAX
# ===========================================================================
def normalizar(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    return (X - X_min) / X_range


X_26_norm = normalizar(X_26)
X_58_norm = normalizar(X_58)


# ===========================================================================
# FUNÇÃO DE EXPERIMENTO
# ===========================================================================
def rodar_experimento(X_norm, y_rotulos, n_features, seed):
    """
    Executa um experimento completo: split, treino, avaliação.
    Retorna a acurácia no conjunto de teste.
    """
    np.random.seed(seed)

    # Split estratificado 70/30
    idx_treino, idx_teste = [], []
    for especie in ESPECIES:
        indices = np.where(y_rotulos == especie)[0].tolist()
        np.random.shuffle(indices)
        corte = max(1, int(len(indices) * SPLIT_TREINO))
        idx_treino.extend(indices[:corte])
        idx_teste.extend(indices[corte:])

    X_treino = X_norm[idx_treino]
    y_treino = y_rotulos[idx_treino]
    X_teste = X_norm[idx_teste]
    y_teste = y_rotulos[idx_teste]

    # Treinamento
    som = SOM(SOM_X, SOM_Y, n_features, seed=seed)
    som.treinar(X_treino, SOM_EPOCAS)
    som.rotular_neuronios(X_treino, y_treino)

    # Avaliação
    acertos = 0
    for i, x in enumerate(X_teste):
        pred = som.classificar(x)
        if pred == y_teste[i]:
            acertos += 1

    return acertos / len(y_teste)


# ===========================================================================
# EXECUÇÃO DO EXPERIMENTO COMPARATIVO
# ===========================================================================
print("\n" + "=" * 70)
print(f"EXPERIMENTO: {len(SEMENTES)} execuções por configuração")
print("=" * 70)

resultados = {26: [], 58: []}

for seed in SEMENTES:
    print(f"\n  Semente {seed}:")

    print(f"    [26 features] Treinando...", end=" ", flush=True)
    acc_26 = rodar_experimento(X_26_norm, y_rotulos, 26, seed)
    resultados[26].append(acc_26)
    print(f"acurácia = {acc_26:.2%}")

    print(f"    [58 features] Treinando...", end=" ", flush=True)
    acc_58 = rodar_experimento(X_58_norm, y_rotulos, 58, seed)
    resultados[58].append(acc_58)
    print(f"acurácia = {acc_58:.2%}")


# ===========================================================================
# ANÁLISE ESTATÍSTICA
# ===========================================================================
print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)

stats = {}
for n_feat in [26, 58]:
    accs = np.array(resultados[n_feat])
    stats[n_feat] = {
        "media": accs.mean() * 100,
        "desvio": accs.std() * 100,
        "min": accs.min() * 100,
        "max": accs.max() * 100,
        "mediana": np.median(accs) * 100,
        "todas": accs * 100,
    }

print(f"\n  {'Configuração':<20} {'Média':>10} {'Desvio':>10} {'Min':>8} {'Max':>8} {'Mediana':>10}")
print(f"  {'-' * 70}")
for n_feat in [26, 58]:
    s = stats[n_feat]
    print(f"  {f'{n_feat} features':<20} "
          f"{s['media']:>9.2f}% {s['desvio']:>9.2f}% "
          f"{s['min']:>7.2f}% {s['max']:>7.2f}% {s['mediana']:>9.2f}%")

# Diferença
diff = stats[58]["media"] - stats[26]["media"]
print(f"\n  Diferença (58 - 26): {diff:+.2f} pontos percentuais")

if abs(diff) < 2:
    interpretacao = "empate técnico — features extras não compensam"
elif diff > 0:
    interpretacao = f"58 features é MELHOR (+{diff:.1f}pp)"
else:
    interpretacao = f"26 features é MELHOR ({diff:.1f}pp) — curse of dimensionality"
print(f"  Interpretação: {interpretacao}")


# ===========================================================================
# SALVAR TABELA TXT
# ===========================================================================
with open("exp_features_resultados.txt", "w", encoding="utf-8") as f:
    f.write("EXPERIMENTO COMPARATIVO: 26 vs 58 FEATURES\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Configuração da rede SOM:\n")
    f.write(f"  Grade: {SOM_X}x{SOM_Y}\n")
    f.write(f"  Épocas: {SOM_EPOCAS}\n")
    f.write(f"  Split treino/teste: {int(SPLIT_TREINO*100)}/{int((1-SPLIT_TREINO)*100)}\n")
    f.write(f"  Sementes testadas: {SEMENTES}\n\n")

    f.write("Resultados por semente:\n")
    f.write(f"  {'Semente':<10} {'26 feat':>12} {'58 feat':>12}\n")
    f.write(f"  {'-' * 36}\n")
    for i, seed in enumerate(SEMENTES):
        f.write(f"  {seed:<10} {resultados[26][i]*100:>11.2f}% {resultados[58][i]*100:>11.2f}%\n")

    f.write("\nEstatísticas:\n")
    f.write(f"  {'Métrica':<15} {'26 feat':>12} {'58 feat':>12}\n")
    f.write(f"  {'-' * 41}\n")
    for metric in ["media", "desvio", "min", "max", "mediana"]:
        f.write(f"  {metric:<15} {stats[26][metric]:>11.2f}% {stats[58][metric]:>11.2f}%\n")

    f.write(f"\nDiferença média (58 - 26): {diff:+.2f}pp\n")
    f.write(f"Interpretação: {interpretacao}\n")

print(f"\n  Salvo: exp_features_resultados.txt")


# ===========================================================================
# GRÁFICO 1: BOXPLOT COMPARATIVO
# ===========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

dados_box = [stats[26]["todas"], stats[58]["todas"]]
labels_box = ["26 features\n(MFCC)", "58 features\n(MFCC + Delta\n+ Espectrais)"]

bp = ax.boxplot(dados_box, labels=labels_box, patch_artist=True,
                widths=0.5, medianprops={"color": "black", "linewidth": 2})

bp["boxes"][0].set_facecolor(CORES_FEATURES[26])
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(CORES_FEATURES[58])
bp["boxes"][1].set_alpha(0.7)

# Pontos individuais
for i, dados in enumerate(dados_box):
    x_pos = np.random.normal(i + 1, 0.04, size=len(dados))
    ax.scatter(x_pos, dados, color="black", alpha=0.6, s=40, zorder=3)

ax.set_ylabel("Acurácia (%)", fontsize=12, fontweight="bold")
ax.set_title("Comparação: 26 vs 58 features\n(5 execuções com sementes diferentes)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim([
    min(stats[26]["min"], stats[58]["min"]) - 5,
    max(stats[26]["max"], stats[58]["max"]) + 5,
])

# Anotações de média
for i, n_feat in enumerate([26, 58]):
    ax.annotate(
        f"μ = {stats[n_feat]['media']:.1f}%\nσ = {stats[n_feat]['desvio']:.1f}%",
        xy=(i + 1, stats[n_feat]["media"]),
        xytext=(i + 1.25, stats[n_feat]["media"]),
        fontsize=10, va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

plt.tight_layout()
plt.savefig("exp_features_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: exp_features_boxplot.png")


# ===========================================================================
# GRÁFICO 2: BARRAS COM ERRO
# ===========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

medias = [stats[26]["media"], stats[58]["media"]]
desvios = [stats[26]["desvio"], stats[58]["desvio"]]
labels = ["26 features\n(MFCC)", "58 features\n(MFCC estendido)"]
cores = [CORES_FEATURES[26], CORES_FEATURES[58]]

bars = ax.bar(labels, medias, yerr=desvios, capsize=10,
              color=cores, alpha=0.8, edgecolor="black", linewidth=1.2,
              error_kw={"linewidth": 2, "ecolor": "black"})

# Valores em cima das barras
for bar, media, desvio in zip(bars, medias, desvios):
    altura = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., altura + desvio + 1,
            f"{media:.1f}% ± {desvio:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Linha do acaso
ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.6, linewidth=1.5)
ax.text(0.5, 35, "Acaso (33.3%)", ha="center", fontsize=10, color="gray", style="italic")

ax.set_ylabel("Acurácia média (%)", fontsize=12, fontweight="bold")
ax.set_title("Acurácia do SOM por representação de features\n(média ± desvio padrão, 5 execuções)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("exp_features_barras.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: exp_features_barras.png")


# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print("\n" + "=" * 70)
print("CONCLUÍDO!")
print("=" * 70)
print(f"""
  Arquivos gerados:
    exp_features_resultados.txt   — Tabela completa
    exp_features_boxplot.png      — Boxplot comparativo
    exp_features_barras.png       — Gráfico de barras com erro

  RESUMO PARA O ARTIGO:
    26 features: {stats[26]['media']:.1f}% ± {stats[26]['desvio']:.1f}%
    58 features: {stats[58]['media']:.1f}% ± {stats[58]['desvio']:.1f}%
    Diferença: {diff:+.2f}pp
    Conclusão: {interpretacao}
""")