"""
=============================================================================
EXPERIMENTO COMPLETO: SOM vs SOM+LVQ × 26 vs 58 features
=============================================================================

Compara 4 configurações:
  1) SOM puro + 26 features
  2) SOM puro + 58 features
  3) SOM + LVQ2.1 + 26 features
  4) SOM + LVQ2.1 + 58 features

Cada configuração é executada 5 vezes com sementes diferentes.

Saídas:
  - exp_completo_resultados.txt       Tabela com todos os resultados
  - exp_completo_tabela2x2.png        Tabela 2x2 visual
  - exp_completo_boxplot.png           Boxplot das 4 configurações
  - exp_completo_barras.png            Gráfico de barras com erro

Estrutura esperada:
    dados/
    ├── araponga/
    ├── bem_te_vi/
    └── urutau/
"""

import os
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURAÇÕES
# ===========================================================================
DADOS_DIR = "dados"
ESPECIES = ["araponga", "bem_te_vi", "urutau"]

N_MFCC = 13
SR = 22050

# SOM
SOM_X, SOM_Y = 5, 5
SOM_EPOCAS = 10000
SIGMA_INICIAL = 2.5
SIGMA_FINAL = 0.5
LR_INICIAL = 0.5
LR_FINAL = 0.01

# LVQ
LVQ_EPOCAS = 5000
LVQ_LR_INICIAL = 0.3
LVQ_LR_FINAL = 0.005
LVQ_JANELA = 0.3       # window width para LVQ2.1

# Experimento
SEMENTES = [42, 123, 456, 789, 1000]
SPLIT_TREINO = 0.7


# ===========================================================================
# SOM
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

    def obter_prototipos(self):
        """
        Retorna lista de (vetor_pesos, rotulo) para cada neurônio rotulado.
        Usado como entrada para o LVQ.
        """
        prototipos = []
        rotulos = []
        posicoes = []
        for (i, j), rotulo in self.mapa_rotulos.items():
            if rotulo is not None:
                prototipos.append(self.pesos[i, j].copy())
                rotulos.append(rotulo)
                posicoes.append((i, j))
        return np.array(prototipos), np.array(rotulos), posicoes


# ===========================================================================
# LVQ 2.1 — REFINAMENTO SUPERVISIONADO
# ===========================================================================
class LVQ21:
    """
    Learning Vector Quantization 2.1

    Refina os protótipos do SOM usando informação supervisionada.
    Para cada amostra:
      - Encontra os 2 protótipos mais próximos
      - Se um tem o rótulo correto e o outro não, E a amostra
        está na "janela" entre eles:
        → puxa o correto para perto
        → empurra o errado para longe

    A "janela" garante que só amostras ambíguas (perto da fronteira)
    sejam usadas para ajustar os protótipos.
    """

    def __init__(self, prototipos, rotulos, posicoes):
        self.prototipos = prototipos.copy()
        self.rotulos = rotulos.copy()
        self.posicoes = posicoes

    def _decaimento(self, vi, vf, epoca, total):
        return vi * (vf / vi) ** (epoca / total)

    def treinar(self, X, y, epocas=LVQ_EPOCAS, lr_i=LVQ_LR_INICIAL,
                lr_f=LVQ_LR_FINAL, janela=LVQ_JANELA):
        """
        Treina o LVQ2.1 refinando os protótipos do SOM.
        """
        n = len(X)

        for epoca in range(epocas):
            lr = self._decaimento(lr_i, lr_f, epoca, epocas)

            # Seleciona amostra aleatória
            idx = np.random.randint(0, n)
            x = X[idx]
            rotulo_real = y[idx]

            # Encontra os 2 protótipos mais próximos
            distancias = np.sqrt(np.sum((self.prototipos - x) ** 2, axis=1))
            idx_ordenados = np.argsort(distancias)

            idx1 = idx_ordenados[0]  # mais próximo
            idx2 = idx_ordenados[1]  # segundo mais próximo

            d1 = distancias[idx1]
            d2 = distancias[idx2]

            rotulo1 = self.rotulos[idx1]
            rotulo2 = self.rotulos[idx2]

            # Condição LVQ2.1:
            # Um dos dois deve ter o rótulo correto, o outro não
            # E a amostra deve estar na janela entre eles

            um_certo = (rotulo1 == rotulo_real) != (rotulo2 == rotulo_real)

            if not um_certo:
                continue

            # Verifica se está na janela
            # Condição: min(d1/d2, d2/d1) > (1 - janela) / (1 + janela)
            if d2 > 0:
                razao = d1 / d2
                limiar = (1 - janela) / (1 + janela)
                na_janela = min(razao, 1.0 / razao) > limiar
            else:
                na_janela = False

            if not na_janela:
                continue

            # Identifica qual é o correto e qual é o errado
            if rotulo1 == rotulo_real:
                idx_correto = idx1
                idx_errado = idx2
            else:
                idx_correto = idx2
                idx_errado = idx1

            # Atualiza: puxa o correto, empurra o errado
            self.prototipos[idx_correto] += lr * (x - self.prototipos[idx_correto])
            self.prototipos[idx_errado] -= lr * (x - self.prototipos[idx_errado])

    def classificar(self, x):
        """Classifica pelo protótipo mais próximo."""
        distancias = np.sqrt(np.sum((self.prototipos - x) ** 2, axis=1))
        idx = np.argmin(distancias)
        return self.rotulos[idx]


# ===========================================================================
# EXTRAÇÃO DE FEATURES
# ===========================================================================
def carregar_audio(caminho, sr=SR):
    y, _ = librosa.load(caminho, sr=sr)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y


def extrair_26(y, sr=SR, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])


def extrair_58(y, sr=SR, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_m = np.mean(mfccs, axis=1)
    mfcc_d = np.std(mfccs, axis=1)
    delta = librosa.feature.delta(mfccs)
    delta_m = np.mean(delta, axis=1)
    delta_d = np.std(delta, axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.concatenate([
        mfcc_m, mfcc_d, delta_m, delta_d,
        [np.mean(cent), np.std(cent)],
        [np.mean(zcr), np.std(zcr)],
        [np.mean(roll), np.std(roll)]
    ])


def normalizar(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    return (X - X_min) / X_range


# ===========================================================================
# CARREGAMENTO DO DATASET
# ===========================================================================
print("=" * 70)
print("CARREGAMENTO DO DATASET")
print("=" * 70)

feat_26, feat_58, rotulos = [], [], []

for especie in ESPECIES:
    pasta = os.path.join(DADOS_DIR, especie)
    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".mp3")])
    print(f"  {especie}: {len(arquivos)} arquivos")
    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)
        try:
            y = carregar_audio(caminho)
            feat_26.append(extrair_26(y))
            feat_58.append(extrair_58(y))
            rotulos.append(especie)
        except Exception as e:
            print(f"    ✗ {arquivo}: {e}")

X_26 = normalizar(np.array(feat_26))
X_58 = normalizar(np.array(feat_58))
y_all = np.array(rotulos)
print(f"\n  Total: {len(y_all)} amostras")
print(f"  X_26: {X_26.shape} | X_58: {X_58.shape}")


# ===========================================================================
# FUNÇÃO DE EXPERIMENTO
# ===========================================================================
def rodar(X, y_all, n_feat, usar_lvq, seed):
    """
    Executa um experimento: split → treinar SOM → (LVQ opcional) → avaliar.
    Retorna acurácia no teste.
    """
    np.random.seed(seed)

    # Split estratificado
    idx_tr, idx_te = [], []
    for esp in ESPECIES:
        ids = np.where(y_all == esp)[0].tolist()
        np.random.shuffle(ids)
        corte = max(1, int(len(ids) * SPLIT_TREINO))
        idx_tr.extend(ids[:corte])
        idx_te.extend(ids[corte:])

    X_tr, y_tr = X[idx_tr], y_all[idx_tr]
    X_te, y_te = X[idx_te], y_all[idx_te]

    # Treinar SOM
    som = SOM(SOM_X, SOM_Y, n_feat, seed=seed)
    som.treinar(X_tr, SOM_EPOCAS)
    som.rotular_neuronios(X_tr, y_tr)

    if not usar_lvq:
        # Classificação SOM puro
        acertos = sum(1 for i, x in enumerate(X_te) if som.classificar(x) == y_te[i])
        # Também retorna métricas por espécie
        metricas = calcular_metricas(X_te, y_te, lambda x: som.classificar(x))
        return acertos / len(y_te), metricas
    else:
        # Extrair protótipos do SOM e refinar com LVQ
        protos, rots, poss = som.obter_prototipos()

        if len(protos) < 2:
            # Fallback: SOM não rotulou neurônios suficientes
            acertos = sum(1 for i, x in enumerate(X_te) if som.classificar(x) == y_te[i])
            metricas = calcular_metricas(X_te, y_te, lambda x: som.classificar(x))
            return acertos / len(y_te), metricas

        lvq = LVQ21(protos, rots, poss)
        lvq.treinar(X_tr, y_tr)

        acertos = sum(1 for i, x in enumerate(X_te) if lvq.classificar(x) == y_te[i])
        metricas = calcular_metricas(X_te, y_te, lambda x: lvq.classificar(x))
        return acertos / len(y_te), metricas


def calcular_metricas(X_te, y_te, classificar_fn):
    """Calcula precisão e recall por espécie."""
    preds = [classificar_fn(x) for x in X_te]
    metricas = {}
    for esp in ESPECIES:
        vp = sum(1 for r, p in zip(y_te, preds) if r == esp and p == esp)
        total_pred = sum(1 for p in preds if p == esp)
        total_real = sum(1 for r in y_te if r == esp)
        prec = vp / total_pred if total_pred > 0 else 0
        rec = vp / total_real if total_real > 0 else 0
        metricas[esp] = {"precisao": prec, "recall": rec}
    return metricas


# ===========================================================================
# EXECUÇÃO DOS 4 EXPERIMENTOS
# ===========================================================================
print("\n" + "=" * 70)
print(f"EXECUTANDO 4 CONFIGURAÇÕES × {len(SEMENTES)} SEMENTES = {4 * len(SEMENTES)} TREINAMENTOS")
print("=" * 70)

configs = [
    ("SOM 26",     X_26, 26, False),
    ("SOM 58",     X_58, 58, False),
    ("SOM+LVQ 26", X_26, 26, True),
    ("SOM+LVQ 58", X_58, 58, True),
]

resultados = {}
metricas_melhores = {}

for nome, X, n_feat, usar_lvq in configs:
    resultados[nome] = []
    melhor_acc = -1
    melhor_met = None

    for seed in SEMENTES:
        metodo = "SOM+LVQ" if usar_lvq else "SOM"
        print(f"  [{nome}] semente {seed}...", end=" ", flush=True)

        acc, met = rodar(X, y_all, n_feat, usar_lvq, seed)
        resultados[nome].append(acc)
        print(f"acurácia = {acc:.2%}")

        if acc > melhor_acc:
            melhor_acc = acc
            melhor_met = met

    metricas_melhores[nome] = melhor_met
    print()


# ===========================================================================
# ANÁLISE ESTATÍSTICA
# ===========================================================================
print("=" * 70)
print("RESULTADOS COMPLETOS")
print("=" * 70)

stats = {}
for nome in resultados:
    accs = np.array(resultados[nome])
    stats[nome] = {
        "media": accs.mean() * 100,
        "desvio": accs.std() * 100,
        "min": accs.min() * 100,
        "max": accs.max() * 100,
        "mediana": np.median(accs) * 100,
        "todas": accs * 100,
    }

print(f"\n  {'Configuração':<16} {'Média':>10} {'±DP':>8} {'Min':>8} {'Máx':>8} {'Mediana':>10}")
print(f"  {'-' * 62}")
for nome in resultados:
    s = stats[nome]
    print(f"  {nome:<16} {s['media']:>9.2f}% {s['desvio']:>7.2f}% "
          f"{s['min']:>7.2f}% {s['max']:>7.2f}% {s['mediana']:>9.2f}%")

# Tabela 2x2
print(f"\n  TABELA 2×2 (acurácia média %):")
print(f"  {'':>16} {'26 features':>14} {'58 features':>14}")
print(f"  {'-' * 46}")
print(f"  {'SOM puro':<16} {stats['SOM 26']['media']:>13.2f}% {stats['SOM 58']['media']:>13.2f}%")
print(f"  {'SOM + LVQ':<16} {stats['SOM+LVQ 26']['media']:>13.2f}% {stats['SOM+LVQ 58']['media']:>13.2f}%")

# Ganho do LVQ
ganho_26 = stats["SOM+LVQ 26"]["media"] - stats["SOM 26"]["media"]
ganho_58 = stats["SOM+LVQ 58"]["media"] - stats["SOM 58"]["media"]
print(f"\n  Ganho do LVQ (26 features): {ganho_26:+.2f}pp")
print(f"  Ganho do LVQ (58 features): {ganho_58:+.2f}pp")

# Melhor configuração
melhor_nome = max(stats, key=lambda k: stats[k]["media"])
print(f"\n  ★ MELHOR CONFIGURAÇÃO: {melhor_nome} ({stats[melhor_nome]['media']:.2f}% ± {stats[melhor_nome]['desvio']:.2f}%)")

# Métricas por espécie da melhor
print(f"\n  Métricas por espécie ({melhor_nome}, melhor execução):")
for esp in ESPECIES:
    m = metricas_melhores[melhor_nome][esp]
    nome_esp = esp.replace("_", " ").title()
    print(f"    {nome_esp:<15} Precisão: {m['precisao']:.2%}  Recall: {m['recall']:.2%}")


# ===========================================================================
# SALVAR RESULTADOS TXT
# ===========================================================================
with open("exp_completo_resultados.txt", "w", encoding="utf-8") as f:
    f.write("EXPERIMENTO COMPLETO: SOM vs SOM+LVQ × 26 vs 58 FEATURES\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"SOM: grade {SOM_X}×{SOM_Y}, {SOM_EPOCAS} épocas\n")
    f.write(f"LVQ: {LVQ_EPOCAS} épocas, janela={LVQ_JANELA}\n")
    f.write(f"Split: {int(SPLIT_TREINO*100)}/{int((1-SPLIT_TREINO)*100)}\n")
    f.write(f"Sementes: {SEMENTES}\n\n")

    # Tabela por semente
    f.write("Resultados por semente:\n")
    f.write(f"  {'Semente':<8}")
    for nome in resultados:
        f.write(f" {nome:>12}")
    f.write("\n  " + "-" * 58 + "\n")
    for i, seed in enumerate(SEMENTES):
        f.write(f"  {seed:<8}")
        for nome in resultados:
            f.write(f" {resultados[nome][i]*100:>11.2f}%")
        f.write("\n")

    # Estatísticas
    f.write("\nEstatísticas:\n")
    f.write(f"  {'Métrica':<10}")
    for nome in resultados:
        f.write(f" {nome:>12}")
    f.write("\n  " + "-" * 58 + "\n")
    for met in ["media", "desvio", "min", "max", "mediana"]:
        f.write(f"  {met:<10}")
        for nome in resultados:
            f.write(f" {stats[nome][met]:>11.2f}%")
        f.write("\n")

    # Tabela 2x2
    f.write(f"\nTabela 2×2 (acurácia média %):\n")
    f.write(f"  {'':>16} {'26 features':>14} {'58 features':>14}\n")
    f.write(f"  SOM puro       {stats['SOM 26']['media']:>13.2f}% {stats['SOM 58']['media']:>13.2f}%\n")
    f.write(f"  SOM + LVQ      {stats['SOM+LVQ 26']['media']:>13.2f}% {stats['SOM+LVQ 58']['media']:>13.2f}%\n")
    f.write(f"\nGanho do LVQ (26 feat): {ganho_26:+.2f}pp\n")
    f.write(f"Ganho do LVQ (58 feat): {ganho_58:+.2f}pp\n")
    f.write(f"\nMelhor: {melhor_nome} ({stats[melhor_nome]['media']:.2f}% ± {stats[melhor_nome]['desvio']:.2f}%)\n")

print(f"\n  Salvo: exp_completo_resultados.txt")


# ===========================================================================
# GRÁFICO 1: BOXPLOT DAS 4 CONFIGURAÇÕES
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

cores = ["#457B9D", "#E63946", "#6A994E", "#BC4749"]
dados = [stats[n]["todas"] for n in resultados]
labels = list(resultados.keys())

bp = ax.boxplot(dados, labels=labels, patch_artist=True, widths=0.5,
                medianprops={"color": "black", "linewidth": 2})

for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(cores[i])
    box.set_alpha(0.7)

for i, d in enumerate(dados):
    x_pos = np.random.normal(i + 1, 0.04, size=len(d))
    ax.scatter(x_pos, d, color="black", alpha=0.6, s=40, zorder=3)

# Anotações
for i, nome in enumerate(resultados):
    s = stats[nome]
    ax.annotate(f"μ={s['media']:.1f}%\nσ={s['desvio']:.1f}%",
                xy=(i + 1, s['media']), xytext=(i + 1.3, s['media']),
                fontsize=9, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
ax.text(0.55, 34.5, "Acaso (33.3%)", fontsize=9, color="gray", style="italic")

ax.set_ylabel("Acurácia (%)", fontsize=12, fontweight="bold")
ax.set_title("Comparação: SOM vs SOM+LVQ × 26 vs 58 features\n(5 execuções por configuração)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

ymin = min(s["min"] for s in stats.values()) - 5
ymax = max(s["max"] for s in stats.values()) + 10
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("exp_completo_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: exp_completo_boxplot.png")


# ===========================================================================
# GRÁFICO 2: BARRAS AGRUPADAS
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.array([0, 1.2])
largura_barra = 0.35

som_medias = [stats["SOM 26"]["media"], stats["SOM 58"]["media"]]
som_desvios = [stats["SOM 26"]["desvio"], stats["SOM 58"]["desvio"]]
lvq_medias = [stats["SOM+LVQ 26"]["media"], stats["SOM+LVQ 58"]["media"]]
lvq_desvios = [stats["SOM+LVQ 26"]["desvio"], stats["SOM+LVQ 58"]["desvio"]]

bars1 = ax.bar(x_pos - largura_barra/2, som_medias, largura_barra,
               yerr=som_desvios, capsize=8, label="SOM puro",
               color="#457B9D", alpha=0.8, edgecolor="black",
               error_kw={"linewidth": 1.5})

bars2 = ax.bar(x_pos + largura_barra/2, lvq_medias, largura_barra,
               yerr=lvq_desvios, capsize=8, label="SOM + LVQ",
               color="#E63946", alpha=0.8, edgecolor="black",
               error_kw={"linewidth": 1.5})

# Valores em cima
for bars, medias, desvios in [(bars1, som_medias, som_desvios), (bars2, lvq_medias, lvq_desvios)]:
    for bar, m, d in zip(bars, medias, desvios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + d + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
ax.text(0, 35, "Acaso", fontsize=9, color="gray", style="italic")

ax.set_xticks(x_pos)
ax.set_xticklabels(["26 features\n(MFCC)", "58 features\n(MFCC estendido)"], fontsize=11)
ax.set_ylabel("Acurácia média (%)", fontsize=12, fontweight="bold")
ax.set_title("SOM vs SOM+LVQ por representação de features\n(média ± desvio padrão, 5 execuções)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("exp_completo_barras.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: exp_completo_barras.png")


# ===========================================================================
# GRÁFICO 3: TABELA 2x2 VISUAL
# ===========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

tabela_dados = [
    [f"{stats['SOM 26']['media']:.1f}% ± {stats['SOM 26']['desvio']:.1f}%",
     f"{stats['SOM 58']['media']:.1f}% ± {stats['SOM 58']['desvio']:.1f}%"],
    [f"{stats['SOM+LVQ 26']['media']:.1f}% ± {stats['SOM+LVQ 26']['desvio']:.1f}%",
     f"{stats['SOM+LVQ 58']['media']:.1f}% ± {stats['SOM+LVQ 58']['desvio']:.1f}%"],
]

tabela = ax.table(
    cellText=tabela_dados,
    rowLabels=["SOM puro", "SOM + LVQ"],
    colLabels=["26 features (MFCC)", "58 features (estendido)"],
    loc="center",
    cellLoc="center",
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(13)
tabela.scale(1.5, 2.2)

# Colorir cells baseado na acurácia
valores = [stats["SOM 26"]["media"], stats["SOM 58"]["media"],
           stats["SOM+LVQ 26"]["media"], stats["SOM+LVQ 58"]["media"]]
v_min, v_max = min(valores), max(valores)

for i in range(2):
    for j in range(2):
        val = float(tabela_dados[i][j].split("%")[0])
        # Gradiente de cor: verde mais forte = melhor
        intensidade = (val - v_min) / (v_max - v_min) if v_max > v_min else 0.5
        r = 1.0 - intensidade * 0.4
        g = 0.85 + intensidade * 0.15
        b = 1.0 - intensidade * 0.4
        tabela[i + 1, j].set_facecolor((r, g, b))

# Cabeçalho
for j in range(2):
    tabela[0, j].set_facecolor("#2C3E50")
    tabela[0, j].set_text_props(color="white", fontweight="bold")

for i in range(2):
    tabela[i + 1, -1].set_facecolor("#34495E")
    tabela[i + 1, -1].set_text_props(color="white", fontweight="bold")

ax.set_title("Acurácia média (%) — Tabela 2×2\nRede × Representação de features",
             fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("exp_completo_tabela2x2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: exp_completo_tabela2x2.png")


# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print("\n" + "=" * 70)
print("CONCLUÍDO!")
print("=" * 70)
print(f"""
  Arquivos gerados:
    exp_completo_resultados.txt     — Tabela completa
    exp_completo_boxplot.png        — Boxplot das 4 configurações
    exp_completo_barras.png         — Barras agrupadas SOM vs SOM+LVQ
    exp_completo_tabela2x2.png      — Tabela 2×2 visual

  TABELA 2×2 PARA O ARTIGO:

                   26 features    58 features
  SOM puro          {stats['SOM 26']['media']:>6.1f}%         {stats['SOM 58']['media']:>6.1f}%
  SOM + LVQ         {stats['SOM+LVQ 26']['media']:>6.1f}%         {stats['SOM+LVQ 58']['media']:>6.1f}%

  Ganho do LVQ com 26 features: {ganho_26:+.1f}pp
  Ganho do LVQ com 58 features: {ganho_58:+.1f}pp

  ★ MELHOR: {melhor_nome} → {stats[melhor_nome]['media']:.1f}% ± {stats[melhor_nome]['desvio']:.1f}%
""")