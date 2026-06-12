"""
=============================================================================
Identificação de Espécies de Aves da Mata Atlântica por Canto (Bioacústica)
Utilizando: Mapa Auto-Organizável de Kohonen (SOM) — Implementação do zero
=============================================================================

Estrutura esperada:
    dados/
    ├── araponga/     (10 arquivos .mp3)
    ├── bem_te_vi/    (10 arquivos .mp3)
    └── urutau/       (10 arquivos .mp3)

Dependências:
    pip install librosa numpy matplotlib
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

# Parâmetros MFCC
N_MFCC = 13
SR = 22050

# Parâmetros SOM
SOM_X, SOM_Y = 5, 5          # tamanho da grade (64 neurônios)
SOM_EPOCAS = 10000            # épocas de treinamento
SIGMA_INICIAL = 3.5           # raio de vizinhança inicial
SIGMA_FINAL = 0.5             # raio de vizinhança final
LR_INICIAL = 0.5              # taxa de aprendizado inicial
LR_FINAL = 0.01               # taxa de aprendizado final


# ===========================================================================
# IMPLEMENTAÇÃO DO SOM (DO ZERO)
# ===========================================================================
class SOM:
    """
    Mapa Auto-Organizável de Kohonen.

    O SOM organiza neurônios em uma grade 2D onde cada neurônio possui
    um vetor de pesos com a mesma dimensão dos dados de entrada.
    Através do aprendizado competitivo, neurônios vizinhos aprendem a
    representar padrões similares, criando um mapa topológico dos dados.
    """

    def __init__(self, largura, altura, n_features, seed=42):
        """
        Parâmetros:
            largura:    número de colunas da grade
            altura:     número de linhas da grade
            n_features: dimensão do vetor de entrada
        """
        self.largura = largura
        self.altura = altura
        self.n_features = n_features

        # Inicializa pesos aleatórios entre 0 e 1
        np.random.seed(seed)
        self.pesos = np.random.rand(largura, altura, n_features)

        # Pré-calcula as coordenadas de cada neurônio na grade
        # Usado para calcular distâncias entre neurônios
        self.coordenadas = np.array(
            [[[i, j] for j in range(altura)] for i in range(largura)]
        )

    def _encontrar_bmu(self, x):
        """
        Encontra o BMU (Best Matching Unit) — o neurônio vencedor.

        Calcula a distância euclidiana entre o vetor de entrada x
        e os pesos de todos os neurônios. O neurônio com menor
        distância é o vencedor da competição.

        Este é o coração do APRENDIZADO COMPETITIVO:
        todos os neurônios competem e apenas um vence.
        """
        # Distância euclidiana de x para cada neurônio
        diferencas = self.pesos - x  # broadcasting: (L, A, F) - (F,)
        distancias = np.sqrt(np.sum(diferencas ** 2, axis=2))  # (L, A)

        # Encontra a posição do neurônio com menor distância
        idx = np.unravel_index(np.argmin(distancias), distancias.shape)
        return idx  # (i, j) na grade

    def _funcao_vizinhanca(self, bmu, sigma):
        """
        Calcula a influência de cada neurônio baseado na distância
        ao BMU usando uma função gaussiana.

        Neurônios próximos ao vencedor recebem alta influência,
        neurônios distantes recebem influência próxima de zero.

        É como jogar uma pedra na água: o impacto é forte no centro
        (BMU) e vai diminuindo nas ondas (vizinhança).
        """
        bmu_coord = np.array([bmu[0], bmu[1]])

        # Distância de cada neurônio ao BMU na grade
        dist_ao_bmu = np.sqrt(np.sum(
            (self.coordenadas - bmu_coord) ** 2, axis=2
        ))

        # Gaussiana: e^(-d² / 2σ²)
        return np.exp(-(dist_ao_bmu ** 2) / (2 * sigma ** 2))

    def _decaimento(self, valor_inicial, valor_final, epoca, total_epocas):
        """
        Calcula o decaimento exponencial de um parâmetro.
        Usado para reduzir gradualmente o raio de vizinhança
        e a taxa de aprendizado ao longo do treinamento.
        """
        return valor_inicial * (valor_final / valor_inicial) ** (epoca / total_epocas)

    def treinar(self, X, epocas, sigma_i=SIGMA_INICIAL, sigma_f=SIGMA_FINAL,
                lr_i=LR_INICIAL, lr_f=LR_FINAL, verbose=True):
        """
        Treina o SOM com os dados X.

        Para cada época:
            1. Seleciona uma amostra aleatória
            2. Encontra o BMU (competição)
            3. Calcula a vizinhança (cooperação)
            4. Atualiza os pesos (adaptação)

        Os parâmetros sigma (raio) e lr (taxa) decaem ao longo
        do treinamento: início = organização grosseira,
        fim = ajuste fino.
        """
        self.historico_erro = []
        n_amostras = len(X)

        for epoca in range(epocas):
            # Decaimento dos parâmetros
            sigma = self._decaimento(sigma_i, sigma_f, epoca, epocas)
            lr = self._decaimento(lr_i, lr_f, epoca, epocas)

            # Seleciona amostra aleatória
            idx = np.random.randint(0, n_amostras)
            x = X[idx]

            # 1. COMPETIÇÃO: encontra o neurônio vencedor
            bmu = self._encontrar_bmu(x)

            # 2. COOPERAÇÃO: calcula influência da vizinhança
            vizinhanca = self._funcao_vizinhanca(bmu, sigma)

            # 3. ADAPTAÇÃO: atualiza pesos do BMU e vizinhos
            # peso_novo = peso_atual + lr * vizinhança * (x - peso_atual)
            for i in range(self.largura):
                for j in range(self.altura):
                    influencia = lr * vizinhanca[i, j]
                    self.pesos[i, j] += influencia * (x - self.pesos[i, j])

            # Calcula erro de quantização (média das distâncias ao BMU)
            if (epoca + 1) % 500 == 0:
                erro = self._erro_quantizacao(X)
                self.historico_erro.append(erro)
                if verbose:
                    print(f"    Época {epoca+1:>5}/{epocas} | "
                          f"σ={sigma:.2f} | lr={lr:.4f} | erro={erro:.4f}")

        if verbose:
            print("    Treinamento concluído!")

    def _erro_quantizacao(self, X):
        """
        Calcula o erro médio de quantização.
        É a média das distâncias entre cada amostra e seu BMU.
        Quanto menor, melhor o mapa representa os dados.
        """
        erro_total = 0
        for x in X:
            bmu = self._encontrar_bmu(x)
            erro_total += np.linalg.norm(x - self.pesos[bmu[0], bmu[1]])
        return erro_total / len(X)

    def vencedor(self, x):
        """Retorna a posição (i, j) do neurônio vencedor para a amostra x."""
        return self._encontrar_bmu(x)

    def u_matrix(self):
        """
        Calcula a U-Matrix (Unified Distance Matrix).

        Para cada neurônio, calcula a média das distâncias aos seus
        vizinhos diretos. Valores altos indicam fronteiras entre
        clusters, valores baixos indicam regiões homogêneas.
        """
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
                            dist = np.linalg.norm(
                                self.pesos[i, j] - self.pesos[ni, nj]
                            )
                            vizinhos.append(dist)
                umat[i, j] = np.mean(vizinhos)

        return umat

    def rotular_neuronios(self, X, rotulos):
        """
        Rotula cada neurônio com a classe mais frequente entre
        as amostras que ele vence. Permite usar o SOM como classificador.
        """
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
        """Classifica uma amostra pelo rótulo do neurônio vencedor."""
        bmu = self._encontrar_bmu(x)
        return self.mapa_rotulos.get(bmu, None)


# ===========================================================================
# ETAPA 1 — CARREGAMENTO E PRÉ-PROCESSAMENTO
# ===========================================================================
print("=" * 60)
print("ETAPA 1: Carregamento e Pré-processamento dos Áudios")
print("=" * 60)


def carregar_audio(caminho, sr=SR):
    """Carrega arquivo de áudio e normaliza o volume."""
    y, _ = librosa.load(caminho, sr=sr)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y


def extrair_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    """
    Extrai features MFCC de um sinal de áudio.
    Retorna vetor de 2*n_mfcc features (média + desvio padrão).
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    media = np.mean(mfccs, axis=1)
    desvio = np.std(mfccs, axis=1)
    return np.concatenate([media, desvio])


# Carregar todos os arquivos
features = []
rotulos = []
nomes_arquivos = []

for especie in ESPECIES:
    pasta = os.path.join(DADOS_DIR, especie)
    arquivos = [f for f in os.listdir(pasta) if f.endswith(".mp3")]
    print(f"\n  {especie}: {len(arquivos)} arquivos encontrados")

    for arquivo in sorted(arquivos):
        caminho = os.path.join(pasta, arquivo)
        try:
            y = carregar_audio(caminho)
            mfcc = extrair_mfcc(y)
            features.append(mfcc)
            rotulos.append(especie)
            nomes_arquivos.append(arquivo)
            print(f"    ✓ {arquivo} ({len(y)/SR:.1f}s)")
        except Exception as e:
            print(f"    ✗ {arquivo}: {e}")

X = np.array(features)
y_rotulos = np.array(rotulos)

print(f"\n  Total: {len(X)} amostras, {X.shape[1]} features cada")

# Normalização Min-Max manual (0 a 1)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_range = X_max - X_min
X_range[X_range == 0] = 1
X_norm = (X - X_min) / X_range


# ===========================================================================
# ETAPA 2 — VISUALIZAÇÃO DOS ESPECTROGRAMAS
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 2: Visualização de Espectrogramas (1 por espécie)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, especie in enumerate(ESPECIES):
    pasta = os.path.join(DADOS_DIR, especie)
    arquivo = sorted([f for f in os.listdir(pasta) if f.endswith(".mp3")])[0]
    y_audio = carregar_audio(os.path.join(pasta, arquivo))
    S = librosa.feature.melspectrogram(y=y_audio, sr=SR)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=SR, x_axis="time", y_axis="mel", ax=axes[i])
    axes[i].set_title(especie.replace("_", " ").title(), fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Tempo (s)")
    axes[i].set_ylabel("Frequência (Hz)" if i == 0 else "")

plt.suptitle("Espectrogramas Mel — Uma Amostra por Espécie", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("01_espectrogramas.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: 01_espectrogramas.png")


# ===========================================================================
# ETAPA 3 — TREINAMENTO DO SOM
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 3: Treinamento do SOM (Aprendizado Competitivo)")
print("=" * 60)

som = SOM(
    largura=SOM_X,
    altura=SOM_Y,
    n_features=X_norm.shape[1],
    seed=42,
)

print(f"  Grade: {SOM_X}x{SOM_Y} = {SOM_X * SOM_Y} neurônios")
print(f"  Features: {X_norm.shape[1]} (MFCC média + desvio)")
print(f"  Épocas: {SOM_EPOCAS}")
print(f"  Treinando...\n")

som.treinar(X_norm, epocas=SOM_EPOCAS)


# ===========================================================================
# ETAPA 4 — VISUALIZAÇÕES DO SOM
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 4: Visualizações do Mapa Auto-Organizável")
print("=" * 60)

# --- 4a: Curva de erro de quantização ---
fig, ax = plt.subplots(figsize=(8, 4))
epocas_plot = [(i + 1) * 500 for i in range(len(som.historico_erro))]
ax.plot(epocas_plot, som.historico_erro, "o-", color="#457B9D", linewidth=1.5, markersize=4)
ax.set_title("Erro de Quantização ao Longo do Treinamento", fontsize=12, fontweight="bold")
ax.set_xlabel("Época")
ax.set_ylabel("Erro Médio de Quantização")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("02_erro_quantizacao.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: 02_erro_quantizacao.png")

# --- 4b: U-Matrix + Mapeamento das amostras ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# U-Matrix
umatrix = som.u_matrix()
im = axes[0].imshow(umatrix.T, cmap="bone_r", origin="lower")
axes[0].set_title("U-Matrix (Distâncias entre Neurônios)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Neurônio X")
axes[0].set_ylabel("Neurônio Y")
plt.colorbar(im, ax=axes[0], fraction=0.046, label="Distância média")

# Mapa com amostras
for i, x in enumerate(X_norm):
    w = som.vencedor(x)
    especie = y_rotulos[i]
    axes[1].plot(
        w[0] + np.random.uniform(-0.25, 0.25),
        w[1] + np.random.uniform(-0.25, 0.25),
        MARCADORES[especie],
        color=CORES[especie],
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

axes[1].set_xlim(-0.5, SOM_X - 0.5)
axes[1].set_ylim(-0.5, SOM_Y - 0.5)
axes[1].set_title("Amostras Mapeadas no SOM", fontsize=12, fontweight="bold")
axes[1].legend(
    handles=[Patch(color=CORES[e], label=e.replace("_", " ").title()) for e in ESPECIES],
    loc="upper right", fontsize=10,
)
axes[1].set_xlabel("Neurônio X")
axes[1].set_ylabel("Neurônio Y")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Mapa Auto-Organizável de Kohonen (SOM)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("03_som_mapa.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: 03_som_mapa.png")

# --- 4c: Mapa de ativação por espécie ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, especie in enumerate(ESPECIES):
    mapa_ativacao = np.zeros((SOM_X, SOM_Y))
    amostras_especie = X_norm[y_rotulos == especie]
    for x in amostras_especie:
        w = som.vencedor(x)
        mapa_ativacao[w[0], w[1]] += 1

    im = axes[idx].imshow(mapa_ativacao.T, cmap="YlOrRd", origin="lower")
    axes[idx].set_title(especie.replace("_", " ").title(), fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Neurônio X")
    axes[idx].set_ylabel("Neurônio Y" if idx == 0 else "")
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

plt.suptitle("Mapa de Ativação por Espécie no SOM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("04_som_ativacao.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: 04_som_ativacao.png")


# ===========================================================================
# ETAPA 5 — CLASSIFICAÇÃO USANDO O SOM
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 5: Classificação com o SOM")
print("=" * 60)

# Split treino/teste manual estratificado (70/30)
np.random.seed(42)
idx_treino, idx_teste = [], []
for especie in ESPECIES:
    indices_especie = np.where(y_rotulos == especie)[0].tolist()
    np.random.shuffle(indices_especie)
    corte = max(1, int(len(indices_especie) * 0.7))
    idx_treino.extend(indices_especie[:corte])
    idx_teste.extend(indices_especie[corte:])

X_treino = X_norm[idx_treino]
y_treino = y_rotulos[idx_treino]
X_teste = X_norm[idx_teste]
y_teste = y_rotulos[idx_teste]

print(f"  Treino: {len(X_treino)} amostras | Teste: {len(X_teste)} amostras")

# Rotular neurônios com dados de treino
som.rotular_neuronios(X_treino, y_treino)

# Contar neurônios rotulados
rotulados = sum(1 for v in som.mapa_rotulos.values() if v is not None)
print(f"  Neurônios rotulados: {rotulados}/{SOM_X * SOM_Y}")

# Classificar amostras de teste
y_pred = []
for x in X_teste:
    pred = som.classificar(x)
    y_pred.append(pred if pred is not None else "desconhecido")
y_pred = np.array(y_pred)

# Acurácia geral
acertos = np.sum(y_pred == y_teste)
acuracia = acertos / len(y_teste)
print(f"\n  Acurácia geral: {acuracia:.2%} ({acertos}/{len(y_teste)})")

# Métricas por espécie
print(f"\n  {'Espécie':<15} {'Precisão':>10} {'Recall':>10} {'Amostras':>10}")
print(f"  {'-' * 45}")
for especie in ESPECIES:
    mask_pred = y_pred == especie
    mask_real = y_teste == especie
    vp = np.sum(mask_pred & mask_real)
    total_pred = np.sum(mask_pred)
    total_real = np.sum(mask_real)
    precisao = vp / total_pred if total_pred > 0 else 0
    recall = vp / total_real if total_real > 0 else 0
    nome = especie.replace("_", " ").title()
    print(f"  {nome:<15} {precisao:>10.2%} {recall:>10.2%} {total_real:>10}")

# Matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_classes = len(ESPECIES)
cm = np.zeros((n_classes, n_classes), dtype=int)
for real, pred in zip(y_teste, y_pred):
    if real in ESPECIES and pred in ESPECIES:
        i = ESPECIES.index(real)
        j = ESPECIES.index(pred)
        cm[i, j] += 1

nomes_display = [e.replace("_", " ").title() for e in ESPECIES]
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks(range(n_classes))
axes[0].set_yticks(range(n_classes))
axes[0].set_xticklabels(nomes_display, rotation=45, ha="right")
axes[0].set_yticklabels(nomes_display)
axes[0].set_xlabel("Predito")
axes[0].set_ylabel("Real")
for i in range(n_classes):
    for j in range(n_classes):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
axes[0].set_title("Matriz de Confusão", fontsize=12, fontweight="bold")

# Mapa de rótulos dos neurônios
mapa_rotulos_visual = np.full((SOM_X, SOM_Y), -1)
for (i, j), rotulo in som.mapa_rotulos.items():
    if rotulo is not None:
        mapa_rotulos_visual[i, j] = ESPECIES.index(rotulo)

cmap_custom = plt.cm.colors.ListedColormap(
    ["#DDDDDD", "#E63946", "#457B9D", "#2A9D8F"]
)
axes[1].imshow(mapa_rotulos_visual.T + 1, cmap=cmap_custom, origin="lower",
               vmin=0, vmax=3)
axes[1].set_title("Mapa de Rótulos dos Neurônios", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Neurônio X")
axes[1].set_ylabel("Neurônio Y")
axes[1].legend(
    handles=[
        Patch(color="#DDDDDD", label="Vazio"),
        *[Patch(color=CORES[e], label=e.replace("_", " ").title()) for e in ESPECIES],
    ],
    loc="upper right", fontsize=9,
)
axes[1].grid(True, alpha=0.2)

plt.suptitle("Classificação com o SOM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("05_classificacao.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Salvo: 05_classificacao.png")


# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print("\n" + "=" * 60)
print("RESUMO DO PROJETO")
print("=" * 60)
print(f"""
  Espécies analisadas: {', '.join(e.replace('_', ' ').title() for e in ESPECIES)}
  Total de amostras:   {len(X)} ({', '.join(f'{e.replace("_", " ").title()}: {np.sum(y_rotulos == e)}' for e in ESPECIES)})
  Features extraídas:  {X.shape[1]} (MFCC: {N_MFCC} médias + {N_MFCC} desvios)

  SOM:
    Grade:             {SOM_X}x{SOM_Y} = {SOM_X * SOM_Y} neurônios
    Épocas:            {SOM_EPOCAS}
    Vizinhança:        σ = {SIGMA_INICIAL} → {SIGMA_FINAL} (decaimento exponencial)
    Taxa aprendizado:  lr = {LR_INICIAL} → {LR_FINAL} (decaimento exponencial)
    Erro final:        {som.historico_erro[-1]:.4f}

  Classificação:
    Acurácia:          {acuracia:.2%}

  Arquivos gerados:
    01_espectrogramas.png      — Espectrogramas Mel (1 por espécie)
    02_erro_quantizacao.png    — Curva de convergência do SOM
    03_som_mapa.png            — U-Matrix + Mapeamento das amostras
    04_som_ativacao.png        — Mapa de ativação por espécie
    05_classificacao.png       — Matriz de confusão + Mapa de rótulos
""")
print("Concluído!")


# ===========================================================================
# ETAPA 6 — TESTE COM ÁUDIO NOVO (SHAZAM DE PASSARINHO)
# ===========================================================================
def identificar_canto(caminho_audio):
    """
    Recebe o caminho de um arquivo MP3 e identifica a espécie.
    Usa o SOM já treinado para classificação.
    """
    print(f"\n  Analisando: {caminho_audio}")

    # Carrega e processa o áudio
    y = carregar_audio(caminho_audio)
    duracao = len(y) / SR
    print(f"  Duração: {duracao:.1f}s")

    # Extrai features MFCC
    mfcc = extrair_mfcc(y)

    # Normaliza com os mesmos parâmetros do treino
    mfcc_norm = (mfcc - X_min) / X_range

    # Encontra o neurônio vencedor
    bmu = som.vencedor(mfcc_norm)
    especie = som.classificar(mfcc_norm)

    # Calcula distância ao BMU (confiança)
    dist_bmu = np.linalg.norm(mfcc_norm - som.pesos[bmu[0], bmu[1]])

    # Calcula distância aos protótipos de cada espécie para comparação
    print(f"\n  ╔══════════════════════════════════════╗")
    if especie:
        nome = especie.replace("_", " ").title()
        print(f"  ║  RESULTADO: {nome:^24s} ║")
    else:
        print(f"  ║  RESULTADO: {'Desconhecido':^24s} ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"  Neurônio vencedor: ({bmu[0]}, {bmu[1]})")
    print(f"  Distância ao BMU: {dist_bmu:.4f}")

    # Mostra distância a cada espécie (média dos protótipos)
    print(f"\n  Similaridade por espécie:")
    for esp in ESPECIES:
        amostras_esp = X_norm[y_rotulos == esp]
        media_esp = np.mean(amostras_esp, axis=0)
        dist = np.linalg.norm(mfcc_norm - media_esp)
        barra = "█" * max(1, int(30 * (1 - dist / 2)))
        nome = esp.replace("_", " ").title()
        marcador = " ← mais provável" if esp == especie else ""
        print(f"    {nome:<15} {barra} ({dist:.3f}){marcador}")

    return especie


# Loop interativo
print("\n" + "=" * 60)
print("MODO IDENTIFICAÇÃO — Shazam de Passarinho 🐦")
print("=" * 60)
print("  Digite o caminho do arquivo MP3 para identificar.")
print("  Digite 'sair' para encerrar.\n")

while True:
    caminho = input("  Arquivo MP3: ").strip().strip('"').strip("'")
    if caminho.lower() in ["sair", "exit", "q", ""]:
        print("\n  Até mais! 🐦")
        break
    if not os.path.exists(caminho):
        print(f"  Arquivo não encontrado: {caminho}")
        continue
    try:
        identificar_canto(caminho)
    except Exception as e:
        print(f"  Erro ao processar: {e}")
    print()