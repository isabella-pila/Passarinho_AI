"""
Módulo SOM para o Django.
Treina o modelo na primeira execução e salva os pesos.
Nas próximas execuções, carrega os pesos salvos.
"""

import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

ESPECIES = ["araponga", "bem_te_vi", "urutau"]
NOMES_DISPLAY = {
    "araponga": "Araponga",
    "bem_te_vi": "Bem-te-vi",
    "urutau": "Urutau",
}
NOMES_CIENTIFICO = {
    "araponga": "Procnias nudicollis",
    "bem_te_vi": "Pitangus sulphuratus",
    "urutau": "Nyctibius griseus",
}
DESCRICOES = {
    "araponga": "Canto metálico e altíssimo, como um sino. Uma das aves mais barulhentas do mundo, encontrada na Mata Atlântica.",
    "bem_te_vi": "Canto onomatopaico inconfundível que dá nome à ave. Comum em áreas urbanas e rurais de todo o Brasil.",
    "urutau": "Canto noturno longo e melancólico, conhecido como 'lamento'. Ave noturna difícil de avistar.",
}
EMOJIS = {
    "araponga": "🔔",
    "bem_te_vi": "💛",
    "urutau": "🌙",
}
IMAGENS = {
    "araponga": "Araponga.jpg",
    "bem_te_vi": "BemTeVi.jpg",
    "urutau": "urutauE.jpg",
}

N_MFCC = 13
SR = 22050
SOM_X, SOM_Y = 5, 5
SOM_EPOCAS = 10000


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

    def treinar(self, X, epocas, sigma_i=None, sigma_f=0.5, lr_i=0.5, lr_f=0.01):
        if sigma_i is None:
            sigma_i = max(self.largura, self.altura) / 2
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
            if (epoca + 1) % 1000 == 0:
                print(f"    Época {epoca+1}/{epocas}")

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

    def vencedor(self, x):
        return self._encontrar_bmu(x)


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


# Modelo global (carregado uma vez)
_modelo = None


def get_modelo(dados_dir):
    """Retorna o modelo treinado. Treina na primeira vez, carrega depois."""
    global _modelo
    if _modelo is not None:
        return _modelo

    pesos_path = os.path.join(dados_dir, "som_pesos.npy")
    rotulos_path = os.path.join(dados_dir, "som_rotulos.npy")
    norm_path = os.path.join(dados_dir, "som_norm.npz")

    som = SOM(SOM_X, SOM_Y, N_MFCC * 2, seed=42)

    # Tenta carregar modelo salvo
    if os.path.exists(pesos_path) and os.path.exists(rotulos_path) and os.path.exists(norm_path):
        print("[SOM] Carregando modelo salvo...")
        som.pesos = np.load(pesos_path)
        rotulos_data = np.load(rotulos_path, allow_pickle=True).item()
        som.mapa_rotulos = rotulos_data
        norm = np.load(norm_path)
        _modelo = {
            "som": som,
            "X_min": norm["X_min"],
            "X_range": norm["X_range"],
        }
        print("[SOM] Modelo carregado!")
        return _modelo

    # Treina do zero
    print("[SOM] Treinando modelo pela primeira vez...")
    features, rotulos = [], []

    for especie in ESPECIES:
        pasta = os.path.join(dados_dir, especie)
        if not os.path.exists(pasta):
            print(f"  AVISO: Pasta {pasta} não encontrada!")
            continue
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

    # Treinar
    som.treinar(X_norm, SOM_EPOCAS)

    # Split treino/teste para medir acurácia
    np.random.seed(42)
    idx_treino, idx_teste = [], []
    for especie in ESPECIES:
        indices = np.where(y_rotulos == especie)[0].tolist()
        np.random.shuffle(indices)
        corte = max(1, int(len(indices) * 0.7))
        idx_treino.extend(indices[:corte])
        idx_teste.extend(indices[corte:])

    som.rotular_neuronios(X_norm[idx_treino], y_rotulos[idx_treino])

    # Calcula acurácia no teste
    acertos = 0
    for i in idx_teste:
        pred = som.classificar(X_norm[i])
        if pred == y_rotulos[i]:
            acertos += 1
    acuracia = acertos / len(idx_teste) * 100

    print(f"\n  ========================================")
    print(f"  ACURÁCIA: {acuracia:.1f}% ({acertos}/{len(idx_teste)})")
    print(f"  ========================================\n")

    # Agora rotula com TODOS os dados para a interface usar
    som.rotular_neuronios(X_norm, y_rotulos)

    # Salvar
    np.save(pesos_path, som.pesos)
    np.save(rotulos_path, som.mapa_rotulos)
    np.savez(norm_path, X_min=X_min, X_range=X_range)
    print("[SOM] Modelo treinado e salvo!")

    _modelo = {
        "som": som,
        "X_min": X_min,
        "X_range": X_range,
    }
    return _modelo


def identificar(caminho_audio, dados_dir):
    """Identifica a espécie de um arquivo de áudio."""
    modelo = get_modelo(dados_dir)
    som = modelo["som"]
    X_min = modelo["X_min"]
    X_range = modelo["X_range"]

    y = carregar_audio(caminho_audio)
    duracao = len(y) / SR
    mfcc = extrair_mfcc(y)
    mfcc_norm = (mfcc - X_min) / X_range

    bmu = som.vencedor(mfcc_norm)
    especie = som.classificar(mfcc_norm)
    dist_bmu = np.linalg.norm(mfcc_norm - som.pesos[bmu[0], bmu[1]])

    # Calcula similaridade com cada espécie
    similaridades = {}
    for esp in ESPECIES:
        dist = np.linalg.norm(mfcc_norm - np.mean(
            [som.pesos[i, j] for (i, j), r in som.mapa_rotulos.items() if r == esp],
            axis=0
        )) if any(r == esp for r in som.mapa_rotulos.values()) else 999
        similaridades[esp] = dist

    # Normaliza similaridades para porcentagem (invertida - menor distância = maior %)
    max_dist = max(similaridades.values())
    confiancas = {}
    total = 0
    for esp, dist in similaridades.items():
        score = max(0, max_dist - dist + 0.1)
        confiancas[esp] = score
        total += score
    for esp in confiancas:
        confiancas[esp] = (confiancas[esp] / total * 100) if total > 0 else 33.3

    return {
        "especie": especie or "desconhecido",
        "nome_display": NOMES_DISPLAY.get(especie, "Desconhecido"),
        "nome_cientifico": NOMES_CIENTIFICO.get(especie, ""),
        "descricao": DESCRICOES.get(especie, ""),
        "emoji": EMOJIS.get(especie, "🐦"),
        "imagem": IMAGENS.get(especie, ""),
        "duracao": round(duracao, 1),
        "neuronio": f"({bmu[0]}, {bmu[1]})",
        "distancia_bmu": round(dist_bmu, 4),
        "confiancas": {
            NOMES_DISPLAY[esp]: round(conf, 1)
            for esp, conf in confiancas.items()
        },
    }