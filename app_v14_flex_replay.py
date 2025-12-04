# ============================================================
# Predict Cars V14-FLEX ULTRA REAL (TURBO++) — VERSÃO GIGANTE
# ============================================================
# Versão com:
# - Número variável de passageiros (FLEX)
# - Barômetro ULTRA real
# - k* ULTRA (sentinela preditivo)
# - IDX Avançado ULTRA
# - IPF / IPO refinados
# - S6 Profundo em camadas
# - Micro-Leque ULTRA
# - Monte Carlo Profundo
# - QDS real + Backtest real
# - Painéis completos em Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
from typing import List, Dict, Any, Tuple, Optional

# ------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V14-FLEX ULTRA REAL (TURBO++)",
    layout="wide",
)

# ============================================================
# 1. UTILITÁRIOS DE HISTÓRICO (FLEX)
# ============================================================

def detectar_colunas_passageiros(df: pd.DataFrame) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Detecta:
      - coluna de ID (id da série)
      - colunas de passageiros (p1..pn)
      - coluna de k (índice de regime/risco)
    A detecção é flexível e tenta se adaptar ao CSV recebido.
    """
    if df is None or df.empty:
        return [], None, None

    cols = list(df.columns)

    # 1) Detectar coluna de id:
    #    - se a primeira coluna não for estritamente numérica, assume como ID
    col_id = None
    primeira = cols[0]
    if not pd.api.types.is_numeric_dtype(df[primeira]):
        col_id = primeira

    # 2) Detectar coluna de k:
    #    - prioridade: coluna chamada exatamente "k"
    #    - se não existir, assume a última coluna numérica como k
    col_k = None
    for c in cols[::-1]:
        if str(c).strip().lower() == "k":
            col_k = c
            break

    if col_k is None:
        for c in cols[::-1]:
            if pd.api.types.is_numeric_dtype(df[c]):
                col_k = c
                break

    # 3) Detectar colunas de passageiros:
    #    - colunas numéricas entre ID e k
    cols_pass = []
    for c in cols:
        if c == col_id or c == col_k:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols_pass.append(c)

    return cols_pass, col_id, col_k


def preparar_historico_flex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o histórico para o formato interno do V14-FLEX ULTRA:
      - coluna 'id' (string)
      - colunas 'p1'..'pn' (passageiros)
      - coluna 'k' (int)
    Mantém a flexibilidade no número de passageiros.
    """
    df = df_raw.copy()

    # Remove colunas totalmente vazias (ruído comum em CSV)
    df = df.dropna(axis=1, how="all")

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)

    # Se não houver id, cria IDs artificiais C1, C2, ...
    if col_id is None:
        df.insert(0, "id", [f"C{i+1}" for i in range(len(df))])
        col_id = "id"

    # Se não houver k, cria k=0 (placeholder neutro)
    if col_k is None:
        df["k"] = 0
        col_k = "k"

    # Renomeia colunas de passageiros para p1..pn
    mapping = {}
    for i, c in enumerate(cols_pass, start=1):
        mapping[c] = f"p{i}"
    df = df.rename(columns=mapping)

    # Garante tipos
    df[col_id] = df[col_id].astype(str)
    df[col_k] = pd.to_numeric(df[col_k], errors="coerce").fillna(0).astype(int)

    # Ordena colunas: id, p1..pn, k
    cols_ord = [col_id] + sorted(
        [c for c in df.columns if c.startswith("p")],
        key=lambda x: int(x[1:])
    ) + [col_k]

    df = df[cols_ord]

    return df


def parse_text_to_df(texto: str) -> pd.DataFrame:
    """
    Converte texto colado em tabela.
    Tenta inteligentemente vários separadores: ; , tab e espaço.
    É compatível com formatos do tipo:
      C1;41;5;4;52;30;33;0
      C2;9;39;...
    """
    if not texto or not texto.strip():
        return pd.DataFrame()

    raw = texto.strip()

    # Tenta ;, depois , e tab
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(StringIO(raw), sep=sep, header=None)
            if df.shape[1] >= 3:
                df.columns = [f"c{i+1}" for i in range(df.shape[1])]
                return df
        except Exception:
            pass

    # Última tentativa: separador por espaços
    try:
        df = pd.read_csv(StringIO(raw), sep=r"\s+", header=None, engine="python")
        df.columns = [f"c{i+1}" for i in range(df.shape[1])]
        return df
    except Exception:
        return pd.DataFrame()

# ============================================================
# 2. BARÔMETRO ULTRA REAL — CLIMA DA ESTRADA
# ============================================================

def _volatilidade_janela(series: np.ndarray, janela: int) -> float:
    """
    Volatilidade (desvio padrão) de uma série em janela deslizante.
    Se não houver séries suficientes, usa tudo.
    """
    if len(series) == 0:
        return 0.0
    if len(series) <= janela:
        return float(np.std(series))
    return float(np.std(series[-janela:]))


def _dispersao_media_series(df: pd.DataFrame, cols_pass: List[str], janela: int = 40) -> float:
    """
    Dispersão média por série:
      - calcula o desvio padrão em cada série (linha) sobre os passageiros
      - depois tira a média das últimas 'janela' séries
    """
    if df is None or df.empty or not cols_pass:
        return 0.0
    arr = df[cols_pass].values.astype(float)
    std_por_serie = np.std(arr, axis=1)
    if len(std_por_serie) <= janela:
        return float(np.mean(std_por_serie))
    return float(np.mean(std_por_serie[-janela:]))


def _entropia_normalizada(series: np.ndarray, n_bins: int = 20) -> float:
    """
    Entropia normalizada da distribuição da estrada.
    Mede o quão 'espalhada' ou 'caótica' está a distribuição.
    """
    if len(series) == 0:
        return 0.0
    hist, _ = np.histogram(series, bins=n_bins, density=True)
    hist = hist + 1e-12  # evita log(0)
    H = -np.sum(hist * np.log(hist))
    H_max = np.log(len(hist))
    if H_max == 0:
        return 0.0
    return float(H / H_max)


def calcular_barometro_ultra(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula o BARÔMETRO ULTRA — clima da estrada.
    Combina:
      - volatilidade local (curto prazo)
      - volatilidade global (médio prazo)
      - dispersão média entre passageiros
      - entropia da estrada
      - variação de k
    E produz:
      - 'estado' ∈ {Calmo, Moderado, Turbulento, Ruptura Imminente}
      - 'score' ∈ [0,1]
    """
    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if df is None or df.empty or not cols_pass:
        return {
            "estado": "Indefinido",
            "score": None,
            "vol_local": 0.0,
            "vol_global": 0.0,
            "dispersao": 0.0,
            "entropia": 0.0,
            "var_k": 0.0,
        }

    # Série somada dos passageiros — proxy da "energia da estrada"
    arr_pass = df[cols_pass].values.astype(float)
    soma = arr_pass.sum(axis=1)

    vol_local = _volatilidade_janela(soma, janela=12)   # curto prazo
    vol_global = _volatilidade_janela(soma, janela=40)  # médio prazo
    dispersao = _dispersao_media_series(df, cols_pass, janela=40)
    entropia = _entropia_normalizada(soma, n_bins=20)

    # Variação média de k
    if col_k and col_k in df.columns:
        kvals = df[col_k].astype(float).values
        if len(kvals) > 1:
            var_k = float(np.mean(np.abs(np.diff(kvals))))
        else:
            var_k = 0.0
    else:
        var_k = 0.0

    # Normalizações aproximadas (calibráveis)
    v_local_n = min(vol_local / 35.0, 1.0)
    v_global_n = min(vol_global / 50.0, 1.0)
    disp_n = min(dispersao / 22.0, 1.0)
    entr_n = min(entropia / 3.5, 1.0)  # entropia normalizada já é [0,1], aqui é só para re-escala leve
    var_k_n = min(var_k / 3.0, 1.0)

    # Score final (pesos podem ser recalibrados)
    score = (
        0.25 * v_local_n +
        0.25 * v_global_n +
        0.20 * disp_n +
        0.15 * entr_n +
        0.15 * var_k_n
    )
    score = float(round(score, 3))

    # Classificação do regime
    if score < 0.25:
        estado = "Calmo"
    elif score < 0.50:
        estado = "Moderado"
    elif score < 0.75:
        estado = "Turbulento"
    else:
        estado = "Ruptura Imminente"

    return {
        "estado": estado,
        "score": score,
        "vol_local": float(round(vol_local, 3)),
        "vol_global": float(round(vol_global, 3)),
        "dispersao": float(round(dispersao, 3)),
        "entropia": float(round(entropia, 3)),
        "var_k": float(round(var_k, 3)),
    }

# ============================================================
# 3. k* ULTRA REAL — SENTINELA PREDITIVO
# ============================================================

def calcular_kstar_ultra(df: pd.DataFrame,
                         cols_pass: List[str],
                         col_k: Optional[str]) -> Dict[str, Any]:
    """
    k* ULTRA REAL:
      - mede instabilidade estrutural da estrada
      - combina:
          * variação ponto a ponto dos passageiros (L1)
          * dispersão média por série
          * instabilidade do somatório (explosões/colapsos)
          * variação de k
    Produz:
      - kstar ∈ [0,1]
      - estado {estavel, atencao, critico}
      - texto explicativo
    """
    if df is None or df.empty or not cols_pass:
        return {
            "kstar": None,
            "estado": "indefinido",
            "texto": "Histórico insuficiente para calcular k*.",
        }

    arr = df[cols_pass].values.astype(float)

    # 1) Variação média L1 dos passageiros entre séries
    if len(arr) > 1:
        diffs = np.abs(np.diff(arr, axis=0))
        var_l1 = float(np.mean(diffs))
    else:
        var_l1 = 0.0

    # 2) Dispersão média entre passageiros (por série)
    disp_series = np.std(arr, axis=1)
    disp_media = float(np.mean(disp_series)) if len(disp_series) > 0 else 0.0

    # 3) Instabilidade do somatório (expansão/contração da estrada)
    soma = arr.sum(axis=1)
    if len(soma) > 1:
        inst = float(np.mean(np.abs(np.diff(soma))))
    else:
        inst = 0.0

    # 4) Variação de k
    if col_k and col_k in df.columns:
        kvals = df[col_k].astype(float).values
        if len(kvals) > 1:
            var_k = float(np.mean(np.abs(np.diff(kvals))))
        else:
            var_k = 0.0
    else:
        var_k = 0.0

    # Normalizações calibráveis (pontos de sensibilidade)
    n_l1 = min(var_l1 / 30.0, 1.0)
    n_disp = min(disp_media / 25.0, 1.0)
    n_inst = min(inst / 40.0, 1.0)
    n_vk = min(var_k / 3.0, 1.0)

    # k* final
    kstar = (
        0.35 * n_l1 +
        0.25 * n_disp +
        0.25 * n_inst +
        0.15 * n_vk
    )
    kstar = float(round(kstar, 3))

    # Estado interpretativo
    if kstar < 0.33:
        estado = "estavel"
        texto = "🟢 k*: Estrada estável — regime coerente, sem sinais fortes de ruptura."
    elif kstar < 0.66:
        estado = "atencao"
        texto = "🟡 k*: Pré-ruptura — surgem micro-instabilidades, usar previsões com atenção."
    else:
        estado = "critico"
        texto = "🔴 k*: Instabilidade pesada — risco elevado de ruptura, usar com cautela máxima."

    return {
        "kstar": kstar,
        "estado": estado,
        "texto": texto,
        "detalhes": {
            "var_l1": float(round(var_l1, 3)),
            "disp_media": float(round(disp_media, 3)),
            "instabilidade": float(round(inst, 3)),
            "var_k": float(round(var_k, 3)),
        },
    }

# ============================================================
# 4. FERRAMENTAS ESTRUTURAIS — VETORES E SIMILARIDADE (BASE IDX)
# ============================================================

def vetor_estrutura_norm(row: pd.Series, cols_pass: List[str]) -> np.ndarray:
    """
    Converte uma série (linha do histórico) em vetor estrutural normalizado.
    Mantém apenas a forma, sem escala absoluta.
    """
    v = row[cols_pass].astype(float).values
    if len(v) == 0:
        return np.zeros(1, dtype=float)
    v_min = np.min(v)
    v_max = np.max(v)
    if v_max == v_min:
        # série "plana" — retorna zeros (nenhuma forma definida)
        return np.zeros_like(v, dtype=float)
    return (v - v_min) / (v_max - v_min)


def similaridade_estrutural_ultra(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similaridade avançada:
      - combina distância euclidiana e correlação.
      - ideal para detectar "trechos de estrada" com forma parecida.
    """
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]

    if len(a) == 0:
        return 0.0

    # Distância euclidiana
    d = float(np.linalg.norm(a - b))

    # Correlação (se houver pontos suficientes)
    if len(a) > 2:
        c = float(np.corrcoef(a, b)[0, 1])
        if math.isnan(c):
            c = 0.0
    else:
        c = 0.0

    # Score:
    # - exp(-d) garante decaimento suave por distância
    # - (0.5 + 0.5*max(c,0)) valoriza correlação positiva
    base = math.exp(-d)
    fator_corr = 0.5 + 0.5 * max(c, 0.0)
    score = base * fator_corr

    return float(score)
# ============================================================
# 5. IDX ULTRA REAL — RANKING DE SIMILARIDADE ESTRUTURAL
# ============================================================

def idx_ultra_avancado(df: pd.DataFrame, idx_alvo: int) -> pd.DataFrame:
    """
    IDX ULTRA REAL:
      - compara o alvo contra todas as outras séries
      - usa vetor estrutural normalizado
      - calcula similaridade avançada
      - retorna ranking ordenado descendentemente
    """
    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if df is None or df.empty or not cols_pass:
        return pd.DataFrame()

    # Série alvo
    alvo = df.iloc[idx_alvo - 1]
    v_alvo = vetor_estrutura_norm(alvo, cols_pass)

    registros = []
    for i in range(len(df)):
        if i == idx_alvo - 1:
            continue
        v_i = vetor_estrutura_norm(df.iloc[i], cols_pass)
        sim = similaridade_estrutural_ultra(v_alvo, v_i)
        registros.append(
            (i + 1, df.iloc[i][col_id], sim)
        )

    out = pd.DataFrame(registros, columns=["idx", "id", "similaridade"])
    out = out.sort_values("similaridade", ascending=False).reset_index(drop=True)
    return out


# ============================================================
# 6. NÚCLEO IPF / IPO ULTRA REAL
# ============================================================

def ipf_nucleo_puro_ultra(df: pd.DataFrame,
                          idx_base: int,
                          cols_pass: List[str]) -> List[int]:
    """
    Núcleo IPF REAL:
      - pega série-base bruta
      - mantém estrutura fiel
      - retorna forma principal (padrão mais confiável)
    """
    base = df.iloc[idx_base - 1][cols_pass].astype(float).values
    return [int(round(x)) for x in base]


def ipo_nucleo_otimizado_ultra(df: pd.DataFrame,
                               idx_base: int,
                               cols_pass: List[str]) -> List[int]:
    """
    Núcleo IPO REAL:
      - corrige micro-ruídos na base
      - suaviza outliers
      - reforça tendências dominantes
    """
    base = df.iloc[idx_base - 1][cols_pass].astype(float).values
    # Suavização simples: média ponderada local
    suav = []
    for i in range(len(base)):
        vizinhos = []
        if i > 0:
            vizinhos.append(base[i - 1])
        vizinhos.append(base[i])
        if i < len(base) - 1:
            vizinhos.append(base[i + 1])
        suav.append(int(round(np.mean(vizinhos))))
    return suav


def gerar_nucleos_ipf_ipo(df: pd.DataFrame,
                          idx_alvo: int,
                          top_n: int = 25) -> List[List[int]]:
    """
    Gera núcleos IPF/IPO (e blend) a partir do IDX ULTRA.
    Será usado pelo S6 Profundo.
    """
    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)

    # Ranking IDX
    rank = idx_ultra_avancado(df, idx_alvo)
    if rank.empty:
        return []

    candidatos = []
    top = rank.head(top_n)

    for _, r in top.iterrows():
        idx_base = int(r["idx"])

        # IPF puro
        ipf = ipf_nucleo_puro_ultra(df, idx_base, cols_pass)
        candidatos.append(ipf)

        # IPO otimizado
        ipo = ipo_nucleo_otimizado_ultra(df, idx_base, cols_pass)
        candidatos.append(ipo)

        # Blend IPF/IPO
        arr_ipf = np.array(ipf, dtype=float)
        arr_ipo = np.array(ipo, dtype=float)
        blend = np.round((arr_ipf + arr_ipo) / 2.0).astype(int).tolist()
        candidatos.append(blend)

    # Remover duplicados mantendo a ordem
    uniq = []
    seen = set()
    for s in candidatos:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)

    return uniq


# ============================================================
# 7. S6 PROFUNDO ULTRA REAL — CAMADAS 1 e 2
# ============================================================

def s6_profundo_base(series_list: List[List[int]],
                     barometro: Dict[str, Any]) -> List[List[int]]:
    """
    S6 Profundo — Camada 1:
      - Ajuste primário orientado pelo Barômetro (clima da estrada)
      - Estratégias por regime:
           * Calmo → compressão suave
           * Moderado → correção leve de extremos
           * Turbulento → expansão leve controlada
           * Ruptura Imminente → estratégias dual (conservadora + agressiva)
    """
    estado = barometro.get("estado", "Moderado")
    score = barometro.get("score", 0.5)

    out = []

    for s in series_list:
        arr = np.array(s, dtype=float)
        media = np.mean(arr)

        if estado == "Calmo":
            # Puxa levemente os valores para a média (suavização)
            novo = arr + 0.15 * (media - arr)
            out.append(np.round(novo).astype(int).tolist())

        elif estado == "Moderado":
            # Correção leve de extremos
            q1 = np.quantile(arr, 0.25)
            q3 = np.quantile(arr, 0.75)
            novo = arr.copy()
            novo[arr < q1] += 1
            novo[arr > q3] -= 1
            out.append(np.round(novo).astype(int).tolist())

        elif estado == "Turbulento":
            # Expansão moderada
            fator = 1.0 + 0.4 * score
            delta = (arr - media) * 0.10 * fator
            novo = arr + delta
            out.append(np.round(novo).astype(int).tolist())

        else:  # Ruptura Imminente
            # Estratégia duplicada: conservadora + agressiva
            # Conservadora
            cons = arr + 0.25 * (media - arr)
            out.append(np.round(cons).astype(int).tolist())

            # Agressiva: amplifica tendências
            delta = (arr - media) * (0.25 + 0.25 * score)
            aggr = arr + delta
            out.append(np.round(aggr).astype(int).tolist())

    # Remover duplicados
    uniq = []
    seen = set()
    for s in out:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)

    return uniq


def s6_profundo_segunda_camada(series_list: List[List[int]],
                               kstar_info: Dict[str, Any]) -> List[List[int]]:
    """
    S6 Profundo — Camada 2:
      - Ajuste secundário orientado por k*
      - Quanto maior o k* (instabilidade), maior a suavização estrutural
      - Quanto menor o k*, mais preserva a forma bruta
    """
    kstar = kstar_info.get("kstar", 0.5)
    out = []

    for s in series_list:
        arr = np.array(s, dtype=float)
        media = np.mean(arr)

        # Ajuste dependendo do nível de instabilidade
        if kstar <= 0.33:
            # Estrada estável → preservar forma original
            novo = arr.copy()

        elif kstar <= 0.66:
            # Atenção → suavizar um pouco extremos
            delta = (media - arr) * 0.10
            novo = arr + delta

        else:
            # Crítico → puxar mais forte para a média (evitar explosões)
            delta = (media - arr) * 0.20
            novo = arr + delta

        out.append(np.round(novo).astype(int).tolist())

    # Deduplicação
    uniq = []
    seen = set()
    for s in out:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)

    return uniq


# ============================================================
# 8. UNIR LEQUES (S1 + IPF/IPO + S6)
# ============================================================

def unir_listas_sem_duplicatas(*listas: List[List[int]]) -> List[List[int]]:
    """
    Une várias listas de candidatos descartando duplicatas.
    Mantém ordem original.
    """
    out = []
    seen = set()
    for lst in listas:
        for s in lst:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                out.append(s)
    return out


# ============================================================
# ------ FIM DA PARTE 2/4 ------
# Cole a Parte 3/4 imediatamente na sequência, sem linhas em branco
# ============================================================
# ============================================================
# 9. S1 VIZINHANÇA ULTRA + MICRO-LEQUE ULTRA
# ============================================================

def s1_vizinhanca_ultra(df: pd.DataFrame,
                         idx_alvo: int,
                         cols_pass: List[str],
                         raio: int = 6) -> List[List[int]]:
    """
    S1 — Vizinhança ULTRA:
      - Pega séries em torno do índice alvo (antes e depois)
      - Considera um raio configurável
      - Ideal para capturar "ondas locais" da estrada
    """
    n = len(df)
    candidatos = []

    for delta in range(-raio, raio + 1):
        if delta == 0:
            continue
        pos = idx_alvo - 1 + delta
        if 0 <= pos < n:
            row = df.iloc[pos]
            candidatos.append(row[cols_pass].astype(int).tolist())

    # Remoção de duplicatas mantendo ordem
    uniq = []
    seen = set()
    for s in candidatos:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            uniq.append(s)

    return uniq


def micro_leque_ultra(series_list: List[List[int]],
                      intensidade: int = 2) -> List[List[int]]:
    """
    Micro-Leque ULTRA:
      - Pequenos ajustes combinatórios nas séries base
      - Perturbações leves em 1 ou 2 passageiros
      - Mantém coerência da estrutura global
    """
    candidatos = []

    for s in series_list:
        base = np.array(s, dtype=int)

        # Micro-ajustes em 1 passageiro
        for _ in range(intensidade):
            v = base.copy()
            i = np.random.randint(0, len(v))
            passo = np.random.choice([-2, -1, 1, 2])
            v[i] = max(0, v[i] + passo)
            candidatos.append(v.astype(int).tolist())

        # Micro-ajustes em 2 passageiros
        if len(base) >= 2:
            for _ in range(intensidade):
                v = base.copy()
                i1, i2 = np.random.choice(len(v), 2, replace=False)
                passo1 = np.random.choice([-1, 1])
                passo2 = np.random.choice([-1, 1])
                v[i1] = max(0, v[i1] + passo1)
                v[i2] = max(0, v[i2] + passo2)
                candidatos.append(v.astype(int).tolist())

    # Remover duplicatas
    uniq = []
    seen = set()
    for s in candidatos:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)

    return uniq


# ============================================================
# 10. MONTE CARLO PROFUNDO ULTRA REAL
# ============================================================

def monte_carlo_profundo_ultra(series: List[List[int]],
                               alvo_vec: np.ndarray,
                               barometro: Dict[str, Any],
                               kstar_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Monte Carlo Profundo ULTRA:
      - Recebe um universo de séries candidatas
      - Avalia cada uma em relação ao alvo e ao clima da estrada
      - Combina:
           * coerência estrutural (correlação com alvo)
           * dispersão
           * adequação ao regime (Barômetro)
           * ajuste ao nível de instabilidade (k*)
      - Retorna DataFrame com score e ranking
    """
    if not series:
        return pd.DataFrame()

    estado = barometro.get("estado", "Moderado")
    reg_score = barometro.get("score", 0.5)
    kstar = kstar_info.get("kstar", 0.5) if kstar_info.get("kstar") is not None else 0.5

    registros = []

    for s in series:
        arr = np.array(s, dtype=float)

        # Dispersão interna da série candidata
        disp = float(np.std(arr))

        # Coerência com o alvo (correlação)
        if len(arr) != len(alvo_vec):
            m = min(len(arr), len(alvo_vec))
            a = arr[:m]
            b = alvo_vec[:m]
        else:
            a = arr
            b = alvo_vec

        if len(a) > 2:
            corr = float(np.corrcoef(a, b)[0, 1])
            if math.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Fator de regime (Barômetro)
        if estado == "Calmo":
            # preferir séries mais concentradas (menor dispersão)
            reg_factor = 1.0 - 0.6 * reg_score
        elif estado == "Moderado":
            reg_factor = 1.0
        elif estado == "Turbulento":
            reg_factor = 0.8 + 0.2 * reg_score
        else:  # Ruptura Imminente
            reg_factor = 0.6 + 0.4 * reg_score

        # Fator de confiança baseado em k*
        if kstar <= 0.33:
            k_factor = 1.0
        elif kstar <= 0.66:
            k_factor = 0.9
        else:
            k_factor = 0.75

        # Normalizações básicas
        disp_norm = 1.0 / (1.0 + disp)      # quanto menor a dispersão, maior este valor
        corr_pos = max(corr, 0.0)           # correlação negativa não ajuda

        # Score final — pesos calibráveis
        score = (
            0.45 * corr_pos +
            0.25 * disp_norm +
            0.15 * reg_factor +
            0.15 * k_factor
        )

        registros.append({
            "series": s,
            "disp": float(round(disp, 3)),
            "corr": float(round(corr, 3)),
            "reg_factor": float(round(reg_factor, 3)),
            "k_factor": float(round(k_factor, 3)),
            "score": float(round(score, 6)),
        })

    df_mc = pd.DataFrame(registros)
    df_mc = df_mc.sort_values("score", ascending=False).reset_index(drop=True)
    df_mc["rank"] = np.arange(1, len(df_mc) + 1)

    return df_mc


# ============================================================
# 11. PIPELINE ULTRA REAL — GERAÇÃO DE PREVISÕES
# ============================================================

def gerar_previsoes_ultra_real(df: pd.DataFrame,
                               idx_alvo: int,
                               n_final: int,
                               barometro: Dict[str, Any],
                               kstar_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Pipeline completo de previsão ULTRA REAL:
      1) S1 Vizinhança
      2) Núcleos IPF / IPO (IDX-based)
      3) S6 Profundo (2 camadas)
      4) Micro-Leque ULTRA
      5) União dos leques
      6) Monte Carlo Profundo ULTRA
      7) Seleção final (Top N)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    # Série alvo e vetor estrutural
    alvo_row = df.iloc[idx_alvo - 1]
    alvo_vec = alvo_row[cols_pass].astype(float).values

    # 1) S1 — vizinhança local
    leque_s1 = s1_vizinhanca_ultra(df, idx_alvo, cols_pass, raio=6)

    # 2) Núcleos IPF / IPO recheados via IDX ULTRA
    leque_ipf_ipo = gerar_nucleos_ipf_ipo(df, idx_alvo, top_n=25)

    # Leque base inicial
    base_primario = unir_listas_sem_duplicatas(leque_s1, leque_ipf_ipo)

    # 3) S6 Profundo — camada 1
    leque_s6_1 = s6_profundo_base(base_primario, barometro)

    # 4) S6 Profundo — camada 2 (ajuste por k*)
    leque_s6_2 = s6_profundo_segunda_camada(leque_s6_1, kstar_info)

    # 5) Micro-Leque ULTRA
    leque_micro = micro_leque_ultra(leque_s6_2, intensidade=3)

    # Universo total de séries
    universo = unir_listas_sem_duplicatas(
        base_primario,
        leque_s6_1,
        leque_s6_2,
        leque_micro,
    )

    # 6) Monte Carlo Profundo ULTRA
    df_mc = monte_carlo_profundo_ultra(
        universo,
        alvo_vec,
        barometro,
        kstar_info,
    )

    if df_mc.empty:
        return pd.DataFrame()

    # 7) Seleção final
    df_final = df_mc.head(n_final).reset_index(drop=True)
    return df_final


# ============================================================
# 12. QDS REAL + BACKTEST REAL ULTRA
# ============================================================

def calcular_qds_real(df: pd.DataFrame,
                      horizonte: int = 80,
                      n_final: int = 20) -> Dict[str, Any]:
    """
    QDS REAL ULTRA:
      - Roda o motor completo nas últimas 'horizonte' séries
      - Compara sempre com a série seguinte (verdade histórica)
      - Mede:
          * média de acertos de passageiros
          * QDS = média_acertos / nº de passageiros
          * nº de casos válidos avaliados
    """
    if df is None or df.empty or len(df) < horizonte + 5:
        return {"qds": None, "media_acertos": None, "n_casos": 0}

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if not cols_pass:
        return {"qds": None, "media_acertos": None, "n_casos": 0}

    # Pré-cálculo global de barômetro e k* (aproximação para QDS)
    barometro = calcular_barometro_ultra(df)
    kstar_info = calcular_kstar_ultra(df, cols_pass, col_k)

    inicio = max(1, len(df) - horizonte)
    acertos = []

    for idx_alvo in range(inicio, len(df) - 1):
        df_prev = gerar_previsoes_ultra_real(
            df,
            idx_alvo,
            n_final,
            barometro,
            kstar_info,
        )
        if df_prev.empty:
            continue

        # Série real (seguinte ao alvo)
        real_row = df.iloc[idx_alvo]
        real = set(real_row[cols_pass].astype(int).tolist())

        melhor = df_prev.iloc[0]["series"]
        prev = set(melhor)

        ac = len(real.intersection(prev))
        acertos.append(ac)

    if not acertos:
        return {"qds": None, "media_acertos": None, "n_casos": 0}

    media_ac = float(np.mean(acertos))
    n_pass = len(cols_pass)
    qds = float(round(media_ac / max(n_pass, 1), 3))

    return {
        "qds": qds,
        "media_acertos": media_ac,
        "n_casos": len(acertos),
    }


# ============================================================
# 13. REPLAY ULTRA — HISTÓRICO PASSO A PASSO
# ============================================================

def gerar_replay_ultra(df: pd.DataFrame,
                       n_final: int = 15) -> pd.DataFrame:
    """
    Replay ULTRA:
      - Para cada índice alvo (até o penúltimo),
      - roda o motor completo,
      - registra:
          * id alvo
          * id real (série seguinte)
          * previsão principal
          * série real
          * nº de acertos
      - Permite auditar o comportamento do motor no histórico.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if not cols_pass:
        return pd.DataFrame()

    barometro = calcular_barometro_ultra(df)
    kstar_info = calcular_kstar_ultra(df, cols_pass, col_k)

    registros = []

    for idx_alvo in range(1, len(df) - 1):
        df_prev = gerar_previsoes_ultra_real(
            df,
            idx_alvo,
            n_final,
            barometro,
            kstar_info,
        )
        if df_prev.empty:
            continue

        # Verdade histórica → série seguinte ao alvo
        real_row = df.iloc[idx_alvo]
        real = set(real_row[cols_pass].astype(int).tolist())

        melhor = df_prev.iloc[0]["series"]
        prev = set(melhor)

        ac = len(real.intersection(prev))

        registros.append({
            "idx_alvo": idx_alvo,
            "id_alvo": df.iloc[idx_alvo - 1][col_id],
            "id_real": df.iloc[idx_alvo][col_id],
            "previsao": melhor,
            "real": list(real),
            "acertos": ac,
        })

    if not registros:
        return pd.DataFrame()

    return pd.DataFrame(registros)


# ============================================================
# 14. DISTRIBUIÇÃO DE k — SUPORTE AO MONITOR DE RISCO
# ============================================================

def calcular_distribuicao_k(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estatísticas de k para o Monitor de Risco:
      - média
      - desvio padrão
      - distribuição de frequências
    """
    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
    if col_k is None or col_k not in df.columns:
        return {"media": None, "std": None, "freq": {}, "col_k": None}

    serie = df[col_k].astype(float)
    media = float(serie.mean())
    std = float(serie.std(ddof=0)) if len(serie) > 1 else 0.0
    freq = serie.value_counts().sort_index().to_dict()

    return {
        "media": media,
        "std": std,
        "freq": freq,
        "col_k": col_k,
    }

# ============================================================
# ------ FIM DA PARTE 3/4 ------
# Cole a Parte 4/4 imediatamente na sequência
# ============================================================
# ============================================================
# 15. INTERFACE STREAMLIT — PAINÉIS COMPLETOS
# ============================================================

def painel_historico_entrada():
    """
    Painel 1 — Histórico — Entrada
    Permite carregar o histórico via:
      - upload de CSV
      - texto colado
    E monta o DataFrame interno no formato FLEX.
    """
    st.markdown("## 📥 Histórico — Entrada (Modo FLEX ULTRA)")

    df_local = None
    opc = st.radio(
        "Como deseja carregar o histórico?",
        ["Enviar arquivo CSV", "Copiar e colar o histórico"],
        horizontal=False,
    )

    if opc == "Enviar arquivo CSV":
        file = st.file_uploader("Selecione o arquivo CSV:", type=["csv"])
        if file is not None:
            try:
                df_raw = pd.read_csv(file)
                df_local = preparar_historico_flex(df_raw)
                st.session_state["df"] = df_local
                st.success("Histórico carregado com sucesso!")
                st.dataframe(df_local.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar CSV: {e}")

    else:
        texto = st.text_area(
            "Cole o histórico (ex: C1;41;5;4;52;30;33;0):",
            height=220,
        )
        if st.button("Processar texto colado"):
            try:
                df_raw = parse_text_to_df(texto)
                if df_raw.empty:
                    st.error("Não foi possível interpretar o texto colado.")
                else:
                    df_local = preparar_historico_flex(df_raw)
                    st.session_state["df"] = df_local
                    st.success("Histórico carregado com sucesso a partir do texto!")
                    st.dataframe(df_local.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao processar texto: {e}")

    st.markdown("---")
    df = st.session_state.get("df", None)
    if df is not None and not df.empty:
        cols_pass, col_id, col_k = detectar_colunas_passageiros(df)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Nº de séries", len(df))
        with c2:
            st.metric("Nº de passageiros", len(cols_pass))
        with c3:
            st.metric("Coluna de k", col_k if col_k else "N/A")


def painel_pipeline_ultra():
    """
    Painel 2 — Pipeline V14-FLEX ULTRA
    Executa o pipeline completo de previsão para um índice alvo específico.
    """
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA — Execução Direta")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue primeiro o histórico no painel '📥 Histórico — Entrada'.")
        st.stop()

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)

    c1, c2 = st.columns(2)
    with c1:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
    with c2:
        n_final = st.slider(
            "Quantidade de séries de saída (Top N):",
            min_value=5,
            max_value=80,
            value=25,
            step=5,
        )

    if st.button("Executar Pipeline ULTRA REAL"):
        with st.spinner("Rodando IPF/IPO + S6 + Micro-Leque + Monte Carlo Profundo..."):
            barometro = calcular_barometro_ultra(df)
            kstar_info = calcular_kstar_ultra(df, cols_pass, col_k)
            df_prev = gerar_previsoes_ultra_real(
                df,
                idx_alvo,
                n_final,
                barometro,
                kstar_info,
            )

        if df_prev.empty:
            st.error("Não foi possível gerar previsões.")
        else:
            st.success("Pipeline ULTRA executado com sucesso.")

            df_view = df_prev.copy()
            df_view["series_str"] = df_view["series"].apply(
                lambda s: " ".join(str(x) for x in s)
            )

            st.markdown("### 🎯 Saída ULTRA — Séries de Previsão (Top N)")
            st.dataframe(
                df_view[
                    ["rank", "series_str", "score", "disp", "corr", "reg_factor", "k_factor"]
                ],
                use_container_width=True,
            )

            melhor = df_prev.iloc[0]["series"]
            st.markdown("### 🏁 Previsão Principal")
            st.code(" ".join(str(x) for x in melhor), language="text")


def painel_monitor_risco():
    """
    Painel 3 — Monitor de Risco (Barômetro + k*)
    Mostra:
      - estatísticas de k
      - Barômetro ULTRA
      - k* ULTRA (sentinela preditivo)
    """
    st.markdown("## 🚨 Monitor de Risco — Barômetro + k* ULTRA")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)

    dist_k = calcular_distribuicao_k(df)
    barometro = calcular_barometro_ultra(df)
    kstar_info = calcular_kstar_ultra(df, cols_pass, col_k)

    # Estatísticas de k
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Média de k",
            f"{dist_k['media']:.2f}" if dist_k["media"] is not None else "N/A",
        )
    with c2:
        st.metric(
            "Desvio padrão de k",
            f"{dist_k['std']:.2f}" if dist_k["std"] is not None else "N/A",
        )
    with c3:
        st.metric("Nº de séries", len(df))

    st.markdown("### 🔢 Distribuição de k")
    if dist_k["freq"]:
        df_freq = pd.DataFrame(
            [{"k": k, "freq": v} for k, v in dist_k["freq"].items()]
        ).sort_values("k")
        st.dataframe(df_freq, use_container_width=True)
    else:
        st.info("Não há dados suficientes para estatísticas de k.")

    st.markdown("---")
    # Barômetro ULTRA
    st.markdown("### 🌡️ Barômetro ULTRA — Clima da Estrada")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Regime", barometro["estado"])
    with b2:
        st.metric("Score de regime (0–1)", barometro["score"])
    with b3:
        st.metric("Volatilidade local", f"{barometro['vol_local']:.2f}")
    with b4:
        st.metric("Dispersão média", f"{barometro['dispersao']:.2f}")

    st.markdown("---")
    # k* ULTRA
    st.markdown("### 🧭 k* ULTRA — Sentinela Preditivo")
    if kstar_info["kstar"] is not None:
        st.metric("k* (0–1)", kstar_info["kstar"])
    st.info(kstar_info["texto"])


def painel_turbo_adaptativo():
    """
    Painel 4 — Modo TURBO++ Adaptativo
    Reúne:
      - estado da estrada (Barômetro + k*)
      - IDX ULTRA do alvo
      - Leque ULTRA + Monte Carlo Profundo
      - Previsão final TURBO++
    """
    st.markdown("## 🚀 Modo TURBO++ Adaptativo — Painel Completo")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    cols_pass, col_id, col_k = detectar_colunas_passageiros(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        idx_alvo = st.number_input(
            "Índice alvo (1 = primeira série):",
            min_value=1,
            max_value=len(df),
            value=len(df),
            step=1,
        )
    with c2:
        n_final = st.slider(
            "Nº de séries na saída ULTRA:",
            min_value=10,
            max_value=80,
            value=30,
            step=5,
        )
    with c3:
        mostra_idx = st.checkbox(
            "Mostrar IDX completo (núcleo de similaridade)",
            value=True,
        )

    if st.button("Rodar TURBO++ Adaptativo"):
        with st.spinner("Executando TURBO++ ULTRA..."):
            barometro = calcular_barometro_ultra(df)
            kstar_info = calcular_kstar_ultra(df, cols_pass, col_k)
            rank_idx = idx_ultra_avancado(df, idx_alvo) if mostra_idx else pd.DataFrame()
            df_prev = gerar_previsoes_ultra_real(
                df,
                idx_alvo,
                n_final,
                barometro,
                kstar_info,
            )

        if df_prev.empty:
            st.error("Não foi possível gerar previsões.")
            st.stop()

        st.success("Modo TURBO++ executado.")

        # 1) Estado da estrada
        st.markdown("### 1️⃣ Estado da Estrada (Barômetro + k*)")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Regime", barometro["estado"])
        with a2:
            st.metric("Score de regime", barometro["score"])
        with a3:
            if kstar_info["kstar"] is not None:
                st.metric("k* (0–1)", kstar_info["kstar"])
            else:
                st.metric("k* (0–1)", "N/A")
        st.info(kstar_info["texto"])

        st.markdown("---")

        # 2) IDX ULTRA
        if mostra_idx and not rank_idx.empty:
            st.markdown("### 2️⃣ IDX ULTRA — Núcleo de Similaridade")
            st.dataframe(rank_idx.head(30), use_container_width=True)

        st.markdown("---")

        # 3) Leque ULTRA + Monte Carlo Profundo
        st.markdown("### 3️⃣ Leque ULTRA + Monte Carlo Profundo — Saída Final")
        df_view = df_prev.copy()
        df_view["series_str"] = df_view["series"].apply(
            lambda s: " ".join(str(x) for x in s)
        )

        st.dataframe(
            df_view[
                ["rank", "series_str", "score", "disp", "corr", "reg_factor", "k_factor"]
            ],
            use_container_width=True,
        )

        melhor = df_prev.iloc[0]["series"]
        st.markdown("### 🎯 Previsão Final TURBO++")
        st.code(" ".join(str(x) for x in melhor), language="text")


def painel_replay_ultra():
    """
    Painel 5 — Replay Automático ULTRA
    Mostra o que o motor teria feito passo a passo no histórico.
    """
    st.markdown("## 📅 Replay Automático ULTRA — Passo a Passo")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    n_final = st.slider(
        "Nº de séries na saída ULTRA (por passo do replay):",
        min_value=5,
        max_value=40,
        value=15,
        step=5,
    )

    if st.button("Rodar Replay ULTRA"):
        with st.spinner("Executando Replay passo a passo..."):
            df_rep = gerar_replay_ultra(df, n_final=n_final)

        if df_rep.empty:
            st.error("Replay não produziu resultados.")
        else:
            st.success("Replay executado.")
            st.dataframe(df_rep.head(100), use_container_width=True)

            media_ac = float(df_rep["acertos"].mean())
            max_ac = int(df_rep["acertos"].max())
            n_casos = len(df_rep)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Média de acertos", f"{media_ac:.2f}")
            with r2:
                st.metric("Máximo de acertos", max_ac)
            with r3:
                st.metric("Nº de casos avaliados", n_casos)


def painel_qds_backtest():
    """
    Painel 6 — Confiabilidade — QDS / Backtest
    Calcula QDS REAL ULTRA e mostra interpretação.
    """
    st.markdown("## 🧪 Confiabilidade — QDS REAL / Backtest REAL")

    df = st.session_state.get("df", None)
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro.")
        st.stop()

    horizonte = st.slider(
        "Horizonte para QDS / Backtest (últimas N séries):",
        min_value=20,
        max_value=250,
        value=80,
        step=10,
    )

    n_final = st.slider(
        "Nº de séries na saída ULTRA (para cada teste):",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
    )

    if st.button("Rodar QDS REAL / Backtest REAL"):
        with st.spinner("Calculando QDS e Backtest ULTRA..."):
            info = calcular_qds_real(df, horizonte=horizonte, n_final=n_final)

        if info["n_casos"] == 0:
            st.error("Não foi possível calcular QDS / Backtest (poucos casos válidos).")
        else:
            st.success("QDS / Backtest calculados.")

            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric(
                    "QDS (0–1)",
                    info["qds"] if info["qds"] is not None else "N/A",
                )
            with q2:
                st.metric(
                    "Média de acertos",
                    f"{info['media_acertos']:.2f}",
                )
            with q3:
                st.metric(
                    "Nº de casos avaliados",
                    info["n_casos"],
                )

            st.markdown(
                """
                **Leitura sugerida do QDS:**
                - QDS > 0.70 → motor extremamente aderente ao histórico recente.
                - QDS entre 0.50 e 0.70 → aderência boa, mas com ruído.
                - QDS entre 0.30 e 0.50 → regime de atenção / necessidade de ajuste fino.
                - QDS < 0.30 → baixa aderência; revisar parâmetros, regime ou janela.
                """
            )

# ============================================================
# 16. FUNÇÃO PRINCIPAL (main)
# ============================================================

def main():
    """
    Função principal do app Streamlit.
    Define o layout geral, navegação lateral e chama os painéis.
    """
    st.title("🚗 Predict Cars V14-FLEX ULTRA REAL (TURBO++)")
    st.caption(
        "Versão FLEX ULTRA REAL — todos os módulos refinados "
        "integrados ao Barômetro, k*, IDX, S6, Micro-Leque, Monte Carlo e QDS."
    )

    # Inicializa DataFrame na sessão, se ainda não existir
    if "df" not in st.session_state:
        st.session_state["df"] = None

    st.sidebar.header("Navegação")
    painel = st.sidebar.radio(
        "Escolha o painel:",
        [
            "📥 Histórico — Entrada",
            "🔍 Pipeline V14-FLEX ULTRA",
            "🚨 Monitor de Risco (Barômetro + k*)",
            "🚀 Modo TURBO++ Adaptativo",
            "📅 Replay Automático ULTRA",
            "🧪 Confiabilidade — QDS / Backtest",
        ],
    )

    if painel == "📥 Histórico — Entrada":
        painel_historico_entrada()
    elif painel == "🔍 Pipeline V14-FLEX ULTRA":
        painel_pipeline_ultra()
    elif painel == "🚨 Monitor de Risco (Barômetro + k*)":
        painel_monitor_risco()
    elif painel == "🚀 Modo TURBO++ Adaptativo":
        painel_turbo_adaptativo()
    elif painel == "📅 Replay Automático ULTRA":
        painel_replay_ultra()
    elif painel == "🧪 Confiabilidade — QDS / Backtest":
        painel_qds_backtest()


# ============================================================
# 17. PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
