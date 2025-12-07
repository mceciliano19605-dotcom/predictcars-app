# ============================================================
# Predict Cars V15-HÍBRIDO ULTRA
# Versão FLEX ULTRA + Replay + Monitor de Risco + Ruído
# + Testes de Confiabilidade REAL + TURBO++ ULTRA ANTI-RUÍDO
# + Painel 📄 Relatório Final V15-HÍBRIDO
#
# Arquivo: app_v15_hibrido_ultra.py
# ============================================================

import io
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Configuração básica da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V15-HÍBRIDO",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Helpers de sessão
# ------------------------------------------------------------
def get_df() -> Optional[pd.DataFrame]:
    return st.session_state.get("df", None)


def set_df(df: pd.DataFrame):
    st.session_state["df"] = df


def init_session_state():
    defaults = {
        "df": None,
        "ultima_previsao": None,
        "ultima_analise_risco": None,
        "ultima_analise_ruido": None,
        "ultima_confiabilidade": None,
        "ultima_previsao_turbo": None,
        "ultimo_relatorio_final": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ------------------------------------------------------------
# Parsing do histórico FLEX ULTRA
#
# Regras:
# - Última coluna = k (guardas que acertaram exatamente o carro)
# - Colunas intermediárias = passageiros (n1..nN)
# - Opcionalmente a primeira coluna pode ser "rótulo da série" (C1, C2...)
#   -> neste caso, guardamos em 'serie_id'
# ------------------------------------------------------------
def detectar_delimitador(sample: str) -> str:
    if ";" in sample and "," in sample:
        if sample.count(";") > sample.count(","):
            return ";"
        else:
            return ","
    if ";" in sample:
        return ";"
    return ","


def carregar_historico_arquivo(
    uploaded_file, formato_linhas: str
) -> pd.DataFrame:
    raw_bytes = uploaded_file.read()
    text_sample = raw_bytes[:2000].decode("utf-8", errors="ignore")
    sep = detectar_delimitador(text_sample)

    df_raw = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=sep,
        header=None,
        engine="python",
    )

    if df_raw.shape[1] < 2:
        raise ValueError("Histórico inválido: é esperado pelo menos 2 colunas (passageiros + k).")

    if formato_linhas == "Série na primeira coluna + passageiros + k":
        if df_raw.shape[1] < 3:
            raise ValueError(
                "Histórico inválido para esse formato: é esperado pelo menos 3 colunas."
            )
        serie_ids = df_raw.iloc[:, 0].astype(str)
        passageiros = df_raw.iloc[:, 1:-1].astype(int)
        k_col = df_raw.iloc[:, -1].astype(int)
    else:
        serie_ids = pd.Series(
            [f"C{i+1}" for i in range(len(df_raw))],
            index=df_raw.index,
        )
        passageiros = df_raw.iloc[:, :-1].astype(int)
        k_col = df_raw.iloc[:, -1].astype(int)

    num_passageiros = passageiros.shape[1]
    col_passageiros = [f"n{i+1}" for i in range(num_passageiros)]
    df = pd.DataFrame(passageiros.values, columns=col_passageiros)
    df["k"] = k_col.values
    df["serie_id"] = serie_ids.values
    df["idx"] = np.arange(1, len(df) + 1)

    cols = ["idx", "serie_id"] + col_passageiros + ["k"]
    df = df[cols]

    return df


def carregar_historico_texto(
    texto: str,
    formato_linhas: str,
) -> pd.DataFrame:
    texto = texto.strip()
    if not texto:
        raise ValueError("Texto vazio para histórico.")

    sample = texto.splitlines()[0]
    sep = detectar_delimitador(sample)

    df_raw = pd.read_csv(
        io.StringIO(texto),
        sep=sep,
        header=None,
        engine="python",
    )

    if df_raw.shape[1] < 2:
        raise ValueError("Histórico inválido: é esperado pelo menos 2 colunas (passageiros + k).")

    if formato_linhas == "Série na primeira coluna + passageiros + k":
        if df_raw.shape[1] < 3:
            raise ValueError(
                "Histórico inválido para esse formato: é esperado pelo menos 3 colunas."
            )
        serie_ids = df_raw.iloc[:, 0].astype(str)
        passageiros = df_raw.iloc[:, 1:-1].astype(int)
        k_col = df_raw.iloc[:, -1].astype(int)
    else:
        serie_ids = pd.Series(
            [f"C{i+1}" for i in range(len(df_raw))],
            index=df_raw.index,
        )
        passageiros = df_raw.iloc[:, :-1].astype(int)
        k_col = df_raw.iloc[:, -1].astype(int)

    num_passageiros = passageiros.shape[1]
    col_passageiros = [f"n{i+1}" for i in range(num_passageiros)]
    df = pd.DataFrame(passageiros.values, columns=col_passageiros)
    df["k"] = k_col.values
    df["serie_id"] = serie_ids.values
    df["idx"] = np.arange(1, len(df) + 1)

    cols = ["idx", "serie_id"] + col_passageiros + ["k"]
    df = df[cols]

    return df


def obter_num_passageiros(df: pd.DataFrame) -> int:
    cols = [c for c in df.columns if c.startswith("n")]
    return len(cols)


def obter_colunas_passageiros(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("n")]

# ------------------------------------------------------------
# Métricas centrais: k, k*, regimes, barômetro e ruído
# ------------------------------------------------------------
def calcular_k_star(df: pd.DataFrame, janela: int = 50) -> pd.Series:
    if "k" not in df.columns:
        raise ValueError("DataFrame sem coluna 'k'.")

    k_values = df["k"].astype(float)
    rolling_mean = k_values.rolling(janela, min_periods=1).mean()
    rolling_max = k_values.rolling(janela, min_periods=1).max()

    eps = 1e-9
    norm = (rolling_mean + 0.5 * rolling_max) / (rolling_max + eps)
    k_star = norm.clip(0.0, 1.0)
    return k_star


def classificar_regime_k_star(valor: float) -> str:
    if valor < 0.25:
        return "🟢 Estrada estável — regime tranquilo."
    elif valor < 0.5:
        return "🟡 Estrada em leve tensão — monitorar."
    elif valor < 0.75:
        return "🟠 Estrada turbulenta — risco elevado."
    else:
        return "🔴 Ruptura / turbulência pesada — extrema cautela."


def calcular_barometro_global(df: pd.DataFrame) -> Dict[str, float]:
    k = df["k"].astype(float)
    k_mean = float(k.mean())
    k_std = float(k.std(ddof=1)) if len(k) > 1 else 0.0
    k_max = float(k.max()) if len(k) > 0 else 0.0

    k_star_series = calcular_k_star(df)
    k_star_mean = float(k_star_series.mean())

    return {
        "k_mean": k_mean,
        "k_std": k_std,
        "k_max": k_max,
        "k_star_mean": k_star_mean,
    }


def classificar_barometro_global(metrics: Dict[str, float]) -> str:
    k_star_mean = metrics.get("k_star_mean", 0.0)
    return classificar_regime_k_star(k_star_mean)


def estimar_ruido_condicional(
    df: pd.DataFrame,
    janela: int = 50,
) -> pd.Series:
    """
    Estimação de ruído condicional (NR%) por janela.
    Heurística:
    - Grandes oscilações de k numa janela indicam ruído.
    - Normaliza a variância local em 0..1.
    """
    k = df["k"].astype(float)
    rolling_var = k.rolling(janela, min_periods=5).var(ddof=1)
    rolling_var = rolling_var.fillna(0.0)

    if rolling_var.max() <= 0:
        nr = pd.Series(0.0, index=df.index)
    else:
        nr = (rolling_var / (rolling_var.max() + 1e-9)).clip(0.0, 1.0)

    return nr


def classificar_nivel_ruido(nr_val: float) -> str:
    if nr_val < 0.2:
        return "🟢 Baixo ruído estrutural."
    elif nr_val < 0.4:
        return "🟡 Ruído moderado."
    elif nr_val < 0.7:
        return "🟠 Ruído elevado."
    else:
        return "🔴 Ruído estrutural muito alto."
# ============================================================
# Motor de Previsões V15-HÍBRIDO
# - S6 Profundo
# - Micro-Leque
# - Monte Carlo Profundo
# - Combinação adaptativa por k* e NR%
# ============================================================

def _extrair_ultimas_series(
    df: pd.DataFrame,
    idx_alvo: int,
    janela_contexto: int,
) -> pd.DataFrame:
    if "idx" not in df.columns:
        raise ValueError("DataFrame sem coluna 'idx'.")

    max_idx = int(df["idx"].max())
    if idx_alvo < 2 or idx_alvo > max_idx + 1:
        raise ValueError(
            f"Índice alvo inválido: {idx_alvo}. Deve ser entre 2 e {max_idx + 1}."
        )

    inicio = max(1, idx_alvo - janela_contexto)
    fim = idx_alvo - 1

    mask = (df["idx"] >= inicio) & (df["idx"] <= fim)
    return df.loc[mask].copy()


def _contar_acertos(previsao: List[int], alvo: List[int]) -> int:
    return len(set(previsao) & set(alvo))


def _s6_profundo(
    df_contexto: pd.DataFrame,
    num_passageiros: int,
    qtd_series: int,
) -> List[List[int]]:
    cols_pass = obter_colunas_passageiros(df_contexto)
    valores = df_contexto[cols_pass].values.ravel()
    valores = valores[valores >= 0]

    if len(valores) == 0:
        return []

    uniques, counts = np.unique(valores, return_counts=True)
    ordem = np.argsort(-counts)
    ordenados = uniques[ordem]

    base = list(ordenados[:num_passageiros])
    previsoes = []

    for i in range(qtd_series):
        rota = i % num_passageiros
        p = base[rota:] + base[:rota]
        previsoes.append(sorted(p[:num_passageiros]))

    return previsoes


def _micro_leque(
    df_contexto: pd.DataFrame,
    num_passageiros: int,
    qtd_series: int,
) -> List[List[int]]:
    cols_pass = obter_colunas_passageiros(df_contexto)
    valores = df_contexto[cols_pass].values.ravel()
    valores = valores[valores >= 0]

    if len(valores) == 0:
        return []

    uniques, counts = np.unique(valores, return_counts=True)
    ordem = np.argsort(-counts)
    ordenados = uniques[ordem]

    top = list(ordenados[: 2 * num_passageiros])
    if len(top) < num_passageiros:
        top = list(ordenados)

    rng = np.random.default_rng(12345)
    previsoes = []
    for _ in range(qtd_series):
        escolha = rng.choice(top, size=num_passageiros, replace=False)
        previsoes.append(sorted(list(escolha)))

    return previsoes


def _monte_carlo_profundo(
    df_contexto: pd.DataFrame,
    num_passageiros: int,
    qtd_series: int,
    n_iter: int = 500,
) -> List[List[int]]:
    cols_pass = obter_colunas_passageiros(df_contexto)
    valores = df_contexto[cols_pass].values.ravel()
    valores = valores[valores >= 0]

    if len(valores) == 0:
        return []

    uniques, counts = np.unique(valores, return_counts=True)
    probs = counts / counts.sum()

    rng = np.random.default_rng(98765)
    previsoes = []
    for _ in range(qtd_series):
        amostra = rng.choice(uniques, size=num_passageiros, replace=False, p=probs)
        previsoes.append(sorted(list(amostra)))

    return previsoes


def combinar_previsoes_hibrido(
    s6_list: List[List[int]],
    micro_list: List[List[int]],
    mc_list: List[List[int]],
    k_star_local: float,
    nr_local: float,
) -> Dict[str, List[List[int]]]:
    if k_star_local < 0.25:
        w_s6 = 0.6
        w_micro = 0.25
        w_mc = 0.15
    elif k_star_local < 0.5:
        w_s6 = 0.5
        w_micro = 0.3
        w_mc = 0.2
    elif k_star_local < 0.75:
        w_s6 = 0.35
        w_micro = 0.3
        w_mc = 0.35
    else:
        w_s6 = 0.2
        w_micro = 0.3
        w_mc = 0.5

    if nr_local > 0.7:
        w_s6 *= 0.6
        w_micro *= 1.2
        w_mc *= 1.3
    elif nr_local > 0.4:
        w_s6 *= 0.8
        w_micro *= 1.1
        w_mc *= 1.1

    total = w_s6 + w_micro + w_mc
    if total <= 0:
        total = 1.0
    w_s6 /= total
    w_micro /= total
    w_mc /= total

    max_len = max(len(s6_list), len(micro_list), len(mc_list))
    if max_len == 0:
        return {
            "S6": [],
            "Micro": [],
            "MC": [],
            "Hibrido": [],
        }

    def safe_get(lst: List[List[int]], i: int) -> Optional[List[int]]:
        if i < len(lst):
            return lst[i]
        return None

    hibrido = []
    for i in range(max_len):
        cand_s6 = safe_get(s6_list, i)
        cand_micro = safe_get(micro_list, i)
        cand_mc = safe_get(mc_list, i)

        pool: Dict[int, float] = {}
        for cand, w in [(cand_s6, w_s6), (cand_micro, w_micro), (cand_mc, w_mc)]:
            if cand is None:
                continue
            for n in cand:
                pool[n] = pool.get(n, 0.0) + w

        if not pool:
            continue

        ordenados = sorted(pool.items(), key=lambda x: -x[1])
        base_len = 0
        for lst in [cand_s6, cand_micro, cand_mc]:
            if lst is not None and len(lst) > 0:
                base_len = len(lst)
                break
        if base_len <= 0:
            base_len = 6

        escolhidos = [x[0] for x in ordenados[:base_len]]
        hibrido.append(sorted(escolhidos))

    return {
        "S6": s6_list,
        "Micro": micro_list,
        "MC": mc_list,
        "Hibrido": hibrido,
    }


def gerar_leque_previsoes_v15(
    df: pd.DataFrame,
    idx_alvo: int,
    janela_contexto: int,
    qtd_series_s6: int,
    qtd_series_micro: int,
    qtd_series_mc: int,
) -> Dict[str, List[List[int]]]:
    df_contexto = _extrair_ultimas_series(df, idx_alvo, janela_contexto)
    num_passageiros = obter_num_passageiros(df)

    df_contexto = df_contexto.copy()
    df_contexto["k_star"] = calcular_k_star(df_contexto)
    df_contexto["nr"] = estimar_ruido_condicional(df_contexto)

    k_star_local = float(df_contexto["k_star"].iloc[-1])
    nr_local = float(df_contexto["nr"].iloc[-1])

    s6_list = _s6_profundo(df_contexto, num_passageiros, qtd_series_s6)
    micro_list = _micro_leque(df_contexto, num_passageiros, qtd_series_micro)
    mc_list = _monte_carlo_profundo(df_contexto, num_passageiros, qtd_series_mc)

    combinado = combinar_previsoes_hibrido(
        s6_list=s6_list,
        micro_list=micro_list,
        mc_list=mc_list,
        k_star_local=k_star_local,
        nr_local=nr_local,
    )

    return combinado


def formatar_previsao(lista: List[int]) -> str:
    return " ".join(str(int(x)) for x in sorted(lista))


def montar_dataframe_leque(
    combinado: Dict[str, List[List[int]]]
) -> pd.DataFrame:
    registros = []
    for tipo in ["S6", "Micro", "MC", "Hibrido"]:
        listas = combinado.get(tipo, [])
        for i, p in enumerate(listas, start=1):
            registros.append(
                {
                    "Módulo": tipo,
                    "Ordem": i,
                    "Previsão": formatar_previsao(p),
                }
            )
    return pd.DataFrame(registros)
# ============================================================
# Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo)
# ============================================================

def _backtest_simple_window(
    df: pd.DataFrame,
    janela_contexto: int,
    qtd_series: int,
    passo: int = 1,
) -> Dict[str, float]:
    max_idx = int(df["idx"].max())
    cols_pass = obter_colunas_passageiros(df)

    resultados_acertos = []

    for idx_alvo in range(janela_contexto + 2, max_idx + 1, passo):
        sub_df = df[df["idx"] < idx_alvo]
        if len(sub_df) < janela_contexto:
            continue

        combinado = gerar_leque_previsoes_v15(
            df=df,
            idx_alvo=idx_alvo,
            janela_contexto=janela_contexto,
            qtd_series_s6=qtd_series,
            qtd_series_micro=qtd_series,
            qtd_series_mc=qtd_series,
        )
        leque_hibrido = combinado.get("Hibrido", [])
        if not leque_hibrido:
            continue

        alvo_row = df[df["idx"] == idx_alvo]
        if alvo_row.empty:
            continue

        alvo_vals = alvo_row[cols_pass].iloc[0].tolist()

        melhor = 0
        for prev in leque_hibrido:
            ac = _contar_acertos(prev, alvo_vals)
            melhor = max(melhor, ac)

        resultados_acertos.append(melhor)

    if not resultados_acertos:
        return {
            "media_acertos": 0.0,
            "qtd_testes": 0,
        }

    media_acertos = float(np.mean(resultados_acertos))
    return {
        "media_acertos": media_acertos,
        "qtd_testes": len(resultados_acertos),
    }


def _qds_qualidade_distribuicao_series(
    df: pd.DataFrame,
    janela_contexto: int,
) -> Dict[str, float]:
    num_passageiros = obter_num_passageiros(df)
    cols_pass = obter_colunas_passageiros(df)

    if len(df) < janela_contexto:
        sub = df.copy()
    else:
        sub = df.iloc[-janela_contexto:].copy()

    valores = sub[cols_pass].values.ravel()
    valores = valores[valores >= 0]
    if len(valores) == 0:
        return {
            "qds": 0.0,
            "diversidade": 0.0,
        }

    uniques, counts = np.unique(valores, return_counts=True)
    probs = counts / counts.sum()

    entropia = -np.sum(probs * np.log(probs + 1e-9))
    entropia_max = math.log(len(uniques) + 1e-9)
    if entropia_max <= 0:
        diversidade = 0.0
    else:
        diversidade = float(entropia / entropia_max)

    sub["k_star"] = calcular_k_star(sub)
    k_star_mean = float(sub["k_star"].mean())

    qds_raw = diversidade * (1.0 - 0.4 * k_star_mean)
    qds = max(0.0, min(1.0, qds_raw))

    return {
        "qds": qds,
        "diversidade": diversidade,
    }


def _monte_carlo_confiabilidade(
    df: pd.DataFrame,
    janela_contexto: int,
    qtd_series: int,
    num_sim: int = 200,
) -> Dict[str, float]:
    max_idx = int(df["idx"].max())
    cols_pass = obter_colunas_passageiros(df)
    rng = np.random.default_rng(112233)

    resultados = []
    for _ in range(num_sim):
        if max_idx <= janela_contexto + 2:
            break

        idx_alvo = int(rng.integers(janela_contexto + 2, max_idx + 1))
        combinado = gerar_leque_previsoes_v15(
            df=df,
            idx_alvo=idx_alvo,
            janela_contexto=janela_contexto,
            qtd_series_s6=qtd_series,
            qtd_series_micro=qtd_series,
            qtd_series_mc=qtd_series,
        )
        leque_hibrido = combinado.get("Hibrido", [])
        if not leque_hibrido:
            continue

        alvo_row = df[df["idx"] == idx_alvo]
        if alvo_row.empty:
            continue

        alvo_vals = alvo_row[cols_pass].iloc[0].tolist()
        melhor = 0
        for prev in leque_hibrido:
            ac = _contar_acertos(prev, alvo_vals)
            melhor = max(melhor, ac)

        resultados.append(melhor)

    if not resultados:
        return {
            "media_acertos": 0.0,
            "qtd_sim": 0,
        }

    media_acertos = float(np.mean(resultados))
    return {
        "media_acertos": media_acertos,
        "qtd_sim": len(resultados),
    }


def consolidar_confiabilidade_real(
    df: pd.DataFrame,
    janela_contexto: int,
    qtd_series: int,
) -> Dict[str, Dict[str, float]]:
    qds_info = _qds_qualidade_distribuicao_series(
        df=df,
        janela_contexto=janela_contexto,
    )
    backtest_info = _backtest_simple_window(
        df=df,
        janela_contexto=janela_contexto,
        qtd_series=qtd_series,
        passo=max(1, janela_contexto // 5),
    )
    mc_info = _monte_carlo_confiabilidade(
        df=df,
        janela_contexto=janela_contexto,
        qtd_series=qtd_series,
        num_sim=150,
    )

    return {
        "QDS": qds_info,
        "Backtest": backtest_info,
        "MonteCarlo": mc_info,
    }

# ============================================================
# Helper — Montagem do Relatório Final V15-HÍBRIDO
# ============================================================

def montar_relatorio_final_v15() -> str:
    df = get_df()
    if df is None or df.empty:
        raise ValueError("Histórico não carregado.")

    max_idx = int(df["idx"].max())
    num_pass = obter_num_passageiros(df)

    # Fonte principal de contexto: TURBO++ ULTRA, se já rodou
    fonte = st.session_state.get("ultima_previsao_turbo")
    if fonte is None:
        fonte = st.session_state.get("ultima_previsao")

    if fonte is not None:
        idx_alvo = int(fonte.get("idx_alvo", max_idx + 1))
        janela_contexto = int(fonte.get("janela_contexto", min(150, max_idx)))
    else:
        idx_alvo = max_idx + 1
        janela_contexto = min(150, max_idx)

    idx_ini = max(1, idx_alvo - janela_contexto)
    idx_fim = min(max_idx, idx_alvo - 1)
    if idx_fim < idx_ini:
        idx_fim = max_idx

    df_contexto = df[(df["idx"] >= idx_ini) & (df["idx"] <= idx_fim)].copy()
    if df_contexto.empty:
        df_contexto = df.copy()

    # Barômetro global
    bar_global = calcular_barometro_global(df)
    df_contexto["k_star"] = calcular_k_star(df_contexto)
    df_contexto["nr"] = estimar_ruido_condicional(df_contexto)

    k_star_local = float(df_contexto["k_star"].iloc[-1])
    nr_local = float(df_contexto["nr"].iloc[-1])
    nr_mean = float(df_contexto["nr"].mean())

    # Confiabilidade
    confi = st.session_state.get("ultima_confiabilidade")
    if confi is None:
        confi = consolidar_confiabilidade_real(
            df=df_contexto,
            janela_contexto=min(janela_contexto, len(df_contexto)),
            qtd_series=5,
        )

    qds = confi["QDS"]
    back = confi["Backtest"]
    mc = confi["MonteCarlo"]

    ambiente_score = qds["qds"]
    ambiente_score += 0.1 * max(0.0, back["media_acertos"] - 2) / 4.0
    ambiente_score += 0.1 * max(0.0, mc["media_acertos"] - 2) / 4.0
    ambiente_score = max(0.0, min(1.0, ambiente_score))

    penal_ruido = nr_local
    penal_k_star = k_star_local
    fator_conf = ambiente_score * (1.0 - 0.5 * penal_ruido - 0.4 * penal_k_star)
    fator_conf = max(0.0, min(1.0, fator_conf))

    # Envelope oficial, se vier do TURBO++ ULTRA
    envelope_oficial = []
    hibrido_list = []
    if st.session_state.get("ultima_previsao_turbo") is not None:
        info_turbo = st.session_state["ultima_previsao_turbo"]
        hibrido_list = info_turbo.get("hibrido_list", [])
        envelope_oficial = info_turbo.get("envelope_oficial", [])
    elif st.session_state.get("ultima_previsao") is not None:
        info = st.session_state["ultima_previsao"]
        combinado = info.get("combinado", {})
        hibrido_list = combinado.get("Hibrido", [])
        envelope_oficial = hibrido_list[:3]

    # Texto do envelope
    def blocos_previsao(lista_series: List[List[int]]) -> str:
        if not lista_series:
            return "Nenhuma série oficial registrada."
        linhas = []
        for i, prev in enumerate(lista_series, start=1):
            linhas.append(f"{i:02d}) {formatar_previsao(prev)}")
        return "\n".join(linhas)

    txt_env = blocos_previsao(envelope_oficial)

    rel = []
    rel.append("=== RELATÓRIO FINAL V15-HÍBRIDO ===")
    rel.append("")
    rel.append(f"Série alvo (hipotética): C{idx_alvo}")
    rel.append(f"Janela de contexto: {janela_contexto} séries (C{idx_ini} até C{idx_fim})")
    rel.append(f"Histórico total: {len(df)} séries, {num_pass} passageiros por série.")
    rel.append("")
    rel.append("--- Ambiente Global ---")
    rel.append(f"k médio global: {bar_global['k_mean']:.2f}")
    rel.append(f"k máximo global: {bar_global['k_max']:.2f}")
    rel.append(f"k* médio global: {bar_global['k_star_mean']:.2f}")
    rel.append(classificar_barometro_global(bar_global))
    rel.append("")
    rel.append("--- Ambiente Local (trecho de contexto) ---")
    rel.append(f"k* local (última série da janela): {k_star_local:.2f}")
    rel.append(classificar_regime_k_star(k_star_local))
    rel.append(f"NR% local: {nr_local:.2f}")
    rel.append(f"NR% médio no trecho: {nr_mean:.2f}")
    rel.append(classificar_nivel_ruido(nr_local))
    rel.append("")
    rel.append("--- Confiabilidade (QDS / Backtest / Monte Carlo) ---")
    rel.append(f"QDS: {qds['qds']:.2f} (diversidade: {qds['diversidade']:.2f})")
    rel.append(
        f"Backtest — média de acertos (HÍBRIDO): {back['media_acertos']:.2f} "
        f"em {back['qtd_testes']} janelas."
    )
    rel.append(
        f"Monte Carlo — média de acertos (HÍBRIDO): {mc['media_acertos']:.2f} "
        f"em {mc['qtd_sim']} simulações."
    )
    rel.append(f"Fator de confiança consolidado: {fator_conf:.2f}")
    rel.append("")
    rel.append("--- Envelope Oficial TURBO++ ULTRA (se disponível) ---")
    rel.append(txt_env)
    rel.append("")
    rel.append("--- Observações gerais ---")
    if fator_conf >= 0.7:
        rel.append(
            "Ambiente favorável: estrada com boa qualidade para previsão híbrida "
            "(nível alto de confiança)."
        )
    elif fator_conf >= 0.4:
        rel.append(
            "Ambiente intermediário: estrada utilizável, mas com pontos de atenção "
            "em ruído ou turbulência."
        )
    else:
        rel.append(
            "Ambiente hostil: ruído/turbulência ou baixa aderência do método. "
            "Usar envelope com cautela, considerar trechos alternativos."
        )

    return "\n".join(rel)
# ============================================================
# UI — Navegação
# ============================================================

st.sidebar.title("🚗 Predict Cars V15-HÍBRIDO")

painel = st.sidebar.radio(
    "📂 Navegação",
    [
        "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)",
        "🔍 Pipeline V14-FLEX ULTRA (V15)",
        "💡 Replay LIGHT",
        "📅 Replay ULTRA",
        "🎯 Replay ULTRA Unitário",
        "🚨 Monitor de Risco (k & k*)",
        "🧪 Testes de Confiabilidade REAL",
        "📊 Ruído Condicional (V15)",
        "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)",
        "📄 Relatório Final V15-HÍBRIDO",
    ],
)

# ============================================================
# PAINEL 1 — Histórico — Entrada FLEX ULTRA
# ============================================================
if painel == "📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)":
    st.markdown("## 📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)")

    st.markdown(
        """
        - Última coluna = **k** (guardas que acertaram exatamente o carro).  
        - Demais colunas (entre série e k) = **passageiros** (n1..nN).  
        - Opcionalmente, a **primeira coluna** pode ser o rótulo da série (C1, C2...).
        """
    )

    modo_entrada = st.radio(
        "Modo de entrada do histórico:",
        ["Carregar arquivo CSV", "Colar texto"],
        horizontal=True,
    )

    formato_linhas = st.selectbox(
        "Formato de cada linha do histórico:",
        [
            "Série na primeira coluna + passageiros + k",
            "Somente passageiros + k",
        ],
    )

    df = get_df()
    if df is not None and not df.empty:
        st.success(
            f"Histórico atualmente carregado: {len(df)} séries, "
            f"{obter_num_passageiros(df)} passageiros por série."
        )
        with st.expander("Visualizar amostra do histórico carregado"):
            st.dataframe(df.head(20), use_container_width=True)

    if modo_entrada == "Carregar arquivo CSV":
        uploaded_file = st.file_uploader(
            "Selecione o arquivo de histórico (.csv):",
            type=["csv", "txt"],
        )

        if uploaded_file is not None:
            if st.button("📥 Carregar histórico do arquivo"):
                try:
                    df_novo = carregar_historico_arquivo(
                        uploaded_file=uploaded_file,
                        formato_linhas=formato_linhas,
                    )
                    set_df(df_novo)
                    st.success(
                        f"Histórico carregado com sucesso: {len(df_novo)} séries, "
                        f"{obter_num_passageiros(df_novo)} passageiros por série."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao carregar histórico: {e}")
    else:
        texto = st.text_area(
            "Cole aqui o histórico (linhas com ';' ou ',' como separador):",
            height=200,
        )
        if st.button("📥 Carregar histórico do texto"):
            try:
                df_novo = carregar_historico_texto(
                    texto=texto,
                    formato_linhas=formato_linhas,
                )
                set_df(df_novo)
                st.success(
                    f"Histórico carregado com sucesso: {len(df_novo)} séries, "
                    f"{obter_num_passageiros(df_novo)} passageiros por série."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao carregar histórico: {e}")

# ============================================================
# PAINEL 2 — Pipeline V14-FLEX ULTRA (V15)
# ============================================================
if painel == "🔍 Pipeline V14-FLEX ULTRA (V15)":
    st.markdown("## 🔍 Pipeline V14-FLEX ULTRA (V15) — Execução Simples")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    num_pass = obter_num_passageiros(df)
    st.info(
        f"Histórico com **{len(df)} séries** e **{num_pass} passageiros** por série."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_alvo = st.number_input(
            "Índice alvo (Cidx) para previsão:",
            min_value=2,
            max_value=int(df["idx"].max()) + 1,
            value=int(df["idx"].max()) + 1,
            step=1,
        )
    with col2:
        janela_contexto = st.slider(
            "Janela de contexto (séries anteriores):",
            min_value=30,
            max_value=min(500, int(df["idx"].max())),
            value=min(150, int(df["idx"].max())),
            step=10,
        )
    with col3:
        qtd_series_leque = st.slider(
            "Quantidade de séries por módulo (S6 / Micro / MC):",
            min_value=2,
            max_value=20,
            value=6,
            step=1,
        )

    if st.button("🚀 Rodar Pipeline V14-FLEX ULTRA (V15)"):
        try:
            combinado = gerar_leque_previsoes_v15(
                df=df,
                idx_alvo=idx_alvo,
                janela_contexto=janela_contexto,
                qtd_series_s6=qtd_series_leque,
                qtd_series_micro=qtd_series_leque,
                qtd_series_mc=qtd_series_leque,
            )
            df_leque = montar_dataframe_leque(combinado)
            st.session_state["ultima_previsao"] = {
                "idx_alvo": idx_alvo,
                "janela_contexto": janela_contexto,
                "qtd_series_leque": qtd_series_leque,
                "df_leque": df_leque,
                "combinado": combinado,
            }

            st.success("Pipeline executado com sucesso. Veja o leque abaixo.")

            st.subheader("Leque de Previsões — V15-HÍBRIDO")
            st.dataframe(df_leque, use_container_width=True)

            hibrido = combinado.get("Hibrido", [])
            if hibrido:
                prev_oficial = hibrido[0]
                st.markdown("### 🔚 Previsão Final TURBO++ HÍBRIDO (V15)")
                st.markdown(
                    f"**Série alvo C{idx_alvo}** (hipotética): "
                    f"`{formatar_previsao(prev_oficial)}`"
                )

                df_contexto = _extrair_ultimas_series(df, idx_alvo, janela_contexto)
                df_contexto["k_star"] = calcular_k_star(df_contexto)
                df_contexto["nr"] = estimar_ruido_condicional(df_contexto)
                k_star_local = float(df_contexto["k_star"].iloc[-1])
                nr_local = float(df_contexto["nr"].iloc[-1])

                st.info(
                    f"**k\* local**: {k_star_local:.2f} — "
                    + classificar_regime_k_star(k_star_local)
                )
                st.info(
                    f"**NR% local (ruído estrutural)**: {nr_local:.2f} — "
                    + classificar_nivel_ruido(nr_local)
                )

        except Exception as e:
            st.error(f"Erro ao rodar o pipeline: {e}")
# ============================================================
# PAINEL 3 — Replay LIGHT
# ============================================================
if painel == "💡 Replay LIGHT":
    st.markdown("## 💡 Replay LIGHT — Navegação rápida pelo histórico")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    max_idx = int(df["idx"].max())

    col1, col2 = st.columns(2)
    with col1:
        idx_ini = st.number_input(
            "Índice inicial (Cidx):",
            min_value=1,
            max_value=max_idx,
            value=max(1, max_idx - 50),
            step=1,
        )
    with col2:
        idx_fim = st.number_input(
            "Índice final (Cidx):",
            min_value=idx_ini,
            max_value=max_idx,
            value=max_idx,
            step=1,
        )

    df_trecho = df[(df["idx"] >= idx_ini) & (df["idx"] <= idx_fim)].copy()
    st.info(
        f"TRECHO selecionado: {len(df_trecho)} séries (C{idx_ini} até C{idx_fim})."
    )

    with st.expander("Visualizar trecho"):
        st.dataframe(df_trecho, use_container_width=True)

    df_trecho["k_star"] = calcular_k_star(df_trecho)
    df_trecho["nr"] = estimar_ruido_condicional(df_trecho)

    k_star_mean = float(df_trecho["k_star"].mean())
    nr_mean = float(df_trecho["nr"].mean())

    st.markdown("### Barômetro do trecho (Replay LIGHT)")
    st.write(f"**k\* médio do trecho**: {k_star_mean:.2f}")
    st.write(classificar_regime_k_star(k_star_mean))
    st.write(f"**NR% médio (ruído estrutural)**: {nr_mean:.2f}")
    st.write(classificar_nivel_ruido(nr_mean))

# ============================================================
# PAINEL 4 — Replay ULTRA
# ============================================================
if painel == "📅 Replay ULTRA":
    st.markdown("## 📅 Replay ULTRA — Backtest visual da estrada")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    max_idx = int(df["idx"].max())

    col1, col2, col3 = st.columns(3)
    with col1:
        janela_contexto = st.slider(
            "Janela de contexto:",
            min_value=30,
            max_value=min(500, max_idx),
            value=min(150, max_idx),
            step=10,
        )
    with col2:
        passo = st.slider(
            "Passo (pular séries para acelerar):",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )
    with col3:
        qtd_series = st.slider(
            "Qtd de séries por módulo (S6 / Micro / MC):",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
        )

    if st.button("▶ Rodar Replay ULTRA (backtest simplificado)"):
        try:
            info_backtest = _backtest_simple_window(
                df=df,
                janela_contexto=janela_contexto,
                qtd_series=qtd_series,
                passo=passo,
            )
            media_acertos = info_backtest["media_acertos"]
            qtd_testes = info_backtest["qtd_testes"]

            st.success("Replay ULTRA executado.")
            st.markdown("### Resultado do Replay ULTRA")
            st.write(f"**Quantidade de janelas testadas**: {qtd_testes}")
            st.write(f"**Média de acertos máximos por janela (HÍBRIDO)**: {media_acertos:.2f}")

            if qtd_testes > 0:
                if media_acertos >= 4:
                    st.info("🟢 Excelente aderência do método híbrido nesse trecho da estrada.")
                elif media_acertos >= 3:
                    st.warning("🟡 Aderência razoável; ambiente exige respeito, mas há potencial.")
                else:
                    st.error("🔴 Aderência baixa: ambiente difícil para previsão robusta.")

        except Exception as e:
            st.error(f"Erro no Replay ULTRA: {e}")

# ============================================================
# PAINEL 5 — Replay ULTRA Unitário
# ============================================================
if painel == "🎯 Replay ULTRA Unitário":
    st.markdown("## 🎯 Replay ULTRA Unitário — Janela específica")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    max_idx = int(df["idx"].max())
    cols_pass = obter_colunas_passageiros(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_alvo_replay = st.number_input(
            "Índice alvo para Replay (série real existe):",
            min_value=2,
            max_value=max_idx,
            value=max_idx,
            step=1,
        )
    with col2:
        janela_contexto = st.slider(
            "Janela de contexto:",
            min_value=30,
            max_value=min(500, max_idx - 1),
            value=min(150, max_idx - 1),
            step=10,
        )
    with col3:
        qtd_series = st.slider(
            "Qtd de séries por módulo (S6 / Micro / MC):",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
        )

    if st.button("▶ Rodar Replay ULTRA Unitário"):
        try:
            combinado = gerar_leque_previsoes_v15(
                df=df,
                idx_alvo=idx_alvo_replay,
                janela_contexto=janela_contexto,
                qtd_series_s6=qtd_series,
                qtd_series_micro=qtd_series,
                qtd_series_mc=qtd_series,
            )
            df_leque = montar_dataframe_leque(combinado)

            alvo_row = df[df["idx"] == idx_alvo_replay]
            if alvo_row.empty:
                st.error("Série alvo não encontrado no histórico.")
            else:
                alvo_vals = alvo_row[cols_pass].iloc[0].tolist()

                st.subheader(f"Série real C{idx_alvo_replay}")
                st.write("Passageiros reais:")
                st.code(formatar_previsao(alvo_vals))

                st.subheader("Leque gerado (HÍBRIDO + módulos)")
                st.dataframe(df_leque, use_container_width=True)

                registros = []
                for _, row in df_leque.iterrows():
                    previsao_lista = [int(x) for x in str(row["Previsão"]).split()]
                    acertos = _contar_acertos(previsao_lista, alvo_vals)
                    registros.append(acertos)

                df_leque["Acertos"] = registros
                st.markdown("### Leque com acertos (Replay ULTRA Unitário)")
                st.dataframe(df_leque, use_container_width=True)

                melhor = max(registros) if registros else 0
                st.info(f"Melhor acerto obtido no leque: **{melhor}** passageiros.")

                if melhor >= 4:
                    st.success("🟢 Excelente ajuste local do método.")
                elif melhor == 3:
                    st.warning("🟡 Ajuste razoável; ambiente ainda respeitável.")
                else:
                    st.error("🔴 Ambiente hostil nesse ponto para o método padrão.")

        except Exception as e:
            st.error(f"Erro no Replay ULTRA Unitário: {e}")

# ============================================================
# PAINEL 6 — Monitor de Risco (k & k*)
# ============================================================
if painel == "🚨 Monitor de Risco (k & k*)":
    st.markdown("## 🚨 Monitor de Risco (k & k*) — Painel dedicado")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    df = df.copy()
    df["k_star"] = calcular_k_star(df)
    df["nr"] = estimar_ruido_condicional(df)

    bar_global = calcular_barometro_global(df)

    st.subheader("Visão global da estrada")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("k médio", f"{bar_global['k_mean']:.2f}")
    with col2:
        st.metric("k máximo", f"{bar_global['k_max']:.2f}")
    with col3:
        st.metric("k* médio", f"{bar_global['k_star_mean']:.2f}")
    with col4:
        nr_global = float(df["nr"].mean())
        st.metric("NR% médio", f"{nr_global:.2f}")

    st.info(classificar_barometro_global(bar_global))
    st.info(classificar_nivel_ruido(nr_global))

    st.markdown("### Zoom local")
    max_idx = int(df["idx"].max())
    col1, col2 = st.columns(2)
    with col1:
        idx_ref = st.number_input(
            "Índice de referência (Cidx):",
            min_value=1,
            max_value=max_idx,
            value=max_idx,
            step=1,
        )
    with col2:
        janela_local = st.slider(
            "Janela local (± séries):",
            min_value=10,
            max_value=200,
            value=60,
            step=10,
        )

    ini = max(1, idx_ref - janela_local)
    fim = min(max_idx, idx_ref + janela_local)

    df_local = df[(df["idx"] >= ini) & (df["idx"] <= fim)].copy()
    st.write(f"Trecho local: C{ini} até C{fim} (centro em C{idx_ref}).")

    with st.expander("Ver tabela local (k, k*, NR%)"):
        st.dataframe(
            df_local[["idx", "serie_id", "k", "k_star", "nr"]],
            use_container_width=True,
        )

    df_centro = df_local[df_local["idx"] == idx_ref]
    if not df_centro.empty:
        k_c = float(df_centro["k"].iloc[0])
        k_star_c = float(df_centro["k_star"].iloc[0])
        nr_c = float(df_centro["nr"].iloc[0])

        st.markdown("### Sentinelas no ponto de referência")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("k (guardas que acertaram)", f"{k_c:.0f}")
        with col2:
            st.metric("k* local", f"{k_star_c:.2f}")
        with col3:
            st.metric("NR% local", f"{nr_c:.2f}")

        st.info(classificar_regime_k_star(k_star_c))
        st.info(classificar_nivel_ruido(nr_c))
# ============================================================
# PAINEL 7 — Testes de Confiabilidade REAL
# ============================================================
if painel == "🧪 Testes de Confiabilidade REAL":
    st.markdown("## 🧪 Testes de Confiabilidade REAL (QDS / Backtest / Monte Carlo)")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    max_idx = int(df["idx"].max())

    col1, col2 = st.columns(2)
    with col1:
        janela_contexto = st.slider(
            "Janela de contexto principal:",
            min_value=30,
            max_value=min(500, max_idx),
            value=min(150, max_idx),
            step=10,
        )
    with col2:
        qtd_series = st.slider(
            "Qtd de séries por módulo (S6 / Micro / MC):",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
        )

    if st.button("🧪 Rodar Testes de Confiabilidade REAL"):
        try:
            consolidado = consolidar_confiabilidade_real(
                df=df,
                janela_contexto=janela_contexto,
                qtd_series=qtd_series,
            )
            st.session_state["ultima_confiabilidade"] = consolidado

            qds = consolidado["QDS"]
            back = consolidado["Backtest"]
            mc = consolidado["MonteCarlo"]

            st.subheader("Resumo das métricas")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("QDS (0–1)", f"{qds['qds']:.2f}")
                st.write(f"Diversidade: {qds['diversidade']:.2f}")
            with col2:
                st.metric(
                    "Backtest — média acertos",
                    f"{back['media_acertos']:.2f}",
                )
                st.write(f"Janelas testadas: {back['qtd_testes']}")
            with col3:
                st.metric(
                    "Monte Carlo — média acertos",
                    f"{mc['media_acertos']:.2f}",
                )
                st.write(f"Sims: {mc['qtd_sim']}")

            st.markdown("### Interpretação qualitativa")

            if qds["qds"] >= 0.7:
                st.info("🟢 QDS alto: distribuição saudável para previsão.")
            elif qds["qds"] >= 0.4:
                st.warning("🟡 QDS intermediário: atenção a trechos de pior qualidade.")
            else:
                st.error("🔴 QDS baixo: estrada com distribuição complicada.")

            if back["qtd_testes"] > 0:
                if back["media_acertos"] >= 4:
                    st.info("🟢 Backtest indica excelente aderência do método híbrido.")
                elif back["media_acertos"] >= 3:
                    st.warning("🟡 Backtest mediano, aderência razoável.")
                else:
                    st.error("🔴 Backtest fraco, ambiente hostil para o método atual.")

            if mc["qtd_sim"] > 0:
                if mc["media_acertos"] >= 4:
                    st.info("🟢 Monte Carlo sugere alta robustez do método.")
                elif mc["media_acertos"] >= 3:
                    st.warning("🟡 Monte Carlo razoável; robustez moderada.")
                else:
                    st.error("🔴 Monte Carlo fraco; método sofre com o ruído.")

        except Exception as e:
            st.error(f"Erro nos testes de confiabilidade: {e}")

# ============================================================
# PAINEL 8 — Ruído Condicional (V15)
# ============================================================
if painel == "📊 Ruído Condicional (V15)":
    st.markdown("## 📊 Ruído Condicional (V15) — Mapa de NR%")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    df = df.copy()
    df["nr"] = estimar_ruido_condicional(df)
    max_idx = int(df["idx"].max())

    st.markdown("### Visão tabular")
    with st.expander("Ver tabela (idx, série, k, NR%)"):
        st.dataframe(
            df[["idx", "serie_id", "k", "nr"]],
            use_container_width=True,
        )

    st.markdown("### Zoom por trecho")

    col1, col2 = st.columns(2)
    with col1:
        idx_ref = st.number_input(
            "Índice de referência (Cidx):",
            min_value=1,
            max_value=max_idx,
            value=max_idx,
            step=1,
        )
    with col2:
        janela_local = st.slider(
            "Janela (± séries):",
            min_value=10,
            max_value=200,
            value=80,
            step=10,
        )

    ini = max(1, idx_ref - janela_local)
    fim = min(max_idx, idx_ref + janela_local)
    df_local = df[(df["idx"] >= ini) & (df["idx"] <= fim)].copy()

    st.write(f"Trecho C{ini} até C{fim}.")
    with st.expander("Ver NR% no trecho"):
        st.dataframe(
            df_local[["idx", "serie_id", "k", "nr"]],
            use_container_width=True,
        )

    nr_mean = float(df_local["nr"].mean())
    st.info(
        f"NR% médio do trecho: {nr_mean:.2f} — "
        + classificar_nivel_ruido(nr_mean)
    )
# ============================================================
# PAINEL 9 — Modo TURBO++ ULTRA ANTI-RUÍDO (V15)
# ============================================================
if painel == "🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)":
    st.markdown("## 🚀 Modo TURBO++ ULTRA ANTI-RUÍDO (V15)")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    max_idx = int(df["idx"].max())

    col1, col2, col3 = st.columns(3)
    with col1:
        idx_alvo = st.number_input(
            "Índice alvo Cidx (próximo carro):",
            min_value=2,
            max_value=max_idx + 1,
            value=max_idx + 1,
            step=1,
        )
    with col2:
        janela_contexto = st.slider(
            "Janela de contexto:",
            min_value=50,
            max_value=min(600, max_idx),
            value=min(200, max_idx),
            step=10,
        )
    with col3:
        qtd_series = st.slider(
            "Qtd de séries HÍBRIDAS no envelope:",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
        )

    st.markdown("### Ajustes anti-ruído")

    col1, col2 = st.columns(2)
    with col1:
        peso_ruido = st.slider(
            "Peso da penalização por NR% (0 = ignorar, 1 = forte):",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )
    with col2:
        peso_k_star = st.slider(
            "Peso da adaptação por k* (0 = ignorar, 1 = forte):",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
        )

    if st.button("🚀 Gerar envelope TURBO++ ULTRA ANTI-RUÍDO"):
        try:
            combinado_base = gerar_leque_previsoes_v15(
                df=df,
                idx_alvo=idx_alvo,
                janela_contexto=janela_contexto,
                qtd_series_s6=qtd_series,
                qtd_series_micro=qtd_series,
                qtd_series_mc=qtd_series,
            )
            hibrido_list = combinado_base.get("Hibrido", [])
            if not hibrido_list:
                st.error("Não foi possível gerar leque híbrido para esse alvo.")
                st.stop()

            df_contexto = _extrair_ultimas_series(df, idx_alvo, janela_contexto)
            df_contexto["k_star"] = calcular_k_star(df_contexto)
            df_contexto["nr"] = estimar_ruido_condicional(df_contexto)
            k_star_local = float(df_contexto["k_star"].iloc[-1])
            nr_local = float(df_contexto["nr"].iloc[-1])

            confi = consolidar_confiabilidade_real(
                df=df_contexto,
                janela_contexto=min(janela_contexto, len(df_contexto)),
                qtd_series=min(qtd_series, 8),
            )

            qds_val = confi["QDS"]["qds"]
            back_ac = confi["Backtest"]["media_acertos"]
            mc_ac = confi["MonteCarlo"]["media_acertos"]

            ambiente_score = qds_val
            ambiente_score += 0.1 * max(0.0, back_ac - 2) / 4.0
            ambiente_score += 0.1 * max(0.0, mc_ac - 2) / 4.0
            ambiente_score = max(0.0, min(1.0, ambiente_score))

            penal_ruido = peso_ruido * nr_local
            penal_k_star = peso_k_star * k_star_local
            fator_conf = ambiente_score * (1.0 - 0.5 * penal_ruido - 0.4 * penal_k_star)
            fator_conf = max(0.0, min(1.0, fator_conf))

            qtd_oficiais = max(1, int(qtd_series * fator_conf))
            qtd_oficiais = min(qtd_oficiais, len(hibrido_list))

            envelope_oficial = hibrido_list[:qtd_oficiais]

            st.session_state["ultima_previsao_turbo"] = {
                "idx_alvo": idx_alvo,
                "janela_contexto": janela_contexto,
                "qtd_series": qtd_series,
                "hibrido_list": hibrido_list,
                "envelope_oficial": envelope_oficial,
                "k_star_local": k_star_local,
                "nr_local": nr_local,
                "confi": confi,
                "fator_conf": fator_conf,
            }

            st.success("Envelope TURBO++ ULTRA ANTI-RUÍDO gerado.")

            st.markdown("### 🔚 Envelope Oficial de Previsões (TURBO++ ULTRA)")
            st.write(f"Série alvo: **C{idx_alvo}** (hipotética).")
            st.write(f"Quantidade de séries oficiais: **{len(envelope_oficial)}** de {len(hibrido_list)} geradas.")

            registros = []
            for i, prev in enumerate(envelope_oficial, start=1):
                registros.append(
                    {
                        "Ordem": i,
                        "Previsão": formatar_previsao(prev),
                    }
                )
            df_env = pd.DataFrame(registros)
            st.dataframe(df_env, use_container_width=True)

            st.markdown("### Ambiente e confiança")

            col1, col2, col3 = st.columns(3)
            qds_info = confi["QDS"]
            back_info = confi["Backtest"]
            mc_info = confi["MonteCarlo"]

            with col1:
                st.metric("QDS", f"{qds_info['qds']:.2f}")
            with col2:
                st.metric("Backtest méd. acertos", f"{back_info['media_acertos']:.2f}")
            with col3:
                st.metric("Monte Carlo méd. acertos", f"{mc_info['media_acertos']:.2f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("k* local", f"{k_star_local:.2f}")
            with col2:
                st.metric("NR% local", f"{nr_local:.2f}")
            with col3:
                st.metric("Fator de confiança", f"{fator_conf:.2f}")

            st.info(classificar_regime_k_star(k_star_local))
            st.info(classificar_nivel_ruido(nr_local))

        except Exception as e:
            st.error(f"Erro no Modo TURBO++ ULTRA ANTI-RUÍDO: {e}")

# ============================================================
# PAINEL 10 — 📄 Relatório Final V15-HÍBRIDO
# ============================================================
if painel == "📄 Relatório Final V15-HÍBRIDO":
    st.markdown("## 📄 Relatório Final V15-HÍBRIDO")

    df = get_df()
    if df is None or df.empty:
        st.warning("Carregue o histórico primeiro no painel '📥 Histórico — Entrada FLEX ULTRA (V15-HÍBRIDO)'.")
        st.stop()

    st.markdown(
        """
        Este painel consolida **ambiente, risco, ruído, confiabilidade e envelope**  
        em um único texto, pronto para você **copiar e colar aqui no chat**  
        para análise conjunta (humano + máquina).
        """
    )

    if st.button("📝 Gerar Relatório Final V15-HÍBRIDO"):
        try:
            rel = montar_relatorio_final_v15()
            st.session_state["ultimo_relatorio_final"] = rel
            st.success("Relatório gerado. Você pode copiar o texto abaixo.")
        except Exception as e:
            st.error(f"Erro ao montar o relatório final: {e}")

    rel_txt = st.session_state.get("ultimo_relatorio_final")
    if rel_txt:
        st.markdown("### Texto do Relatório Final (selecione e copie):")
        st.text_area(
            "Relatório Final",
            value=rel_txt,
            height=400,
        )
    else:
        st.info("Ainda não há relatório gerado nesta sessão.")
