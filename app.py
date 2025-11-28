# =========================================================
# BLOCO 1 — app.py TURBO
# Imports, configuração e funções básicas de parsing/métricas
# =========================================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Configuração geral do app
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predict Cars V13.8-TURBO",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------
# Modelos básicos de dados
# ---------------------------------------------------------

@dataclass
class SeriesRecord:
    """
    Representa uma série individual no histórico.

    Atributos
    ---------
    idx : Optional[str]
        Identificador da série, por exemplo 'C2943'. Pode ser None.
    passengers : List[int]
        Lista de passageiros (números principais da série).
    k_label : Optional[int]
        Rótulo numérico adicional opcional (k).
    """
    idx: Optional[str]
    passengers: List[int]
    k_label: Optional[int] = None


@dataclass
class RegimeState:
    """
    Descreve o estado da estrada (regime) para o trecho mais recente.
    """
    nome: str
    score_resiliencia: float
    score_turbulencia: float
    comentario_curto: str


# ---------------------------------------------------------
# Funções utilitárias gerais
# ---------------------------------------------------------

def _safe_int(x: str) -> Optional[int]:
    """
    Converte string em inteiro de forma segura.
    Retorna None em caso de erro.
    """
    x = x.strip()
    if not x:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def parse_history_text(text: str, max_passengers: int = 6) -> List[SeriesRecord]:
    """
    Lê o histórico em formato texto e converte em uma lista de SeriesRecord.

    Formatos aceitos (por linha):
    - C2943;8;29;30;36;39;60
    - 8;29;30;36;39;60
    - C2943;8;29;30;36;39;60;7
    - 8;29;30;36;39;60;7

    Regras:
    - Ignora linhas vazias.
    - Aceita tanto ponto e vírgula ';' quanto vírgula ',' como separador.
    - Remove espaços em excesso.
    """
    records: List[SeriesRecord] = []

    # Normaliza quebras de linha
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Normaliza separadores
        if ";" in line:
            parts = [p.strip() for p in line.split(";")]
        else:
            parts = [p.strip() for p in line.split(",")]

        if not parts:
            continue

        idx: Optional[str] = None
        nums: List[int] = []
        k_label: Optional[int] = None

        # Detecta se o primeiro elemento é um identificador do tipo Cxxxx
        first = parts[0]
        if first.upper().startswith("C") and len(parts) > 1:
            idx = first.strip()
            num_parts = parts[1:]
        else:
            num_parts = parts

        # Converte tudo para int, ignorando vazios
        temp_nums: List[int] = []
        for p in num_parts:
            val = _safe_int(p)
            if val is not None:
                temp_nums.append(val)

        if not temp_nums:
            continue

        # Se houver mais que max_passengers, o último é tratado como k_label
        if len(temp_nums) > max_passengers:
            passengers = temp_nums[:max_passengers]
            k_label = temp_nums[max_passengers]
        else:
            passengers = temp_nums

        # Garante unicidade básica dos passageiros
        passengers = list(dict.fromkeys(passengers))[:max_passengers]

        if len(passengers) == 0:
            continue

        record = SeriesRecord(idx=idx, passengers=passengers, k_label=k_label)
        records.append(record)

    return records


def records_to_dataframe(records: List[SeriesRecord]) -> pd.DataFrame:
    """
    Converte a lista de SeriesRecord em DataFrame tabular.

    Colunas:
    - idx: identificador textual opcional
    - p1..pN: passageiros
    - k: rótulo opcional
    """
    if not records:
        return pd.DataFrame(columns=["idx", "k"])

    max_len = max(len(r.passengers) for r in records)
    data = []
    for r in records:
        row: Dict[str, Any] = {
            "idx": r.idx,
            "k": r.k_label,
        }
        for i in range(max_len):
            col = f"p{i + 1}"
            row[col] = r.passengers[i] if i < len(r.passengers) else np.nan
        data.append(row)

    df = pd.DataFrame(data)
    # Cria um índice numérico contínuo, mesmo que idx textual exista
    df["row_id"] = np.arange(1, len(df) + 1)
    return df


def load_history(
    uploaded_file, pasted_text: str
) -> Tuple[List[SeriesRecord], pd.DataFrame, str]:
    """
    Carrega o histórico a partir de:
    - arquivo enviado, se existir
    - caso contrário, texto colado

    Retorna:
    - lista de SeriesRecord
    - DataFrame correspondente
    - origem ('file', 'text' ou 'empty')
    """
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        # Tenta detectar encoding simples
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_bytes.decode("latin-1")
            except UnicodeDecodeError:
                text = raw_bytes.decode(errors="ignore")
        origin = "file"
    else:
        text = pasted_text or ""
        origin = "text" if text.strip() else "empty"

    if not text.strip():
        return [], pd.DataFrame(columns=["idx", "k", "row_id"]), origin

    records = parse_history_text(text)
    df = records_to_dataframe(records)
    return records, df, origin


# ---------------------------------------------------------
# Métricas básicas e leitura do estado da estrada
# ---------------------------------------------------------

def compute_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas agregadas da estrada a partir do DataFrame.

    Saída:
    - n_series: quantidade de linhas
    - n_passengers: quantidade média de passageiros por série
    - amplitude: max(n) - min(n)
    - dispersion: desvio padrão dos passageiros
    - vibration: média do módulo da variação entre séries consecutivas
    - pairs_activity: densidade de pares recorrentes
    """
    metrics: Dict[str, Any] = {
        "n_series": 0,
        "n_passengers": 0.0,
        "amplitude": 0.0,
        "dispersion": 0.0,
        "vibration": 0.0,
        "pairs_activity": 0.0,
    }

    if df.empty:
        return metrics

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return metrics

    # Matriz de passageiros
    values = df[passenger_cols].to_numpy(dtype=float)
    mask = ~np.isnan(values)

    # Número de séries e passageiros médios
    metrics["n_series"] = len(df)
    metrics["n_passengers"] = float(mask.sum(axis=1).mean())

    # Flatten de todos os passageiros válidos
    all_vals = values[mask]
    if all_vals.size > 0:
        metrics["amplitude"] = float(all_vals.max() - all_vals.min())
        metrics["dispersion"] = float(all_vals.std(ddof=1) if all_vals.size > 1 else 0.0)

    # Vibração: variação média entre séries consecutivas (média dos mínimos deslocamentos)
    if len(df) > 1:
        diffs = []
        for i in range(1, len(df)):
            prev = set(v for v in values[i - 1] if not math.isnan(v))
            cur = set(v for v in values[i] if not math.isnan(v))
            if prev and cur:
                # distância média de cada número atual ao mais próximo da série anterior
                d_list = []
                for v in cur:
                    d_list.append(min(abs(v - u) for u in prev))
                diffs.append(np.mean(d_list))
        metrics["vibration"] = float(np.mean(diffs)) if diffs else 0.0

    # Atividade de pares (pares recorrentes ao longo do histórico)
    from collections import Counter

    pair_counter: Counter[Tuple[int, int]] = Counter()
    for row in values:
        row_vals = [int(v) for v in row if not math.isnan(v)]
        row_vals = sorted(set(row_vals))
        for i in range(len(row_vals)):
            for j in range(i + 1, len(row_vals)):
                pair_counter[(row_vals[i], row_vals[j])] += 1

    if pair_counter:
        total_pairs = sum(pair_counter.values())
        distinct_pairs = len(pair_counter)
        metrics["pairs_activity"] = float(total_pairs / max(distinct_pairs, 1))
    else:
        metrics["pairs_activity"] = 0.0

    return metrics


def infer_regime(metrics: Dict[str, Any]) -> RegimeState:
    """
    Infere o regime da estrada a partir de métricas básicas.

    Lógica heurística:
    - baixa vibração + baixa dispersão -> Resiliente
    - vibração moderada + dispersão moderada -> Intermediário
    - vibração alta + dispersão alta -> Turbulento
    - vibração muito alta com aumento recente -> Pré-Ruptura / Ruptura
    """
    vib = float(metrics.get("vibration", 0.0) or 0.0)
    disp = float(metrics.get("dispersion", 0.0) or 0.0)

    # Normalização simples para faixas de decisão
    vib_level = "low"
    if vib > 6.0:
        vib_level = "high"
    elif vib > 3.0:
        vib_level = "mid"

    disp_level = "low"
    if disp > 20.0:
        disp_level = "high"
    elif disp > 10.0:
        disp_level = "mid"

    # Combinação de níveis para regime
    if vib_level == "low" and disp_level == "low":
        nome = "Resiliente"
        score_res = 0.9
        score_turb = 0.1
        comment = "Estrada estável, núcleo tende a se manter coerente."
    elif vib_level == "mid" and disp_level in ("low", "mid"):
        nome = "Intermediário"
        score_res = 0.5
        score_turb = 0.5
        comment = "Estrada em transição, equilíbrio entre repetição e renovação."
    elif vib_level == "high" and disp_level == "high":
        nome = "Turbulento"
        score_res = 0.2
        score_turb = 0.9
        comment = "Estrada agitada, movimentos amplos e menos previsíveis."
    else:
        # Zona cinza interpretada como estado pré-ruptura / pós-ruptura leve
        nome = "Pré-Ruptura"
        score_res = 0.3
        score_turb = 0.7
        comment = "Estrada em fase sensível, núcleo exige proteção extra."

    return RegimeState(
        nome=nome,
        score_resiliencia=score_res,
        score_turbulencia=score_turb,
        comentario_curto=comment,
    )


# ---------------------------------------------------------
# Inicialização de session_state (para uso nos próximos blocos)
# ---------------------------------------------------------

def init_session_state() -> None:
    """
    Garante que chaves essenciais estejam presentes em st.session_state.
    """
    defaults = {
        "history_records": [],
        "history_df": pd.DataFrame(),
        "history_origin": "empty",
        "basic_metrics": {},
        "regime_state": None,
        "turbo_output": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# FIM DO BLOCO 1 — app.py TURBO
# (Copiar exatamente como está para o início do arquivo)
# =========================================================# =========================================================
# BLOCO 2 — app.py TURBO
# Interface inicial Streamlit + carregamento do histórico +
# exibição + leitura de estado + métricas e regime
# =========================================================

# ---------------------------------------------------------
# Inicialização e layout inicial
# ---------------------------------------------------------

init_session_state()

st.title("🚗 Predict Cars V13.8 — Modo TURBO")

st.markdown("""
Sistema completo de previsão **Ultra-Híbrido TURBO**  
com todas as camadas profundas do motor V13.8:

- IDX Avançado  
- IPF / IPO Profundo  
- ASB Anti-SelfBias  
- ICA / HLA Profundo  
- ADN (leve / médio / profundo)  
- Dependências Ocultas  
- Trechos Espelhados  
- S6 Avançado  
- Monte Carlo Profundo  
- Backtest Interno + Backtest do Futuro  
- Núcleo Resiliente Final  
""")

st.divider()


# ---------------------------------------------------------
# Painel lateral — Entrada do histórico
# ---------------------------------------------------------

st.sidebar.header("📥 Entrada do Histórico")

uploaded_file = st.sidebar.file_uploader(
    "Enviar arquivo (.txt ou .csv)",
    type=["txt", "csv"],
    accept_multiple_files=False,
)

pasted_text = st.sidebar.text_area(
    "Ou colar o histórico aqui",
    height=200,
    placeholder="Exemplo:\nC2943; 8; 29; 30; 36; 39; 60\n8; 29; 30; 36; 39; 60\n..."
)

btn_load = st.sidebar.button("Carregar Histórico")


# ---------------------------------------------------------
# Carregamento do histórico
# ---------------------------------------------------------

if btn_load:
    records, df, origin = load_history(uploaded_file, pasted_text)

    st.session_state["history_records"] = records
    st.session_state["history_df"] = df
    st.session_state["history_origin"] = origin

    # computa métricas
    metrics = compute_basic_metrics(df)
    st.session_state["basic_metrics"] = metrics

    # inferir regime
    regime = infer_regime(metrics)
    st.session_state["regime_state"] = regime

    st.success("Histórico carregado com sucesso.")


# ---------------------------------------------------------
# Exibição do histórico
# ---------------------------------------------------------

df = st.session_state["history_df"]

if df.empty:
    st.warning("Nenhum histórico carregado ainda.")
else:
    st.subheader("📊 Histórico Carregado")
    st.dataframe(df, use_container_width=True)

    metrics = st.session_state["basic_metrics"]
    regime = st.session_state["regime_state"]

    st.divider()

    # -----------------------------------------------------
    # Painel de métricas gerais
    # -----------------------------------------------------
    st.subheader("📡 Métricas da Estrada")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Séries", metrics.get("n_series", 0))
    c2.metric("Passageiros médios", f"{metrics.get('n_passengers', 0):.2f}")
    c3.metric("Amplitude", f"{metrics.get('amplitude', 0):.1f}")
    c4.metric("Dispersão", f"{metrics.get('dispersion', 0):.2f}")
    c5.metric("Vibração", f"{metrics.get('vibration', 0):.2f}")
    c6.metric("Atividade de pares", f"{metrics.get('pairs_activity', 0):.2f}")

    st.divider()

    # -----------------------------------------------------
    # Painel do regime
    # -----------------------------------------------------
    st.subheader("🌡️ Estado da Estrada (Regime)")

    regime_box = st.container()
    with regime_box:
        if regime:
            if regime.nome == "Resiliente":
                color = "#4caf50"
            elif regime.nome == "Intermediário":
                color = "#ff9800"
            elif regime.nome == "Turbulento":
                color = "#f44336"
            else:
                color = "#9c27b0"  # Pré-Ruptura

            st.markdown(
                f"""
                <div style="
                    padding: 15px;
                    border-radius: 10px;
                    background-color: {color}22;
                    border-left: 4px solid {color};
                ">
                    <h4 style="margin:0;">{regime.nome}</h4>
                    <p style="margin:0;">
                        {regime.comentario_curto}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"**Resiliência:** {regime.score_resiliencia:.2f}")
            st.write(f"**Turbulência:** {regime.score_turbulencia:.2f}")

    st.divider()

    # (Os demais blocos — IDX, IPF, IPO, ICA, HLA, ASB, etc. —
    #  serão adicionados nos BLOCO 3, 4, 5... até a finalização.)
    

# =========================================================
# FIM DO BLOCO 2 — app.py TURBO
# =========================================================
# =========================================================
# BLOCO 3 — app.py TURBO
# Implementação do IDX Avançado:
# - similaridade estrutural
# - similaridade de faixas
# - similaridade de pares
# - similaridade de ritmo
# - ranking de trechos gêmeos
# - painel Streamlit
# =========================================================

# ---------------------------------------------------------
# Funções internas de similaridade para o IDX Avançado
# ---------------------------------------------------------

def similarity_structural(a: List[int], b: List[int]) -> float:
    """
    Similaridade estrutural: mede alinhamento bruto entre conjuntos.
    Retorna valor entre 0 e 1.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union > 0 else 0.0


def similarity_ranges(a: List[int], b: List[int]) -> float:
    """
    Similaridade por faixas (low/mid/high).
    Agrupa passageiros em: 1-26 (low), 27-53 (mid), 54-80 (high).
    """
    def band(x):
        if x <= 26: return "L"
        if x <= 53: return "M"
        return "H"

    bands_a = [band(x) for x in a]
    bands_b = [band(x) for x in b]

    sa, sb = set(bands_a), set(bands_b)
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union > 0 else 0.0


def similarity_pairs(a: List[int], b: List[int]) -> float:
    """
    Similaridade de pares (pares recorrentes).
    Quanto mais pares coincidem, maior a similaridade.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0

    def make_pairs(lst):
        lst = sorted(set(lst))
        return {(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))}

    pa = make_pairs(a)
    pb = make_pairs(b)

    if not pa and not pb:
        return 0.0

    inter = len(pa.intersection(pb))
    union = len(pa.union(pb))
    return inter / union if union > 0 else 0.0


def similarity_rhythm(a: List[int], b: List[int]) -> float:
    """
    Similaridade de ritmo:
    compara a forma dos deslocamentos internos (diferenças ordenadas).
    Quanto mais parecida a estrutura de variações, maior a similaridade.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0

    da = sorted(a)
    db = sorted(b)

    diffa = [da[i + 1] - da[i] for i in range(len(da) - 1)]
    diffb = [db[i + 1] - db[i] for i in range(len(db) - 1)]

    # Ajuste para tamanhos diferentes
    m = min(len(diffa), len(diffb))
    if m == 0:
        return 0.0

    da2 = np.array(diffa[:m], dtype=float)
    db2 = np.array(diffb[:m], dtype=float)

    # Similaridade inversa da distância normalizada
    dist = np.linalg.norm(da2 - db2)
    maxdist = np.linalg.norm(np.maximum(da2, db2))

    if maxdist == 0:
        return 1.0
    score = 1.0 - (dist / maxdist)
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------
# IDX Avançado completo (unifica todas as similaridades)
# ---------------------------------------------------------

def run_IDX_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o IDX Avançado:

    - extrai a última série
    - compara com todas as anteriores
    - computa 4 similaridades:
        estrutural
        faixas
        pares
        ritmo
    - unifica tudo em um ranking final
    """

    if df.empty:
        return pd.DataFrame()

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    values = df[passenger_cols].to_numpy(dtype=float)

    # Última série
    last = [int(v) for v in values[-1] if not math.isnan(v)]

    rows = []
    for i in range(len(df) - 1):  # compara com todas menos a última
        cur = [int(v) for v in values[i] if not math.isnan(v)]

        s_struct = similarity_structural(last, cur)
        s_range = similarity_ranges(last, cur)
        s_pairs = similarity_pairs(last, cur)
        s_rhythm = similarity_rhythm(last, cur)

        # Combinação oficial do IDX Avançado
        score = (
            0.40 * s_struct +
            0.20 * s_range +
            0.20 * s_pairs +
            0.20 * s_rhythm
        )

        rows.append({
            "row_id": df.iloc[i]["row_id"],
            "idx": df.iloc[i]["idx"],
            "structural": s_struct,
            "ranges": s_range,
            "pairs": s_pairs,
            "rhythm": s_rhythm,
            "score": score,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Painel Streamlit do IDX Avançado
# ---------------------------------------------------------

if not df.empty:
    st.subheader("🔍 IDX Avançado — Trechos Gêmeos")

    idx_df = run_IDX_advanced(df)

    st.session_state["idx_result"] = idx_df

    if idx_df.empty:
        st.warning("Não foi possível calcular o IDX.")
    else:
        st.dataframe(idx_df.style.format({
            "structural": "{:.3f}",
            "ranges": "{:.3f}",
            "pairs": "{:.3f}",
            "rhythm": "{:.3f}",
            "score": "{:.3f}",
        }), use_container_width=True)

        # Destaque do Top 3
        st.markdown("### 🏆 Top 3 Trechos Mais Semelhantes")
        top3 = idx_df.head(3)
        st.table(top3[["row_id", "idx", "score"]])

    st.divider()

# =========================================================
# FIM DO BLOCO 3 — app.py TURBO
# =========================================================# =========================================================
# =========================================================
# BLOCO 4 — app.py TURBO (LIMPO)
# IPF (Puro Focado) + IPO (Otimizado Profundo)
# =========================================================

from collections import Counter

# ---------------------------------------------------------
# Funções auxiliares — IPF / IPO
# ---------------------------------------------------------

def get_passengers_from_row(row: pd.Series) -> List[int]:
    vals = []
    for col in row.index:
        if col.startswith("p"):
            v = row[col]
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(int(v))
    return vals


def build_candidate_universe(df: pd.DataFrame, idx_df: pd.DataFrame, top_k: int = 10):
    if df.empty or idx_df.empty:
        return [], {}

    sub = idx_df.head(top_k)
    weights = Counter()

    for _, r in sub.iterrows():
        row_id = r["row_id"]
        score = float(r["score"])
        gain = 1.0 + score

        base_row = df[df["row_id"] == row_id]
        if base_row.empty:
            continue
        passengers = get_passengers_from_row(base_row.iloc[0])

        for n in passengers:
            weights[n] += gain

    if not weights:
        return [], {}

    ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    candidates = [n for n, _ in ordered]
    weight_dict = {n: float(w) for n, w in ordered}
    return candidates, weight_dict


def compute_strong_pairs_from_candidates(df, idx_df, top_k=10, max_pairs=10):
    if df.empty or idx_df.empty:
        return []

    pair_counter = Counter()
    sub = idx_df.head(top_k)

    for _, r in sub.iterrows():
        row_id = r["row_id"]
        base_row = df[df["row_id"] == row_id]
        if base_row.empty:
            continue
        passengers = sorted(set(get_passengers_from_row(base_row.iloc[0])))

        for i in range(len(passengers)):
            for j in range(i + 1, len(passengers)):
                pair_counter[(passengers[i], passengers[j])] += 1

    if not pair_counter:
        return []

    ordered = sorted(pair_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [p for p, _ in ordered[:max_pairs]]


def infer_dominant_band(candidates: List[int]) -> str:
    if not candidates:
        return "Indefinida"

    def band(x):
        if x <= 26: return "L"
        if x <= 53: return "M"
        return "H"

    c = Counter(band(x) for x in candidates)
    code, _ = c.most_common(1)[0]
    mapping = {"L": "Baixa (1–26)", "M": "Média (27–53)", "H": "Alta (54–80)"}
    return mapping.get(code, "Indefinida")


def compute_core_energy(weight_dict):
    if not weight_dict:
        return 0.0
    arr = np.array(list(weight_dict.values())).astype(float)
    mean_w = float(arr.mean())
    max_w = float(arr.max())
    if max_w <= 0:
        return 0.0
    return float(max(0.0, min(1.0, mean_w / max_w)))


def select_ipf_core(candidates, core_size=6):
    if not candidates:
        return []
    core = sorted(set(candidates[:core_size]))[:core_size]
    return core


def quality_against_neighbors(core, df, idx_df, top_k=10):
    if not core or df.empty or idx_df.empty:
        return 0.0

    sub = idx_df.head(top_k)
    scores = []

    for _, r in sub.iterrows():
        row_id = r["row_id"]
        base_row = df[df["row_id"] == row_id]
        if base_row.empty:
            continue

        passengers = get_passengers_from_row(base_row.iloc[0])

        s_struct = similarity_structural(core, passengers)
        s_range  = similarity_ranges(core, passengers)
        s_pairs  = similarity_pairs(core, passengers)
        s_rhythm = similarity_rhythm(core, passengers)

        score = (
            0.40 * s_struct +
            0.20 * s_range +
            0.25 * s_pairs +
            0.15 * s_rhythm
        )
        scores.append(score)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def enforce_regime_overlap(core, last_series, regime,
                           desired_resilient_overlap=(3, 5),
                           max_turbulent_overlap=2):

    if not core or not last_series or regime is None:
        return core

    core_set = set(core)
    last_set = set(last_series)

    current_overlap = len(core_set & last_set)
    new_core = list(core)
    pool = sorted(set(core) | set(last_series))

    if regime.nome == "Resiliente":
        low, high = desired_resilient_overlap

        if current_overlap < low:
            missing = list(last_set - core_set)
            i = 0
            for m in missing:
                new_core.sort(reverse=True)
                if len(new_core) > i:
                    new_core[i] = m
                    i += 1
                if len(set(new_core) & last_set) >= low:
                    break

        if len(set(new_core) & last_set) > high:
            excess = len(set(new_core) & last_set) - high
            for _ in range(excess):
                victim = None
                for n in sorted(new_core, reverse=True):
                    if n in last_set:
                        victim = n
                        break
                if victim:
                    new_core.remove(victim)
                    for c in pool:
                        if c not in new_core:
                            new_core.append(c)
                            break

    elif regime.nome == "Turbulento":
        if current_overlap > max_turbulent_overlap:
            to_remove = current_overlap - max_turbulent_overlap
            for _ in range(to_remove):
                victim = None
                for n in new_core:
                    if n in last_set:
                        victim = n
                        break
                if victim:
                    new_core.remove(victim)
                    for c in pool:
                        if c not in new_core and c not in last_set:
                            new_core.append(c)
                            break

    return sorted(set(new_core))[:len(core)]


def run_IPF_IPO(df, idx_df, regime, core_size=6, neighbor_k=10, optimization_steps=80):
    ipf = {}
    ipo = {}

    if df.empty or idx_df.empty:
        return ipf, ipo

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    values = df[passenger_cols].to_numpy(dtype=float)
    last_series = [int(v) for v in values[-1] if not math.isnan(v)]

    candidates, weight_dict = build_candidate_universe(df, idx_df, top_k=neighbor_k)
    strong_pairs = compute_strong_pairs_from_candidates(df, idx_df, top_k=neighbor_k)
    dominant_band = infer_dominant_band(candidates)
    energy = compute_core_energy(weight_dict)

    pre_core = select_ipf_core(candidates, core_size=core_size)

    ipf = {
        "pre_core": pre_core,
        "candidates": candidates,
        "weights": weight_dict,
        "strong_pairs": strong_pairs,
        "dominant_band": dominant_band,
        "energy": energy,
    }

    if not pre_core:
        return ipf, ipo

    extra_pool = sorted(set(candidates[: core_size * 3]) | set(last_series))
    current_core = enforce_regime_overlap(pre_core, last_series, regime)
    current_score = quality_against_neighbors(current_core, df, idx_df, top_k=neighbor_k)

    for _ in range(optimization_steps):
        if not extra_pool:
            break

        pos = random.randrange(len(current_core))
        old_val = current_core[pos]

        choices = [x for x in extra_pool if x not in current_core]
        if not choices:
            continue

        new_val = random.choice(choices)

        trial = list(current_core)
        trial[pos] = new_val
        trial = sorted(set(trial))[:core_size]
        trial = enforce_regime_overlap(trial, last_series, regime)

        t_score = quality_against_neighbors(trial, df, idx_df, top_k=neighbor_k)

        if t_score > current_score:
            current_core = trial
            current_score = t_score

    overlap_last = len(set(current_core) & set(last_series))

    ipo = {
        "structural_core": sorted(current_core),
        "quality": float(current_score),
        "overlap_last": overlap_last,
        "regime": regime.nome if regime else None,
    }

    return ipf, ipo


# ---------------------------------------------------------
# Painel Streamlit — IPF / IPO
# ---------------------------------------------------------

if not df.empty:
    st.subheader("🧠 Núcleo Estrutural (IPF / IPO)")

    idx_res = st.session_state.get("idx_result", pd.DataFrame())
    regime_state = st.session_state.get("regime_state", None)

    if idx_res.empty:
        st.info("IDX ainda não calculado.")
    else:
        ipf_out, ipo_out = run_IPF_IPO(df, idx_res, regime_state)

        st.session_state["ipf_core"] = ipf_out
        st.session_state["ipo_core"] = ipo_out

        if not ipf_out or not ipo_out:
            st.warning("Não foi possível computar IPF/IPO.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Núcleo Pré-Bruto (IPF)")
                st.write(f"**IPF:** {ipf_out['pre_core']}")
                st.write(f"**Energia estrutural:** {ipf_out['energy']:.3f}")
                st.write(f"**Faixa dominante:** {ipf_out['dominant_band']}")
                st.write("**Pares fortes:**")
                st.write(ipf_out["strong_pairs"] or "Nenhum.")

            with c2:
                st.markdown("### Núcleo Otimizado (IPO)")
                st.write(f"**IPO:** {ipo_out['structural_core']}")
                st.write(f"**Qualidade vs IDX:** {ipo_out['quality']:.3f}")
                st.write(f"**Overlap última série:** {ipo_out['overlap_last']}")
                if ipo_out["regime"]:
                    st.write(f"**Regime usado:** {ipo_out['regime']}")

            st.divider()

# =========================================================
# FIM DO BLOCO 4 — LIMPO — app.py TURBO
# =========================================================
# =========================================================
# BLOCO 5 — app.py TURBO
# ASB (Anti-SelfBias)
# ADN (Ajuste Dinâmico: leve / médio / profundo)
# ICA (Iterative Core Adjustment Profundo)
# HLA (High-Level Adjustment)
# =========================================================

# ---------------------------------------------------------
# ASB — Anti-SelfBias
# ---------------------------------------------------------

def apply_ASB(core: List[int], last_series: List[int], regime: Optional[RegimeState]) -> List[int]:
    """
    Política Anti-SelfBias:
    - Em regime Resiliente: repetição permitida, mas limitada.
    - Em regime Intermediário: controle moderado de repetição.
    - Em regime Turbulento: repetição mínima.
    """
    if not core or not last_series or regime is None:
        return core

    core_set = set(core)
    last_set = set(last_series)
    overlap = len(core_set & last_set)
    new_core = list(core)

    if regime.nome == "Resiliente":
        # Permite algum overlap, mas não total
        if overlap > 4:
            # substitui excedentes
            for n in core:
                if n in last_set:
                    new_core.remove(n)
                    break

    elif regime.nome == "Intermediário":
        # overlap moderado
        if overlap > 3:
            for n in core:
                if n in last_set:
                    new_core.remove(n)
                    break

    elif regime.nome == "Turbulento":
        # proteger contra repetição excessiva
        if overlap > 2:
            for n in core:
                if n in last_set:
                    new_core.remove(n)
                    break

    # Repreenche com números próximos ao universo original (de forma neutra)
    pool = list(range(1, 81))
    for p in pool:
        if len(new_core) >= len(core):
            break
        if p not in new_core:
            new_core.append(p)

    return sorted(new_core)[:len(core)]


# ---------------------------------------------------------
# ADN — Ajuste Dinâmico
# ---------------------------------------------------------

def apply_ADN(core: List[int], energy: float, mode: str = "leve") -> List[int]:
    """
    Ajuste Dinâmico:
    - leve: pequenas trocas baseadas em energia
    - medio: substitui pontos fracos + reforça vizinhanças
    - profundo: reestrutura o núcleo mantendo identidade
    """
    if not core:
        return core

    new_core = list(core)

    if mode == "leve":
        # pequenas variações
        if energy < 0.45:
            # tenta reforçar com números vizinhos
            for i in range(len(new_core)):
                v = new_core[i]
                if v + 1 <= 80:
                    new_core[i] = v + 1

    elif mode == "medio":
        # substitui ponto de menor densidade
        weakest = min(new_core)
        new_core.remove(weakest)
        # reforço estrutural na média
        avg = int(sum(new_core) / len(new_core))
        if avg not in new_core:
            new_core.append(avg)
        else:
            new_core.append((avg % 80) + 1)

    elif mode == "profundo":
        # reorganiza totalmente com redução / expansão suave
        for i in range(len(new_core)):
            v = new_core[i]
            shift = (-3 + i) if energy < 0.5 else (2 - i)
            nv = v + shift
            if 1 <= nv <= 80:
                new_core[i] = nv

    return sorted(set(new_core))[:len(core)]


# ---------------------------------------------------------
# ICA — Iterative Core Adjustment (Profundo)
# ---------------------------------------------------------

def run_ICA(core: List[int], df: pd.DataFrame, idx_df: pd.DataFrame,
            iterations: int = 60, neighbor_k: int = 10) -> List[int]:

    if not core or df.empty or idx_df.empty:
        return core

    current_core = list(core)
    current_score = quality_against_neighbors(core, df, idx_df, top_k=neighbor_k)

    for _ in range(iterations):
        pos = random.randrange(len(current_core))
        old = current_core[pos]

        # Nova tentativa aleatória
        new_val = random.randint(1, 80)
        if new_val in current_core:
            continue

        trial = list(current_core)
        trial[pos] = new_val
        trial = sorted(set(trial))[:len(core)]

        t_score = quality_against_neighbors(trial, df, idx_df, top_k=neighbor_k)

        # critério de aceitação estritamente superior
        if t_score > current_score:
            current_core = trial
            current_score = t_score

    return current_core


# ---------------------------------------------------------
# HLA — High-Level Adjustment
# ---------------------------------------------------------

def run_HLA(core: List[int], candidates: List[int],
            strong_pairs: List[Tuple[int,int]],
            desired_size: int = 6) -> List[int]:

    if not core:
        return core

    new_core = list(core)

    # 1. Reforça pares fortes
    for (a, b) in strong_pairs:
        if a in new_core and b not in new_core:
            # adiciona b substituindo o maior elemento
            victim = max(new_core)
            if b not in new_core:
                new_core.remove(victim)
                new_core.append(b)

    # 2. Se houver muita divergência estrutural, aproxima do universo candidato
    if len(set(new_core) & set(candidates[:10])) <= 2:
        new_core = sorted(set(new_core + candidates[:4]))[:desired_size]

    return sorted(set(new_core))[:desired_size]


# ---------------------------------------------------------
# Pipeline completo do BLOCO 5 (ASB + ADN + ICA + HLA)
# ---------------------------------------------------------

def run_block5_adjustments(df: pd.DataFrame,
                           idx_df: pd.DataFrame,
                           ipf_out: Dict[str,Any],
                           ipo_out: Dict[str,Any],
                           regime: Optional[RegimeState]) -> Dict[str, Any]:

    if not ipf_out or not ipo_out:
        return {}

    structural_core = list(ipo_out["structural_core"])
    energy = ipf_out["energy"]
    candidates = ipf_out["candidates"]
    strong_pairs = ipf_out["strong_pairs"]

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    last_series = [int(v) for v in df[passenger_cols].to_numpy(dtype=float)[-1]
                   if not math.isnan(v)]

    # 1) ASB
    core_asb = apply_ASB(structural_core, last_series, regime)

    # 2) ADN
    if energy > 0.6:
        core_adn = apply_ADN(core_asb, energy, mode="leve")
    elif energy > 0.4:
        core_adn = apply_ADN(core_asb, energy, mode="medio")
    else:
        core_adn = apply_ADN(core_asb, energy, mode="profundo")

    # 3) ICA
    core_ica = run_ICA(core_adn, df, idx_df)

    # 4) HLA
    core_hla = run_HLA(core_ica, candidates, strong_pairs)

    return {
        "after_asb": core_asb,
        "after_adn": core_adn,
        "after_ica": core_ica,
        "after_hla": core_hla,
    }


# ---------------------------------------------------------
# Painel Streamlit — Ajustes Profundos
# ---------------------------------------------------------

if not df.empty:
    st.subheader("⚙️ Ajustes Profundos (ASB + ADN + ICA + HLA)")

    idx_res = st.session_state.get("idx_result", pd.DataFrame())
    ipf_out = st.session_state.get("ipf_core", {})
    ipo_out = st.session_state.get("ipo_core", {})
    regime_state = st.session_state.get("regime_state", None)

    if idx_res.empty or not ipf_out or not ipo_out:
        st.info("IPF/IPO ainda não disponíveis para ajustes profundos.")
    else:
        adj = run_block5_adjustments(df, idx_res, ipf_out, ipo_out, regime_state)
        st.session_state["adjusted_core"] = adj

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("### Após ASB")
            st.write(adj["after_asb"])

        with c2:
            st.markdown("### Após ADN")
            st.write(adj["after_adn"])

        with c3:
            st.markdown("### Após ICA")
            st.write(adj["after_ica"])

        with c4:
            st.markdown("### Núcleo Final (HLA)")
            st.write(adj["after_hla"])

        st.success("Núcleo pós-ajustes concluído (antes de Monte Carlo).")
        st.divider()

# =========================================================
# FIM DO BLOCO 5 — app.py TURBO
# =========================================================
# =========================================================
# BLOCO 6 — app.py TURBO
# Dependências Ocultas + Modo S6 Profundo
# =========================================================

# ---------------------------------------------------------
# Dependências Ocultas
# ---------------------------------------------------------

def compute_hidden_dependencies(
    df: pd.DataFrame,
    max_window: int = 200,
    max_number: int = 80,
) -> Dict[str, Any]:
    """
    Analisa o trecho mais recente do histórico para extrair:
    - força individual dos números (freq + contexto)
    - pares naturais (muito frequentes)
    - pares ocultos (lift alto)
    """
    if df.empty:
        return {
            "number_scores": {},
            "top_numbers": [],
            "natural_pairs": [],
            "hidden_pairs": [],
        }

    passenger_cols = [c for c in df.columns if c.startswith("p")]
    if not passenger_cols:
        return {
            "number_scores": {},
            "top_numbers": [],
            "natural_pairs": [],
            "hidden_pairs": [],
        }

    # janela recente
    recent = df.tail(max_window)
    arr = recent[passenger_cols].to_numpy(dtype=float)

    num_counter = Counter()
    pair_counter = Counter()

    for row in arr:
        vals = sorted({int(v) for v in row if not math.isnan(v) and 1 <= v <= max_number})
        for v in vals:
            num_counter[v] += 1
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pair_counter[(vals[i], vals[j])] += 1

    if not num_counter:
        return {
            "number_scores": {},
            "top_numbers": [],
            "natural_pairs": [],
            "hidden_pairs": [],
        }

    # Frequências individuais
    max_freq = max(num_counter.values())
    number_scores: Dict[int, float] = {}
    for n in range(1, max_number + 1):
        freq = num_counter.get(n, 0)
        score = freq / max_freq if max_freq > 0 else 0.0
        number_scores[n] = float(score)

    # Pares naturais (frequência absoluta)
    natural_pairs = []
    if pair_counter:
        nat_sorted = sorted(
            pair_counter.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
        natural_pairs = [p for p, _ in nat_sorted[:20]]

    # Pares ocultos (lift aproximado)
    hidden_pairs = []
    if pair_counter:
        pair_lift: List[Tuple[Tuple[int, int], float]] = []
        total_series = len(recent)
        for (a, b), c_ab in pair_counter.items():
            p_ab = c_ab / total_series
            p_a = num_counter[a] / total_series
            p_b = num_counter[b] / total_series
            denom = p_a * p_b if p_a * p_b > 0 else 1e-9
            lift = p_ab / denom
            pair_lift.append(((a, b), lift))

        lift_sorted = sorted(
            pair_lift,
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
        # evita duplicar os "naturais" mais óbvios
        hidden_pairs = [p for p, _ in lift_sorted[:20]]

    # Top números por score
    top_numbers = [
        n for n, s in sorted(number_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if s > 0
    ][:25]

    return {
        "number_scores": number_scores,
        "top_numbers": top_numbers,
        "natural_pairs": natural_pairs,
        "hidden_pairs": hidden_pairs,
    }


# ---------------------------------------------------------
# Modo S6 Profundo — geração de séries com alto potencial
# ---------------------------------------------------------

def score_s6_candidate(
    series: List[int],
    base_core: List[int],
    idx_df: pd.DataFrame,
    df: pd.DataFrame,
    hidden_info: Dict[str, Any],
    neighbor_k: int = 10,
) -> float:
    """
    Avalia uma série candidata S6 combinando:
    - alinhamento com o núcleo ajustado (base_core)
    - alinhamento com trechos IDX
    - reforço de dependências ocultas
    - uso de pares naturais/ocultos
    """
    if not series:
        return 0.0

    series = sorted(set(series))
    if len(series) < 3:
        return 0.0

    # 1) similaridade com o núcleo final ajustado
    s_core = similarity_structural(series, base_core)

    # 2) alinhamento com trechos gêmeos
    s_idx = 0.0
    if not idx_df.empty and not df.empty:
        passenger_cols = [c for c in df.columns if c.startswith("p")]
        sub = idx_df.head(neighbor_k)
        scores = []
        for _, r in sub.iterrows():
            row_id = r["row_id"]
            base_row = df[df["row_id"] == row_id]
            if base_row.empty:
                continue
            passengers = get_passengers_from_row(base_row.iloc[0])
            scores.append(similarity_structural(series, passengers))
        if scores:
            s_idx = float(np.mean(scores))

    # 3) reforço de números com alta dependência oculta
    num_scores = hidden_info.get("number_scores", {}) if hidden_info else {}
    if num_scores:
        vals = [num_scores.get(n, 0.0) for n in series]
        hidden_boost = float(np.mean(vals))
    else:
        hidden_boost = 0.0

    # 4) presença de pares naturais/ocultos
    nat_pairs = set(hidden_info.get("natural_pairs", [])) if hidden_info else set()
    hid_pairs = set(hidden_info.get("hidden_pairs", [])) if hidden_info else set()

    def make_pairs(lst):
        lst = sorted(set(lst))
        return {(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))}

    pairs = make_pairs(series)
    if pairs:
        nat_share = len(pairs & nat_pairs) / len(pairs)
        hid_share = len(pairs & hid_pairs) / len(pairs)
    else:
        nat_share = 0.0
        hid_share = 0.0

    # combinação final
    score = (
        0.35 * s_core +
        0.25 * s_idx +
        0.20 * hidden_boost +
        0.10 * nat_share +
        0.10 * hid_share
    )
    return float(score)


def generate_s6_series(
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    adjusted_core: Dict[str, Any],
    hidden_info: Dict[str, Any],
    n_series: int = 12,
) -> pd.DataFrame:
    """
    Gera um conjunto de séries S6 com maior potencial de acertos,
    variando o núcleo final ajustado + reforço de dependências ocultas.
    """
    if not adjusted_core:
        return pd.DataFrame()

    base_core = adjusted_core.get("after_hla") or adjusted_core.get("after_ica") or []
    base_core = sorted(set(base_core))
    if not base_core:
        return pd.DataFrame()

    num_scores = hidden_info.get("number_scores", {}) if hidden_info else {}
    top_numbers = hidden_info.get("top_numbers", []) if hidden_info else []

    # pool de reforço: top numbers + núcleo + vizinhos numéricos
    pool = set(base_core) | set(top_numbers[:30])
    for n in list(pool):
        if n - 1 >= 1:
            pool.add(n - 1)
        if n + 1 <= 80:
            pool.add(n + 1)
    pool = sorted(pool)

    # geração de candidatos por variação controlada
    candidates_set = set()
    max_attempts = n_series * 40
    rng = np.random.default_rng()

    while len(candidates_set) < n_series * 6 and max_attempts > 0:
        max_attempts -= 1
        series = list(base_core)

        # decide quantos elementos trocar
        k_change = rng.integers(1, min(3, len(series)) + 1)

        indices_to_change = rng.choice(len(series), size=k_change, replace=False)
        for idx_pos in indices_to_change:
            # novo número vindo do pool + ruído suave
            candidate_pool = pool.copy()
            for _ in range(5):
                cand = int(rng.integers(1, 81))
                candidate_pool.append(cand)
            replacement = rng.choice(candidate_pool)
            series[idx_pos] = int(replacement)

        series = sorted(set(series))
        if len(series) != 6:
            # completa / ajusta para ter 6 elementos
            while len(series) < 6:
                candidate = int(rng.integers(1, 81))
                if candidate not in series:
                    series.append(candidate)
            series = sorted(series)[:6]

        key = tuple(series)
        candidates_set.add(key)

    # Avaliação dos candidatos
    scored = []
    for s in candidates_set:
        s_list = list(s)
        score = score_s6_candidate(s_list, base_core, idx_df, df, hidden_info)
        scored.append({"series": s_list, "score": score})

    if not scored:
        return pd.DataFrame()

    scored_sorted = sorted(scored, key=lambda x: -x["score"])[:n_series]

    data = []
    for i, item in enumerate(scored_sorted, start=1):
        row = {
            "rank": i,
            "score": float(item["score"]),
        }
        for j, n in enumerate(item["series"], start=1):
            row[f"p{j}"] = n
        data.append(row)

    return pd.DataFrame(data)


# ---------------------------------------------------------
# Painel Streamlit — Dependências Ocultas + S6 Profundo
# ---------------------------------------------------------

if not df.empty:
    st.subheader("🔗 Dependências Ocultas & Modo S6 Profundo")

    idx_res = st.session_state.get("idx_result", pd.DataFrame())
    adjusted_core = st.session_state.get("adjusted_core", {})
    ipf_out = st.session_state.get("ipf_core", {})

    if idx_res.empty or not adjusted_core or not ipf_out:
        st.info("É necessário ter IDX + IPF/IPO + Ajustes Profundos para ativar o Modo S6 Profundo.")
    else:
        # Dependências ocultas
        hidden_info = compute_hidden_dependencies(df)
        st.session_state["hidden_dependencies"] = hidden_info

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Números com maior dependência oculta")
            top_nums = hidden_info["top_numbers"]
            if top_nums:
                df_top = pd.DataFrame(
                    {
                        "número": top_nums,
                        "score": [hidden_info["number_scores"].get(n, 0.0) for n in top_nums],
                    }
                )
                st.dataframe(df_top, use_container_width=True)
            else:
                st.write("Nenhum padrão relevante identificado.")

        with c2:
            st.markdown("### Pares naturais e ocultos (principais)")
            nat_pairs = hidden_info["natural_pairs"]
            hid_pairs = hidden_info["hidden_pairs"]

            df_nat = pd.DataFrame(
                [{"a": a, "b": b} for (a, b) in nat_pairs[:15]]
            )
            df_hid = pd.DataFrame(
                [{"a": a, "b": b} for (a, b) in hid_pairs[:15]]
            )

            st.write("**Pares naturais:**")
            if not df_nat.empty:
                st.table(df_nat)
            else:
                st.write("—")

            st.write("**Pares ocultos:**")
            if not df_hid.empty:
                st.table(df_hid)
            else:
                st.write("—")

        st.markdown("---")
        st.markdown("### 🎯 Séries S6 Profundas (alto potencial)")

        s6_df = generate_s6_series(df, idx_res, adjusted_core, hidden_info, n_series=12)
        st.session_state["s6_series"] = s6_df

        if s6_df.empty:
            st.warning("Não foi possível gerar séries S6 profundas com os dados atuais.")
        else:
            st.dataframe(
                s6_df.style.format({"score": "{:.3f}"}),
                use_container_width=True,
            )

        st.divider()

# =========================================================
# FIM DO BLOCO 6 — app.py TURBO
# =========================================================
# =========================================================
# BLOCO 7 — app.py TURBO
# Monte Carlo Profundo + Simulação Retroativa (Backtest Interno)
# =========================================================

# ---------------------------------------------------------
# Utilitários de avaliação de séries contra o histórico
# ---------------------------------------------------------

def count_hits(series: List[int], row: pd.Series) -> int:
    """
    Conta quantos números de 'series' aparecem em uma linha do DataFrame.
    """
    vals = get_passengers_from_row(row)
    return len(set(series) & set(vals))


def evaluate_series_against_history(
    series: List[int],
    df: pd.DataFrame,
    window: int = 300,
) -> Dict[str, Any]:
    """
    Avalia uma série fixa contra as últimas 'window' séries do histórico.

    Retorna:
    - hits_por_posicao: lista com número de acertos em cada série
    - max_hits: máximo de acertos obtidos
    - media_hits: média de acertos
    - freq_ge3: frequência de ocorrências com 3+ acertos
    - freq_ge4: frequência de ocorrências com 4+ acertos
    """
    if df.empty or not series:
        return {
            "hits_por_posicao": [],
            "max_hits": 0,
            "media_hits": 0.0,
            "freq_ge3": 0.0,
            "freq_ge4": 0.0,
        }

    series = sorted(set(series))
    if len(series) == 0:
        return {
            "hits_por_posicao": [],
            "max_hits": 0,
            "media_hits": 0.0,
            "freq_ge3": 0.0,
            "freq_ge4": 0.0,
        }

    sub = df.tail(window).reset_index(drop=True)
    hits_list: List[int] = []
    for _, row in sub.iterrows():
        hits_list.append(count_hits(series, row))

    if not hits_list:
        return {
            "hits_por_posicao": [],
            "max_hits": 0,
            "media_hits": 0.0,
            "freq_ge3": 0.0,
            "freq_ge4": 0.0,
        }

    arr = np.array(hits_list, dtype=int)
    max_hits = int(arr.max())
    media_hits = float(arr.mean())
    freq_ge3 = float(np.mean(arr >= 3))
    freq_ge4 = float(np.mean(arr >= 4))

    return {
        "hits_por_posicao": hits_list,
        "max_hits": max_hits,
        "media_hits": media_hits,
        "freq_ge3": freq_ge3,
        "freq_ge4": freq_ge4,
    }


# ---------------------------------------------------------
# Monte Carlo Profundo — variações suaves do núcleo
# ---------------------------------------------------------

def monte_carlo_core_scenarios(
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    base_core: List[int],
    n_scenarios: int = 200,
    max_shift: int = 4,
    neighbor_k: int = 10,
) -> pd.DataFrame:
    """
    Gera cenários Monte Carlo em torno de um núcleo base:

    - aplica ruído suave (shifts pequenos e substituições leves)
    - avalia qualidade contra trechos gêmeos (IDX)
    - retorna DataFrame com núcleos simulados e seus scores
    """
    if df.empty or idx_df.empty or not base_core:
        return pd.DataFrame()

    base_core = sorted(set(base_core))
    rng = np.random.default_rng()
    scenarios: List[Dict[str, Any]] = []

    for i in range(n_scenarios):
        # Trabalha com cópia
        core = list(base_core)

        # Número de posições perturbadas
        k_change = rng.integers(1, min(3, len(core)) + 1)
        idx_positions = rng.choice(len(core), size=k_change, replace=False)

        for pos in idx_positions:
            v = core[pos]
            shift = int(rng.integers(-max_shift, max_shift + 1))
            nv = v + shift
            if nv < 1 or nv > 80:
                nv = v  # mantém se sair fora da faixa
            core[pos] = nv

        core = sorted(set(core))
        # garante exatamente o mesmo tamanho
        while len(core) < len(base_core):
            candidate = int(rng.integers(1, 81))
            if candidate not in core:
                core.append(candidate)
        core = sorted(core)[: len(base_core)]

        score_q = quality_against_neighbors(core, df, idx_df, top_k=neighbor_k)

        scenario = {
            "scenario_id": i + 1,
            "score_quality": float(score_q),
        }
        for j, n in enumerate(core, start=1):
            scenario[f"p{j}"] = n

        scenarios.append(scenario)

    if not scenarios:
        return pd.DataFrame()

    mc_df = pd.DataFrame(scenarios)
    mc_df = mc_df.sort_values("score_quality", ascending=False).reset_index(drop=True)
    return mc_df


# ---------------------------------------------------------
# Simulação Retroativa (Backtest Interno)
# ---------------------------------------------------------

def backtest_internal_for_cores(
    df: pd.DataFrame,
    cores: List[List[int]],
    window: int = 300,
) -> pd.DataFrame:
    """
    Executa simulação retroativa simples para um conjunto de núcleos:

    - calcula desempenho em termos de acertos contra o histórico recente
    - retorna DataFrame com estatísticas agregadas por núcleo
    """
    if df.empty or not cores:
        return pd.DataFrame()

    results = []
    for i, core in enumerate(cores, start=1):
        stats = evaluate_series_against_history(core, df, window=window)
        row = {
            "core_id": i,
            "max_hits": stats["max_hits"],
            "media_hits": stats["media_hits"],
            "freq_ge3": stats["freq_ge3"],
            "freq_ge4": stats["freq_ge4"],
        }
        for j, n in enumerate(sorted(set(core)), start=1):
            row[f"p{j}"] = n
        results.append(row)

    bt_df = pd.DataFrame(results)
    if not bt_df.empty:
        bt_df = bt_df.sort_values(
            ["max_hits", "media_hits", "freq_ge4", "freq_ge3"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return bt_df


# ---------------------------------------------------------
# Painel Streamlit — Monte Carlo Profundo + Backtest Interno
# ---------------------------------------------------------

if not df.empty:
    st.subheader("🎲 Monte Carlo Profundo & Backtest Interno")

    idx_res = st.session_state.get("idx_result", pd.DataFrame())
    adjusted_core = st.session_state.get("adjusted_core", {})
    s6_df = st.session_state.get("s6_series", pd.DataFrame())

    if idx_res.empty or not adjusted_core:
        st.info("É necessário ter IDX e Núcleo Ajustado para rodar Monte Carlo e Backtest Interno.")
    else:
        base_core = adjusted_core.get("after_hla") or adjusted_core.get("after_ica") or []
        base_core = sorted(set(base_core))

        if not base_core:
            st.warning("Núcleo ajustado indisponível para Monte Carlo.")
        else:
            # Parâmetros simples (podem ser ajustados no futuro via sidebar se desejar)
            n_scenarios = 200
            max_shift = 4

            mc_df = monte_carlo_core_scenarios(
                df,
                idx_res,
                base_core,
                n_scenarios=n_scenarios,
                max_shift=max_shift,
            )
            st.session_state["mc_core_scenarios"] = mc_df

            if mc_df.empty:
                st.warning("Não foi possível gerar cenários Monte Carlo.")
            else:
                st.markdown("### Distribuição de Cenários Monte Carlo (Núcleo)")

                st.dataframe(
                    mc_df.head(30).style.format({"score_quality": "{:.3f}"}),
                    use_container_width=True,
                )

                # Pequeno resumo estatístico dos scores
                score_series = mc_df["score_quality"]
                st.write("**Resumo de scores (Monte Carlo):**")
                st.write(
                    f"Média: {score_series.mean():.3f} | "
                    f"Mediana: {score_series.median():.3f} | "
                    f"Máximo: {score_series.max():.3f}"
                )

            st.markdown("---")
            st.markdown("### 🧪 Backtest Interno — Núcleos Selecionados")

            # Núcleo base + alguns melhores cenários MC + eventualmente top S6
            candidate_cores: List[List[int]] = []
            if base_core:
                candidate_cores.append(list(base_core))

            if "mc_core_scenarios" in st.session_state:
                mc_top = st.session_state["mc_core_scenarios"].head(5)
                for _, row in mc_top.iterrows():
                    core_vals = []
                    for col in sorted([c for c in row.index if c.startswith("p")]):
                        v = row[col]
                        if not (isinstance(v, float) and math.isnan(v)):
                            core_vals.append(int(v))
                    core_vals = sorted(set(core_vals))
                    if core_vals and core_vals not in candidate_cores:
                        candidate_cores.append(core_vals)

            # Também testa o 1º S6 como referência estrutural
            if s6_df is not None and not s6_df.empty:
                first_s6 = s6_df.iloc[0]
                s6_core = []
                for col in sorted([c for c in s6_df.columns if c.startswith("p")]):
                    v = first_s6[col]
                    if not (isinstance(v, float) and math.isnan(v)):
                        s6_core.append(int(v))
                s6_core = sorted(set(s6_core))
                if s6_core and s6_core not in candidate_cores:
                    candidate_cores.append(s6_core)

            bt_df = backtest_internal_for_cores(df, candidate_cores, window=300)
            st.session_state["internal_backtest"] = bt_df

            if bt_df.empty:
                st.warning("Backtest interno não pôde ser executado com os núcleos atuais.")
            else:
                st.dataframe(
                    bt_df.style.format(
                        {
                            "media_hits": "{:.2f}",
                            "freq_ge3": "{:.2%}",
                            "freq_ge4": "{:.2%}",
                        }
                    ),
                    use_container_width=True,
                )

                # Breve destaque do melhor núcleo no backtest
                best_row = bt_df.iloc[0]
                best_core = [
                    int(best_row[c])
                    for c in sorted([c for c in bt_df.columns if c.startswith("p")])
                    if not (isinstance(best_row[c], float) and math.isnan(best_row[c]))
                ]
                st.success(
                    f"Melhor núcleo no backtest interno: {best_core} "
                    f"(máx {best_row['max_hits']} acertos, média {best_row['media_hits']:.2f})"
                )

            st.divider()

# =========================================================
# FIM DO BLOCO 7 — app.py TURBO
# =========================================================

# =========================================================
# BLOCO 8 — app.py TURBO
# Backtest do Futuro + Leque Final TURBO
# =========================================================

# ---------------------------------------------------------
# Backtest do Futuro — Coerência Retroativa
# ---------------------------------------------------------

def backtest_do_futuro_serie(
    series: List[int],
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    window: int = 600,
    neighbor_k: int = 10,
) -> Dict[str, Any]:
    """
    Backtest do Futuro (coerência retroativa) para UMA série.

    Ideia:
    - verifica se a série "cabe" no passado:
      • não é completamente alienígena em relação às séries reais
      • apresenta algum nível de acertos em janelas históricas
    - verifica consistência estrutural com trechos gêmeos (IDX)
    - gera uma nota de coerência retroativa entre 0 e 1

    Não depende do futuro real, apenas da consistência
    da série quando colocada em contexto histórico.
    """
    base_stats = evaluate_series_against_history(series, df, window=window)

    # 1) estrutura mínima de acertos
    avg_hits = base_stats["media_hits"]
    max_hits = base_stats["max_hits"]
    freq_ge3 = base_stats["freq_ge3"]
    freq_ge4 = base_stats["freq_ge4"]

    # 2) similaridade média com trechos gêmeos
    s_idx = 0.0
    if not df.empty and not idx_df.empty:
        passenger_cols = [c for c in df.columns if c.startswith("p")]
        arr = df[passenger_cols].to_numpy(dtype=float)

        sub = idx_df.head(neighbor_k)
        scores = []
        for _, r in sub.iterrows():
            row_id = r["row_id"]
            base_row = df[df["row_id"] == row_id]
            if base_row.empty:
                continue
            passengers = get_passengers_from_row(base_row.iloc[0])
            scores.append(similarity_structural(series, passengers))
        if scores:
            s_idx = float(np.mean(scores))

    # Combinação de coerência
    # (pesos calibrados de forma simples, mas estáveis)
    coherence_raw = (
        0.35 * s_idx +
        0.25 * (avg_hits / 6.0) +
        0.20 * freq_ge3 +
        0.20 * freq_ge4
    )
    coherence = float(max(0.0, min(1.0, coherence_raw)))

    # Critério de validade:
    # • coerência mínima
    # • pelo menos alguma ocorrência razoável (>=3 acertos)
    is_valid = (coherence >= 0.25) and (freq_ge3 > 0.0 or max_hits >= 3)

    return {
        "series": sorted(set(series)),
        "avg_hits": avg_hits,
        "max_hits": max_hits,
        "freq_ge3": freq_ge3,
        "freq_ge4": freq_ge4,
        "s_idx": s_idx,
        "coherence": coherence,
        "valid": is_valid,
    }


def backtest_do_futuro_conjunto(
    candidate_series: List[List[int]],
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    window: int = 600,
    neighbor_k: int = 10,
) -> pd.DataFrame:
    """
    Aplica o Backtest do Futuro a um conjunto de séries
    retornando DataFrame com métricas e flag de validade.
    """
    if df.empty or not candidate_series:
        return pd.DataFrame()

    rows = []
    for i, s in enumerate(candidate_series, start=1):
        result = backtest_do_futuro_serie(
            s, df, idx_df, window=window, neighbor_k=neighbor_k
        )
        row = {
            "id": i,
            "series": result["series"],
            "avg_hits": result["avg_hits"],
            "max_hits": result["max_hits"],
            "freq_ge3": result["freq_ge3"],
            "freq_ge4": result["freq_ge4"],
            "s_idx": result["s_idx"],
            "coherence": result["coherence"],
            "valid": result["valid"],
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["valid", "coherence", "max_hits", "avg_hits"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Construção do Leque Final TURBO
# ---------------------------------------------------------

def build_final_leque_turbo(
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
    adjusted_core: Dict[str, Any],
    s6_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    internal_bt_df: pd.DataFrame,
    max_premium: int = 10,
    max_estruturais: int = 12,
    max_cobertura: int = 16,
) -> Dict[str, Any]:
    """
    Constrói o Leque Final TURBO:

    - Núcleo TURBO
    - Séries Premium
    - Séries Estruturais
    - Séries de Cobertura
    - Séries S6 (já geradas no BLOCO 6)
    - Ensamble final

    Todas as séries passam pelo Backtest do Futuro.
    """

    if df.empty or adjusted_core is None:
        return {}

    # 1) Núcleo base (ponto de partida)
    base_core = adjusted_core.get("after_hla") or adjusted_core.get("after_ica") or []
    base_core = sorted(set(base_core))

    # Se ainda assim estiver vazio, tenta IPO
    if not base_core:
        ipo_out = st.session_state.get("ipo_core", {})
        base_core = ipo_out.get("structural_core", []) if ipo_out else []
        base_core = sorted(set(base_core))

    if not base_core:
        return {}

    # 2) Núcleo "melhor no backtest interno", se disponível
    best_bt_core = None
    if internal_bt_df is not None and not internal_bt_df.empty:
        best_row = internal_bt_df.iloc[0]
        temp_core = []
        for col in sorted([c for c in internal_bt_df.columns if c.startswith("p")]):
            v = best_row[col]
            if not (isinstance(v, float) and math.isnan(v)):
                temp_core.append(int(v))
        temp_core = sorted(set(temp_core))
        if temp_core:
            best_bt_core = temp_core

    # 3) Séries S6 profundas
    s6_candidates: List[List[int]] = []
    if s6_df is not None and not s6_df.empty:
        for _, row in s6_df.iterrows():
            s = []
            for col in sorted([c for c in s6_df.columns if c.startswith("p")]):
                v = row[col]
                if not (isinstance(v, float) and math.isnan(v)):
                    s.append(int(v))
            s = sorted(set(s))
            if len(s) >= 3:
                s6_candidates.append(s)

    # 4) Núcleos Monte Carlo principais
    mc_candidates: List[List[int]] = []
    if mc_df is not None and not mc_df.empty:
        mc_top = mc_df.head(15)
        for _, row in mc_top.iterrows():
            s = []
            for col in sorted([c for c in mc_df.columns if c.startswith("p")]):
                v = row[col]
                if not (isinstance(v, float) and math.isnan(v)):
                    s.append(int(v))
            s = sorted(set(s))
            if len(s) >= 3:
                mc_candidates.append(s)

    # 5) Conjunto bruto de candidatos
    candidate_series: List[List[int]] = []

    if base_core:
        candidate_series.append(base_core)
    if best_bt_core and best_bt_core != base_core:
        candidate_series.append(best_bt_core)

    # adiciona S6
    for s in s6_candidates:
        if s not in candidate_series:
            candidate_series.append(s)

    # adiciona MC
    for s in mc_candidates:
        if s not in candidate_series:
            candidate_series.append(s)

    # normaliza tudo para exatamente 6 passageiros
    normalized_candidates: List[List[int]] = []
    for s in candidate_series:
        s = sorted(set(s))
        if len(s) < 6:
            # completa com números próximos à média
            mean_val = int(sum(s) / len(s))
            pool = list(range(max(1, mean_val - 10), min(80, mean_val + 10) + 1))
            rng = np.random.default_rng()
            while len(s) < 6 and pool:
                cand = int(rng.choice(pool))
                if cand not in s:
                    s.append(cand)
        s = sorted(s)[:6]
        if s not in normalized_candidates:
            normalized_candidates.append(s)

    if not normalized_candidates:
        return {}

    # 6) Backtest do Futuro em todas as séries candidatas
    btf_df = backtest_do_futuro_conjunto(
        normalized_candidates,
        df,
        idx_df,
        window=600,
        neighbor_k=10,
    )

    if btf_df.empty:
        return {}

    # Apenas séries válidas
    valid_df = btf_df[btf_df["valid"] == True].copy()
    if valid_df.empty:
        # Em último caso, mantém tudo como alerta, mesmo sem validade forte
        valid_df = btf_df.copy()

    # 7) Classificação em categorias

    # Identifica Núcleo TURBO:
    # • prioridade: série igual ao best_bt_core
    # • senão: série com maior coerência
    nucleo_series = None

    def series_equal(a: List[int], b: List[int]) -> bool:
        return sorted(set(a)) == sorted(set(b))

    if best_bt_core:
        for _, row in valid_df.iterrows():
            s = row["series"]
            if series_equal(s, best_bt_core):
                nucleo_series = s
                break

    if nucleo_series is None:
        nucleo_series = valid_df.iloc[0]["series"]

    # Marca Núcleo TURBO
    valid_df["category"] = "Cobertura"  # default

    # Semelhantes ao núcleo (premium/estruturais) por similaridade estrutural
    sim_values = []
    for i, row in valid_df.iterrows():
        s = row["series"]
        sim = similarity_structural(nucleo_series, s)
        sim_values.append(sim)
    valid_df["sim_nucleo"] = sim_values

    # ordena por coerência + similaridade
    valid_df = valid_df.sort_values(
        ["coherence", "sim_nucleo", "max_hits", "avg_hits"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # Define categorias:
    # • Núcleo TURBO: a série principal
    # • Premium: alta coerência e alta similaridade
    # • Estruturais: coerência boa e similaridade média
    # • Cobertura: o restante

    categorias = []
    for i, row in valid_df.iterrows():
        s = row["series"]
        if series_equal(s, nucleo_series):
            categorias.append("Núcleo TURBO")
        else:
            coh = row["coherence"]
            sim = row["sim_nucleo"]
            if coh >= 0.55 and sim >= 0.65:
                categorias.append("Premium")
            elif coh >= 0.40 and sim >= 0.40:
                categorias.append("Estrutural")
            else:
                categorias.append("Cobertura")
    valid_df["category"] = categorias

    # Limites de quantidade por categoria
    def limit_category(df_cat: pd.DataFrame, max_n: int) -> pd.DataFrame:
        return df_cat.head(max_n).reset_index(drop=True)

    nucleo_df = valid_df[valid_df["category"] == "Núcleo TURBO"].copy()
    premium_df = limit_category(valid_df[valid_df["category"] == "Premium"], max_premium)
    estrutural_df = limit_category(
        valid_df[valid_df["category"] == "Estrutural"], max_estruturais
    )
    cobertura_df = limit_category(
        valid_df[valid_df["category"] == "Cobertura"], max_cobertura
    )

    # Ajusta ranks internos
    for df_cat in (nucleo_df, premium_df, estrutural_df, cobertura_df):
        if not df_cat.empty:
            df_cat["rank_cat"] = np.arange(1, len(df_cat) + 1)

    # 8) Ensamble final:
    # média ponderada das séries Premium + Núcleo TURBO,
    # ponderando pela coerência retroativa.

    ensamble_series: List[int] = []
    if not nucleo_df.empty:
        base = nucleo_df.iloc[0]
        nucleo = base["series"]
        weight_nucleo = base["coherence"]
    else:
        nucleo = nucleo_series
        weight_nucleo = 0.8

    # agrega Premium
    weighted_counts = Counter()
    total_weight = 0.0

    # Núcleo com peso extra
    for n in nucleo:
        weighted_counts[n] += weight_nucleo
    total_weight += weight_nucleo

    for _, row in premium_df.iterrows():
        s = row["series"]
        w = float(row["coherence"])
        for n in s:
            weighted_counts[n] += w
        total_weight += w

    if weighted_counts and total_weight > 0:
        scoring = [
            (n, float(w) / total_weight) for n, w in weighted_counts.items()
        ]
        scoring_sorted = sorted(scoring, key=lambda kv: (-kv[1], kv[0]))
        ensamble_series = [n for n, _ in scoring_sorted[:6]]
        ensamble_series = sorted(set(ensamble_series))[:6]
    else:
        ensamble_series = nucleo

    # 9) Expectativa de acertos por série (usa média de hits do Backtest do Futuro)
    def attach_expectation(df_cat: pd.DataFrame) -> pd.DataFrame:
        if df_cat.empty:
            return df_cat
        df_cat = df_cat.copy()
        df_cat["expect_acertos"] = df_cat["avg_hits"]
        return df_cat

    nucleo_df = attach_expectation(nucleo_df)
    premium_df = attach_expectation(premium_df)
    estrutural_df = attach_expectation(estrutural_df)
    cobertura_df = attach_expectation(cobertura_df)

    return {
        "nucleo": nucleo_df,
        "premium": premium_df,
        "estruturais": estrutural_df,
        "cobertura": cobertura_df,
        "s6": s6_df,
        "ensamble": ensamble_series,
        "btf_raw": valid_df,
    }


# ---------------------------------------------------------
# Painel Streamlit — Leque Final TURBO
# ---------------------------------------------------------

if not df.empty:
    st.subheader("🚀 Leque Final V13.8-TURBO")

    idx_res = st.session_state.get("idx_result", pd.DataFrame())
    adjusted_core = st.session_state.get("adjusted_core", {})
    s6_df = st.session_state.get("s6_series", pd.DataFrame())
    mc_df = st.session_state.get("mc_core_scenarios", pd.DataFrame())
    internal_bt_df = st.session_state.get("internal_backtest", pd.DataFrame())

    if idx_res.empty or not adjusted_core:
        st.info("É necessário ter IDX, Núcleo Ajustado, S6 e Monte Carlo para montar o Leque TURBO.")
    else:
        leque = build_final_leque_turbo(
            df,
            idx_res,
            adjusted_core,
            s6_df,
            mc_df,
            internal_bt_df,
        )
        st.session_state["leque_turbo"] = leque

        if not leque:
            st.warning("Não foi possível construir o Leque TURBO com os dados atuais.")
        else:
            nucleo_df = leque["nucleo"]
            premium_df = leque["premium"]
            estrutural_df = leque["estruturais"]
            cobertura_df = leque["cobertura"]
            s6_final = leque["s6"]
            ensamble_series = leque["ensamble"]

            # Núcleo TURBO
            st.markdown("### ⭐ Núcleo TURBO")
            if nucleo_df is None or nucleo_df.empty:
                st.warning("Núcleo TURBO não identificado com alta confiança. Usando núcleo ajustado como referência.")
                st.write(sorted(st.session_state.get("adjusted_core", {}).get("after_hla", [])))
            else:
                row = nucleo_df.iloc[0]
                st.write(f"**Série Núcleo:** {row['series']}")
                st.write(f"**Expectativa de acertos:** {row['expect_acertos']:.2f}")
                st.write(f"**Coerência retroativa:** {row['coherence']:.3f}")

            st.markdown("---")

            # Ensamble
            st.markdown("### 🔁 Ensamble TURBO (fusão Núcleo + Premium)")
            st.write(f"**Série Ensamble:** {ensamble_series}")

            st.markdown("---")

            # Premium
            st.markdown("### 💎 Séries Premium")
            if premium_df is None or premium_df.empty:
                st.write("Nenhuma série Premium qualificada.")
            else:
                df_show = premium_df[["rank_cat", "series", "expect_acertos", "coherence"]]
                st.dataframe(
                    df_show.style.format({"expect_acertos": "{:.2f}", "coherence": "{:.3f}"}),
                    use_container_width=True,
                )

            # Estruturais
            st.markdown("### 🧱 Séries Estruturais")
            if estrutural_df is None or estrutural_df.empty:
                st.write("Nenhuma série Estrutural qualificada.")
            else:
                df_show = estrutural_df[["rank_cat", "series", "expect_acertos", "coherence"]]
                st.dataframe(
                    df_show.style.format({"expect_acertos": "{:.2f}", "coherence": "{:.3f}"}),
                    use_container_width=True,
                )

            # Cobertura
            st.markdown("### 🌐 Séries de Cobertura")
            if cobertura_df is None or cobertura_df.empty:
                st.write("Nenhuma série de Cobertura selecionada.")
            else:
                df_show = cobertura_df[["rank_cat", "series", "expect_acertos", "coherence"]]
                st.dataframe(
                    df_show.style.format({"expect_acertos": "{:.2f}", "coherence": "{:.3f}"}),
                    use_container_width=True,
                )

            # S6 (como bloco explícito, pois já é parte do TURBO)
            st.markdown("### 🎯 Séries S6 Profundas (referência)")
            if s6_final is None or s6_final.empty:
                st.write("Nenhuma série S6 disponível.")
            else:
                st.dataframe(
                    s6_final.style.format({"score": "{:.3f}"}),
                    use_container_width=True,
                )

            st.success("Leque Final TURBO construído e validado com Backtest do Futuro.")
            st.divider()

# =========================================================
# FIM DO BLOCO 8 — app.py TURBO
# =========================================================


# =========================================================
# BLOCO 9 — app.py TURBO
# Sidebar + Controle de Saída + Acertos (esperados) integrados
# =========================================================

# ---------------------------------------------------------
# Sidebar — Controle do Leque TURBO
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Controle do Leque TURBO")

output_mode = st.sidebar.radio(
    "Modo de geração do Leque:",
    options=["Automático (por regime)", "Quantidade fixa", "Confiabilidade mínima"],
    index=0,
    help=(
        "Automático: quantidade de séries definida pelo regime.\n"
        "Quantidade fixa: define o total de séries desejadas.\n"
        "Confiabilidade mínima: filtra séries por coerência mínima."
    ),
)

n_series_fixed = st.sidebar.slider(
    "Quantidade total de séries (se modo for 'Quantidade fixa')",
    min_value=5,
    max_value=25,
    value=12,
)

min_conf_pct = st.sidebar.slider(
    "Confiabilidade mínima (%) (se modo for 'Confiabilidade mínima')",
    min_value=30,
    max_value=85,
    value=55,
)


# ---------------------------------------------------------
# Utilitários — cálculo de acertos (esperados) em inteiro
# ---------------------------------------------------------

def expected_hits_from_stats(
    avg_hits: float,
    max_hits: int,
    freq_ge3: float,
    freq_ge4: float,
    category: str,
) -> int:
    """
    Converte estatísticas de backtest em um número inteiro de
    'Acertos (esperados)' entre 1 e 6.
    """
    # S6 sempre mira 6
    if category.lower().startswith("s6"):
        return 6

    # Proteção
    if avg_hits is None or np.isnan(avg_hits):
        avg_hits = 0.0

    # Faixas principais baseadas em média, moduladas por máximos
    if avg_hits >= 5.4 or (avg_hits >= 5.0 and max_hits >= 6):
        return 6
    if avg_hits >= 4.5 or (avg_hits >= 4.0 and freq_ge4 > 0.25):
        return 5
    if avg_hits >= 3.5:
        return 4
    if avg_hits >= 2.5:
        return 3
    if avg_hits >= 1.5:
        return 2
    return 1


def join_btf_stats_on_series(
    df_cat: pd.DataFrame,
    btf_df: pd.DataFrame,
    category_name: str,
) -> pd.DataFrame:
    """
    Associa as estatísticas de Backtest do Futuro (btf_df) às séries
    de um DataFrame de categoria (df_cat), com base na lista de números
    da série.

    Adiciona colunas:
    - avg_hits
    - max_hits
    - freq_ge3
    - freq_ge4
    - s_idx
    - coherence
    - expected_hits (inteiro)
    - category_name (texto)
    """
    if df_cat is None or df_cat.empty or btf_df is None or btf_df.empty:
        return pd.DataFrame()

    # Mapa {tuple(sorted(series)): stats_row}
    btf_map: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for _, row in btf_df.iterrows():
        s = row["series"]
        key = tuple(sorted(set(s)))
        btf_map[key] = {
            "avg_hits": float(row["avg_hits"]),
            "max_hits": int(row["max_hits"]),
            "freq_ge3": float(row["freq_ge3"]),
            "freq_ge4": float(row["freq_ge4"]),
            "s_idx": float(row["s_idx"]),
            "coherence": float(row["coherence"]),
        }

    rows_out = []
    for _, row in df_cat.iterrows():
        s = row["series"]
        key = tuple(sorted(set(s)))
        stats = btf_map.get(
            key,
            {
                "avg_hits": 0.0,
                "max_hits": 0,
                "freq_ge3": 0.0,
                "freq_ge4": 0.0,
                "s_idx": 0.0,
                "coherence": float(row.get("coherence", 0.0)),
            },
        )

        avg_hits = stats["avg_hits"]
        max_hits = stats["max_hits"]
        freq_ge3 = stats["freq_ge3"]
        freq_ge4 = stats["freq_ge4"]
        coherence = stats["coherence"]

        expected_int = expected_hits_from_stats(
            avg_hits=avg_hits,
            max_hits=max_hits,
            freq_ge3=freq_ge3,
            freq_ge4=freq_ge4,
            category=category_name,
        )

        rows_out.append(
            {
                "series": s,
                "category": category_name,
                "avg_hits": avg_hits,
                "max_hits": max_hits,
                "freq_ge3": freq_ge3,
                "freq_ge4": freq_ge4,
                "coherence": coherence,
                "expected_hits": expected_int,
            }
        )

    return pd.DataFrame(rows_out)


def build_flat_series_table(leque: Dict[str, Any]) -> pd.DataFrame:
    """
    Constrói uma tabela plana com TODAS as séries (Núcleo, Premium,
    Estruturais, Cobertura), já com:

    - category
    - coherence
    - expected_hits (Acertos (esperados))
    """
    if not leque:
        return pd.DataFrame()

    btf_df = leque.get("btf_raw", pd.DataFrame())
    nucleo_df = leque.get("nucleo", pd.DataFrame())
    premium_df = leque.get("premium", pd.DataFrame())
    estrutural_df = leque.get("estruturais", pd.DataFrame())
    cobertura_df = leque.get("cobertura", pd.DataFrame())

    parts = []

    if nucleo_df is not None and not nucleo_df.empty:
        parts.append(join_btf_stats_on_series(nucleo_df, btf_df, "Núcleo TURBO"))
    if premium_df is not None and not premium_df.empty:
        parts.append(join_btf_stats_on_series(premium_df, btf_df, "Premium"))
    if estrutural_df is not None and not estrutural_df.empty:
        parts.append(join_btf_stats_on_series(estrutural_df, btf_df, "Estrutural"))
    if cobertura_df is not None and not cobertura_df.empty:
        parts.append(join_btf_stats_on_series(cobertura_df, btf_df, "Cobertura"))

    if not parts:
        return pd.DataFrame()

    flat = pd.concat(parts, ignore_index=True)

    # Remove duplicadas por série + categoria (se houver)
    flat["series_key"] = flat["series"].apply(lambda s: tuple(sorted(set(s))))
    flat = (
        flat.sort_values(
            ["category", "coherence", "avg_hits"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["series_key", "category"])
        .reset_index(drop=True)
    )
    return flat


def limit_by_mode(
    flat_df: pd.DataFrame,
    regime: Optional[RegimeState],
    mode: str,
    n_series_fixed: int,
    min_conf_pct: int,
) -> pd.DataFrame:
    """
    Aplica o modo de controle:

    - Automático (por regime)
    - Quantidade fixa
    - Confiabilidade mínima
    """
    if flat_df.empty:
        return flat_df

    # Ordenação base: Núcleo > Premium > Estrutural > Cobertura
    cat_order = {"Núcleo TURBO": 0, "Premium": 1, "Estrutural": 2, "Cobertura": 3}
    flat_df = flat_df.copy()
    flat_df["cat_rank"] = flat_df["category"].map(cat_order).fillna(99).astype(int)

    flat_df = flat_df.sort_values(
        ["cat_rank", "coherence", "avg_hits"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    if mode.startswith("Automático"):
        # Define alvo total pela estrada
        if regime is None:
            target_total = 10
        else:
            if regime.nome == "Resiliente":
                target_total = 8
            elif regime.nome == "Intermediário":
                target_total = 12
            elif regime.nome == "Turbulento":
                target_total = 15
            else:
                target_total = 10

        return flat_df.head(target_total).reset_index(drop=True)

    if mode.startswith("Quantidade"):
        return flat_df.head(n_series_fixed).reset_index(drop=True)

    if mode.startswith("Confiabilidade"):
        thr = float(min_conf_pct) / 100.0
        df_f = flat_df[flat_df["coherence"] >= thr].copy()
        if df_f.empty:
            # fallback: nenhuma bate o threshold, retorna tudo (mas sinaliza)
            return flat_df
        return df_f.reset_index(drop=True)

    return flat_df


def evaluate_ensamble_series(
    ensamble: List[int],
    df: pd.DataFrame,
    idx_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calcula coerência e acertos (esperados) para a série Ensamble.
    """
    if not ensamble:
        return {}

    # Usa Backtest do Futuro da própria série
    res = backtest_do_futuro_serie(
        ensamble,
        df,
        idx_df,
        window=600,
        neighbor_k=10,
    )
    expected_int = expected_hits_from_stats(
        avg_hits=res["avg_hits"],
        max_hits=res["max_hits"],
        freq_ge3=res["freq_ge3"],
        freq_ge4=res["freq_ge4"],
        category="Ensamble",
    )
    res["expected_hits"] = expected_int
    return res


# ---------------------------------------------------------
# Painel — Saída Final Controlada (com Acertos (esperados))
# ---------------------------------------------------------

if not df.empty:
    st.subheader("📦 Saída Final Controlada — V13.8-TURBO")

    leque = st.session_state.get("leque_turbo", {})
    regime_state = st.session_state.get("regime_state", None)
    idx_res = st.session_state.get("idx_result", pd.DataFrame())

    if not leque:
        st.info("O Leque TURBO ainda não foi construído (ver painel anterior).")
    else:
        flat_df = build_flat_series_table(leque)

        if flat_df.empty:
            st.warning("Não há séries suficientes para montar a saída controlada.")
        else:
            # Aplica modo de controle
            controlled_df = limit_by_mode(
                flat_df,
                regime_state,
                output_mode,
                n_series_fixed,
                min_conf_pct,
            )

            if controlled_df.empty:
                st.warning("Nenhuma série qualificada após aplicar o modo de controle.")
            else:
                # Monta uma tabela legível com os campos principais
                display_rows = []
                for i, row in controlled_df.iterrows():
                    display_rows.append(
                        {
                            "Rank": i + 1,
                            "Categoria": row["category"],
                            "Série": " ".join(str(x) for x in row["series"]),
                            "Confiabilidade": f"{row['coherence']*100:.1f}%",
                            "Acertos (esperados)": int(row["expected_hits"]),
                        }
                    )

                df_display = pd.DataFrame(display_rows)
                st.dataframe(df_display, use_container_width=True)

                st.caption(
                    "Todas as séries acima já passaram por: IPF, IPO, ASB, ADN, ICA, "
                    "HLA, S6 (quando aplicável), Monte Carlo, Backtest Interno e Backtest do Futuro."
                )

            st.markdown("---")

            # Ensamble TURBO com Acertos (esperados)
            ensamble_series = leque.get("ensamble", [])
            if ensamble_series:
                st.markdown("### 🔁 Ensamble TURBO — com Acertos (esperados)")

                ensamble_stats = evaluate_ensamble_series(
                    ensamble_series,
                    df,
                    idx_res,
                )

                if ensamble_stats:
                    st.write(f"**Série Ensamble:** {' '.join(str(x) for x in ensamble_series)}")
                    st.write(
                        f"**Confiabilidade (coerência retroativa):** "
                        f"{ensamble_stats['coherence']*100:.1f}%"
                    )
                    st.write(
                        f"**Acertos (esperados):** {ensamble_stats['expected_hits']}"
                    )
                else:
                    st.write("Não foi possível calcular estatísticas para o Ensamble.")
            else:
                st.markdown("### 🔁 Ensamble TURBO")
                st.write("Nenhum ensamble disponível.")

        st.divider()

# =========================================================
# FIM DO BLOCO 9 — app.py TURBO
# =========================================================

# =========================================================
# BLOCO 10 — app.py TURBO
# NAVEGAÇÃO MODULAR + LISTA PURA NUMERADA (Somente Séries Controladas)
# =========================================================

# ---------------------------------------------------------
# MENU DE NAVEGAÇÃO — PAINÉIS DO V13.8-TURBO
# ---------------------------------------------------------

st.sidebar.markdown("## 📂 Navegação")
painel = st.sidebar.radio(
    "Escolha o painel:",
    [
        "Histórico",
        "Estado Atual",
        "IDX Avançado",
        "Núcleo IPF / IPO",
        "Ajustes (ASB / ADN / ICA / HLA)",
        "Dependências Ocultas",
        "S6 Profundo",
        "Monte Carlo Profundo",
        "Backtest Interno",
        "Backtest do Futuro",
        "Leque TURBO",
        "Saída Final Controlada",   # 🔥 onde estará a lista pura
    ],
    index=0,
)

# =========================================================
# A partir daqui, mostramos APENAS o painel selecionado
# =========================================================

# ---------------------------------------------------------
# PAINEL — Histórico
# ---------------------------------------------------------
if painel == "Histórico":
    st.subheader("📘 Histórico Carregado")
    if df.empty:
        st.warning("Nenhum histórico foi carregado.")
    else:
        st.write("Quantidade de séries:", len(df))
        st.dataframe(df.tail(20), use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — Estado Atual
# ---------------------------------------------------------
if painel == "Estado Atual":
    st.subheader("🌡️ Estado Atual da Estrada")
    regime_state = st.session_state.get("regime_state", None)

    if regime_state is None:
        st.info("O estado ainda não foi calculado.")
    else:
        st.write("**Regime:**", regime_state.nome)
        st.write("**Dispersão:**", regime_state.dispersao)
        st.write("**Amplitude:**", regime_state.amplitude)
        st.write("**Vibração:**", regime_state.vibracao)
        st.write("**Pares Ativos:**", regime_state.pares)
    st.stop()

# ---------------------------------------------------------
# PAINEL — IDX Avançado
# ---------------------------------------------------------
if painel == "IDX Avançado":
    st.subheader("🔎 IDX Avançado — Trechos Semelhantes")
    idx_df = st.session_state.get("idx_result", pd.DataFrame())

    if idx_df.empty:
        st.info("IDX ainda não foi calculado.")
    else:
        st.dataframe(idx_df, use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — IPF / IPO
# ---------------------------------------------------------
if painel == "Núcleo IPF / IPO":
    st.subheader("🧱 Núcleo Estrutural — IPF / IPO")
    nucleo_ipf = st.session_state.get("nucleo_ipf", None)
    nucleo_ipo = st.session_state.get("nucleo_ipo", None)

    if nucleo_ipf:
        st.markdown("### IPF (Núcleo Pré-Bruto)")
        st.write(nucleo_ipf)

    if nucleo_ipo:
        st.markdown("### IPO (Núcleo Otimizado)")
        st.write(nucleo_ipo)

    if not nucleo_ipf and not nucleo_ipo:
        st.info("IPF/IPO ainda não foi calculado.")
    st.stop()

# ---------------------------------------------------------
# PAINEL — Ajustes Profundos
# ---------------------------------------------------------
if painel == "Ajustes (ASB / ADN / ICA / HLA)":
    st.subheader("🛠️ Ajustes Profundos — ASB / ADN / ICA / HLA")
    ajustes_log = st.session_state.get("ajustes_log", [])

    if ajustes_log:
        for bloco in ajustes_log:
            st.markdown(f"### {bloco['nome']}")
            st.write(bloco["dados"])
    else:
        st.info("Ainda não foram registrados ajustes.")
    st.stop()

# ---------------------------------------------------------
# PAINEL — Dependências Ocultas
# ---------------------------------------------------------
if painel == "Dependências Ocultas":
    st.subheader("🧬 Dependências Ocultas")
    deps = st.session_state.get("dependencias", None)

    if deps:
        st.write(deps)
    else:
        st.info("Nenhuma dependência encontrada.")
    st.stop()

# ---------------------------------------------------------
# PAINEL — S6 Profundo
# ---------------------------------------------------------
if painel == "S6 Profundo":
    st.subheader("🎯 S6 Profundo — Séries com Convergência Máxima")
    s6_df = st.session_state.get("s6_df", pd.DataFrame())

    if s6_df.empty:
        st.info("Nenhuma série S6 encontrada.")
    else:
        st.dataframe(s6_df, use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — Monte Carlo Profundo
# ---------------------------------------------------------
if painel == "Monte Carlo Profundo":
    st.subheader("🌪️ Monte Carlo Profundo — Perturbações Válidas")
    mc_df = st.session_state.get("mc_df", pd.DataFrame())

    if mc_df.empty:
        st.info("Monte Carlo ainda não foi executado.")
    else:
        st.dataframe(mc_df, use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — Backtest Interno
# ---------------------------------------------------------
if painel == "Backtest Interno":
    st.subheader("📉 Backtest Interno")
    bti = st.session_state.get("backtest_interno", pd.DataFrame())

    if bti.empty:
        st.info("Backtest Interno não disponível.")
    else:
        st.dataframe(bti, use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — Backtest do Futuro
# ---------------------------------------------------------
if painel == "Backtest do Futuro":
    st.subheader("🔮 Backtest do Futuro — Coerência Retroativa")
    btf = st.session_state.get("btf_raw", pd.DataFrame())

    if btf.empty:
        st.info("Backtest do Futuro ainda não foi realizado.")
    else:
        st.dataframe(btf, use_container_width=True)
    st.stop()

# ---------------------------------------------------------
# PAINEL — Leque TURBO
# ---------------------------------------------------------
if painel == "Leque TURBO":
    st.subheader("🚀 LEQUE TURBO — Séries Preditivas Completas")

    leque = st.session_state.get("leque_turbo", {})

    if not leque:
        st.info("O Leque TURBO ainda não foi construído.")
    else:
        st.write("### Núcleo TURBO")
        st.dataframe(leque.get("nucleo", pd.DataFrame()), use_container_width=True)

        st.write("### Premium")
        st.dataframe(leque.get("premium", pd.DataFrame()), use_container_width=True)

        st.write("### Estruturais")
        st.dataframe(leque.get("estruturais", pd.DataFrame()), use_container_width=True)

        st.write("### Cobertura")
        st.dataframe(leque.get("cobertura", pd.DataFrame()), use_container_width=True)

        st.write("### S6 Profundo")
        st.dataframe(leque.get("s6", pd.DataFrame()), use_container_width=True)

        st.write("### Ensamble TURBO")
        st.write(leque.get("ensamble", []))

    st.stop()

# ---------------------------------------------------------
# PAINEL — Saída Final Controlada
# (Tabela + Ensamble + Lista Pura numerada)
# ---------------------------------------------------------
if painel == "Saída Final Controlada":

    st.subheader("🎯 SAÍDA FINAL CONTROLADA — V13.8-TURBO")

    # Usa a tabela do BLOCO 9
    leque = st.session_state.get("leque_turbo", {})
    regime_state = st.session_state.get("regime_state", None)
    idx_df = st.session_state.get("idx_result", pd.DataFrame())

    if not leque:
        st.info("O Leque TURBO ainda não foi construído.")
        st.stop()

    # Reconstrói tabela plana com acertos esperados
    flat_df = build_flat_series_table(leque)

    if flat_df.empty:
        st.warning("Não há séries suficientes para montar a saída final.")
        st.stop()

    # Aplica modo de controle (do BLOCO 9)
    controlled_df = limit_by_mode(
        flat_df,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct,
    )

    if controlled_df.empty:
        st.warning("Nenhuma série passou pelo filtro escolhido.")
        st.stop()

    # -----------------------------------------------------
    # 1) Tabela organizada
    # -----------------------------------------------------
    display_rows = []
    for i, row in controlled_df.iterrows():
        display_rows.append(
            {
                "Rank": i + 1,
                "Categoria": row["category"],
                "Série": " ".join(str(x) for x in row["series"]),
                "Confiabilidade": f"{row['coherence']*100:.1f}%",
                "Acertos (esperados)": int(row["expected_hits"]),
            }
        )
    df_display = pd.DataFrame(display_rows)
    st.write("### 📊 Séries Selecionadas")
    st.dataframe(df_display, use_container_width=True)

    # -----------------------------------------------------
    # 2) Ensamble TURBO
    # -----------------------------------------------------
    st.write("---")
    st.write("### 🔁 Ensamble TURBO")

    ens_series = leque.get("ensamble", [])
    ens_stats = evaluate_ensamble_series(ens_series, df, idx_df)

    if ens_stats:
        st.write("**Série Ensamble:**", " ".join(str(x) for x in ens_series))
        st.write(f"**Confiabilidade:** {ens_stats['coherence']*100:.1f}%")
        st.write(f"**Acertos (esperados):** {ens_stats['expected_hits']}")
    else:
        st.write("Nenhum ensamble disponível.")

    # -----------------------------------------------------
    # 3) LISTA PURA NUMERADA — Somente Séries Controladas
    # -----------------------------------------------------
    st.write("---")
    st.markdown("### 📄 Lista Pura — Séries Filtradas (Para Copiar)")

    for i, row in controlled_df.iterrows():
        s = " ".join(str(x) for x in row["series"])
        st.write(f"{i+1}) {s}")

    st.stop()


# =========================================================
# FIM DO BLOCO 10
# =========================================================

# =========================================================
# BLOCO 11 — Painel de Logs Técnicos Internos
# =========================================================

# ---------------------------------------------------------
# Função auxiliar para registrar logs internos de qualquer etapa
# ---------------------------------------------------------

def add_log(etapa: str, dados: Any):
    """
    Armazena logs técnicos do pipeline TURBO.
    Cada log contém:
    - etapa: nome da etapa (IDX, IPF, IPO, ICA, S6 etc.)
    - dados: conteúdo técnico estruturado
    """
    if "logs_tecnicos" not in st.session_state:
        st.session_state["logs_tecnicos"] = []

    st.session_state["logs_tecnicos"].append({
        "etapa": etapa,
        "dados": dados
    })


# ---------------------------------------------------------
# PAINEL DE NAVEGAÇÃO (adicionar item)
# ---------------------------------------------------------

# Adiciona o painel "Logs Técnicos" ao menu, APÓS BLOCO 10
# (Somente adicionar o nome à lista de opções)

# NO BLOCO 10, SUBSTITUIR:
# 
# painel = st.sidebar.radio(
#     "Escolha o painel:",
#     [
#         ...
#         "Saída Final Controlada",
#     ],
# )

# POR:
#
# painel = st.sidebar.radio(
#     "Escolha o painel:",
#     [
#         "Histórico",
#         "Estado Atual",
#         "IDX Avançado",
#         "Núcleo IPF / IPO",
#         "Ajustes (ASB / ADN / ICA / HLA)",
#         "Dependências Ocultas",
#         "S6 Profundo",
#         "Monte Carlo Profundo",
#         "Backtest Interno",
#         "Backtest do Futuro",
#         "Leque TURBO",
#         "Saída Final Controlada",
#         "Logs Técnicos",  # <-- ADICIONADO AQUI
#     ],
#     index=0,
# )


# ---------------------------------------------------------
# PAINEL — Logs Técnicos
# ---------------------------------------------------------
if painel == "Logs Técnicos":
    
    st.subheader("🧰 Logs Técnicos — V13.8-TURBO")

    logs = st.session_state.get("logs_tecnicos", [])

    if not logs:
        st.info("Nenhum log técnico registrado ainda.")
        st.stop()

    # Exibe cada log como um bloco collapsible
    for registro in logs:
        with st.expander(f"Etapa: {registro['etapa']}"):
            st.write(registro["dados"])

    st.stop()

# =========================================================
# BLOCO 12 — Diagnóstico Profundo (Gráficos e Análises Estruturais)
# =========================================================

# Este bloco adiciona um painel completo de diagnóstico:
# - curva de dispersão
# - curva de amplitude
# - vibração
# - heatmap de similaridade IDX
# - distribuição de acertos do Backtest Interno
# - distribuição de coerência do Backtest do Futuro
# - convergência S6
# - estabilidade da estrada (índice composto)
#
# Tudo opcionalmente exibido via navegação modular.


import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Funções auxiliares de diagnóstico
# ---------------------------------------------------------

def plot_line(data, title, ylabel):
    """Gera gráfico de linha simples."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(data, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Índice")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)


def plot_hist(data, title, xlabel):
    """Histograma simples."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequência")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)


def plot_heatmap(df_matrix, title):
    """Heatmap de matriz (ex: similaridade IDX)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_matrix, cmap="viridis", linewidths=.5, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def calcular_indice_estabilidade(regime_state):
    """
    Índice composto de estabilidade da estrada:
    Combina dispersão, amplitude, vibração e pares.
    Valores aproximados:
    >0.75 → Estável
    0.5–0.75 → Intermediário
    <0.5 → Instável / Turbulento
    """
    if not regime_state:
        return None

    score = 0.0

    # pesos heurísticos derivados do comportamento da estrada
    disp_peso = max(0, 1 - regime_state.dispersao / 40)
    amp_peso = max(0, 1 - regime_state.amplitude / 60)
    vib_peso = max(0, 1 - regime_state.vibracao / 30)
    par_peso = min(1.0, len(regime_state.pares) / 10)

    score = (disp_peso + amp_peso + vib_peso + par_peso) / 4
    return score


# ---------------------------------------------------------
# Adicionar item de navegação: BLOCO 12
# (Adicionar ao menu do BLOCO 10)
# ---------------------------------------------------------

# Em BLOCO 10, adicionar "Diagnóstico Profundo" à lista:
#
#     "Leque TURBO",
#     "Saída Final Controlada",
#     "Logs Técnicos",
#     "Diagnóstico Profundo",  # <-- ADICIONAR
#


# ---------------------------------------------------------
# PAINEL — Diagnóstico Profundo
# ---------------------------------------------------------
if painel == "Diagnóstico Profundo":

    st.subheader("🧭 Diagnóstico Profundo — V13.8-TURBO")

    if df.empty:
        st.warning("Carregue um histórico para visualizar o diagnóstico.")
        st.stop()

    # =====================================================
    # Seção 1 — Curvas Históricas Básicas
    # =====================================================
    st.markdown("### 📈 Curvas Estruturais da Estrada")

    dispersoes = df.apply(lambda row: np.std(list(row[:-1])), axis=1)
    amplitudes = df.apply(lambda row: max(row[:-1]) - min(row[:-1]), axis=1)

    plot_line(dispersoes, "Dispersão ao longo da Estrada", "Dispersão")
    plot_line(amplitudes, "Amplitude ao longo da Estrada", "Amplitude")

    # =====================================================
    # Seção 2 — Vibração
    # =====================================================
    st.markdown("### 🌐 Vibração Estrutural")
    vib = np.abs(dispersoes.diff().fillna(0))
    plot_line(vib, "Vibração Estrutural", "Vibração")

    # =====================================================
    # Seção 3 — Similaridade IDX (Heatmap)
    # =====================================================
    st.markdown("### 🔎 Heatmap de Similaridade (IDX)")
    idx_df = st.session_state.get("idx_result", pd.DataFrame())

    if not idx_df.empty and "similarity_vector" in idx_df.columns:
        # Construir matriz de similaridade
        sim_vectors = np.array(idx_df["similarity_vector"].tolist())
        if sim_vectors.ndim == 2:
            df_sim = pd.DataFrame(sim_vectors)
            plot_heatmap(df_sim, "Mapa de Similaridade IDX")
    else:
        st.info("Nenhum dado IDX detalhado disponível.")

    # =====================================================
    # Seção 4 — Distribuição de acertos (Backtest Interno)
    # =====================================================
    st.markdown("### 🎯 Distribuição de Acertos — Backtest Interno")
    bti = st.session_state.get("backtest_interno", pd.DataFrame())

    if not bti.empty and "max_hits" in bti.columns:
        plot_hist(bti["max_hits"], "Distribuição do Máximo de Acertos", "Acertos")
    else:
        st.info("Backtest Interno não disponível para análise.")

    # =====================================================
    # Seção 5 — Coerência Retroativa (Backtest do Futuro)
    # =====================================================
    st.markdown("### 🔮 Coerência Retroativa — Backtest do Futuro")
    btf = st.session_state.get("btf_raw", pd.DataFrame())

    if not btf.empty and "coherence" in btf.columns:
        plot_hist(
            btf["coherence"],
            "Distribuição da Coerência Retroativa",
            "Coerência",
        )
    else:
        st.info("Backtest do Futuro ainda não foi realizado.")

    # =====================================================
    # Seção 6 — Convergência S6
    # =====================================================
    st.markdown("### ⭐ Convergência S6 — Intensidade")
    s6_df = st.session_state.get("s6_df", pd.DataFrame())

    if not s6_df.empty and "score" in s6_df.columns:
        plot_hist(s6_df["score"], "Distribuição de Convergência S6", "Score")
    else:
        st.info("Nenhum dado S6 disponível.")

    # =====================================================
    # Seção 7 — Índice de Estabilidade da Estrada
    # =====================================================
    st.markdown("### 🧩 Índice de Estabilidade da Estrada")

    regime_state = st.session_state.get("regime_state", None)
    estabilidade = calcular_indice_estabilidade(regime_state)

    if estabilidade is None:
        st.info("Estado da estrada ainda não foi calculado.")
    else:
        st.metric(
            label="Estabilidade Estrutural",
            value=f"{estabilidade*100:.1f}%"
        )

        if estabilidade >= 0.75:
            st.success("A estrada está ESTÁVEL (cenário Resiliente).")
        elif estabilidade >= 0.50:
            st.warning("A estrada está MODERADA (Intermediária).")
        else:
            st.error("A estrada está INSTÁVEL / Turbulenta.")

    st.stop()


# =========================================================
# FIM DO BLOCO 12 — Diagnóstico Profundo
# =========================================================
# =========================================================
# BLOCO 13 — Exportação TXT / CSV
# =========================================================

# Este bloco adiciona um painel extra na navegação:
# "Exportar Resultados"
#
# Ele permite baixar:
# - Lista Pura (TXT)
# - Séries Controladas com metadados (CSV)
# - Leque TURBO completo (CSV)
# - Logs Técnicos (TXT)
# - Diagnóstico profundo (CSV quando aplicável)
#
# Todos os arquivos são produzidos dinamicamente.


import io


# ---------------------------------------------------------
# Funções de Exportação
# ---------------------------------------------------------

def export_txt_list(lista_pura):
    """Gera TXT da lista pura numerada."""
    buffer = io.StringIO()
    for item in lista_pura:
        buffer.write(item + "\n")
    return buffer.getvalue().encode("utf-8")


def export_csv_df(df):
    """Exporta qualquer DataFrame para CSV."""
    return df.to_csv(index=False).encode("utf-8")


def export_logs_txt(logs):
    """Exporta logs técnicos em TXT estruturado."""
    buffer = io.StringIO()
    for registro in logs:
        buffer.write(f"Etapa: {registro['etapa']}\n")
        buffer.write(str(registro["dados"]) + "\n")
        buffer.write("\n" + "-"*50 + "\n\n")
    return buffer.getvalue().encode("utf-8")


def export_leque_csv(leque):
    """Exporta o Leque TURBO inteiro em múltiplas seções."""
    buffer = io.StringIO()
    for nome_secao, df_secao in leque.items():
        buffer.write(f"=== {nome_secao.upper()} ===\n")
        if isinstance(df_secao, list):  # ensamble
            buffer.write("Ensamble: " + " ".join(str(x) for x in df_secao) + "\n\n")
        else:
            buffer.write(df_secao.to_csv(index=False))
            buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------
# Adicionar item de navegação (após BLOCO 12)
# ---------------------------------------------------------
# Adicionar ao menu do BLOCO 10:
#
# "Diagnóstico Profundo",
# "Exportar Resultados",      # <-- ADICIONAR AQUI
#


# ---------------------------------------------------------
# PAINEL — Exportar Resultados
# ---------------------------------------------------------
if painel == "Exportar Resultados":

    st.subheader("📤 Exportação — V13.8-TURBO")

    leque = st.session_state.get("leque_turbo", {})
    regime_state = st.session_state.get("regime_state", None)
    logs = st.session_state.get("logs_tecnicos", [])
    idx_df = st.session_state.get("idx_result", pd.DataFrame())

    # Reconstruir flat_df
    flat_df = build_flat_series_table(leque)
    if flat_df.empty:
        st.warning("Nenhum dado disponível para exportação.")
        st.stop()

    # Reconstruir séries controladas (mesmo algoritmo da Saída Final)
    controlled_df = limit_by_mode(
        flat_df,
        regime_state,
        output_mode,
        n_series_fixed,
        min_conf_pct
    )

    # Criar a lista pura numerada
    lista_pura = []
    for i, row in controlled_df.iterrows():
        ss = " ".join(str(x) for x in row["series"])
        lista_pura.append(f"{i+1}) {ss}")

    st.markdown("### 📄 1) Lista Pura (somente séries filtradas)")
    st.write("Séries numeradas, prontas para cópia ou exportação.")

    txt_data = export_txt_list(lista_pura)
    st.download_button(
        label="⬇️ Baixar Lista Pura (TXT)",
        data=txt_data,
        file_name="lista_pura.txt",
        mime="text/plain"
    )

    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 2) Séries Controladas com Metadados (CSV)")

    df_ctrl = pd.DataFrame([
        {
            "rank": i+1,
            "categoria": row["category"],
            "series": " ".join(str(x) for x in row["series"]),
            "confiabilidade": row["coherence"],
            "acertos_esperados": int(row["expected_hits"])
        }
        for i, row in controlled_df.iterrows()
    ])

    st.dataframe(df_ctrl, use_container_width=True)

    csv_ctrl = export_csv_df(df_ctrl)
    st.download_button(
        label="⬇️ Baixar Séries Filtradas (CSV)",
        data=csv_ctrl,
        file_name="series_controladas.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🚀 3) Leque TURBO completo (CSV/TXT)")

    csv_leque = export_leque_csv(leque)
    st.download_button(
        label="⬇️ Baixar Leque TURBO Completo",
        data=csv_leque,
        file_name="leque_turbo_completo.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧪 4) Logs Técnicos do Pipeline (TXT)")

    if logs:
        logs_txt = export_logs_txt(logs)
        st.download_button(
            label="⬇️ Baixar Logs Técnicos (TXT)",
            data=logs_txt,
            file_name="logs_tecnicos.txt",
            mime="text/plain"
        )
    else:
        st.info("Nenhum log técnico disponível.")

    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 5) Diagnóstico (quando aplicável)")

    # Exporta dados básicos de diagnóstico
    diag_df = pd.DataFrame({
        "dispersao": df.apply(lambda row: np.std(list(row[:-1])), axis=1),
        "amplitude": df.apply(lambda row: max(row[:-1]) - min(row[:-1]), axis=1),
    })

    diag_csv = export_csv_df(diag_df)
    st.download_button(
        label="⬇️ Baixar Diagnóstico Estrutural (CSV)",
        data=diag_csv,
        file_name="diagnostico_basico.csv",
        mime="text/csv"
    )

    st.stop()


# =========================================================
# FIM DO BLOCO 13 — Exportação
# =========================================================
# =========================================================
# BLOCO 14 — Exportar Sessão Completa (ZIP)
# =========================================================

import zipfile
import json
from datetime import datetime


def build_session_zip():
    """
    Cria um ZIP com todos os dados disponíveis da sessão do app TURBO.
    Inclui:
    - histórico
    - IDX
    - IPF / IPO
    - ajustes
    - dependências
    - S6
    - Monte Carlo
    - backtests
    - leque TURBO
    - séries controladas
    - lista pura numerada
    - diagnóstico básico
    - logs técnicos
    - relatório técnico completo
    - estado da sessão (JSON)
    """

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as z:

        # ----------------------------------------------
        # 1) Histórico
        # ----------------------------------------------
        if not df.empty:
            z.writestr("historico.csv", df.to_csv(index=False))

        # ----------------------------------------------
        # 2) IDX
        # ----------------------------------------------
        idx_df = st.session_state.get("idx_result", pd.DataFrame())
        if not idx_df.empty:
            z.writestr("idx.csv", idx_df.to_csv(index=False))

        # ----------------------------------------------
        # 3) IPF / IPO
        # ----------------------------------------------
        ipf = st.session_state.get("nucleo_ipf", None)
        ipo = st.session_state.get("nucleo_ipo", None)

        if ipf:
            z.writestr("ipf.txt", str(ipf))
        if ipo:
            z.writestr("ipo.txt", str(ipo))

        # ----------------------------------------------
        # 4) Ajustes profundos
        # ----------------------------------------------
        ajustes = st.session_state.get("ajustes_log", [])
        if ajustes:
            txt = ""
            for bloco in ajustes:
                txt += f"[{bloco['nome']}]\n{bloco['dados']}\n\n"
            z.writestr("ajustes.txt", txt)

        # ----------------------------------------------
        # 5) Dependências ocultas
        # ----------------------------------------------
        deps = st.session_state.get("dependencias", None)
        if deps:
            z.writestr("dependencias.txt", str(deps))

        # ----------------------------------------------
        # 6) S6 Profundo
        # ----------------------------------------------
        s6_df = st.session_state.get("s6_df", pd.DataFrame())
        if not s6_df.empty:
            z.writestr("s6.csv", s6_df.to_csv(index=False))

        # ----------------------------------------------
        # 7) Monte Carlo Profundo
        # ----------------------------------------------
        mc_df = st.session_state.get("mc_df", pd.DataFrame())
        if not mc_df.empty:
            z.writestr("montecarlo.csv", mc_df.to_csv(index=False))

        # ----------------------------------------------
        # 8) Backtest Interno
        # ----------------------------------------------
        bti = st.session_state.get("backtest_interno", pd.DataFrame())
        if not bti.empty:
            z.writestr("backtest_interno.csv", bti.to_csv(index=False))

        # ----------------------------------------------
        # 9) Backtest do Futuro
        # ----------------------------------------------
        btf = st.session_state.get("btf_raw", pd.DataFrame())
        if not btf.empty:
            z.writestr("backtest_futuro.csv", btf.to_csv(index=False))

        # ----------------------------------------------
        # 10) Leque TURBO completo
        # ----------------------------------------------
        leque = st.session_state.get("leque_turbo", {})
        if leque:
            z.writestr("leque_turbo.csv", export_leque_csv(leque))

        # ----------------------------------------------
        # 11) Séries controladas (CSV)
        # ----------------------------------------------
        flat_df = build_flat_series_table(leque)
        regime_state = st.session_state.get("regime_state", None)

        ctrl_df = limit_by_mode(
            flat_df,
            regime_state,
            output_mode,
            n_series_fixed,
            min_conf_pct
        )
        if not ctrl_df.empty:
            df_ctrl = pd.DataFrame([
                {
                    "rank": i + 1,
                    "categoria": row["category"],
                    "series": " ".join(str(x) for x in row["series"]),
                    "confiabilidade": row["coherence"],
                    "acertos_esperados": int(row["expected_hits"])
                }
                for i, row in ctrl_df.iterrows()
            ])
            z.writestr("series_controladas.csv", df_ctrl.to_csv(index=False))

        # ----------------------------------------------
        # 12) Lista pura numerada
        # ----------------------------------------------
        lista_pura = []
        for i, row in ctrl_df.iterrows():
            ss = " ".join(str(x) for x in row["series"])
            lista_pura.append(f"{i + 1}) {ss}")

        if lista_pura:
            z.writestr(
                "lista_pura.txt",
                export_txt_list(lista_pura)
            )

        # ----------------------------------------------
        # 13) Diagnóstico básico
        # ----------------------------------------------
        diag_df = pd.DataFrame({
            "dispersao": df.apply(lambda row: np.std(list(row[:-1])), axis=1),
            "amplitude": df.apply(lambda row: max(row[:-1]) - min(row[:-1]), axis=1),
        })

        z.writestr("diagnostico_basico.csv", diag_df.to_csv(index=False))

        # ----------------------------------------------
        # 14) Logs técnicos (TXT)
        # ----------------------------------------------
        logs = st.session_state.get("logs_tecnicos", [])
        if logs:
            z.writestr("logs_tecnicos.txt", export_logs_txt(logs))

        # ----------------------------------------------
        # 15) Relatório técnico completo
        # ----------------------------------------------
        rel = io.StringIO()
        rel.write("=== RELATÓRIO TÉCNICO — Predict Cars V13.8-TURBO ===\n\n")

        if regime_state:
            rel.write(f"Regime: {regime_state.nome}\n")
            rel.write(f"Dispersão: {regime_state.dispersao}\n")
            rel.write(f"Amplitude: {regime_state.amplitude}\n")
            rel.write(f"Vibração: {regime_state.vibracao}\n")
            rel.write(f"Pares: {regime_state.pares}\n\n")

        rel.write("=== Parâmetros ===\n")
        rel.write(f"Modo de saída: {output_mode}\n")
        rel.write(f"N séries fixas: {n_series_fixed}\n")
        rel.write(f"Confiabilidade mínima: {min_conf_pct}\n\n")

        z.writestr("relatorio_completo.txt", rel.getvalue())

        # ----------------------------------------------
        # 16) Estado completo da sessão (JSON)
        # ----------------------------------------------
        state_json = json.dumps(
            {k: str(v) for k, v in st.session_state.items()},
            indent=2
        )
        z.writestr("estado_sessao.json", state_json)

    buffer.seek(0)
    return buffer


# ---------------------------------------------------------
# Painel de Navegação (incluir no menu)
# ---------------------------------------------------------
# Adicionar ao menu no BLOCO 10:
#
#   "Exportar Sessão Completa",
#


# ---------------------------------------------------------
# PAINEL — Exportar Sessão Completa (ZIP)
# ---------------------------------------------------------
if painel == "Exportar Sessão Completa":

    st.subheader("📦 Exportar Sessão Completa — V13.8-TURBO")

    if df.empty:
        st.warning("Carregue um histórico antes de exportar.")
        st.stop()

    if st.button("⬇️ Gerar e Baixar ZIP da Sessão Completa"):
        zip_file = build_session_zip()
        st.download_button(
            label="⬇️ Baixar Sessão Completa (ZIP)",
            data=zip_file,
            file_name=f"sessao_predictcars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

    st.stop()


# =========================================================
# FIM DO BLOCO 14 — Exportar Sessão Completa
# =========================================================
