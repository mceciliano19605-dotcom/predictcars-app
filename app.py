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
# =========================================================
