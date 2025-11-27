import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
#  FUNÇÕES UTILITÁRIAS
# =========================
def parse_series_text(text: str) -> List[int]:
    if not text:
        return []
    separators = [",", ";", " "]
    tmp = text
    for sep in separators[1:]:
        tmp = tmp.replace(sep, separators[0])
    parts = [p.strip() for p in tmp.split(separators[0]) if p.strip()]
    return [int(p) for p in parts]


def series_to_str(series: List[int]) -> str:
    return " ".join(str(x) for x in series)


def ensure_session_state():
    defaults = {
        "data": None,
        "current_index": None,
        "current_series": [],
        "nucleo_resiliente": [],
        "idx_info": None,
        "series_puras": [],
        "series_avaliadas": pd.DataFrame(),
        "series_extras": [],
        "s6_ensamble": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
#  CAMADA DE DADOS
# =========================
def carregar_arquivo(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";|,", engine="python")
    else:
        df = pd.read_csv(uploaded_file, sep=";|,", engine="python", header=None)

    if df.shape[1] >= 8:
        df.columns = ["idx", "n1", "n2", "n3", "n4", "n5", "n6", "k"]
    elif df.shape[1] == 7:
        df.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
        df.insert(0, "idx", range(1, len(df) + 1))
    elif df.shape[1] == 6:
        df.columns = ["n1", "n2", "n3", "n4", "n5", "k"]
        df.insert(0, "idx", range(1, len(df) + 1))
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def extrair_passageiros_linha(linha: pd.Series) -> List[int]:
    passageiros = []
    for i in range(1, 7):
        col = f"n{i}"
        if col in linha.index:
            passageiros.append(int(linha[col]))
    return passageiros


def selecionar_indice_alvo(df: pd.DataFrame):
    if df.empty:
        return None, []

    idx_col = "idx"
    indices = df[idx_col].tolist()

    alvo = st.selectbox("Série alvo:", indices, index=len(indices) - 1)
    linha = df[df[idx_col] == alvo].iloc[0]

    return alvo, extrair_passageiros_linha(linha)


# =========================
#  BARÔMETRO
# =========================
def calcular_medidas_basicas(series):
    if not series:
        return {"media": 0, "desvio": 0, "amplitude": 0}
    arr = np.array(series)
    return {
        "media": float(arr.mean()),
        "desvio": float(arr.std()),
        "amplitude": float(arr.max() - arr.min()),
    }


def barometro_basico(series):
    m = calcular_medidas_basicas(series)
    if m["desvio"] < 10 and m["amplitude"] < 20:
        return "Resiliente"
    if m["desvio"] < 15:
        return "Intermediário"
    if m["amplitude"] > 35:
        return "Pré-Ruptura"
    return "Turbulento"


# =========================
#  IDX AVANÇADO (versão leve)
# =========================
def idx_avancado(df, target_series):
    if df.empty or not target_series:
        return {"indice_referencia": None, "trecho_referencia": None, "similaridade": 0}

    idx_col = "idx"
    target_set = set(target_series)

    best_score = -1
    best_idx = None
    best_series = None

    total = len(df)

    for pos, (_, row) in enumerate(df.iterrows()):
        serie = extrair_passageiros_linha(row)
        if not serie:
            continue

        intersec = len(target_set & set(serie)) / len(target_set)
        rec = (pos + 1) / total
        score = 0.7 * intersec + 0.3 * rec

        if score > best_score:
            best_score = score
            best_idx = row[idx_col]
            best_series = serie

    return {
        "indice_referencia": best_idx,
        "trecho_referencia": best_series,
        "similaridade": round(best_score, 3),
        "motorista_dominante": "Cluster_1" if best_score > 0.6 else "Cluster_2",
    }


# =========================
#  NÚCLEO RESILIENTE
# =========================
def ipf_basico(target):
    return target.copy()


def ipo_basico(nucleo):
    return sorted(nucleo)


def aplicar_anti_self_bias(nucleo, modo="B"):
    if modo == "B":
        rot = nucleo[1:] + nucleo[:1]
        return sorted(rot)
    return nucleo


def construir_nucleo_resiliente(target, modo_asb="B"):
    ipf = ipf_basico(target)
    ipo = ipo_basico(ipf)
    asb = aplicar_anti_self_bias(ipo, modo_asb)
    combinado = sorted(set(ipo + asb))
    return combinado[:6]


# =========================
#  SÉRIES PURAS
# =========================
def gerar_series_puras(nucleo):
    if not nucleo:
        return []
    base = nucleo
    rot1 = base[1:] + base[:1]
    rot2 = base[2:] + base[:2]
    cres = sorted(base)
    uniq = []
    seen = set()
    for s in [base, rot1, rot2, cres]:
        t = tuple(s)
        if t not in seen:
            uniq.append(s)
            seen.add(t)
    return uniq


# =========================
#  AVALIAÇÃO (ICA + HLA)
# =========================
def score_ica(s, nucleo):
    return len(set(s) & set(nucleo)) / len(nucleo)


def score_hla(s):
    arr = np.array(s)
    return 1 - ((arr.max() - arr.min()) / 80)


def avaliar_series(series, nucleo):
    registros = []
    for s in series:
        ica = score_ica(s, nucleo)
        hla = score_hla(s)
        conf = 0.6 * ica + 0.4 * hla

        farol = "Verde" if conf >= 0.75 else "Amarelo" if conf >= 0.55 else "Vermelho"

        registros.append({
            "serie": series_to_str(s),
            "ICA": round(ica, 3),
            "HLA": round(hla, 3),
            "Confiabilidade": round(conf, 3),
            "Farol": farol
        })

    df = pd.DataFrame(registros)
    return df.sort_values(by=["Confiabilidade"], ascending=False)


# =========================
#  GERADOR EXTRA
# =========================
def gerar_series_extras(nucleo, max_series=10):
    extras = []
    seen = set()
    for delta in [-2, -1, 1, 2]:
        for i in range(len(nucleo)):
            s = nucleo.copy()
            nv = s[i] + delta
            if 1 <= nv <= 80:
                s[i] = nv
                t = tuple(sorted(s))
                if t not in seen:
                    extras.append(sorted(s))
                    seen.add(t)
                if len(extras) >= max_series:
                    return extras
    return extras


# =========================
#  S6 + ENSAMBLE
# =========================
def construir_s6_ensamble(df_av, extras, top_n=5):
    registros = []

    if not df_av.empty:
        for _, row in df_av.head(top_n).iterrows():
            registros.append({
                "Origem": "Avaliadas",
                "Série": row["serie"],
                "Confiabilidade": row["Confiabilidade"],
            })

    for e in extras:
        registros.append({
            "Origem": "Extras",
            "Série": series_to_str(e),
            "Confiabilidade": None,
        })

    return pd.DataFrame(registros)


# =========================
#  INTERFACE STREAMLIT
# =========================
def main():
    ensure_session_state()
    st.set_page_config(page_title="Predict Cars V13.8", layout="wide")
    st.title("🔥 Predict Cars V13.8")

    menu = st.sidebar.radio(
        "Menu",
        [
            "Entrada de Dados",
            "Estado Atual",
            "IDX Avançado",
            "Núcleo Resiliente",
            "Séries Puras",
            "Séries Avaliadas",
            "Gerador Extra",
            "S6 + Ensamble",
            "Resumo"
        ],
    )

    if menu == "Entrada de Dados":
        arquivo = st.file_uploader("Selecione o arquivo histórico")
        if arquivo:
            df = carregar_arquivo(arquivo)
            st.session_state["data"] = df
            st.dataframe(df.head())

    if menu == "Estado Atual":
        df = st.session_state.get("data")
        if df is None or df.empty:
            st.warning("Carregue um arquivo primeiro.")
            return
        idx, serie = selecionar_indice_alvo(df)
        st.session_state["current_index"] = idx
        st.session_state["current_series"] = serie
        st.write("Série:", series_to_str(serie))
        st.write("Barômetro:", barometro_basico(serie))

    if menu == "IDX Avançado":
        df = st.session_state.get("data")
        serie = st.session_state.get("current_series")
        if df is None or df.empty or not serie:
            st.warning("Carregue dados e selecione série alvo.")
            return
        if st.button("Calcular IDX"):
            info = idx_avancado(df, serie)
            st.session_state["idx_info"] = info
            st.write(info)

    if menu == "Núcleo Resiliente":
        serie = st.session_state.get("current_series")
        if not serie:
            st.warning("Nenhuma série alvo.")
            return
        modo = st.selectbox("ASB", ["A", "B"], index=1)
        if st.button("Construir Núcleo"):
            nucleo = construir_nucleo_resiliente(serie, modo)
            st.session_state["nucleo_resiliente"] = nucleo
            st.write("Núcleo:", series_to_str(nucleo))

    if menu == "Séries Puras":
        nucleo = st.session_state.get("nucleo_resiliente")
        if not nucleo:
            st.warning("Construa o Núcleo primeiro.")
            return
        if st.button("Gerar Séries Puras"):
            sp = gerar_series_puras(nucleo)
            st.session_state["series_puras"] = sp
            st.code("\n".join(series_to_str(s) for s in sp))

    if menu == "Séries Avaliadas":
        nucleo = st.session_state.get("nucleo_resiliente")
        sp = st.session_state.get("series_puras")
        if not nucleo or not sp:
            st.warning("Gere Séries Puras antes.")
            return
        if st.button("Avaliar Séries"):
            df_av = avaliar_series(sp, nucleo)
            st.session_state["series_avaliadas"] = df_av
            st.dataframe(df_av)

    if menu == "Gerador Extra":
        nucleo = st.session_state.get("nucleo_resiliente")
        if not nucleo:
            st.warning("Núcleo ausente.")
            return
        max_s = st.slider("Quantas séries extras?", 3, 20, 10)
        if st.button("Gerar Extras"):
            ex = gerar_series_extras(nucleo, max_s)
            st.session_state["series_extras"] = ex
            st.code("\n".join(series_to_str(s) for s in ex))

    if menu == "S6 + Ensamble":
        df_av = st.session_state.get("series_avaliadas")
        ex = st.session_state.get("series_extras")
        if df_av is None and not ex:
            st.warning("Nada a consolidar.")
            return
        topn = st.slider("Top N avaliadas", 1, 10, 5)
        df2 = construir_s6_ensamble(df_av, ex, topn)
        st.session_state["s6_ensamble"] = df2
        st.dataframe(df2)

    if menu == "Resumo":
        st.write("👉 Resumo técnico do estado atual")
        st.write("Série alvo:", series_to_str(st.session_state.get("current_series", [])))
        st.write("Núcleo:", series_to_str(st.session_state.get("nucleo_resiliente", [])))
        st.write("IDX:", st.session_state.get("idx_info"))
        st.write("Qtd Séries Puras:", len(st.session_state.get("series_puras", [])))
        st.write("Qtd Avaliadas:", len(st.session_state.get("series_avaliadas", [])))
        st.write("Qtd Extras:", len(st.session_state.get("series_extras", [])))


if __name__ == "__main__":
    main()
