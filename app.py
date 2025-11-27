import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================
#  UTILITÁRIOS BÁSICOS
# ============================================

def parse_series_text(text: str) -> List[int]:
    """Converte texto colado em lista de números."""
    if not text:
        return []
    txt = text.replace(",", " ").replace(";", " ")
    parts = [p.strip() for p in txt.split() if p.strip().isdigit()]
    return [int(p) for p in parts]


def series_to_str(series: List[int]) -> str:
    """Transforma lista em formato '1 2 3 4 5 6'."""
    return " ".join(str(x) for x in series)


def ensure_session_state():
    """Garante variáveis internas do app."""
    defaults = {
        "data": None,
        "data_text": "",
        "current_index": None,
        "current_series": [],
        "idx_info": None,
        "nucleo_resiliente": [],
        "series_puras": [],
        "series_avaliadas": pd.DataFrame(),
        "series_extras": [],
        "s6_ensamble": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================
#  ENTRADA DE DADOS — (A) ARQUIVO
# ============================================

def carregar_arquivo(uploaded_file) -> pd.DataFrame:
    """Carrega arquivo .txt ou .csv com 6 passageiros + k."""
    try:
        df = pd.read_csv(uploaded_file, sep=";|,|\s+", engine="python", header=None)

        if df.shape[1] == 7:
            df.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
            df.insert(0, "idx", range(1, len(df) + 1))

        elif df.shape[1] == 8:
            df.columns = ["idx", "n1", "n2", "n3", "n4", "n5", "n6", "k"]

        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        return df

    except Exception:
        return pd.DataFrame()


# ============================================
#  ENTRADA DE DADOS — (B) TEXTO COLADO
# ============================================

def carregar_texto_colado(texto: str) -> pd.DataFrame:
    """Interpreta séries coladas manualmente no campo de texto."""
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    registros = []

    for i, linha in enumerate(linhas, start=1):
        nums = parse_series_text(linha)
        if len(nums) >= 6:
            passageiros = nums[:6]
            k = nums[6] if len(nums) > 6 else 0
            registros.append([i] + passageiros + [k])

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros, columns=["idx", "n1", "n2", "n3", "n4", "n5", "n6", "k"])
    return df


# ============================================
#  EXTRAIR PASSAGEIROS
# ============================================

def extrair_passageiros_linha(linha: pd.Series) -> List[int]:
    """Extrai n1..n6 de uma linha."""
    return [int(linha[f"n{i}"]) for i in range(1, 7) if f"n{i}" in linha]


# ============================================
#  SELEÇÃO DA SÉRIE ALVO
# ============================================

def selecionar_indice_alvo(df: pd.DataFrame):
    """Seleciona índice alvo e extrai passageiros."""
    if df.empty:
        return None, []

    indices = df["idx"].tolist()
    alvo = st.selectbox("Série alvo:", indices, index=len(indices) - 1)
    linha = df[df["idx"] == alvo].iloc[0]

    return alvo, extrair_passageiros_linha(linha)


# ============================================
#  BARÔMETRO — versão leve
# ============================================

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


# ============================================
#  IDX AVANÇADO (versão leve garantida)
# ============================================

def idx_avancado(df: pd.DataFrame, alvo: List[int]):
    if df.empty or not alvo:
        return {"indice_referencia": None, "trecho_referencia": None, "similaridade": 0}

    alvo_set = set(alvo)
    idx_col = "idx"

    best_score = -1
    best_idx = None
    best_serie = None

    total = len(df)

    for pos, (_, row) in enumerate(df.iterrows()):
        serie = extrair_passageiros_linha(row)
        if not serie:
            continue

        intersec = len(alvo_set & set(serie)) / len(alvo_set)
        recencia = (pos + 1) / total

        score = 0.7 * intersec + 0.3 * recencia

        if score > best_score:
            best_score = score
            best_idx = row[idx_col]
            best_serie = serie

    return {
        "indice_referencia": best_idx,
        "trecho_referencia": best_serie,
        "similaridade": round(best_score, 3),
        "motorista_dominante": "Cluster_1" if best_score >= 0.6 else "Cluster_2",
    }


# ============================================
#  NÚCLEO RESILIENTE (IPF + IPO + ASB)
# ============================================

def ipf_basico(series: List[int]) -> List[int]:
    return series.copy()


def ipo_basico(series: List[int]) -> List[int]:
    return sorted(series)


def aplicar_anti_self_bias(series: List[int], modo="B") -> List[int]:
    if modo == "B":
        rot = series[1:] + series[:1]
        return sorted(rot)
    return series


def construir_nucleo_resiliente(series: List[int], modo_asb="B") -> List[int]:
    ipf = ipf_basico(series)
    ipo = ipo_basico(ipf)
    asb = aplicar_anti_self_bias(ipo, modo_asb)
    uni = sorted(set(ipo + asb))
    return uni[:6]


# ============================================
#  SÉRIES PURAS
# ============================================

def gerar_series_puras(nucleo: List[int]) -> List[List[int]]:
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


# ============================================
#  AVALIAÇÃO (ICA + HLA)
# ============================================

def score_ica(serie: List[int], nucleo: List[int]) -> float:
    return len(set(serie) & set(nucleo)) / len(nucleo)


def score_hla(serie: List[int]) -> float:
    arr = np.array(serie)
    return 1 - ((arr.max() - arr.min()) / 80)


def avaliar_series(series: List[List[int]], nucleo: List[int]) -> pd.DataFrame:
    registros = []

    for s in series:
        ica = score_ica(s, nucleo)
        hla = score_hla(s)
        conf = 0.6 * ica + 0.4 * hla

        farol = (
            "Verde" if conf >= 0.75
            else "Amarelo" if conf >= 0.55
            else "Vermelho"
        )

        registros.append({
            "serie": series_to_str(s),
            "ICA": round(ica, 3),
            "HLA": round(hla, 3),
            "Confiabilidade": round(conf, 3),
            "Farol": farol,
        })

    df = pd.DataFrame(registros)
    return df.sort_values(by="Confiabilidade", ascending=False)


# ============================================
#  GERADOR EXTRA
# ============================================

def gerar_series_extras(nucleo: List[int], max_series: int = 12) -> List[List[int]]:
    extras = []
    seen = set()

    for delta in [-2, -1, 1, 2]:
        for i in range(len(nucleo)):
            s = nucleo.copy()
            nv = s[i] + delta
            if 1 <= nv <= 80:
                s[i] = nv
                key = tuple(sorted(s))
                if key not in seen:
                    seen.add(key)
                    extras.append(sorted(s))
                if len(extras) >= max_series:
                    return extras

    return extras


# ============================================
#  ENSAMBLE FINAL (S6)
# ============================================

def construir_s6_ensamble(df_av: pd.DataFrame, extras: List[List[int]], top_n: int = 5) -> pd.DataFrame:
    registros = []

    if not df_av.empty:
        for _, row in df_av.head(top_n).iterrows():
            registros.append({
                "Origem": "Avaliada",
                "Série": row["serie"],
                "Confiabilidade": row["Confiabilidade"],
            })

    for s in extras:
        registros.append({
            "Origem": "Extra",
            "Série": series_to_str(s),
            "Confiabilidade": None,
        })

    return pd.DataFrame(registros)
# ============================================
#  INTERFACE STREAMLIT
# ============================================

def painel_entrada_dados():
    st.subheader("1. Entrada de Dados")

    col1, col2 = st.columns(2)

    # (A) Upload de arquivo
    with col1:
        st.markdown("**A) Carregar arquivo (.txt / .csv)**")
        uploaded = st.file_uploader(
            "Selecione o arquivo histórico",
            type=["txt", "csv"],
            key="uploader_arquivo",
        )
        if uploaded is not None:
            df = carregar_arquivo(uploaded)
            if df.empty:
                st.error("Falha ao interpretar o arquivo.")
            else:
                st.session_state["data"] = df
                st.success("Arquivo carregado com sucesso.")
                with st.expander("Visualizar primeiras linhas"):
                    st.dataframe(df.head(30))

    # (B) Texto colado
    with col2:
        st.markdown("**B) Colar séries manualmente**")
        texto = st.text_area(
            "Cole aqui as linhas de séries (cada linha = 6 passageiros + k opcional):",
            value=st.session_state.get("data_text", ""),
            height=220,
        )

        colb1, colb2 = st.columns([1, 1])
        with colb1:
            if st.button("Carregar do texto colado"):
                df_txt = carregar_texto_colado(texto)
                if df_txt.empty:
                    st.error("Não foi possível interpretar nenhuma linha válida.")
                else:
                    st.session_state["data"] = df_txt
                    st.session_state["data_text"] = texto
                    st.success("Texto carregado como base de dados.")
        with colb2:
            if st.button("Limpar texto colado"):
                st.session_state["data_text"] = ""
                st.experimental_rerun()

    df_atual = st.session_state.get("data")
    if df_atual is not None and not df_atual.empty:
        st.markdown("---")
        st.markdown("**Status dos dados atuais:**")
        st.write(f"Linhas: {len(df_atual)} | Colunas: {list(df_atual.columns)}")


def painel_estado_atual():
    st.subheader("2. Estado Atual da Série Alvo")

    df = st.session_state.get("data")
    if df is None or df.empty:
        st.warning("Nenhum dado carregado. Use a aba 'Entrada de Dados'.")
        return

    idx, serie = selecionar_indice_alvo(df)
    st.session_state["current_index"] = idx
    st.session_state["current_series"] = serie

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Índice alvo:** `{idx}`")
        st.markdown(f"**Passageiros (série alvo):** `{series_to_str(serie)}`")

    with col2:
        medidas = calcular_medidas_basicas(serie)
        st.markdown("**Medidas básicas:**")
        st.json(medidas)
        st.markdown(f"**Barômetro (versão leve):** `{barometro_basico(serie)}`")


def painel_idx():
    st.subheader("3. IDX Avançado (versão leve)")

    df = st.session_state.get("data")
    serie = st.session_state.get("current_series")

    if df is None or df.empty:
        st.warning("Nenhum dado carregado.")
        return
    if not serie:
        st.warning("Nenhuma série alvo selecionada (ver aba Estado Atual).")
        return

    if st.button("Calcular IDX para a série alvo"):
        info = idx_avancado(df, serie)
        st.session_state["idx_info"] = info
        st.success("IDX calculado.")
        st.markdown("**Trecho mais parecido (IDX):**")
        st.json(
            {
                "indice_referencia": info["indice_referencia"],
                "trecho_referencia": series_to_str(info["trecho_referencia"])
                if info["trecho_referencia"]
                else None,
                "similaridade": info["similaridade"],
                "motorista_dominante": info["motorista_dominante"],
            }
        )


def painel_nucleo():
    st.subheader("4. Núcleo Resiliente (IPF + IPO + ASB)")

    serie = st.session_state.get("current_series")
    if not serie:
        st.warning("Nenhuma série alvo selecionada.")
        return

    modo = st.selectbox(
        "Modo Anti-SelfBias (ASB)",
        ["A", "B"],
        index=1,
        help="Modo B é mais recomendado para robustez.",
    )

    if st.button("Construir Núcleo Resiliente"):
        nucleo = construir_nucleo_resiliente(serie, modo)
        st.session_state["nucleo_resiliente"] = nucleo
        st.success("Núcleo Resiliente construído.")
        st.markdown(f"**Núcleo Resiliente:** `{series_to_str(nucleo)}`")

    nucleo_atual = st.session_state.get("nucleo_resiliente", [])
    if nucleo_atual:
        texto = st.text_input(
            "Ajuste manual do núcleo (opcional):",
            value=series_to_str(nucleo_atual),
            help="Formato: 8 29 30 34 39 60",
        )
        if st.button("Aplicar ajuste manual do núcleo"):
            lista = parse_series_text(texto)
            if len(lista) >= 5:
                st.session_state["nucleo_resiliente"] = lista[:6]
                st.success("Núcleo atualizado manualmente.")
            else:
                st.error("Informe ao menos 5 números para o núcleo.")


def painel_series_puras():
    st.subheader("5. Séries Puras (6 passageiros)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    if not nucleo:
        st.warning("Núcleo Resiliente ainda não definido.")
        return

    if st.button("Gerar Séries Puras a partir do Núcleo"):
        sp = gerar_series_puras(nucleo)
        st.session_state["series_puras"] = sp
        st.success("Séries Puras geradas.")
        st.code("\n".join(series_to_str(s) for s in sp))


def painel_series_avaliadas():
    st.subheader("6. Séries Avaliadas (ICA + HLA + Confiabilidade)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    series_puras = st.session_state.get("series_puras", [])

    if not nucleo:
        st.warning("Núcleo Resiliente não definido.")
        return
    if not series_puras:
        st.warning("Nenhuma Série Pura gerada.")
        return

    if st.button("Avaliar Séries Puras"):
        df_av = avaliar_series(series_puras, nucleo)
        st.session_state["series_avaliadas"] = df_av
        st.success("Séries avaliadas com sucesso.")
        st.dataframe(df_av, use_container_width=True)


def painel_gerador_extra():
    st.subheader("7. Gerador Extra (abrindo faixas)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    if not nucleo:
        st.warning("Núcleo Resiliente não definido.")
        return

    max_s = st.slider("Quantidade máxima de séries extras", 3, 30, 12)
    if st.button("Gerar Séries Extras"):
        extras = gerar_series_extras(nucleo, max_s)
        st.session_state["series_extras"] = extras
        st.success("Séries Extras geradas.")
        st.code("\n".join(series_to_str(s) for s in extras))


def painel_s6_ensamble():
    st.subheader("8. S6 + Ensamble (visão consolidada)")

    df_av = st.session_state.get("series_avaliadas", pd.DataFrame())
    extras = st.session_state.get("series_extras", [])

    if (df_av is None or df_av.empty) and not extras:
        st.warning("Nenhuma série avaliada ou extra disponível.")
        return

    top_n = st.slider("Top N Séries Avaliadas para o Ensamble", 1, 15, 5)

    df_ens = construir_s6_ensamble(df_av if df_av is not None else pd.DataFrame(),
                                   extras or [],
                                   top_n)
    st.session_state["s6_ensamble"] = df_ens

    st.markdown("**Pacote consolidado (S6 + Ensamble):**")
    st.dataframe(df_ens, use_container_width=True)

    if not df_ens.empty:
        todas = []
        for _, row in df_ens.iterrows():
            s = parse_series_text(row["Série"])
            todas.append(s)
        flat = list(itertools.chain.from_iterable(todas))
        freq = pd.Series(flat).value_counts().reset_index()
        freq.columns = ["passageiro", "frequencia"]
        st.markdown("**Frequência de passageiros no pacote consolidado:**")
        st.dataframe(freq, use_container_width=True)


def painel_resumo():
    st.subheader("9. Resumo Técnico do Estado Atual")

    idx = st.session_state.get("current_index")
    serie = st.session_state.get("current_series", [])
    nucleo = st.session_state.get("nucleo_resiliente", [])
    idx_info = st.session_state.get("idx_info", None)
    sp = st.session_state.get("series_puras", [])
    df_av = st.session_state.get("series_avaliadas", pd.DataFrame())
    extras = st.session_state.get("series_extras", [])
    ens = st.session_state.get("s6_ensamble", pd.DataFrame())

    st.markdown(f"**Índice alvo:** `{idx}`")
    st.markdown(f"**Série alvo:** `{series_to_str(serie)}`")
    st.markdown(f"**Núcleo Resiliente:** `{series_to_str(nucleo)}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Qtd Séries Puras", len(sp))
    with col2:
        st.metric("Qtd Séries Avaliadas", 0 if df_av is None or df_av.empty else len(df_av))
    with col3:
        st.metric("Qtd Séries Extras", len(extras))

    st.markdown("---")
    st.markdown("**Resumo IDX (versão leve):**")
    if idx_info is None:
        st.info("IDX ainda não foi calculado.")
    else:
        st.json(
            {
                "indice_referencia": idx_info["indice_referencia"],
                "trecho_referencia": series_to_str(idx_info["trecho_referencia"])
                if idx_info["trecho_referencia"]
                else None,
                "similaridade": idx_info["similaridade"],
                "motorista_dominante": idx_info["motorista_dominante"],
            }
        )

    st.markdown("---")
    st.markdown("**Pacote S6 + Ensamble (visão geral):**")
    if ens is None or ens.empty:
        st.info("Nenhum pacote consolidado ainda.")
    else:
        st.dataframe(ens, use_container_width=True)


# ============================================
#  MAIN
# ============================================

def main():
    ensure_session_state()
    st.set_page_config(page_title="Predict Cars V13.8", layout="wide")
    st.title("🔥 Predict Cars V13.8 — App Ultra-Híbrido (Versão Estrutural)")

    menu = st.sidebar.radio(
        "Navegação",
        [
            "Entrada de Dados",
            "Estado Atual",
            "IDX Avançado",
            "Núcleo Resiliente",
            "Séries Puras",
            "Séries Avaliadas",
            "Gerador Extra",
            "S6 + Ensamble",
            "Resumo",
        ],
    )

    if menu == "Entrada de Dados":
        painel_entrada_dados()
    elif menu == "Estado Atual":
        painel_estado_atual()
    elif menu == "IDX Avançado":
        painel_idx()
    elif menu == "Núcleo Resiliente":
        painel_nucleo()
    elif menu == "Séries Puras":
        painel_series_puras()
    elif menu == "Séries Avaliadas":
        painel_series_avaliadas()
    elif menu == "Gerador Extra":
        painel_gerador_extra()
    elif menu == "S6 + Ensamble":
        painel_s6_ensamble()
    elif menu == "Resumo":
        painel_resumo()


if __name__ == "__main__":
    main()
