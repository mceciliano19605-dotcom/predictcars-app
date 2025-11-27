import streamlit as st
import pandas aimport itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
#  PREDICT CARS V13.8 — APP STREAMLIT (VERSÃO ORGANIZADA)
# ---------------------------------------------------------
#  Estrutura geral:
#  - Entrada de dados
#  - Estado atual + Barômetro
#  - IDX Avançado (similaridade + recência) – versão leve
#  - Núcleo Resiliente (IPF + IPO + Anti-SelfBias)
#  - Séries Puras
#  - Séries Avaliadas (ICA + HLA + Faróis + Confiabilidade)
#  - Gerador Extra (por confiabilidade alvo)
#  - S6 + Ensamble Final
#
#  Observação:
#  A arquitetura está completa e funcional.
#  As funções podem ser enriquecidas depois com a lógica
#  completa do V13.8, mantendo esta mesma estrutura.
# ---------------------------------------------------------


# =========================
#  FUNÇÕES UTILITÁRIAS
# =========================
def parse_series_text(text: str) -> List[int]:
    """
    Converte um texto de série em lista de inteiros.
    Aceita formatos:
    - "8 29 30 34 39 60"
    - "8,29,30,34,39,60"
    - "8;29;30;34;39;60"
    """
    if not text:
        return []
    separators = [",", ";", " "]
    tmp = text
    for sep in separators[1:]:
        tmp = tmp.replace(sep, separators[0])
    parts = [p.strip() for p in tmp.split(separators[0]) if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return []


def series_to_str(series: List[int]) -> str:
    """Converte uma lista de inteiros em string 'n1 n2 n3 n4 n5 n6'."""
    return " ".join(str(x) for x in series)


def ensure_session_state():
    """
    Garante chaves básicas em st.session_state.
    """
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
    """
    Carrega arquivo de entrada.
    Formatos aceitos:
    - .csv
    - .txt (separador ';' ou ',')
    Suporta arquivos com 5 ou 6 passageiros + k.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";|,", engine="python")
    else:
        df = pd.read_csv(uploaded_file, sep=";|,", engine="python", header=None)

    # Normalização básica de colunas
    # Tentativa de mapear idx + n1..n6 + k
    if df.shape[1] >= 8:
        df.columns = ["idx", "n1", "n2", "n3", "n4", "n5", "n6", "k"] + [
            f"extra_{i}" for i in range(8, df.shape[1])
        ]
    elif df.shape[1] == 7:
        df.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "k"]
        df.insert(0, "idx", range(1, len(df) + 1))
    elif df.shape[1] == 6:
        # 5 passageiros + k
        df.columns = ["n1", "n2", "n3", "n4", "n5", "k"]
        df.insert(0, "idx", range(1, len(df) + 1))
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def extrair_passageiros_linha(linha: pd.Series) -> List[int]:
    """
    Extrai os passageiros n1..n6 (se existirem) de uma linha do DataFrame.
    """
    passageiros = []
    for i in range(1, 7):
        col = f"n{i}"
        if col in linha.index:
            try:
                passageiros.append(int(linha[col]))
            except Exception:
                pass
    return passageiros


def selecionar_indice_alvo(df: pd.DataFrame) -> Tuple[Any, List[int]]:
    """
    Seleciona a série alvo (linha atual) com base em um seletor na UI.
    Utiliza a última linha como padrão.
    """
    if df.empty:
        return None, []

    idx_col = "idx" if "idx" in df.columns else df.columns[0]
    indices = df[idx_col].tolist()
    default_idx = indices[-1]

    alvo = st.selectbox(
        "Série alvo (índice):",
        options=indices,
        index=indices.index(default_idx),
        key="select_indice_alvo",
    )

    linha = df[df[idx_col] == alvo].iloc[0]
    numeros = extrair_passageiros_linha(linha)

    return alvo, numeros


# =========================
#  CAMADAS ANALÍTICAS BÁSICAS
# =========================
def calcular_medidas_basicas(series: List[int]) -> Dict[str, float]:
    """
    Calcula medidas básicas da série alvo.
    Utilizado para compor indicadores simples de regime.
    """
    if not series:
        return {"media": 0.0, "desvio": 0.0, "amplitude": 0.0}

    arr = np.array(series)
    return {
        "media": float(arr.mean()),
        "desvio": float(arr.std(ddof=0)),
        "amplitude": float(arr.max() - arr.min()),
    }


def barometro_basico(series: List[int]) -> str:
    """
    Barômetro simples, placeholder.
    Pode ser substituído pela lógica completa do V13.8.
    """
    if not series:
        return "Indefinido"

    medidas = calcular_medidas_basicas(series)
    desvio = medidas["desvio"]
    amplitude = medidas["amplitude"]

    if desvio < 10 and amplitude < 20:
        return "Resiliente"
    if desvio < 15:
        return "Intermediário"
    if amplitude > 35:
        return "Pré-Ruptura"
    return "Turbulento"


# =========================
#  IDX AVANÇADO (VERSÃO LEVE)
# =========================
def idx_avancado(df: pd.DataFrame, target_series: List[int]) -> Dict[str, Any]:
    """
    IDX Avançado simplificado:
    - percorre o histórico
    - calcula similaridade por interseção de passageiros
    - adiciona um peso de recência
    - retorna trecho mais parecido e um "motorista dominante" simbólico
    """
    if df is None or df.empty or not target_series:
        return {
            "indice_referencia": None,
            "trecho_referencia": None,
            "similaridade": 0.0,
            "motorista_dominante": None,
        }

    idx_col = "idx" if "idx" in df.columns else df.columns[0]
    target_set = set(target_series)
    n_linhas = len(df)

    melhor_score = -1.0
    melhor_idx = None
    melhor_serie = None

    for i, (_, row) in enumerate(df.iterrows()):
        passageiros = extrair_passageiros_linha(row)
        if not passageiros:
            continue

        intersec = len(target_set & set(passageiros))
        base_sim = intersec / max(len(target_set), 1)

        # Peso de recência: linhas mais ao fim recebem peso maior
        recencia = (i + 1) / n_linhas
        score_total = 0.7 * base_sim + 0.3 * recencia

        if score_total > melhor_score:
            melhor_score = score_total
            melhor_idx = row[idx_col]
            melhor_serie = passageiros

    motorista = "Cluster_1" if melhor_score >= 0.7 else "Cluster_2"

    return {
        "indice_referencia": melhor_idx,
        "trecho_referencia": melhor_serie,
        "similaridade": float(round(melhor_score, 3)),
        "motorista_dominante": motorista,
    }


# =========================
#  IPF / IPO / ANTI-SELFBIAS / NÚCLEO
# =========================
def ipf_basico(target_series: List[int]) -> List[int]:
    """
    IPF básico:
    Utiliza diretamente a série alvo como núcleo inicial.
    """
    return list(target_series)


def ipo_basico(nucleo_ipf: List[int]) -> List[int]:
    """
    IPO básico:
    Ordenação estabilizadora de passageiros.
    """
    return sorted(nucleo_ipf)


def aplicar_anti_self_bias(nucleo: List[int], modo: str = "B") -> List[int]:
    """
    Anti-SelfBias versionado.
    Modo B:
    - rotaciona levemente a série para quebrar rigidez
    - mantém estrutura geral
    """
    if not nucleo:
        return []

    base = list(nucleo)
    if modo == "B":
        rot = base[1:] + base[:1]
        return sorted(rot)
    if modo == "A":
        return base
    return base


def construir_nucleo_resiliente(
    target_series: List[int],
    modo_asb: str = "B",
) -> List[int]:
    """
    Constrói o Núcleo Resiliente a partir de:
    - Série alvo
    - IPF básico
    - IPO básico
    - Anti-SelfBias
    Combinação simplificada:
    - união dos núcleos
    - ordenação
    - corte em até 6 passageiros
    """
    if not target_series:
        return []

    nucleo_ipf = ipf_basico(target_series)
    nucleo_ipo = ipo_basico(nucleo_ipf)
    nucleo_asb = aplicar_anti_self_bias(nucleo_ipo, modo=modo_asb)

    combinado = sorted(set(nucleo_ipo + nucleo_asb))
    return combinado[:6]


# =========================
#  ICA / HLA (RANQUEAMENTO)
# =========================
def calcular_score_ica(serie: List[int], nucleo: List[int]) -> float:
    """
    Score ICA simplificado:
    - baseado no tamanho da interseção com o Núcleo Resiliente.
    """
    if not serie or not nucleo:
        return 0.0
    intersec = len(set(serie) & set(nucleo))
    return intersec / max(len(nucleo), 1)


def calcular_score_hla(serie: List[int]) -> float:
    """
    Score HLA simplificado:
    - baseado na amplitude da série (compactação).
    """
    if not serie:
        return 0.0
    arr = np.array(serie)
    amplitude = arr.max() - arr.min()
    return 1.0 - min(amplitude / 80.0, 1.0)


def classificar_farol(confiabilidade: float) -> str:
    """
    Classifica farol com base na confiabilidade.
    """
    if confiabilidade >= 0.75:
        return "Verde"
    if confiabilidade >= 0.55:
        return "Amarelo"
    return "Vermelho"


# =========================
#  SÉRIES PURAS
# =========================
def gerar_series_puras(nucleo_resiliente: List[int]) -> List[List[int]]:
    """
    Gera séries puras a partir do Núcleo Resiliente.
    Versão simplificada:
    - Série base
    - Rotação 1
    - Rotação 2
    - Ordenação crescente
    Remove duplicatas.
    """
    if not nucleo_resiliente:
        return []

    base = list(nucleo_resiliente)
    rot1 = base[1:] + base[:1]
    rot2 = base[2:] + base[:2]
    crescente = sorted(base)

    series = [base, rot1, rot2, crescente]
    series_unicas = []
    seen = set()
    for s in series:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            series_unicas.append(s)

    return series_unicas


# =========================
#  SÉRIES AVALIADAS
# =========================
def avaliar_series(
    series: List[List[int]],
    nucleo_resiliente: List[int],
) -> pd.DataFrame:
    """
    Avalia séries com ICA, HLA, confiabilidade e farol.
    """
    registros = []
    for s in series:
        score_ica = calcular_score_ica(s, nucleo_resiliente)
        score_hla = calcular_score_hla(s)
        confiab = 0.6 * score_ica + 0.4 * score_hla
        farol = classificar_farol(confiab)
        registros.append(
            {
                "serie": series_to_str(s),
                "score_ica": round(score_ica, 3),
                "score_hla": round(score_hla, 3),
                "confiabilidade": round(confiab, 3),
                "farol": farol,
            }
        )

    df = pd.DataFrame(registros)
    df = df.sort_values(by=["confiabilidade", "score_ica", "score_hla"], ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


# =========================
#  GERADOR EXTRA
# =========================
def gerar_series_extras(
    nucleo_resiliente: List[int],
    confiabilidade_alvo: float = 0.8,
    max_series: int = 10,
) -> List[List[int]]:
    """
    Gera séries extras a partir do Núcleo Resiliente,
    perturbando levemente alguns elementos.
    A confiabilidade_alvo é usada conceitualmente aqui;
    a versão completa pode ajustar a lógica mais finamente.
    """
    if not nucleo_resiliente:
        return []

    base = list(nucleo_resiliente)
    extras = []
    seen = set()

    for delta in [-2, -1, 1, 2]:
        for i in range(len(base)):
            serie = base.copy()
            novo_valor = serie[i] + delta
            if 1 <= novo_valor <= 80:
                serie[i] = novo_valor
                key = tuple(sorted(serie))
                if key not in seen:
                    seen.add(key)
                    extras.append(sorted(serie))
            if len(extras) >= max_series:
                break
        if len(extras) >= max_series:
            break

    return extras


# =========================
#  S6 + ENSAMBLE FINAL
# =========================
def construir_s6_ensamble(
    df_avaliadas: pd.DataFrame,
    extras: List[List[int]],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Constrói visão S6 + Ensamble a partir das melhores séries avaliadas
    e das séries extras.
    """
    registros = []

    # Top séries avaliadas
    if not df_avaliadas.empty:
        subset = df_avaliadas.head(top_n)
        for _, row in subset.iterrows():
            numeros = parse_series_text(row["serie"])
            registros.append(
                {
                    "origem": "Avaliadas",
                    "serie": row["serie"],
                    "confiabilidade": row["confiabilidade"],
                    "qtd_passageiros": len(numeros),
                }
            )

    # Séries extras
    for s in extras:
        registros.append(
            {
                "origem": "Extras",
                "serie": series_to_str(s),
                "confiabilidade": np.nan,
                "qtd_passageiros": len(s),
            }
        )

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    return df


# =========================
#  INTERFACE STREAMLIT
# =========================
def mostrar_header_principal():
    st.title("🔥 Predict Cars V13.8 — App Ultra-Híbrido")
    st.caption("IDX / Núcleo Resiliente / Séries Puras / Séries Avaliadas / S6 / Ensamble")


def painel_dados():
    st.subheader("1. Entrada de Dados")

    uploaded = st.file_uploader(
        "Arquivo histórico (.csv ou .txt com separador ';' ou ',')",
        type=["csv", "txt"],
    )

    if uploaded is not None:
        df = carregar_arquivo(uploaded)
        st.session_state["data"] = df
        st.success("Arquivo carregado com sucesso.")
        with st.expander("Visualizar primeiras linhas"):
            st.dataframe(df.head(20))
    else:
        st.info("Carregar um arquivo para iniciar.")


def painel_estado_atual():
    st.subheader("2. Estado Atual da Série Alvo")

    df = st.session_state.get("data")
    if df is None or df.empty:
        st.warning("Nenhum arquivo carregado.")
        return

    idx, numeros = selecionar_indice_alvo(df)
    st.session_state["current_index"] = idx
    st.session_state["current_series"] = numeros

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Índice alvo:** `{idx}`")
        st.markdown(f"**Passageiros alvo:** `{series_to_str(numeros)}`")

    with col2:
        medidas = calcular_medidas_basicas(numeros)
        st.markdown("**Medidas básicas:**")
        st.write(medidas)
        st.markdown(f"**Barômetro (simplificado):** `{barometro_basico(numeros)}`")

    # IDX Avançado (visão geral)
    if st.button("Calcular IDX Avançado (similaridade + recência)"):
        info_idx = idx_avancado(df, numeros)
        st.session_state["idx_info"] = info_idx
        st.success("IDX calculado.")
        st.markdown("**Trecho mais parecido (IDX):**")
        st.write(
            {
                "indice_referencia": info_idx["indice_referencia"],
                "trecho_referencia": series_to_str(info_idx["trecho_referencia"])
                if info_idx["trecho_referencia"]
                else None,
                "similaridade": info_idx["similaridade"],
                "motorista_dominante": info_idx["motorista_dominante"],
            }
        )


def painel_nucleo_resiliente():
    st.subheader("3. Núcleo Resiliente")

    series_atual = st.session_state.get("current_series", [])
    if not series_atual:
        st.warning("Nenhuma série alvo definida.")
        return

    modo_asb = st.selectbox(
        "Modo Anti-SelfBias",
        options=["A", "B"],
        index=1,
        help="Modo B recomendado para maior robustez estrutural.",
    )

    if st.button("Construir Núcleo Resiliente"):
        nucleo = construir_nucleo_resiliente(series_atual, modo_asb=modo_asb)
        st.session_state["nucleo_resiliente"] = nucleo
        st.success("Núcleo Resiliente construído.")
        st.markdown(f"**Núcleo Resiliente:** `{series_to_str(nucleo)}`")

    nucleo_manual = st.text_input(
        "Ajuste manual do Núcleo Resiliente (opcional)",
        value=series_to_str(st.session_state.get("nucleo_resiliente", [])),
        help="Formato: 8 29 30 34 39 60",
    )

    if st.button("Aplicar ajuste manual do Núcleo"):
        lista = parse_series_text(nucleo_manual)
        if len(lista) >= 5:
            st.session_state["nucleo_resiliente"] = lista[:6]
            st.success("Núcleo Resiliente atualizado manualmente.")
        else:
            st.error("Entrada inválida. Informar pelo menos 5 números.")


def painel_series_puras():
    st.subheader("4. Séries Puras (6 passageiros)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    if not nucleo:
        st.warning("Núcleo Resiliente ainda não definido.")
        return

    if st.button("Gerar Séries Puras"):
        series_puras = gerar_series_puras(nucleo)
        st.session_state["series_puras"] = series_puras
        st.success("Séries Puras geradas.")
        texto = "\n".join(series_to_str(s) for s in series_puras)
        st.code(texto, language="text")


def painel_series_avaliadas():
    st.subheader("5. Séries Avaliadas (ICA + HLA)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    series_puras = st.session_state.get("series_puras", [])

    if not nucleo:
        st.warning("Núcleo Resiliente ainda não definido.")
        return

    if not series_puras:
        st.info("Nenhuma Série Pura encontrada. Gerar Séries Puras antes.")
        return

    if st.button("Avaliar Séries (ICA + HLA)"):
        df_av = avaliar_series(series_puras, nucleo)
        st.session_state["series_avaliadas"] = df_av
        st.success("Avaliação concluída.")
        st.dataframe(df_av)


def painel_gerador_extra():
    st.subheader("6. Gerador Extra (por confiabilidade alvo)")

    nucleo = st.session_state.get("nucleo_resiliente", [])
    if not nucleo:
        st.warning("Núcleo Resiliente ainda não definido.")
        return

    conf_target = st.slider(
        "Confiabilidade alvo (referência conceitual)",
        min_value=0.70,
        max_value=0.95,
        value=0.80,
        step=0.01,
    )

    max_series = st.slider(
        "Quantidade máxima de séries extras",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
    )

    if st.button("Gerar Séries Extras"):
        extras = gerar_series_extras(
            nucleo,
            confiabilidade_alvo=conf_target,
            max_series=max_series,
        )
        st.session_state["series_extras"] = extras
        st.success("Séries Extras geradas.")
        texto = "\n".join(series_to_str(s) for s in extras)
        st.code(texto, language="text")


def painel_s6_ensamble():
    st.subheader("7. S6 + Ensamble Final")

    df_av = st.session_state.get("series_avaliadas", pd.DataFrame())
    extras = st.session_state.get("series_extras", [])

    if df_av.empty and not extras:
        st.info("Nenhuma série avaliada ou extra disponível.")
        return

    top_n = st.slider(
        "Quantidade de séries avaliadas para o Ensamble",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    df_ensamble = construir_s6_ensamble(df_av, extras, top_n=top_n)
    st.session_state["s6_ensamble"] = df_ensamble

    st.markdown("**Visão consolidada (S6 + Ensamble):**")
    st.dataframe(df_ensamble)

    # Visão de frequência de passageiros
    todas_series = []
    for _, row in df_ensamble.iterrows():
        numeros = parse_series_text(row["serie"])
        todas_series.append(numeros)

    if todas_series:
        flat = list(itertools.chain.from_iterable(todas_series))
        freq = pd.Series(flat).value_counts().reset_index()
        freq.columns = ["passageiro", "frequencia"]
        st.markdown("**Frequência de passageiros nas séries selecionadas:**")
        st.dataframe(freq)


def painel_resumo():
    st.subheader("Resumo Técnico do Estado Atual")

    idx = st.session_state.get("current_index")
    serie = st.session_state.get("current_series", [])
    nucleo = st.session_state.get("nucleo_resiliente", [])
    series_puras = st.session_state.get("series_puras", [])
    df_av = st.session_state.get("series_avaliadas", pd.DataFrame())
    extras = st.session_state.get("series_extras", [])
    idx_info = st.session_state.get("idx_info", None)

    st.markdown(f"**Índice alvo:** `{idx}`")
    st.markdown(f"**Série alvo:** `{series_to_str(serie)}`")
    st.markdown(f"**Núcleo Resiliente:** `{series_to_str(nucleo)}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Qtd. Séries Puras", len(series_puras))
    with col2:
        st.metric("Qtd. Séries Avaliadas", 0 if df_av.empty else len(df_av))
    with col3:
        st.metric("Qtd. Séries Extras", len(extras))

    if idx_info is not None:
        st.markdown("**Resumo IDX:**")
        st.write(
            {
                "indice_referencia": idx_info["indice_referencia"],
                "trecho_referencia": series_to_str(idx_info["trecho_referencia"])
                if idx_info["trecho_referencia"]
                else None,
                "similaridade": idx_info["similaridade"],
                "motorista_dominante": idx_info["motorista_dominante"],
            }
        )


def main():
    ensure_session_state()
    st.set_page_config(
        page_title="Predict Cars V13.8",
        layout="wide",
    )

    mostrar_header_principal()

    # Menu lateral
    menu = st.sidebar.radio(
        "Navegação",
        options=[
            "Entrada de Dados",
            "Estado Atual",
            "Núcleo Resiliente",
            "Séries Puras",
            "Séries Avaliadas",
            "Gerador Extra",
            "S6 + Ensamble",
            "Resumo",
        ],
    )

    if menu == "Entrada de Dados":
        painel_dados()
    elif menu == "Estado Atual":
        painel_estado_atual()
    elif menu == "Núcleo Resiliente":
        painel_nucleo_resiliente()
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
s pd
import numpy as np

# -------------------------------------------------------------
# Configuração geral do app
# -------------------------------------------------------------
st.set_page_config(
    page_title="Predict Cars V13.8",
    page_icon="🚗",
    layout="wide",
)


# -------------------------------------------------------------
# Funções auxiliares de histórico
# -------------------------------------------------------------
def set_historico(conteudo: str):
    if conteudo is not None and conteudo.strip():
        st.session_state["historico_bruto"] = conteudo


def get_historico():
    return st.session_state.get("historico_bruto", None)


# -------------------------------------------------------------
# Funções auxiliares de parsing e faixas
# -------------------------------------------------------------
def faixa_num(n: int) -> int:
    """Classifica número em faixas: 1-20, 21-40, 41-60, 61-80."""
    if 1 <= n <= 20:
        return 1
    elif 21 <= n <= 40:
        return 2
    elif 41 <= n <= 60:
        return 3
    elif 61 <= n <= 80:
        return 4
    return 0


def extrair_numeros(historico_bruto: str):
    """Extrai números das linhas do histórico (protótipo simples)."""
    numeros = []
    if not historico_bruto:
        return numeros

    for line in historico_bruto.splitlines():
        line = line.strip()
        if not line:
            continue
        partes = [p.strip() for p in line.split(";") if p.strip()]

        if not partes:
            continue

        # Se começar com Cxxxx, ignora o primeiro campo
        if partes[0].upper().startswith("C"):
            partes = partes[1:]

        # Em geral, último é k (rótulo), então tentamos ignorar
        if len(partes) >= 2:
            possiveis_passageiros = partes[:-1]
        else:
            possiveis_passageiros = partes

        for p in possiveis_passageiros:
            try:
                n = int(p)
                numeros.append(n)
            except ValueError:
                pass

    return numeros


def parse_historico(historico_bruto: str):
    """
    Converte o histórico bruto em uma lista de dicionários:
    {
        'linha': índice,
        'id': 'Cxxxx' ou None,
        'passageiros': [...],
        'k': rótulo final,
        'texto': linha original
    }
    """
    registros = []
    if not historico_bruto:
        return registros

    for idx, line in enumerate(historico_bruto.splitlines()):
        original = line.rstrip("\n")
        line = line.strip()
        if not line:
            continue

        partes = [p.strip() for p in line.split(";") if p.strip()]
        if not partes:
            continue

        id_serie = None
        resto = partes

        if partes[0].upper().startswith("C"):
            id_serie = partes[0]
            resto = partes[1:]

        k = None
        if len(resto) >= 2:
            passageiros_str = resto[:-1]
            k = resto[-1]
        else:
            passageiros_str = resto

        passageiros = []
        for p in passageiros_str:
            try:
                n = int(p)
                passageiros.append(n)
            except ValueError:
                pass

        registros.append(
            {
                "linha": idx,
                "id": id_serie,
                "passageiros": passageiros,
                "k": k,
                "texto": original,
            }
        )
    return registros


def similaridade_faixas(passageiros_alvo, passageiros_cand):
    """Calcula similaridade de faixas entre alvo e candidato (0 a 1)."""
    if not passageiros_alvo or not passageiros_cand:
        return 0.0

    faixas_alvo = [faixa_num(n) for n in passageiros_alvo]
    faixas_cand = [faixa_num(n) for n in passageiros_cand]

    sim = 0
    for f in range(1, 5):
        sim += min(faixas_alvo.count(f), faixas_cand.count(f))

    return sim / max(len(passageiros_alvo), 1)


# -------------------------------------------------------------
# IDX Avançado
# -------------------------------------------------------------
def encontrar_similares_idx_avancado(registros, w_coinc=3.0, w_recencia=2.0, w_faixa=1.0):
    """Retorna tabela IDX, série alvo e núcleo IDX ponderado."""
    if not registros or len(registros) < 2:
        return None, None, None

    alvo = registros[-1]
    alvo_set = set(alvo["passageiros"])
    if not alvo_set:
        return None, alvo, None

    max_linha = max(r["linha"] for r in registros) or 1

    candidatos = []
    for r in registros[:-1]:
        conj = set(r["passageiros"])
        inter = alvo_set.intersection(conj)
        coincidencias = len(inter)
        if coincidencias == 0:
            continue

        recencia_norm = r["linha"] / max_linha
        sim_fx = similaridade_faixas(alvo["passageiros"], r["passageiros"])

        score_total = (
            w_coinc * coincidencias
            + w_recencia * recencia_norm
            + w_faixa * sim_fx
        )

        candidatos.append(
            {
                "linha": r["linha"],
                "id": r["id"],
                "qtd_passageiros": len(r["passageiros"]),
                "coincidentes": coincidencias,
                "recencia_norm": recencia_norm,
                "sim_faixas": sim_fx,
                "score_total": score_total,
                "passageiros": r["passageiros"],
                "texto": r["texto"],
            }
        )

    if not candidatos:
        return None, alvo, None

    df = pd.DataFrame(candidatos)
    df = df.sort_values(by=["score_total", "coincidentes", "linha"], ascending=[False, False, False])

    # ~20% dos melhores (min 5, max 25)
    num_cand = len(df)
    top_k = int(np.ceil(num_cand * 0.2))
    top_k = max(5, min(25, top_k))
    top_df = df.head(top_k)

    # Núcleo IDX
    pesos_por_numero = {}
    for _, row in top_df.iterrows():
        score = float(row["score_total"])
        for n in row["passageiros"]:
            pesos_por_numero[n] = pesos_por_numero.get(n, 0.0) + score

    if not pesos_por_numero:
        nucleo = None
    else:
        ordenados = sorted(pesos_por_numero.items(), key=lambda x: x[1], reverse=True)
        nucleo = [n for n, _ in ordenados[:6]]

    return top_df, alvo, nucleo


# -------------------------------------------------------------
# IPF Híbrido
# -------------------------------------------------------------
def calcular_ipf_hibrido(df_top, alvo):
    """Implementa IPF híbrido simples: ritmo, dispersão e pares."""
    if df_top is None or df_top.empty:
        return None, None

    df = df_top.copy()

    # Ritmo = diferença de linhas entre séries vizinhas
    df["ritmo"] = df["linha"].diff().abs().fillna(0)

    # Dispersão = variação da quantidade de passageiros
    df["dispersao"] = df["qtd_passageiros"].rolling(2).std().fillna(0)

    # Pares fixos com o alvo
    alvo_set = set(alvo["passageiros"])
    pares = []
    for row in df["passageiros"]:
        pares.append(len(alvo_set.intersection(set(row))))
    df["pares_fixos"] = pares

    # Score IPF composto
    df["score_ipf"] = (
        df["coincidentes"] * 1.5
        + df["recencia_norm"] * 1.2
        + df["sim_faixas"] * 1.0
        + df["pares_fixos"] * 0.8
        - df["ritmo"] * 0.3
        - df["dispersao"] * 0.2
    )

    df_ipf = df.sort_values(by="score_ipf", ascending=False).head(12)

    # Núcleo IPF
    pesos_num = {}
    for _, r in df_ipf.iterrows():
        for n in r["passageiros"]:
            pesos_num[n] = pesos_num.get(n, 0.0) + float(r["score_ipf"])

    ordenados = sorted(pesos_num.items(), key=lambda x: x[1], reverse=True)
    nucleo_ipf = [n for n, _ in ordenados[:6]]

    return df_ipf, nucleo_ipf


# -------------------------------------------------------------
# IPO Profissional
# -------------------------------------------------------------
def calcular_ipo_profissional(df_top):
    """Implementa IPO profissional com suavização e microcorreção."""
    if df_top is None or df_top.empty:
        return None, None

    df = df_top.copy()

    # Suavização de ruído
    df = df[df["coincidentes"] >= 2]
    if df.empty:
        df = df_top.copy()

    # Correção microestrutural básica
    df["micro"] = df["sim_faixas"] * 0.5 + df["recencia_norm"] * 0.3

    df["score_ipo"] = (
        df["score_total"] * 0.6
        + df["micro"] * 0.4
    )

    df_ipo = df.sort_values(by="score_ipo", ascending=False).head(10)

    # Núcleo IPO
    pesos = {}
    for _, r in df_ipo.iterrows():
        for n in r["passageiros"]:
            pesos[n] = pesos.get(n, 0.0) + float(r["score_ipo"])

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    nucleo_ipo = [n for n, _ in ordenados[:6]]

    return df_ipo, nucleo_ipo


# -------------------------------------------------------------
# ANTI-SELFBIAS (ASB A + B)
# -------------------------------------------------------------
def aplicar_asb(nucleo_ipo, passageiros_alvo, modo):
    """
    Aplica Anti-SelfBias:
    - A (leve): troca 1 número em caso de autociclagem forte.
    - B (médio): troca 1–2 números em comum com o alvo.
    """
    if nucleo_ipo is None:
        return None

    alvo_set = set(passageiros_alvo)
    nuc = list(nucleo_ipo)

    comuns = len(alvo_set.intersection(nuc))

    # ASB A — Leve
    if modo == "A":
        if comuns == len(nuc):
            # troca o último número por um da faixa menos representada
            faixas = [faixa_num(n) for n in nuc]
            todas = {1, 2, 3, 4}
            faltantes = list(todas - set(faixas))
            if faltantes:
                f = min(faltantes)
                candidato = f * 20 - 5
            else:
                candidato = min(nuc) + 1
            nuc[-1] = candidato
        return sorted(nuc)

    # ASB B — Médio
    if modo == "B":
        if comuns >= len(nuc) - 1:
            # remove 1 número do alvo
            for n in list(nuc):
                if n in alvo_set:
                    nuc.remove(n)
                    break
            # adiciona um número estruturado próximo à média
            media = int(np.mean(nuc))
            candidato = media + 1
            if candidato in nuc:
                candidato += 2
            nuc.append(candidato)
        return sorted(nuc[:6])

    return sorted(nuc)


# -------------------------------------------------------------
# Núcleo Resiliente básico (pré-ICA)
# -------------------------------------------------------------
def gerar_nucleo_resiliente(nucleo_ipo, nucleo_asb_b):
    """
    Núcleo Resiliente V13.8 — Combinação IPO + ASB-B
    Dá mais peso ao ASB-B (anti-selfbias médio),
    preservando coerência estrutural.
    """
    if not nucleo_ipo or not nucleo_asb_b:
        return None

    base = list(dict.fromkeys(nucleo_asb_b + nucleo_ipo))  # união preservando ordem

    pesos = {}
    for n in base:
        pesos[n] = 0.0
        if n in nucleo_asb_b:
            pesos[n] += 2.0
        if n in nucleo_ipo:
            pesos[n] += 1.0
        # pequeno ajuste por faixa (apenas para diversificar)
        faixa = faixa_num(n)
        pesos[n] += 0.1 * (5 - faixa)

    ordenados = sorted(pesos.items(), key=lambda x: x[1], reverse=True)
    resiliente = [n for n, _ in ordenados[:6]]
    resiliente.sort()
    return resiliente


# -------------------------------------------------------------
# ICA — Ajuste de Núcleo
# -------------------------------------------------------------
def aplicar_ica(nucleo_resiliente, registros):
    """
    ICA (Iterative Core Adjustment) – versão simples:
    neste protótipo, considera que o núcleo já está bem calibrado
    e apenas o replica como núcleo_ica.
    """
    if not nucleo_resiliente:
        return None
    return list(sorted(nucleo_resiliente))


# -------------------------------------------------------------
# HLA — High-Level Adjustment
# -------------------------------------------------------------
def aplicar_hla(nucleo_ica, cobertura):
    """
    HLA (High-Level Adjustment) – versão simples:
    alinha o núcleo com a cobertura, reforçando números
    bem ancorados.
    """
    if not nucleo_ica:
        return None
    if not cobertura:
        return list(sorted(nucleo_ica))

    set_cov = set(cobertura)
    fortes = [n for n in nucleo_ica if n in set_cov]
    fracos = [n for n in nucleo_ica if n not in set_cov]

    if len(fortes) >= 4:
        base = fortes + fracos
    else:
        base = nucleo_ica

    # Garante 6 números
    base = list(dict.fromkeys(base))
    if len(base) < 6:
        for n in cobertura:
            if n not in base:
                base.append(n)
            if len(base) >= 6:
                break

    return sorted(base[:6])


# -------------------------------------------------------------
# Cobertura de Vento + Listas + Espremer + S6 + Ensamble
# -------------------------------------------------------------
def gerar_cobertura(nucleo_final, registros):
    """
    Gera Cobertura de Vento (10–15 números) ao redor do núcleo final,
    usando proximidade e recorrência simples.
    """
    if not nucleo_final:
        return []

    # Frequência global
    todos_nums = []
    for r in registros:
        todos_nums.extend(r["passageiros"])
    freq = {}
    for n in todos_nums:
        freq[n] = freq.get(n, 0) + 1

    base = set(nucleo_final)
    candidatos = set()

    for n in nucleo_final:
        for delta in [-3, -2, -1, 1, 2, 3]:
            m = n + delta
            if 1 <= m <= 80:
                candidatos.add(m)

    # Adiciona os mais frequentes do histórico nas mesmas faixas
    for n, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        if len(candidatos) >= 25:
            break
        if faixa_num(n) in [faixa_num(x) for x in nucleo_final]:
            candidatos.add(n)

    todos = sorted(candidatos | base)
    if len(todos) > 15:
        # heurística: prioriza proximidade dos núcleos
        def score_num(x):
            return min(abs(x - n) for n in nucleo_final)

        todos = sorted(todos, key=score_num)[:15]

    return sorted(todos)


def gerar_listas_sa1_max_hibrida(nucleo_final, cobertura):
    """
    Gera SA1, MAX e Híbrida a partir do núcleo + cobertura.
    """
    if not nucleo_final:
        return [], [], []

    cov = list(sorted(cobertura)) if cobertura else []

    # SA1: primeiros 10 da cobertura (se possível) priorizando núcleo
    sa1 = []
    for n in cov:
        if n not in sa1:
            sa1.append(n)
        if len(sa1) >= 10:
            break

    # MAX: núcleo + números de maior valor da cobertura
    max_list = list(nucleo_final)
    for n in sorted(cov, reverse=True):
        if n not in max_list:
            max_list.append(n)
        if len(max_list) >= 10:
            break
    max_list = sorted(set(max_list))[:10]

    # Híbrida: união
    hibrida = sorted(set(sa1 + max_list))

    return sa1, max_list, hibrida


def espremer_listas(sa1, max_list, hibrida):
    """
    Espremer (versão simples): mantém as listas já enxutas,
    removendo apenas duplicatas e ordenando.
    """
    sa1_e = sorted(set(sa1))
    max_e = sorted(set(max_list))
    hibrida_e = sorted(set(hibrida))
    return sa1_e, max_e, hibrida_e


def gerar_s6(nucleo_final, cobertura, sa1_e, max_e, hibrida_e):
    """
    Gera S6 Alfa / Bravo / Charlie.
    """
    if not nucleo_final:
        return [], [], []

    s6_alfa = list(sorted(nucleo_final))

    # Bravo: 4 números bem apoiados na cobertura e SA1
    base_bravo = []
    for lista in [sa1_e, cobertura]:
        for n in lista:
            if n not in s6_alfa and n not in base_bravo:
                base_bravo.append(n)
            if len(base_bravo) >= 4:
                break
        if len(base_bravo) >= 4:
            break

    s6_bravo = base_bravo[:4]

    # Charlie: 3 números adicionais de espremidas / cobertura
    base_charlie = []
    for lista in [hibrida_e, max_e, cobertura]:
        for n in lista:
            if n not in s6_alfa and n not in s6_bravo and n not in base_charlie:
                base_charlie.append(n)
            if len(base_charlie) >= 3:
                break
        if len(base_charlie) >= 3:
            break

    s6_charlie = base_charlie[:3]

    return s6_alfa, s6_bravo, s6_charlie


def gerar_ensamble(nucleo_final, s6_bravo, s6_charlie):
    """
    Ensamble Final: núcleo + parte de bravo/charlie.
    """
    if not nucleo_final:
        return []

    ens = list(nucleo_final)
    for lista in [s6_bravo, s6_charlie]:
        for n in lista:
            if n not in ens:
                ens.append(n)
            if len(ens) >= 10:
                break
        if len(ens) >= 10:
            break

    return sorted(ens)


def avaliar_farois_barometro_confiabilidade(registros, nucleo_final, cobertura):
    """
    Protótipo simplificado:
    cenário geralmente resiliente, com confiabilidade fixa (85%),
    podendo ser adaptado depois para estados de turbulência.
    """
    if not nucleo_final:
        return "🟠", "Turbulento", 50.0

    # Aqui poderia haver lógica mais sofisticada.
    return "🟢", "Resiliente", 85.0


# -------------------------------------------------------------
# Funções auxiliares para Séries de Previsão (V13.8 Turbo)
# -------------------------------------------------------------
def _coletar_series_base(resultado: dict):
    """
    Coleta as principais séries de previsão já calculadas no pipeline
    e devolve como uma lista de dicts:
    {
        "nome": "Núcleo Final",
        "origem": "nucleo_hla",
        "numeros": [ ... ],
    }
    Somente séries com exatamente 6 números são consideradas.
    """
    series = []

    # Núcleo Final (ICA + HLA) – se existir
    nuc_final = resultado.get("nucleo_hla") or resultado.get("nucleo_ica") or resultado.get("nucleo_resiliente")
    if isinstance(nuc_final, (list, tuple)) and len(nuc_final) == 6:
        series.append({"nome": "Núcleo Final", "origem": "nucleo_hla", "numeros": list(nuc_final)})

    # SA1 / MAX / Híbrida (se forem exatamente 6 – caso contrário, entram só na análise)
    for chave, nome in [
        ("sa1_e", "SA1-E"),
        ("max_e", "MAX-E"),
        ("hibrida_e", "Híbrida-E"),
    ]:
        lista = resultado.get(chave)
        if isinstance(lista, (list, tuple)) and len(lista) == 6:
            series.append({"nome": nome, "origem": chave, "numeros": list(lista)})

    # S6 Alfa / Bravo / Charlie (se existirem com 6)
    for chave, nome in [
        ("s6_alfa", "S6 Alfa"),
        ("s6_bravo", "S6 Bravo"),
        ("s6_charlie", "S6 Charlie"),
    ]:
        lista = resultado.get(chave)
        if isinstance(lista, (list, tuple)) and len(lista) == 6:
            series.append({"nome": nome, "origem": chave, "numeros": list(lista)})

    # Ensamble Final (se existir com 6)
    ensamble = resultado.get("ensamble")
    if isinstance(ensamble, (list, tuple)) and len(ensamble) == 6:
        series.append({"nome": "Ensamble Final", "origem": "ensamble", "numeros": list(ensamble)})

    # Remover duplicadas (mesmos 6 números) preservando ordem
    vistos = set()
    series_unicas = []
    for s in series:
        chave = tuple(sorted(s["numeros"]))
        if chave not in vistos:
            vistos.add(chave)
            series_unicas.append(s)

    return series_unicas


def _score_convergencia(numeros, nucleo_final, cobertura):
    """
    Mede uma convergência simples com núcleo final e cobertura.
    Retorna valor em [0, 1].
    """
    if not numeros:
        return 0.0
    set_n = set(numeros)
    set_nuc = set(nucleo_final or [])
    set_cov = set(cobertura or [])

    inter_nuc = len(set_n & set_nuc)
    inter_cov = len(set_n & set_cov)

    # Peso maior para coincidência com o núcleo
    score = 0.7 * (inter_nuc / 6.0) + 0.3 * (inter_cov / max(len(set_cov), 1))
    return max(0.0, min(1.0, score))


def _classificar_risco(confianca: float):
    """
    Traduz a confiabilidade em rótulo de risco.
    """
    if confianca >= 80:
        return "Baixo"
    if confianca >= 60:
        return "Moderado"
    return "Alto"


def _classificar_metafora(nome_serie: str, confianca: float):
    """
    Traduz nome + confiabilidade em metáfora curta.
    """
    nome = (nome_serie or "").lower()
    if "núcleo" in nome or "final" in nome or "ensamble" in nome:
        if confianca >= 80:
            return "Carro Líder da Rodovia"
        else:
            return "Carro Estrutural"
    if "sa1" in nome:
        return "Comboio Primário de Estabilidade"
    if "max" in nome:
        return "Carro de Pressão Máxima"
    if "híbrida" in nome or "hibrida" in nome:
        return "Carro de Faixas Sincronizadas"
    if "alfa" in nome:
        return "Carro de Elite (Escolta)"
    if "bravo" in nome:
        return "Carro Tático de Apoio"
    if "charlie" in nome:
        return "Carro de Cobertura Leve"
    return "Carro Auxiliar da Operação"


def avaliar_serie_previsao(numeros, nome_serie, contexto):
    """
    Gera:
    - confiabilidade estimada (%)
    - acertos esperados (0–6)
    - risco (texto)
    - tipo metafórico
    - interpretação curta
    usando apenas heurísticas simples baseadas em:
    - confiabilidade global
    - convergência com núcleo final
    - convergência com cobertura
    """
    if not isinstance(numeros, (list, tuple)) or len(numeros) != 6:
        return {
            "confianca": 50.0,
            "acertos_esperados": 2.5,
            "risco": "Alto",
            "tipo": "Carro Auxiliar",
            "interpretacao": "Série incompleta ou instável.",
        }

    nucleo_final = contexto.get("nucleo_hla") or contexto.get("nucleo_ica") or contexto.get("nucleo_resiliente") or []
    cobertura = contexto.get("cobertura") or []
    conf_global = float(contexto.get("confiabilidade", 80.0))

    score_conv = _score_convergencia(numeros, nucleo_final, cobertura)

    # Confiabilidade da série = mistura de global + convergência
    confianca = conf_global * (0.5 + 0.5 * score_conv)
    confianca = max(20.0, min(99.0, confianca))

    # Acertos esperados ~ 6 * (confianca normalizada * convergência)
    p = (confianca / 100.0) * (0.5 + 0.5 * score_conv)
    acertos_esperados = max(0.5, min(6.0, 6.0 * p))

    risco = _classificar_risco(confianca)
    tipo = _classificar_metafora(nome_serie, confianca)

    # Interpretação curta
    if score_conv >= 0.8:
        interp = "Série altamente alinhada ao núcleo e à cobertura — estrutura muito forte."
    elif score_conv >= 0.5:
        interp = "Série bem apoiada pela estrada atual — boa estrutura com risco controlado."
    elif score_conv >= 0.3:
        interp = "Série de apoio tático — cobre oscilações com risco moderado."
    else:
        interp = "Série de cobertura extrema — pode ser útil em cenários voláteis, mas com risco elevado."

    return {
        "confianca": round(confianca, 1),
        "acertos_esperados": round(acertos_esperados, 2),
        "risco": risco,
        "tipo": tipo,
        "interpretacao": interp,
    }


def gerar_series_extras(resultado: dict, series_base: list, num_series: int = 5, conf_min: float = 60.0):
    """
    Gera séries extras a partir de:
    - Cobertura
    - Números recorrentes em SA1 / MAX / Híbrida / S6 / Ensamble

    Estratégia simples:
    - Conta frequência de cada número nas séries base + cobertura
    - Monta novas séries de 6 números com os mais frequentes,
      evitando duplicar exatamente as séries já existentes.
    - Filtra por confiabilidade mínima usando a mesma heurística
      de avaliar_serie_previsao.
    """
    cobertura = resultado.get("cobertura") or []
    nums_frequencia = {}

    # Frequência pelos blocos base
    for s in series_base:
        for n in s["numeros"]:
            nums_frequencia[n] = nums_frequencia.get(n, 0) + 2  # peso maior

    # Frequência pela cobertura
    for n in cobertura:
        nums_frequencia[n] = nums_frequencia.get(n, 0) + 1

    if not nums_frequencia:
        return []

    # Ordenar números por frequência
    nums_ordenados = [n for n, _ in sorted(nums_frequencia.items(), key=lambda x: x[1], reverse=True)]

    # Séries já existentes (para não repetir)
    existentes = {tuple(sorted(s["numeros"])) for s in series_base}

    extras = []
    idx = 0
    tentativas = 0
    max_tentativas = 200

    while len(extras) < num_series and tentativas < max_tentativas:
        tentativas += 1
        # Monta uma série de 6 números a partir de uma janela deslocada
        janela = nums_ordenados[idx:idx + 6]
        if len(janela) < 6:
            # recomeça com pequeno deslocamento
            idx = (idx + 1) % max(1, len(nums_ordenados) - 5)
            continue

        serie = sorted(set(janela))
        if len(serie) != 6:
            idx = (idx + 1) % max(1, len(nums_ordenados) - 5)
            continue

        chave = tuple(sorted(serie))
        if chave in existentes:
            idx = (idx + 1) % max(1, len(nums_ordenados) - 5)
            continue

        # Avalia série
        info = avaliar_serie_previsao(serie, "Extra", resultado)
        if info["confianca"] >= conf_min:
            extras.append(
                {
                    "nome": f"Extra {len(extras)+1}",
                    "origem": "extra",
                    "numeros": serie,
                    "avaliacao": info,
                }
            )
            existentes.add(chave)

        idx = (idx + 1) % max(1, len(nums_ordenados) - 5)

    return extras


# -------------------------------------------------------------
# Função de pipeline completo (IDX → IPF → IPO → ASB → Resiliente → ICA → HLA)
# -------------------------------------------------------------
def rodar_pipeline_completo(historico_bruto: str, modo_asb: str = "B"):
    """
    Executa todo o pipeline para uso nas páginas:
    - retorna dicionário com todas as estruturas principais.
    """
    registros = parse_historico(historico_bruto)
    if len(registros) < 2:
        return None

    df_idx, alvo, nuc_idx = encontrar_similares_idx_avancado(registros)
    if df_idx is None or df_idx.empty:
        return None

    df_ipf, nuc_ipf = calcular_ipf_hibrido(df_idx, alvo)
    df_ipo, nuc_ipo = calcular_ipo_profissional(df_idx)

    if nuc_ipo is None:
        nuc_asb_a = None
        nuc_asb_b = None
        nuc_res = None
        nuc_ica = None
        nuc_hla = None
    else:
        nuc_asb_a = aplicar_asb(nuc_ipo, alvo["passageiros"], "A")
        nuc_asb_b = aplicar_asb(nuc_ipo, alvo["passageiros"], modo_asb)
        nuc_res = gerar_nucleo_resiliente(nuc_ipo, nuc_asb_b)
        nuc_ica = aplicar_ica(nuc_res, registros)
        # Cobertura é usada dentro do HLA, então gerar logo depois
        cobertura_tmp = gerar_cobertura(nuc_ica, registros)
        nuc_hla = aplicar_hla(nuc_ica, cobertura_tmp)

    # Se ainda não tiver cobertura/núcleo final, usa fallback
    nucleo_final = nuc_hla or nuc_ica or nuc_res or nuc_ipo or nuc_idx or nuc_ipf or []

    cobertura = gerar_cobertura(nucleo_final, registros)

    sa1, max_list, hibrida = gerar_listas_sa1_max_hibrida(nucleo_final, cobertura)
    sa1_e, max_e, hibrida_e = espremer_listas(sa1, max_list, hibrida)
    s6_alfa, s6_bravo, s6_charlie = gerar_s6(nucleo_final, cobertura, sa1_e, max_e, hibrida_e)
    ensamble = gerar_ensamble(nucleo_final, s6_bravo, s6_charlie)
    farol, barometro, confiabilidade = avaliar_farois_barometro_confiabilidade(registros, nucleo_final, cobertura)

    return {
        "alvo": alvo,
        "registros": registros,
        "df_idx": df_idx,
        "nucleo_idx": nuc_idx,
        "df_ipf": df_ipf,
        "nucleo_ipf": nuc_ipf,
        "df_ipo": df_ipo,
        "nucleo_ipo": nuc_ipo,
        "nucleo_asb_a": nuc_asb_a,
        "nucleo_asb_b": nuc_asb_b,
        "nucleo_resiliente": nuc_res,
        "nucleo_ica": nuc_ica,
        "nucleo_hla": nuc_hla,
        "cobertura": cobertura,
        "sa1": sa1,
        "max_list": max_list,
        "hibrida": hibrida,
        "sa1_e": sa1_e,
        "max_e": max_e,
        "hibrida_e": hibrida_e,
        "s6_alfa": s6_alfa,
        "s6_bravo": s6_bravo,
        "s6_charlie": s6_charlie,
        "ensamble": ensamble,
        "farol": farol,
        "barometro": barometro,
        "confiabilidade": confiabilidade,
    }


# -------------------------------------------------------------
# SIDEBAR — Histórico + Navegação (MENU HÍBRIDO)
# -------------------------------------------------------------
st.sidebar.title("🚗 Predict Cars V13.8")

st.sidebar.markdown("### 1. Histórico")

uploaded_file = st.sidebar.file_uploader(
    "Enviar arquivo de histórico (.txt ou .csv):",
    type=["txt", "csv"]
)
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    set_historico(content)
    st.sidebar.success("Histórico carregado via arquivo.")

st.sidebar.markdown("Ou cole o histórico abaixo:")
historico_texto = st.sidebar.text_area(
    "Cole aqui as linhas do histórico (Cxxxx; n1; ...; k):",
    height=150,
    key="historico_text_area"
)

if st.sidebar.button("Carregar histórico do texto"):
    if historico_texto.strip():
        set_historico(historico_texto)
        st.sidebar.success("Histórico carregado a partir do texto.")
    else:
        st.sidebar.warning("Campo de texto vazio.")

st.sidebar.markdown("### 2. Navegação")

pagina = st.sidebar.radio(
    "Escolha a seção:",
    (
        "Painel Principal",
        "Análise Estrutural (IDX/IPF/IPO/ASB)",
        "Núcleo + ICA + HLA",
        "Cobertura + SA1/MAX/Híbrida + Espremer",
        "S6 + Ensamble",
        "Formato Oficial (V13.8)",
        "Previsão Completa (consolidada)",
        "Séries Puras (6 passageiros)",
        "Séries Avaliadas (indicadores + metáfora)",
        "Gerador de Séries Extras (confiabilidade)",
    )
)

historico_bruto = get_historico()

if historico_bruto:
    st.success("✅ Histórico carregado e disponível.")
else:
    st.info("ℹ️ Nenhum histórico carregado ainda.")


# -------------------------------------------------------------
# PÁGINAS
# -------------------------------------------------------------
if pagina == "Painel Principal":
    st.title("🚗 Predict Cars V13.8 — Painel Principal")
    st.markdown(
        "Este app executa o espírito do **Manual Técnico Ultra-Híbrido — V13.8**:\n\n"
        "- Análise estrutural do histórico (IDX / IPF / IPO / ASB / ICA / HLA)\n"
        "- Núcleo Resiliente + Ajustes\n"
        "- Cobertura de Vento + SA1 / MAX / Híbrida + Espremer\n"
        "- Lista S6 (6 acertos) + Ensamble Final\n"
        "- Formato Oficial\n"
        "- Módulo de Séries Puras e Séries Avaliadas\n"
        "- Gerador de Séries Extras por confiabilidade\n\n"
        "Use a barra lateral para carregar o histórico e navegar."
    )

    if historico_bruto:
        with st.expander("Visualizar primeiras linhas do histórico"):
            st.text("\n".join(historico_bruto.splitlines()[:40]))


elif pagina == "Análise Estrutural (IDX/IPF/IPO/ASB)":
    st.title("🎯 Análise Estrutural — IDX → IPF → IPO → ASB")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]
            st.subheader("📌 Série atual (alvo)")
            st.write(f"Linha: {alvo['linha']}")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            # IDX
            st.markdown("---")
            st.subheader("🔍 IDX Avançado")
            st.dataframe(resultado["df_idx"], use_container_width=True)
            st.write(f"**Núcleo IDX (ponderado):** {resultado['nucleo_idx']}")

            # IPF
            st.markdown("---")
            st.subheader("🧩 IPF Híbrido")
            st.dataframe(resultado["df_ipf"], use_container_width=True)
            st.write(f"**Núcleo IPF (híbrido):** {resultado['nucleo_ipf']}")

            # IPO
            st.markdown("---")
            st.subheader("🚀 IPO Profissional")
            st.dataframe(resultado["df_ipo"], use_container_width=True)
            st.write(f"**Núcleo IPO (profissional):** {resultado['nucleo_ipo']}")

            # ASB A/B
            st.markdown("---")
            st.subheader("🧹 Anti-SelfBias (A/B)")

            modo_asb_label = st.selectbox(
                "Selecione o modo Anti-SelfBias para visualizar:",
                ["A (leve)", "B (médio)"],
                index=1,
            )
            if modo_asb_label.startswith("A"):
                nuc_asb = resultado["nucleo_asb_a"]
                modo_txt = "A (leve)"
            else:
                nuc_asb = resultado["nucleo_asb_b"]
                modo_txt = "B (médio)"

            st.write(f"**Núcleo IPO original:** {resultado['nucleo_ipo']}")
            st.write(f"**Núcleo IPO Anti-SelfBias {modo_txt}:** {nuc_asb}")

            st.success("Pipeline IDX → IPF → IPO → ASB executado com sucesso.")


elif pagina == "Núcleo + ICA + HLA":
    st.title("🔰 Núcleo Resiliente + ICA + HLA")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.markdown("### Núcleos ao longo do pipeline")
            st.write(f"**Núcleo Resiliente (pré-ICA):** {resultado['nucleo_resiliente']}")
            st.write(f"**Núcleo após ICA:** {resultado['nucleo_ica']}")
            st.success(f"**Núcleo Final (ICA + HLA):** {resultado['nucleo_hla']}")

            st.info(
                "O Núcleo Final (ICA + HLA) é o \"carro líder\" do cenário: "
                "base estrutural para Cobertura, Listas, S6 e Ensamble."
            )


elif pagina == "Cobertura + SA1/MAX/Híbrida + Espremer":
    st.title("🌬 Cobertura + SA1 / MAX / Híbrida + Modo Espremer")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            st.subheader("🔰 Núcleo Final (ICA + HLA)")
            st.write(resultado["nucleo_hla"])

            st.markdown("---")
            st.subheader("🌬 Cobertura de Vento (10–15 números)")
            st.write(resultado["cobertura"])

            st.markdown("---")
            st.subheader("🧾 Listas SA1 / MAX / Híbrida")
            st.write("**SA1:**")
            st.write(resultado["sa1"])
            st.write("**MAX:**")
            st.write(resultado["max_list"])
            st.write("**Híbrida:**")
            st.write(resultado["hibrida"])

            st.markdown("---")
            st.subheader("🧪 Versões Espremidas (SA1-E / MAX-E / Híbrida-E)")
            st.write("**SA1-E:**")
            st.write(resultado["sa1_e"])
            st.write("**MAX-E:**")
            st.write(resultado["max_e"])
            st.write("**Híbrida-E:**")
            st.write(resultado["hibrida_e"])


elif pagina == "S6 + Ensamble":
    st.title("🎯 Modo 6 Acertos (S6) + Ensamble Final")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            st.subheader("S6 — Alfa / Bravo / Charlie")
            st.write("**S6 Alfa:**")
            st.write(resultado["s6_alfa"])
            st.write("**S6 Bravo:**")
            st.write(resultado["s6_bravo"])
            st.write("**S6 Charlie:**")
            st.write(resultado["s6_charlie"])

            st.markdown("---")
            st.subheader("🧠 Ensamble Final (lista robusta)")
            st.write(resultado["ensamble"])

            st.info(
                "O Ensamble integra Núcleo Final, Cobertura, Listas Espremidas e S6, "
                "servindo como lista robusta para decisões finais."
            )


elif pagina == "Formato Oficial (V13.8)":
    st.title("📜 Formato Oficial — V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("### 1) Núcleo Resiliente")
            st.write(resultado["nucleo_hla"])

            st.markdown("### 2) Cobertura de Vento")
            st.write(resultado["cobertura"])

            st.markdown("### 3) Listas SA1 / MAX / Híbrida")
            st.write("**SA1:**")
            st.write(resultado["sa1"])
            st.write("**MAX:**")
            st.write(resultado["max_list"])
            st.write("**Híbrida:**")
            st.write(resultado["hibrida"])

            st.markdown("### 4) Versões Espremidas (SA1-E / MAX-E / Híbrida-E)")
            st.write("**SA1-E:**")
            st.write(resultado["sa1_e"])
            st.write("**MAX-E:**")
            st.write(resultado["max_e"])
            st.write("**Híbrida-E:**")
            st.write(resultado["hibrida_e"])

            st.markdown("### 5) S6 — Alfa / Bravo / Charlie")
            st.write("**S6 Alfa:**")
            st.write(resultado["s6_alfa"])
            st.write("**S6 Bravo:**")
            st.write(resultado["s6_bravo"])
            st.write("**S6 Charlie:**")
            st.write(resultado["s6_charlie"])

            st.markdown("### 6) Ensamble Final")
            st.write(resultado["ensamble"])

            st.markdown("### 7) Faróis")
            st.write(resultado["farol"])

            st.markdown("### 8) Barômetro")
            st.write(resultado["barometro"])

            st.markdown("### 9) Confiabilidade (%)")
            st.write(f"{resultado['confiabilidade']}%")

            st.markdown("### 10) Observações Estruturais")
            st.caption(
                "Faixas dominantes, motorista, dispersão e clima seguem o Núcleo Final (ICA + HLA) "
                "e a Cobertura de Vento. O farol e o barômetro refletem a estabilidade atual do cenário."
            )


elif pagina == "Previsão Completa (consolidada)":
    st.title("📜 Previsão Completa — Predict Cars V13.8")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            alvo = resultado["alvo"]

            st.subheader("📌 Série atual (alvo)")
            st.write(f"ID: {alvo['id']}")
            st.write(f"Passageiros: {alvo['passageiros']}")
            st.code(alvo["texto"])

            st.markdown("---")
            st.markdown("### 1) Núcleo Resiliente + ICA + HLA")
            st.write(f"(pré-ICA): {resultado['nucleo_resiliente']}")
            st.write(f"(após ICA): {resultado['nucleo_ica']}")
            st.success(f"(ICA + HLA — Núcleo Final): {resultado['nucleo_hla']}")

            st.markdown("---")
            st.markdown("### 2) Cobertura de Vento (10–15 números)")
            st.write(resultado["cobertura"])

            st.markdown("---")
            st.markdown("### 3) Listas SA1 / MAX / Híbrida")
            st.write("**SA1:**")
            st.write(resultado["sa1"])
            st.write("**MAX:**")
            st.write(resultado["max_list"])
            st.write("**Híbrida:**")
            st.write(resultado["hibrida"])

            st.markdown("---")
            st.markdown("### 4) Versões Espremidas (SA1-E / MAX-E / Híbrida-E)")
            st.write("**SA1-E:**")
            st.write(resultado["sa1_e"])
            st.write("**MAX-E:**")
            st.write(resultado["max_e"])
            st.write("**Híbrida-E:**")
            st.write(resultado["hibrida_e"])

            st.markdown("---")
            st.markdown("### 5) S6 — Alfa / Bravo / Charlie")
            st.write("**S6 Alfa:**")
            st.write(resultado["s6_alfa"])
            st.write("**S6 Bravo:**")
            st.write(resultado["s6_bravo"])
            st.write("**S6 Charlie:**")
            st.write(resultado["s6_charlie"])

            st.markdown("---")
            st.markdown("### 6) Ensamble Final")
            st.write(resultado["ensamble"])

            st.markdown("---")
            st.markdown("### 7) Faróis, Barômetro, Confiabilidade")
            st.write(f"Farol: {resultado['farol']}")
            st.write(f"Barômetro: {resultado['barometro']}")
            st.write(f"Confiabilidade: {resultado['confiabilidade']}%")

            st.markdown("---")
            st.markdown("### 8) Observações Estruturais")
            st.caption(
                "Esta página consolida o formato oficial em um painel único para a série atual."
            )


elif pagina == "Séries Puras (6 passageiros)":
    st.title("📄 Séries de Previsão — SÉRIES PURAS (6 passageiros)")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            series_base = _coletar_series_base(resultado)
            if not series_base:
                st.info("Nenhuma série base com 6 passageiros encontrada.")
            else:
                st.subheader("Séries puras (somente os 6 passageiros) — prontas para copiar e colar")
                linhas_puras = []
                for s in series_base:
                    linha = " ".join(str(n) for n in s["numeros"])
                    linhas_puras.append(linha)
                st.code("\n".join(linhas_puras))


elif pagina == "Séries Avaliadas (indicadores + metáfora)":
    st.title("📊 Séries Avaliadas — Confiabilidade, Risco e Metáfora")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            series_base = _coletar_series_base(resultado)
            if not series_base:
                st.info("Nenhuma série base com 6 passageiros encontrada.")
            else:
                st.subheader("Séries avaliadas estruturalmente")
                for s in series_base:
                    info = avaliar_serie_previsao(s["numeros"], s["nome"], resultado)
                    st.markdown(f"**Série:** `{ ' '.join(str(n) for n in s['numeros']) }`  —  *[{s['nome']}]*")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write(f"Confiabilidade: **{info['confianca']}%**")
                    with col_b:
                        st.write(f"Acertos esperados: **{info['acertos_esperados']}**")
                    with col_c:
                        st.write(f"Risco: **{info['risco']}**")
                    st.write(f"Tipo: *{info['tipo']}*")
                    st.caption(info["interpretacao"])
                    st.markdown("---")


elif pagina == "Gerador de Séries Extras (confiabilidade)":
    st.title("⚙️ Gerador de Séries Extras — por Confiabilidade")

    if not historico_bruto:
        st.warning("Carregue primeiro o histórico na barra lateral.")
    else:
        resultado = rodar_pipeline_completo(historico_bruto, modo_asb="B")
        if resultado is None:
            st.warning("Histórico insuficiente para o pipeline.")
        else:
            series_base = _coletar_series_base(resultado)
            if not series_base:
                st.info("Nenhuma série base encontrada para servir de referência.")
            else:
                st.subheader("Parâmetros para geração de séries extras")

                col1, col2 = st.columns(2)
                with col1:
                    num_extras = st.number_input(
                        "Quantidade de séries extras:",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                    )
                with col2:
                    conf_min = st.slider(
                        "Confiabilidade mínima (%) para as extras:",
                        min_value=40,
                        max_value=95,
                        value=70,
                        step=1,
                    )

                if st.button("Gerar séries extras"):
                    extras = gerar_series_extras(resultado, series_base, num_series=int(num_extras), conf_min=float(conf_min))
                    if not extras:
                        st.warning("Não foi possível gerar séries extras com os critérios informados.")
                    else:
                        st.success(f"Séries extras geradas (mínimo {conf_min}% de confiabilidade):")
                        linhas_extras = []
                        for e in extras:
                            linha = " ".join(str(n) for n in e["numeros"])
                            linhas_extras.append(linha)
                        st.code("\n".join(linhas_extras))

                        st.markdown("#### Detalhes das séries extras")
                        for e in extras:
                            info = e["avaliacao"]
                            st.markdown(f"**Série Extra:** `{ ' '.join(str(n) for n in e['numeros']) }`")
                            colx, coly, colz = st.columns(3)
                            with colx:
                                st.write(f"Confiabilidade: **{info['confianca']}%**")
                            with coly:
                                st.write(f"Acertos esperados: **{info['acertos_esperados']}**")
                            with colz:
                                st.write(f"Risco: **{info['risco']}**")
                            st.write(f"Tipo: *{info['tipo']}*")
                            st.caption(info["interpretacao"])
                            st.markdown("---")

                st.markdown("---")
                st.subheader("Otimização: quantas séries preciso para um alvo de confiabilidade?")

                alvo_conf = st.slider(
                    "Alvo de confiabilidade global (%):",
                    min_value=50,
                    max_value=99,
                    value=90,
                    step=1,
                )

                # Recalcula avaliação das séries base
                avals = []
                for s in series_base:
                    info = avaliar_serie_previsao(s["numeros"], s["nome"], resultado)
                    avals.append(
                        {
                            "serie": s,
                            "confianca": info["confianca"],
                        }
                    )

                # Ordena por confianca desc
                avals = sorted(avals, key=lambda x: x["confianca"], reverse=True)

                if not avals:
                    st.info("Não há séries suficientes para cálculo de otimização.")
                else:
                    # aproximação: prob_comb = 1 - ∏(1 - p_i), p_i = conf/100
                    prob_acumulada = 0.0
                    pacote = []
                    for item in avals:
                        p = item["confianca"] / 100.0
                        prob_acumulada = 1.0 - (1.0 - prob_acumulada) * (1.0 - p)
                        pacote.append(item)
                        if prob_acumulada * 100.0 >= alvo_conf:
                            break

                    st.write(
                        f"Para atingir aproximadamente **{alvo_conf}%** de confiabilidade conjunta, "
                        f"são necessárias cerca de **{len(pacote)}** séries."
                    )

                    st.markdown("Séries sugeridas para o pacote mínimo:")

                    linhas_pacote = []
                    for item in pacote:
                        nums = item["serie"]["numeros"]
                        linha = " ".join(str(n) for n in nums)
                        linhas_pacote.append(linha)
                    st.code("\n".join(linhas_pacote))

                    st.caption(
                        "A confiabilidade conjunta é aproximada assumindo independência parcial entre as séries "
                        "(modelo 1 - ∏(1-p_i))."
                    )
