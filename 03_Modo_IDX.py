
import streamlit as st

st.title("🎯 Modo IDX — IDX Puro Focado / Otimizado (Protótipo)")

st.markdown(
    "Esta página representa o **Modo IDX** do V13.8 (IPF e IPO).\n"
    "A lógica aqui será focada em encontrar trechos historicamente semelhantes ao momento atual "
    "e construir um núcleo previsivo baseado em similaridade estrutural."
)

if "historico_bruto" not in st.session_state:
    st.warning("Nenhum histórico foi carregado ainda. Volte à página principal e envie o arquivo.")
    st.stop()

raw = st.session_state["historico_bruto"]
lines = [l.strip() for l in raw.splitlines() if l.strip()]

st.subheader("📥 Resumo do Histórico")
st.write(f"Total de linhas disponíveis: **{len(lines)}**")

st.subheader("🧪 Protótipo de Similaridade (conceitual)")
st.markdown(
    "Nesta primeira versão, não estamos executando o IDX real, apenas marcando o local "
    "onde a lógica de similaridade será implementada.\n\n"
    "Mais tarde, aqui entraremos com: identificação do trecho alvo, cálculo de similaridade, "
    "seleção dos trechos mais parecidos e construção do núcleo puro."
)

st.info("Modo IDX pronto para receber a lógica detalhada do manual.")
