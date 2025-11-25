
import streamlit as st

st.title("📊 Previsões Finais — Núcleo, Cobertura e Listas (Protótipo)")

st.markdown(
    "Esta página consolida as **previsões finais**: núcleo, núcleo resiliente, coberturas, "
    "listas SA1/MAX e demais saídas previstas no V13.8."
)

if "historico_bruto" not in st.session_state:
    st.warning("Nenhum histórico foi carregado ainda. Volte à página principal e envie o arquivo.")
    st.stop()

st.subheader("🧪 Área de prototipagem de previsões")
st.markdown(
    "Neste primeiro momento, não estamos gerando previsões reais.\n"
    "Esta página serve como espaço para estruturarmos a apresentação das previsões, "
    "que depois serão preenchidas com a lógica detalhada do sistema."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Núcleo (protótipo)")
    st.write("[ ] [ ] [ ] [ ] [ ] [ ]")

with col2:
    st.markdown("### Cobertura de Vento (protótipo)")
    st.write("[ ] [ ] [ ] [ ] [ ] [ ]")

st.markdown("---")
st.markdown("### Listas SA1 / MAX (protótipo)")
st.write("Aqui exibiremos as listas estruturadas, com rótulos claros (SA1, MAX, híbridas, etc.).")

st.info(
    "Quando a lógica de previsão estiver implementada, esta página será o painel principal de resultados "
    "para você copiar e trazer para o ChatGPT discutir."
)
