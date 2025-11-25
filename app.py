
import streamlit as st

st.set_page_config(
    page_title="Predict Cars V13.8",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Predict Cars V13.8 — Painel Principal")
st.markdown(
    "Bem-vindo ao painel web do **Predict Cars V13.8**.\n\n"
    "Use o menu lateral para navegar entre as seções: Manual, Modo Normal, Modo IDX, Ajuste Dinâmico e Previsões Finais."
)

st.header("📂 Carregar histórico")
st.write(
    "Envie aqui o arquivo de histórico no formato esperado (linhas do tipo "
    "`C1234; n1; n2; n3; n4; n5; k`). Este arquivo ficará disponível para todas as páginas."
)

uploaded_file = st.file_uploader(
    "Escolha o arquivo de histórico (.txt ou .csv):",
    type=["txt", "csv"]
)

if uploaded_file is not None:
    # Guardar o conteúdo bruto na sessão para outras páginas usarem
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    st.session_state["historico_bruto"] = content
    st.success("Histórico carregado e disponível para as demais páginas.")

    with st.expander("Pré-visualização das primeiras linhas"):
        preview_lines = "\n".join(content.splitlines()[:20])
        st.text(preview_lines)
else:
    st.info("Nenhum arquivo enviado ainda. As outras páginas só funcionam plenamente após o upload do histórico.")
