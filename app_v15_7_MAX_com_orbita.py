from __future__ import annotations

# ============================================================
# PARTE 1/8 — INÍCIO
# ============================================================
"""PredictCars V15.7 MAX — V16 Premium
Âncora Estável (base: app_v15_7_MAX_com_orbita.py)
P1 — Ajuste de Pacote (pré-C4) · Backtest A/B BLOCO C (N=60 primeiro)
Arquivo único, íntegro e operacional.
"""


import streamlit as st
from datetime import datetime
import re

# ============================================================
# V16h57CJ — MODE6 FUNCTION TRACE HELPERS
# ============================================================
def pc_packet_audit_dict(listas, label=""):
    try:
        pkt = []
        for l in (listas or []):
            try:
                li = [int(x) for x in list(l)[:6]]
                if li:
                    pkt.append(li)
            except Exception:
                pass
        if not pkt:
            return {
                "label": label,
                "n_listas": 0,
                "hash": None,
                "passageiros_unicos": 0,
                "sobreposicao_media": 0.0,
                "exemplo": [],
            }
        flat = [x for l in pkt for x in l]
        inter = []
        for i in range(len(pkt)):
            si = set(pkt[i])
            for j in range(i + 1, len(pkt)):
                inter.append(len(si.intersection(pkt[j])))
        return {
            "label": label,
            "n_listas": len(pkt),
            "hash": hash(str(pkt)),
            "passageiros_unicos": len(set(flat)),
            "sobreposicao_media": round(sum(inter) / len(inter), 2) if inter else 0.0,
            "exemplo": pkt[:3],
        }
    except Exception as e:
        return {
            "label": label,
            "n_listas": 0,
            "hash": None,
            "passageiros_unicos": 0,
            "sobreposicao_media": 0.0,
            "exemplo": [],
            "erro": str(e),
        }

def pc_exec_trace(step, payload=None):
    try:
        key = "v16h57FL_exec_trace"
        arr = st.session_state.get(key)
        if not isinstance(arr, list):
            arr = []
        item = {"step": str(step)}
        if isinstance(payload, dict):
            item.update(payload)
        arr.append(item)
        st.session_state[key] = arr
    except Exception:
        pass


def pc_list_source_detector(step, listas=None, extra=None):
    try:
        import inspect
        key = "v16h57FL_source_detector"
        arr = st.session_state.get(key)
        if not isinstance(arr, list):
            arr = []
        info = pc_packet_audit_dict(listas or [], label=str(step))
        callers = []
        try:
            for fr in inspect.stack()[1:6]:
                fn = str(fr.function)
                if fn not in callers:
                    callers.append(fn)
        except Exception:
            callers = []
        item = {
            "step": str(step),
            "caller_chain": callers,
            "list_obj_id": int(id(listas)) if listas is not None else None,
            "is_session_modo6_listas": bool(listas is st.session_state.get("modo6_listas")) if listas is not None else False,
            "is_session_pacote_atual": bool(listas is st.session_state.get("pacote_listas_atual")) if listas is not None else False,
            "is_session_pacote_baseline": bool(listas is st.session_state.get("pacote_listas_baseline")) if listas is not None else False,
            "session_keys_present": [k for k in ["modo6_listas","pacote_listas_atual","pacote_listas_baseline","v16_execucao","ultima_previsao"] if k in st.session_state],
            "packet": info,
        }
        if isinstance(extra, dict):
            item.update(extra)
        arr.append(item)
        st.session_state[key] = arr
    except Exception:
        pass


# ============================================================
# V16h57CF — INTERNAL MODE6 TRACE HELPERS
# ============================================================
def pc_packet_audit_dict(listas, label=""):
    try:
        if not listas:
            return {
                "label": label,
                "n_listas": 0,
                "hash": None,
                "passageiros_unicos": 0,
                "sobreposicao_media": 0.0,
                "exemplo": [],
            }
        pkt = []
        for l in (listas or []):
            try:
                li = [int(x) for x in list(l)[:6]]
                if li:
                    pkt.append(li)
            except Exception:
                pass
        flat = [x for l in pkt for x in l]
        inter = []
        for i in range(len(pkt)):
            si = set(pkt[i])
            for j in range(i + 1, len(pkt)):
                inter.append(len(si.intersection(pkt[j])))
        return {
            "label": label,
            "n_listas": len(pkt),
            "hash": hash(str(pkt)),
            "passageiros_unicos": len(set(flat)),
            "sobreposicao_media": round(sum(inter) / len(inter), 2) if inter else 0.0,
            "exemplo": pkt[:3],
        }
    except Exception as e:
        return {
            "label": label,
            "n_listas": 0,
            "hash": None,
            "passageiros_unicos": 0,
            "sobreposicao_media": 0.0,
            "exemplo": [],
            "erro": str(e),
        }

def pc_trace_store(key, listas, label=None):
    try:
        info = pc_packet_audit_dict(listas, label or key)
        st.session_state[key] = info
        return info
    except Exception as e:
        try:
            st.session_state[f"{key}_error"] = str(e)
        except Exception:
            pass
        return None


# ============================================================
# ============================================================
# V16h57CP — COHESION AUTO-TUNE (plugável real)
# - Mede o pacote real antes do controller
# - Ajusta a intensidade da coesão com base em U/O
# - Não cria motor novo e não altera Camada 4
# ============================================================
def pc_packet_metrics(listas):
    try:
        return pc_packet_audit_dict(listas, "packet_metrics")
    except Exception:
        return {
            "label": "packet_metrics",
            "n_listas": 0,
            "hash": None,
            "passageiros_unicos": 0,
            "sobreposicao_media": 0.0,
            "exemplo": [],
        }

def pc_cohesion_auto_tune(
    packet_metrics,
    intensidade_atual=0.5,
    u_target_mid=21.0,
    o_target_mid=1.9,
    alpha=0.20,
):
    try:
        U = float((packet_metrics or {}).get("passageiros_unicos", 0.0) or 0.0)
        O = float((packet_metrics or {}).get("sobreposicao_media", 0.0) or 0.0)

        if U <= 0:
            return {
                "intensidade_nova": float(intensidade_atual),
                "e_u": 0.0,
                "e_o": 0.0,
                "motivo": "pacote_invalido",
            }

        e_u = (U - float(u_target_mid)) / max(float(u_target_mid), 1e-9)
        e_o = (O - float(o_target_mid)) / max(float(o_target_mid), 1e-9)
        intensidade_nova = float(intensidade_atual) + float(alpha) * (-float(e_u) + float(e_o))
        intensidade_nova = max(0.0, min(1.0, intensidade_nova))

        return {
            "intensidade_nova": round(float(intensidade_nova), 4),
            "e_u": round(float(e_u), 4),
            "e_o": round(float(e_o), 4),
            "motivo": "ok",
        }
    except Exception as _e:
        return {
            "intensidade_nova": float(intensidade_atual),
            "e_u": 0.0,
            "e_o": 0.0,
            "motivo": f"erro: {_e}",
        }


# ============================================================
# V16h57BT — PACKET COHESION CONTROLLER (safe hook)
# ============================================================
def packet_cohesion_controller(listas, intensidade=0.5):
    try:
        pc_exec_trace("ENTER packet_cohesion_controller", {"arg_n": len(listas or []), "intensidade": float(intensidade if intensidade is not None else 0.5)})
        if not listas:
            print("\n🔎 POST MODO6 BEFORE CONTROLLER")
            print({"hash": None, "passageiros_unicos": 0, "sobreposicao_media": 0})
            print("\n🔎 POST MODO6 AFTER CONTROLLER")
            print({"hash": None, "passageiros_unicos": 0, "sobreposicao_media": 0})
            return listas

        def _stats(pkt):
            flat = [x for l in pkt for x in l]
            inter = []
            for i in range(len(pkt)):
                si = set(pkt[i])
                for j in range(i + 1, len(pkt)):
                    inter.append(len(si.intersection(pkt[j])))
            return {
                "hash": hash(str(pkt)),
                "passageiros_unicos": len(set(flat)),
                "sobreposicao_media": round(sum(inter) / len(inter), 2) if inter else 0
            }

        pkt = []
        for l in (listas or []):
            try:
                li = [int(x) for x in list(l)[:6]]
                if len(li) >= 6:
                    pkt.append(li[:6])
            except Exception:
                pass
        if not pkt:
            return listas

        before = _stats(pkt)
        print("\n🔎 POST MODO6 BEFORE CONTROLLER")
        print(before)

        from collections import Counter
        intensidade = float(intensidade if intensidade is not None else 0.5)
        intensidade = max(0.0, min(1.0, intensidade))

        flat = [x for l in pkt for x in l]
        freq = Counter(flat)

        dominant_rank = [p for p, _ in freq.most_common(max(6, min(12, len(freq))))]
        rare_rank = [p for p, _ in sorted(freq.items(), key=lambda kv: (kv[1], kv[0]))]

        dominant_cut = max(4, int(round(4 + 4 * intensidade)))
        dominant_set = set(dominant_rank[:dominant_cut])
        rare_pool = [p for p in rare_rank if p not in dominant_set]
        if not rare_pool:
            rare_pool = list(rare_rank)

        if intensidade < 0.34:
            early_keep = 5
            tail_replace = 1
            tail_start = 6
        elif intensidade < 0.67:
            early_keep = 4
            tail_replace = 2
            tail_start = 4
        else:
            early_keep = 3
            tail_replace = 3
            tail_start = 3

        novas = []
        rot = 0
        for idx, l in enumerate(pkt):
            orig = [int(x) for x in l[:6]]

            preserved = []
            for p in sorted(orig, key=lambda v: (freq.get(v, 0), v)):
                if p not in preserved:
                    preserved.append(p)
                if len(preserved) >= early_keep:
                    break

            nl = list(preserved)

            while len(nl) < 6 and rare_pool:
                cand = rare_pool[rot % len(rare_pool)]
                rot += 1
                if cand not in nl and cand not in orig:
                    nl.append(int(cand))
                if len(nl) >= 6:
                    break

            if len(nl) < 6:
                for p in sorted(orig, key=lambda v: (v in dominant_set, freq.get(v, 0), v)):
                    if p not in nl:
                        nl.append(int(p))
                    if len(nl) >= 6:
                        break

            replace_budget = 0 if idx < tail_start else tail_replace
            replaced = 0
            if replace_budget > 0 and rare_pool:
                for pos, val in enumerate(list(nl)):
                    if replaced >= replace_budget:
                        break
                    if val in dominant_set:
                        for cand in rare_pool:
                            if cand not in nl:
                                old = nl[pos]
                                nl[pos] = int(cand)
                                if len(set(nl)) == 6:
                                    replaced += 1
                                    break
                                nl[pos] = old

            dedup = []
            for p in nl:
                if p not in dedup:
                    dedup.append(int(p))
            nl = dedup[:6]

            if len(nl) < 6:
                for p in sorted(orig + rare_pool, key=lambda v: (v in nl, freq.get(v, 0), v)):
                    if p not in nl:
                        nl.append(int(p))
                    if len(nl) >= 6:
                        break

            novas.append(sorted(nl[:6]))

        after = _stats(novas)
        print("\n🔎 POST MODO6 AFTER CONTROLLER")
        print(after)
        try:
            pc_exec_trace("AFTER packet_cohesion_controller_structural_break", {
                "intensidade": float(intensidade),
                "before_unique": int(before.get("passageiros_unicos", 0)),
                "after_unique": int(after.get("passageiros_unicos", 0)),
                "before_overlap": float(before.get("sobreposicao_media", 0)),
                "after_overlap": float(after.get("sobreposicao_media", 0)),
            })
        except Exception:
            pass
        return novas
    except Exception:
        return listas

# ============================================================
# V16h57CG — GENERATOR OPENING CONTROL (anti-compression at source)
# Atua no nascimento do pacote, antes das camadas de coesão.
# Objetivo: abrir o universo útil sem trocar o motor do Predicart.
# ============================================================
def pc_v16_generator_opening_control(listas_totais, *, ranking_vals=None, n_alvo=6, target_unique_min=28, max_replace_per_list=4):
    try:
        base = []
        for lst in (listas_totais or []):
            try:
                li = sorted(dict.fromkeys(int(x) for x in lst[:int(n_alvo)]))
                if len(li) == int(n_alvo):
                    base.append(li)
            except Exception:
                pass

        if not base:
            return listas_totais, {
                "active": False,
                "applied": False,
                "reason": "base_vazia",
                "passageiros_unicos_antes": 0,
                "passageiros_unicos_depois": 0,
            }

        flat = [x for l in base for x in l]
        unique_before = len(set(flat))

        from collections import Counter
        freq = Counter(flat)

        ranking_pool = []
        for x in (ranking_vals or []):
            try:
                xi = int(x)
                if xi not in ranking_pool:
                    ranking_pool.append(xi)
            except Exception:
                pass

        if not ranking_pool:
            ranking_pool = [v for v, _ in freq.most_common()]

        dominant = [int(x) for x, c in freq.most_common() if c >= max(4, len(base)//2)]
        non_packet = [x for x in ranking_pool if x not in set(flat)]
        low_inside = [x for x, _ in sorted(freq.items(), key=lambda kv: (kv[1], kv[0])) if x in set(flat)]
        candidate_pool = non_packet + low_inside
        if not candidate_pool:
            return base, {
                "active": True,
                "applied": False,
                "reason": "sem_pool_candidatos",
                "passageiros_unicos_antes": int(unique_before),
                "passageiros_unicos_depois": int(unique_before),
            }

        novas = []
        rotate_idx = 0
        swaps_total = 0
        for idx, lst in enumerate(base):
            orig = [int(x) for x in lst[:int(n_alvo)]]
            nova = list(orig)

            # more aggressive in later lists: break dominant structure before top10 cut
            replace_budget = 1 if idx < max(2, len(base)//4) else int(max_replace_per_list)
            replace_positions = [p for p, val in enumerate(nova) if val in dominant]
            replace_positions += [p for p in range(len(nova)) if p not in replace_positions]

            done = 0
            for pos in replace_positions:
                if done >= replace_budget:
                    break
                tries = 0
                candidate = None
                while tries < len(candidate_pool):
                    cand = candidate_pool[(rotate_idx + tries) % len(candidate_pool)]
                    tries += 1
                    if cand in nova:
                        continue
                    candidate = int(cand)
                    rotate_idx = (rotate_idx + 1) % max(1, len(candidate_pool))
                    break
                if candidate is None:
                    continue
                old = nova[pos]
                nova[pos] = candidate
                if len(set(nova)) != int(n_alvo):
                    nova[pos] = old
                    continue
                done += 1
                swaps_total += 1

            # reorder to keep deterministic shape but new composition
            novas.append(sorted(nova))

        saneadas = []
        seen = set()
        for lst in novas:
            key = tuple(sorted(lst))
            if key in seen:
                continue
            seen.add(key)
            saneadas.append(list(lst))

        for lst in base:
            if len(saneadas) >= len(base):
                break
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                saneadas.append(list(lst))

        final_out = saneadas[:len(base)]
        final_flat = [x for l in final_out for x in l]
        unique_after = len(set(final_flat))

        return final_out, {
            "active": True,
            "applied": bool(swaps_total > 0 and final_out != base),
            "reason": "ok" if swaps_total > 0 else "sem_trocas",
            "passageiros_unicos_antes": int(unique_before),
            "passageiros_unicos_depois": int(unique_after),
            "delta_unicos": int(unique_after - unique_before),
            "listas_qtd": int(len(final_out)),
            "dominant_vals": dominant[:8],
            "swaps_total": int(swaps_total),
            "target_unique_min": int(target_unique_min),
            "mode": "generator_diversity_break",
        }
    except Exception as _e:
        return listas_totais, {
            "active": False,
            "applied": False,
            "reason": f"generator_diversity_break_erro: {_e}",
            "passageiros_unicos_antes": 0,
            "passageiros_unicos_depois": 0,
        }


# ============================================================
# PredictCars V15.7 MAX — BUILD AUDITÁVEL v16h57FJ — FG + PRESSAO FINAL DE CONVERSAO + FAMILIA ESTAVEL + BANNER OK
# ============================================================

BUILD_TAG = "v16h57FL — PRESSAO CONVERSAO PROGRESSIVA FAMILIA ESTAVEL + BANNER OK"
BUILD_REAL_FILE = "app_v15_7_MAX_com_orbita_BUILD_AUDITAVEL_v16h57FL_PRESSAO_CONVERSAO_PROGRESSIVA_FAMILIA_ESTAVEL_BANNER_OK.py"
BUILD_CANONICAL_FILE = "app_v15_7_MAX_com_orbita.py"
BUILD_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
WATERMARK = "2026-03-02_01 (UNI50_60_AUDIT_FIX)"

# ⚠️ st.set_page_config precisa ser a PRIMEIRA chamada Streamlit
st.set_page_config(page_title="PredictCars V15.7 MAX — v16h57FL — BUILD AUDITÁVEL (micro encaixe de conversão + família estável)", page_icon="🚗", layout="wide")

# ================= BANNER AUDITÁVEL (GIGANTE) =================
st.markdown(
    f"""
    <div style="background-color:#111; border:3px solid #ff4b4b; padding:18px; border-radius:12px; margin-bottom:14px;">
        <h2 style="color:#ff4b4b; margin:0;">
            EXECUTANDO AGORA (BUILD REAL): {BUILD_REAL_FILE}
        </h2>
        <p style="color:white; margin:8px 0 0 0; font-size:15px;">
            <b>Arquivo canônico no GitHub/Streamlit:</b> {BUILD_CANONICAL_FILE}<br>
            <b>BUILD:</b> {BUILD_TAG}<br>
            <b>TIMESTAMP:</b> {BUILD_TIME}<br>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.warning(
    f"EXECUTANDO AGORA (BUILD REAL): {BUILD_REAL_FILE}\n"
    f"Arquivo canônico no GitHub/Streamlit: {BUILD_CANONICAL_FILE}\n"
    f"BUILD: {BUILD_TAG}\n"
    f"TIMESTAMP: {BUILD_TIME}\n"
    f"WATERMARK: {WATERMARK}"
)

# ------------------------------------------------------------
# V16h6 — BOOT CLEAN (anti-resíduo de sessão)

# ============================================================
# V16h57AR — COOCCURRENCE PACKET (pré‑C4 · auditável)
# Matriz simples de co‑ocorrência entre passageiros para
# favorecer pares historicamente frequentes nas listas.
# Não altera Camada 4.
# ============================================================
def pc_v16_cooccurrence_matrix(series_hist):
    try:
        from collections import defaultdict
        co = defaultdict(int)
        for s in series_hist:
            nums = list(dict.fromkeys(s))
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair = tuple(sorted((nums[i], nums[j])))
                    co[pair] += 1
        return co
    except Exception:
        return {}

# ============================================================
# V16h57AS — COOCCURRENCE LIST GENERATOR (pré‑C4 · auditável)
# Novo gerador de listas baseado em pares fortes de co‑ocorrência.
# Substitui parcialmente o gerador anterior ao montar listas
# usando pares historicamente frequentes.
# ============================================================

def pc_v16_generate_lists_cooccurrence(ranking, co_matrix, n=6, k_lists=12):
    try:
        if not isinstance(ranking, list) or len(ranking) < n:
            return []

        from random import shuffle
        top = ranking[:20]  # universo mais relevante
        pairs = []

        # construir pares fortes
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                pair = tuple(sorted((top[i], top[j])))
                score = co_matrix.get(pair, 0)
                if score > 0:
                    pairs.append((score, pair))

        pairs.sort(reverse=True)

        lists = []
        for score, pair in pairs[:k_lists*3]:
            base = list(pair)
            rest = [x for x in top if x not in base]
            shuffle(rest)
            while len(base) < n and rest:
                base.append(rest.pop())
            if len(base) == n:
                lists.append(base)

        return lists[:k_lists]
    except Exception:
        return []


def pc_v16_apply_cooccurrence(ranking, co_matrix):
    try:
        if not isinstance(ranking, list) or not co_matrix:
            return ranking
        
        top = ranking[:10]
        scored = []
        for n in top:
            score = 0
            for m in top:
                if n == m:
                    continue
                pair = tuple(sorted((n, m)))
                score += co_matrix.get(pair, 0)
            scored.append((score, n))
        
        scored.sort(reverse=True)
        new_top = [n for _, n in scored]
        rest = ranking[10:]
        return new_top + rest
    except Exception:
        return ranking



# ============================================================
# V16h57CT — CONVERSION PRESSURE (pré-MODO 6 · auditável)
# Usa memória ex-post recente para reponderar o ranking com
# pressão de conversão: números que aparecem em listas "quase"
# (best_hit >= 3) e particularmente em listas com best_hit >= 4.
# Não altera Camada 4.
# ============================================================
def pc_v16_conversion_pressure_scores(snapshot_p0_canonic, lookback=60):
    try:
        if not isinstance(snapshot_p0_canonic, dict) or not snapshot_p0_canonic:
            return {"ok": False, "scores": {}, "motivo": "sem_snapshots"}

        ks = []
        for k in snapshot_p0_canonic.keys():
            try:
                ks.append(int(k))
            except Exception:
                pass
        ks = sorted(ks)[-int(max(1, lookback)):]
        if not ks:
            return {"ok": False, "scores": {}, "motivo": "ks_vazios"}

        scores = {}
        usados = 0
        for k in ks:
            snap = snapshot_p0_canonic.get(k) or {}
            listas = snap.get("listas_top10") or snap.get("listas") or []
            evals = snap.get("eval_top10") or snap.get("best_hits_top10") or []

            if not isinstance(listas, list) or not listas:
                continue

            local_weights = []
            if isinstance(evals, list) and len(evals) == len(listas):
                for bh in evals:
                    try:
                        b = int(bh)
                    except Exception:
                        b = 0
                    if b >= 4:
                        local_weights.append(2.0)
                    elif b >= 3:
                        local_weights.append(1.0)
                    else:
                        local_weights.append(0.0)
            else:
                core = set()
                quase = set()
                try:
                    core = {int(x) for x in (snap.get("snap_v8", {}) or {}).get("core", [])}
                except Exception:
                    core = set()
                try:
                    quase = {int(x) for x in (snap.get("snap_v8", {}) or {}).get("quase_core", [])}
                except Exception:
                    quase = set()
                for lst in listas:
                    w = 0.0
                    try:
                        li = [int(x) for x in list(lst)[:6]]
                    except Exception:
                        li = []
                    if any(x in core for x in li):
                        w += 0.5
                    if any(x in quase for x in li):
                        w += 0.25
                    local_weights.append(w)

            for lst, w in zip(listas, local_weights):
                if float(w) <= 0.0:
                    continue
                usados += 1
                try:
                    li = [int(x) for x in list(lst)[:6]]
                except Exception:
                    li = []
                for x in li:
                    scores[int(x)] = float(scores.get(int(x), 0.0)) + float(w)

        if not scores:
            return {"ok": False, "scores": {}, "motivo": "scores_vazios"}

        mx = max(scores.values()) if scores else 1.0
        norm = {int(k): float(v) / float(mx if mx else 1.0) for k, v in scores.items()}
        top = sorted(norm.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        return {
            "ok": True,
            "scores": norm,
            "top": [{"p": int(k), "score": round(float(v), 4)} for k, v in top],
            "snapshots_usados": int(len(ks)),
            "listas_ponderadas": int(usados),
            "motivo": "ok",
        }
    except Exception as e:
        return {"ok": False, "scores": {}, "motivo": f"conversion_pressure_erro: {e}"}


# ============================================================
# V16h57EL — FINAL PACKET MOUNT (deep conversion-oriented)
# Atua APENAS na montagem final do pacote já gerado.
# Não cria motor novo. Não altera Camada 4. Não aumenta volume.
# ============================================================
def pc_v16_packet_final_mount_deep(listas_packet, ranking_vals=None, cp_scores=None, co_matrix=None, n_alvo=6, top_k=10):
    try:
        pkt = []
        for lst in (listas_packet or []):
            try:
                li = sorted(dict.fromkeys(int(x) for x in list(lst)[:int(n_alvo)]))
                if len(li) >= int(n_alvo):
                    pkt.append(li[:int(n_alvo)])
            except Exception:
                pass
        if not pkt:
            return listas_packet, {"active": False, "applied": False, "reason": "pacote_vazio"}

        top_k = int(max(1, min(int(top_k), len(pkt))))
        top = [list(x) for x in pkt[:top_k]]
        tail = [list(x) for x in pkt[top_k:]]

        # frequência local do pacote
        freq = {}
        for lst in top:
            for v in lst:
                freq[int(v)] = freq.get(int(v), 0) + 1

        ranking = []
        for v in (ranking_vals or []):
            try:
                iv = int(v)
                if iv not in ranking:
                    ranking.append(iv)
            except Exception:
                pass

        if not ranking:
            ranking = [v for v, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]

        cp_scores = cp_scores if isinstance(cp_scores, dict) else {}
        co_matrix = co_matrix if isinstance(co_matrix, dict) else {}
        ranking_pos = {int(v): i for i, v in enumerate(ranking)}

        # pool operativo: topo do ranking + elementos já presentes no pacote
        pool = []
        for v in ranking[:18]:
            if int(v) not in pool:
                pool.append(int(v))
        for v in sorted(freq.keys(), key=lambda x: (-freq.get(x, 0), x)):
            if int(v) not in pool:
                pool.append(int(v))

        border_pool = []
        for v in ranking[10:24]:
            iv = int(v)
            if iv not in border_pool:
                border_pool.append(iv)
        for lst in pkt:
            for v in lst:
                iv = int(v)
                if iv not in border_pool and iv not in ranking[:10]:
                    border_pool.append(iv)

        def pair_score(cand, base_now):
            score = float(cp_scores.get(int(cand), 0.0)) * 3.0
            for b in base_now:
                pair = tuple(sorted((int(cand), int(b))))
                score += float(co_matrix.get(pair, 0)) * 0.158
            # leve preferência por candidatos já "na borda" do pacote
            score += float(freq.get(int(cand), 0)) * 0.15
            return score

        new_top = []
        changed = 0
        border_injections = 0
        for idx, lst in enumerate(top):
            base = list(lst)

            # primeiras listas preservam mais o núcleo; últimas recebem pequeno alívio seletivo
            preserve_n = 3 if idx < 4 else 2
            preserved = sorted(
                base,
                key=lambda v: (
                    -(float(cp_scores.get(int(v), 0.0)) * 3.0 + float(freq.get(int(v), 0))),
                    ranking_pos.get(int(v), 9999),
                    int(v),
                )
            )[:preserve_n]

            fill = list(dict.fromkeys(int(x) for x in preserved))

            # v16h57FJ — alívio seletivo: em parte do Top10, preserva 1 elemento de borda do pacote original
            border_anchor = None
            if idx >= 4:
                border_candidates_from_base = sorted(
                    [int(v) for v in base if int(v) not in fill],
                    key=lambda v: (
                        freq.get(int(v), 0),
                        -float(cp_scores.get(int(v), 0.0)),
                        ranking_pos.get(int(v), 9999),
                        int(v),
                    )
                )
                for cand in border_candidates_from_base:
                    if cand not in fill:
                        border_anchor = int(cand)
                        break
                if border_anchor is None:
                    for cand in border_pool:
                        if cand not in fill and cand not in base:
                            border_anchor = int(cand)
                            break
                if border_anchor is not None and border_anchor not in fill:
                    fill.append(int(border_anchor))
                    border_injections += 1

            candidates = [int(v) for v in pool if int(v) not in fill]
            while len(fill) < int(n_alvo) and candidates:
                best = sorted(
                    candidates,
                    key=lambda c: (-pair_score(int(c), fill), ranking_pos.get(int(c), 9999), int(c))
                )[0]
                fill.append(int(best))
                candidates = [c for c in candidates if int(c) != int(best)]

            nova = sorted(dict.fromkeys(fill))[:int(n_alvo)]
            if len(nova) < int(n_alvo):
                for v in base:
                    if int(v) not in nova:
                        nova.append(int(v))
                    if len(nova) >= int(n_alvo):
                        break
            nova = sorted(dict.fromkeys(nova))[:int(n_alvo)]

            if nova != sorted(lst):
                changed += 1
            new_top.append(nova)

        # mede pressão do top antes do alívio final
        def _packet_metrics(packet_lists):
            flat = [int(x) for lst in (packet_lists or []) for x in lst[:int(n_alvo)]]
            inter = []
            for i in range(len(packet_lists or [])):
                si = set(packet_lists[i][:int(n_alvo)])
                for j in range(i + 1, len(packet_lists or [])):
                    inter.append(len(si.intersection(packet_lists[j][:int(n_alvo)])))
            return {
                "passageiros_unicos": len(set(flat)),
                "sobreposicao_media": round(sum(inter) / len(inter), 2) if inter else 0.0,
            }

        top_metrics_before_relief = _packet_metrics(new_top)

        # v16h57FJ — alívio cirúrgico do Top10 só quando a compressão estiver elevada
        relief_applied = False
        relief_swaps = 0
        if (
            len(new_top) >= 8
            and int(top_metrics_before_relief.get("passageiros_unicos", 0)) <= 15
            and float(top_metrics_before_relief.get("sobreposicao_media", 0.0)) >= 2.55
        ):
            dominant_vals = [
                int(v) for v, c in sorted(freq.items(), key=lambda kv: (-kv[1], -float(cp_scores.get(int(kv[0]), 0.0)), kv[0]))
                if c >= max(4, int(round(len(new_top) * 0.50)))
            ]
            relief_pool = [int(v) for v in border_pool if int(v) not in dominant_vals]
            if relief_pool:
                for idx in range(min(4, len(new_top)-1), len(new_top)):
                    lst = list(new_top[idx])
                    drop_candidates = [int(v) for v in sorted(lst, key=lambda v: (-freq.get(int(v), 0), ranking_pos.get(int(v), 9999), int(v))) if int(v) in dominant_vals]
                    drop = drop_candidates[0] if drop_candidates else None
                    add = None
                    for cand in relief_pool:
                        if int(cand) not in lst:
                            add = int(cand)
                            break
                    if drop is None or add is None:
                        continue
                    nova = sorted(dict.fromkeys([int(v) for v in lst if int(v) != int(drop)] + [int(add)]))[:int(n_alvo)]
                    if len(nova) >= int(n_alvo) and sorted(nova) != sorted(lst):
                        new_top[idx] = sorted(nova)
                        relief_applied = True
                        relief_swaps += 1
                        try:
                            relief_pool.remove(int(add))
                        except Exception:
                            pass
                    if relief_swaps >= 2:
                        break

        top_metrics_after_relief = _packet_metrics(new_top)

        # v16h57FJ REAL — pressão de conversão interna da família
        # Objetivo: aumentar interseção útil dentro da família já estável, sem colapsar o envelope.
        internal_conv_applied = False
        internal_conv_swaps = 0
        top_metrics_before_internal = dict(top_metrics_after_relief)

        if (
            len(new_top) >= 8
            and int(top_metrics_after_relief.get("passageiros_unicos", 0)) >= 17
            and float(top_metrics_after_relief.get("sobreposicao_media", 0.0)) <= 2.30
        ):
            family_freq = {}
            for lst in new_top:
                for v in lst[:int(n_alvo)]:
                    family_freq[int(v)] = family_freq.get(int(v), 0) + 1

            recurring_candidates = [
                int(v) for v, c in sorted(
                    family_freq.items(),
                    key=lambda kv: (-kv[1], -float(cp_scores.get(int(kv[0]), 0.0)), ranking_pos.get(int(kv[0]), 9999), int(kv[0]))
                )
                if c >= 3
            ]

            def _score_val(v):
                return (
                    float(cp_scores.get(int(v), 0.0)) * 3.0
                    + float(freq.get(int(v), 0)) * 0.25
                    + max(0.0, 1.0 - (ranking_pos.get(int(v), 9999) / max(1, len(ranking_pos) or 1)))
                )

            if recurring_candidates:
                for idx in range(3, len(new_top)):
                    lst = list(new_top[idx])
                    preserved = sorted(lst, key=lambda v: (-_score_val(int(v)), int(v)))[:2]
                    weak_candidates = [
                        int(v) for v in sorted(
                            lst,
                            key=lambda v: (_score_val(int(v)), family_freq.get(int(v), 0), int(v))
                        )
                        if int(v) not in preserved
                    ]
                    if not weak_candidates:
                        continue

                    add = None
                    for cand in recurring_candidates:
                        if int(cand) not in lst:
                            score_now = pair_score(int(cand), preserved + [x for x in lst if x in preserved])
                            if score_now >= 1.0:
                                add = int(cand)
                                break
                    if add is None:
                        continue

                    drop = int(weak_candidates[0])
                    nova = sorted(dict.fromkeys([int(v) for v in lst if int(v) != drop] + [int(add)]))[:int(n_alvo)]
                    if len(nova) >= int(n_alvo) and sorted(nova) != sorted(lst):
                        new_top[idx] = sorted(nova)
                        internal_conv_applied = True
                        internal_conv_swaps += 1

                    if internal_conv_swaps >= 2:
                        break

        top_metrics_after_internal = _packet_metrics(new_top)

        # v16h57FJ — pressão final de conversão
        # Objetivo: dar um último empurrão em 1-2 listas do Top10, sem recompactar o pacote inteiro.
        final_push_applied = False
        final_push_swaps = 0
        top_metrics_before_final_push = dict(top_metrics_after_internal)

        if (
            len(new_top) >= 8
            and int(top_metrics_after_internal.get("passageiros_unicos", 0)) >= 16
            and float(top_metrics_after_internal.get("sobreposicao_media", 0.0)) <= 2.60
        ):
            family_freq = {}
            for lst in new_top:
                for v in lst[:int(n_alvo)]:
                    family_freq[int(v)] = family_freq.get(int(v), 0) + 1

            strong_family = [
                int(v) for v, c in sorted(
                    family_freq.items(),
                    key=lambda kv: (-kv[1], -float(cp_scores.get(int(kv[0]), 0.0)), ranking_pos.get(int(kv[0]), 9999), int(kv[0]))
                ) if c >= 4
            ]

            def _val_score(v):
                return (
                    float(cp_scores.get(int(v), 0.0)) * 3.2
                    + float(freq.get(int(v), 0)) * 0.35
                    + float(family_freq.get(int(v), 0)) * 0.45
                    + max(0.0, 1.0 - (ranking_pos.get(int(v), 9999) / max(1, len(ranking_pos) or 1)))
                )

            if strong_family:
                for idx in range(2, min(len(new_top), 8)):
                    lst = list(new_top[idx])
                    preserve = sorted(lst, key=lambda v: (-_val_score(int(v)), int(v)))[:3]
                    weak = [
                        int(v) for v in sorted(
                            lst,
                            key=lambda v: (_val_score(int(v)), family_freq.get(int(v), 0), int(v))
                        ) if int(v) not in preserve
                    ]
                    add = None
                    for cand in strong_family:
                        if int(cand) not in lst:
                            local_score = pair_score(int(cand), preserve)
                            if local_score >= 1.0:
                                add = int(cand)
                                break
                    if not weak or add is None:
                        continue
                    drop = int(weak[0])
                    nova = sorted(dict.fromkeys([int(v) for v in lst if int(v) != drop] + [int(add)]))[:int(n_alvo)]
                    if len(nova) >= int(n_alvo) and sorted(nova) != sorted(lst):
                        new_top[idx] = sorted(nova)
                        final_push_applied = True
                        final_push_swaps += 1
                    if final_push_swaps >= 2:
                        break

        top_metrics_after_final_push = _packet_metrics(new_top)

        # dedup + recomposição mantendo volume
        out = []
        seen = set()
        for lst in new_top + tail:
            t = tuple(sorted(int(x) for x in lst[:int(n_alvo)]))
            if len(t) >= int(n_alvo) and t not in seen:
                seen.add(t)
                out.append(list(t))
        for lst in pkt:
            t = tuple(sorted(int(x) for x in lst[:int(n_alvo)]))
            if len(out) >= len(pkt):
                break
            if t not in seen:
                seen.add(t)
                out.append(list(t))

        out = out[:len(pkt)]
        return out, {
            "active": True,
            "applied": bool(out != pkt),
            "reason": "ok" if out != pkt else "sem_mudanca",
            "top_k_montado": int(top_k),
            "changed_lists": int(changed),
            "border_injections": int(border_injections),
            "relief_applied": bool(relief_applied),
            "relief_swaps": int(relief_swaps),
            "top_unique_before_relief": int(top_metrics_before_relief.get("passageiros_unicos", 0)),
            "top_overlap_before_relief": float(top_metrics_before_relief.get("sobreposicao_media", 0.0)),
            "top_unique_after_relief": int(top_metrics_after_relief.get("passageiros_unicos", 0)),
            "top_overlap_after_relief": float(top_metrics_after_relief.get("sobreposicao_media", 0.0)),
            "internal_conv_applied": bool(internal_conv_applied),
            "internal_conv_swaps": int(internal_conv_swaps),
            "top_unique_before_internal": int(top_metrics_before_internal.get("passageiros_unicos", 0)),
            "top_overlap_before_internal": float(top_metrics_before_internal.get("sobreposicao_media", 0.0)),
            "top_unique_after_internal": int(top_metrics_after_internal.get("passageiros_unicos", 0)),
            "top_overlap_after_internal": float(top_metrics_after_internal.get("sobreposicao_media", 0.0)),
            "final_push_applied": bool(final_push_applied),
            "final_push_swaps": int(final_push_swaps),
            "top_unique_before_final_push": int(top_metrics_before_final_push.get("passageiros_unicos", 0)),
            "top_overlap_before_final_push": float(top_metrics_before_final_push.get("sobreposicao_media", 0.0)),
            "top_unique_after_final_push": int(top_metrics_after_final_push.get("passageiros_unicos", 0)),
            "top_overlap_after_final_push": float(top_metrics_after_final_push.get("sobreposicao_media", 0.0)),
            "hash_antes": hash(str(pkt)),
            "hash_depois": hash(str(out)),
        }
    except Exception as e:
        return listas_packet, {"active": False, "applied": False, "reason": f"final_mount_erro: {e}"}

# ============================================================
# V16h57AT — NEW PACKET GENERATOR (pré-C4 · auditável)
# Atua no gerador REAL do Modo 6.
# Usa coocorrência + top_pool para regenerar parte do pacote.
# ============================================================
def pc_v16_new_packet_generator(listas_totais, *, ranking_vals=None, historico_df=None, n_alvo=6, seed=0, max_lists=None):
    try:
        pc_exec_trace("ENTER pc_v16_new_packet_generator", {"arg_n": len(listas_totais or [])})
        base = []
        for lst in (listas_totais or []):
            try:
                li = [int(x) for x in lst][:int(n_alvo)]
                if len(li) >= int(n_alvo):
                    t = sorted(dict.fromkeys(li))[:int(n_alvo)]
                    if len(t) >= int(n_alvo):
                        base.append(t)
            except Exception:
                pass
        if not base:
            return listas_totais, {"active": False, "applied": False, "reason": "base_vazia", "listas_regeneradas_qtd": 0}

        max_lists = int(max_lists or len(base) or 0)
        if max_lists <= 0:
            return listas_totais, {"active": False, "applied": False, "reason": "max_lists_zero", "listas_regeneradas_qtd": 0}

        # histórico -> matriz de coocorrência
        series_hist = []
        try:
            dfh = historico_df
            if dfh is not None and hasattr(dfh, "columns"):
                cols = [c for c in dfh.columns if str(c).startswith("p")]
                for _, row in dfh.iterrows():
                    vals = []
                    for c in cols[:int(n_alvo)]:
                        try:
                            if pd.notna(row[c]):
                                vals.append(int(row[c]))
                        except Exception:
                            pass
                    if len(vals) >= int(n_alvo):
                        series_hist.append(vals[:int(n_alvo)])
        except Exception:
            series_hist = []

        co = pc_v16_cooccurrence_matrix(series_hist) if series_hist else {}

        ranking = []
        try:
            ranking = [int(x) for x in (ranking_vals or [])]
        except Exception:
            ranking = []

        if not ranking:
            freq = {}
            for lst in base:
                for v in lst:
                    freq[v] = freq.get(v, 0) + 1
            ranking = [v for v, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]

        if not ranking:
            return listas_totais, {"active": False, "applied": False, "reason": "ranking_vazio", "listas_regeneradas_qtd": 0}

        ranking2 = pc_v16_apply_cooccurrence(ranking, co) if co else list(ranking)
        ranking_before_cp = [int(v) for v in ranking2]
        try:
            snapshot_p0_map = (
                st.session_state.get("snapshot_p0_canonic")
                or st.session_state.get("snapshot_p0")
                or st.session_state.get("snapshot_p0_replay")
                or {}
            )
            cp_info = pc_v16_conversion_pressure_scores(snapshot_p0_map, lookback=60)
            cp_scores = cp_info.get("scores", {}) if isinstance(cp_info, dict) and cp_info.get("ok") else {}
            # Fallback: derive minimal scores from snapshot core to ensure observable activation
            if not cp_scores and isinstance(snapshot_p0_map, dict) and snapshot_p0_map:
                try:
                    core = snapshot_p0_map.get("snap_v8", {}).get("core", []) if isinstance(snapshot_p0_map.get("snap_v8", {}), dict) else []
                    cp_scores = {int(v): 1.0 for v in core} if core else {}
                    if cp_scores:
                        cp_info = dict(cp_info or {})
                        cp_info["ok"] = True
                        cp_info["motivo"] = "fallback_core"
                except Exception:
                    pass
            if cp_scores:
                _base_idx = {int(v): i for i, v in enumerate(ranking2)}
                ranking2 = sorted(
                    [int(v) for v in ranking2],
                    key=lambda v: (
                        -(cp_scores.get(int(v), 0.0) * 2.25 + max(0.0, 1.0 - (_base_idx.get(int(v), 9999) / max(1, len(ranking2))))),
                        _base_idx.get(int(v), 9999),
                        int(v),
                    )
                )

                # v16h57FJ — INJECAO BORDA-PERTO REAL
                # objetivo: trazer alguns candidatos da borda util para o topo operativo,
                # sem inventar motor novo e sem quebrar o ranking base.
                try:
                    top_cut = ranking2[:10]
                    border_pool = ranking2[10:18]
                    border_pool = sorted(
                        [int(v) for v in border_pool],
                        key=lambda v: (-float(cp_scores.get(int(v), 0.0)), _base_idx.get(int(v), 9999), int(v))
                    )
                    inj_candidates = []
                    for cand in border_pool:
                        if cand not in top_cut and cand not in inj_candidates:
                            inj_candidates.append(int(cand))
                        if len(inj_candidates) >= 2:
                            break

                    if inj_candidates:
                        # v16h57FJ — injecao mais agressiva: até 3 candidatos subindo até a posição 7
                        extra_pool = ranking2[18:22]
                        extra_pool = sorted(
                            [int(v) for v in extra_pool],
                            key=lambda v: (-float(cp_scores.get(int(v), 0.0)), _base_idx.get(int(v), 9999), int(v))
                        )
                        for cand in extra_pool:
                            if cand not in top_cut and cand not in inj_candidates:
                                inj_candidates.append(int(cand))
                            if len(inj_candidates) >= 3:
                                break

                        replace_positions = [7, 8, 9]
                        ranking2_mod = list(ranking2)
                        injected = []
                        for pos, cand in zip(replace_positions, inj_candidates):
                            if pos < len(ranking2_mod):
                                old_v = int(ranking2_mod[pos])
                                if cand != old_v and cand not in ranking2_mod[:10]:
                                    ranking2_mod.remove(cand)
                                    ranking2_mod.insert(pos, int(cand))
                                    injected.append({"pos": int(pos), "entrou": int(cand), "saiu": int(old_v)})
                        if injected:
                            ranking2 = ranking2_mod
                            cp_info = dict(cp_info or {})
                            cp_info["borda_perto_injetada"] = True
                            cp_info["borda_injetada_qtd"] = int(len(injected))
                            cp_info["borda_injetada_eventos"] = injected
                        else:
                            cp_info = dict(cp_info or {})
                            cp_info["borda_perto_injetada"] = False
                            cp_info["borda_injetada_qtd"] = 0
                    else:
                        cp_info = dict(cp_info or {})
                        cp_info["borda_perto_injetada"] = False
                        cp_info["borda_injetada_qtd"] = 0
                except Exception as _inj_e:
                    cp_info = dict(cp_info or {})
                    cp_info["borda_perto_injetada"] = False
                    cp_info["borda_injecao_erro"] = str(_inj_e)

                cp_info = dict(cp_info)
                cp_info["snapshot_disponivel"] = bool(isinstance(snapshot_p0_map, dict) and len(snapshot_p0_map) > 0)
                cp_info["qtd_snapshots"] = int(len(snapshot_p0_map) if isinstance(snapshot_p0_map, dict) else 0)
                cp_info["ranking_antes_hash"] = hash(str(ranking_before_cp))
                cp_info["ranking_depois_hash"] = hash(str(ranking2))
                cp_info["ranking_mudou"] = bool(ranking_before_cp != ranking2)
                cp_info["top10_antes"] = ranking_before_cp[:10]
                cp_info["top10_depois"] = [int(v) for v in ranking2[:10]]
                cp_info["dif_posicoes_top10"] = int(sum(1 for a, b in zip(ranking_before_cp[:10], ranking2[:10]) if a != b))
                cp_info["cp_weight"] = 2.25
            else:
                cp_info = {
                    "ok": False,
                    "motivo": "sem_cp",
                    "snapshot_disponivel": bool(isinstance(snapshot_p0_map, dict) and len(snapshot_p0_map) > 0),
                    "qtd_snapshots": int(len(snapshot_p0_map) if isinstance(snapshot_p0_map, dict) else 0),
                    "ranking_antes_hash": hash(str(ranking_before_cp)),
                    "ranking_depois_hash": hash(str(ranking2)),
                    "ranking_mudou": False,
                    "top10_antes": ranking_before_cp[:10],
                    "top10_depois": [int(v) for v in ranking2[:10]],
                    "dif_posicoes_top10": 0,
                    "cp_weight": 2.25,
                }
        except Exception as _e:
            cp_info = {"ok": False, "motivo": f"cp_apply_erro: {_e}"}

        geradas = pc_v16_generate_lists_cooccurrence(
            ranking2,
            co,
            n=int(n_alvo),
            k_lists=min(max_lists, max(9, min(12, max_lists)))
        ) if co else []

        if not geradas:
            try:
                rng = np.random.default_rng(int(seed) if seed is not None else 0)
                top = ranking2[:max(12, int(n_alvo))]
                geradas = []
                for _ in range(min(max_lists, max(9, min(12, max_lists)))):
                    pick = list(top[:int(n_alvo)])
                    if len(pick) < int(n_alvo):
                        pool = [x for x in ranking2 if x not in pick]
                        while len(pick) < int(n_alvo) and pool:
                            idx = int(rng.integers(0, len(pool)))
                            pick.append(pool.pop(idx))
                    if len(pick) >= int(n_alvo):
                        t = sorted(dict.fromkeys([int(x) for x in pick]))[:int(n_alvo)]
                        if len(t) >= int(n_alvo):
                            geradas.append(t)
            except Exception:
                geradas = []

        if not geradas:
            return listas_totais, {"active": True, "applied": False, "reason": "gerador_sem_saida", "listas_regeneradas_qtd": 0}

        out = []
        seen = set()
        for lst in geradas + base:
            try:
                t = tuple(sorted(dict.fromkeys([int(x) for x in lst[:int(n_alvo)]]))[:int(n_alvo)])
                if len(t) >= int(n_alvo) and t not in seen:
                    seen.add(t)
                    out.append(list(t))
            except Exception:
                pass

        out = out[:max_lists]
        if len(out) < len(base):
            for lst in base:
                t = tuple(lst[:int(n_alvo)])
                if t not in seen:
                    seen.add(t)
                    out.append(list(t))
                if len(out) >= len(base):
                    break

        # v16h57FJ — montagem final profunda para conversão
        out_mounted, final_mount_info = pc_v16_packet_final_mount_deep(
            out,
            ranking_vals=ranking2,
            cp_scores=(cp_info.get("scores", {}) if isinstance(cp_info, dict) else {}),
            co_matrix=co,
            n_alvo=int(n_alvo),
            top_k=min(10, len(out)),
        )
        if isinstance(out_mounted, list) and out_mounted:
            out = out_mounted

        return out, {
            "active": True,
            "applied": bool(out != base),
            "reason": "ok" if out else "saida_vazia",
            "listas_regeneradas_qtd": int(min(len(geradas), len(base))),
            "co_pairs_qtd": int(len(co)),
            "ranking_base_qtd": int(len(ranking2)),
            "listas_antes_gerador_hash": hash(str(base)),
            "listas_finais_hash": hash(str(out)),
            "mudou_no_pacote_final": bool(out != base),
            "conversion_pressure": cp_info,
            "final_mount_info": final_mount_info,
        }
    except Exception as e:
        return listas_totais, {"active": False, "applied": False, "reason": f"new_packet_generator_erro: {e}", "listas_regeneradas_qtd": 0}


# - Se não há histórico carregado, remove saídas antigas que podem
#   "vazar" na UI (ex.: pacote final / previsões antigas).
# ------------------------------------------------------------
try:
    if "historico_df" not in st.session_state:
        for _k in ["ultima_previsao", "listas_geradas", "pacote_listas_atual", "pacote_listas_origem", "pacote_pre_bloco_c", "pacote_pre_bloco_c_origem"]:
            if _k in st.session_state:
                del st.session_state[_k]
except Exception:
    pass

# ============================================================
# V16h57CN — SESSION STATE CONTROL (FORCE FRESH PACKET)
# Limpa pacotes/listas persistidas antes de uma nova execução do Modo 6,
# para evitar reutilização de estado antigo na sessão.
# ============================================================
def v16h57FL_clear_mode6_packet_state():
    removed = []
    keys = [
        "modo6_listas",
        "modo6_listas_top10",
        "modo6_listas_totais",
        "pacote_listas_atual",
        "pacote_listas_origem",
        "pacote_pre_bloco_c",
        "pacote_pre_bloco_c_origem",
        "pacote_listas_baseline",
        "listas_geradas",
        "ultima_previsao",
        "bloco_c_info",
        "postura_respiravel_info",
        "postura_respiravel_memoria",
    ]
    try:
        for k in keys:
            if k in st.session_state:
                removed.append(k)
                del st.session_state[k]
    except Exception:
        pass
    st.session_state["v16h57FL_fresh_packet_removed_keys"] = removed
    st.session_state["v16h57FL_fresh_packet_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return removed

# ============================================================
# V16 — POSTURA OPERACIONAL (pré-C4)
# Estados: ESTÁVEL / RESPIRÁVEL / RUPTURA
# - Não decide ataque (Camada 4 intocável)
# - Apenas ajusta a execução do P0/Modo 6 em ambiente "jogável"
# ============================================================

def _pc_safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        if s == "" or s.lower() in ("n/d", "nd", "nan", "none"):
            return default
        return float(s)
    except Exception:
        return default

def pc_classificar_postura(pipeline_regime: str | None, k_star, nr_percent, div_s6_mc):
    """Classifica postura operacional canônica.
    Critérios são conservadores e usam apenas sinais já existentes.
    """
    k = _pc_safe_float(k_star, None)
    nr = _pc_safe_float(nr_percent, None)
    dv = _pc_safe_float(div_s6_mc, None)

    regime = (pipeline_regime or "").strip()

    # Heurística: estrada quente puxa para RESPIRÁVEL, mas não força RUPTURA sozinha.
    estrada_quente = ("quente" in regime.lower()) or ("alta volatilidade" in regime.lower())

    # RUPTURA: risco alto/saturação (quando sinais existem)
    if (k is not None and k >= 0.40) or (nr is not None and nr >= 40.0) or (dv is not None and dv >= 4.5):
        return "RUPTURA"

    # RESPIRÁVEL: ruim, mas jogável (zona cinzenta)
    if estrada_quente:
        return "RESPIRÁVEL"
    if (k is not None and 0.18 <= k < 0.40) or (nr is not None and 25.0 <= nr < 40.0) or (dv is not None and 2.5 <= dv < 4.5):
        return "RESPIRÁVEL"

    return "ESTÁVEL"


def pc_classificar_postura_motor(pipeline_regime: str | None, nr_percent, div_s6_mc):
    """
    Postura do MOTOR (isolada de k/k*).
    Regra canônica desta fase: k/k* é diagnóstico externo e NÃO pode vazar para decisões
    que alterem listas, volumes ou composição de pacote.

    Portanto, esta função IGNORA k/k* e usa apenas sinais do próprio ambiente/execução
    já existentes (NR% e divergência + leitura de estrada).
    """
    return pc_classificar_postura(
        pipeline_regime=pipeline_regime,
        k_star=None,  # isolamento: k não entra
        nr_percent=nr_percent,
        div_s6_mc=div_s6_mc,
    )

def pc_resp_aplicar_diversificacao(listas_totais, listas_top10, universo, seed=0, n_alvo=6, memoria_sufocadores=None, cap_pct=0.65, core_min=None):
    if core_min is None:
        core_min = 0.60

    """Em RESPIRÁVEL, aplicamos *elasticidade mínima* no pacote (pré-C4):
    - Anti-clone leve (remove duplicatas)
    - Anti-core leve (troca 1 passageiro em algumas listas para reduzir compressão)
    Mantém volume e não inventa motor novo.
    """
    try:
        import random
        rng = random.Random(int(seed) if seed is not None else 0)

        if not isinstance(listas_totais, list) or len(listas_totais) == 0:
            return listas_totais, listas_top10, {"aplicado": False, "motivo": "listas_totais vazias"}

        top = listas_top10 if isinstance(listas_top10, list) else []
        top = [lst for lst in top if isinstance(lst, (list, tuple)) and len(lst) >= n_alvo]
        if not top:
            top = [lst for lst in listas_totais if isinstance(lst, (list, tuple)) and len(lst) >= n_alvo][:10]

        # Frequência no Top 10
        freq = {}
        for lst in top:
            for v in lst[:n_alvo]:
                try:
                    vv = int(v)
                except Exception:
                    continue
                freq[vv] = freq.get(vv, 0) + 1

        # CORE: presença >= core_min do top
        core_thr = max(1, int(float(core_min) * max(1, len(top))))
        core = {v for v, c in freq.items() if c >= core_thr}

        # Memória estrutural (sufocadores) — opcional e auditável
        sufocadores = set()
        try:
            if memoria_sufocadores is not None:
                for x in memoria_sufocadores:
                    try:
                        sufocadores.add(int(x))
                    except Exception:
                        pass
        except Exception:
            sufocadores = set()

        # Cap de dominância (só atua em sufocadores)
        cap_pct = float(cap_pct) if cap_pct is not None else 0.65
        cap_pct = max(0.40, min(0.85, cap_pct))

        # Candidatos de baixa frequência (para "respirar")
        uni = [int(x) for x in universo] if isinstance(universo, (list, tuple)) else []
        uni = sorted({u for u in uni if u > 0})
        if not uni:
            return listas_totais, listas_top10, {"aplicado": False, "motivo": "universo indisponível"}

        low_all = sorted(uni, key=lambda u: (freq.get(u, 0), u))
        low_pref = [u for u in low_all if freq.get(u, 0) <= 2]

        def _norm(lst):
            out = []
            for x in lst[:n_alvo]:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            out = sorted(set(out))
            # completa se perdeu algo
            if len(out) < n_alvo:
                for u in low_all:
                    if u not in out:
                        out.append(u)
                    if len(out) >= n_alvo:
                        break
            return sorted(out[:n_alvo])

        def _pick_candidate(base, blocked=None):
            blocked = set(blocked or [])
            pool = [u for u in low_pref if u not in base and u not in blocked]
            if not pool:
                pool = [u for u in low_all if u not in base and u not in blocked]
            for cand in pool:
                novo_pct = float(freq.get(int(cand), 0) + 1) / float(max(1, len(top)))
                if novo_pct <= cap_pct:
                    return int(cand)
            return None

        # Aplica em poucas listas (elasticidade mínima), evitando mexer no pacote inteiro
        new_top = []
        trocas = 0
        for idx, lst in enumerate(top):
            base = _norm(lst)

            # Anti-core leve nas primeiras listas do top
            if idx < 6:
                removiveis = [v for v in base if v in core]
                if len(removiveis) < 2:
                    removiveis = [v for v in base if (float(freq.get(v, 0)) / float(max(1, len(top)))) >= 0.30]
                if not removiveis:
                    removiveis = sorted(base, key=lambda v: (-freq.get(v, 0), v))

                if removiveis:
                    drop = rng.choice(removiveis[:min(len(removiveis), 3)])
                    cand = _pick_candidate(base, blocked={drop})
                    if cand is not None and cand not in base:
                        base2 = [v for v in base if v != drop]
                        base2.append(cand)
                        base2 = sorted(set(base2))
                        while len(base2) < n_alvo:
                            extra = _pick_candidate(base2, blocked={drop})
                            if extra is None:
                                break
                            base2.append(extra)
                            base2 = sorted(set(base2))
                        base2 = sorted(base2[:n_alvo])
                        if base2 != base:
                            base = base2
                            trocas += 1

            # Enforce cap de dominância para sufocadores (apenas nas primeiras listas do top)
            try:
                if sufocadores and idx < 8:
                    for s in list(sufocadores):
                        c0 = freq.get(int(s), 0)
                        if len(top) > 0 and (float(c0) / float(len(top))) > cap_pct and int(s) in base:
                            cand2 = _pick_candidate(base, blocked={int(s)})
                            if cand2 is not None and cand2 not in base:
                                base2 = [v for v in base if v != int(s)]
                                base2.append(int(cand2))
                                base2 = sorted(set(base2))
                                while len(base2) < n_alvo:
                                    extra = _pick_candidate(base2, blocked={int(s)})
                                    if extra is None:
                                        break
                                    base2.append(extra)
                                    base2 = sorted(set(base2))
                                base2 = sorted(base2[:n_alvo])
                                if base2 != base:
                                    base = base2
                                    trocas += 1
                                    break
            except Exception:
                pass

            new_top.append(base)

        # Reconstrói listas_totais: substitui prefixo correspondente ao top, preserva o resto
        listas_totais_norm = [_norm(lst) for lst in listas_totais if isinstance(lst, (list, tuple))]

        # anti-clone global leve
        seen = set()
        uniq = []
        for lst in listas_totais_norm:
            t = tuple(lst)
            if t not in seen:
                seen.add(t)
                uniq.append(lst)

        # injeta new_top na frente (sem duplicar)
        uniq2 = []
        seen2 = set()
        for lst in new_top + uniq:
            t = tuple(lst)
            if t not in seen2:
                seen2.add(t)
                uniq2.append(lst)

        new_tot = uniq2
        new_top10 = new_tot[:10]

        # fallback v16h57FJ: se nada mudou, força 1 troca mínima na 1a lista do top
        if trocas == 0 and new_top10:
            try:
                base = list(new_top10[0])
                drop = sorted(base, key=lambda v: (-freq.get(v, 0), v))[0]
                cand = None
                for u in low_all:
                    if u not in base:
                        cand = int(u)
                        break
                if cand is not None:
                    base2 = sorted(set([v for v in base if v != drop] + [cand]))[:n_alvo]
                    if base2 != sorted(base):
                        new_top10[0] = base2
                        if new_tot:
                            new_tot[0] = base2
                        trocas = 1
            except Exception:
                pass

        
        # v16h57FJ safety: guarantee at least one minimal swap if calibration active
        try:
            if trocas == 0 and new_top10:
                base = list(new_top10[0])
                drop = sorted(base)[0]
                cand = None
                for u in low_all:
                    if u not in base:
                        cand = u
                        break
                if cand is not None:
                    base2 = sorted(set([v for v in base if v != drop] + [cand]))[:n_alvo]
                    if base2 != sorted(base):
                        new_top10[0] = base2
                        if new_tot:
                            new_tot[0] = base2
                        trocas = 1
        except Exception:
            pass
        print("DEBUG_RESP", "trocas=", trocas, "core_sz=", len(core), "low_pref_sz=", len(low_pref), "top_sz=", len(top))

        try:
            _mudou_top_bruto = bool(new_top != [_norm(lst) for lst in top])
        except Exception:
            _mudou_top_bruto = bool(trocas > 0)
        try:
            _mudou_pacote_final = bool(new_tot != listas_totais_norm)
        except Exception:
            _mudou_pacote_final = bool(trocas > 0)
        try:
            _dedup_removeu_qtd = max(0, int(len(listas_totais_norm) - len(uniq)))
        except Exception:
            _dedup_removeu_qtd = 0

        info = {
            "aplicado": bool(trocas > 0),
            "trocas": int(trocas),
            "trocas_brutas": int(trocas),
            "trocas_final": int(trocas),
            "core_sz": len(core),
            "core_thr": int(core_thr),
            "low_pref_sz": len(low_pref),
            "sufocadores_sz": len(sufocadores),
            "cap_pct": float(cap_pct),
            "mudou_top_bruto": bool(_mudou_top_bruto),
            "mudou_pacote_final": bool(_mudou_pacote_final),
            "normalizacao_ocorreu": True,
            "dedup_removeu_qtd": int(_dedup_removeu_qtd),
            "pacote_entrada_qtd": int(len(listas_totais_norm)),
            "pacote_saida_qtd": int(len(new_tot)),
            "fallback_forcado": bool(trocas > 0 and not _mudou_top_bruto),
            "motivo": "respiravel_diversificacao_minima" if trocas > 0 else "sem_trocas_validas",
        }
        return new_tot, new_top10, info
    except Exception as e:
        return listas_totais, listas_top10, {"aplicado": False, "motivo": f"falha_resp: {e}"}

def pc_v16_aplicar_resp_no_pacote(*, pacote_atual, k_reg, universo_min=1, universo_max=60, calib_leve_summary=None, top10_override=None, memoria_sufocadores=None, cap_pct=None, core_min=0.40):
    """Rotina canônica de aplicação RESP no pacote (pré-C4, auditável)."""
    try:
        pacote_baseline = []
        for lst in (pacote_atual or []):
            try:
                li = [int(x) for x in lst]
                if len(li) >= 6:
                    pacote_baseline.append(li[:6])
            except Exception:
                continue
        if not pacote_baseline:
            return {
                "pacote_baseline": [],
                "pacote_store": [],
                "top10_final": [],
                "resp_info": {"aplicado": False, "motivo": "pacote_vazio"},
                "calib_active": False,
                "calib_should_apply": False,
                "calib_applied": False,
                "I_mean": 0.0,
                "I_max": 0.0,
                "I2_mean": 0.0,
                "I2_max": 0.0,
                "thr_base": 0.25,
                "diff_conteudo_qtd": 0,
                "diff_posicao_qtd": 0,
                "diff_dedup_qtd": 0,
            }

        calib_leve = calib_leve_summary if isinstance(calib_leve_summary, dict) else (st.session_state.get("v16_calib_leve_last_summary") or {})
        if not isinstance(calib_leve, dict):
            calib_leve = {}

        def _get_num(d, keys, default=0.0):
            for k in keys:
                try:
                    v = d.get(k, None)
                    if v is None:
                        continue
                    return float(v)
                except Exception:
                    continue
            return float(default)

        I_mean = _get_num(calib_leve, ["I_mean", "I_media", "I", "I_val"], 0.0)
        I_max  = _get_num(calib_leve, ["I_max", "Imax", "I_maximo"], I_mean)
        I2_mean = _get_num(calib_leve, ["I2_mean", "I2_media", "I2"], 0.0)
        I2_max  = _get_num(calib_leve, ["I2_max", "I2max", "I2_maximo"], I2_mean)
        THR_BASE = float(calib_leve.get("thr_base", 0.25) or 0.25)

        I2_val = float(I2_max if I2_max is not None else I2_mean)
        I_val  = float(I_max if I_max is not None else I_mean)
        calib_active = bool(I2_val > 0.0 or I_val > 0.0)
        calib_should_apply = bool(calib_active and (I2_val >= THR_BASE))

        pacote_store = [list(lst) for lst in pacote_baseline]
        top10_final = list(top10_override) if isinstance(top10_override, list) and top10_override else pacote_store[:10]
        resp_info = {"aplicado": False, "motivo": "nao_aplicado"}

        if calib_should_apply:
            try:
                universo_resp = list(range(int(universo_min), int(universo_max) + 1))
            except Exception:
                universo_resp = list(range(1, 61))
            universo_resp = [int(x) for x in universo_resp if int(x) > 0]
            top10_resp = list(top10_override) if isinstance(top10_override, list) and top10_override else (pacote_store[:10] if len(pacote_store) >= 10 else list(pacote_store))
            _cap_pct = float(cap_pct) if cap_pct is not None else (0.35 if (I2_val >= 0.65) else 0.45)
            new_tot, new_top10, resp_info = pc_resp_aplicar_diversificacao(
                listas_totais=pacote_store,
                listas_top10=top10_resp,
                universo=universo_resp,
                seed=int(k_reg),
                n_alvo=6,
                memoria_sufocadores=memoria_sufocadores,
                cap_pct=_cap_pct,
                core_min=core_min,
            )
            if isinstance(new_tot, list) and len(new_tot) > 0:
                _tmp = []
                for lst in new_tot:
                    try:
                        li = [int(x) for x in lst]
                        if len(li) >= 6:
                            _tmp.append(li[:6])
                    except Exception:
                        continue
                if _tmp:
                    pacote_store = _tmp
            if isinstance(new_top10, list) and len(new_top10) > 0:
                _top = []
                for lst in new_top10:
                    try:
                        li = [int(x) for x in lst]
                        if len(li) >= 6:
                            _top.append(li[:6])
                    except Exception:
                        continue
                top10_final = _top if _top else pacote_store[:10]
            else:
                top10_final = pacote_store[:10]

        def _tup(lst):
            try:
                return tuple(int(x) for x in lst[:6])
            except Exception:
                return tuple()

        base_tups = [_tup(lst) for lst in pacote_baseline]
        store_tups = [_tup(lst) for lst in pacote_store]
        base_set = set(base_tups)
        store_set = set(store_tups)

        diff_conteudo_qtd = int(len(base_set.symmetric_difference(store_set)))
        diff_posicao_qtd = int(sum(1 for i in range(min(len(base_tups), len(store_tups))) if base_tups[i] != store_tups[i]))
        diff_dedup_qtd = int(abs(len(store_tups) - len(base_tups)))

        _aplicado_flag = bool(resp_info.get("aplicado", False)) if isinstance(resp_info, dict) else False
        _mudou_flag = bool(pacote_store != pacote_baseline)
        calib_applied = bool(_aplicado_flag or _mudou_flag)

        return {
            "pacote_baseline": pacote_baseline,
            "pacote_store": pacote_store,
            "top10_final": top10_final,
            "resp_info": resp_info if isinstance(resp_info, dict) else {"aplicado": False, "motivo": "resp_info_invalido"},
            "calib_active": bool(calib_active),
            "calib_should_apply": bool(calib_should_apply),
            "calib_applied": bool(calib_applied),
            "I_mean": float(I_mean),
            "I_max": float(I_max),
            "I2_mean": float(I2_mean),
            "I2_max": float(I2_max),
            "thr_base": float(THR_BASE),
            "diff_conteudo_qtd": int(diff_conteudo_qtd),
            "diff_posicao_qtd": int(diff_posicao_qtd),
            "diff_dedup_qtd": int(diff_dedup_qtd),
        }
    except Exception as e:
        try:
            _base = [list(map(int, lst[:6])) for lst in (pacote_atual or []) if isinstance(lst, (list, tuple)) and len(lst) >= 6]
        except Exception:
            _base = []
        return {
            "pacote_baseline": _base,
            "pacote_store": list(_base),
            "top10_final": list(_base[:10]),
            "resp_info": {"aplicado": False, "motivo": f"falha_apply_resp: {e}"},
            "calib_active": False,
            "calib_should_apply": False,
            "calib_applied": False,
            "I_mean": 0.0,
            "I_max": 0.0,
            "I2_mean": 0.0,
            "I2_max": 0.0,
            "thr_base": 0.25,
            "diff_conteudo_qtd": 0,
            "diff_posicao_qtd": 0,
            "diff_dedup_qtd": 0,
        }

def pc_v16_mc_observacional_pacote_pre_c4(
    *,
    modo6_listas_totais,
    modo6_listas_top10,
    historico_df_full,
    nocivos_consistentes=None,
    w_alvos: int = 60,
    sims: int = 200,
) -> dict:
    """MC Observacional do Pacote (pré-C4)
    Objetivo: medir (sem decisão e sem alterar listas) se rigidez/overlap e nocivos
    estão estatisticamente derrubando a chance de ≥3/≥4 no filme curto.

    Regras: sem novos botões/controles; parâmetros fixos e auditáveis.
    """
    nocivos_consistentes = set(nocivos_consistentes or [])
    # --- alvos reais (últimos W) ---
    df = historico_df_full.copy()
    # tenta detectar colunas com 6 números
    cols_nums = [c for c in df.columns if isinstance(c, str) and c.lower() in ["n1","n2","n3","n4","n5","n6","a","b","c","d","e","f"]]
    if len(cols_nums) >= 6:
        cols_nums = cols_nums[:6]
        alvos = df[cols_nums].tail(w_alvos).values.tolist()
    else:
        # fallback: procurar coluna 'nums'/'lista' com iterável
        col_cand = None
        for c in df.columns:
            if str(c).lower() in ["nums","numeros","lista","sorteio","target","alvo"]:
                col_cand = c
                break
        if col_cand is None:
            raise ValueError("Histórico não possui colunas de números reconhecíveis para MC.")
        alvos = []
        for v in df[col_cand].tail(w_alvos).tolist():
            if isinstance(v, (list, tuple)) and len(v) >= 6:
                alvos.append(list(v)[:6])
            else:
                s = str(v)
                nums = [int(x) for x in re.findall(r"\d+", s)]
                if len(nums) >= 6:
                    alvos.append(nums[:6])
        if not alvos:
            raise ValueError("Não foi possível extrair alvos (últimos W) do histórico para MC.")

    # --- avaliador ---
    def eval_pacote(pacote_listas):
        bests = []
        hit3 = 0
        hit4 = 0
        for alvo in alvos:
            bh = _pc_melhor_hit_do_pacote(alvo, pacote_listas)
            bests.append(bh)
            if bh >= 3:
                hit3 += 1
            if bh >= 4:
                hit4 += 1
        n = max(1, len(alvos))
        return {
            "avg_best": float(sum(bests) / n),
            "rate_3p": float(hit3 / n),
            "rate_4p": float(hit4 / n),
            "max_best": int(max(bests) if bests else 0),
        }

    # --- métricas de rigidez (simples, auditáveis) ---
    def rigidez(pacote_listas):
        if not pacote_listas:
            return {"ov_mean": 0.0, "core_sz": 0, "frac_colados": 0.0}
        L = len(pacote_listas)
        ovs = []
        for i in range(L):
            si = set(pacote_listas[i])
            for j in range(i + 1, L):
                sj = set(pacote_listas[j])
                ovs.append(len(si & sj))
        ov_mean = float(sum(ovs) / max(1, len(ovs)))
        freq = {}
        for li in pacote_listas:
            for p in li:
                freq[p] = freq.get(p, 0) + 1
        core = [p for p, f in freq.items() if f / L >= 0.6]
        frac_colados = float(sum(1 for x in ovs if x >= 3) / max(1, len(ovs)))
        return {"ov_mean": ov_mean, "core_sz": len(core), "frac_colados": frac_colados, "core": sorted(core)}

    def nocivo_share(pacote_listas):
        if not pacote_listas or not nocivos_consistentes:
            return 0.0
        flat = [p for li in pacote_listas for p in li]
        return float(sum(1 for p in flat if p in nocivos_consistentes) / max(1, len(flat)))

    baseline = list(modo6_listas_top10 or [])
    baseline_eval = eval_pacote(baseline)
    baseline_rig = rigidez(baseline)
    baseline_noc = nocivo_share(baseline)

    def sim_diversificado(cap_pct: float):
        import random
        rates4 = []
        rates3 = []
        avgs = []
        ovm = []
        noc = []
        resp_called = 0
        resp_aplicado = 0
        fallback_baseline = 0
        pacote_origem_counts = {"baseline": 0, "resp_top10": 0, "resp_tot10": 0}
        resp_motivos = {}
        try:
            universo_resp = sorted({int(x) for lst in (modo6_listas_totais or []) for x in (lst or []) if int(x) > 0})
        except Exception:
            universo_resp = []
        if not universo_resp:
            universo_resp = list(range(1, 61))
        top10_resp = list(modo6_listas_top10 or [])

        for s in range(sims):
            random.seed(1337 + s)
            pacote = baseline
            origem = "baseline"
            try:
                resp_called += 1
                new_tot, new_top10, resp_info = pc_resp_aplicar_diversificacao(
                    listas_totais=modo6_listas_totais,
                    listas_top10=top10_resp,
                    universo=universo_resp,
                    cap_pct=cap_pct,
                    seed=1337 + s,
                    n_alvo=6,
                )
                if isinstance(resp_info, dict):
                    motivo = str(resp_info.get("motivo", "sem_motivo"))
                    resp_motivos[motivo] = resp_motivos.get(motivo, 0) + 1
                    if bool(resp_info.get("aplicado", False)):
                        resp_aplicado += 1
                if isinstance(new_top10, list) and len(new_top10) > 0:
                    pacote = [list(map(int, lst[:6])) for lst in new_top10 if isinstance(lst, (list, tuple)) and len(lst) >= 6]
                    origem = "resp_top10"
                elif isinstance(new_tot, list) and len(new_tot) > 0:
                    pacote = [list(map(int, lst[:6])) for lst in new_tot[:10] if isinstance(lst, (list, tuple)) and len(lst) >= 6]
                    origem = "resp_tot10"
                else:
                    fallback_baseline += 1
            except Exception as e:
                motivo = f"falha_mc_resp: {e}"
                resp_motivos[motivo] = resp_motivos.get(motivo, 0) + 1
                fallback_baseline += 1
                pacote = baseline
                origem = "baseline"
            pacote_origem_counts[origem] = pacote_origem_counts.get(origem, 0) + 1
            ev = eval_pacote(pacote)
            rg = rigidez(pacote)
            rates4.append(ev["rate_4p"])
            rates3.append(ev["rate_3p"])
            avgs.append(ev["avg_best"])
            ovm.append(rg["ov_mean"])
            noc.append(nocivo_share(pacote))
        def q(a, p):
            a2 = sorted(a)
            if not a2:
                return None
            k = int(round((len(a2) - 1) * p))
            k = max(0, min(len(a2) - 1, k))
            return float(a2[k])
        return {
            "cap_pct": cap_pct,
            "rate_4p_mean": float(sum(rates4)/max(1,len(rates4))),
            "rate_4p_p10": q(rates4, 0.10),
            "rate_4p_p90": q(rates4, 0.90),
            "rate_3p_mean": float(sum(rates3)/max(1,len(rates3))),
            "avg_best_mean": float(sum(avgs)/max(1,len(avgs))),
            "ov_mean_mean": float(sum(ovm)/max(1,len(ovm))),
            "nocivo_share_mean": float(sum(noc)/max(1,len(noc))),
            "resp_called": int(resp_called),
            "resp_aplicado": int(resp_aplicado),
            "fallback_baseline": int(fallback_baseline),
            "pacote_mc_origem_counts": pacote_origem_counts,
            "universo_resp_qtd": int(len(universo_resp)),
            "resp_motivos": resp_motivos,
        }

    scen_loose = sim_diversificado(0.60)
    scen_tight = sim_diversificado(0.85)

    delta_4p_pp = (scen_loose["rate_4p_mean"] - baseline_eval["rate_4p"]) * 100.0
    rigidez_matando = (
        (delta_4p_pp >= 0.5)
        and (scen_loose["ov_mean_mean"] < baseline_rig["ov_mean"])
        and (scen_loose["nocivo_share_mean"] <= baseline_noc + 1e-9)
    )

    return {
        "w_alvos": int(len(alvos)),
        "sims": int(sims),
        "baseline": {**baseline_eval, **baseline_rig, "nocivo_share": baseline_noc},
        "scenario_loose": scen_loose,
        "scenario_tight": scen_tight,
        "rigidez_matando": bool(rigidez_matando),
        "delta_4p_pp_loose_vs_base": float(delta_4p_pp),
        "nota": "Pré-C4 · Observacional · Não altera listas · Parâmetros fixos (W=60, sims=200).",
    }



def pc_resp_memoria_estrutural_from_snapshots(snapshot_p0_canonic, lookback: int = 25, top_n: int = 8, min_lists: int = 5):
    """Memória Estrutural do RESPIRÁVEL (pré-C4, auditável, sem motor novo).

    Fonte: snapshots P0 canônicos já registrados (Replay Progressivo / CAP Invisível).
    Ideia: identificar passageiros que tendem a *dominar* o pacote (compressão) de forma recorrente.

    Retorna:
      {
        "ok": bool,
        "sufocadores": [int...],   # top_n
        "stats": {...},            # auditável (média de dominância / top ocorrências)
        "motivo": str
      }
    """
    try:
        if not isinstance(snapshot_p0_canonic, dict) or len(snapshot_p0_canonic) == 0:
            return {"ok": False, "sufocadores": [], "stats": {}, "motivo": "sem_snapshots"}

        # pega os últimos ks (ordem crescente)
        ks = []
        for k in snapshot_p0_canonic.keys():
            try:
                ks.append(int(k))
            except Exception:
                continue
        ks = sorted(ks)
        if not ks:
            return {"ok": False, "sufocadores": [], "stats": {}, "motivo": "ks_invalidos"}

        ks_use = ks[-int(max(1, lookback)):]
        score = {}        # soma de freq_norm
        occ = {}          # contagem de aparições
        dom_vals = []     # dominância do top1 por snapshot

        for k in ks_use:
            snap = snapshot_p0_canonic.get(k) or {}
            try:
                qtd = int(snap.get("qtd_listas", 0))
            except Exception:
                qtd = 0
            if qtd < int(min_lists):
                continue

            freq = snap.get("freq_passageiros") or {}
            items = []
            for pk, pv in freq.items():
                try:
                    p = int(pk)
                    v = int(pv)
                except Exception:
                    continue
                if v <= 0:
                    continue
                items.append((p, v))
            if not items:
                continue

            items.sort(key=lambda kv: (-kv[1], kv[0]))
            top1 = items[0][1]
            dom_vals.append(float(top1) / float(max(1, qtd)))

            for p, v in items:
                fn = float(v) / float(max(1, qtd))
                score[p] = score.get(p, 0.0) + fn
                occ[p] = occ.get(p, 0) + 1

        if not score:
            return {"ok": False, "sufocadores": [], "stats": {}, "motivo": "score_vazio"}

        # ranking por "dominância média" (score / occ), com desempate por score total
        rank = []
        for p, sc in score.items():
            o = int(occ.get(p, 1))
            rank.append((p, sc / float(max(1, o)), sc, o))
        rank.sort(key=lambda t: (-t[1], -t[2], -t[3], t[0]))

        suf = [int(p) for p, _, _, _ in rank[: int(max(1, top_n))]]

        stats = {
            "snapshots_usados": int(len(ks_use)),
            "k_min": int(ks_use[0]),
            "k_max": int(ks_use[-1]),
            "dominancia_top1_media": float(sum(dom_vals) / max(1, len(dom_vals))) if dom_vals else 0.0,
            "top_ocorrencias": [
                {"p": int(p), "occ": int(o), "score_total": float(sc), "score_medio": float(sc / max(1, o))}
                for p, _, sc, o in rank[: min(12, len(rank))]
            ],
        }

        return {"ok": True, "sufocadores": suf, "stats": stats, "motivo": "ok"}
    except Exception as e:
        return {"ok": False, "sufocadores": [], "stats": {}, "motivo": f"falha_memoria_resp: {e}"}



def _p2_series_from_df(df, idx, n=6):
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p")]
    cols = sorted(cols, key=lambda x: int(x[1:]))[:int(n)]
    return [int(df.loc[idx, c]) for c in cols if c in df.columns]

def _p2_avaliar_fora(universo, alvo):
    return list(set(alvo) - set(universo))


# =====================
# FIX P2 — RESOLUÇÃO ROBUSTA DE k NO HISTÓRICO FULL
# =====================
def _p2_resolver_posicao_k(_df_full_safe, k):
    # 1) Se k for índice explícito
    if k in _df_full_safe.index:
        return list(_df_full_safe.index).index(k)
    # 2) Se houver coluna 'k' ou 'serie'
    for col in ["k", "serie", "serie_id"]:
        if col in _df_full_safe.columns:
            matches = _df_full_safe.index[_df_full_safe[col] == k].tolist()
            if matches:
                return list(_df_full_safe.index).index(matches[0])
    # 3) Fallback 1-based -> 0-based
    if isinstance(k, int) and 1 <= k <= len(_df_full_safe):
        return k - 1
    raise ValueError(f"k={k} não encontrado no histórico FULL")


def p2_calcular_cap_dinamico(universo_len, fora_longe, fora_total, score_rigidez):
    if fora_total <= 0:
        return 0
    p_longe = fora_longe / max(fora_total, 1)
    if p_longe >= 0.70:
        f_longe = 1.5
    elif p_longe >= 0.40:
        f_longe = 1.0
    else:
        f_longe = 0.0
    if score_rigidez >= 0.75:
        f_rig = 0.7
    elif score_rigidez >= 0.40:
        f_rig = 1.0
    else:
        f_rig = 1.3
    cap_base = math.ceil(0.20 * max(1, int(universo_len)))
    return max(0, int(round(cap_base * f_longe * f_rig)))

def p2_h1_freq(df_hist, universo, cap, n=6):
    if cap <= 0:
        return [], universo
    freq = {}
    for i in df_hist.index:
        for v in _p2_series_from_df(df_hist, i, n=n):
            freq[v] = freq.get(v, 0) + 1
    cand = [(v, c) for v, c in freq.items() if v not in universo]
    cand.sort(key=lambda x: x[1], reverse=True)
    adds = [v for v, _ in cand[:cap]]
    return adds, sorted(set(universo) | set(adds))

def p2_h2_dist(df_hist, universo, cap, n=6):
    if cap <= 0:
        return [], universo
    dist = {}
    idxs = list(df_hist.index)
    for i in range(1, len(idxs)):
        a = set(_p2_series_from_df(df_hist, idxs[i-1], n=n))
        b = set(_p2_series_from_df(df_hist, idxs[i], n=n))
        for v in (a | b):
            dist[v] = dist.get(v, 0) + len(a.symmetric_difference(b))
    cand = [(v, d) for v, d in dist.items() if v not in universo]
    cand.sort(key=lambda x: x[1], reverse=True)
    adds = [v for v, _ in cand[:cap]]
    return adds, sorted(set(universo) | set(adds))


# ============================================================
# CAMADA 3 — PARABÓLICA (GOVERNO PRÉ-C4) — MULTI-ESCALA + VETORIAL + PERSISTÊNCIA
# ============================================================
# Regra canônica:
# - Leitura apenas (pré-C4)
# - Governa permissões estruturais (P1/P2) sem decidir ataque
# - Critérios objetivos (sem chute): multi-escala + persistência mínima
# ============================================================

def _parab_safe_float(x, default=0.0):
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return default
        return xf
    except Exception:
        return default

def _parab_state_from_curvature(c: float, eps: float) -> str:
    if c < -eps:
        return "DESCENDO"
    if c > eps:
        return "SUBINDO"
    return "PLANA"

def _parab_auto_eps(dE, C):
    # eps objetivo: escala pelo ruído real observado (dE e curvatura)
    try:
        mdE = float(np.median(np.abs(dE))) if len(dE) else 0.0
        mC = float(np.median(np.abs(C))) if len(C) else 0.0
        # piso mínimo evita “tremedeira” quando valores são pequenos
        return max(0.05, 0.22 * mdE + 0.05 * mC)
    except Exception:
        return 0.05

def _parab_resolver_pos_k(df: pd.DataFrame, k_val: int):
    idxs = list(df.index)
    if k_val in idxs:
        return idxs.index(k_val), idxs
    if (k_val - 1) in idxs:
        return idxs.index(k_val - 1), idxs
    if (k_val + 1) in idxs:
        return idxs.index(k_val + 1), idxs
    try:
        if hasattr(df.index, "start") and hasattr(df.index, "stop"):
            if df.index.start <= k_val < df.index.stop:
                return int(k_val - df.index.start), idxs
    except Exception:
        pass
    return None, idxs

def _parab_series_from_df(df: pd.DataFrame, idx, n: int = 6):
    cols = [c for c in df.columns if str(c).startswith("p")]
    if len(cols) >= n:
        return [int(df.loc[idx, c]) for c in cols[:n]]
    vals = []
    for c in df.columns:
        try:
            vals.append(int(df.loc[idx, c]))
        except Exception:
            pass
    return vals[:n]

def _parab_erro_snapshot(df: pd.DataFrame, snap: dict, n: int = 6):
    universo = snap.get("universo_pacote") or snap.get("universo") or snap.get("universo_p0") or []
    try:
        universo = [int(x) for x in universo]
    except Exception:
        universo = []
    uni_set = set(universo)

    k_val = int(snap.get("k", -1))
    pos, idxs = _parab_resolver_pos_k(df, k_val)
    if pos is None:
        return None

    alvos = []
    for off in (1, 2):
        if pos + off < len(idxs):
            alvos.append(_parab_series_from_df(df, idxs[pos + off], n=n))

    fora_total = 0
    fora_perto = 0
    fora_longe = 0
    dist_medias = []

    for alvo in alvos:
        alvo_set = set(alvo)
        try:
            r = _v9_trave_proximidade(alvo_set, uni_set, thr=2)
            fora_total += int(r.get("fora_total", 0))
            fora_perto += int(r.get("fora_perto", 0))
            fora_longe += int(r.get("fora_longe", 0))
            dist_medias.append(float(r.get("dist_media", 0.0)))
        except Exception:
            fora = list(alvo_set - uni_set)
            fora_total += len(fora)
            fora_longe += len(fora)

    dist_media = float(np.mean(dist_medias)) if dist_medias else 0.0

    return {
        "k": k_val,
        "universo_len": len(universo),
        "score_rigidez": _parab_safe_float(snap.get("score_rigidez", 0.0)),
        "fora_total": int(fora_total),
        "fora_perto": int(fora_perto),
        "fora_longe": int(fora_longe),
        "dist_media": float(dist_media),
    }

def _parab_compute_curvature(vals):
    # retorna dE, C (ddE), eps_auto, curvatura_atual
    E = [_parab_safe_float(v) for v in vals]
    dE = np.diff(E) if len(E) >= 2 else np.array([])
    C = np.diff(dE) if len(dE) >= 2 else np.array([])
    eps = _parab_auto_eps(dE, C)
    curv = float(C[-1]) if len(C) else 0.0
    return E, dE.tolist() if len(dE) else [], C.tolist() if len(C) else [], float(eps), float(curv)

def parabola_multiescala_vetorial(_df_full_safe: pd.DataFrame, snapshots_map: dict, n: int = 6):
    # snapshots_map: dict[k]->snapshot(dict)
    ks = sorted([int(k) for k in snapshots_map.keys() if str(k).isdigit()])
    if len(ks) < 3:
        return None

    # escalas objetivas (sem chute): curta/média/longa dentro do que existir
    W_short = min(5, len(ks))
    W_mid = min(9, len(ks))
    W_long = min(15, len(ks))

    def _serie(W):
        ks_w = ks[-W:]
        serie = []
        for kk in ks_w:
            s = snapshots_map.get(kk)
            if isinstance(s, dict):
                e = _parab_erro_snapshot(_df_full_safe, s, n=n)
                if e is not None:
                    serie.append(e)
        return serie

    out = {"ks_all": ks, "Ws": {"short": W_short, "mid": W_mid, "long": W_long}}

    # métrica primária e confirmadoras (vetorial)
    metrics = {
        "fora_longe": {"label": "fora_longe"},
        "fora_total": {"label": "fora_total"},
        "dist_media": {"label": "dist_media"},
        "universo_len": {"label": "universo_len"},
        "score_rigidez": {"label": "score_rigidez"},
    }

    scales = {"short": _serie(W_short), "mid": _serie(W_mid), "long": _serie(W_long)}
    out["series"] = scales

    # estados por (escala, métrica)
    estados = {}
    debug = {}

    for sname, serie in scales.items():
        estados[sname] = {}
        debug[sname] = {}
        if len(serie) < 3:
            for mname in metrics.keys():
                estados[sname][mname] = "N/D"
            debug[sname]["motivo"] = "serie_insuficiente"
            continue
        for mname in metrics.keys():
            vals = [row.get(mname, 0.0) for row in serie]
            E, dE, C, eps, curv = _parab_compute_curvature(vals)
            stt = _parab_state_from_curvature(curv, eps)
            estados[sname][mname] = stt
            debug[sname][mname] = {"E": E, "dE": dE, "C": C, "eps": eps, "curv": curv}

    out["estados"] = estados
    out["debug"] = debug

    # persistência objetiva:
    # - exige ao menos 2 curvaturas consecutivas na mesma direção na escala longa (métrica primária)
    pers = {"ok_subindo": False, "ok_descendo": False, "motivo": None}
    try:
        dbg = debug.get("long", {}).get("fora_longe", {})
        C = dbg.get("C", [])
        eps = float(dbg.get("eps", 0.05))
        if len(C) >= 2:
            c1, c2 = float(C[-2]), float(C[-1])
            pers["ok_subindo"] = (c1 > eps and c2 > eps)
            pers["ok_descendo"] = (c1 < -eps and c2 < -eps)
        else:
            pers["motivo"] = "C_insuficiente"
    except Exception:
        pers["motivo"] = "erro_persistencia"

    out["persistencia"] = pers

    # estado global (governo): usa LONG primário como “mundo”, SHORT como “momento”
    st_long = estados.get("long", {}).get("fora_longe", "N/D")
    st_short = estados.get("short", {}).get("fora_longe", "N/D")

    if st_long == "SUBINDO" and pers["ok_subindo"]:
        estado_global = "SUBINDO"
    elif st_long == "DESCENDO" and pers["ok_descendo"]:
        estado_global = "DESCENDO"
    else:
        # se curto diverge do longo, tratamos como PLANA (instável/ambíguo) por governança
        estado_global = "PLANA"

    # P2 “acorda” somente se:
    # - global SUBINDO (com persistência)
    # - e pelo menos 1 confirmador (fora_total OU dist_media) também SUBINDO em long ou mid
    conf = False
    for conf_metric in ("fora_total", "dist_media"):
        if estados.get("long", {}).get(conf_metric) == "SUBINDO" or estados.get("mid", {}).get(conf_metric) == "SUBINDO":
            conf = True
    p2_permitido = (estado_global == "SUBINDO") and conf

    out["estado_global"] = estado_global
    out["p2_permitido"] = bool(p2_permitido)
    out["confirmacao"] = {"confirmadores_subindo": bool(conf)}

    return out


def v16_detector_ritmo_danca_expost(gov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """🕺 Detector de Ritmo/Dança (ex-post, pré-C4)
    - OBSERVACIONAL: não decide, não altera Camada 4
    - Usa apenas a saída já existente da Parabólica (gov/estados/persistência)
    - Objetivo: marcar quando o erro está *abrindo caminho* (fora_longe DESCENDO com persistência)
      com confirmação mínima em (fora_total/dist_media), para servir de base à Memória Estrutural do RESPIRÁVEL.
    """
    out: Dict[str, Any] = {
        "ritmo_global": "N/D",
        "motivos": [],
        "sinais": {},
        "map_por_k": [],
        "obs": "Pré-C4. Observacional. Usa Parabólica (multi-escala) como fonte.",
    }

    if not isinstance(gov, dict) or not gov:
        out["motivos"].append("gov_indisponivel")
        return out

    estados = gov.get("estados") or {}
    persist = gov.get("persistencia") or {}
    debug = gov.get("debug") or {}

    st_long = (estados.get("long") or {}).get("fora_longe", "N/D")
    st_mid = (estados.get("mid") or {}).get("fora_longe", "N/D")
    st_short = (estados.get("short") or {}).get("fora_longe", "N/D")

    ok_desc = bool(persist.get("ok_descendo"))
    ok_sub = bool(persist.get("ok_subindo"))

    # confirmadores: queremos ver também DESCENDO (melhora) em pelo menos 1 confirmador
    conf_desc = 0
    for m in ("fora_total", "dist_media"):
        if (estados.get("long", {}) or {}).get(m) == "DESCENDO" or (estados.get("mid", {}) or {}).get(m) == "DESCENDO":
            conf_desc += 1

    sinais = {
        "fora_longe_long": st_long,
        "fora_longe_mid": st_mid,
        "fora_longe_short": st_short,
        "persist_ok_descendo": ok_desc,
        "persist_ok_subindo": ok_sub,
        "confirmadores_descendo_qtd": conf_desc,
    }
    out["sinais"] = sinais

    # Regras canônicas (simples, auditáveis):
    # - Ritmo CLARO: fora_longe DESCENDO em LONG + persistência + pelo menos 1 confirmador DESCENDO
    # - Ritmo FRACO: fora_longe DESCENDO em LONG + persistência (sem confirmador) OU LONG DESCENDO e MID/SHORT DESCENDO
    # - Sem ritmo: demais casos (inclui PLANA/SUBINDO)
    if st_long == "DESCENDO" and ok_desc and conf_desc >= 1:
        out["ritmo_global"] = "RITMO_CLARO"
        out["motivos"].append("fora_longe_descendo_persistente_com_confirmacao")
    elif st_long == "DESCENDO" and ok_desc and (st_mid == "DESCENDO" or st_short == "DESCENDO"):
        out["ritmo_global"] = "RITMO_FRACO"
        out["motivos"].append("fora_longe_descendo_persistente_sem_confirmacao_forte")
    else:
        out["ritmo_global"] = "SEM_RITMO"
        if st_long in ("SUBINDO", "PLANA", "N/D"):
            out["motivos"].append(f"fora_longe_long_{st_long}")
        if ok_sub:
            out["motivos"].append("persistencia_subindo")
        if not ok_desc:
            out["motivos"].append("sem_persistencia_descendo")

    # Mapa por k (últimos ks na escala LONG, se disponível)
    try:
        ks = gov.get("ks") or []
        ws = gov.get("Ws") or {}
        w_long = int(ws.get("long", 0) or 0)
        if isinstance(ks, list) and w_long > 0:
            out["map_por_k"] = [int(k) for k in ks[-min(10, w_long):] if str(k).isdigit()]
    except Exception:
        pass

    # Auditoria extra: curvatura final fora_longe (se houver)
    try:
        dbg = (debug.get("long") or {}).get("fora_longe") or {}
        C = dbg.get("C") or []
        eps = dbg.get("eps")
        if isinstance(C, list) and C:
            out["sinais"]["curvatura_fora_longe_ult"] = float(C[-1])
        if eps is not None:
            out["sinais"]["eps"] = float(eps)
    except Exception:
        pass

    return out



# ============================================================
# >>> P1 AUTOMÁTICO (pré-C4) — Governado pela Parabólica
# - Não toca Camada 4
# - Auditável
# - Usa apenas histórico + Snapshot P0 canônico
# ============================================================

def _p1__build_ub_from_snapshot(snapshot: dict, umin: int, umax: int) -> dict:
    """Constrói universo UB (P1.B) a partir do Snapshot P0 canônico.
    Estratégia defensiva: adiciona vizinhos (+/-1) dos passageiros mais frequentes do pacote,
    limitado e sem explosão. Sempre respeita [umin, umax]."""
    if not isinstance(snapshot, dict):
        return {"UB": [], "adds_B": [], "motivo": "snapshot_invalido"}

    u0 = snapshot.get("universo_pacote") or []
    try:
        u0 = sorted({int(x) for x in u0 if int(umin) <= int(x) <= int(umax)})
    except Exception:
        u0 = []
    if not u0:
        return {"UB": [], "adds_B": [], "motivo": "u0_vazio"}

    # Top frequentes do pacote (ex-ante, vindo do snapshot)
    freq = snapshot.get("freq_passageiros") or {}
    top = []
    try:
        # freq pode estar como dict[str]->int
        items = []
        for k, v in freq.items():
            try:
                items.append((int(k), int(v)))
            except Exception:
                continue
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        top = [k for k, _ in items[:12]]
    except Exception:
        top = []

    if not top:
        # fallback: usa o próprio universo do pacote como "top"
        top = u0[:12]

    def clamp(v: int) -> int:
        return max(int(umin), min(int(umax), int(v)))

    cand = []
    for x in top:
        try:
            xi = int(x)
        except Exception:
            continue
        cand.append(clamp(xi - 1))
        cand.append(clamp(xi + 1))

    # remove já existentes e limita
    u0_set = set(u0)
    adds = []
    for x in cand:
        if x not in u0_set and x not in adds:
            adds.append(x)
        if len(adds) >= 8:
            break

    ub = sorted(u0_set | set(adds))
    return {"UB": ub, "adds_B": adds, "motivo": "ok"}


def _ambiente_ruim(*, k_star: float, indice_risco: float | None, regime_txt: str | None) -> bool:
    """Heurística conservadora para 'RUIM/TURBULENTO' (pré-C4).
    Não decide ataque; apenas governa se P1 pode ser aplicado defensivamente."""
    try:
        ks = float(k_star or 0.0)
    except Exception:
        ks = 0.0
    try:
        ir = float(indice_risco) if indice_risco is not None else None
    except Exception:
        ir = None
    rt = (regime_txt or "").lower()

    if "ruptura" in rt or "turbul" in rt:
        return True
    if ir is not None and ir >= 0.55:
        return True
    if ks >= 0.20:
        return True
    return False


def _p1_auto_decidir(_df_full_safe, snaps_map: dict, k_ref: int) -> dict:
    """Decide (pré-C4) se P1 deve ser aplicado automaticamente.
    Retorna: {eligivel, motivo, estado_global, ub, adds_B} (tudo auditável)."""
    # CAP calibrada?
    cap_status = str(st.session_state.get("cap_status") or "")
    cap_ok = cap_status.startswith("CALIBRADA")
    # Parabólica (se não visitou o painel ainda, calcula aqui em modo leitura)
    estado_global = st.session_state.get("parabola_estado_global")
    gov = st.session_state.get("parabola_gov")

    if estado_global is None or not isinstance(gov, dict):
        try:
            n = int(st.session_state.get("n_alvo") or 6)
            gov = parabola_multiescala_vetorial(_df_full_safe, snaps_map, n=n) or {}
            estado_global = gov.get("estado_global")
            st.session_state["parabola_estado_global"] = estado_global
            st.session_state["parabola_gov"] = gov
        except Exception:
            gov = {}
            estado_global = None

    if estado_global not in ("PLANA", "DESCENDO"):
        return {"eligivel": False, "motivo": f"estado_global_{estado_global}"}

    # Persistência: se há melhora sustentada, não corrige (desliga P1)
    persist = (gov.get("persistencia") or {}) if isinstance(gov, dict) else {}
    ok_subindo = bool(persist.get("ok_subindo"))
    if ok_subindo:
        return {"eligivel": False, "motivo": "melhora_persistente"}


    # CAP "mincal": calibração mínima suficiente para liberar P1 defensivo (sem liberar P2).
    Ws = gov.get("Ws") if isinstance(gov, dict) else {}
    try:
        ws_short = int((Ws or {}).get("short") or 0)
        ws_mid = int((Ws or {}).get("mid") or 0)
        ws_long = int((Ws or {}).get("long") or 0)
    except Exception:
        ws_short = ws_mid = ws_long = 0
    ws_ok = (ws_short >= 5) or (ws_mid >= 5) or (ws_long >= 5)

    # Se CAP não está "CALIBRADA", só libera P1 defensivo quando a Parabólica tem calibração mínima suficiente.
    if (not cap_ok) and (not ws_ok):
        return {"eligivel": False, "motivo": "cap_nao_calibrada"}
    # Ambiente RUIM/TURBULENTO (defensivo)
    k_star = float(st.session_state.get("sentinela_kstar") or 0.0)
    indice_risco = st.session_state.get("indice_risco")
    regime_txt = st.session_state.get("regime_ambiente") or st.session_state.get("diagnostico_regime")
    if not _ambiente_ruim(k_star=k_star, indice_risco=indice_risco, regime_txt=str(regime_txt) if regime_txt else None):
        # Se não está ruim, ainda podemos permitir em PLANA/DESCENDO, mas conservador: exige RUIM
        return {"eligivel": False, "motivo": "ambiente_nao_ruim"}

    # Snapshot P0 do k de referência
    snap = None
    try:
        snap = snaps_map.get(int(k_ref))
    except Exception:
        snap = None
    if not isinstance(snap, dict):
        return {"eligivel": False, "motivo": f"snapshot_ausente_k_{k_ref}"}

    p1 = _p1__build_ub_from_snapshot(snap, umin=int(st.session_state.get("universo_min", 1) or 1), umax=int(st.session_state.get("universo_max", 60) or 60))
    ub = p1.get("UB") or []
    if len(ub) < 10:
        return {"eligivel": False, "motivo": "ub_insuficiente"}

    return {
        "eligivel": True,
        "motivo": "P1_DEFENSIVO_PLANO_RUIM",
        "estado_global": estado_global,
        "ub": ub,
        "adds_B": p1.get("adds_B") or [],
        "cap_status": cap_status,
        "persistencia": persist,
    }

def p2_executar(snapshot, _df_full_safe):
    k = snapshot.get("k")
    universo = list(snapshot.get("universo_pacote", []))
    score_rigidez = float(snapshot.get("score_rigidez", 0.0) or 0.0)
    n = int(st.session_state.get("n_alvo") or 6)
    idxs = list(_df_full_safe.index)
    pos = _p2_resolver_posicao_k(_df_full_safe, k)
    alvos = []
    for off in (1, 2):
        if pos + off < len(idxs):
            alvos.append(_p2_series_from_df(_df_full_safe, idxs[pos + off], n=n))
    fora_total = 0
    fora_longe = 0
    for alvo in alvos:
        fora = _p2_avaliar_fora(universo, alvo)
        fora_total += len(fora)
        fora_longe += len(fora)
    cap = p2_calcular_cap_dinamico(len(universo), fora_longe, fora_total, score_rigidez)

    # Governo estrutural (Parabólica): P2 só pode "acordar" quando houver SUBIDA com persistência (multi-escala).
    # Critério objetivo: parabola_multiescala_vetorial (sem chute).
    estado_parabola = st.session_state.get("parabola_estado_global") or st.session_state.get("parabola_estado")
    p2_permitido = st.session_state.get("parabola_p2_permitido")

    if p2_permitido is None:
        # se a Parabólica ainda não foi visitada, calculamos aqui (usando apenas histórico + snapshots)
        snaps_map = st.session_state.get("snapshot_p0_canonic") or st.session_state.get("snapshots_p0_map", {}) or {}
        st.session_state["snapshots_p0_map"] = snaps_map
        if not snaps_map:
            # fallback: tenta converter lista->map
            snaps_list = st.session_state.get("snapshots_p0", [])
            if isinstance(snaps_list, list) and snaps_list:
                try:
                    snaps_map = {int(s.get("k")): s for s in snaps_list if isinstance(s, dict) and s.get("k") is not None}
                except Exception:
                    snaps_map = {}
        if snaps_map:
            gov = parabola_multiescala_vetorial(_df_full_safe, snaps_map, n=n)
            if gov:
                estado_parabola = gov.get("estado_global")
                p2_permitido = bool(gov.get("p2_permitido"))
                st.session_state["parabola_estado_global"] = estado_parabola
                st.session_state["parabola_p2_permitido"] = p2_permitido

    motivo_p2 = None
    if not p2_permitido:
        motivo_p2 = f"P2 vetado pela Parabólica (estado_global={estado_parabola}). Mantido cap=0 (P2 dormindo)."
        cap = 0
    df_hist = _df_full_safe.iloc[:pos+1]
    adds_h1, U_h1 = p2_h1_freq(df_hist, universo, cap, n=n)
    adds_h2, U_h2 = p2_h2_dist(df_hist, universo, cap, n=n)
    return {
        "cap": cap,
        "parabola_estado": estado_parabola,
        "motivo": motivo_p2,
        "H1": {"adds": adds_h1, "universo_len": len(U_h1)},
        "H2": {"adds": adds_h2, "universo_len": len(U_h2)},
    }


def _pc_fmt_num(x, decimals: int = 4, nd: str = "N/D") -> str:
    """Formata número para UX (evita mostrar nan/inf cru)."""
    try:
        if x is None:
            return nd
        if isinstance(x, (int, float)):
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return nd
            return f"{xf:.{decimals}f}"
        # tenta converter
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return nd
        return f"{xf:.{decimals}f}"
    except Exception:
        return nd




# ============================================================
# PredictCars V15.7 MAX — Âncora Estável
# (sem governança / sem fases extras / sem 'próximo passo')
# ============================================================

st.sidebar.info(f"Rodando arquivo (canônico): {BUILD_CANONICAL_FILE}")
# ============================================================
# Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO
# Núcleo + Coberturas + Interseção Estatística
# Pipeline V14-FLEX ULTRA + Replay LIGHT/ULTRA + TURBO++ HÍBRIDO
# + TURBO++ ULTRA + Painel de Ruído Condicional
# + Painel de Divergência S6 vs MC + Monitor de Risco (k & k*)
# + Testes de Confiabilidade REAL + Modo 6 Acertos V15.7 MAX
# + Relatório Final COMPLETO V15.7 MAX
# Arquivo oficial: app_v15_7_MAX.py
# ============================================================
import math
import itertools
import textwrap
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DEBUG TEMPORÁRIO — PROVA DE EXECUÇÃO DO ARQUIVO
st.sidebar.caption("🧪 DEBUG: arquivo carregado")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ============================================================
# MÓDULO 1 — GOVERNANÇA & MIRROR (OBSERVACIONAL)
# Camada de VISIBILIDADE (read-only)
# - NÃO executa motores
# - NÃO altera comportamento
# - NÃO bloqueia operações
# - Falha silenciosa (nunca derruba o app)
# ============================================================

def _m1_collect_mirror_snapshot() -> Dict[str, Any]:
    """Coleta um snapshot read-only do estado atual (session_state + sinais básicos).
    Regra: nunca levanta exceção para o app; devolve N/D quando não existir.
    """

    ss = st.session_state

    # V16 — guarda anti-NameError (nocivos_set)
    try:
        nocivos_set = set(ss.get('nocivos_set', []) or [])
    except Exception:
        nocivos_set = set()

    # ------------------------------------------------------------------
    # UNIVERSO (auto-derivação) — especialmente para "Carregar Histórico (Colar)"
    # ------------------------------------------------------------------
    if (("universo_min" not in ss) or ("universo_max" not in ss)) and ss.get("historico_df") is not None:
        try:
            dfu = ss.get("historico_df")
            cols = [c for c in ["p1","p2","p3","p4","p5","p6"] if c in getattr(dfu, "columns", [])]
            if cols:
                ser = pd.to_numeric(dfu[cols].stack(), errors="coerce")
                vmin = ser.min()
                vmax = ser.max()
                if pd.notna(vmin) and pd.notna(vmax):
                    ss["universo_min"] = int(vmin)
                    ss["universo_max"] = int(vmax)
        except Exception:
            pass

    def g(key: str, default: Any = "N/D") -> Any:
        try:
            return ss.get(key, default)
        except Exception:
            return default

    # Base
    historico_df = g("historico_df", None)
    historico_ok = historico_df is not None

    # Sinais universais (quando existirem)
    n_alvo = g("n_alvo", "N/D")
    universo_min = g("universo_min", "N/D")
    universo_max = g("universo_max", "N/D")

    # Pipeline / Diagnóstico (nomes variam no arquivo; usar fallback N/D)
    pipeline_ok = bool(g("pipeline_flex_ultra_concluido", False) or g("pipeline_executado", False))
    regime = g("regime_identificado", g("regime", "N/D"))
    energia = g("energia_media", g("energia_media_estrada", "N/D"))
    volatilidade = g("volatilidade_media", "N/D")
    clusters = g("clusters_formados", "N/D")

    # Monitor de risco
    k_star = g("k_star", g("k*", "N/D"))
    nr_percent = g("nr_percent", g("nr%", "N/D"))
    divergencia = g("divergencia_s6_mc", g("divergencia", "N/D"))
    indice_risco = g("indice_risco", "N/D")
    classe_risco = g("classe_risco", "N/D")

    # Execução / TURBO / Modo 6
    turbo_tentado = bool(g("turbo_ultra_executado", False) or g("turbo_executado", False))
    turbo_bloqueado = bool(g("turbo_bloqueado", False))
    turbo_motivo = g("turbo_motivo_bloqueio", g("motivo_bloqueio", "N/D"))
    modo6_executado = bool(g("modo_6_ativo", False) or g("modo6_executado", False) or g("modo_6_executado", False))
    # Listas geradas (numérico quando possível)
    _lg = g("listas_geradas", None)
    if _lg is None:
        _lg = g("listas_finais", None)
    if _lg is None:
        _lg = g("pacote_atual", None)

    if isinstance(_lg, list):
        listas_geradas = len(_lg)
    elif isinstance(_lg, int):
        listas_geradas = _lg
    else:
        listas_geradas = "<não definido>"
    volumes_usados = g("volumes_usados", "N/D")
    estado_alvo = g("estado_alvo", "N/D")
    eco_status = g("eco_status", g("eco", "N/D"))
    dmo_status = g("estado_dmo", "N/D")

    # Rastro de navegação (quando existir)
    painel_atual = g("NAV_V157_CANONICA", "N/D")

    # Keys (para auditoria leve)
    try:
        keys = sorted([str(k) for k in ss.keys()])
    except Exception:
        keys = []

    return {
        "historico_ok": historico_ok,
        "historico_df": "definido" if historico_ok else "<não definido>",
        "n_alvo": n_alvo,
        "universo_min": universo_min,
        "universo_max": universo_max,
        "nocivos_consistentes": sorted(list(nocivos_set))[:30],
        "nocivos_qtd": int(len(nocivos_set)),
        "pipeline_ok": pipeline_ok,
        "regime": regime,
        "energia_media": energia,
        "volatilidade_media": volatilidade,
        "clusters": clusters,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia_s6_mc": divergencia,
        "indice_risco": indice_risco,
        "classe_risco": classe_risco,
        "turbo_tentado": turbo_tentado,
        "turbo_bloqueado": turbo_bloqueado,
        "turbo_motivo": turbo_motivo,
        "modo6_executado": modo6_executado,
        "listas_geradas": "definidas" if (listas_geradas not in (None, "N/D", "<não definido>")) else "<não definido>",
        "volumes_usados": volumes_usados,
        "estado_alvo": estado_alvo,
        "eco_status": eco_status,
        "dmo_status": dmo_status,
        "painel_atual": painel_atual,
        "keys": keys,
    }


# ============================================================
# V16 — UNIVERSO (mín/max) — REGISTRO CANÔNICO
# - Objetivo: garantir que o snapshot mostre 1–50 / 1–60 etc
# - Regra: NÃO decide nada; só registra leitura.
# - Compatível com Upload e Colar.
# ============================================================

def v16_detectar_universo_do_historico(df, n_alvo=6):
    try:
        if df is None or len(df) == 0:
            return (None, None)
        cols = []
        for i in range(1, int(n_alvo) + 1):
            c = f"p{i}"
            if c in df.columns:
                cols.append(c)
        if not cols:
            cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p") and c[1:].isdigit()]
            cols = sorted(cols, key=lambda x: int(x[1:]))[:int(n_alvo)]
        if not cols:
            return (None, None)

        vals = df[cols].values.ravel()
        clean = []
        for v in vals:
            if v is None:
                continue
            try:
                if isinstance(v, float) and (v != v):
                    continue
                clean.append(int(v))
            except Exception:
                continue

        if not clean:
            return (None, None)

        return (min(clean), max(clean))
    except Exception:
        return (None, None)


def v16_registrar_universo_session_state(df, n_alvo=6):
    try:
        umin, umax = v16_detectar_universo_do_historico(df, n_alvo=n_alvo)
        if umin is not None and umax is not None:
            st.session_state["universo_min"] = int(umin)
            st.session_state["universo_max"] = int(umax)
            st.session_state["universo_str"] = f"{int(umin)}–{int(umax)}"
        else:
            st.session_state.setdefault("universo_min", None)
            st.session_state.setdefault("universo_max", None)
            st.session_state.setdefault("universo_str", "N/D")
    except Exception:
        st.session_state.setdefault("universo_str", "N/D")


# ============================================================
# CAP INVISÍVEL (V0) — REGISTRO AUTOMÁTICO DO SNAPSHOT P0
# ------------------------------------------------------------
# Objetivo (pré-C4):
# - Eliminar o clique manual de "Registrar pacote" no Replay Progressivo
# - Sempre que o operador roda o 🎯 Modo 6, o pacote da janela ativa vira um Snapshot P0 canônico
# - NÃO decide ataque, NÃO muda geração, NÃO altera Camada 4
# - Preparação direta para o CAP Invisível completo (auto-preencher ks)
# ============================================================

def pc_snapshot_p0_autoregistrar(pacote_atual, k_reg, universo_min=1, universo_max=60):
    """Registra automaticamente um snapshot P0 canônico (pré-C4) para a janela k_reg.

    Observacional • auditável • não altera listas • não altera Camada 4.
    """
    try:
        if pacote_atual is None:
            return False

        # Normaliza pacote -> lista de listas[int] (somente listas com 6 passageiros)
        pacote_norm = []
        for lst in (pacote_atual or []):
            try:
                li = [int(x) for x in lst]
                if len(li) == 6:
                    pacote_norm.append(li)
            except Exception:
                continue

        if len(pacote_norm) == 0:
            return False

        pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
        snapshot_p0_reg = st.session_state.get("snapshot_p0_canonic", {})

        from datetime import datetime
        import hashlib, json

        # V8 (borda) — reaproveita/recupera (se não existir ou estiver inválido, reclassifica)
        try:
            v8_snap = st.session_state.get("v8_borda_qualificada") or {}
            if not isinstance(v8_snap, dict) or v8_snap.get("meta", {}).get("status") not in ("ok", "presenca_vazia"):
                base_n = int(min(10, len(pacote_norm)))
                v8_snap = v8_classificar_borda_qualificada(
                    listas=[list(map(int, lst)) for lst in pacote_norm],
                    base_n=base_n,
                    core_presenca_min=0.60,
                    quase_delta=0.12,
                    max_borda_interna=6,
                    universo_min=int(universo_min),
                    universo_max=int(universo_max),
                    rigidez_info=st.session_state.get("v16_rigidez_info"),
                )
        except Exception:
            v8_snap = {"core": [], "quase_core": [], "borda_interna": [], "borda_externa": [], "meta": {"status": "snap_falhou"}}

        # Universo do pacote
        try:
            universo_pacote = sorted({int(x) for lst in pacote_norm for x in lst})
        except Exception:
            universo_pacote = []

        # Replay (mapa por janela)

        # --- v16h57K: garantir snapshot de métricas do Mirror no momento do registro ---
        # A calib_leve depende de mirror_rank_meta; se o usuário não abriu o painel Mirror,
        # nós forçamos um refresh silencioso aqui (read-only) para não registrar pacotes "cegos".
        mirror_meta = None
        try:
            mirror_meta = st.session_state.get("mirror_rank_meta")
            if not isinstance(mirror_meta, dict):
                mirror_meta = None
            if mirror_meta is None:
                try:
                    _m1_obter_ranking_structural_df()  # preenche mirror_rank_meta (read-only)
                except Exception:
                    pass
                mirror_meta = st.session_state.get("mirror_rank_meta")
                if not isinstance(mirror_meta, dict):
                    mirror_meta = None
        except Exception:
            mirror_meta = None

        # Se calib_leve ainda não existe (ou veio incompleto), cria um resumo mínimo a partir do mirror_meta
        try:
            calib_snap = st.session_state.get("v16_calib_leve_last_summary")
            if not isinstance(calib_snap, dict):
                calib_snap = {}
            # preenche insumos e I/I2 se estiverem faltando
            if mirror_meta is not None:
                calib_snap.setdefault("C_top", float(mirror_meta.get("C_top", 0.0) or 0.0))
                calib_snap.setdefault("Slope", float(mirror_meta.get("Slope", 0.0) or 0.0))
                calib_snap.setdefault("Stab", float(mirror_meta.get("Stab", mirror_meta.get("StabTop6", 0.0) or 0.0) or 0.0))
                calib_snap.setdefault("Gap", float(mirror_meta.get("Gap", mirror_meta.get("gap", 0.0) or 0.0) or 0.0))
                # I canônico (se não existir)
                if "I" not in calib_snap and "I_mean" not in calib_snap:
                    C_top = float(calib_snap.get("C_top", 0.0))
                    Slope = float(calib_snap.get("Slope", 0.0))
                    Stab = float(calib_snap.get("Stab", 0.0))
                    Gap  = float(calib_snap.get("Gap", 0.0))
                    I1 = max(0.0, min(1.0, (C_top - 1.20) / 1.20))
                    I2_ = max(0.0, min(1.0, (Slope - 0.0020) / 0.0030))
                    I3 = max(0.0, min(1.0, (Stab - 0.55) / 0.35))
                    I4 = max(0.0, min(1.0, (Gap - 0.0015) / 0.0030))
                    I = float((I1 + I2_ + I3 + I4) / 4.0)
                    calib_snap["I"] = float(I)
                    calib_snap["I_mean"] = float(I)
                    calib_snap["I_max"] = float(I)
                # I2 contraste (se não existir)
                if "I2" not in calib_snap and "I2_mean" not in calib_snap:
                    C_top = float(calib_snap.get("C_top", 0.0))
                    Slope = float(calib_snap.get("Slope", 0.0))
                    Stab = float(calib_snap.get("Stab", 0.0))
                    Gap  = float(calib_snap.get("Gap", 0.0))
                    J1 = max(0.0, min(1.0, (C_top - 1.00) / 0.80))
                    J2 = max(0.0, min(1.0, (Slope) / 0.0040))
                    J3 = max(0.0, min(1.0, (Stab - 0.50) / 0.25))
                    J4 = max(0.0, min(1.0, (Gap) / 0.0030))
                    I2_contraste = float((J1 + J2 + J3 + J4) / 4.0)
                    calib_snap["I2"] = float(I2_contraste)
                    calib_snap["I2_mean"] = float(I2_contraste)
                    calib_snap["I2_max"] = float(I2_contraste)
                calib_snap.setdefault("active", True)
                # guarda snapshot do mirror para auditoria robusta
                calib_snap.setdefault("insumos", {})
                calib_snap["insumos"].setdefault("mirror", {})
                calib_snap["insumos"]["mirror"].setdefault("metrics", mirror_meta)
            st.session_state["v16_calib_leve_last_summary"] = calib_snap.copy()
        except Exception:
            pass

        # --- V16: calibração leve (pré-C4) aplicada NO REGISTRO do pacote (auditável) ---
        calib_leve = st.session_state.get("v16_calib_leve_last_summary") or {}
        if not isinstance(calib_leve, dict):
            calib_leve = {}

        try:
            _top10_tmp = pacote_norm[:10]
            _freq = {}
            for _lst in _top10_tmp:
                for _x in _lst:
                    _freq[_x] = _freq.get(_x, 0) + 1
            _suf_freq = [x for x, c in _freq.items() if (c / max(1, len(_top10_tmp))) >= 0.40]
        except Exception:
            _suf_freq = []
        try:
            _nocivos = list(st.session_state.get("pc_nocivos_consistentes", []))
        except Exception:
            _nocivos = []
        _suf = sorted(set(_suf_freq) | set(_nocivos))

        _resp_apply = pc_v16_aplicar_resp_no_pacote(
            pacote_atual=pacote_norm,
            k_reg=int(k_reg),
            universo_min=int(universo_min),
            universo_max=int(universo_max),
            calib_leve_summary=calib_leve,
            top10_override=pacote_norm[:10],
            memoria_sufocadores=_suf,
            cap_pct=None,
            core_min=0.40,
        )

        pacote_baseline = _resp_apply.get("pacote_baseline", [])
        pacote_store = _resp_apply.get("pacote_store", pacote_baseline)
        resp_info = _resp_apply.get("resp_info", {"aplicado": False, "motivo": "nao_aplicado"})
        calib_active = bool(_resp_apply.get("calib_active", False))
        calib_applied = bool(_resp_apply.get("calib_applied", False))
        I_mean = float(_resp_apply.get("I_mean", 0.0))
        I_max = float(_resp_apply.get("I_max", I_mean))
        I2_mean = float(_resp_apply.get("I2_mean", 0.0))
        I2_max = float(_resp_apply.get("I2_max", I2_mean))
        THR_BASE = float(_resp_apply.get("thr_base", 0.25))

        calib_leve_store = dict(calib_leve)
        calib_leve_store.update({
            "active": calib_active,
            "applied": calib_applied,
            "aplicada_no_pacote": calib_applied,
            "thr_base": THR_BASE,
            "I_mean": float(I_mean),
            "I_max": float(I_max),
            "I2_mean": float(I2_mean),
            "I2_max": float(I2_max),
            "resp_info": resp_info,
            "diff_conteudo_qtd": int(_resp_apply.get("diff_conteudo_qtd", 0)),
            "diff_posicao_qtd": int(_resp_apply.get("diff_posicao_qtd", 0)),
            "diff_dedup_qtd": int(_resp_apply.get("diff_dedup_qtd", 0)),
            "reason": "pacote_modificado" if calib_applied else ("I2<thr_base" if calib_active else "I2=0"),
        })

        pacotes_reg[int(k_reg)] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "qtd": int(len(pacote_store)),
            "listas": [list(map(int, lst)) for lst in pacote_store],
            "snap_v9": {
                "core": list(map(int, (v8_snap.get("core") or []))),
                "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                "universo_pacote": list(map(int, universo_pacote)),
                "meta": v8_snap.get("meta") or {},
            },
        }

        # Snapshot P0 (canônico)
        try:
            freq_passageiros = {}
            for lst in pacote_store:
                for x in lst:
                    xi = int(x)
                    freq_passageiros[xi] = freq_passageiros.get(xi, 0) + 1

            sig_raw = json.dumps([list(map(int, lst)) for lst in pacote_store], ensure_ascii=False, sort_keys=True)
            sig = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()[:16]
        except Exception:
            freq_passageiros = {}
            sig = "N/D"

        snapshot_p0_reg[int(k_reg)] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "k": int(k_reg),
            "qtd_listas": int(len(pacote_store)),
            "universo_pacote_len": int(len(universo_pacote)),
            "listas": [list(map(int, lst)) for lst in pacote_store],
            "freq_passageiros": {str(int(k)): int(v) for k, v in sorted(freq_passageiros.items(), key=lambda kv: (-kv[1], kv[0]))},
            "snap_v8": {
                "core": list(map(int, (v8_snap.get("core") or []))),
                "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                "meta": v8_snap.get("meta") or {},
            },
            "assinatura": sig,
            "nota": "Snapshot P0 canônico — leitura apenas (pré-C4). Não altera Camada 4.",
        }

        st.session_state["snapshot_p0_canonic"] = snapshot_p0_reg
        st.session_state["replay_progressivo_pacotes"] = pacotes_reg

        # Atualiza Memória Estrutural automaticamente ao registrar snapshot
        try:
            _df_full_me = st.session_state.get("_df_full_safe") or st.session_state.get("historico_df_full") or st.session_state.get("historico_df")
            v16_me_update_auto(_df_full_safe=_df_full_me, snapshots_map=st.session_state.get("snapshot_p0_canonic") or {})
        except Exception:
            pass

        return True
    except Exception:
        return False
def _pc_replay_limpar_chaves_dependentes_silent():
    """Limpa chaves dependentes do histórico/pipeline/pacote (versão silenciosa)."""
    chaves = [
        "pipeline_col_pass",
        "pipeline_clusters",
        "pipeline_centroides",
        "pipeline_matriz_norm",
        "pipeline_estrada",
        "pipeline_flex_ultra_concluido",
        "pipeline_executado",
        "m1_selo_pipeline_ok",
        "m1_ts_pipeline_ok",
        "sentinela_kstar",
        "k_star",
        "k*",
        "nr_percent",
        "nr_percent_v16",
        "div_s6_mc",
        "divergencia_s6_mc",
        "indice_risco",
        "classe_risco",
        "ultima_previsao",
        "pacote_listas_atual",
        "pacote_listas_origem",
        "listas_geradas",
        "listas_finais",
        "modo6_executado",
        "modo_6_executado",
        "modo_6_ativo",
        "turbo_executado",
        "turbo_ultra_executado",
        "turbo_ultra_rodou",
        "turbo_ultra_rodou_ok",
        "turbo_ultra_rodou_flag",
        "turbo_rodou",
        "turbo_bloqueado",
        "turbo_motivo_bloqueio",
        "motor_turbo_executado",
        "listas_intercept_orbita",
        "modo6_listas_totais",
        "modo6_listas_top10",
        "modo6_listas",
    ]
    for k in chaves:
        try:
            if k in st.session_state:
                del st.session_state[k]
        except Exception:
            pass


def pc_exec_pipeline_flex_ultra_silent(df: pd.DataFrame) -> bool:
    """Executa o Pipeline V14-FLEX ULTRA (silencioso) e grava chaves canônicas em session_state."""
    try:
        if df is None or df.empty:
            return False

        col_pass = [c for c in df.columns if str(c).startswith("p")]
        if not col_pass:
            return False

        matriz = df[col_pass].astype(float).values

        minimo = float(np.nanmin(matriz))
        maximo = float(np.nanmax(matriz))
        amplitude = (maximo - minimo) if maximo != minimo else 1.0
        matriz_norm = (matriz - minimo) / amplitude

        medias = np.mean(matriz_norm, axis=1)
        desvios = np.std(matriz_norm, axis=1)

        media_geral = float(np.mean(medias))
        desvio_geral = float(np.mean(desvios))

        if media_geral < 0.35:
            estrada = "🟦 Estrada Fria (Baixa energia)"
        elif media_geral < 0.65:
            estrada = "🟩 Estrada Neutra / Estável"
        else:
            estrada = "🟥 Estrada Quente (Alta volatilidade)"

        try:
            from sklearn.cluster import KMeans
            n_clusters = 3
            modelo = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            clusters = modelo.fit_predict(matriz_norm)
            centroides = modelo.cluster_centers_
        except Exception:
            clusters = np.zeros(len(matriz_norm))
            centroides = np.zeros((1, matriz_norm.shape[1]))

        st.session_state["pipeline_col_pass"] = col_pass
        st.session_state["pipeline_clusters"] = clusters
        st.session_state["pipeline_centroides"] = centroides
        st.session_state["pipeline_matriz_norm"] = matriz_norm
        # V16h35 — Persistência canônica (não depende de SAFE)
        st.session_state["pipeline_matriz_norm_base"] = matriz_norm

        st.session_state["pipeline_estrada"] = estrada

        st.session_state["regime_identificado"] = estrada
        st.session_state["energia_media"] = float(media_geral)
        st.session_state["energia_media_estrada"] = float(media_geral)
        st.session_state["volatilidade_media"] = float(desvio_geral)
        st.session_state["clusters_formados"] = int(max(clusters) + 1) if len(np.atleast_1d(clusters)) else 0

        st.session_state["pipeline_flex_ultra_concluido"] = True
        st.session_state["pipeline_executado"] = True
        st.session_state["m1_selo_pipeline_ok"] = True
        try:
            from datetime import datetime
            st.session_state["m1_ts_pipeline_ok"] = datetime.now().isoformat(timespec="seconds")
        except Exception:
            pass
        return True
    except Exception:
        return False



# ============================================================
# Semi-automação segura (por k) — helpers silenciosos
# ============================================================

def pc_sentinelas_kstar_silent(df: pd.DataFrame) -> float | None:
    """Calcula k* (sentinela) de forma silenciosa e grava em session_state.
    Regra: mesmo espírito do painel Sentinelas, sem UI.
    """
    try:
        if df is None or df.empty or "k" not in df.columns:
            return None

        k_vals = df["k"].astype(int).values

        def _media_movel(vetor, janela):
            if len(vetor) < janela:
                return float(np.mean(vetor))
            return float(np.mean(vetor[-janela:]))

        janela_curta = 12
        janela_media = 30
        janela_longa = 60

        k_curto = _media_movel(k_vals, janela_curta)
        k_medio = _media_movel(k_vals, janela_media)
        k_longo = _media_movel(k_vals, janela_longa)

        k_star = (0.50 * k_curto + 0.35 * k_medio + 0.22 * k_longo)

        st.session_state["sentinela_kstar"] = float(k_star)
        # alias canônico
        try:
            st.session_state["k_star"] = float(k_star)
        except Exception:
            pass
        return float(k_star)
    except Exception:
        return None


def pc_monitor_risco_silent(df: pd.DataFrame) -> dict:
    """Atualiza o Monitor de Risco (k & k*) de forma silenciosa.
    Observacional (pré-C4): não altera listas, apenas grava chaves de contexto.
    """
    try:
        metricas = calcular_metricas_basicas_historico(df)
        qtd_series = metricas.get("qtd_series", 0)
        min_k = metricas.get("min_k")
        max_k = metricas.get("max_k")
        media_k = metricas.get("media_k")

        k_star = st.session_state.get("sentinela_kstar")
        nr_percent = st.session_state.get("nr_percent")
        divergencia = st.session_state.get("div_s6_mc")

        # Garantias (defaults neutros, como no painel)
        if k_star is None:
            k_star = 0.25
        if nr_percent is None:
            nr_percent = 35.0
        if divergencia is None:
            divergencia = 4.0

        kstar_norm = min(1.0, float(k_star) / 0.50)
        nr_norm = min(1.0, float(nr_percent) / 70.0)
        div_norm = min(1.0, float(divergencia) / 8.0)

        indice_risco = float(0.40 * kstar_norm + 0.35 * nr_norm + 0.25 * div_norm)

        if indice_risco < 0.30:
            classe_risco = "🟢 Risco Baixo (Janela Favorável)"
        elif indice_risco < 0.55:
            classe_risco = "🟡 Risco Moderado"
        elif indice_risco < 0.80:
            classe_risco = "🟠 Risco Elevado"
        else:
            classe_risco = "🔴 Risco Crítico"

        # grava chaves canônicas
        st.session_state["k_star"] = float(k_star)
        st.session_state["nr_percent"] = float(nr_percent)
        st.session_state["div_s6_mc"] = float(divergencia)
        st.session_state["divergencia_s6_mc"] = float(divergencia)

        st.session_state["indice_risco"] = float(indice_risco)
        st.session_state["classe_risco"] = classe_risco

        st.session_state["m1_selo_risco_ok"] = True
        st.session_state["m1_ts_risco_ok"] = __import__("time").time()

        return {
            "qtd_series": qtd_series,
            "min_k": min_k,
            "max_k": max_k,
            "media_k": media_k,
            "k_star": float(k_star),
            "nr_percent": float(nr_percent),
            "divergencia": float(divergencia),
            "indice_risco": float(indice_risco),
            "classe_risco": classe_risco,
            "status": "ok",
        }
    except Exception as e:
        return {"status": "erro", "erro": str(e)}



def pc_replay_registrar_pacote_silent(*, k_reg: int, pacote_atual: list, universo_min: int, universo_max: int) -> bool:
    """Registra pacote e Snapshot P0 canônico para a janela k_reg (silencioso).
    Caminho canônico alinhado ao registro manual, com baseline interno real e calibração leve persistida.
    """
    try:
        if not isinstance(pacote_atual, list) or len(pacote_atual) == 0:
            return False

        # containers
        if "replay_progressivo_pacotes" not in st.session_state or not isinstance(st.session_state.get("replay_progressivo_pacotes"), dict):
            st.session_state["replay_progressivo_pacotes"] = {}
        if "snapshot_p0_canonic" not in st.session_state or not isinstance(st.session_state.get("snapshot_p0_canonic"), dict):
            st.session_state["snapshot_p0_canonic"] = {}

        pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
        snapshot_p0_reg = st.session_state.get("snapshot_p0_canonic", {})

        from datetime import datetime
        import hashlib, json

        # baseline interno real + aplicação RESP canônica
        calib_leve = st.session_state.get("v16_calib_leve_last_summary")
        if isinstance(calib_leve, dict):
            calib_leve_store = dict(calib_leve)
        else:
            calib_leve_store = {}

        _resp_apply = pc_v16_aplicar_resp_no_pacote(
            pacote_atual=pacote_atual,
            k_reg=int(k_reg),
            universo_min=int(universo_min),
            universo_max=int(universo_max),
            calib_leve_summary=calib_leve_store,
            top10_override=None,
            memoria_sufocadores=None,
            cap_pct=0.65,
            core_min=0.40,
        )

        pacote_baseline = _resp_apply.get("pacote_baseline", [])
        if not pacote_baseline:
            return False
        pacote_store = _resp_apply.get("pacote_store", pacote_baseline)
        resp_info = _resp_apply.get("resp_info", {"aplicado": False, "motivo": "nao_executado"})
        calib_active = bool(_resp_apply.get("calib_active", False))
        calib_applied = bool(_resp_apply.get("calib_applied", False))

        calib_leve_store.update({
            "active": bool(calib_active),
            "applied": bool(calib_applied),
            "aplicada_no_pacote": bool(calib_applied),
            "resp_info": resp_info,
            "diff_conteudo_qtd": int(_resp_apply.get("diff_conteudo_qtd", 0)),
            "diff_posicao_qtd": int(_resp_apply.get("diff_posicao_qtd", 0)),
            "diff_dedup_qtd": int(_resp_apply.get("diff_dedup_qtd", 0)),
            "reason": "pacote_modificado" if calib_applied else "sem_aplicacao_real",
        })

        # V8 (borda) — reaproveita/recupera
        try:
            v8_snap = st.session_state.get("v8_borda_qualificada") or {}
            if not isinstance(v8_snap, dict) or v8_snap.get("meta", {}).get("status") not in ("ok", "presenca_vazia"):
                v8_snap = v8_classificar_borda_qualificada(
                    listas=[list(map(int, lst)) for lst in pacote_store],
                    base_n=10,
                    core_presenca_min=0.60,
                    quase_delta=0.12,
                    max_borda_interna=6,
                    universo_min=universo_min,
                    universo_max=universo_max,
                    rigidez_info=st.session_state.get("v16_rigidez_info"),
                )
        except Exception:
            v8_snap = {"core": [], "quase_core": [], "borda_interna": [], "borda_externa": [], "meta": {"status": "snap_falhou"}}

        # Universo do pacote
        try:
            universo_pacote = sorted({int(x) for lst in pacote_store for x in lst})
        except Exception:
            universo_pacote = []

        pacotes_reg[int(k_reg)] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "qtd": int(len(pacote_store)),
            "listas": [list(map(int, lst)) for lst in pacote_store],
            "calib_leve": calib_leve_store,
            "listas_baseline": [list(map(int, lst)) for lst in pacote_baseline] if calib_applied else None,
            "snap_v9": {
                "core": list(map(int, (v8_snap.get("core") or []))),
                "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                "universo_pacote": list(map(int, universo_pacote)),
                "meta": v8_snap.get("meta") or {},
            },
        }

        # Snapshot P0 (canônico)
        try:
            freq_passageiros = {}
            for lst in pacote_store:
                for x in lst:
                    xi = int(x)
                    freq_passageiros[xi] = freq_passageiros.get(xi, 0) + 1

            sig_raw = json.dumps([list(map(int, lst)) for lst in pacote_store], ensure_ascii=False, sort_keys=True)
            sig = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()[:16]
        except Exception:
            freq_passageiros = {}
            sig = "N/D"

        snapshot_p0_reg[int(k_reg)] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "k": int(k_reg),
            "qtd_listas": int(len(pacote_store)),
            "listas": [list(map(int, lst)) for lst in pacote_store],
            "calib_leve": calib_leve_store,
            "listas_baseline": [list(map(int, lst)) for lst in pacote_baseline] if calib_applied else None,
            "freq_passageiros": {str(int(k)): int(v) for k, v in sorted(freq_passageiros.items(), key=lambda kv: (-kv[1], kv[0]))},
            "snap_v8": {
                "core": list(map(int, (v8_snap.get("core") or []))),
                "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                "meta": v8_snap.get("meta") or {},
            },
            "assinatura": sig,
            "nota": "Snapshot P0 canônico — leitura apenas (pré-C4). Não altera Camada 4.",
        }

        st.session_state["snapshot_p0_canonic"] = snapshot_p0_reg
        st.session_state["replay_progressivo_pacotes"] = pacotes_reg

        # Atualiza Memória Estrutural automaticamente ao registrar snapshot
        try:
            _df_full_me = st.session_state.get("_df_full_safe") or st.session_state.get("historico_df_full") or st.session_state.get("historico_df")
            v16_me_update_auto(_df_full_safe=_df_full_me, snapshots_map=st.session_state.get("snapshot_p0_canonic") or {})
        except Exception:
            pass

        return True
    except Exception:
        return False

def pc_semi_auto_processar_um_k(*, _df_full_safe: pd.DataFrame, k_exec: int) -> dict:
    """Executa a sequência mínima segura por k (sem decisão automática):
    recorta janela → (sentinela/monitor) → pipeline → modo6 → registra snapshot.
    """
    try:
        if _df_full_safe is None or _df_full_safe.empty:
            return {"ok": False, "erro": "df_full_vazio"}

        k_exec = int(k_exec)
        k_exec = max(10, min(k_exec, int(len(_df_full_safe))))

        # recorte + limpeza
        df_recorte = _df_full_safe.head(k_exec).copy()
        st.session_state["historico_df"] = df_recorte
        st.session_state["_df_full_safe"] = _df_full_safe  # referência full para governança (P1) em modo SAFE
        st.session_state["replay_janela_k_active"] = int(k_exec)

        try:
            _pc_replay_limpar_chaves_dependentes_silent()
        except Exception:
            pass

        # universo canônico
        try:
            uinfo = v16_detectar_universo_do_historico(df_recorte)
            v16_registrar_universo_session_state(uinfo)
        except Exception:
            pass

        # sentinela + monitor (observacionais)
        try:
            pc_sentinelas_kstar_silent(df_recorte)
        except Exception:
            pass
        try:
            pc_monitor_risco_silent(df_recorte)
        except Exception:
            pass

        # pipeline
        ok_pipe = pc_exec_pipeline_flex_ultra_silent(df_recorte)
        if not ok_pipe:
            return {"ok": False, "k": int(k_exec), "erro": "pipeline_falhou"}

        # modo 6
        pacote_calib, calib_meta = pc_modo6_gerar_pacote_top10_silent(df_recorte, calib_override=None)
        pacote_base, _meta_base = pc_modo6_gerar_pacote_top10_silent(df_recorte, calib_override=False)
        st.session_state['pacote_listas_atual'] = pacote_calib
        st.session_state['pacote_listas_baseline'] = pacote_base
        st.session_state['v16_calib_leve_last_summary'] = calib_meta.copy()
        pacote = pacote_calib
        if not pacote:
            # Bootstrap de emergência (SAFE): se o gerador canônico falhar por estado faltante,
            # geramos um pacote mínimo determinístico baseado no universo observado.
            try:
                col_pass_boot = [c for c in df_recorte.columns if str(c).startswith("p")]
                vals_boot = []
                if col_pass_boot:
                    for _, row in df_recorte.iterrows():
                        for c in col_pass_boot:
                            try:
                                if pd.notna(row[c]):
                                    vals_boot.append(int(row[c]))
                            except Exception:
                                pass
                universo_boot = sorted({v for v in vals_boot if isinstance(v, int) and v > 0})
                if len(universo_boot) >= 6:
                    seed = pc_stable_seed(f"PC-SAFE-BOOT-{len(df_recorte)}-{k_exec}")
                    rng = np.random.default_rng(seed)
                    pacote = [sorted(rng.choice(universo_boot, size=6, replace=False).tolist()) for _ in range(9)]
                else:
                    pacote = []
            except Exception:
                pacote = []
        if not pacote:
            return {"ok": False, "k": int(k_exec), "erro": "modo6_sem_pacote"}

        st.session_state["pacote_listas_atual"] = pacote

        umin = int(st.session_state.get("universo_min", 1) or 1)
        umax = int(st.session_state.get("universo_max", 60) or 60)

        ok_reg = pc_replay_registrar_pacote_silent(k_reg=int(k_exec), pacote_atual=pacote, universo_min=umin, universo_max=umax)
        if not ok_reg:
            return {"ok": False, "k": int(k_exec), "erro": "registro_falhou"}

        return {"ok": True, "k": int(k_exec), "qtd_listas": int(len(pacote))}
    except Exception as e:
        return {"ok": False, "erro": str(e)}


def pc_v16_aplicar_top_cohesion_pacote(listas_totais, *, n_alvo: int = 6, seed: int = 0, i2_hint: float | None = None, strong_vals=None, calib_active: bool = True):
    """Top Cohesion leve do pacote (pré-C4, auditável).
    - Só atua quando a calibração está ativa.
    - Mantém a diversidade da maior parte do pacote.
    - Concentra parcialmente passageiros fortes em poucas listas do topo.
    """
    try:
        norm = []
        for lst in (listas_totais or []):
            try:
                li = [int(x) for x in lst[:n_alvo]]
                if len(li) >= n_alvo:
                    li = sorted(set(li))
                    if len(li) >= n_alvo:
                        norm.append(li[:n_alvo])
            except Exception:
                continue

        if not calib_active:
            return list(listas_totais or []), {
                "active": False,
                "applied": False,
                "reason": "top_cohesion_desligada_calib_inativa",
                "listas_coesas_qtd": 0,
                "cohesion_frac_efetivo": 0.0,
                "top_anchor_efetivo": 0,
                "strong_vals": [],
            }

        if not norm:
            return list(listas_totais or []), {
                "active": False,
                "applied": False,
                "reason": "pacote_vazio",
                "listas_coesas_qtd": 0,
                "cohesion_frac_efetivo": 0.0,
                "top_anchor_efetivo": 0,
                "strong_vals": [],
            }

        try:
            i2 = float(i2_hint) if i2_hint is not None else 0.0
        except Exception:
            i2 = 0.0

        # Coesão leve: poucas listas recebem âncoras fortes, o resto fica intacto.
        if i2 >= 0.90:
            cohesion_frac = 0.30
            top_anchor = min(4, n_alvo)
        elif i2 >= 0.75:
            cohesion_frac = 0.24
            top_anchor = min(4, n_alvo)
        else:
            cohesion_frac = 0.18
            top_anchor = min(3, n_alvo)

        import random
        rng = random.Random(int(seed) if seed is not None else 0)

        freq = {}
        for lst in norm:
            for v in lst[:n_alvo]:
                freq[int(v)] = freq.get(int(v), 0) + 1

        strong = []
        if isinstance(strong_vals, list) and strong_vals:
            for v in strong_vals:
                try:
                    vi = int(v)
                    if vi not in strong:
                        strong.append(vi)
                except Exception:
                    pass
        if not strong:
            strong = [v for v, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max(top_anchor + 2, 6)]]

        n_cohesive = int(round(len(norm) * cohesion_frac))
        n_cohesive = max(1, n_cohesive) if len(norm) >= 8 else min(1, len(norm))
        n_cohesive = min(n_cohesive, max(1, min(4, len(norm))))

        out = []
        for idx, base in enumerate(norm):
            if idx < n_cohesive:
                anchors = []
                for v in strong:
                    if v not in anchors:
                        anchors.append(v)
                    if len(anchors) >= top_anchor:
                        break
                preserved = [v for v in base if v not in anchors]
                rng.shuffle(preserved)
                new = []
                for v in anchors:
                    if v not in new:
                        new.append(v)
                for v in preserved:
                    if len(new) >= n_alvo:
                        break
                    if v not in new:
                        new.append(v)
                if len(new) < n_alvo:
                    rest = [v for v, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if v not in new]
                    for v in rest:
                        if v not in new:
                            new.append(int(v))
                        if len(new) >= n_alvo:
                            break
                out.append(sorted(new[:n_alvo]))
            else:
                out.append(list(base))

        seen = set()
        uniq = []
        for lst in out:
            t = tuple(lst)
            if t not in seen:
                seen.add(t)
                uniq.append(lst)
        if len(uniq) < len(norm):
            for lst in norm:
                t = tuple(lst)
                if t not in seen:
                    seen.add(t)
                    uniq.append(lst)
                if len(uniq) >= len(norm):
                    break

        info = {
            "active": True,
            "applied": bool(uniq != norm),
            "reason": "top_cohesion_aplicada" if uniq != norm else "top_cohesion_sem_delta",
            "listas_coesas_qtd": int(n_cohesive),
            "cohesion_frac_efetivo": float(cohesion_frac),
            "top_anchor_efetivo": int(top_anchor),
            "strong_vals": [int(v) for v in strong[:8]],
        }
        return uniq, info
    except Exception as e:
        return list(listas_totais or []), {
            "active": False,
            "applied": False,
            "reason": f"top_cohesion_falha: {e}",
            "listas_coesas_qtd": 0,
            "cohesion_frac_efetivo": 0.0,
            "top_anchor_efetivo": 0,
            "strong_vals": [],
        }


def pc_modo6_gerar_pacote_top10_silent(df: pd.DataFrame, calib_override=None) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Gera pacote Top10 do Modo 6 (silencioso) para a janela atual.
    Regra: é o mesmo espírito do painel, mas sem UI e com falhas silenciosas.
    v16h57FJ:
    - aceita calib_override (compatível com SAFE/CAP)
    - sempre retorna (pacote, calib_meta)
    - protege o SAFE contra abortos por assinatura/estado mínimo
    """
    calib_meta: Dict[str, Any] = {
        "active": False,
        "applied": False,
        "I_mean": 0.0,
        "I_max": 0.0,
        "I2_mean": 0.0,
        "I2_max": 0.0,
        "I": 0.0,
        "I2": 0.0,
        "ruido_lim": 3,
        "reason": "",
        "rule": {"ctop_min": 1.50, "slope_min": 0.0035, "stab_min": 0.66},
        "aplicada_no_pacote": False,
    }
    try:
        if df is None or df.empty:
            calib_meta["reason"] = "df_vazio"
            return [], calib_meta

        # k* (fallback)
        _kstar_raw = st.session_state.get("sentinela_kstar")
        k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

        nr_pct = st.session_state.get("nr_percent")
        divergencia_s6_mc = st.session_state.get("div_s6_mc")
        risco_composto = st.session_state.get("indice_risco")
        ultima_prev = st.session_state.get("ultima_previsao")

        # Ajuste de ambiente (PRÉ-ECO) — usa função canônica existente
        try:
            config = ajustar_ambiente_modo6(
                df=df,
                k_star=k_star,
                nr_pct=nr_pct,
                divergencia_s6_mc=divergencia_s6_mc,
                risco_composto=risco_composto,
                previsibilidade="alta",
            ) or {}
        except Exception:
            config = {}

        volume = int(config.get("volume_recomendado", 6) or 6)
        volume_max = int(config.get("volume_max", max(volume, 12)) or max(volume, 12))
        volume = max(1, min(volume, volume_max))

        # Detectar n_real e universo real
        col_pass = [c for c in df.columns if str(c).startswith("p")]
        if not col_pass:
            _cand = []
            for c in df.columns:
                cn = str(c).strip().lower()
                if cn in ("k", "serie", "série", "concurso", "id", "idx", "index"):
                    continue
                if cn.startswith("n") or cn.isdigit():
                    _cand.append(c)
            if _cand:
                col_pass = _cand

        universo_tmp = []
        contagens = []
        for _, row in df.iterrows():
            vals = []
            for c in col_pass:
                try:
                    if pd.notna(row[c]):
                        vals.append(int(row[c]))
                except Exception:
                    pass
            if vals:
                contagens.append(len(vals))
                universo_tmp.extend(vals)

        if not contagens or not universo_tmp:
            calib_meta["reason"] = "historico_sem_passageiros"
            return [], calib_meta

        n_real = int(pd.Series(contagens).mode().iloc[0])
        st.session_state["n_alvo"] = n_real

        universo = sorted({int(v) for v in universo_tmp if int(v) > 0})
        if not universo:
            calib_meta["reason"] = "universo_vazio"
            return [], calib_meta
        umin, umax = int(min(universo)), int(max(universo))

        st.session_state["universo_min"] = umin
        st.session_state["universo_max"] = umax
        st.session_state["universo_str"] = f"{umin}–{umax}"

        seed = pc_stable_seed(f"PC-M6-{len(df)}-{n_real}-{umin}-{umax}")
        rng = np.random.default_rng(seed)

        universo_idx = list(range(len(universo)))
        valor_por_idx = {i: universo[i] for i in universo_idx}
        idx_por_valor = {v: i for i, v in valor_por_idx.items()}

        universo_idx_use = universo_idx
        try:
            df_full_for_gov = st.session_state.get("_df_full_safe") or st.session_state.get("historico_df_full") or st.session_state.get("historico_df") or df
            snaps_map_for_gov = st.session_state.get("snapshot_p0_canonic") or {}
            k_ref = int(st.session_state.get("replay_janela_k_active", len(df)))
            decisao_p1 = _p1_auto_decidir(df_full_for_gov, snaps_map_for_gov, k_ref) if df_full_for_gov is not None else {"eligivel": False, "motivo": "df_full_ausente"}
        except Exception:
            decisao_p1 = {"eligivel": False, "motivo": "erro_decisao_p1"}

        if isinstance(decisao_p1, dict) and decisao_p1.get("eligivel"):
            ub = decisao_p1.get("ub") or []
            foco = sorted({int(v) for v in ub if umin <= int(v) <= umax})
            foco_set = set(foco)
            universo_idx_foco = [i for i, v in enumerate(universo) if int(v) in foco_set]
            if len(universo_idx_foco) >= max(2 * n_real, 10):
                universo_idx_use = universo_idx_foco

        def ajustar_para_n(lista_vals):
            out_idx = []
            for v in lista_vals:
                try:
                    vi = int(v)
                    if vi in idx_por_valor:
                        ix = idx_por_valor[vi]
                        if ix not in out_idx:
                            out_idx.append(ix)
                except Exception:
                    pass
            while len(out_idx) < n_real:
                cand = int(rng.choice(universo_idx_use))
                if cand not in out_idx:
                    out_idx.append(cand)
            return out_idx[:n_real]

        if ultima_prev:
            try:
                base_vals = ultima_prev if isinstance(ultima_prev[0], int) else ultima_prev[0]
            except Exception:
                base_vals = []
            base_idx = ajustar_para_n(base_vals)
        else:
            base_idx = rng.choice(universo_idx_use, size=n_real, replace=False).tolist()

        pool_idx = universo_idx
        pool_mode = "full"

        try:
            meta = st.session_state.get("mirror_rank_meta", None)
            if not isinstance(meta, dict):
                meta = None
            if meta is None:
                try:
                    _m1_obter_ranking_structural_df()
                except Exception:
                    pass
                meta = st.session_state.get("mirror_rank_meta", None)
                if not isinstance(meta, dict):
                    meta = None

            C_top = float(meta.get("C_top", 0.0)) if meta else 0.0
            Slope = float(meta.get("Slope", 0.0)) if meta else 0.0
            Stab = float(meta.get("Stab", 0.0)) if meta else 0.0
            Gap = meta.get("Gap", None) if meta else None
            Gap = float(Gap) if Gap is not None else 0.0

            I1 = max(0.0, min(1.0, (C_top - 1.20) / 1.20))
            I2m = max(0.0, min(1.0, (Slope - 0.0020) / 0.0030))
            I3 = max(0.0, min(1.0, (Stab - 0.55) / 0.35))
            I4 = max(0.0, min(1.0, (Gap - 0.0015) / 0.0030))
            I = float((I1 + I2m + I3 + I4) / 4.0)

            J1 = max(0.0, min(1.0, (C_top - 1.00) / 0.80))
            J2 = max(0.0, min(1.0, (Slope) / 0.0040))
            J3 = max(0.0, min(1.0, (Stab - 0.50) / 0.25))
            J4 = max(0.0, min(1.0, (Gap) / 0.0030))
            I2_contraste = float((J1 + J2 + J3 + J4) / 4.0)

            calib_meta["active"] = True
            calib_meta["C_top"] = float(C_top)
            calib_meta["Slope"] = float(Slope)
            calib_meta["Stab"] = float(Stab)
            calib_meta["Gap"] = float(Gap)
            calib_meta["I_mean"] = float(I)
            calib_meta["I_max"] = float(I)
            calib_meta["I"] = float(I)
            calib_meta["I2"] = float(I2_contraste)
            calib_meta["I2_mean"] = float(I2_contraste)
            calib_meta["I2_max"] = float(I2_contraste)

            suggested = (C_top >= calib_meta["rule"]["ctop_min"]) and (Slope >= calib_meta["rule"]["slope_min"]) and (Stab >= calib_meta["rule"]["stab_min"])

            if calib_override is True:
                apply = True
            elif calib_override is False:
                apply = False
            else:
                apply = bool(suggested)

            if apply:
                calib_meta["ruido_lim"] = 2 if I >= 0.35 else 3
                calib_meta["applied"] = True
                calib_meta["reason"] = f"aplicada (C_top={C_top:.2f}, Slope={Slope:.4f}, Stab={Stab:.2f}, Gap={Gap:.4f}, I={I:.2f})"
            else:
                calib_meta["ruido_lim"] = 3
                calib_meta["applied"] = False
                if calib_override is False:
                    calib_meta["reason"] = "baseline_forçado (calib_override=False)"
                elif calib_override is True:
                    calib_meta["reason"] = "calib_forçada (calib_override=True)"
                else:
                    calib_meta["reason"] = f"não aplicada (C_top={C_top:.2f}, Slope={Slope:.4f}, Stab={Stab:.2f})"
        except Exception as _e:
            calib_meta["active"] = True
            calib_meta["applied"] = False
            calib_meta["ruido_lim"] = 3
            calib_meta["reason"] = f"calib_erro: {type(_e).__name__}: {_e}"

        calib_meta["aplicada_no_pacote"] = bool(calib_meta.get("applied", False))
        st.session_state["v16_calib_leve_last_summary"] = calib_meta.copy()

        inv_pos = None
        try:
            if isinstance(universo_idx_use, list) and universo_idx_use != universo_idx:
                pool_idx = list(universo_idx_use)
                pool_mode = "foco_p1"
                inv_pos = {int(ix): j for j, ix in enumerate(pool_idx)}
        except Exception:
            pool_idx = universo_idx
            pool_mode = "full"
            inv_pos = None

        listas_brutas = []
        for _ in range(int(volume)):
            ruido_lim = int(st.session_state.get('v16_calib_leve_last_summary', {}).get('ruido_lim', 3))
            ruido_lim = 3 if ruido_lim not in (1, 2, 3) else ruido_lim
            ruido = rng.integers(-ruido_lim, ruido_lim + 1, size=n_real)
            if pool_mode == "foco_p1" and inv_pos is not None:
                nova_idx = []
                for idx, r in zip(base_idx, ruido):
                    pos = inv_pos.get(int(idx), None)
                    if pos is None:
                        nova_idx.append(max(0, min(len(universo) - 1, int(idx))))
                    else:
                        new_pos = max(0, min(len(pool_idx) - 1, int(pos) + int(r)))
                        nova_idx.append(int(pool_idx[new_pos]))
            else:
                nova_idx = [max(0, min(len(universo_idx) - 1, int(idx) + int(r))) for idx, r in zip(base_idx, ruido)]
            nova = [int(valor_por_idx[i]) for i in nova_idx]
            listas_brutas.append(nova)

        listas_filtradas = []
        for lista in listas_brutas:
            try:
                if all(umin <= int(v) <= umax for v in lista):
                    listas_filtradas.append([int(v) for v in lista])
            except Exception:
                pass

        # ------------------------------------------------------------
        # CT / GENERATOR PATH CANÔNICO (DQ)
        # - Atua ANTES da sanidade final
        # - Não altera replay
        # - Não altera pipeline
        # ------------------------------------------------------------
        _listas_pre_sanidade = list(listas_filtradas)

        try:
            _opening_ranking_vals = []
            try:
                _opening_ranking_vals = [int(v) for v in (calib_meta.get("top_pool") or [])]
            except Exception:
                _opening_ranking_vals = []
            _listas_pre_sanidade, _opening_info = pc_v16_generator_opening_control(
                _listas_pre_sanidade,
                ranking_vals=_opening_ranking_vals,
                n_alvo=n_real,
                target_unique_min=28,
                max_replace_per_list=4,
            )
            calib_meta["generator_opening_control"] = dict(_opening_info or {})
            try:
                pc_trace_store("pc_trace_after_generator_opening", _listas_pre_sanidade, "1.0) PRE SANIDADE GENERATOR OPENING CONTROL")
            except Exception:
                pass
        except Exception as _e:
            calib_meta["generator_opening_control"] = {
                "active": False,
                "applied": False,
                "reason": f"generator_opening_control_erro: {_e}",
            }

        try:
            _ranking_vals_at = []
            try:
                _ranking_vals_at = [int(v) for v in (calib_meta.get("top_pool") or [])]
            except Exception:
                _ranking_vals_at = []

            _listas_pre_sanidade, _npgen_info = pc_v16_new_packet_generator(
                _listas_pre_sanidade,
                ranking_vals=_ranking_vals_at,
                historico_df=df,
                n_alvo=n_real,
                seed=seed,
                max_lists=len(_listas_pre_sanidade),
            )
            try:
                pc_trace_store("pc_trace_after_npg", _listas_pre_sanidade, "1.5) PRE SANIDADE NEW PACKET GENERATOR")
            except Exception:
                pass
            calib_meta["new_packet_generator"] = dict(_npgen_info or {})
            try:
                st.session_state["v16_ct_last_real_generator"] = dict(_npgen_info or {})
            except Exception:
                pass
            pc_exec_trace(
                "AFTER pc_v16_new_packet_generator_PRE_SANIDADE",
                dict(_npgen_info or {}, **pc_packet_audit_dict(_listas_pre_sanidade, "after_new_packet_pre_sanidade"))
            )
        except Exception as _e:
            calib_meta["new_packet_generator"] = {
                "active": False,
                "applied": False,
                "reason": f"new_packet_generator_erro: {_e}",
                "listas_regeneradas_qtd": 0,
            }
            try:
                st.session_state["v16_ct_last_real_generator"] = dict(calib_meta["new_packet_generator"])
            except Exception:
                pass

        # ------------------------------------------------------------
        # DR — CT no gerador INTERNO do Mode6, antes da sanidade
        # ------------------------------------------------------------
        try:
            _ranking_vals_internal = []
            try:
                _ranking_vals_internal = [int(v) for v in (calib_meta.get("top_pool") or [])]
            except Exception:
                _ranking_vals_internal = []
            _listas_pre_sanidade, _npgen_internal_info = pc_v16_new_packet_generator(
                _listas_pre_sanidade,
                ranking_vals=_ranking_vals_internal,
                historico_df=df,
                n_alvo=n_real,
                seed=seed,
                max_lists=len(_listas_pre_sanidade),
            )
            calib_meta["new_packet_generator"] = dict(_npgen_internal_info or {})
            try:
                st.session_state["v16_ct_last_real_generator"] = dict(_npgen_internal_info or {})
            except Exception:
                pass
            try:
                pc_trace_store("pc_trace_after_npg_internal", _listas_pre_sanidade, "1.75) PRE SANIDADE INTERNAL NEW PACKET GENERATOR")
            except Exception:
                pass
        except Exception as _e_internal:
            calib_meta["new_packet_generator"] = {
                "active": False,
                "applied": False,
                "reason": f"new_packet_generator_internal_erro: {_e_internal}",
                "listas_regeneradas_qtd": 0,
            }
            try:
                st.session_state["v16_ct_last_real_generator"] = dict(calib_meta["new_packet_generator"])
            except Exception:
                pass

        listas_totais = sanidade_final_listas(_listas_pre_sanidade)
        pc_exec_trace("AFTER sanidade_final_listas", pc_packet_audit_dict(listas_totais, "after_sanidade"))
        try:
            pc_trace_store("pc_trace_after_sanidade", listas_totais, "2) POST SANIDADE FINAL LISTAS")
        except Exception:
            pass
        # ------------------------------------------------------------
                # TOP COHESION DO PACOTE (AN)
        # - Pré-C4 · auditável · não altera volume
                # - Só atua quando a calibração está ativa/aplicada
        # - Mantém baseline intacto quando calib_override=False
        # ------------------------------------------------------------
        try:
            _top_pool_vals = []
            try:
                _top_pool_vals = [int(v) for v in (calib_meta.get("top_pool") or [])]
            except Exception:
                _top_pool_vals = []
            if calib_override is False:
                _comp_info = {
                    "active": False,
                    "applied": False,
                    "reason": "top_cohesion_desligada_baseline",
                    "listas_comprimidas_qtd": 0,
                    "compress_frac_efetivo": 0.0,
                    "top_keep_efetivo": 0,
                    "strong_vals": [],
                }
            else:
                listas_totais, _comp_info = pc_v16_aplicar_top_cohesion_pacote(
                    listas_totais,
                    n_alvo=n_real,
                    seed=seed,
                    i2_hint=calib_meta.get("I2_max", calib_meta.get("I2_mean", calib_meta.get("I2", 0.0))),
                    strong_vals=_top_pool_vals,
                    calib_active=bool(calib_meta.get("applied", False)),
                )
            calib_meta["packet_compression"] = dict(_comp_info)
            pc_exec_trace("AFTER pc_v16_aplicar_top_cohesion_pacote", dict(_comp_info or {}, **pc_packet_audit_dict(listas_totais, "after_top_cohesion")))
        except Exception as _e:
            calib_meta["packet_compression"] = {"active": False, "applied": False, "reason": f"top_cohesion_erro: {_e}"}

        try:
            pc_trace_store("pc_trace_before_controller", listas_totais, "3) BEFORE CONTROLLER")
        except Exception:
            pass

        try:
            _packet_before = pc_packet_metrics(listas_totais)
            _intensidade_prev = float(st.session_state.get("pc_cohesion_intensity", 0.50) or 0.50)
            _auto_tune_info = pc_cohesion_auto_tune(
                _packet_before,
                intensidade_atual=_intensidade_prev,
                u_target_mid=21.0,
                o_target_mid=1.9,
                alpha=0.20,
            )
            _intensidade_usada = float((_auto_tune_info or {}).get("intensidade_nova", _intensidade_prev) or _intensidade_prev)
            st.session_state["pc_cohesion_intensity_prev"] = _intensidade_prev
            st.session_state["pc_cohesion_intensity"] = _intensidade_usada
            listas_totais = packet_cohesion_controller(listas_totais, intensidade=_intensidade_usada)
            _packet_after = pc_packet_metrics(listas_totais)
            calib_meta["cohesion_auto_tune"] = {
                "active": True,
                "intensidade_anterior": _intensidade_prev,
                "intensidade_nova": _intensidade_usada,
                "packet_before": _packet_before,
                "packet_after": _packet_after,
                "auto_tune_info": _auto_tune_info,
            }
            pc_exec_trace("AFTER packet_cohesion_controller", {
                **(_packet_after or {}),
                "intensidade_anterior": _intensidade_prev,
                "intensidade_nova": _intensidade_usada,
                "auto_tune_motivo": (_auto_tune_info or {}).get("motivo", "ok"),
            })
        except Exception as _e:
            calib_meta["cohesion_auto_tune"] = {
                "active": False,
                "motivo": f"cohesion_auto_tune_erro: {_e}",
            }
            listas_totais = packet_cohesion_controller(listas_totais, intensidade=float(st.session_state.get("pc_cohesion_intensity", 0.50) or 0.50))
            pc_exec_trace("AFTER packet_cohesion_controller", pc_packet_audit_dict(listas_totais, "after_controller"))

        try:
            pc_trace_store("pc_trace_after_controller", listas_totais, "4) AFTER CONTROLLER")
            pc_trace_store("pc_trace_before_top10", listas_totais, "5) BEFORE TOP10")
        except Exception:
            pass
        listas_top10 = listas_totais[:10]
        pc_exec_trace("AFTER listas_top10", pc_packet_audit_dict(listas_top10, "after_top10"))
        try:
            pc_trace_store("pc_trace_top10_raw", listas_top10, "6) TOP10 RAW APOS CONTROLLER")
        except Exception:
            pass

        try:
            _v8_info = st.session_state.get("v8_borda_qualificada_info", None)
            _v9_info = st.session_state.get("v9_memoria_borda", None)
            try:
                st.session_state["pacote_pre_bloco_c"] = [list(x) for x in (listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais)]
                st.session_state["pacote_pre_bloco_c_origem"] = "CAP Invisível (V1) — Modo 6 (pré-BLOCO C)"
            except Exception:
                pass

            _c_out = v10_bloco_c_aplicar_ajuste_fino_numerico(
                listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais,
                n_real=n_real,
                v8_borda_info=_v8_info,
                v9_memoria_info=_v9_info,
            )
            if _c_out.get("aplicado"):
                _aj = _c_out.get("listas_ajustadas", [])
                if isinstance(listas_top10, list) and len(listas_top10) > 0:
                    listas_top10 = _aj
                else:
                    listas_totais = _aj
                    listas_top10 = listas_totais[:10]
        except Exception:
            pass

        pacote = listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais
        st.session_state["pacote_listas_atual"] = pacote
        st.session_state["pacote_listas_origem"] = "CAP Invisível (V1) — Modo 6 (Top10)" if pacote is listas_top10 else "CAP Invisível (V1) — Modo 6 (Total)"
        try:
            st.session_state["v16_packet_compression_last"] = (calib_meta.get("packet_compression") or {}).copy() if isinstance(calib_meta, dict) else {}
        except Exception:
            pass
        return pacote, calib_meta
    except Exception as e:
        calib_meta["reason"] = f"modo6_silent_erro: {type(e).__name__}: {e}"
        return [], calib_meta


def pc_cap_invisivel_v1_processar_um_k(_df_full_safe: pd.DataFrame, k_alvo: int) -> bool:
    """Processa 1 janela k_alvo: recorta, executa pipeline+modo6 e registra snapshot P0."""
    try:
        if _df_full_safe is None or _df_full_safe.empty:
            return False
        k_alvo = int(k_alvo)
        k_alvo = max(10, min(k_alvo, len(_df_full_safe)))

        # recorte
        df_k = _df_full_safe.head(k_alvo).copy()
        st.session_state["historico_df"] = df_k
        st.session_state["replay_janela_k_active"] = int(k_alvo)

        # limpar dependências e rodar pipeline
        _pc_replay_limpar_chaves_dependentes_silent()
        ok_pipe = pc_exec_pipeline_flex_ultra_silent(df_k)
        if not ok_pipe:
            return False

        # gerar pacote e registrar snapshot
        pacote_calib, calib_meta = pc_modo6_gerar_pacote_top10_silent(df_k, calib_override=None)
        pacote_base, _meta_base = pc_modo6_gerar_pacote_top10_silent(df_k, calib_override=False)
        st.session_state['pacote_listas_atual'] = pacote_calib
        st.session_state['pacote_listas_baseline'] = pacote_base
        st.session_state['v16_calib_leve_last_summary'] = calib_meta.copy()
        pacote = pacote_calib
        if not pacote:
            return False

        try:
            umin = int(st.session_state.get("universo_min", 1) or 1)
            umax = int(st.session_state.get("universo_max", 60) or 60)
        except Exception:
            umin, umax = 1, 60

        pc_snapshot_p0_autoregistrar(pacote, k_reg=int(k_alvo), universo_min=umin, universo_max=umax)
        return True
    except Exception:
        return False
def _m1_classificar_estado(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Classifica estado S0–S6 (canônico) com base no snapshot.
    Regra: conservador; se faltar evidência, não avança estado.
    """

    S = "S0"
    nome = "Histórico inexistente"
    faltas: List[str] = []
    avisos: List[str] = []

    if snapshot.get("historico_ok"):
        S = "S1"
        nome = "Histórico carregado"
    else:
        faltas.append("Carregar Histórico")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    if snapshot.get("pipeline_ok"):
        S = "S2"
        nome = "Pipeline consolidado"
    else:
        faltas.append("Rodar Pipeline V14-FLEX ULTRA")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    # Diagnóstico completo (não exigir tudo; se houver k* e NR% já é bom indicativo)
    if snapshot.get("k_star") != "N/D" or snapshot.get("nr_percent") != "N/D":
        S = "S3"
        nome = "Diagnóstico disponível"
    else:
        avisos.append("Diagnóstico ainda parcial (k*/NR% N/D)")
        return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}

    if snapshot.get("turbo_tentado"):
        S = "S4"
        nome = "Sondagem de execução (TURBO)"
    else:
        avisos.append("TURBO não executado nesta sessão (permitido, mas reduz visibilidade do envelope)")

    if snapshot.get("modo6_executado"):
        S = "S5"
        nome = "Execução real (Modo 6)"
    else:
        faltas.append("Executar Modo 6")

    # Pós-execução: listas e/ou relatório já gerados na sessão (heurística conservadora)
    if snapshot.get("modo6_executado") and snapshot.get("listas_geradas") == "definidas":
        S = "S6"
        nome = "Pós-execução (Relatório / Governança)"

    return {"estado": S, "nome": nome, "faltas": faltas, "avisos": avisos}


def _m1_render_barra_estados(estado: str) -> None:
    ordem = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    marcadores = []
    for s in ordem:
        if ordem.index(s) < ordem.index(estado):
            marcadores.append("✓")
        elif s == estado:
            marcadores.append("●")
        else:
            marcadores.append("○")
    st.write(" ".join([f"[{s}]" for s in ordem]))
    st.write(" ".join([f" {m} " for m in marcadores]))



def _m1_obter_ranking_structural_df():
    """Ranking REAL de passageiros (universo 1–N) com base no histórico.

    • Somente leitura (pré‑C4).
    • Não depende do SAFE.
    • Não altera listas nem Camada 4.

    Observação:
    - Usa "historico_df" como fonte principal, com fallback para chaves comuns,
      porque o app mantém múltiplas variações (full/safe) ao longo do fluxo.

    V16h46 — Instrumentação de Concentração (pré‑calibração)
    - Suporta operação intencional em 1–50 OU 1–60 (override auditável) sem quebrar o AUTO.
    - Calcula métricas numéricas (somente leitura):
        C_top  : z-score do topo (média Top6 vs distribuição global)
        Slope  : média Top6 − média Borda(8–15)
        Gap    : score(rank6) − score(rank15)
        Stab   : estabilidade do Top6 ao variar a janela (w−20 / w+20)
    """
    # (v16h43+) Imports locais defensivos: evita NameError se algo sobrescrever nomes globais
    import re as _re
    import numpy as _np
    import pandas as _pd

    try:
        st.session_state.pop("mirror_rank_err", None)

        # Fonte principal + fallbacks (não muda fluxo, só leitura)
        df = st.session_state.get("historico_df", None)
        if df is None or (not isinstance(df, _pd.DataFrame)) or df.empty:
            for k in ["historico_df_full", "historico_df_full_safe", "_df_full_safe"]:
                cand = st.session_state.get(k, None)
                if isinstance(cand, _pd.DataFrame) and not cand.empty:
                    df = cand
                    break

        if df is None or (not isinstance(df, _pd.DataFrame)) or df.empty:
            st.session_state["mirror_rank_err"] = "historico_df ausente/vazio"
            return None

        # colunas de passageiros (p1..pN)
        pcols = [c for c in df.columns if isinstance(c, str) and _re.match(r"^p\d+$", c)]
        if not pcols:
            pcols = [c for c in df.columns if isinstance(c, str) and c.startswith("p")]
        if not pcols:
            st.session_state["mirror_rank_err"] = "colunas p* não encontradas"
            return None

        # -----------------------------
        # Universo (AUTO vs override 1–50 / 1–60)
        # -----------------------------
        mode = str(st.session_state.get("mirror_universe_mode", "AUTO")).strip().upper()
        override_umax = None
        if mode in ("1–50", "1-50", "50"):
            override_umax = 50
        elif mode in ("1–60", "1-60", "60"):
            override_umax = 60
        else:
            override_umax = None  # AUTO

        # universo (preferir session_state; senão inferir do df)
        umin = st.session_state.get("universo_min", None)
        umax = st.session_state.get("universo_max", None)

        def _infer_umax_umin(_df):
            try:
                a = _pd.to_numeric(_df[pcols].stack(), errors="coerce").dropna()
                if a.empty:
                    return None, None
                # remove zeros/negativos
                a = a[a > 0]
                if a.empty:
                    return None, None
                return int(a.min()), int(a.max())
            except Exception:
                return None, None

        if umin is None or umax is None:
            iu, ia = _infer_umax_umin(df)
            if umin is None:
                umin = iu
            if umax is None:
                umax = ia

        # fallback conservador
        if umin is None:
            umin = 1
        if umax is None:
            umax = 60

        # aplica override (se solicitado)
        if override_umax is not None:
            umin = 1
            umax = int(override_umax)

        # -----------------------------
        # Janela recente (fixa e auditável nesta fase)
        # -----------------------------
        w = int(min(180, len(df)))
        w = max(30, w)  # piso mínimo para evitar instabilidade extrema em histórico muito curto

        # -----------------------------
        # Computa ranking (helper) para estabilidade
        # -----------------------------
        def _rank_for_window(_w: int):
            _w = int(max(10, min(len(df), _w)))
            dfr = df.tail(_w)[pcols].copy()
            dfl = df[pcols].copy()

            # flat
            a_r = _pd.to_numeric(dfr.stack(), errors="coerce").dropna()
            a_l = _pd.to_numeric(dfl.stack(), errors="coerce").dropna()

            # filtra universo atual
            a_r = a_r[(a_r >= int(umin)) & (a_r <= int(umax))]
            a_l = a_l[(a_l >= int(umin)) & (a_l <= int(umax))]

            # frequência normalizada (proporção)
            fr = a_r.value_counts(normalize=True).to_dict()
            fl = a_l.value_counts(normalize=True).to_dict()

            rows = []
            for p in range(int(umin), int(umax) + 1):
                rr = float(fr.get(p, 0.0))
                ll = float(fl.get(p, 0.0))
                eta = rr - ll
                rows.append({"passageiro": int(p), "freq_recente": rr, "freq_longo": ll, "score": float(eta)})
            out_ = _pd.DataFrame(rows)
            out_ = out_.sort_values(["score", "passageiro"], ascending=[False, True]).reset_index(drop=True)
            out_["rank"] = _np.arange(1, len(out_) + 1)
            return out_

        out = _rank_for_window(w)

        # -----------------------------
        # Métricas de Concentração (pré‑calibração)
        # -----------------------------
        def _safe_mean(vals):
            try:
                vals = [float(v) for v in vals if v is not None]
                return float(sum(vals) / max(1, len(vals)))
            except Exception:
                return 0.0

        scores_all = [float(x) for x in out["score"].tolist()] if "score" in out.columns else []
        mu = _safe_mean(scores_all)
        sd = float(_np.std(_np.array(scores_all), ddof=0)) if scores_all else 0.0

        top6 = out.head(6)
        borda = out[(out["rank"] >= 8) & (out["rank"] <= 15)]
        top6_mean = _safe_mean(top6["score"].tolist() if not top6.empty else [])
        borda_mean = _safe_mean(borda["score"].tolist() if not borda.empty else [])

        C_top = float((top6_mean - mu) / sd) if sd > 1e-12 else 0.0
        Slope = float(top6_mean - borda_mean)

        try:
            s6 = float(out.loc[out["rank"] == 6, "score"].iloc[0])
        except Exception:
            s6 = None
        try:
            s15 = float(out.loc[out["rank"] == 15, "score"].iloc[0])
        except Exception:
            s15 = None
        Gap = float(s6 - s15) if (s6 is not None and s15 is not None) else None

        # Estabilidade do Top6 ao variar janela (w-20 / w+20)
        try:
            base_top = set(int(x) for x in top6["passageiro"].tolist())
            w_lo = int(max(10, w - 20))
            w_hi = int(min(len(df), w + 20))
            top_lo = set(int(x) for x in _rank_for_window(w_lo).head(6)["passageiro"].tolist())
            top_hi = set(int(x) for x in _rank_for_window(w_hi).head(6)["passageiro"].tolist())

            def _overlap(a, b):
                if not a or not b:
                    return 0.0
                return float(len(a & b) / float(len(a | b)))

            Stab = float((_overlap(base_top, top_lo) + _overlap(base_top, top_hi)) / 2.0)
        except Exception:
            Stab = 0.0

        # persistir para a sessão (só leitura)
        st.session_state["mirror_rank_df"] = out.copy()
        st.session_state["mirror_rank_meta"] = {
            "w_recente": int(w),
            "umin": int(umin),
            "umax": int(umax),
            "pcols_n": int(len(pcols)),
            "mode": str(mode),
            "C_top": float(C_top),
            "Slope": float(Slope),
            "Gap": (None if Gap is None else float(Gap)),
            "Stab": float(Stab),
        }
        return out

    except Exception as e:
        st.session_state["mirror_rank_err"] = f"exceção: {type(e).__name__}: {e}"
        return None


def _m1_mirror_robustez_wr_table(wr_list=(160, 180, 200, 220), wr_base=180):
    """
    Mirror (somente leitura): calcula robustez das métricas de concentração variando a janela recente (Wr).

    - Não altera listas / motor / Camada 4.
    - Não grava nada além de leitura local (não persiste em session_state).
    - Funciona para universo 1–50 e 1–60 (AUTO/override) igual ao ranking principal.

    Retorna: pandas.DataFrame com colunas:
      Wr, C_top(z), Slope, Gap(6-15), Stab_vs_base
    """
    # imports defensivos (evita NameError se algo sobrescrever globais)
    import re as _re
    import numpy as _np
    import pandas as _pd

    try:
        # Fonte principal + fallbacks (mesma lógica do ranking principal)
        df = st.session_state.get("historico_df", None)
        if df is None or (not isinstance(df, _pd.DataFrame)) or df.empty:
            for k in ["historico_df_full", "historico_df_full_safe", "_df_full_safe"]:
                cand = st.session_state.get(k, None)
                if isinstance(cand, _pd.DataFrame) and not cand.empty:
                    df = cand
                    break
        if df is None or (not isinstance(df, _pd.DataFrame)) or df.empty:
            return None

        # colunas p1..pN
        pcols = [c for c in df.columns if isinstance(c, str) and _re.match(r"^p\d+$", c)]
        if not pcols:
            pcols = [c for c in df.columns if isinstance(c, str) and c.startswith("p")]
        if not pcols:
            return None

        # Universo (AUTO vs override 1–50 / 1–60)
        mode = str(st.session_state.get("mirror_universe_mode", "AUTO")).strip().upper()
        override_umax = None
        if mode in ("1–50", "1-50", "50"):
            override_umax = 50
        elif mode in ("1–60", "1-60", "60"):
            override_umax = 60
        else:
            override_umax = None

        umin = st.session_state.get("universo_min", None)
        umax = st.session_state.get("universo_max", None)

        def _infer_umax_umin(_df):
            try:
                a = _pd.to_numeric(_df[pcols].stack(), errors="coerce").dropna()
                if a.empty:
                    return None, None
                a = a[a > 0]
                if a.empty:
                    return None, None
                return int(a.min()), int(a.max())
            except Exception:
                return None, None

        if umin is None or umax is None:
            iu, ia = _infer_umax_umin(df)
            if umin is None:
                umin = iu
            if umax is None:
                umax = ia

        if umin is None:
            umin = 1
        if umax is None:
            umax = 60

        if override_umax is not None:
            umin = 1
            umax = int(override_umax)

        # helper: ranking para um Wr específico (score = p_recente - p_longo)
        def _rank_for_window(_w: int):
            _w = int(max(10, min(len(df), _w)))
            dfr = df.tail(_w)[pcols].copy()
            dfl = df[pcols].copy()

            a_r = _pd.to_numeric(dfr.stack(), errors="coerce").dropna()
            a_l = _pd.to_numeric(dfl.stack(), errors="coerce").dropna()

            a_r = a_r[(a_r >= int(umin)) & (a_r <= int(umax))]
            a_l = a_l[(a_l >= int(umin)) & (a_l <= int(umax))]

            fr = a_r.value_counts(normalize=True).to_dict()
            fl = a_l.value_counts(normalize=True).to_dict()

            rows = []
            for p in range(int(umin), int(umax) + 1):
                rr = float(fr.get(p, 0.0))
                ll = float(fl.get(p, 0.0))
                rows.append({"passageiro": int(p), "score": float(rr - ll)})
            out_ = _pd.DataFrame(rows)
            out_ = out_.sort_values(["score", "passageiro"], ascending=[False, True]).reset_index(drop=True)
            out_["rank"] = _np.arange(1, len(out_) + 1)
            return out_

        def _safe_mean(vals):
            try:
                vals = [float(v) for v in vals if v is not None]
                return float(sum(vals) / max(1, len(vals)))
            except Exception:
                return 0.0

        def _metricas(_out):
            scores_all = [float(x) for x in _out["score"].tolist()] if "score" in _out.columns else []
            mu = _safe_mean(scores_all)
            sd = float(_np.std(_np.array(scores_all), ddof=0)) if scores_all else 0.0

            top6 = _out.head(6)
            borda = _out[(_out["rank"] >= 8) & (_out["rank"] <= 15)]
            top6_mean = _safe_mean(top6["score"].tolist() if not top6.empty else [])
            borda_mean = _safe_mean(borda["score"].tolist() if not borda.empty else [])

            c_top = float((top6_mean - mu) / sd) if sd > 1e-12 else 0.0
            slope = float(top6_mean - borda_mean)

            try:
                s6 = float(_out.loc[_out["rank"] == 6, "score"].iloc[0])
            except Exception:
                s6 = None
            try:
                s15 = float(_out.loc[_out["rank"] == 15, "score"].iloc[0])
            except Exception:
                s15 = None
            gap = float(s6 - s15) if (s6 is not None and s15 is not None) else None
            top_set = set(int(x) for x in top6["passageiro"].tolist()) if not top6.empty else set()
            return c_top, slope, gap, top_set

        base_w = int(max(10, min(len(df), wr_base)))
        base_rank = _rank_for_window(base_w)
        base_c, base_s, base_g, base_top = _metricas(base_rank)

        rows = []
        for w in wr_list:
            ww = int(max(10, min(len(df), int(w))))
            outw = _rank_for_window(ww)
            c_top, slope, gap, top_set = _metricas(outw)
            stab = float(len(top_set & base_top) / 6.0) if base_top else 0.0
            rows.append({
                "Wr": int(ww),
                "C_top(z)": float(c_top),
                "Slope": float(slope),
                "Gap(6-15)": (None if gap is None else float(gap)),
                "Stab_vs_180": float(stab),
            })

        df_out = _pd.DataFrame(rows)
        return df_out
    except Exception:
        return None

# ============================================================
# V16h51 — CALIBRAÇÃO LEVE (pré-C4) baseada em CONCENTRAÇÃO (somente leitura)
# - NÃO cria painel novo
# - NÃO altera Camada 4
# - NÃO altera risco / CAP / ECO
# - Apenas ajusta levemente núcleo × dispersão no gerador (Modo 6) quando
#   a concentração estrutural estiver sugerida e robusta.
# ============================================================

def v16_calib_leve_computar_da_concentracao(force_recompute: bool = False) -> dict:
    """Computa (ou reutiliza) a calibração leve da sessão a partir das métricas do Mirror.

    Retorna dict com:
      - active (bool)
      - I (float 0..1): intensidade
      - n_from_top (int): qtde de picks puxados do Top
      - noise_amp (int): amplitude do ruído em idx-space (1..3)
      - top_pool (list[int]): pool Top (por score) dentro do universo atual
      - meta (dict): espelho das métricas usadas

    Importante:
      - Se o Mirror não foi aberto, tenta usar mirror_rank_meta existente; se não houver,
        tenta recomputar ranking a partir do histórico (somente leitura).
    """
    try:
        if (not force_recompute) and isinstance(st.session_state.get("calib_leve_v16"), dict):
            return st.session_state["calib_leve_v16"]

        # 1) garantir meta do ranking (somente leitura)
        rank_meta = st.session_state.get("mirror_rank_meta", None)
        df_rank = st.session_state.get("mirror_rank_df", None)

        if (rank_meta is None) or (df_rank is None) or (getattr(df_rank, "empty", True)):
            try:
                df_rank = _m1_obter_ranking_structural_df()
                rank_meta = st.session_state.get("mirror_rank_meta", None)
            except Exception:
                df_rank = None
                rank_meta = None

        if not isinstance(rank_meta, dict) or df_rank is None or (not hasattr(df_rank, "empty")) or df_rank.empty:
            out = {"active": False, "I": 0.0, "n_from_top": 0, "noise_amp": 3, "top_pool": [], "meta": {"motivo": "mirror_meta_indisponivel"}}
            st.session_state["calib_leve_v16"] = out
            return out

        C_top = float(rank_meta.get("C_top", 0.0) or 0.0)
        Slope = float(rank_meta.get("Slope", 0.0) or 0.0)
        Gap = float(rank_meta.get("Gap", 0.0) or 0.0)
        Stab = float(rank_meta.get("Stab", 0.0) or 0.0)
        umin = int(rank_meta.get("umin", st.session_state.get("universo_min", 1)) or 1)
        umax = int(rank_meta.get("umax", st.session_state.get("universo_max", 60)) or 60)

        # 2) robustez por janelas (Wr) — somente leitura
        stab_wr_min = None
        try:
            wr_tbl = _m1_mirror_robustez_wr_table(wr_list=(160, 180, 200, 220), wr_base=int(rank_meta.get("w_recente", 180) or 180))
            # wr_tbl: lista de tuplas (wr, Ctop, Slope, Gap, Stab)
            if isinstance(wr_tbl, (list, tuple)) and len(wr_tbl) > 0:
                stabs = []
                for row in wr_tbl:
                    try:
                        stabs.append(float(row[4]))
                    except Exception:
                        pass
                stab_wr_min = min(stabs) if stabs else None
        except Exception:
            stab_wr_min = None

        # 3) intensidade I (0..1) — simples e estável
        def _clip01(x: float) -> float:
            try:
                return float(max(0.0, min(1.0, x)))
            except Exception:
                return 0.0

        # normalizações simples (limiares conservadores)
        i_ctop  = _clip01((C_top - 1.20) / 0.80)      # 1.20→0 ; 2.00→1
        i_stab  = _clip01((Stab - 0.60) / 0.25)       # 0.60→0 ; 0.85→1
        i_slope = _clip01((Slope - 0.0025) / 0.0025)  # 0.0025→0 ; 0.0050→1

        I = float(_clip01(i_ctop * i_stab * (0.65 + 0.35 * i_slope)))

        # 4) condições de ativação (governança)
        classe = str(st.session_state.get("classe_risco", "") or "")
        regime = str(st.session_state.get("regime", "") or st.session_state.get("regime_detectado", "") or "")
        risco_ok = (not classe.strip().startswith("🔴"))
        regime_ok = ("Ruptura" not in regime)

        robust_ok = True
        if stab_wr_min is not None:
            robust_ok = (float(stab_wr_min) >= 0.66)

        # 4b) threshold dinâmico (ANTI-RUÍDO): escala pelo histórico de I no Replay
        # Ideia: não aplicar por um valor fixo fora de escala; aplicar somente quando I estiver
        # significativamente acima do "nível típico" (mediana) do próprio Replay.
        # Isso evita: (a) calibração morta por threshold alto demais; (b) calibração barulhenta por threshold baixo demais.
        try:
            pac_hist = st.session_state.get("replay_progressivo_pacotes") or {}
            I_hist = []
            for _k, _p in pac_hist.items():
                try:
                    _cl = (_p or {}).get("calib_leve") or {}
                    _Iv = _cl.get("I")
                    if _Iv is None:
                        continue
                    I_hist.append(float(_Iv))
                except Exception:
                    continue
            if len(I_hist) >= 8:
                I_med_hist = float(statistics.median(I_hist))
                mad = float(statistics.median([abs(x - I_med_hist) for x in I_hist]))
                I_thr_dyn = I_med_hist + (3.0 * mad)
            elif len(I_hist) > 0:
                I_med_hist = float(statistics.median(I_hist))
                I_thr_dyn = I_med_hist  # sem MAD suficiente → conservador: não "adianta" o limiar
            else:
                I_med_hist = None
                I_thr_dyn = 0.25  # fallback seguro
            # Piso mínimo de escala: abaixo disso é quase sempre ruído estrutural (valor empírico do regime atual)
            I_threshold = float(max(0.02, I_thr_dyn))
        except Exception:
            I_med_hist = None
            I_thr_dyn = 0.25
            I_threshold = 0.25

        active = bool((I >= I_threshold) and robust_ok and risco_ok and regime_ok)

        # 5) parâmetros de aplicação (leves, com teto)
        n_carro = int(st.session_state.get("n_real", st.session_state.get("n_alvo", 6)) or 6)
        n_from_top = int(max(0, min(n_carro, round(I * n_carro))))
        if active and n_from_top <= 0:
            n_from_top = 1  # mínimo de efeito quando ativo

        # ruído (dispersão) — reduz levemente quando ativo
        # 3 → 2 → 1 conforme I sobe
        noise_amp = 3
        if active:
            if I >= 0.75:
                noise_amp = 1
            elif I >= 0.45:
                noise_amp = 2
            else:
                noise_amp = 3

        # top_pool (Top12 por score dentro do universo atual)
        top_pool = []
        try:
            df_top = df_rank.head(12).copy()
            top_pool = [int(x) for x in df_top["passageiro"].tolist() if (umin <= int(x) <= umax)]
        except Exception:
            top_pool = []

        out = {
            "active": bool(active),
            "applied": bool(active),  # quando ativo, influencia a dispersão (noise_amp) e o mix do topo
            "I": float(I),
            "I_threshold": float(I_threshold),
            "I_thr_dyn": (float(I_thr_dyn) if I_thr_dyn is not None else None),
            "I_med_hist": (float(I_med_hist) if I_med_hist is not None else None),
            "I_hist_n": int(len(I_hist)) if isinstance(I_hist, list) else 0,
            "n_from_top": int(n_from_top),
            "noise_amp": int(noise_amp),
            "top_pool": list(top_pool),
            "meta": {
                "C_top": float(C_top),
                "Slope": float(Slope),
                "Gap": float(Gap),
                "Stab": float(Stab),
                "stab_wr_min": (None if stab_wr_min is None else float(stab_wr_min)),
                "umin": int(umin),
                "umax": int(umax),
                "classe_risco": classe,
                "regime": regime,
                "robust_ok": bool(robust_ok),
            },
        }
        st.session_state["calib_leve_v16"] = out
        return out

    except Exception:
        out = {"active": False, "I": 0.0, "n_from_top": 0, "noise_amp": 3, "top_pool": [], "meta": {"motivo": "excecao"}}
        st.session_state["calib_leve_v16"] = out
        return out




def _m1_render_mirror_panel() -> None:
    """Painel Mirror canônico (observacional). Nunca derruba o app."""
    try:
        snapshot = _m1_collect_mirror_snapshot()
        meta = _m1_classificar_estado(snapshot)

        # MODULO 2 (infraestrutura): registrar memoria no gatilho canonico
        _m2_registrar_minimo_se_preciso(snapshot, meta)
        _m2_registrar_fechamento_se_preciso(snapshot, meta)

        st.markdown("## 🔍 Diagnóstico Espelho (Mirror)")
        st.caption("Painel somente leitura — estado real da execução · governança informativa · sem decisão")

        # ----------------------------------------------
        # V16h35 — MIRROR: Ranking estrutural (Top20) + Captura 8–15
        # (apenas leitura; não depende do SAFE; não altera listas)
        # ----------------------------------------------
        
        # ----------------------------------------------
        # V16h46 — MIRROR: Universo alvo (AUTO / 1–50 / 1–60) + Métrica de Concentração (pré‑calibração)
        # Regra: somente leitura, auditável, sem decisão e sem mexer na Camada 4.
        # ----------------------------------------------
        with st.expander("⚙️ Config do Mirror (auditoria) — Universo 1–50 / 1–60", expanded=False):
            try:
                cur = st.session_state.get("mirror_universe_mode", "AUTO")
                opts = ["AUTO", "1–50", "1–60"]
                if cur not in opts:
                    cur = "AUTO"
                sel = st.selectbox(
                    "Universo-alvo do ranking (somente leitura)",
                    options=opts,
                    index=opts.index(cur),
                    help="AUTO = inferir do histórico. 1–50/1–60 = força o ranking a considerar apenas esse universo (ignora números fora).",
                )
                st.session_state["mirror_universe_mode"] = sel
                st.caption("Obs.: isto NÃO altera listas, não altera motor e não decide nada — é apenas leitura/diagnóstico.")
            except Exception:
                pass

        df_rank = _m1_obter_ranking_structural_df()
        if df_rank is None or (not isinstance(df_rank, pd.DataFrame)) or df_rank.empty:
            err = st.session_state.get("mirror_rank_err", "N/D")
            st.warning(f"Ranking estrutural indisponível nesta sessão. (motivo: {err})")
            st.caption("Dica: carregue o histórico (Arquivo/Colar) e rode o Pipeline FLEX ULTRA. O Mirror só lê; não cria dados.")
        else:
            rank_meta = st.session_state.get("mirror_rank_meta", {})
            umin = rank_meta.get("umin", st.session_state.get("universo_min", None))
            umax = rank_meta.get("umax", st.session_state.get("universo_max", None))
            w = rank_meta.get("w_recente", 0)
            # --- v16h45: sincroniza universo (session_state + snapshot já montado)
            # Motivo: o snapshot é calculado antes do bloco do ranking; então, se o Mirror inferir 1–50 aqui,
            # precisamos refletir isso também no dict snapshot que será impresso logo abaixo.
            try:
                if isinstance(umin, int) and isinstance(umax, int):
                    st.session_state["universo_min"] = umin
                    st.session_state["universo_max"] = umax
                    st.session_state["universo_str"] = f"{umin}–{umax}"
                    # snapshot local (já criado acima) — atualiza para não ficar None–None
                    try:
                        snapshot["universo_min"] = umin
                        snapshot["universo_max"] = umax
                    except Exception:
                        pass
            except Exception:
                pass
            st.markdown("### 🧮 Ranking de Passageiros (1–N real) — somente leitura")
            st.caption(f"Fonte: histórico (p1..pN). Score = freq_recente − freq_longo. Janela recente = últimos {('N/D' if (w is None or w==0) else w)} registros.")

            # --- V16h46: Métricas numéricas de concentração (somente leitura)
            try:
                C_top = rank_meta.get("C_top", None)
                Slope = rank_meta.get("Slope", None)
                Gap = rank_meta.get("Gap", None)
                Stab = rank_meta.get("Stab", None)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("C_top (z)", f"{C_top:.2f}" if isinstance(C_top, (int, float)) else "N/D")
                with c2:
                    st.metric("Slope (Top6−Borda)", f"{Slope:.4f}" if isinstance(Slope, (int, float)) else "N/D")
                with c3:
                    st.metric("Gap (6−15)", f"{Gap:.4f}" if isinstance(Gap, (int, float)) else "N/D")
                with c4:
                    st.metric("Stab Top6", f"{Stab:.2f}" if isinstance(Stab, (int, float)) else "N/D")

                # leitura conservadora (não decide; só etiqueta)
                leitura = "NEUTRO"
                try:
                    if isinstance(C_top, (int, float)) and isinstance(Slope, (int, float)) and isinstance(Stab, (int, float)):
                        if (C_top >= 1.00) and (Slope > 0.0) and (Stab >= 0.50):
                            leitura = "CONCENTRAÇÃO SUGERIDA"
                        elif (C_top <= 0.40) or (abs(Slope) <= 0.0001) or (Stab < 0.35):
                            leitura = "BORDA PLANA / INSTÁVEL"
                except Exception:
                    leitura = "NEUTRO"

                st.caption(f"Leitura automática (sem decisão): **{leitura}** · Use isto apenas para orientar a próxima etapa (métrica → calibração → MC).")
            except Exception:
                pass



            # --- V16h50: Robustez por múltiplas janelas (Wr fixos 160/180/200/220) — somente leitura
            # Regra: não altera listas, não altera motor e não decide nada.
            with st.expander("📊 Robustez Estrutural por Janela (Somente Leitura) — Wr 160/180/200/220", expanded=False):
                try:
                    df_wr = _m1_mirror_robustez_wr_table(wr_list=(160, 180, 200, 220), wr_base=180)
                    if df_wr is None or (not isinstance(df_wr, pd.DataFrame)) or df_wr.empty:
                        st.caption("Robustez indisponível (histórico insuficiente ou estado incompleto).")
                    else:
                        # Formatação simples (não decide nada): apenas exibe números
                        st.dataframe(df_wr, use_container_width=True, hide_index=True)
                        st.caption("Stab_vs_180 = |Top6(Wr) ∩ Top6(180)| / 6 · Use isto apenas para validar se o sinal é robusto (sem decisão).")
                except Exception as _e:
                    st.caption(f"Robustez indisponível (exceção: {type(_e).__name__}).")

            st.markdown("**Top 20 (por score):**")
            st.dataframe(df_rank.head(20), use_container_width=True, hide_index=True)

            with st.expander("📌 Ver Top 50 completo (auditoria)", expanded=False):
                st.dataframe(df_rank.head(50), use_container_width=True, hide_index=True)

            st.markdown("**Borda do corte (posições 8–15):**")
            borda = df_rank[(df_rank['rank'] >= 8) & (df_rank['rank'] <= 15)].copy()
            st.dataframe(borda, use_container_width=True, hide_index=True)

            st.markdown("**Universo inferido/ativo:**")
            st.code(f"{umin}–{umax}")
            st.markdown("### 🧭 Estado Operacional Atual")
        st.markdown(f"**{meta['estado']} — {meta['nome']}**")
        _m1_render_barra_estados(meta["estado"])

        if meta.get("faltas"):
            st.info("Ainda não percorrido (na sessão): " + " · ".join(meta["faltas"]))
        if meta.get("avisos"):
            for a in meta["avisos"]:
                st.warning(a)

        st.markdown("---")
        st.markdown("### 📋 Snapshot (read-only)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write({
                "historico_df": snapshot.get("historico_df"),
                "n_alvo": snapshot.get("n_alvo"),
                "universo": f"{snapshot.get('universo_min')}–{snapshot.get('universo_max')}",
                "pipeline_ok": snapshot.get("pipeline_ok"),
                "regime": snapshot.get("regime"),
            })
        with col2:
            st.write({
                "k_star": snapshot.get("k_star"),
                "nr_percent": snapshot.get("nr_percent"),
                "divergencia_s6_mc": snapshot.get("divergencia_s6_mc"),
                "indice_risco": snapshot.get("indice_risco"),
                "classe_risco": snapshot.get("classe_risco"),
            })
        with col3:
            st.write({
                "turbo_tentado": snapshot.get("turbo_tentado"),
                "turbo_bloqueado": snapshot.get("turbo_bloqueado"),
                "turbo_motivo": snapshot.get("turbo_motivo"),
                "modo6_executado": snapshot.get("modo6_executado"),
                "listas_geradas": snapshot.get("listas_geradas"),
            })

        with st.expander("🧪 Chaves do session_state (auditoria leve)"):
            st.write(snapshot.get("keys", []))


        with st.expander('🧠 M2 - Memoria de Estados (auditoria controlada)'):
            st.write(_m2_resumo_auditavel())

    except Exception as _e:
        # Falha silenciosa: não derrubar o app.
        st.warning(f"⚠️ Mirror falhou (silencioso): {_e}")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — TOPO DO ARQUIVO
# (DESATIVADA — substituída pela CAMADA D real)
# Mantida apenas como registro histórico
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# def v16_registrar_estado_alvo():
#     return {
#         "tipo": "indefinido",
#         "velocidade": "indefinida",
#         "comentario": "Estado ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_expectativa():
#     return {
#         "previsibilidade": "indefinida",
#         "erro_esperado": "indefinido",
#         "chance_janela_ouro": "baixa",
#         "comentario": "Expectativa ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# def v16_registrar_volume_e_confiabilidade():
#     return {
#         "minimo": "-",
#         "recomendado": "-",
#         "maximo_tecnico": "-",
#         "confiabilidades_estimadas": {},
#         "comentario": "Volume ainda não disponível (carregue histórico e rode Sentinelas/Pipeline).",
#     }

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIM — V16 — GUARDA CANÔNICA (ANTI-NAMEERROR) — DESATIVADA
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# ============================================================
# FUNÇÃO — CARREGAMENTO UNIVERSAL DE HISTÓRICO (FLEX ULTRA)
# REGRA FIXA:
# - Último valor da linha = k
# - Quantidade de passageiros é LIVRE
# - Universo é derivado do histórico (SANIDADE)
# ============================================================




def carregar_historico_universal(linhas):
    import pandas as pd
    
    # Auditoria (governança): quantas linhas foram descartadas por invalidade estrutural
    skipped_total = 0
    skipped_len = 0
    skipped_nonnum = 0
    skipped_empty_k = 0
    skipped_examples = []
    
    registros = []
    
    for idx, linha in enumerate(linhas, start=1):
        if linha is None:
            continue
        
        linha = linha.strip()
        if not linha:
            continue
        
        # Governança: só processa linhas que representam séries (ID começando com 'C')
        # Isso elimina cabeçalhos, comentários e lixo residual sem afrouxar o determinismo 6+k.
        if not linha.startswith("C"):
            continue
        
        partes = linha.split(";")
        
        # Se houver ';' no final, pode gerar campo vazio — removemos de forma canônica.
        if len(partes) > 0 and partes[-1] == "":
            # Campo vazio final indica "k vazio" => linha inválida (descarta), não derruba app
            skipped_total += 1
            skipped_empty_k += 1
            if len(skipped_examples) < 3:
                skipped_examples.append((idx, "k_vazio", linha))
            continue
        
        # Esperado: ID + 6 passageiros + k  => total 8 campos
        if len(partes) != 8:
            skipped_total += 1
            skipped_len += 1
            if len(skipped_examples) < 3:
                skipped_examples.append((idx, f"len={len(partes)}", linha))
            continue
        
        identificador = partes[0]
        
        try:
            passageiros = [int(x) for x in partes[1:7]]
            k = int(partes[7])
        except ValueError:
            skipped_total += 1
            skipped_nonnum += 1
            if len(skipped_examples) < 3:
                skipped_examples.append((idx, "nao_numerico", linha))
            continue
        
        registro = {
            "id": identificador,
            "p1": passageiros[0],
            "p2": passageiros[1],
            "p3": passageiros[2],
            "p4": passageiros[3],
            "p5": passageiros[4],
            "p6": passageiros[5],
            "k": k
        }
        
        registros.append(registro)
    
    df = pd.DataFrame(registros)
    
    # Registrar auditoria no session_state (sem quebrar caso Streamlit não esteja disponível)
    try:
        import streamlit as st
        st.session_state["HIST_PARSER_SKIPPED_TOTAL"] = skipped_total
        st.session_state["HIST_PARSER_SKIPPED_LEN"] = skipped_len
        st.session_state["HIST_PARSER_SKIPPED_NONNUM"] = skipped_nonnum
        st.session_state["HIST_PARSER_SKIPPED_EMPTY_K"] = skipped_empty_k
        st.session_state["HIST_PARSER_SKIPPED_EXAMPLES"] = skipped_examples
    except Exception:
        pass
    
    
    # --- v16h37 (AUDIT): inferir universo real 1..N a partir do histórico (p1..pN)
    try:
        import streamlit as st
        pcols = [c for c in df.columns if isinstance(c, str) and c.startswith("p")]
        if pcols:
            _vals = pd.to_numeric(df[pcols], errors="coerce")
            _umin = int(_vals.min().min())
            _umax = int(_vals.max().max())
            st.session_state["universo_min"] = _umin
            st.session_state["universo_max"] = _umax
            st.session_state["universo_str"] = f"{_umin}–{_umax}"
    except Exception:
        pass

    return df
# ============================================================
# V16 PREMIUM — IMPORTAÇÃO OFICIAL
# (Não altera nada do V15.7, apenas registra os painéis novos)
# ============================================================

from app_v16_premium import (
    v16_obter_paineis,
    v16_renderizar_painel,
)

# ============================================================
# Configuração da página (obrigatório V15.7 MAX)
# ============================================================

# ============================================================
# V16 — CAMADA ORBITA (E1) + GRADIENTE (G0–G3) + N_EXTRA
# (sem interceptação automática; sem travas; sem painel novo)
# ============================================================

def v16_orbita__interseccao(a, b):
    return len(set(a).intersection(set(b)))

def v16_orbita__pares_interseccao(listas):
    # retorna contagem de pares com intersecção >=2 e >=3
    if not listas or len(listas) < 2:
        return {"pares_total": 0, "pares_ge2": 0, "pares_ge3": 0}
    pares_total = 0
    ge2 = 0
    ge3 = 0
    for i in range(len(listas)):
        for j in range(i+1, len(listas)):
            pares_total += 1
            inter = v16_orbita__interseccao(listas[i], listas[j])
            if inter >= 2:
                ge2 += 1
            if inter >= 3:
                ge3 += 1
    return {"pares_total": pares_total, "pares_ge2": ge2, "pares_ge3": ge3}

def v16_calcular_orbita_pacote(listas_topN, universo_min, universo_max):
    """Calcula ORBITA_E0/E1 + métricas (f_max, range_8, pares>=2/3).
    Não altera listas; apenas descreve o pacote.
    """
    info = {
        "estado": "E0",
        "selo": "E0",
        "f_max": 0.0,
        "range_8": None,
        "range_lim": None,
        "pares_ge2": 0.0,
        "pares_ge3": 0.0,
        "ancoras": [],
        "top_passageiros": [],
        "listas_top": [],
    }
    try:
        if not listas_topN:
            return info

        info["listas_top"] = listas_topN

        # Frequências por passageiro
        from collections import Counter
        flat = [p for lst in listas_topN for p in lst]
        if not flat:
            return info
        c = Counter(flat)
        top_pass = [p for p, _ in c.most_common(12)]
        info["top_passageiros"] = top_pass

        # f_max normalizado por N (em quantas listas aparece o passageiro mais recorrente)
        # Atenção: usamos presença por lista (não contagem bruta).
        pres = Counter()
        for lst in listas_topN:
            for p in set(lst):
                pres[p] += 1
        if not pres:
            return info
        f_max = max(pres.values()) / float(len(listas_topN))
        info["f_max"] = float(round(f_max, 4))

        # âncoras: passageiros com presença >= 50% no pacote TopN
        ancoras = [p for p, v in pres.items() if (v / float(len(listas_topN))) >= 0.50]
        ancoras = sorted(ancoras)[:10]
        info["ancoras"] = ancoras

        # compressão de faixa (Top8 por frequência bruta)
        top8 = [p for p, _ in c.most_common(8)]
        if top8:
            r8 = max(top8) - min(top8)
            info["range_8"] = int(r8)
        else:
            info["range_8"] = None

        # limite de compressão depende do universo
        universo_size = int(universo_max) - int(universo_min) + 1
        lim = int(round(universo_size * 0.44))  # ~22 em 1–50, ~26 em 1–60
        info["range_lim"] = lim

        # coerência de interseção
        pares = v16_orbita__pares_interseccao(listas_topN)
        if pares["pares_total"] > 0:
            info["pares_ge2"] = float(round(pares["pares_ge2"] / pares["pares_total"], 4))
            info["pares_ge3"] = float(round(pares["pares_ge3"] / pares["pares_total"], 4))

        # decisão E1 (quase-órbita) — criteriosa mas sem "freio" no operador:
        # - f_max em zona de quase-âncora (0.35..0.70)
        # - range_8 comprimido (<= lim)
        # - pares>=2 moderado (>= 0.35)
        # - pares>=3 não explosivo (<= 0.35)
        passa_f = (info["f_max"] >= 0.35) and (info["f_max"] <= 0.70)
        passa_range = (info["range_8"] is not None) and (info["range_8"] <= lim)
        passa_ge2 = (info["pares_ge2"] >= 0.35)
        passa_ge3 = (info["pares_ge3"] <= 0.35)

        if passa_f and passa_range and passa_ge2 and passa_ge3:
            info["estado"] = "E1"
            info["selo"] = "E1"
        else:
            info["estado"] = "E0"
            info["selo"] = "E0"

        return info
    except Exception:
        # falha silenciosa: não derruba app
        return info

def v16_calcular_gradiente_E1(info_orbita):
    """Retorna G0..G3.
    G0: E0 (mar aberto)
    G1: E1 fraco
    G2: E1 consistente
    G3: E1 comprimido (quase E2)
    """
    try:
        if not info_orbita or info_orbita.get("estado") != "E1":
            return {"gradiente": "G0", "score": 0.0}

        f = float(info_orbita.get("f_max") or 0.0)
        ge2 = float(info_orbita.get("pares_ge2") or 0.0)
        ge3 = float(info_orbita.get("pares_ge3") or 0.0)
        r8 = info_orbita.get("range_8")
        lim = info_orbita.get("range_lim") or 1

        # componentes normalizados
        # f ideal ~0.55 (meio termo)
        f_score = 1.0 - min(1.0, abs(f - 0.55) / 0.20)  # tolerância 0.20
        # range: quanto menor que lim, melhor
        if r8 is None:
            r_score = 0.0
        else:
            r_score = 1.0 - min(1.0, max(0.0, (r8 / float(lim)) - 0.75) / 0.75)  # bom até 0.75*lim
        # ge2: quanto maior, melhor (até 0.85)
        ge2_score = min(1.0, ge2 / 0.85)
        # ge3: penaliza explosão de iguais
        ge3_pen = min(1.0, max(0.0, (ge3 - 0.22) / 0.35))

        score = (0.35*f_score + 0.35*r_score + 0.30*ge2_score) * (1.0 - 0.35*ge3_pen)
        score = float(round(max(0.0, min(1.0, score)), 4))

        if score >= 0.78:
            g = "G3"
        elif score >= 0.60:
            g = "G2"
        else:
            g = "G1"

        return {"gradiente": g, "score": score}
    except Exception:
        return {"gradiente": "G0", "score": 0.0}

def v16_calcular_N_extra(estado_orbita, gradiente, n_base, eco_forca=None, eco_acionabilidade=None):
    """Expansão condicional do pacote.
    - Não divide pacote (mantém N_BASE intacto)
    - Apenas adiciona N_EXTRA quando justificável
    - Sem travar operador (apenas informa + gera listas)
    """
    try:
        n_base = int(n_base or 0)
        if n_base <= 0:
            return 0

        # qualificador ECO (se disponível)
        eco_ok = True
        if eco_acionabilidade is not None:
            eco_ok = (str(eco_acionabilidade).lower() != "não_acionável") and (str(eco_acionabilidade).lower() != "nao_acionavel")
        # se não existe ECO, não bloqueia

        if not eco_ok and estado_orbita != "E2":
            return 0

        if estado_orbita == "E2":
            return int(max(2, min(8, round(0.5 * n_base))))
        if gradiente == "G3":
            return int(max(2, min(6, round(0.3 * n_base))))
        if gradiente == "G2":
            return int(max(1, min(4, round(0.28 * n_base))))
        return 0
    except Exception:
        return 0

def v16_gerar_listas_extra_por_orbita(info_orbita, universo_min, universo_max, n_carro, qtd, seed=0):
    """Gera listas extras (N_EXTRA) com viés de interseção/âncoras.
    Sem interceptação automática: é só expansão condicional do pacote.
    """
    try:
        import random
        rnd = random.Random(int(seed or 0) + 991)
        universo = list(range(int(universo_min), int(universo_max)+1))

        ancoras = list(info_orbita.get("ancoras") or [])
        top = list(info_orbita.get("top_passageiros") or [])
        base_pool = [p for p in (ancoras + top) if (p in universo)]
        base_pool = list(dict.fromkeys(base_pool))  # unique preserve order

        listas = []
        alvo_anchors = min(3, max(1, len(base_pool)//3)) if base_pool else 0

        for _ in range(int(qtd or 0)):
            lst = []
            # fixa 2–3 âncoras/top
            if base_pool:
                kfix = min(alvo_anchors + rnd.randint(0, 1), max(1, min(3, len(base_pool))))
                lst.extend(rnd.sample(base_pool, kfix))
            # completa aleatório do universo, evitando duplicatas
            while len(lst) < int(n_carro):
                p = rnd.choice(universo)
                if p not in lst:
                    lst.append(p)
            lst = sorted(lst)
            listas.append(lst)

        # remove duplicatas exatas
        uniq = []
        seen = set()
        for l in listas:
            t = tuple(l)
            if t not in seen:
                seen.add(t)
                uniq.append(l)
        return uniq
    except Exception:
        return []



# ============================================================
# ============================================================
# V16 — CAMADA REMOVIDA (âncora estável)
# Motivo: este arquivo âncora opera sem camadas experimentais.
# ============================================================

# ============================================================
# V16 — APS (Auditoria de Postura do Sistema) — Observacional
# ============================================================
def v16_calcular_aps_postura(nr_percent=None, orbita_selo=None, eco_acionabilidade=None, anti_exato_level=None):
    """APS = Auditoria de Postura do Sistema.
    - Observacional: NÃO muda listas, NÃO decide volume.
    - Classifica risco/postura e sugere forma compatível (denso/espalhado/duplo pacote) sem impor.
    """
    try:
        nr = float(nr_percent) if nr_percent is not None else None
    except Exception:
        nr = None

    selo = (orbita_selo or "E0").strip()
    eco = (eco_acionabilidade or "N/D").strip()

    # Anti-exato: "alto"/"médio"/"baixo" ou None
    ae = (str(anti_exato_level).strip().lower() if anti_exato_level is not None else "")

    # Regras deliberadamente conservadoras
    if (nr is not None) and (nr >= 75):
        return ("🔴", "Postura Crítica", "Ruído crítico (NR alto). Evitar ancoragem forte; preferir pacote espalhado e baixo volume. Observação: agir com cautela.")
    if (selo.startswith("E0")) and (nr is not None) and (nr >= 55):
        return ("🟡", "Postura Sensível", "E0 + ruído alto: ancoragem excessiva é perigosa. Preferir duplo pacote (base + anti-âncora) sem aumentar universo.")
    if (selo.startswith("E1") or selo.startswith("E2")) and (nr is not None) and (nr <= 55):
        return ("🟢", "Postura Operável", "Órbita emergente com ruído sob controle. Densidade moderada pode ser compatível; manter governança e observar persistência.")
    if eco.lower() in ("acionável", "acionavel") and (nr is not None) and (nr <= 60):
        return ("🟢", "Postura Operável", "ECO acionável com ruído aceitável. Operar com disciplina: microvariações/envelope e testes de consistência.")
    # fallback
    return ("⚪", "Postura Neutra", "Sem evidência suficiente para postura ativa. Manter pacote base e acompanhar série a série (detecção/sensibilidade/gradiente).")


# ============================================================
# V16 — SINCRONIZAÇÃO CANÔNICA (ALIASES) + ANTI-ÂNCORA (OBS)
# ============================================================

def v16_sync_aliases_canonicos(force: bool = False) -> dict:
    """Sincroniza variáveis canônicas usadas no Relatório Final / Registro.
    - Observacional: NÃO altera motores, NÃO altera listas, NÃO decide.
    - Objetivo: evitar N/D indevido quando dados existem sob chaves antigas/alternativas.
    """
    mudancas = {}

    # k* (sentinela)
    if force or (st.session_state.get("k_star") is None):
        if st.session_state.get("sentinela_kstar") is not None:
            st.session_state["k_star"] = st.session_state.get("sentinela_kstar")
            mudancas["k_star<=sentinela_kstar"] = True

    # Divergência S6 vs MC
    if force or (st.session_state.get("divergencia_s6_mc") is None):
        if st.session_state.get("div_s6_mc") is not None:
            st.session_state["divergencia_s6_mc"] = st.session_state.get("div_s6_mc")
            mudancas["divergencia_s6_mc<=div_s6_mc"] = True

    # ECO/Estado (painel V16 mastigado)
    diag = st.session_state.get("diagnostico_eco_estado_v16")
    if isinstance(diag, dict):
        eco_txt = f"{diag.get('eco_forca','')} · {diag.get('eco_persistencia','')} · {diag.get('eco_acionabilidade','')}"
        eco_txt = eco_txt.strip(" ·").strip()
        if eco_txt and (force or (not st.session_state.get("eco_status")) or (st.session_state.get("eco_status") in ("N/D", "DESCONHECIDO"))):
            st.session_state["eco_status"] = eco_txt
            mudancas["eco_status<=diagnostico_eco_estado_v16"] = True

        est_txt = str(diag.get("estado") or "").strip()
        if est_txt and (force or (not st.session_state.get("estado_atual")) or (st.session_state.get("estado_atual") in ("N/D", "DESCONHECIDO"))):
            st.session_state["estado_atual"] = est_txt
            mudancas["estado_atual<=diagnostico_eco_estado_v16"] = True

    return mudancas


def v16_analisar_duplo_pacote_base_anti_ancora(
    listas: list,
    base_n: int = 10,
    max_anti: int = 4,
    core_presenca_min: float = 0.60,
) -> dict:
    """Analisa ancoragem do pacote (OBSERVACIONAL).

    - Define um 'CORE' do pacote base (Top N) por presença em listas.
    - Mede overlap (0..len(core)) de cada lista com o CORE.
    - Sugere (sem impor) quais listas existentes podem servir como 'anti-âncora':
        listas com overlap baixo com o CORE (mas ainda dentro do mesmo universo).

    Retorna um dict com:
    - core: lista de passageiros do CORE
    - overlaps: lista de overlaps por lista (índice alinhado a 'listas')
    - base_idx: índices (1-based) do pacote base
    - anti_idx: índices (1-based) sugeridos como anti-âncora (existentes)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {"core": [], "overlaps": [], "base_idx": [], "anti_idx": [], "nota": "sem_listas"}

        base_n = int(base_n or 0)
        base_n = max(1, min(base_n, len(listas)))
        base = listas[:base_n]

        # Presença por lista (não contagem bruta)
        from collections import Counter

        pres = Counter()
        for L in base:
            if not isinstance(L, (list, tuple)):
                continue
            for p in set(int(x) for x in L):
                pres[p] += 1

        if not pres:
            return {"core": [], "overlaps": [0 for _ in listas], "base_idx": list(range(1, base_n + 1)), "anti_idx": [], "nota": "sem_presenca"}

        # CORE: passageiros com presença >= core_presenca_min no pacote base.
        core = [p for p, v in pres.items() if (v / float(base_n)) >= float(core_presenca_min)]
        core = sorted(core)

        # fallback: se core ficou vazio, usa TOP3 por presença (mais conservador)
        if not core:
            core = [p for p, _ in pres.most_common(3)]
            core = sorted(core)

        core_set = set(core)

        overlaps = []
        for L in listas:
            if not isinstance(L, (list, tuple)):
                overlaps.append(0)
                continue
            overlaps.append(len(set(int(x) for x in L).intersection(core_set)))

        # Anti-âncora: fora do TopN (preferência), overlap baixo com CORE.
        # Threshold: <=1 quando core >=3, senão <=0.
        thr = 1 if len(core) >= 3 else 0

        candidatos = []
        for idx in range(base_n, len(listas)):
            ov = overlaps[idx]
            if ov <= thr:
                candidatos.append((ov, idx))

        # ordena por overlap menor e por índice estável
        candidatos = sorted(candidatos, key=lambda t: (t[0], t[1]))
        anti_idx = [i + 1 for _, i in candidatos[:max(0, int(max_anti))]]

        return {
            "core": core,
            "overlaps": overlaps,
            "base_idx": list(range(1, base_n + 1)),
            "anti_idx": anti_idx,
            "thr": thr,
            "base_n": base_n,
            "max_anti": int(max_anti),
            "core_presenca_min": float(core_presenca_min),
        }
    except Exception:
        return {"core": [], "overlaps": [], "base_idx": [], "anti_idx": [], "nota": "falha_silenciosa"}




# ============================================================
# V16 — DIAGNÓSTICO: RIGIDEZ DO "JEITÃO" (OBSERVACIONAL)
# - NÃO decide, NÃO altera listas, NÃO muda volume.
# - Objetivo: detectar quando o pacote está "rigidamente preso" a um jeitão
#   (ex.: concentração alta em faixa/âncoras) e sugerir apenas UMA "folga"
#   (folga qualitativa) como alerta diagnóstico — não como decisão.
# ============================================================

def v16_diagnostico_rigidez_jeitao(
    listas: list,
    universo_min: int = None,
    universo_max: int = None,
    base_n: int = 10,
    core_presenca_min: float = 0.60,
) -> dict:
    """
    Retorna:
      - rigido (bool)
      - score (0..1)
      - folga_qualitativa (str)   # apenas diagnóstico (sem prescrição numérica)
      - sinais (dict)           # métricas usadas
      - mensagem (str)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"motivo": "sem_listas"}, "mensagem": "Sem listas para diagnóstico."}

        base_n = int(base_n or 0)
        base_n = max(3, min(base_n, len(listas)))
        topN = listas[:base_n]

        # reaproveita lógica de CORE + overlaps
        anti = v16_analisar_duplo_pacote_base_anti_ancora(
            listas=listas,
            base_n=base_n,
            max_anti=4,
            core_presenca_min=float(core_presenca_min),
        )
        core = anti.get("core") or []
        overlaps = anti.get("overlaps") or []
        anti_idx = anti.get("anti_idx") or []

        core_sz = len(core)
        if core_sz <= 0 or not overlaps:
            return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"core_sz": core_sz, "motivo": "core_indisponivel"}, "mensagem": "CORE indisponível — diagnóstico de rigidez não aplicado."}

        # overlap médio e proporção de listas muito coladas no CORE
        ov_mean = float(sum([o for o in overlaps if isinstance(o, (int, float))]) / max(1, len(overlaps)))
        # "muito colado": overlap >= core_sz - 1 (quando core>=3), senão overlap==core_sz
        if core_sz >= 3:
            thr_colado = core_sz - 1
        else:
            thr_colado = core_sz
        colados = 0
        for o in overlaps:
            try:
                if int(o) >= int(thr_colado):
                    colados += 1
            except Exception:
                pass
        frac_colados = float(colados / max(1, len(overlaps)))

        # métricas de faixa / âncoras via órbita (se universo estiver disponível)
        umin = universo_min
        umax = universo_max
        if (umin is None or umax is None):
            try:
                umin = st.session_state.get("universo_min")
                umax = st.session_state.get("universo_max")
            except Exception:
                umin, umax = None, None

        orb = {}
        if (umin is not None) and (umax is not None):
            orb = v16_calcular_orbita_pacote(topN, int(umin), int(umax))
        f_max = float(orb.get("f_max") or 0.0)
        range_8 = orb.get("range_8")
        range_lim = orb.get("range_lim")

        # score de rigidez (conservador)
        score = 0.0

        # 1) colagem no CORE pesa muito
        score += 0.45 * min(1.0, frac_colados / 0.80)  # saturação em 80%

        # 2) overlap médio alto (normalizado por core_sz)
        score += 0.25 * min(1.0, (ov_mean / max(1.0, float(core_sz))) / 0.85)

        # 3) f_max alto = ancoragem forte (se disponível)
        if f_max > 0.0:
            score += 0.20 * min(1.0, max(0.0, (f_max - 0.45) / 0.35))  # acima de ~0.45 começa pesar

        # 4) faixa top8 comprimida (se disponível)
        if (range_8 is not None) and (range_lim is not None) and (range_lim > 0):
            # quanto menor a faixa vs limite, mais rígido
            comp = 1.0 - min(1.0, float(range_8) / float(range_lim))
            score += 0.10 * max(0.0, comp)

        score = float(round(max(0.0, min(1.0, score)), 4))

        # rigidez: score >= 0.62 (limiar deliberadamente conservador)
        rigido = score >= 0.62

        # folga qualitativa (diagnóstico, não decisão)
        folga_qual = "nenhuma"
        if rigido:
            if score >= 0.85:
                folga_qual = "moderada"
                msg = (
                    "Jeitão **muito rígido**: pode haver compressão excessiva. "
                    "Diagnóstico sugere **folga moderada** (alerta, não decisão)."
                )
            else:
                folga_qual = "mínima"
                msg = (
                    "Jeitão **rígido**: pode haver compressão excessiva. "
                    "Diagnóstico sugere **folga mínima** (alerta, não decisão)."
                )
        else:
            msg = (
                "Jeitão **não aparenta rigidez excessiva** (ou há folga/anti-âncora suficiente)."
            )
        sinais = {
            "core_sz": core_sz,
            "ov_mean": round(ov_mean, 4),
            "frac_colados": round(frac_colados, 4),
            "f_max": round(f_max, 4) if f_max is not None else None,
            "range_8": range_8,
            "range_lim": range_lim,
            "anti_idx_detectados": anti_idx,
        }

        return {"rigido": rigido, "score": score, "folga_qualitativa": folga_qual, "sinais": sinais, "mensagem": msg}

    except Exception:
        return {"rigido": False, "score": 0.0, "folga_qualitativa": "nenhuma", "sinais": {"motivo": "falha_silenciosa"}, "mensagem": "Falha silenciosa no diagnóstico de rigidez."}



# ============================================================
# V8 — AJUSTE FINO · ETAPA 2 — BORDA QUALIFICADA (PRÉ-CAMADA 4)
# - Observacional / governança legível
# - NÃO altera Modo 6 / TURBO / Bala Humano
# - NÃO decide ataque / volume
# - Classifica "borda interna" vs "borda externa" a partir do pacote (Top N)
# ============================================================

def v8_classificar_borda_qualificada(
    listas: list,
    base_n: int = 10,
    core_presenca_min: float = 0.60,
    quase_delta: float = 0.12,
    max_borda_interna: int = 6,
    universo_min: int = None,
    universo_max: int = None,
    rigidez_info: dict = None,
) -> dict:
    """V8 — Borda Qualificada (Etapa 2).
    Retorna um dict com:
      - core (lista)
      - quase_core (lista)
      - borda_interna (lista) + motivos
      - borda_externa (lista) + motivos
      - meta (base_n, thresholds, rigidez, etc.)
    """
    try:
        if not listas or not isinstance(listas, list):
            return {
                "core": [],
                "quase_core": [],
                "borda_interna": [],
                "borda_externa": [],
                "candidatos": [],
                "meta": {"status": "sem_listas"},
            }

        # base_n conservador
        base_n = int(base_n or 0)
        base_n = max(3, min(base_n, len(listas)))
        base = listas[:base_n]

        # universo (se não vier explícito, tenta session_state)
        umin = universo_min
        umax = universo_max
        if (umin is None or umax is None):
            try:
                umin = st.session_state.get("universo_min")
                umax = st.session_state.get("universo_max")
            except Exception:
                umin, umax = None, None

        # presença por lista (não contagem bruta)
        from collections import Counter
        pres = Counter()
        for lst in base:
            try:
                for p in set(lst):
                    pres[int(p)] += 1
            except Exception:
                continue

        if not pres:
            return {
                "core": [],
                "quase_core": [],
                "borda_interna": [],
                "borda_externa": [],
                "candidatos": [],
                "meta": {"status": "presenca_vazia", "base_n": base_n},
            }

        # thresholds
        cmin = float(core_presenca_min)
        delta = float(quase_delta)
        thr_quase_min = max(0.0, cmin - delta)

        # CORE e QUASE-CORE (por presença)
        core = []
        quase = []
        ratios = {}
        for p, v in pres.items():
            r = float(v) / float(base_n)
            ratios[int(p)] = float(round(r, 4))
            if r >= cmin:
                core.append(int(p))
            elif r >= thr_quase_min:
                quase.append(int(p))

        core = sorted(core)
        quase = sorted(quase)

        # rigidez (se não vier, tenta calcular)
        rig = rigidez_info or {}
        try:
            if not rig:
                rig = v16_diagnostico_rigidez_jeitao(
                    listas=listas,
                    universo_min=umin,
                    universo_max=umax,
                    base_n=base_n,
                    core_presenca_min=float(core_presenca_min),
                )
        except Exception:
            rig = rigidez_info or {}

        rigido = bool(rig.get("rigido", False))
        score_rig = rig.get("score", 0.0)
        folga_qual = rig.get("folga_qualitativa", "nenhuma")

        # geometria simples do CORE (para decidir borda interna vs externa)
        core_min = min(core) if core else None
        core_max = max(core) if core else None
        universo_size = None
        try:
            if (umin is not None) and (umax is not None):
                universo_size = int(umax) - int(umin) + 1
        except Exception:
            universo_size = None

        candidatos = []
        for p in quase:
            r = ratios.get(int(p), 0.0)
            motivos = []

            # 1) presença (sempre)
            motivos.append(f"quase-CORE por presença ({int(round(r*100))}%)")

            # 2) rigidez / compressão (condicional)
            if rigido:
                motivos.append(f"jeitão rígido (score {score_rig})")
                if str(folga_qual).strip().lower() in ("mínima", "minima", "moderada"):
                    motivos.append(f"folga {folga_qual} (alerta)")
            else:
                motivos.append("jeitão não rígido (diagnóstico)")

            # 3) distância do CORE (se disponível)
            dist = None
            if core_min is not None and core_max is not None:
                if int(p) < int(core_min):
                    dist = int(core_min) - int(p)
                elif int(p) > int(core_max):
                    dist = int(p) - int(core_max)
                else:
                    dist = 0
                motivos.append(f"distância do CORE: {dist}")

            # 4) sanidade: evitar "borda externa" que abre universo demais (heurística conservadora)
            externo = False
            if universo_size is not None and dist is not None:
                # se o candidato estiver muito fora do "miolo" do CORE em proporção ao universo, tende a dispersar
                if dist >= int(round(0.28 * float(universo_size))):
                    externo = True
                    motivos.append("muito distante do CORE (risco de dispersão)")

            # 5) qualificador final: "interna" exige presença alta dentro do quase-CORE + compatibilidade com rigidez
            # (não é regra de ataque; é governança legível)
            interna = False
            if not externo:
                if r >= (cmin - (delta * 0.50)):
                    # se jeitão rígido, prioriza borda interna; se não rígido, exige ainda mais presença
                    if rigido:
                        interna = True
                        motivos.append("classificada como BORDA INTERNA (rigidez + presença alta)")
                    else:
                        if r >= (cmin - (delta * 0.25)):
                            interna = True
                            motivos.append("classificada como BORDA INTERNA (presença muito alta)")
                        else:
                            motivos.append("presença boa, mas não suficiente para interna sem rigidez")
                else:
                    motivos.append("presença insuficiente para BORDA INTERNA (fica como externa/observacional)")

            classe = "borda_interna" if interna else "borda_externa"
            candidatos.append({
                "p": int(p),
                "ratio": r,
                "classe": classe,
                "motivos": motivos,
            })

        # ordena candidatos por ratio desc, depois por p
        candidatos = sorted(candidatos, key=lambda d: (-float(d.get("ratio") or 0.0), int(d.get("p") or 0)))

        borda_interna = [d["p"] for d in candidatos if d.get("classe") == "borda_interna"][:int(max_borda_interna)]
        borda_externa = [d["p"] for d in candidatos if d.get("classe") == "borda_externa"]

        # motivos por grupo (compacto)
        motivos_in = {d["p"]: d["motivos"] for d in candidatos if d.get("classe") == "borda_interna" and d.get("p") in borda_interna}
        motivos_ex = {d["p"]: d["motivos"] for d in candidatos if d.get("classe") == "borda_externa"}

        return {
            "core": core,
            "quase_core": quase,
            "borda_interna": borda_interna,
            "borda_externa": borda_externa,
            "motivos_interna": motivos_in,
            "motivos_externa": motivos_ex,
            "candidatos": candidatos,
            "meta": {
                "status": "ok",
                "base_n": base_n,
                "core_presenca_min": cmin,
                "quase_delta": delta,
                "thr_quase_min": round(thr_quase_min, 4),
                "rigido": rigido,
                "score_rigidez": score_rig,
                "folga_qualitativa": folga_qual,
                "universo_min": umin,
                "universo_max": umax,
            },
        }

    except Exception:
        return {
            "core": [],
            "quase_core": [],
            "borda_interna": [],
            "borda_externa": [],
            "candidatos": [],
            "meta": {"status": "falha_silenciosa"},
        }


# ============================================================
# Estilos globais — preservando jeitão V14-FLEX + V15.6 MAX
# ============================================================
st.markdown(
    """
    <style>
    .big-title { font-size: 32px; font-weight: bold; }
    .sub-title { font-size: 22px; font-weight: bold; margin-top: 25px; }
    .danger { color: red; font-weight: bold; }
    .success { color: green; font-weight: bold; }
    .warning { color: orange; font-weight: bold; }
    .gray-text { color: #888; }
    .info-box {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-left: 4px solid #4c8bf5;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# BLINDAGEM FINAL — SANIDADE DE UNIVERSO (V16)
# Aplica automaticamente o universo real do histórico
# em qualquer lista de previsão antes do uso operacional
# ------------------------------------------------------------
# ============================================================
# V16 — ÓRBITA: listas de interceptação automática (E2)
# (sem painel novo; muda listas quando justificado)
# ============================================================

def v16_gerar_listas_interceptacao_orbita(info_orbita: dict,
                                         universo_min: int,
                                         universo_max: int,
                                         n_carro: int,
                                         qtd: int = 4,
                                         seed: int = 0):
    """Gera listas densas adicionais quando ORBITA entra em E2.
    Objetivo: aumentar interseção e repetição controlada sem explodir universo.
    Retorna uma lista de listas (cada uma com n_carro passageiros).
    """
    import random

    try:
        qtd = int(qtd)
    except Exception:
        qtd = 4
    qtd = max(0, min(12, qtd))

    if qtd <= 0:
        return []

    rng = random.Random(int(seed) + 9173)

    # âncoras / candidatos principais (se não vierem, recalcula a partir das listas do pacote)
    anchors = list(info_orbita.get("ancoras") or info_orbita.get("anchors") or [])
    pool_top = list(info_orbita.get("top_passageiros") or info_orbita.get("top_passengers") or [])

    # fallback robusto: usa os passageiros mais frequentes no pacote
    if not anchors or not pool_top:
        listas = info_orbita.get("listas") or info_orbita.get("listas_top") or info_orbita.get("listas_topN") or info_orbita.get("listas_top_n") or []
        freq = {}
        for L in listas:
            for x in L:
                if isinstance(x, int):
                    freq[x] = freq.get(x, 0) + 1
        ordenados = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        pool_top = [k for k, v in ordenados[:max(12, n_carro * 3)]]
        anchors = [k for k, v in ordenados[:max(3, min(8, n_carro + 1))]]

    # garante domínio
    anchors = [x for x in anchors if isinstance(x, int) and universo_min <= x <= universo_max]
    pool_top = [x for x in pool_top if isinstance(x, int) and universo_min <= x <= universo_max]

    # fallback final: universo inteiro (último recurso)
    if not pool_top:
        pool_top = list(range(universo_min, universo_max + 1))

    # modelo: base fixa (2–4 âncoras) + completar com top, preservando diversidade mínima
    # densidade-alvo: mais forte em E2 (pelo menos 3 âncoras se possível)
    base_k = 3 if len(anchors) >= 3 else max(1, min(2, len(anchors)))
    if info_orbita.get("gradiente") in ("G2", "G3"):
        base_k = min(max(3, base_k), max(1, min(4, len(anchors))))

    geradas = []
    vistos = set()

    for i in range(qtd * 3):  # tenta mais para evitar duplicatas
        L = []

        # 1) âncoras (fixa)
        if anchors:
            picks = anchors[:]  # copia
            rng.shuffle(picks)
            L.extend(picks[:base_k])

        # 2) completar com top
        if len(L) < n_carro:
            picks = pool_top[:]
            rng.shuffle(picks)
            for x in picks:
                if x not in L:
                    L.append(x)
                if len(L) >= n_carro:
                    break

        # 3) completar (se ainda faltar) com universo
        if len(L) < n_carro:
            uni = list(range(universo_min, universo_max + 1))
            rng.shuffle(uni)
            for x in uni:
                if x not in L:
                    L.append(x)
                if len(L) >= n_carro:
                    break

        if len(L) != n_carro:
            continue

        L = sorted(L)
        key = tuple(L)
        if key in vistos:
            continue
        vistos.add(key)
        geradas.append(L)

        if len(geradas) >= qtd:
            break

    return geradas



def v16_blindar_ultima_previsao_universo():
    """
    Blindagem estrutural:
    - Garante que nenhuma lista contenha passageiros fora do universo real
    - Usa universo_min / universo_max detectados no carregamento
    - Atua como última barreira antes do uso operacional
    """

    if "ultima_previsao" not in st.session_state:
        return

    listas = st.session_state.get("ultima_previsao")
    if not listas or not isinstance(listas, list):
        return

    umin = st.session_state.get("universo_min")
    umax = st.session_state.get("universo_max")

    if umin is None or umax is None:
        return

    listas_sanas = []

    for lista in listas:
        if not isinstance(lista, (list, tuple)):
            continue

        lista_filtrada = [
            int(x) for x in lista
            if isinstance(x, (int, np.integer)) and umin <= int(x) <= umax
        ]

        if len(lista_filtrada) == len(lista):
            listas_sanas.append(lista)
        else:
            # se houve ajuste, preserva ordem e tamanho quando possível
            lista_corrigida = lista_filtrada[:len(lista)]
            if len(lista_corrigida) == len(lista):
                listas_sanas.append(lista_corrigida)

    if listas_sanas:
        st.session_state["ultima_previsao"] = listas_sanas

# ============================================================
# Sessão Streamlit — persistência para V15.7 MAX
# ============================================================

# Inicialização de estado
if "historico_df" not in st.session_state:
    st.session_state["historico_df"] = None

if "ultima_previsao" not in st.session_state:
    st.session_state["ultima_previsao"] = None

if "sentinela_kstar" not in st.session_state:
    st.session_state["sentinela_kstar"] = None

if "diagnostico_risco" not in st.session_state:
    st.session_state["diagnostico_risco"] = None

if "n_alvo" not in st.session_state:
    st.session_state["n_alvo"] = None




if "aps_postura_selo" not in st.session_state:
    st.session_state["aps_postura_selo"] = None

if "aps_postura_titulo" not in st.session_state:
    st.session_state["aps_postura_titulo"] = None

if "aps_postura_msg" not in st.session_state:
    st.session_state["aps_postura_msg"] = None



# ============================================================
# DETECÇÃO CANÔNICA DE n_alvo (PASSAGEIROS REAIS DA RODADA)
# REGRA FIXA:
# - Última coluna SEMPRE é k
# - Todas as colunas p* anteriores são passageiros
# - n_alvo é definido pela ÚLTIMA SÉRIE VÁLIDA
# ============================================================

def detectar_n_alvo(historico_df):
    if historico_df is None or historico_df.empty:
        return None

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    if not col_pass:
        return None

    ultima_linha = historico_df[col_pass].iloc[-1]
    return int(ultima_linha.dropna().shape[0])


# Atualização automática de n_alvo
if st.session_state.get("historico_df") is not None:
    st.session_state["n_alvo"] = detectar_n_alvo(
        st.session_state["historico_df"]
    )


# ============================================================
# 🧪 SÉRIE SUFICIENTE (SS) — V16 PREMIUM (INFORMATIVO)
# ------------------------------------------------------------
# Função:
# - Explicitar ao operador se já há base mínima para confiar em leituras EX-POST
#   (Ritmo/Dança, V9, Parabólica, Memórias), sem bloquear execução e sem tocar Camada 4.
# - SS NÃO decide, NÃO prevê, NÃO muda listas e NÃO altera volumes.
# Fonte de verdade:
# - snapshots do Replay Progressivo: st.session_state["snapshot_p0_canonic"] (mapa k -> snapshot)
# Critério (auditável, simples, sem magia):
# - mínimo de janelas (ks) registradas
# - mínimo de janelas com EX-POST disponível (k+1 existente no histórico FULL)
# ============================================================

SS_MIN_KS = 5
SS_MIN_EXPOST = 5

def v16_calcular_ss(_df_full_safe: Optional[pd.DataFrame], snapshots_map: Optional[dict]) -> dict:
    """Calcula o status de SS (informativo).
    Retorna dict auditável: {status, ks_total, ks_expost, motivos, Ws}.

    - ks_total: quantidade de snapshots (janelas) registrados
    - ks_expost: quantos snapshots têm alvo ex-post disponível (k < len(_df_full_safe))
    - Ws: calibração Parabólica (se já calculada), apenas para contextualizar
    """
    snaps = snapshots_map if isinstance(snapshots_map, dict) else {}
    ks = []
    for k in snaps.keys():
        try:
            ks.append(int(k))
        except Exception:
            continue
    ks = sorted(list(set(ks)))
    ks_total = int(len(ks))

    n_full = int(len(_df_full_safe) if _df_full_safe is not None else 0)
    ks_expost = 0
    if n_full > 0:
        for k in ks:
            # Se a janela é C1..Ck, o alvo ex-post mínimo é C(k+1) => existe se k < N
            if int(k) < int(n_full):
                ks_expost += 1

    motivos = []
    if ks_total < int(SS_MIN_KS):
        motivos.append(f"poucas_janelas_registradas ({ks_total} < {SS_MIN_KS})")
    if ks_expost < int(SS_MIN_EXPOST):
        motivos.append(f"poucas_janelas_com_expost ({ks_expost} < {SS_MIN_EXPOST})")

    # Contexto Parabólica (se já existir em sessão)
    gov = st.session_state.get("parabola_gov")
    Ws = (gov or {}).get("Ws") if isinstance(gov, dict) else {}
    try:
        Ws = {k: int(v) for k, v in (Ws or {}).items()}
    except Exception:
        Ws = {}

    status = (ks_total >= int(SS_MIN_KS)) and (ks_expost >= int(SS_MIN_EXPOST))

    return {
        "status": bool(status),
        "ks_total": int(ks_total),
        "ks_expost": int(ks_expost),
        "motivos": motivos,
        "Ws": Ws,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
    }

def v16_render_bloco_ss(ss_info: dict):
    """Renderiza o bloco SS de forma visível e consistente (sem criar painel novo)."""
    info = ss_info if isinstance(ss_info, dict) else {}
    ok = bool(info.get("status"))
    ks_total = int(info.get("ks_total") or 0)
    ks_expost = int(info.get("ks_expost") or 0)
    motivos = info.get("motivos") or []
    Ws = info.get("Ws") or {}

    st.markdown("### 🧪 Série Suficiente (SS)")
    st.caption("Condição de estabilidade (informativa). Não bloqueia execução. Não altera listas. Não mexe na Camada 4.")

    if ok:
        st.success(f"✅ SS ATINGIDA — base mínima presente. Janelas: {ks_total} · com EX-POST: {ks_expost}.")
    else:
        st.warning(f"⚠️ SS AINDA NÃO ATINGIDA — leituras podem variar. Janelas: {ks_total} · com EX-POST: {ks_expost}.")
        if motivos:
            st.write("**Motivos:**")
            for m in motivos[:6]:
                st.write(f"- {m}")

    # Contexto Parabólica (se já houver)
    if isinstance(Ws, dict) and Ws:
        try:
            st.caption(f"Contexto Parabólica (calibração Ws): short={Ws.get('short',0)} · mid={Ws.get('mid',0)} · long={Ws.get('long',0)}")
        except Exception:
            pass

# ============================================================
# V16 PREMIUM — INFRAESTRUTURA UNIVERSAL
# (REGRAS CANÔNICAS + ORÇAMENTO CONDICIONADO)
# ============================================================

# -----------------------------
# REGRA CANÔNICA: LISTAS >= n_real
# -----------------------------

# ============================================================
# 🧠 V16 — MEMÓRIA ESTRUTURAL DO RESPIRÁVEL (SEM_RITMO) — pré-C4
# - Observacional, auditável, reversível
# - NÃO altera Camada 4
# - NÃO aprende online (atualiza apenas quando um Snapshot P0 é registrado)
# - Atua apenas (se habilitada) quando: Postura=RESPIRÁVEL e Ritmo=SEM_RITMO
# ============================================================

ME_LOOKBACK = 25
ME_TOP_N = 8
ME_MIN_JANELAS = 5  # base mínima para produzir ranking útil (auditável)

def v16_me_build_from_snapshots(snapshots_map: Optional[dict], lookback: int = ME_LOOKBACK, top_n: int = ME_TOP_N) -> dict:
    """Constrói a Memória Estrutural a partir dos Snapshots P0 canônicos.
    Não usa frequência bruta isolada: usa *efeito estrutural* via proxies já existentes nos snapshots (rigidez/folga e camadas V8).
    Retorna dict auditável (não decide):
      {
        "ok": bool,
        "base": {"janelas": int, "ks": [...], "obs": str},
        "sufocadores": [int...],
        "scores": {p: float},
        "debug": {...}
      }
    """
    try:
        if not isinstance(snapshots_map, dict) or not snapshots_map:
            return {"ok": False, "base": {"janelas": 0, "ks": [], "obs": "sem_snapshots"}, "sufocadores": [], "scores": {}, "debug": {"motivo": "sem_snapshots"}}

        # ordenar ks e pegar lookback
        ks_all = []
        for k in snapshots_map.keys():
            try:
                ks_all.append(int(k))
            except Exception:
                pass
        ks_all = sorted(set(ks_all))
        if not ks_all:
            return {"ok": False, "base": {"janelas": 0, "ks": [], "obs": "ks_invalidos"}, "sufocadores": [], "scores": {}, "debug": {"motivo": "ks_invalidos"}}

        ks_used = ks_all[-int(lookback):] if lookback and len(ks_all) > int(lookback) else ks_all[:]
        snaps = []
        for k in ks_used:
            s = snapshots_map.get(k)
            if isinstance(s, dict):
                snaps.append((k, s))
        if not snaps:
            return {"ok": False, "base": {"janelas": 0, "ks": ks_used, "obs": "snapshots_vazios"}, "sufocadores": [], "scores": {}, "debug": {"motivo": "snapshots_vazios"}}

        # Proxies de dano estrutural (pré-C4) — auditáveis
        def _damage_proxy(snap: dict) -> float:
            try:
                v8 = (snap.get("snap_v8") or {}) if isinstance(snap, dict) else {}
                meta = (v8.get("meta") or {}) if isinstance(v8, dict) else {}
                score_rig = _pc_safe_float(meta.get("score_rigidez"), 0.0) or 0.0
                rigido = bool(meta.get("rigido"))
                folga = str(meta.get("folga_qualitativa") or "").strip().lower()
                # penaliza "nenhuma" folga (tende a compressão) e rigidez marcada
                dmg = float(score_rig)
                if rigido:
                    dmg += 0.20
                if folga in ("nenhuma", "0", "zero"):
                    dmg += 0.10
                # pacote muito pequeno costuma ser mais comprimido (proxy leve)
                up = _pc_safe_float(snap.get("universo_pacote_len"), None)
                if up is not None and up <= 12:
                    dmg += 0.05
                # clamp
                if dmg < 0:
                    dmg = 0.0
                if dmg > 1.5:
                    dmg = 1.5
                return dmg
            except Exception:
                return 0.0

        # Dominância por camada V8 (core/quase/bordas)
        def _dom_weight(layer: str) -> float:
            if layer == "core":
                return 1.0
            if layer == "quase_core":
                return 0.70
            if layer == "borda_interna":
                return 0.40
            if layer == "borda_externa":
                return 0.20
            return 0.0

        # Acúmulos
        sum_dom = {}
        sum_w = {}
        count = {}
        dmg_list = []

        for k, snap in snaps:
            dmg = _damage_proxy(snap)
            dmg_list.append(dmg)

            v8 = snap.get("snap_v8") or {}
            if not isinstance(v8, dict):
                continue

            for layer in ("core", "quase_core", "borda_interna", "borda_externa"):
                arr = v8.get(layer) or []
                if not isinstance(arr, list):
                    continue
                w_dom = _dom_weight(layer)
                for p in arr:
                    try:
                        pi = int(p)
                    except Exception:
                        continue
                    # presença conta sempre; dominância pesa por dano (efeito estrutural)
                    count[pi] = count.get(pi, 0) + 1
                    sum_dom[pi] = sum_dom.get(pi, 0.0) + (w_dom * dmg)
                    sum_w[pi] = sum_w.get(pi, 0.0) + dmg

        janelas = len(snaps)
        if janelas < 1:
            return {"ok": False, "base": {"janelas": 0, "ks": ks_used, "obs": "snapshots_invalidos"}, "sufocadores": [], "scores": {}, "debug": {"motivo": "snapshots_invalidos"}}

        # Se dano total é 0 (sem proxies), ainda assim produz ranking por presença em camadas, mas marca obs
        scores = {}
        for p, c in count.items():
            w = sum_w.get(p, 0.0)
            if w > 0:
                avg_dom = sum_dom.get(p, 0.0) / max(w, 1e-9)
            else:
                # fallback: usa presença relativa como proxy (auditável)
                avg_dom = float(c) / float(janelas)
            # índice final (simples, auditável): dominância média * presença relativa
            pres_rel = float(c) / float(janelas)
            idx = float(avg_dom) * pres_rel
            scores[p] = idx

        # top-N sufocadores
        suf = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
        suf = [int(p) for p in suf[: int(top_n)]]

        debug = {
            "lookback": int(lookback),
            "top_n": int(top_n),
            "dmg_min": min(dmg_list) if dmg_list else None,
            "dmg_max": max(dmg_list) if dmg_list else None,
            "dmg_media": (sum(dmg_list) / len(dmg_list)) if dmg_list else None,
            "janelas_total": int(janelas),
        }

        ok = bool(janelas >= 1 and len(suf) > 0)
        return {"ok": ok, "base": {"janelas": int(janelas), "ks": ks_used, "obs": "pre-C4 | efeito estrutural (rigidez/folga/camadas V8)"}, "sufocadores": suf, "scores": scores, "debug": debug}
    except Exception as e:
        return {"ok": False, "base": {"janelas": 0, "ks": [], "obs": "falha"}, "sufocadores": [], "scores": {}, "debug": {"motivo": f"falha_build: {e}"}}

def v16_me_status(postura: str, ritmo_global: str, me_enabled: bool, ss_info: Optional[dict], me_info: Optional[dict]) -> dict:
    """Define estado canônico da Memória (não decide; só governa a leitura)."""
    postura = str(postura or "").strip().upper()
    ritmo = str(ritmo_global or "").strip().upper()
    enabled = bool(me_enabled)

    if not enabled:
        return {"status": "DESLIGADA", "motivo": "me_enabled_false"}

    if postura != "RESPIRÁVEL":
        return {"status": "INATIVA", "motivo": f"postura_{postura or 'N/D'}"}

    if ritmo != "SEM_RITMO":
        return {"status": "INATIVA", "motivo": f"ritmo_{ritmo or 'N/D'}"}

    base_j = int(((me_info or {}).get("base") or {}).get("janelas") or 0)
    if base_j < ME_MIN_JANELAS:
        return {"status": "INSUFICIENTE", "motivo": f"poucas_janelas ({base_j} < {ME_MIN_JANELAS})"}

    ss_ok = bool((ss_info or {}).get("status"))
    if not ss_ok:
        return {"status": "BASE_FRACA", "motivo": "SS_nao_atingida"}

    return {"status": "ATIVA", "motivo": "ok"}

def v16_me_update_auto(_df_full_safe: Optional[pd.DataFrame], snapshots_map: Optional[dict], postura: Optional[str] = None, ritmo_global: Optional[str] = None) -> dict:
    """Atualiza Memória Estrutural (Jogador B) automaticamente quando um snapshot entra.
    Não dispara execução; só atualiza st.session_state.
    """
    # toggle padrão (mantém compatibilidade: ON por padrão, mas operador pode desligar)
    if "me_enabled" not in st.session_state:
        st.session_state["me_enabled"] = True

    me_enabled = bool(st.session_state.get("me_enabled", True))
    me_info = v16_me_build_from_snapshots(snapshots_map=snapshots_map, lookback=ME_LOOKBACK, top_n=ME_TOP_N)
    st.session_state["me_info"] = me_info

    # usa SS já calculado quando existir; senão calcula rápido
    ss_info = st.session_state.get("ss_info")
    if not isinstance(ss_info, dict):
        try:
            ss_info = v16_calcular_ss(_df_full_safe=_df_full_safe, snapshots_map=snapshots_map)
        except Exception:
            ss_info = {"status": False, "ks_total": 0, "ks_expost": 0, "motivos": ["ss_indisponivel"]}
    st.session_state["ss_info"] = ss_info
    st.session_state["ss_status"] = "ATINGIDA" if ss_info.get("status") else "NAO_ATINGIDA"

    # postura/ritmo — se não vier, pega do session_state
    pst = postura or st.session_state.get("postura_estado") or ""
    rg = ritmo_global or st.session_state.get("ritmo_global_expost") or (st.session_state.get("ritmo_danca_info") or {}).get("ritmo_global") or "N/D"

    me_st = v16_me_status(postura=str(pst), ritmo_global=str(rg), me_enabled=me_enabled, ss_info=ss_info, me_info=me_info)
    st.session_state["me_status"] = me_st.get("status")
    st.session_state["me_status_info"] = me_st
    st.session_state["me_last_update"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return {"me_info": me_info, "me_status": me_st, "ss_info": ss_info}

def v16_render_bloco_me(me_info: Optional[dict], me_status_info: Optional[dict], ss_info: Optional[dict]):
    """Render do bloco informativo da Memória Estrutural (SEM_RITMO)."""
    st.markdown("### 🧠 Memória Estrutural (SEM_RITMO)")
    st.caption("Pré-C4 · Observacional · Auditável. Não altera Camada 4. Não aprende online (atualiza só ao registrar Snapshot P0).")

    st.checkbox("Ativar Memória Estrutural (SEM_RITMO)", key="me_enabled", value=bool(st.session_state.get("me_enabled", True)))

    stt = (me_status_info or {}).get("status", "N/D")
    mot = (me_status_info or {}).get("motivo", "")
    base = (me_info or {}).get("base") or {}
    j = int(base.get("janelas") or 0)

    ss_ok = bool((ss_info or {}).get("status"))
    st.markdown(f"**Status:** {stt}  \\n**Motivo:** {mot}  \\n**Base:** {j} janelas | SS: {'ATINGIDA' if ss_ok else 'NÃO ATINGIDA'}")

    suf = list((me_info or {}).get("sufocadores") or [])
    if suf:
        st.write("**Top sufocadores (efeito estrutural):** " + ", ".join(map(str, suf)))
    else:
        st.info("Ainda não há sufocadores suficientes para exibir (base curta ou dados insuficientes).")

    dbg = (me_info or {}).get("debug") or {}
    if dbg:
        st.json({"debug": dbg, "obs": base.get("obs")})

def v16_calc_lce_b(
    ss_info: Optional[dict],
    ritmo_info: Optional[dict],
    df_eval: Optional[pd.DataFrame],
    snap_last: Optional[dict],
    anti_idx_detectados: Optional[list] = None,
) -> dict:
    """LCE‑B (Leitura Canônica de E1 para Jogador B) — painel silencioso.

    Regras canônicas:
    - Pré‑C4, observacional, auditável.
    - NÃO decide nada, NÃO altera listas, NÃO altera Camada 4.
    - NÃO cria métricas fundamentais novas: apenas combina leituras já existentes.
    """
    ss_ok = bool((ss_info or {}).get("status"))
    ritmo_global = str((ritmo_info or {}).get("ritmo_global") or (st.session_state.get("ritmo_global_expost") or "N/D")).strip()

    # Rigidez / folga (já existente no Snapshot P0 V8)
    rig_score = None
    rig_flag = None
    core_sz = None
    try:
        meta = (((snap_last or {}).get("snap_v8") or {}).get("meta") or {})
        rig_score = float(meta.get("score_rigidez")) if meta.get("score_rigidez") is not None else None
        rig_flag = bool(meta.get("rigido")) if meta.get("rigido") is not None else None
        core = ((snap_last or {}).get("snap_v8") or {}).get("core") or []
        core_sz = int(len(core))
    except Exception:
        rig_score, rig_flag, core_sz = None, None, None

    # Métricas do df_eval (já existente): melhor acerto e trave (fora_perto/fora_longe)
    avg_best = None
    rate_4p = 0.0
    rate_5p = 0.0
    p3_hits = 0
    p3_rate = 0.0
    trave_ratio = None
    total_rows = 0
    try:
        if df_eval is not None and len(df_eval) > 0:
            total_rows = int(len(df_eval))
            # melhor acerto por alvo (já calculado no replay)
            c1 = "best_acerto_alvo_1"
            c2 = "best_acerto_alvo_2"
            bests = []
            if c1 in df_eval.columns:
                bests.extend([x for x in df_eval[c1].tolist() if isinstance(x, (int, float))])
            if c2 in df_eval.columns:
                bests.extend([x for x in df_eval[c2].tolist() if isinstance(x, (int, float))])
            if bests:
                avg_best = float(sum(bests) / max(1, len(bests)))
                rate_4p = float(sum(1 for b in bests if b >= 4) / len(bests))
                rate_5p = float(sum(1 for b in bests if b >= 5) / len(bests))
                p3_hits = int(sum(1 for b in bests if b >= 3))
                p3_rate = float(p3_hits / max(1, len(bests)))

            # trave/perto/longe (já calculado no replay)
            fp = 0
            fl = 0
            for col in ["fora_perto_1", "fora_perto_2"]:
                if col in df_eval.columns:
                    fp += int(sum(1 for x in df_eval[col].tolist() if x in (1, True)))
            for col in ["fora_longe_1", "fora_longe_2"]:
                if col in df_eval.columns:
                    fl += int(sum(1 for x in df_eval[col].tolist() if x in (1, True)))
            denom = fp + fl
            if denom > 0:
                trave_ratio = float(fp / denom)
    except Exception:
        pass

    # Estado STE‑E1 (silencioso): nasce quando SS ok, ritmo ainda N/D, e já há sinais de "trave" + piso 4/6
    # (combinação de leituras existentes; nenhum sensor novo)
    ste_e1 = False
    try:
        if ss_ok and (ritmo_global == "N/D"):
            rig_ok = (rig_score is not None and rig_score >= 0.70) or (rig_flag is True) or (core_sz is not None and core_sz <= 1)
            trv_ok = (trave_ratio is not None and trave_ratio >= 0.55)
            piso4 = (rate_4p >= 0.20)  # 1 em 5 alvos/rodadas batendo 4/6 dentro do pacote
            ste_e1 = bool(rig_ok and trv_ok and piso4)
    except Exception:
        ste_e1 = False

    # ZEE‑B

    # Pré‑E1 (silencioso): sinal invisível antes do primeiro 4 consistente
    pre_e1 = False
    try:
        if ss_ok and (total_rows >= 10):
            p3_ok = (p3_rate >= 0.70)
            trv_ok2 = (trave_ratio is not None and trave_ratio <= 0.55)
            pre_e1 = bool((not ste_e1) and p3_ok and trv_ok2)
    except Exception:
        pre_e1 = False

    # ETAPA 2.3 — MICRO‑E1 (silencioso): primeiro micro‑sinal estatístico antes do 4 consistente
    # Regra: NÃO cria sensor novo; apenas usa o df_eval (já existente) em janela móvel.
    micro_e1 = False
    p3_rate_w = None
    trave_ratio_w = None
    zero_hit_rate_w = None
    w_used = None
    try:
        if ss_ok and df_eval is not None and len(df_eval) > 0:
            W = 12  # janela mínima canônica (evita ruído)
            w_used = int(min(W, len(df_eval)))
            dfw = df_eval.tail(w_used).copy()

            # bests na janela (alvos 1 e 2)
            bests_w = []
            c1 = "best_acerto_alvo_1"
            c2 = "best_acerto_alvo_2"
            if c1 in dfw.columns:
                bests_w.extend([x for x in dfw[c1].tolist() if isinstance(x, (int, float))])
            if c2 in dfw.columns:
                bests_w.extend([x for x in dfw[c2].tolist() if isinstance(x, (int, float))])

            if bests_w:
                p3_rate_w = float(sum(1 for b in bests_w if b >= 3) / len(bests_w))
                zero_hit_rate_w = float(sum(1 for b in bests_w if b == 0) / len(bests_w))

            # trave_ratio na janela (fora_perto vs fora_longe)
            fpw = 0
            flw = 0
            for col in ["fora_perto_1", "fora_perto_2"]:
                if col in dfw.columns:
                    fpw += int(sum(1 for x in dfw[col].tolist() if x in (1, True)))
            for col in ["fora_longe_1", "fora_longe_2"]:
                if col in dfw.columns:
                    flw += int(sum(1 for x in dfw[col].tolist() if x in (1, True)))
            denomw = fpw + flw
            if denomw > 0:
                trave_ratio_w = float(fpw / denomw)

            # Gatilho micro (canônico, mínimo): 3+ sustentado e trave reduzindo
            # (não exige 4+ ainda, mas exige sustentação)
            if (w_used is not None and w_used >= W) and (p3_rate_w is not None) and (trave_ratio_w is not None):
                T_P3 = 0.25
                T_TRAVE = 0.45
                micro_e1 = bool((not ste_e1) and (not pre_e1) and (p3_rate_w >= T_P3) and (trave_ratio_w <= T_TRAVE))
    except Exception:
        micro_e1 = False
    # ZEE‑B) (Zona de Efeito Estatístico do B) — ainda silenciosa nesta fase
    # OFF: sem SS ou sem df_eval
    # STE_E1: sinal inicial (não é "ativo", não decide nada)
    if (not ss_ok) or (total_rows <= 0):
        zee_state = "OFF"
    else:
        if ste_e1:
            zee_state = "STE_E1"
        elif pre_e1:
            zee_state = "PRE_E1"
        elif micro_e1:
            zee_state = "MICRO_E1"
        else:
            zee_state = "OBS"

    # B1 sugerido (Base + Anti‑âncora) — apenas quando já existe anti‑âncora detectada por métricas existentes
    b1 = "Base (Top) apenas"
    anti_idx = []
    try:
        anti_idx = list(anti_idx_detectados or [])
        anti_idx = [int(x) for x in anti_idx if str(x).strip().isdigit()]
    except Exception:
        anti_idx = []
    if anti_idx:
        # indices são 0‑based no código; exibimos como L1..Ln para o operador
        lbls = [f"L{int(i)+1}" for i in anti_idx[:3]]
        b1 = "Base + Anti‑âncora (" + ", ".join(lbls) + ")"

    # prontidão 6E (tentativa de cravar) — CANÔNICO: apenas informativo, nunca autorização automática
    pront_6e = "NÃO"
    try:
        if ss_ok and ste_e1 and (rate_5p >= 0.10) and (trave_ratio is not None and trave_ratio >= 0.60):
            pront_6e = "PRÉ (observacional)"
    except Exception:
        pass

    # ------------------------------
    # V16_CURV_SUST_DETECTOR — curvatura sustentada a partir do próprio df_eval
    # ------------------------------
    try:
        _curv_info = v16_detector_curvatura_sustentada_df_eval(
            df_eval,
            w_smooth=5,
            L_sust=4,
            eps_mult=0.60,
            lookback_max=None,
        )
    except Exception as _e_curv_calc:
        _curv_info = {"ok": False, "motivo": f"erro: {_e_curv_calc}"}

    gatilho_curvatura = bool(
        isinstance(_curv_info, dict)
        and _curv_info.get("ok")
        and _curv_info.get("curvatura_sustentada_recente")
        and _curv_info.get("troca_sinal_recente")
    )

    return {
        "estado_zee_b": zee_state,
        "gatilho_curvatura": gatilho_curvatura,
        "curvatura_info": _curv_info,
        "ste_e1": bool(ste_e1),
        "b1_sugerido": b1,
        "prontidao_6e": pront_6e,
        "_debug": {
            "ss_ok": ss_ok,
            "ritmo_global": ritmo_global,
            "rig_score": rig_score,
            "rig_flag": rig_flag,
                "pre4_gate": bool((st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("pre4_gate", False)),
                "pre4_motivo": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("pre4_motivo", "N/A"),
                "gamma_rank": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("gamma_rank", None),
                "gamma_gap": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("gamma_gap", None),
                "gamma_gap_norm": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("gamma_gap_norm", None),
                "rank_n": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("rank_n", None),
                "thr_min": (st.session_state.get("bloco_c_fase6_dir_diag") or {}).get("thr_min", None),

            "core_sz": core_sz,
            "avg_best": avg_best,
                "avg_best_from_dist": avg_best_from_dist,
            "rate_4p": rate_4p,
            "rate_5p": rate_5p,
            "p3_hits": p3_hits,
            "p3_rate": p3_rate,
            "pre_e1": bool(pre_e1),
            "micro_e1": bool(micro_e1),
            "p3_rate_w": p3_rate_w,
            "trave_ratio_w": trave_ratio_w,
            "zero_hit_rate_w": zero_hit_rate_w,
            "w_used": w_used,
            "trave_ratio": trave_ratio,
            "rows_eval": total_rows,
            "anti_idx_detectados": anti_idx,
        },
    }




# ============================================================
# V16_CURV_SUST_DETECTOR — Detector matemático de Curvatura Sustentada (gatilho real de ataque)
# (não bifurca / não decide por você; apenas mede e torna legível)
# ============================================================
def v16_detector_curvatura_sustentada_df_eval(
    df_eval,
    w_smooth: int = 5,
    L_sust: int = 4,
    eps_mult: float = 0.60,
    lookback_max: int | None = None,
):
    """
    Extrai o 'relógio geométrico' do Predicar a partir do df_eval (prova objetiva),
    usando o melhor acerto por alvo (best_acerto_geral) e sua curvatura discreta.

    Conceitos (discretos):
      b[k]   = suavização (rolling mean) do best_acerto_geral
      d1[k]  = Δ b[k]  (derivada 1)
      d2[k]  = Δ² b[k] (curvatura / derivada 2)

    Detector de CURVATURA SUSTENTADA:
      - abs(d2) pequeno por L_sust passos (corredor quase-reto)
      - e a derivada muda de sinal (Δb sai de <=0 para >0) perto do corredor

    Retorna um dict com:
      - estado recente (d1/d2)
      - flags (sustentada, troca_sinal)
      - distância desde o último 4 (se existir)
      - estimativa histórica do atraso típico até um 4 após o corredor
    """
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        return {"ok": False, "motivo": "numpy/pandas indisponível"}

    if df_eval is None or len(df_eval) == 0:
        return {"ok": False, "motivo": "df_eval vazio"}

    df = df_eval.copy()

    # Garantir colunas canônicas (best_acerto_geral)
    if "best_acerto_geral" not in df.columns:
        if ("best_acerto_alvo_1" in df.columns) and ("best_acerto_alvo_2" in df.columns):
            df["best_acerto_geral"] = _np.maximum(
                _pd.to_numeric(df["best_acerto_alvo_1"], errors="coerce").fillna(0.0).values,
                _pd.to_numeric(df["best_acerto_alvo_2"], errors="coerce").fillna(0.0).values,
            )
        else:
            return {"ok": False, "motivo": "coluna best_acerto_geral ausente e não há alvo_1/alvo_2 para derivar"}

    # Ordenação temporal: preferir k_base; fallback para índice original
    if "k_base" in df.columns:
        df = df.sort_values("k_base").reset_index(drop=True)
        eixo = "k_base"
    else:
        df = df.reset_index(drop=True)
        eixo = "index"

    # Lookback opcional (para reduzir custo em históricos gigantes)
    if lookback_max is not None and len(df) > int(lookback_max):
        df = df.iloc[-int(lookback_max):].reset_index(drop=True)

    best = _pd.to_numeric(df["best_acerto_geral"], errors="coerce").fillna(0.0)
    b = best.rolling(int(max(1, w_smooth)), min_periods=1).mean()

    d1 = b.diff().fillna(0.0)
    d2 = d1.diff().fillna(0.0)

    # Epsilon robusto baseado em MAD de d2 (evita depender de escala fixa)
    med = float(_np.nanmedian(d2.values)) if len(d2) else 0.0
    mad = float(_np.nanmedian(_np.abs(d2.values - med))) if len(d2) else 0.0
    sigma_rob = 1.4826 * mad
    eps = float(max(1e-9, eps_mult * (sigma_rob + 1e-9)))

    # Flag: curvatura sustentada nos últimos L_sust passos
    L = int(max(2, L_sust))
    tail_d2 = d2.tail(L)
    sust_recente = bool((tail_d2.abs() <= eps).all()) if len(tail_d2) == L else False

    # Troca de sinal (Δb: <=0 -> >0) recente, preferencialmente dentro do corredor
    d1_prev = d1.shift(1).fillna(0.0)
    troca_sinal = (d1_prev <= 0.0) & (d1 > 0.0)
    troca_sinal_idx = df.index[troca_sinal.fillna(False)].tolist()

    # Distância desde o último 4 real (best>=4)
    idx_4 = df.index[(best >= 4.0).fillna(False)].tolist()
    if len(idx_4) > 0:
        last_4_i = int(idx_4[-1])
        dist_desde_4 = int(len(df) - 1 - last_4_i)
    else:
        last_4_i = None
        dist_desde_4 = None

    # Estimar o "relógio geométrico":
    # Para cada 4 real, encontrar o último corredor (d2 ~ 0 por L) com troca de sinal antes dele,
    # e medir quantos passos depois veio o 4.
    atraso_list = []
    corredor_list = []

    # máscara de corredor (d2 ~ 0 por L consecutivos)
    is_flat = (d2.abs() <= eps).fillna(False)
    # rolling all-true por janela L
    flat_run = is_flat.rolling(L, min_periods=L).apply(lambda x: 1.0 if (x == 1.0).all() else 0.0).fillna(0.0) > 0.5

    for i4 in idx_4:
        # procurar para trás um ponto i0 tal que:
        # - flat_run em i0 (significa que termina um corredor de L)
        # - houve troca de sinal em algum ponto dentro dos últimos L..2L antes de i0 (robusto)
        search_start = max(0, int(i4) - 400)  # buffer local (não precisa varrer o mundo todo)
        candidates = [i for i in range(search_start, int(i4)) if bool(flat_run.iloc[i])]
        if not candidates:
            continue
        # pegar o mais recente candidato antes do 4
        i0 = int(candidates[-1])

        # checar troca de sinal próxima do corredor
        # janela: [i0-2L, i0] (inclui o corredor)
        j0 = max(0, i0 - 2*L)
        has_flip = False
        flip_at = None
        for j in range(j0, i0 + 1):
            if j < len(troca_sinal) and bool(troca_sinal.iloc[j]):
                has_flip = True
                flip_at = j
                break

        if not has_flip:
            continue

        atraso = int(i4 - i0)
        atraso_list.append(atraso)
        corredor_list.append({
            "i_corredor": i0,
            "i_flip": int(flip_at) if flip_at is not None else None,
            "i_4": int(i4),
        })

    if len(atraso_list) > 0:
        atraso_med = float(_np.median(atraso_list))
        atraso_p25 = float(_np.percentile(atraso_list, 25))
        atraso_p75 = float(_np.percentile(atraso_list, 75))
    else:
        atraso_med = None
        atraso_p25 = None
        atraso_p75 = None

    # Estado recente (últimos valores)
    d1_last = float(d1.iloc[-1]) if len(d1) else 0.0
    d2_last = float(d2.iloc[-1]) if len(d2) else 0.0

    if d1_last > 0.0 and abs(d2_last) <= eps:
        estado = "SUBINDO_COM_CURVATURA_BAIXA"
    elif d1_last < 0.0 and abs(d2_last) <= eps:
        estado = "DESCENDO_COM_CURVATURA_BAIXA"
    elif abs(d2_last) <= eps:
        estado = "PLANO_CURVATURA_BAIXA"
    elif d2_last < 0.0:
        estado = "CURVATURA_NEGATIVA"
    else:
        estado = "CURVATURA_POSITIVA"

    # troca de sinal "recente" (últimos 2L)
    tail_flip = troca_sinal.tail(2*L)
    troca_sinal_recente = bool(tail_flip.any()) if len(tail_flip) else False

    out = {
        "ok": True,
        "eixo": eixo,
        "n": int(len(df)),
        "params": {"w_smooth": int(w_smooth), "L_sust": int(L), "eps_mult": float(eps_mult)},
        "eps": float(eps),
        "estado_recente": estado,
        "d1_last": float(d1_last),
        "d2_last": float(d2_last),
        "curvatura_sustentada_recente": bool(sust_recente),
        "troca_sinal_recente": bool(troca_sinal_recente),
        "dist_desde_ultimo_4": dist_desde_4,
        "relogio_geometrico": {
            "atraso_mediano": atraso_med,
            "p25": atraso_p25,
            "p75": atraso_p75,
            "amostras": int(len(atraso_list)),
        },
        "debug": {
            "last_4_i": int(last_4_i) if last_4_i is not None else None,
            "corredores_usados": corredor_list[-3:] if len(corredor_list) > 0 else [],
        },
    }
    return out


def validar_lista_vs_n_real(lista, n_real):
    return isinstance(lista, (list, tuple)) and len(lista) >= int(n_real)

# -----------------------------
# ORÇAMENTOS CONDICIONADOS (TABELAS)
# -----------------------------
ORCAMENTOS_CONDICIONADOS = {
    5: {
        5: 3,
        6: 18,
        7: 63,
        8: 168,
        9: 378,
        10: 756,
    },
    6: {
        6: 6,
        7: 42,
        8: 168,
        9: 504,
        10: 1260,
        11: 2772,
    },
    15: {
        15: 3.5,
        16: 56,
        17: 476,
    },
}

# -----------------------------
# RESOLUÇÃO DE ORÇAMENTO
# -----------------------------
def resolver_orcamento(n_real, tamanho_lista, orcamento_manual=None):
    """
    Prioridade:
    1) Orçamento manual (se fornecido)
    2) Tabela condicionada por n_real
    3) None (não avalia custo)
    """
    if orcamento_manual is not None:
        try:
            return float(orcamento_manual)
        except Exception:
            return None

    tabela = ORCAMENTOS_CONDICIONADOS.get(int(n_real))
    if not tabela:
        return None

    return tabela.get(int(tamanho_lista))

# -----------------------------
# AVALIAÇÃO UNIVERSAL (OBSERVACIONAL)
# -----------------------------
def avaliar_listas_universal(listas, alvo_real, n_real, orcamento_manual=None):
    """
    Retorna métricas OBSERVACIONAIS:
    - acertos / n_real
    - custo (se disponível)
    """
    resultados = []
    alvo_set = set(int(v) for v in alvo_real if int(v) > 0)

    for idx, lst in enumerate(listas, start=1):
        if not validar_lista_vs_n_real(lst, n_real):
            continue

        lst_set = set(int(v) for v in lst if int(v) > 0)
        acertos = len(alvo_set.intersection(lst_set))
        custo = resolver_orcamento(n_real, len(lst), orcamento_manual)

        resultados.append({
            "lista_id": idx,
            "tamanho_lista": len(lst),
            "acertos": acertos,
            "n_real": int(n_real),
            "score": f"{acertos}/{int(n_real)}",
            "custo": custo,
        })

    return resultados





# ============================================================
# GUARDAS DE SEGURANÇA POR n_alvo
# (INFRAESTRUTURA — NÃO APLICADA A NENHUM PAINEL)
# ============================================================

def guarda_n_alvo(n_esperado, nome_modulo):
    n_alvo = st.session_state.get("n_alvo")

    if n_alvo is None:
        st.warning(
            f"⚠️ {nome_modulo}: n_alvo não detectado. "
            f"Carregue um histórico válido antes de executar este painel."
        )
        return False

    if n_alvo != n_esperado:
        st.warning(
            f"🚫 {nome_modulo} BLOQUEADO\n\n"
            f"n detectado = {n_alvo}\n"
            f"n esperado por este módulo = {n_esperado}\n\n"
            f"Este painel assume n fixo e foi bloqueado para evitar "
            f"cálculo incorreto ou truncamento silencioso."
        )
        return False

    return True



# ============================================================
# V16 PREMIUM — INSTRUMENTAÇÃO RETROSPECTIVA (ERRO POR REGIME)
# (PAINEL OBSERVACIONAL PERMANENTE — NÃO MUDA MOTOR)
# ============================================================

def _pc16_normalizar_series_6(historico_df: pd.DataFrame) -> np.ndarray:
    """
    Extrai exatamente as colunas p1..p6 do histórico V15.7 MAX.
    Retorna matriz shape (N, 6) com cada série ordenada.
    """
    if historico_df is None or historico_df.empty:
        return np.zeros((0, 6), dtype=float)

    colunas_esperadas = ["p1", "p2", "p3", "p4", "p5", "p6"]
    for c in colunas_esperadas:
        if c not in historico_df.columns:
            return np.zeros((0, 6), dtype=float)

    try:
        dfp = historico_df[colunas_esperadas].astype(float).dropna()
    except Exception:
        return np.zeros((0, 6), dtype=float)

    if len(dfp) < 10:
        return np.zeros((0, 6), dtype=float)

    arr = dfp.values
    arr.sort(axis=1)
    return arr



def _pc16_distancia_media(v: np.ndarray, centro: np.ndarray) -> float:
    """
    Distância média absoluta (L1 média) entre vetor de 6 e centro de 6.
    """
    return float(np.mean(np.abs(v - centro)))



def pc16_calcular_continuidade_por_janelas(
    historico_df: pd.DataFrame,
    janela: int = 60,
    step: int = 1,
    usar_quantis: bool = True
) -> Dict[str, Any]:
    """
    Analisa retrospectivamente o histórico em janelas móveis.
    Para cada janela [t-janela, t), calcula:
      - 'dx_janela': dispersão média das séries da janela em relação ao centróide da janela
      - 'erro_prox': erro da PRÓXIMA série (t) em relação ao centróide da janela (proxy de 'erro contido')
    Classifica regime por dx_janela (ECO / PRE / RUIM) e compara erro_prox por regime.

    Retorna dict com DataFrame e resumo.
    """
    X = _pc16_normalizar_series_6(historico_df)
    n = X.shape[0]
    if n < (janela + 5):
        return {
            "ok": False,
            "motivo": f"Histórico insuficiente para janela={janela}. Séries válidas: {n}.",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    rows = []
    # percorre janelas, garantindo que exista a "próxima" série t
    for t in range(janela, n - 1, step):
        bloco = X[t - janela:t, :]
        centro = np.mean(bloco, axis=0)

        # dx_janela: média das distâncias das séries da janela ao centróide
        dists = [ _pc16_distancia_media(bloco[i], centro) for i in range(bloco.shape[0]) ]
        dx_janela = float(np.mean(dists))

        # erro_prox: distância da série seguinte (t) ao centróide da janela
        prox = X[t, :]
        erro_prox = _pc16_distancia_media(prox, centro)

        rows.append({
            "t": t,  # índice da série (0-based dentro do array)
            "dx_janela": dx_janela,
            "erro_prox": erro_prox
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "ok": False,
            "motivo": "Não foi possível gerar janelas (df vazio).",
            "df": pd.DataFrame(),
            "resumo": {}
        }

    # Classificação de regime (ECO/PRE/RUIM) baseada em dx_janela
    if usar_quantis:
        q1 = float(df["dx_janela"].quantile(0.33))
        q2 = float(df["dx_janela"].quantile(0.66))
    else:
        # fallback conservador: thresholds fixos (raramente usado)
        q1, q2 = 0.30, 0.45

    def _rotulo(dx: float) -> str:
        if dx <= q1:
            return "ECO"
        elif dx <= q2:
            return "PRE"
        return "RUIM"

    df["regime"] = df["dx_janela"].apply(_rotulo)

    # Métricas resumo
    resumo = {}
    for reg in ["ECO", "PRE", "RUIM"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            resumo[reg] = {"n": 0}
            continue

        resumo[reg] = {
            "n": int(len(sub)),
            "dx_janela_medio": float(sub["dx_janela"].mean()),
            "erro_prox_medio": float(sub["erro_prox"].mean()),
            "erro_prox_mediana": float(sub["erro_prox"].median()),
        }

    # Métrica única que queremos: diferença ECO vs RUIM no erro_prox médio
    if resumo.get("ECO", {}).get("n", 0) > 0 and resumo.get("RUIM", {}).get("n", 0) > 0:
        diff = resumo["RUIM"]["erro_prox_medio"] - resumo["ECO"]["erro_prox_medio"]
    else:
        diff = None

    resumo_geral = {
        "janela": int(janela),
        "step": int(step),
        "q1_dx": q1,
        "q2_dx": q2,
        "diff_ruim_menos_eco_no_erro": diff,
        "n_total_janelas": int(len(df))
    }

    return {
        "ok": True,
        "motivo": "",
        "df": df,
        "resumo": resumo,
        "resumo_geral": resumo_geral
    }



# ============================================================
# Função utilitária — formatador geral
# ============================================================
def formatar_lista_passageiros(lista: List[int]) -> str:
    """Formata lista no padrão compacto V15.7 MAX"""
    return ", ".join(str(x) for x in lista)

# ============================================================
# Parsing FLEX ULTRA — versão robusta V15.7 MAX
# ============================================================
def analisar_historico_flex_ultra(conteudo: str) -> pd.DataFrame:
    """
    Parser oficial V15.7 MAX — leitura de histórico com:
    - prefixo C1, C2, C3 ...
    - 5 ou 6 passageiros
    - sensor k sempre na última coluna
    """
    linhas = conteudo.strip().split("\n")
    registros = []

    for linha in linhas:
        partes = linha.replace(" ", "").split(";")
        if len(partes) < 7:
            continue

        try:
            serie = partes[0]
            nums_raw = list(map(int, partes[1:-1]))
            if len(nums_raw) < 6:
                continue
            nums = nums_raw[:6]  # CANÔNICO: sempre 6 passageiros (k é o último campo)
            k_val = int(partes[-1])
            registros.append([serie] + nums + [k_val])
        except:
            continue

    colunas = ["serie", "p1", "p2", "p3", "p4", "p5", "p6", "k"]
    if not registros:
        return pd.DataFrame(columns=colunas)
    df = pd.DataFrame(registros, columns=colunas)

    return df

# ============================================================
# Utilitários de texto e apresentação — V15.7 MAX
# ============================================================
def texto_em_blocos(texto: str, largura: int = 100) -> List[str]:
    if not texto:
        return []
    return textwrap.wrap(texto, width=largura)


def exibir_bloco_mensagem(
    titulo: str,
    corpo: str,
    tipo: str = "info",
) -> None:

    blocos = texto_em_blocos(corpo, largura=110)

    if tipo == "info":
        st.info(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "warning":
        st.warning(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "error":
        st.error(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    elif tipo == "success":
        st.success(f"**{titulo}**\n\n" + "\n\n".join(blocos))
    else:
        st.markdown(
            f"""
            <div class="info-box">
                <div class="sub-title">{titulo}</div>
                <p>{"<br>".join(blocos)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Configurações Anti-Zumbi — limites globais
# ============================================================
LIMITE_SERIES_REPLAY_ULTRA: int = 8000
LIMITE_SERIES_TURBO_ULTRA: int = 8000
LIMITE_PREVISOES_TURBO: int = 600
LIMITE_PREVISOES_MODO_6: int = 800


def limitar_operacao(
    qtd_series: int,
    limite_series: int,
    contexto: str = "",
    painel: str = "",
) -> bool:

    if qtd_series is None:
        return True

    if qtd_series <= limite_series:
        return True

    msg = (
        f"🔒 **Operação bloqueada pela Proteção Anti-Zumbi ({contexto}).**\n\n"
        f"- Séries detectadas: **{qtd_series}**\n"
        f"- Limite seguro: **{limite_series}**\n"
        f"Painel: **{painel}**\n\n"
        "👉 Evitamos travamento no Streamlit."
    )
    exibir_bloco_mensagem("Proteção Anti-Zumbi", msg, tipo="warning")
    return False


# ============================================================
# NÚCLEO V16 — Premium Profundo (Diagnóstico & Calibração)
# Compatível com V15.7 MAX, 100% opcional e retrocompatível
# ============================================================
from typing import Dict, Any, Optional, Tuple  # Reimportar não faz mal


def v16_identificar_df_base() -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Tenta descobrir qual DataFrame de histórico está ativo no app.
    Busca em chaves comuns do st.session_state para não quebrar nada.
    Se não encontrar nada, retorna (None, None).
    """
    candidatos = []
    for chave in ["historico_df", "df_historico", "df_base", "df", "df_hist"]:
        if chave in st.session_state:
            objeto = st.session_state[chave]
            if isinstance(objeto, pd.DataFrame) and not objeto.empty:
                candidatos.append((chave, objeto))

    if not candidatos:
        return None, None

    chave_escolhida, df_escolhido = candidatos[0]
    return chave_escolhida, df_escolhido


def v16_resumo_basico_historico(
    df: pd.DataFrame,
    limite_linhas: int = 3000,
) -> Dict[str, Any]:
    """
    Gera um resumo leve do histórico para diagnóstico:
    - Quantidade total de séries
    - Janela usada para diagnóstico (anti-zumbi)
    - Distribuição de k (se existir)
    - Presença de colunas relevantes (k*, NR%, QDS)
    Tudo protegido contra KeyError e DataFrames pequenos.
    """
    resumo: Dict[str, Any] = {}

    n_total = int(len(df))
    if n_total <= 0:
        resumo["n_total"] = 0
        resumo["n_usado"] = 0
        resumo["colunas"] = list(df.columns)
        resumo["dist_k"] = {}
        resumo["info_extra"] = {}
        return resumo

    limite_seguro = max(100, min(limite_linhas, n_total))
    df_uso = df.tail(limite_seguro).copy()

    resumo["n_total"] = n_total
    resumo["n_usado"] = int(len(df_uso))
    resumo["colunas"] = list(df_uso.columns)

    dist_k: Dict[Any, int] = {}
    if "k" in df_uso.columns:
        try:
            contagem_k = df_uso["k"].value_counts().sort_index()
            for k_val, qtd in contagem_k.items():
                dist_k[int(k_val)] = int(qtd)
        except Exception:
            dist_k = {}
    resumo["dist_k"] = dist_k

    info_extra: Dict[str, Any] = {}
    for col in df_uso.columns:
        col_lower = str(col).lower()
        if "k*" in col_lower or "k_est" in col_lower or "kstar" in col_lower:
            info_extra["tem_k_estrela"] = True
        if "nr" in col_lower and "%" in col_lower:
            info_extra["tem_nr_percent"] = True
        if "qds" in col_lower:
            info_extra["tem_qds"] = True
    resumo["info_extra"] = info_extra

    return resumo


def v16_mapear_confiabilidade_session_state() -> Dict[str, Any]:
    """
    Varre st.session_state e tenta localizar informações de confiabilidade,
    QDS, k*, NR%, etc., sem assumir nomes fixos.
    Não quebra o app se nada for encontrado.
    """
    mapeamento: Dict[str, Any] = {}

    try:
        for chave, valor in st.session_state.items():
            nome_lower = str(chave).lower()
            if any(token in nome_lower for token in ["confiab", "qds", "k_estrela", "k*", "nr%", "ruido"]):
                if isinstance(valor, (int, float, str)):
                    mapeamento[chave] = valor
                elif isinstance(valor, dict):
                    mapeamento[chave] = {"tipo": "dict", "tamanho": len(valor)}
                elif isinstance(valor, pd.DataFrame):
                    mapeamento[chave] = {
                        "tipo": "DataFrame",
                        "linhas": len(valor),
                        "colunas": list(valor.columns)[:10],
                    }
                else:
                    mapeamento[chave] = {"tipo": type(valor).__name__}
    except Exception:
        pass

    return mapeamento


# ============================================================
# Métricas básicas do histórico — V15.7 MAX
# ============================================================
def calcular_metricas_basicas_historico(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas simples do histórico para uso em:
    - Painel de carregamento
    - Monitor de Risco — k & k*
    Tudo de forma leve, sem quebrar se faltarem colunas.
    """
    metricas: Dict[str, Any] = {}

    if df is None or df.empty:
        metricas["qtd_series"] = 0
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0
        return metricas

    metricas["qtd_series"] = int(len(df))

    if "k" in df.columns:
        try:
            k_vals = df["k"].astype(float)
            metricas["min_k"] = float(k_vals.min())
            metricas["max_k"] = float(k_vals.max())
            metricas["media_k"] = float(k_vals.mean())
        except Exception:
            metricas["min_k"] = None
            metricas["max_k"] = None
            metricas["media_k"] = 0.0
    else:
        metricas["min_k"] = None
        metricas["max_k"] = None
        metricas["media_k"] = 0.0

    return metricas


def exibir_resumo_inicial_historico(metricas: Dict[str, Any]) -> None:
    """
    Exibe um resumo amigável logo após o carregamento do histórico.
    Usado no Painel 1 (Carregar Histórico) e como base para o Monitor de Risco.
    """
    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k", 0.0)

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
    )

    exibir_bloco_mensagem(
        "Resumo inicial do histórico (V15.7 MAX)",
        corpo,
        tipo="info",
    )

# ============================================================
# Cabeçalho visual principal
# ============================================================
st.markdown(
    '<div class="big-title">🚗 Predict Cars V15.7 MAX — V16 PREMIUM PROFUNDO</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="gray-text">
    Núcleo + Coberturas + Interseção Estatística · Pipeline V14-FLEX ULTRA ·
    Replay LIGHT/ULTRA · TURBO++ HÍBRIDO · TURBO++ ULTRA · Monitor de Risco (k & k*) ·
    Painel de Ruído Condicional · Divergência S6 vs MC · Testes de Confiabilidade REAL ·
    Modo 6 Acertos V15.7 MAX · Relatório Final Integrado.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Construção da  — V15.7 MAX
# (LAUDO DE CÓDIGO — BLOCO 1-FIX | ORDEM OPERACIONAL FINAL)
# ============================================================

def construir_navegacao_v157() -> str:

    st.sidebar.markdown("## 🚦  PredictCars V15.7 MAX")
    st.sidebar.markdown("📌 Selecione o painel:")

    n_alvo = st.session_state.get("n_alvo")

    # =========================================================
    # ORDEM OPERACIONAL DEFINITIVA — USO DIÁRIO
    # =========================================================
    opcoes = [

        # -----------------------------------------------------
        # BLOCO 0 — ENTRADA
        # -----------------------------------------------------
        "📁 Carregar Histórico (Arquivo)",
        "📄 Carregar Histórico (Colar)",

        # -----------------------------------------------------
        # BLOCO 1 — ORIENTAÇÃO
        # -----------------------------------------------------
        "🧭 Modo Guiado Oficial — PredictCars",
        "🔍 Diagnóstico Espelho (Mirror)",

        # -----------------------------------------------------
        # BLOCO 2 — LEITURA DO AMBIENTE
        # -----------------------------------------------------
        "🛰️ Sentinelas — k* (Ambiente de Risco)",
        "🧭 Monitor de Risco — k & k*",
        "📊 Observação Histórica — Eventos k",
        "⏱️ Duração da Janela — Análise Histórica",

        # -----------------------------------------------------
        # BLOCO 3 — QUALIDADE DO SINAL
        # -----------------------------------------------------
        "📡 Painel de Ruído Condicional",
        "📉 Painel de Divergência S6 vs MC",

        # -----------------------------------------------------
        # BLOCO 4 — RITMO DO ALVO
        # -----------------------------------------------------
        "🔁 Replay LIGHT",
        "🔁 Replay ULTRA",
                "🧭 Replay Progressivo — Janela Móvel (Assistido)",
        "🧪 P1 — Ajuste de Pacote (pré-C4) — Comparativo",
        "🧪 MC Observacional do Pacote (pré-C4)",
        "📐 Parabólica — Curvatura do Erro (Governança Pré-C4)",
        "📡 CAP — Calibração Assistida da Parabólica (pré-C4)",
    "🧪 P2 — Hipóteses de Família (pré-C4)",
        "🧪 Replay Curto — Expectativa 1–3 Séries",

        # -----------------------------------------------------
        # BLOCO 5 — EIXO 1 | ESTRUTURA DAS LISTAS
        # -----------------------------------------------------
        "🧼 B1 — Higiene de Passageiros",
        "🧩 B2 — Coerência Interna das Listas",
        "🟢 B3 — Prontidão (Refinamento)",
        "🟣 B4 — Refinamento Leve de Passageiros",

        # -----------------------------------------------------
        # BLOCO 6 — DECISÃO ÚNICA
        # -----------------------------------------------------
        "🧭 Checklist Operacional — Decisão (AGORA)",

        # -----------------------------------------------------
        # BLOCO 7 — MOTOR
        # -----------------------------------------------------
        "🛣️ Pipeline V14-FLEX ULTRA",
        "⚙️ Modo TURBO++ HÍBRIDO",
        "⚙️ Modo TURBO++ ULTRA",

        # -----------------------------------------------------
        # BLOCO 7.5 — EIXO 2 | MOMENTO & ANTECIPAÇÃO
        # -----------------------------------------------------
        "📊 V16 Premium — Backtest Rápido do Pacote (N=60)",
        "📊 P1 — Backtest Comparativo BLOCO C (A/B) — N=60",
        "🧭 V16 Premium — Rodadas Estratificadas (A/B)",

        "🧠 M5 — Pulo do Gato (Coleta Automática de Estados)",

        "📈 Expectativa Histórica — Contexto do Momento (V16)",

        # -----------------------------------------------------
        # BLOCO 8 — EXECUÇÃO
        # -----------------------------------------------------
        "🎯 Modo 6 Acertos — Execução",
        "🧪 Testes de Confiabilidade REAL",
        "📘 Relatório Final",

        # -----------------------------------------------------
        # BLOCO 9 — EXTENSÃO
        # -----------------------------------------------------
        "🔵 MODO ESPECIAL — Evento Condicionado",

        # -----------------------------------------------------
        # BLOCO 10 — CAMADA UNIVERSAL
        # -----------------------------------------------------
        "💰 MVP-U2 — Orçamento Universal",
        "🧩 MVP-U3 — Cobertura Universal",
        "📈 MVP-U4 — Eficiência Marginal por Custo",

        # -----------------------------------------------------
        # BLOCO 11 — DEPOIS | APRENDIZADO (EIXO 3)
        # -----------------------------------------------------
        "🧠 Memória Operacional",
        "🧠 Memória Operacional — Registro Semi-Automático",
        "🧠 Laudo Operacional V16",
        "🧠 Diagnóstico ECO & Estado (V16)",
        "🧭 RMO/DMO — Retrato do Momento (V16)",
        "🧾 APS — Auditoria de Postura (V16)",
        "📊 V16 Premium — Erro por Regime (Retrospectivo)",
        "📊 V16 Premium — EXATO por Regime (Proxy)",
        "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)",
        "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)",
        "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros",
        "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos",
        "🎯 Compressão do Alvo — Observacional (V16)",
        "🔮 V16 Premium Profundo — Diagnóstico & Calibração",
    ]

    # ------------------------------------------------------------
    # INSERÇÃO CONDICIONAL — MODO N EXPERIMENTAL (n≠6)
    # ------------------------------------------------------------
    if (n_alvo is not None) and (int(n_alvo) != 6):
        opcoes.insert(
            opcoes.index("🎯 Modo 6 Acertos — Execução"),
            "🧪 Modo N Experimental (n≠6)"
        )
    # --- estabilidade do NAV (evita painel=None / TypeError)
    _nav_key = "NAV_V157_CANONICA"
    _prev = st.session_state.get(_nav_key, None)
    if (_prev is None) or (_prev not in opcoes):
        _prev = opcoes[0] if opcoes else "Carregar Histórico (Colar)"
        st.session_state[_nav_key] = _prev
    painel = st.sidebar.radio("📌 Selecione o painel:", opcoes, index=opcoes.index(_prev) if _prev in opcoes else 0, key=_nav_key, label_visibility="collapsed")
    return painel


# ============================================================
# FIM — Construção da  — V15.7 MAX
# ============================================================



# ============================================================
# Ativação da Navegação — V15.7 MAX
# ============================================================

painel = construir_navegacao_v157()
if painel is None:
    # fallback absoluto (protege contra TypeError em `in painel`)
    painel = "Carregar Histórico (Colar)"
st.sidebar.caption(f"Painel ativo: {painel}")

# ============================================================
# DEBUG — CARIMBO DE BUILD (QUAL ARQUIVO REALMENTE ESTÁ RODANDO)
# ============================================================
try:
    st.sidebar.markdown("---")
    st.sidebar.caption("✅ BUILD-ID: NAV_ANCORA_ESTAVEL_2026-01-18")
    # Observação: build-id atualizado quando há alteração canônica na navegação.
    st.sidebar.caption(f"📄 __file__: {__file__}")
    st.sidebar.caption(f"🔎 Primeiro item NAV: {construir_navegacao_v157.__name__}")
    st.sidebar.caption("🧭 TOP-5: (debug desativado — não chamar construir_navegacao_v157() aqui)")
except Exception as _e:
    st.sidebar.caption(f"⚠️ DEBUG build falhou: {_e}")



# ============================================================
# DEBUG MINIMAL — CONFIRMA PAINEL ATIVO
# (manter por enquanto para auditoria)
# ============================================================
st.sidebar.caption(f"Painel ativo: {painel}")



# ============================================================
# MODO ESPECIAL — EVENTO CONDICIONADO (C2955)
# AVALIAÇÃO MULTI-ORÇAMENTO | OBSERVACIONAL | 6 OU NADA
# ============================================================

def pc_especial_avaliar_pacote_contem_6(carro, alvo):
    """
    Retorna True se o carro contém TODOS os 6 números do alvo.
    Régua BINÁRIA: 6 ou nada.
    """
    try:
        return set(alvo).issubset(set(carro))
    except Exception:
        return False


def pc_especial_avaliar_historico_pacote(historico_df, pacote):
    """
    Percorre o histórico rodada a rodada e verifica se,
    em alguma rodada, algum carro do pacote contém os 6.
    Retorna contagem de sucessos.
    """
    if historico_df is None or historico_df.empty:
        return {
            "rodadas": 0,
            "sucessos": 0,
        }

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    rodadas = 0
    sucessos = 0

    for _, row in historico_df.iterrows():
        try:
            alvo = [int(row[c]) for c in col_pass[:6]]
        except Exception:
            continue

        rodadas += 1

        for carro in pacote:
            if pc_especial_avaliar_pacote_contem_6(carro, alvo):
                sucessos += 1
                break  # sucesso binário por rodada

    return {
        "rodadas": rodadas,
        "sucessos": sucessos,
    }


# ============================================================
# 🔵 MODO ESPECIAL — MVP2 (2–6 acertos + Estado do Alvo PROXY)
# OBSERVACIONAL | NÃO decide | NÃO gera pacotes | NÃO aprende
# ============================================================

def _pc_contar_hits_lista_vs_alvo(lista, alvo_set):
    """
    Retorna quantidade de acertos (interseção) entre uma lista (carro) e o alvo (set).
    """
    try:
        s = set(int(x) for x in lista)
    except Exception:
        return 0
    return len(s & alvo_set)


def _pc_melhor_hit_do_pacote(pacote_listas, alvo_set):
    """
    Dado um pacote (listas de previsão), retorna o MELHOR hit (0..6) encontrado contra o alvo.
    """
    if not pacote_listas:
        return 0

    best = 0
    for lst in pacote_listas:
        h = _pc_contar_hits_lista_vs_alvo(lst, alvo_set)
        if h > best:
            best = h
            if best >= 6:
                break
    return best


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do histórico.
    Espera colunas p1..p6 (padrão do PredictCars).
    """
    try:
        return [int(row[f"p{i}"]) for i in range(1, 7)]
    except Exception:
        return None


def _pc_distancia_carros(carro_a, carro_b):
    """
    Distância simples entre dois carros (proxy):
    número de passageiros diferentes.
    """
    if carro_a is None or carro_b is None:
        return None
    try:
        return len(set(carro_a) ^ set(carro_b))
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Classificação simples do estado do alvo (proxy),
    baseada na distância entre carros consecutivos.
    """
    if dist is None:
        return "None"

    try:
        d = float(dist)
    except Exception:
        return "None"

    if d <= 1:
        return "parado"
    elif d <= 3:
        return "movimento_lento"
    else:
        return "movimento_brusco"


def pc_modo_especial_mvp2_avaliar_pacote(df_hist, pacote_listas):
    """
    MVP2:
    - Para cada série do histórico, computa:
        estado_alvo_proxy (parado/lento/brusco/None)
        melhor_hit (0..6) do pacote contra o alvo daquela série
    - Consolida em tabela: Estado x Hits(2..6) [contagem EXATA]
    Retorna (df_resumo, total_series_avaliadas).
    """
    if df_hist is None or df_hist.empty:
        return pd.DataFrame(), 0

    if not pacote_listas:
        return pd.DataFrame(), int(len(df_hist))

    cont = {
        "parado": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_lento": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "movimento_brusco": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        "None": {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    }

    rows = list(df_hist.iterrows())
    carro_prev = None

    for _, row in rows:
        carro_atual = _pc_extrair_carro_row(row)

        dist = (
            _pc_distancia_carros(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado = _pc_estado_alvo_proxy(dist)
        estado_key = estado if estado in cont else "None"

        if carro_atual is None:
            carro_prev = carro_atual
            continue

        alvo_set = set(carro_atual)
        best_hit = _pc_melhor_hit_do_pacote(pacote_listas, alvo_set)

        if best_hit in [2, 3, 4, 5, 6]:
            cont[estado_key][best_hit] += 1

        carro_prev = carro_atual

    out = []
    for estado_key in ["parado", "movimento_lento", "movimento_brusco", "None"]:
        linha = {"Estado": estado_key}
        for h in [2, 3, 4, 5, 6]:
            linha[str(h)] = int(cont[estado_key][h])
        out.append(linha)

    df_out = pd.DataFrame(out)

    ordem = {"parado": 0, "movimento_lento": 1, "movimento_brusco": 2, "None": 3}
    df_out["__ord"] = df_out["Estado"].map(ordem).fillna(9).astype(int)
    df_out = df_out.sort_values("__ord").drop(columns=["__ord"])

    return df_out, int(len(df_hist))

# ============================================================
# 🔵 FIM — FUNÇÕES DO MODO ESPECIAL MVP2
# ============================================================


# ============================================================
# PAINEL — 🔵 MODO ESPECIAL (Evento Condicionado C2955)
# Avaliação MULTI-ORÇAMENTO | Observacional
# ============================================================

if painel == "🔵 MODO ESPECIAL — Evento Condicionado":

    st.markdown("## 🔵 MODO ESPECIAL — Evento Condicionado (C2955)")
    st.caption(
        "Avaliação OBSERVACIONAL de pacotes já gerados.\n\n"
        "✔ Régua extrema: **6 ou nada** (MVP1)\n"
        "✔ Avaliação realista: **2–6 por estado do alvo** (MVP2)\n"
        "✔ Sem aprendizado\n"
        "✔ Sem interferência no Modo Normal\n"
        "✔ Decisão HUMANA (Rogério + Auri)"
    )

    historico_df = st.session_state.get("historico_df")

    # ============================================================
    # 🔵 SELETOR DE FONTE DO PACOTE (TURBO × MODO 6)
    # OBSERVACIONAL | NÃO decide | NÃO aprende | NÃO interfere
    # ============================================================

    pacote_turbo_raw = st.session_state.get("ultima_previsao")

    pacote_m6_total = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    pacote_m6_top10 = st.session_state.get("modo6_listas_top10") or []

    fontes = []
    if pacote_turbo_raw:
        fontes.append("TURBO (núcleo)")
    if pacote_m6_total:
        fontes.append("MODO 6 (TOTAL)")
    if pacote_m6_top10:
        fontes.append("MODO 6 (TOP 10)")
    if pacote_turbo_raw and pacote_m6_total:
        fontes.append("MIX (TURBO + M6 TOTAL)")

    if not fontes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "É necessário:\n"
            "- Histórico carregado\n"
            "- Pacotes gerados pelo TURBO ou Modo 6",
            tipo="warning",
        )
        st.stop()

    idx_default = fontes.index("MODO 6 (TOTAL)") if "MODO 6 (TOTAL)" in fontes else 0

    fonte_escolhida = st.selectbox(
        "Fonte do pacote para avaliação (observacional):",
        options=fontes,
        index=idx_default,
    )

    # -----------------------------
    # Construção do pacote ativo
    # -----------------------------
    if fonte_escolhida == "TURBO (núcleo)":
        pacotes_raw = pacote_turbo_raw
    elif fonte_escolhida == "MODO 6 (TOTAL)":
        pacotes_raw = pacote_m6_total
    elif fonte_escolhida == "MODO 6 (TOP 10)":
        pacotes_raw = pacote_m6_top10
    else:
        mix = []

        if isinstance(pacote_turbo_raw, list):
            if pacote_turbo_raw and isinstance(pacote_turbo_raw[0], int):
                mix.append(pacote_turbo_raw)
            else:
                mix.extend(pacote_turbo_raw)

        if isinstance(pacote_m6_total, list):
            mix.extend(pacote_m6_total)

        pacotes_raw = mix

    # ============================================================
    # ✅ NORMALIZAÇÃO FINAL — LISTA DE LISTAS
    # ============================================================
    if pacotes_raw is None:
        pacotes = []
    elif isinstance(pacotes_raw, list) and pacotes_raw and isinstance(pacotes_raw[0], int):
        pacotes = [pacotes_raw]
    elif isinstance(pacotes_raw, list):
        pacotes = pacotes_raw
    else:
        pacotes = []

    st.caption(
        f"Pacote ativo: **{fonte_escolhida}** | "
        f"Listas avaliadas: **{len(pacotes)}**"
    )

    if historico_df is None or historico_df.empty or not pacotes:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Histórico vazio ou pacote inválido.",
            tipo="warning",
        )
        st.stop()

    # ============================================================
    # 🔵 MVP4 — ANÁLISE DE COMPOSIÇÕES DE COBERTURA (OBSERVACIONAL)
    # Núcleo / Fronteira automáticos — NÃO executa
    # ============================================================

    st.markdown("### 🔵 MVP4 — Análise de Composições de Cobertura")
    st.caption(
        "Painel analítico: sugere **composições candidatas** (6×6 até 1×9),\n"
        "com base em núcleo/fronteira extraídos automaticamente.\n"
        "❌ Não gera listas | ❌ Não decide | ❌ Não interfere"
    )

    from collections import Counter
    from math import comb

    todas = [n for lista in pacotes for n in lista]
    freq = Counter(todas)

    nucleo = sorted([n for n, c in freq.items() if c >= 3])
    fronteira = sorted([n for n, c in freq.items() if c == 2])
    ruido = sorted([n for n, c in freq.items() if c == 1])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧱 Núcleo**")
        st.write(nucleo if nucleo else "—")
        if len(nucleo) < 4:
            st.warning("Núcleo fraco (<4).")
        if len(nucleo) > 5:
            st.warning("Núcleo grande (>5).")

    with col2:
        st.markdown("**🟡 Fronteira**")
        st.write(fronteira if fronteira else "—")
        if len(fronteira) > 6:
            st.warning("Fronteira extensa (ambiguidade elevada).")

    with col3:
        st.markdown("**🔴 Ruído**")
        st.write(ruido if ruido else "—")
        st.caption("Ruído excluído de carros >6.")

    st.markdown("#### 📦 Composições Candidatas (comparação teórica)")

    composicoes = [
        ("C1 — Foco puro", [(6, 6)]),
        ("C2 — Proteção leve", [(6, 4), (7, 1)]),
        ("C3 — Proteção + ambiguidade", [(6, 2), (7, 1), (8, 1)]),
        ("C4 — Envelope compacto", [(8, 1)]),
        ("C5 — Envelope amplo", [(9, 1)]),
    ]

    for nome, mix in composicoes:
        custo = 0
        combs = 0
        for m, q in mix:
            c = comb(m, 6)
            custo += c * 6 * q
            combs += c * q

        with st.expander(f"📘 {nome}"):
            st.write(f"Mix: {mix}")
            st.write(f"• Combinações de 6 cobertas: **{combs}**")
            st.write(f"• Custo teórico (régua): **{custo}**")

            if len(nucleo) < 4:
                st.warning("⚠️ Núcleo fraco — envelope pode diluir sinal.")
            if len(fronteira) > 6:
                st.warning("⚠️ Fronteira grande — risco de ilusão de cobertura.")

    # ============================================================
    # MVP2 — Avaliação 2–6 × Estado do Alvo (OBSERVACIONAL)
    # ============================================================

    st.markdown("### 📊 Resultado comparativo — MVP2 (2–6 × Estado do Alvo)")
    st.caption(
        "Leitura realista de aproximação.\n"
        "🟢 parado | 🟡 movimento lento | 🔴 movimento brusco\n"
        "O sistema **não decide**."
    )

    linhas = []

    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamentos_sel = st.multiselect(
        "Selecione os orçamentos a avaliar (observacional):",
        options=orcamentos_disponiveis,
        default=[42],
    )

    if not orcamentos_sel:
        st.warning("Selecione ao menos um orçamento.")
        st.stop()

    for orc in orcamentos_sel:
        df_mvp2, total_series = pc_modo_especial_mvp2_avaliar_pacote(
            df_hist=historico_df,
            pacote_listas=pacotes,
        )

        if df_mvp2 is None or df_mvp2.empty:
            linhas.append({
                "Orçamento": orc,
                "Estado": "N/A",
                "Séries": int(total_series),
                "2": 0, "3": 0, "4": 0, "5": 0, "6": 0
            })
            continue

        for _, r in df_mvp2.iterrows():
            linhas.append({
                "Orçamento": int(orc),
                "Estado": str(r["Estado"]),
                "Séries": int(total_series),
                "2": int(r["2"]),
                "3": int(r["3"]),
                "4": int(r["4"]),
                "5": int(r["5"]),
                "6": int(r["6"]),
            })

    df_cmp = pd.DataFrame(linhas)
    st.dataframe(df_cmp, use_container_width=True, height=420)

    st.info(
        "📌 Interpretação HUMANA:\n"
        "- 🟢 Mais 4/5 em 'parado' → janela boa\n"
        "- 🟡 Predomínio de 3/4 → cautela\n"
        "- 🔴 Quase só 2/3 → reduzir agressividade\n"
        "- 6 é raro; 4/5 indicam proximidade real"
    )




# ============================================================
# CAMADA A — ESTADO DO ALVO (V16)
# Observador puro — NÃO decide, NÃO bloqueia, NÃO gera previsões
# ============================================================

def v16_diagnosticar_eco_estado():
    """
    Diagnóstico OBSERVACIONAL enriquecido (ECO A):
    - ECO: força + persistência + acionabilidade
    - ESTADO: parado / movimento_lento / movimento_brusco + confiabilidade
    NÃO altera motores | NÃO decide | NÃO bloqueia
    """

    historico_df = st.session_state.get("historico_df")

    # -----------------------------
    # Fallback seguro
    # -----------------------------
    if historico_df is None or historico_df.empty:
        diag = {
            "eco_forca": "indefinido",
            "eco_persistencia": "indefinida",
            "eco_acionabilidade": "não_acionável",
            "estado": "indefinido",
            "estado_confiavel": False,
            "contradicoes": ["histórico insuficiente"],
            "leitura_geral": "Histórico insuficiente para diagnóstico.",
        }
        st.session_state["diagnostico_eco_estado_v16"] = diag
        v16_sync_aliases_canonicos()
        return diag

    # =========================================================
    # ECO — sinais já existentes
    # =========================================================
    k_star = st.session_state.get("sentinela_kstar")
    nr_pct = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    sinais = 0
    motivos = []

    if isinstance(k_star, (int, float)) and k_star < 0.22:
        sinais += 1
        motivos.append("k* favorável")

    if isinstance(nr_pct, (int, float)) and nr_pct < 30:
        sinais += 1
        motivos.append("ruído controlado")

    if isinstance(divergencia, (int, float)) and divergencia < 5:
        sinais += 1
        motivos.append("baixa divergência")

    if sinais >= 3:
        eco_forca = "forte"
    elif sinais == 2:
        eco_forca = "médio"
    else:
        eco_forca = "fraco"

    # Persistência curta
    hist_eco = st.session_state.get("historico_eco_v16", [])
    hist_eco.append(eco_forca)
    hist_eco = hist_eco[-5:]
    st.session_state["historico_eco_v16"] = hist_eco

    eco_persistencia = "persistente" if hist_eco.count(eco_forca) >= 3 else "instável"

    # =========================================================
    # CONTRADIÇÕES (leitura turva)
    # =========================================================
    contradicoes = []
    if eco_forca in ("fraco", "médio") and isinstance(divergencia, (int, float)) and divergencia > 20:
        contradicoes.append("divergência elevada")
    if eco_persistencia == "persistente" and isinstance(divergencia, (int, float)) and divergencia > 30:
        contradicoes.append("persistência enganosa")

    # =========================================================
    # ACIONABILIDADE (OBSERVACIONAL)
    # =========================================================
    if eco_forca == "forte" and eco_persistencia == "persistente" and not contradicoes:
        eco_acionabilidade = "favorável"
    elif eco_forca in ("médio", "forte") and not contradicoes:
        eco_acionabilidade = "cautela"
    else:
        eco_acionabilidade = "não_acionável"

    # =========================================================
    # ESTADO DO ALVO (proxy existente)
    # =========================================================
    estado_proxy = None
    try:
        col_pass = [c for c in historico_df.columns if c.startswith("p")]
        if len(col_pass) >= 6 and len(historico_df) >= 2:
            a = [int(historico_df.iloc[-1][c]) for c in col_pass[:6]]
            b = [int(historico_df.iloc[-2][c]) for c in col_pass[:6]]
            dist = len(set(a) ^ set(b))
            if dist <= 1:
                estado_proxy = "parado"
            elif dist <= 3:
                estado_proxy = "movimento_lento"
            else:
                estado_proxy = "movimento_brusco"
    except Exception:
        estado_proxy = None

    if estado_proxy is None:
        estado = "indefinido"
        estado_confiavel = False
    else:
        estado = estado_proxy
        hist_estado = st.session_state.get("historico_estado_v16", [])
        hist_estado.append(estado)
        hist_estado = hist_estado[-5:]
        st.session_state["historico_estado_v16"] = hist_estado
        estado_confiavel = hist_estado.count(estado) >= 3

    # =========================================================
    # LEITURA FINAL (MASTIGADA)
    # =========================================================
    leitura = (
        f"ECO {eco_forca}, {eco_persistencia}, {eco_acionabilidade}. "
        f"Estado {estado}. "
        f"{'Confiável' if estado_confiavel else 'Em transição'}."
    )
    if contradicoes:
        leitura += " Atenção: " + "; ".join(contradicoes) + "."

    diagnostico = {
        "eco_forca": eco_forca,
        "eco_persistencia": eco_persistencia,
        "eco_acionabilidade": eco_acionabilidade,
        "estado": estado,
        "estado_confiavel": estado_confiavel,
        "contradicoes": contradicoes,
        "leitura_geral": leitura,
        "motivos_eco": motivos,
    }

    st.session_state["diagnostico_eco_estado_v16"] = diagnostico
    return diagnostico

# ============================================================
# ATIVAÇÃO SILENCIOSA — DIAGNÓSTICO ECO & ESTADO (V16)
# ============================================================
if "historico_df" in st.session_state:
    try:
        v16_diagnosticar_eco_estado()
    except Exception:
        pass




# ============================================================
# CAMADA B — EXPECTATIVA DE CURTO PRAZO (V16)
# Laudo observacional: horizonte 1–3 séries (NÃO decide)
# ============================================================


def v16_calcular_expectativa_curto_prazo(
    df: Optional[pd.DataFrame],
    estado_alvo: Optional[Dict[str, Any]],
    k_star: Optional[float],
    nr_percent: Optional[float],
    divergencia: Optional[float],
) -> Dict[str, Any]:

    if df is None or df.empty:
        return {
            "horizonte": "1–3 séries",
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Histórico insuficiente para expectativa.",
        }

    k = float(k_star) if isinstance(k_star, (int, float)) else 0.25
    nr = float(nr_percent) if isinstance(nr_percent, (int, float)) else 35.0
    div = float(divergencia) if isinstance(divergencia, (int, float)) else 4.0

    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")

    # Índice simples de previsibilidade
    risco_norm = min(1.0, (nr / 70.0) * 0.4 + (div / 10.0) * 0.3 + (k / 0.5) * 0.3)
    previsibilidade_score = max(0.0, 1.0 - risco_norm)

    if previsibilidade_score >= 0.65:
        previsibilidade = "alta"
        erro = "baixo"
    elif previsibilidade_score >= 0.40:
        previsibilidade = "média"
        erro = "médio"
    else:
        previsibilidade = "baixa"
        erro = "alto"

    # Chance de janela de ouro (qualitativa)
    if tipo == "parado" and previsibilidade_score >= 0.60:
        chance_ouro = "alta"
    elif tipo == "movimento_lento" and previsibilidade_score >= 0.45:
        chance_ouro = "média"
    else:
        chance_ouro = "baixa"

    comentario = (
        f"Alvo {tipo}. Previsibilidade {previsibilidade}. "
        f"Erro esperado {erro}. Chance de janela de ouro {chance_ouro}."
    )

    return {
        "horizonte": "1–3 séries",
        "previsibilidade": previsibilidade,
        "erro_esperado": erro,
        "chance_janela_ouro": chance_ouro,
        "score_previsibilidade": round(previsibilidade_score, 4),
        "comentario": comentario,
    }


def v16_registrar_expectativa():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = v16_calcular_expectativa_curto_prazo(
        st.session_state.get("historico_df"),
        estado,
        st.session_state.get("sentinela_kstar"),
        st.session_state.get("nr_percent"),
        st.session_state.get("div_s6_mc"),
    )
    st.session_state["expectativa_v16"] = expectativa
    return expectativa

# ============================================================
# CAMADA C — VOLUME & CONFIABILIDADE (V16)
# Sistema INFORMA; humano DECIDE
# ============================================================

def v16_estimativa_confiabilidade_por_volume(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    base_confiabilidade: Optional[float] = None,
) -> Dict[int, float]:
    """
    Retorna um mapa {volume: confiabilidade_estimada}.
    Não bloqueia execução; apenas informa trade-offs.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    score_prev = (expectativa or {}).get("score_previsibilidade", 0.4)

    # Base de confiabilidade (fallback seguro)
    base = float(base_confiabilidade) if isinstance(base_confiabilidade, (int, float)) else score_prev

    # Ajuste por tipo de alvo
    if tipo == "parado":
        fator = 1.15
    elif tipo == "movimento_lento":
        fator = 1.00
    else:
        fator = 0.80

    volumes = [3, 6, 12, 20, 30, 50, 80]
    estimativas: Dict[int, float] = {}

    for v in volumes:
        # Ganho marginal decrescente
        ganho = 1.0 - (1.0 / max(1.0, np.log(v + 1)))
        conf = base * fator * ganho
        estimativas[v] = round(max(0.05, min(1.05, conf)), 3)

    return estimativas


def v16_calcular_volume_operacional(
    estado_alvo: Optional[Dict[str, Any]],
    expectativa: Optional[Dict[str, Any]],
    confiabilidades: Dict[int, float],
) -> Dict[str, Any]:
    """
    Consolida recomendações de volume sem impor decisão.
    """
    tipo = (estado_alvo or {}).get("tipo", "movimento_lento")
    prev = (expectativa or {}).get("previsibilidade", "média")

    # Volume recomendado por heurística qualitativa
    if tipo == "parado" and prev == "alta":
        recomendado = 30
    elif tipo == "movimento_lento":
        recomendado = 20
    else:
        recomendado = 6

    # Limites técnicos (anti-zumbi conceitual, não bloqueante)
    minimo = 3
    maximo = max(confiabilidades.keys()) if confiabilidades else 30

    return {
        "minimo": minimo,
        "recomendado": recomendado,
        "maximo_tecnico": maximo,
        "confiabilidades_estimadas": confiabilidades,
        "comentario": (
            "O sistema informa volumes e confiabilidades. "
            "A decisão final de quantas previsões gerar é do usuário."
        ),
    }


def v16_registrar_volume_e_confiabilidade():
    estado = st.session_state.get("estado_alvo_v16")
    expectativa = st.session_state.get("expectativa_v16")

    confiabs = v16_estimativa_confiabilidade_por_volume(
        estado_alvo=estado,
        expectativa=expectativa,
        base_confiabilidade=(expectativa or {}).get("score_previsibilidade"),
    )

    volume_op = v16_calcular_volume_operacional(
        estado_alvo=estado,
        expectativa=expectativa,
        confiabilidades=confiabs,
    )

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op



# ============================================================
# PARTE 1/8 — FIM
# ============================================================


# ============================================================
# MODULO 2 - MEMORIA HISTORICA DE ESTADOS (OBSERVACIONAL)
# Objetivo:
# - Registrar estados canonicos (S0..S6) por rodada, sem listas e sem acerto
# - Alimentar o MODULO 3 (Expectativa) com base comparavel e sem vies
# Regras:
# - NUNCA decide
# - NUNCA altera motores
# - NUNCA bloqueia
# - Falha silenciosa (nao derruba o app)
# ============================================================

import json
import hashlib


# ============================================================
# 🔒 Determinismo canônico (sem bifurcar): seed estável
# Motivo: hash() do Python é randomizado por processo, gerando listas diferentes no refresh.
# Este helper cria um seed reprodutível (32-bit) a partir de um texto.
# ============================================================
def pc_stable_seed(*parts) -> int:
    """Seed estável canônica.
    Aceita 1 string (compatível com versões antigas) OU múltiplas partes (FIX6),
    gerando seed determinístico (0..2**32-1)."""
    try:
        if len(parts) == 1:
            tag = str(parts[0])
        else:
            tag = "|".join(str(p) for p in parts)
        h = hashlib.sha256(tag.encode("utf-8", errors="ignore")).digest()
        return int.from_bytes(h[:8], "big", signed=False) % (2**32)
    except Exception:
        return 0

def pc_dirichlet_smooth_probs(counts: dict, universe: list, alpha: float = 1.0, eps: float = 0.02):
    """FIX6 TAILSTAB — suavização Dirichlet (Laplace) + mistura mínima uniforme.
    Não corta cauda; só estabiliza probabilidades em amostra pequena."""
    K = max(1, len(universe))
    total = 0.0
    for p in universe:
        total += float(counts.get(p, 0))
    denom = total + float(alpha) * K
    probs = {}
    if denom <= 0:
        # uniforme puro
        u = 1.0 / K
        for p in universe:
            probs[p] = u
        return probs

    # posterior mean
    for p in universe:
        probs[p] = (float(counts.get(p, 0)) + float(alpha)) / denom

    # mistura mínima uniforme (anti-colapso)
    u = 1.0 / K
    eps = max(0.0, min(0.25, float(eps)))  # teto canônico para não diluir demais
    for p in universe:
        probs[p] = (1.0 - eps) * probs[p] + eps * u

    # normalização final (segurança numérica)
    s = sum(probs.values())
    if s > 0:
        for p in universe:
            probs[p] /= s
    return probs

def pc_fill_lists_to_target(listas, target_n: int, universe_candidates: list, n_por_lista: int, seed: int):
    """FIX6 — garante mínimo determinístico de listas únicas (ex.: 10).
    Preenche com perturbações controladas privilegiando cauda (pouco usados),
    mas com TAILSTAB (Dirichlet+eps-uniforme) para não colapsar."""
    try:
        target_n = int(target_n)
        n_por_lista = int(n_por_lista)
    except Exception:
        return listas

    if target_n <= 0 or n_por_lista <= 0:
        return listas

    # já suficiente
    if listas is None:
        listas = []
    if len(listas) >= target_n:
        return listas

    # universo seguro
    uni = [int(x) for x in universe_candidates if isinstance(x, (int, float, str)) and str(x).strip().isdigit()]
    uni = sorted(set([u for u in uni if u > 0]))
    if len(uni) < n_por_lista:
        return listas

    # contagem de frequência (para empurrar cauda)
    freq = {}
    for lst in listas:
        try:
            for p in lst:
                freq[int(p)] = freq.get(int(p), 0) + 1
        except Exception:
            pass

    # prob proporcional ao "inverso" (cauda), mas estabilizada
    # base: score = 1/(1+freq)
    counts_tail = {p: 0 for p in uni}
    for p in uni:
        counts_tail[p] = max(0.0, 1.0 / (1.0 + float(freq.get(p, 0))))

    probs = pc_dirichlet_smooth_probs(counts_tail, uni, alpha=1.0, eps=0.02)

    rnd = random.Random(int(seed) % (2**32))

    # set de unicidade
    seen = set()
    for lst in listas:
        try:
            seen.add(tuple(sorted(int(x) for x in lst)))
        except Exception:
            pass

    # amostragem ponderada sem reposição por lista (via sorteio sequencial)
    def sample_weighted_without_replacement():
        pool = uni[:]
        w = [probs.get(p, 0.0) for p in pool]
        out = []
        for _ in range(n_por_lista):
            s = sum(w)
            if s <= 0:
                # fallback uniforme no restante
                idx = rnd.randrange(len(pool))
            else:
                r = rnd.random() * s
                acc = 0.0
                idx = 0
                for i, wi in enumerate(w):
                    acc += wi
                    if acc >= r:
                        idx = i
                        break
            out.append(pool.pop(idx))
            w.pop(idx)
        return sorted(out)

    guard = 0
    while len(listas) < target_n and guard < 500:
        guard += 1
        cand = sample_weighted_without_replacement()
        t = tuple(cand)
        if t in seen:
            continue
        seen.add(t)
        listas.append(cand)

    return listas

from datetime import datetime


def _m2_init_memoria() -> None:
    """Inicializa a memoria em session_state (infraestrutura invisivel)."""
    try:
        ss = st.session_state
        if "m2_memoria_estados" not in ss or not isinstance(ss.get("m2_memoria_estados"), list):
            ss["m2_memoria_estados"] = []
        if "m2_memoria_selo_s3" not in ss:
            ss["m2_memoria_selo_s3"] = set()
        if "m2_memoria_selo_s6" not in ss:
            ss["m2_memoria_selo_s6"] = set()
    except Exception:
        # falha silenciosa
        pass


def _m2_guess_serie_id(snapshot: dict) -> str:
    """Tenta inferir o id da serie atual. Regra: nunca falhar."""
    try:
        ss = st.session_state
        # Preferencias: chaves explicitas (quando existirem)
        for k in ("serie_id", "serie_atual", "serie_corrente", "concurso_atual", "c_atual"):
            v = ss.get(k)
            if v not in (None, "", "N/D"):
                return str(v)
        # Fallback: tamanho do historico
        df = ss.get("historico_df")
        if df is not None:
            try:
                n = int(len(df))
                return f"C{n}"
            except Exception:
                pass
        # ultimo fallback
        return "N/D"
    except Exception:
        return "N/D"


def _m2_build_registro_minimo(snapshot: dict, meta: dict) -> dict:
    """Registro minimo (obrigatorio) - comparavel e sem vies."""
    ss = st.session_state
    serie_id = _m2_guess_serie_id(snapshot)
    # normalizar universo
    umin = snapshot.get("universo_min", ss.get("universo_min", "N/D"))
    umax = snapshot.get("universo_max", ss.get("universo_max", "N/D"))
    # normalizar diagnostico
    regime = snapshot.get("regime", ss.get("regime_identificado", "N/D"))
    classe = snapshot.get("classe_risco", ss.get("classe_risco", "N/D"))
    nrp = snapshot.get("nr_percent", ss.get("nr_percent", "N/D"))
    div = snapshot.get("divergencia_s6_mc", ss.get("divergencia_s6_mc", "N/D"))

    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "serie_id": serie_id,
        "estado": str(meta.get("estado", "N/D")),
        "n_alvo": snapshot.get("n_alvo", ss.get("n_alvo", "N/D")),
        "universo_min": umin,
        "universo_max": umax,
        "regime": regime,
        "classe_risco": classe,
        "nr_percent": nrp,
        "divergencia_s6_mc": div,
    }


def _m2_build_registro_estendido(snapshot: dict, meta: dict) -> dict:
    """Registro estendido (opcional) - fechamento de rodada, sem acerto."""
    ss = st.session_state
    base = _m2_build_registro_minimo(snapshot, meta)

    # postura humana (se existir) - NUNCA vira gatilho automatico
    postura = ss.get("postura_humana", ss.get("checklist_decisao", "N/D"))

    # exposicao e execucao
    base.update({
        "houve_execucao": bool(snapshot.get("modo6_executado")),
        "turbo_tentado": bool(snapshot.get("turbo_tentado")),
        "turbo_bloqueado": bool(snapshot.get("turbo_bloqueado")),
        "turbo_motivo": snapshot.get("turbo_motivo", "N/D"),
        "postura_humana": postura,
        "listas_geradas": snapshot.get("listas_geradas", "<nao definido>"),
    })

    return base


def _m2_persistir_linha_jsonl(registro: dict) -> None:
    """Persistencia best-effort em JSONL. Falha silenciosa (Streamlit Cloud pode restringir)."""
    try:
        # arquivo local no diretorio do app (best-effort)
        path = "memoria_estados_v16.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _m2_registrar_minimo_se_preciso(snapshot: dict, meta: dict) -> None:
    """Gatilho canonico: registrar no primeiro S3 auditavel (sem duplicar)."""
    try:
        _m2_init_memoria()
        estado = str(meta.get("estado", ""))
        if estado not in ("S3", "S4", "S5", "S6"):
            return

        serie_id = _m2_guess_serie_id(snapshot)
        chave = f"{serie_id}|S3"
        selos = st.session_state.get("m2_memoria_selo_s3")
        if isinstance(selos, set) and chave in selos:
            return

        reg = _m2_build_registro_minimo(snapshot, {"estado": "S3"})
        st.session_state["m2_memoria_estados"].append(reg)
        if isinstance(selos, set):
            selos.add(chave)
        st.session_state["m2_ts_ultimo_registro_s3"] = reg.get("ts")

        _m2_persistir_linha_jsonl(reg)
    except Exception:
        pass


def _m2_registrar_fechamento_se_preciso(snapshot: dict, meta: dict) -> None:
    """Gatilho complementar: registrar fechamento (S6 heuristico), sem duplicar."""
    try:
        _m2_init_memoria()
        estado = str(meta.get("estado", ""))
        if estado != "S6":
            return

        serie_id = _m2_guess_serie_id(snapshot)
        chave = f"{serie_id}|S6"
        selos = st.session_state.get("m2_memoria_selo_s6")
        if isinstance(selos, set) and chave in selos:
            return

        reg = _m2_build_registro_estendido(snapshot, {"estado": "S6"})
        st.session_state["m2_memoria_estados"].append(reg)
        if isinstance(selos, set):
            selos.add(chave)
        st.session_state["m2_ts_ultimo_registro_s6"] = reg.get("ts")

        _m2_persistir_linha_jsonl(reg)
    except Exception:
        pass


def _m2_resumo_auditavel() -> dict:
    """Resumo leve para aparecer no Mirror (auditoria controlada)."""
    try:
        _m2_init_memoria()
        mem = st.session_state.get("m2_memoria_estados", [])
        total = len(mem) if isinstance(mem, list) else 0

        # contagens simples por estado
        cont = {}
        if isinstance(mem, list):
            for r in mem:
                try:
                    e = str(r.get("estado", "N/D"))
                    cont[e] = cont.get(e, 0) + 1
                except Exception:
                    pass

        # diversidade de contexto (n e universo)
        ns = set()
        universos = set()
        if isinstance(mem, list):
            for r in mem:
                try:
                    ns.add(str(r.get("n_alvo", "N/D")))
                    universos.add(f"{r.get('universo_min','N/D')}–{r.get('universo_max','N/D')}")
                except Exception:
                    pass

        return {
            "memoria_total": total,
            "por_estado": cont,
            "n_distintos": sorted(list(ns))[:8],
            "universos_distintos": sorted(list(universos))[:8],
            "ts_ultimo_s3": st.session_state.get("m2_ts_ultimo_registro_s3", "N/D"),
            "ts_ultimo_s6": st.session_state.get("m2_ts_ultimo_registro_s6", "N/D"),
        }
    except Exception:
        return {"memoria_total": "N/D"}

# ============================================================
# PARTE 2/8 — INÍCIO
# ============================================================

# ============================================================
# >>> FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# (UNIVERSAL — respeita o fenômeno detectado)
# ============================================================

def ajustar_ambiente_modo6(
    *,
    df,
    k_star,
    nr_pct,
    divergencia_s6_mc,
    risco_composto,
    previsibilidade="baixa",
):
    """
    Ajusta volumes do Modo 6 sem bloquear execução.
    Sempre retorna configuração válida.

    BLOCO UNIVERSAL C:
    - Não assume n = 6
    - Lê PC_N_EFETIVO e PC_UNIVERSO_ATIVO se existirem
    - Não força alteração de comportamento
    """

    # --------------------------------------------------------
    # Leitura do fenômeno ativo (Blocos A + B + C)
    # --------------------------------------------------------
    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    # --------------------------------------------------------
    # Valores base (comportamento LEGADO preservado)
    # --------------------------------------------------------
    volume_min = 3
    volume_recomendado = 6
    volume_max = 80

    # --------------------------------------------------------
    # Ajuste simples por previsibilidade (V16)
    # --------------------------------------------------------
    if previsibilidade == "alta":
        volume_min = 6
        volume_recomendado = 12
        volume_max = 40
    elif previsibilidade == "baixa":
        volume_min = 3
        volume_recomendado = 6
        volume_max = 20

    # --------------------------------------------------------
    # Ajuste UNIVERSAL SUAVE (não forçador)
    # --------------------------------------------------------
    aviso_universal = ""

    if pc_n_efetivo is not None:
        aviso_universal += f" | Fenômeno n={pc_n_efetivo}"

        # Regra conservadora:
        # quanto maior n, menor o volume máximo recomendado
        if pc_n_efetivo > 6:
            volume_max = min(volume_max, 20)
            volume_recomendado = min(volume_recomendado, 6)
            aviso_universal += " (redução preventiva)"

        elif pc_n_efetivo < 6:
            # Fenômenos menores toleram leve expansão
            volume_max = min(volume_max, 40)
            aviso_universal += " (fenômeno compacto)"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        aviso_universal += f" | Univ:{u_min}-{u_max}"

    # --------------------------------------------------------
    # Retorno PADRÃO (compatível com todo o app)
    # --------------------------------------------------------
    return {
        "volume_min": volume_min,
        "volume_recomendado": volume_recomendado,
        "volume_max": volume_max,
        "confiabilidade_estimada": 0.05,
        "aviso_curto": (
            f"Modo 6 ativo | Volumes: "
            f"{volume_min}/{volume_recomendado}/{volume_max}"
            f"{aviso_universal}"
        ),
    }

# ============================================================
# <<< FIM — FUNÇÃO AUXILIAR — AJUSTE DE AMBIENTE PARA MODO 6
# ============================================================


# ============================================================
# GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# NÃO decide | NÃO expande | NÃO altera volumes
# Apenas sinaliza prontidão para ECO
# (UNIVERSAL — consciente do fenômeno)
# ============================================================

def avaliar_gatilho_eco(
    k_star_atual: float,
    nr_pct: float,
    divergencia_s6_mc: float,
):
    """
    Avalia se o ambiente está tecnicamente pronto para ECO.
    BLOCO UNIVERSAL C:
    - Leitura do fenômeno ativo
    - Nenhuma decisão automática
    """

    pc_n_efetivo = st.session_state.get("PC_N_EFETIVO")
    pc_universo = st.session_state.get("PC_UNIVERSO_ATIVO")

    pronto_eco = False
    motivos = []

    # --------------------------------------------------------
    # Critérios técnicos (LEGADOS)
    # --------------------------------------------------------
    if k_star_atual < 0.22:
        motivos.append("k* favorável")

    if nr_pct < 0.30:
        motivos.append("ruído controlado")

    if divergencia_s6_mc < 5.0:
        motivos.append("baixa divergência S6 vs MC")

    if len(motivos) >= 2:
        pronto_eco = True

    # --------------------------------------------------------
    # Informação universal (observacional)
    # --------------------------------------------------------
    info_universal = ""

    if pc_n_efetivo is not None:
        info_universal += f" | Fenômeno n={pc_n_efetivo}"

    if pc_universo is not None:
        u_min, u_max = pc_universo
        info_universal += f" | Univ:{u_min}-{u_max}"

    return {
        "pronto_eco": pronto_eco,
        "motivos": motivos,
        "mensagem": (
            "ECO tecnicamente possível"
            if pronto_eco
            else "ECO ainda não recomendado"
        )
        + info_universal,
    }

# ============================================================
# <<< FIM — GATILHO ECO — OBSERVADOR PASSIVO (V16 PREMIUM)
# ============================================================






# ============================================================
# MÓDULO 3 — EXPECTATIVA HISTÓRICA (CONTEXTO DO MOMENTO) — V16
# Observacional | Retrospectivo | NÃO decide | NÃO gera listas
# Depende de S2 (Pipeline) + S3 (Diagnóstico de Risco mínimo: k*/NR%/classe)
# ============================================================

M3_PAINEL_EXPECTATIVA_NOME = "📈 Expectativa Histórica — Contexto do Momento (V16)"

# ============================================================
# MÓDULO 5 (V16) — “PULO DO GATO” OPERACIONAL
# Coleta automática de estados (para dar massa ao M2/M3)
# ============================================================

M5_PAINEL_PULO_GATO_NOME = "🧠 M5 — Pulo do Gato (Coleta Automática de Estados)"


def _m5_identidade_historico_para_coleta(_df_full_safe, n_alvo, universo_min, universo_max):
    """ID estável (best-effort) para limitar coleta por histórico sem depender de hash pesado."""
    try:
        tam = _df_full_len if _df_full_safe is not None else -1
    except Exception:
        tam = -1
    return f"H|n={n_alvo}|U={universo_min}-{universo_max}|len={tam}"


def _m5_leitura_regime_light(df_cut, universo_min, universo_max):
    """Leitura LIGHT (rápida) para regime/volatilidade sem rodar pipeline completo.

    Objetivo: registrar um *sinal* coerente para M2 (não substituir o pipeline).
    """
    try:
        # janela curta para captar irregularidade recente
        w = min(120, max(30, int(len(df_cut) * 0.158)))
        dfw = df_cut.tail(w)
        # tenta extrair colunas numéricas de passageiros
        cols_num = [c for c in dfw.columns if str(c).strip().isdigit()]
        if not cols_num:
            # fallback: tenta padrão comum (p1..p6)
            cols_num = [c for c in dfw.columns if str(c).lower().startswith("p")]
        vals = []
        for _, row in dfw.iterrows():
            linha = []
            for c in cols_num:
                try:
                    v = int(row[c])
                    if universo_min <= v <= universo_max:
                        linha.append(v)
                except Exception:
                    pass
            vals.append(len(set(linha)))
        if not vals:
            return "N/D", None, None

        # “energia” simples = diversidade média normalizada
        u = max(1, (universo_max - universo_min + 1))
        energia = float(np.mean(vals)) / float(min(u, 60))
        # “volatilidade” simples = desvio da diversidade
        volatilidade = float(np.std(vals)) / float(max(1, np.mean(vals)))

        if volatilidade >= 0.28:
            regime = "🟥 Estrada Quente (Alta volatilidade)"
        elif volatilidade >= 0.18:
            regime = "🟨 Estrada Mista / Irregular"
        else:
            regime = "🟩 Estrada Neutra / Estável"

        return regime, energia, volatilidade
    except Exception:
        return "N/D", None, None


def m5_painel_pulo_do_gato_v16():
    """Painel canônico: coleta automática de estados para dar massa mínima ao M2/M3.

    - Não mexe no núcleo
    - Não decide
    - Não substitui pipeline/monitor
    """
    st.subheader("🧠 M5 — Pulo do Gato (Coleta Automática de Estados)")
    st.caption("Coleta assistida para preencher Memória de Estados (M2) e habilitar Expectativa Histórica (M3) sem exigir que você rode manualmente dezenas de vezes.")

    _df_full_safe = st.session_state.get("historico_df")
    if _df_full_safe is None or len(_df_full_safe) < 50:
        st.warning("Carregue um histórico válido antes de usar o M5.")
        return

    n_alvo = int(st.session_state.get("n_alvo", 6) or 6)
    universo_min = int(st.session_state.get("universo_min", 0) or 0)
    universo_max = int(st.session_state.get("universo_max", 50) or 50)

    # limites (anti-zumbi canônico do M5)
    max_sessao = int(st.session_state.get("m5_max_por_sessao", 25) or 25)
    max_hist = int(st.session_state.get("m5_max_por_historico", 50) or 50)

    st.markdown("""
**Como funciona (sem mágica):**

- Você escolhe quantas “fotos” o sistema deve coletar.
- Cada foto simula um ponto de corte do histórico (C... menor) e registra **um estado S3** com um *regime light*.
- Isso alimenta o M2; o M3 passa a ter base mínima para expectativa histórica.

**Importante:**

- O M5 **não** roda o Pipeline completo em cada corte (para não zumbizar).
- Ele registra um sinal *light* e honesto para memória.
""")

    restante_sessao = max(0, max_sessao - int(st.session_state.get("m5_contador_sessao", 0) or 0))
    hist_id = _m5_identidade_historico_para_coleta(_df_full_safe, n_alvo, universo_min, universo_max)
    por_hist = st.session_state.get("m5_contador_por_historico", {})
    restante_hist = max(0, max_hist - int(por_hist.get(hist_id, 0) or 0))

    st.info(f"Limites ativos: Máx. por sessão: {max_sessao} (restante: {restante_sessao}) · Máx. por histórico: {max_hist} (restante: {restante_hist})")

    if restante_sessao <= 0 or restante_hist <= 0:
        st.warning("Limite atingido. Para coletar mais: reinicie a sessão (para limite por sessão) ou troque o histórico (para limite por histórico).")
        return

    n_solicitado = st.slider("Quantas fotos coletar agora?", 1, min(25, restante_sessao, restante_hist), 12)
    passo = st.slider("Passo entre cortes (em séries)", 1, 10, 1)
    janela_recente = st.slider("Janela recente para coleta (últimas séries)", 60, 600, 180, step=30)

    btn = st.button("📸 Coletar agora (M5)")

    if not btn:
        return

    # prepara M2
    if "m2_memoria_estados" not in st.session_state:
        st.session_state["m2_memoria_estados"] = []

    # pontos de corte: fatiamos a janela recente para não varrer o histórico inteiro (anti-zumbi)
    total = len(_df_full_safe)
    base_ini = max(50, total - int(janela_recente))
    pontos = list(range(total, base_ini, -int(passo)))
    pontos = pontos[: int(n_solicitado)]

    adicionados = 0
    falhas = 0

    for cut_len in pontos:
        try:
            df_cut = _df_full_safe.iloc[:cut_len].copy()
            regime_light, energia_light, vol_light = _m5_leitura_regime_light(df_cut, universo_min, universo_max)

            registro = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "n_alvo": int(n_alvo),
                "universo": f"{universo_min}–{universo_max}",
                "tamanho": int(cut_len),
                "estado": "S3",
                "regime": regime_light,
                "energia_light": energia_light,
                "volatilidade_light": vol_light,
                "origem": "M5",
            }
            st.session_state["m2_memoria_estados"].append(registro)
            adicionados += 1
        except Exception:
            falhas += 1
            continue

    # atualiza contadores
    st.session_state["m5_contador_sessao"] = int(st.session_state.get("m5_contador_sessao", 0) or 0) + int(adicionados)
    por_hist = st.session_state.get("m5_contador_por_historico", {})
    por_hist[hist_id] = int(por_hist.get(hist_id, 0) or 0) + int(adicionados)
    st.session_state["m5_contador_por_historico"] = por_hist

    st.success(f"M5 concluído: {adicionados} fotos adicionadas à Memória de Estados (M2).")
    if falhas:
        st.caption(f"Falhas silenciosas (válidas): {falhas}")

    st.markdown("""
**Próximo passo canônico:**

1) Abra **📈 Expectativa Histórica — Contexto do Momento (V16)**
2) Veja se o N mínimo já foi atingido.
""")


# ============================================================
# M5 — ALIAS CANÔNICO (compatibilidade de chamada)
# ============================================================

def m5_painel_pulo_do_gato_operacional():
    """Painel canônico (nome histórico).

    Mantém compatibilidade com chamadas antigas/âncoras que apontam para
    `m5_painel_pulo_do_gato_operacional()`.
    """
    return m5_painel_pulo_do_gato_v16()


def _m3_has_s2_pipeline() -> bool:
    return bool(st.session_state.get("pipeline_flex_ultra_concluido") or st.session_state.get("pipeline_executado") or st.session_state.get("m1_ts_pipeline_ok"))


def _m3_has_s3_risco_minimo() -> bool:
    risco = st.session_state.get("diagnostico_risco") or {}

    k_star = risco.get("k_star", risco.get("kstar", None))
    nr = risco.get("nr_percent", None)
    classe = risco.get("classe_risco", None)

    # fallback: algumas chaves aparecem fora do pack
    if k_star is None:
        k_star = st.session_state.get("k_star")
    if nr is None:
        nr = st.session_state.get("nr_percent")
    if classe is None:
        classe = st.session_state.get("classe_risco")

    ok_k = isinstance(k_star, (int, float))
    ok_nr = isinstance(nr, (int, float))
    ok_classe = isinstance(classe, str) and classe.strip() not in ("", "N/D")

    return bool(ok_k and ok_nr and ok_classe)


def _m3_norm_int(v):
    try:
        return int(float(str(v).strip().replace(",", ".")))
    except Exception:
        return None


def _m3_dx_janela(df_window, cols_pass):
    vals = []
    for c in cols_pass:
        s = [_m3_norm_int(x) for x in df_window[c].values]
        s = [x for x in s if x is not None]
        if len(s) >= 2:
            try:
                vals.append(float(np.std(s, ddof=1)))
            except Exception:
                pass
    if not vals:
        return None
    try:
        return float(np.mean(vals))
    except Exception:
        return None


def _m3_classificar_regime_dx(dx, q1, q2):
    if dx is None:
        return "N/D"
    if dx <= q1:
        return "ECO"
    if dx <= q2:
        return "PRÉ-ECO"
    return "RUIM"


def m3_painel_expectativa_historica_contexto():
    st.markdown("## 📈 Expectativa Histórica — Contexto do Momento (V16)")
    st.caption(
        "Observacional e retrospectivo. Não gera listas. Não muda motores. "
        "Serve para responder: *em momentos parecidos no passado, o que costuma acontecer nas próximas 1–3 séries?*"
    )

    # --------------------------------------------------------
    # Governança (dependências)
    # --------------------------------------------------------
    # ============================================================
    # 📤 Auditoria externa (opcional) — NÃO é necessária no uso normal
    # ============================================================
    with st.expander("📤 Auditoria externa (opcional) — usar outro histórico (sem substituir a sessão)", expanded=False):
        st.caption("Uso normal: este painel usa o histórico já carregado na sessão.\n"
                   "Este uploader é apenas para auditoria/estudo com outro arquivo, sem afetar o histórico atual.")
        arquivo = st.file_uploader("Envie um histórico FLEX ULTRA (opcional)", type=["csv", "txt"], key="m3_upload_auditoria")
        if arquivo is not None:
            try:
                df_aud = carregar_historico_flex_ultra(arquivo)
                st.success("Histórico de auditoria carregado (não substitui a sessão).")
                metricas = calcular_metricas_basicas_historico(df_aud)
                exibir_resumo_inicial_historico(metricas)
                st.info("✅ Auditoria concluída. Para operar o fluxo normal, use o histórico carregado em 📁/📄 Carregar Histórico.")
            except Exception as e:
                st.error(f"Falha ao carregar histórico de auditoria: {e}")

    # --------------------------------------------------------
    # 1) Calcula dx nas janelas recentes para quantis
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}

    # --------------------------------------------------------
    # Fonte CANÔNICA: histórico da sessão + colunas do Pipeline
    # (M3 é observacional: não gera listas, não decide)
    # --------------------------------------------------------
    if "pipeline_col_pass" not in st.session_state:
        st.warning("Execute primeiro o painel 🛣️ Pipeline V14-FLEX ULTRA (fonte canônica de passageiros).")
        return

    cols_pass = st.session_state["pipeline_col_pass"]
    nome_df, df_base = v16_identificar_df_base()
    if df_base is None:
        st.warning("Histórico não encontrado. Carregue o histórico e rode o Pipeline.")
        return

    # Janela canônica (alinhada aos painéis V16 Premium PRÉ-ECO/ECO)
    W = 60
    n = int(len(df_base))
    if n < (W + 5):
        st.warning("Histórico insuficiente para Expectativa Histórica (M3).")
        return

    # Intervalo automático (sem sliders): recorte recente suficiente para quantis e estabilidade
    t_final = n - 1
    max_janelas = min(600, max(180, n - (W + 2)))  # 180–600 janelas, conforme tamanho do histórico
    t_inicial = max(W, t_final - max_janelas)

    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = _m3_dx_janela(wdf, cols_pass)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 120:
        st.warning("⚠️ Poucas janelas válidas para estimar quantis com estabilidade. (Resultados ainda são informativos.)")

    try:
        q1 = float(np.quantile(dx_list, 0.33))
        q2 = float(np.quantile(dx_list, 0.66))
    except Exception:
        st.error("❌ Falha ao calcular quantis do dx (dados insuficientes ou inválidos).")
        return

    dx_atual = dx_por_t.get(t_final)
    regime_atual = _m3_classificar_regime_dx(dx_atual, q1, q2)

    # --------------------------------------------------------
    # 2) Regime por t (mesma regra do painel PRÉ-ECO→ECO)
    # --------------------------------------------------------
    regime_por_t = {}
    for t, dx in dx_por_t.items():
        regime_por_t[t] = _m3_classificar_regime_dx(dx, q1, q2)

    # --------------------------------------------------------
    # 3) Expectativa: quando estava no MESMO regime, o que ocorreu em 1–3 séries?
    # --------------------------------------------------------
    total = 0
    vira_eco_1 = 0
    vira_eco_2 = 0
    vira_eco_3 = 0

    permanece_mesmo_1 = 0
    permanece_mesmo_2 = 0
    permanece_mesmo_3 = 0

    for t in sorted(regime_por_t.keys()):
        if t + 1 > t_final:
            continue
        if regime_por_t[t] != regime_atual:
            continue

        total += 1

        r1 = regime_por_t.get(t + 1)
        r2 = regime_por_t.get(t + 2)
        r3 = regime_por_t.get(t + 3)

        if r1 == "ECO":
            vira_eco_1 += 1
        if (r1 == "ECO") or (r2 == "ECO"):
            vira_eco_2 += 1
        if (r1 == "ECO") or (r2 == "ECO") or (r3 == "ECO"):
            vira_eco_3 += 1

        if r1 == regime_atual:
            permanece_mesmo_1 += 1
        if (r1 == regime_atual) or (r2 == regime_atual):
            permanece_mesmo_2 += 1
        if (r1 == regime_atual) or (r2 == regime_atual) or (r3 == regime_atual):
            permanece_mesmo_3 += 1

    # --------------------------------------------------------
    # 4) Exibição
    # --------------------------------------------------------
    st.markdown("### 🧭 Momento atual (classificação por dx)")

    colA, colB, colC = st.columns(3)
    colA.metric("dx (janela)", f"{dx_atual:.6f}" if isinstance(dx_atual, (int, float)) else "N/D")
    colB.metric("Regime (dx)", regime_atual)
    colC.metric("Janelas analisadas", f"{len(dx_list)}")

    st.caption(
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM. "
        "(Quantis calculados nas últimas janelas, com anti-zumbi interno.)"
    )

    st.markdown("### 📈 Expectativa histórica (condicional ao regime atual)")

    if total == 0:
        st.warning("⚠️ Não houve eventos suficientes no histórico para estimar expectativa condicional neste regime.")
        return

    taxa_eco_1 = vira_eco_1 / total
    taxa_eco_2 = vira_eco_2 / total
    taxa_eco_3 = vira_eco_3 / total

    taxa_perm_1 = permanece_mesmo_1 / total
    taxa_perm_2 = permanece_mesmo_2 / total
    taxa_perm_3 = permanece_mesmo_3 / total

    df_out = pd.DataFrame(
        [
            {
                "Regime atual (dx)": regime_atual,
                "Eventos similares": int(total),
                "Vira ECO em 1": round(taxa_eco_1, 4),
                "Vira ECO em 2": round(taxa_eco_2, 4),
                "Vira ECO em 3": round(taxa_eco_3, 4),
                "Permanece no mesmo (1)": round(taxa_perm_1, 4),
                "Permanece no mesmo (2)": round(taxa_perm_2, 4),
                "Permanece no mesmo (3)": round(taxa_perm_3, 4),
            }
        ]
    )

    st.dataframe(df_out, use_container_width=True, hide_index=True)



    # --- M3: exporta um resumo mínimo para uso em outros painéis (read-only)

    try:

        st.session_state["m3_ts"] = datetime.utcnow().isoformat() + "Z"

        st.session_state["m3_regime_dx"] = regime_atual

        st.session_state["m3_eventos_similares"] = int(total)

        st.session_state["m3_taxa_eco1"] = float(taxa_eco_1)

        st.session_state["m3_taxa_estado_bom"] = float(taxa_estado_bom)

        st.session_state["m3_taxa_transicao"] = float(taxa_transicao)

    except Exception:

        pass
    st.info("📌 Interpretação correta (sem viés):\n- Isso NÃO prevê o próximo alvo.\n- Isso mede *o que costuma acontecer* quando o ambiente cai no mesmo tipo de regime.\n- Serve para calibrar expectativa, postura e paciência — não para aumentar convicção por '3 acertos'.")

    st.markdown(
        "Envie um arquivo de histórico em formato **FLEX ULTRA**.\n\n"
        "📌 Regra universal: o **último valor da linha é sempre k**, "
        "independente da quantidade de passageiros."
    )

    arquivo = st.file_uploader(
        "Envie o arquivo de histórico",
        type=["txt", "csv"],
    )

    if arquivo is None:
        exibir_bloco_mensagem(
            "Aguardando arquivo de histórico",
            "Envie seu arquivo para iniciar o processamento do PredictCars.",
            tipo="info",
        )
        st.stop()

    try:
        conteudo = arquivo.getvalue().decode("utf-8")
        linhas = conteudo.strip().split("\n")

        if not limitar_operacao(
            len(linhas),
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Carregar Histórico (Arquivo)",
            painel="📁 Carregar Histórico (Arquivo)",
        ):
            st.stop()

        df = carregar_historico_universal(linhas)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro ao processar histórico",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    st.session_state["historico_df"] = df

    # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF

    v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

    metricas = calcular_metricas_basicas_historico(df)
    exibir_resumo_inicial_historico(metricas)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL A — DETECTOR DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Perfil do Fenômeno (detecção automática)")
    st.caption(
        "Detecção automática do formato real do fenômeno.\n"
        "✔ Última coluna = k\n"
        "✔ Quantidade de passageiros livre\n"
        "✔ Universo variável\n"
        "❌ Não há decisão automática"
    )

    import hashlib

    colunas = list(df.columns)
    col_id = colunas[0]
    col_k = colunas[-1]
    col_passageiros = colunas[1:-1]

    passageiros_por_linha = []
    todos_passageiros = []

    for _, row in df.iterrows():
        valores = [int(v) for v in row[col_passageiros] if pd.notna(v)]
        passageiros_por_linha.append(len(valores))
        todos_passageiros.extend(valores)

    n_set = sorted(set(passageiros_por_linha))
    mix_n_detectado = len(n_set) > 1
    n_passageiros = n_set[0] if not mix_n_detectado else None

    universo_min = int(min(todos_passageiros)) if todos_passageiros else None
    universo_max = int(max(todos_passageiros)) if todos_passageiros else None
    universo_set = sorted(set(todos_passageiros))

    hash_base = f"{n_set}-{universo_min}-{universo_max}"
    fenomeno_id = hashlib.md5(hash_base.encode()).hexdigest()[:8]

    st.session_state["pc_n_passageiros"] = n_passageiros
    st.session_state["pc_n_set_detectado"] = n_set
    st.session_state["pc_mix_n_detectado"] = mix_n_detectado
    st.session_state["pc_universo_min"] = universo_min
    st.session_state["pc_universo_max"] = universo_max
    st.session_state["pc_universo_set"] = universo_set
    st.session_state["pc_fenomeno_id"] = fenomeno_id

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📐 Estrutura**")
        st.write(f"Passageiros por série (n): **{n_set}**")
        if mix_n_detectado:
            st.warning("Mistura de n detectada no mesmo histórico.")
        st.write(f"Coluna ID: `{col_id}`")
        st.write(f"Coluna k: `{col_k}`")

    with col2:
        st.markdown("**🌍 Universo observado**")
        st.write(f"Mínimo: **{universo_min}**")
        st.write(f"Máximo: **{universo_max}**")
        st.write(f"Total distintos: **{len(universo_set)}**")

    st.markdown("**🆔 Fenômeno ID (auditoria)**")
    st.code(fenomeno_id)

    # ============================================================
    # 🌐 BLOCO UNIVERSAL B — PARAMETRIZAÇÃO DO FENÔMENO
    # ============================================================

    st.markdown("### 🌐 Parâmetros Ativos do Fenômeno")
    st.caption(
        "Parâmetros universais derivados do histórico.\n"
        "✔ Não executa\n"
        "✔ Não interfere\n"
        "✔ Não altera módulos existentes"
    )

    if not mix_n_detectado:
        pc_n_alvo = n_passageiros
        pc_n_status = "fixo"
    else:
        pc_n_alvo = None
        pc_n_status = "misto"

    st.session_state["pc_n_alvo"] = pc_n_alvo
    st.session_state["pc_range_min"] = universo_min
    st.session_state["pc_range_max"] = universo_max

    if pc_n_alvo:
        st.session_state["pc_regua_extrema"] = f"{pc_n_alvo} ou nada"
        st.session_state["pc_regua_mvp2"] = f"2–{pc_n_alvo}"
    else:
        st.session_state["pc_regua_extrema"] = "indefinida"
        st.session_state["pc_regua_mvp2"] = "indefinida"

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🎯 n alvo**")
        st.write(f"Status: **{pc_n_status}**")
        st.write(f"n alvo: **{pc_n_alvo if pc_n_alvo else 'MISTO'}**")

    with col4:
        st.markdown("**📏 Universo ativo**")
        st.write(f"{universo_min} – {universo_max}")
        st.write("Origem: histórico observado")

    if mix_n_detectado:
        st.warning(
            "⚠️ Histórico contém mistura de quantidades de passageiros.\n\n"
            "Recomenda-se separar fenômenos antes de previsões."
        )

    if pc_n_alvo and pc_n_alvo != 6:
        st.info(
            f"ℹ️ Fenômeno com n = {pc_n_alvo} detectado.\n"
            "Módulos legados ainda podem assumir n=6.\n"
            "➡️ Próximo passo: BLOCO UNIVERSAL C."
        )

    st.success("Perfil e parâmetros do fenômeno definidos.")

    st.success("Histórico carregado com sucesso!")
    st.dataframe(df.head(20))


# ============================================================

# ============================================================
# Painel 1A — 📁 Carregar Histórico (Arquivo)
# ============================================================


def v16_painel_mc_observacional_pacote_pre_c4():
    """
    🧪 MC Observacional do Pacote (pré-C4)
    Observacional, auditável. NÃO altera Camada 4.
    Usa df_eval (Replay Progressivo / avaliações) já calculado.
    """
    import numpy as np
    import pandas as pd
    import random
    import math

    st.title("🧪 MC Observacional do Pacote (pré-C4)")
    st.caption("Observacional • auditável • não altera listas • não altera Camada 4.")

    # Fonte canônica: df_eval salvo pelo Replay Progressivo
    df_eval = st.session_state.get("df_eval", None)
    if df_eval is None or not hasattr(df_eval, "columns") or len(df_eval) == 0:
        st.warning("Não encontrei `df_eval` na sessão. Rode primeiro **🧭 Replay Progressivo — Janela Móvel (Assistido)** para gerar a base de avaliação, e depois volte aqui.")
        return

    cols_needed = ["best_acerto_alvo_1", "best_acerto_alvo_2"]
    for c in cols_needed:
        if c not in df_eval.columns:
            st.warning("O `df_eval` encontrado não tem as colunas esperadas para MC observacional (best_acerto_alvo_1/2). Rode novamente o Replay Progressivo com a versão atual do app.")
            return

    # 1) Constrói a base de alvos avaliados (flatten dos 2 alvos por snapshot)
    hits = []
    for c in cols_needed:
        vals = df_eval[c].tolist()
        for v in vals:
            if v is None:
                continue
            try:
                vv = int(v)
            except Exception:
                continue
            hits.append(vv)

    if len(hits) == 0:
        st.warning("`df_eval` existe, mas não há alvos válidos avaliados ainda (hits vazios).")
        return

    # 2) Métricas objetivas
    def _rates(arr):
        arr = np.asarray(arr, dtype=float)
        out = {}
        out["n"] = int(len(arr))
        out["avg_best"] = float(np.nanmean(arr))
        out["max_best"] = int(np.nanmax(arr))
        out["rate_3p"] = float(np.mean(arr >= 3))
        out["rate_4p"] = float(np.mean(arr >= 4))
        out["rate_5p"] = float(np.mean(arr >= 5))
        out["rate_6p"] = float(np.mean(arr >= 6))
        out["dist"] = {str(i): int(np.sum(arr == i)) for i in range(0,7)}
        return out

    base = _rates(hits)

    st.subheader("📌 Base avaliada (flatten dos alvos)")
    st.json({
        "targets_avaliados": base["n"],
        "avg_best": round(base["avg_best"], 4),
        "max_best": base["max_best"],
        "rate_3p": round(base["rate_3p"], 6),
        "rate_4p": round(base["rate_4p"], 6),
        "rate_5p": round(base["rate_5p"], 6),
        "rate_6p": round(base["rate_6p"], 6),
        "dist_best_hit_0_6": base["dist"],
    })


    # ----------------------------------------
    # Split observacional (baseline interno) por calibração
    # Nota: isso NÃO cria lista nova; apenas separa os hits já avaliados no Replay.
    if "calib_applied" in df_eval.columns:
        try:
            df_on = df_eval[df_eval["calib_applied"] == True].copy()
            df_off = df_eval[df_eval["calib_applied"] == False].copy()

            def _collect_hits(_df):
                _hits = []
                for _c in cols_needed:
                    for _v in _df[_c].tolist():
                        if _v is None:
                            continue
                        try:
                            _hits.append(int(_v))
                        except Exception:
                            continue
                return _hits

            hits_on = _collect_hits(df_on) if len(df_on) else []
            hits_off = _collect_hits(df_off) if len(df_off) else []

            if len(hits_on) > 0 or len(hits_off) > 0:
                st.subheader("🔎 Split observacional — baseline interno (por calib_applied)")
                st.caption("Sem depender de nada fora do Predicar: comparamos 'pacotes com calib_applied=True' vs 'False' dentro do mesmo Replay/SAFE.")
                if len(hits_off) > 0:
                    base_off = _rates(hits_off)
                    st.json({
                        "grupo": "BASELINE_INTERNO (calib_applied=False)",
                        "targets_avaliados": base_off["n"],
                        "avg_best": round(base_off["avg_best"], 4),
                        "max_best": base_off["max_best"],
                        "rate_3p": round(base_off["rate_3p"], 6),
                        "rate_4p": round(base_off["rate_4p"], 6),
                        "rate_5p": round(base_off["rate_5p"], 6),
                        "rate_6p": round(base_off["rate_6p"], 6),
                        "dist_best_hit_0_6": base_off["dist"],
                    })
                else:
                    st.info("Não há amostras suficientes com calib_applied=False neste Replay.")

                if len(hits_on) > 0:
                    base_on = _rates(hits_on)
                    st.json({
                        "grupo": "CALIB_ATIVA (calib_applied=True)",
                        "targets_avaliados": base_on["n"],
                        "avg_best": round(base_on["avg_best"], 4),
                        "max_best": base_on["max_best"],
                        "rate_3p": round(base_on["rate_3p"], 6),
                        "rate_4p": round(base_on["rate_4p"], 6),
                        "rate_5p": round(base_on["rate_5p"], 6),
                        "rate_6p": round(base_on["rate_6p"], 6),
                        "dist_best_hit_0_6": base_on["dist"],
                    })
                else:
                    st.info("Não há amostras suficientes com calib_applied=True neste Replay.")
        except Exception as _e:
            st.warning(f"Falha ao gerar split observacional por calib_applied: {_e}")

    # ------------------------------------------------------------
    # AUDIT — calibração leve (resumo dos pacotes registrados)
    # ------------------------------------------------------------
    try:
        pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {}) or {}
        calib_items = []
        for _k, _v in pacotes_reg.items():
            c = (_v or {}).get("calib_leve")
            if isinstance(c, dict):
                calib_items.append(c)
        if calib_items:
            n_tot = len(calib_items)
            n_active = sum(1 for c in calib_items if bool(c.get("active", True)))
            n_aplic = sum(1 for c in calib_items if bool(c.get("aplicada_no_pacote", c.get("applied", False))))
            # I / I2 / insumos — auditoria robusta (aceita metadados legados)
            def _get_first(d, keys, default=0.0):
                if not isinstance(d, dict):
                    return default
                for k in keys:
                    if k in d and d.get(k) is not None:
                        try:
                            return float(d.get(k))
                        except Exception:
                            pass
                return default

            def _get_nested(d, outer_keys, inner_keys, default=0.0):
                if not isinstance(d, dict):
                    return default
                for ok in outer_keys:
                    subd = d.get(ok)
                    if isinstance(subd, dict):
                        v = _get_first(subd, inner_keys, default=None)
                        if v is not None:
                            return v
                return default

            # I "canônico" — aceitar várias chaves que já existiram em builds anteriores
            I_keys = ["I", "I_mean", "I_media", "I_canon", "I_can", "I_val", "I_score", "score_I", "compressao_I"]
            I_vals = [ _get_first(c, I_keys, default=_get_nested(c, ["insumos","mirror","metrics"], I_keys, default=0.0)) for c in calib_items ]
            I_med = (sum(I_vals) / float(n_tot)) if n_tot else 0.0
            I_min = min(I_vals) if I_vals else 0.0
            I_max = max(I_vals) if I_vals else 0.0

            # I2 (contraste topo×borda)
            I2_keys = ["I2", "I2_mean", "I2_media", "I2_val", "I2_score", "contraste_I2", "contrast_I2"]
            I2_vals = [ _get_first(c, I2_keys, default=_get_nested(c, ["insumos","mirror","metrics"], I2_keys, default=0.0)) for c in calib_items ]
            I2_med = (sum(I2_vals) / float(n_tot)) if n_tot else 0.0
            I2_min = min(I2_vals) if I2_vals else 0.0
            I2_max = max(I2_vals) if I2_vals else 0.0

            # Insumos (C_top / Slope / Stab / Gap) — aceitar nomes legados e/ou bloco aninhado
            C_keys = ["C_top", "Ctop", "C_top_z", "Cz", "C"]
            S_keys = ["Slope", "slope", "slope_top_borda", "Top6_minus_Borda", "delta_top_borda"]
            T_keys = ["Stab", "StabTop6", "stab_top6", "stability", "T"]
            G_keys = ["Gap", "gap", "gap_6_15", "gap_6_15_norm", "G"]

            C_vals = [ _get_first(c, C_keys, default=_get_nested(c, ["insumos","mirror","metrics"], C_keys, default=0.0)) for c in calib_items ]
            S_vals = [ _get_first(c, S_keys, default=_get_nested(c, ["insumos","mirror","metrics"], S_keys, default=0.0)) for c in calib_items ]
            T_vals = [ _get_first(c, T_keys, default=_get_nested(c, ["insumos","mirror","metrics"], T_keys, default=0.0)) for c in calib_items ]
            G_vals = [ _get_first(c, G_keys, default=_get_nested(c, ["insumos","mirror","metrics"], G_keys, default=0.0)) for c in calib_items ]
            C_med = (sum(C_vals) / float(n_tot)) if n_tot else 0.0
            S_med = (sum(S_vals) / float(n_tot)) if n_tot else 0.0
            T_med = (sum(T_vals) / float(n_tot)) if n_tot else 0.0
            G_med = (sum(G_vals) / float(n_tot)) if n_tot else 0.0

            st.subheader("🧩 Auditoria — Calibração Leve (pré-C4)")
            st.json({
                "pacotes_registrados": int(n_tot),
                "active_em": f"{n_active}/{n_tot}",
                "aplicada_em": f"{n_aplic}/{n_tot}",
                "I_media": round(float(I_med), 6),
                "I_min": round(float(I_min), 6),
                "I_max": round(float(I_max), 6),
                "I2_media": round(float(I2_med), 6),
                "I2_min": round(float(I2_min), 6),
                "I2_max": round(float(I2_max), 6),
                "insumos_media": {
                    "C_top": round(float(C_med), 6),
                    "Slope": round(float(S_med), 6),
                    "Stab": round(float(T_med), 6),
                    "Gap": round(float(G_med), 6),
                },
                "nota": "I>0 indica que a calibração foi calculada; 'aplicada_em' indica que ela influenciou o sorteio do pacote (somente leitura)."
            })
        else:
            st.subheader("🧩 Auditoria — Calibração Leve (pré-C4)")
            st.info("Sem metadados de calibração nos pacotes desta sessão (ok se o Replay ainda não registrou pacotes).")
    except Exception:
        pass


    # 3) Janela móvel (alvos) — padrão 60 (mesmo espírito do sistema)
    st.subheader("🪟 Janela móvel (alvos) — MC observacional")
    w_default = 60
    w = st.number_input("Tamanho da janela (alvos, não séries)", min_value=20, max_value=240, value=w_default, step=5)
    hits_w = hits[-int(w):] if len(hits) >= int(w) else hits[:]
    win = _rates(hits_w)
    st.json({
        "w_used": win["n"],
        "avg_best_w": round(win["avg_best"], 4),
        "max_best_w": win["max_best"],
        "rate_3p_w": round(win["rate_3p"], 6),
        "rate_4p_w": round(win["rate_4p"], 6),
        "rate_5p_w": round(win["rate_5p"], 6),
        "rate_6p_w": round(win["rate_6p"], 6),
        "dist_best_hit_0_6_w": win["dist"],
    })

    # 4) MC (bootstrap) — "foi sorte?" (incerteza estatística da janela)
    st.subheader("🎲 MC Bootstrap — Foi sorte ou é sinal?")
    B = st.number_input("Rodadas MC (bootstrap)", min_value=200, max_value=10000, value=2000, step=200)
    B = int(B)
    rng = random.Random(1337)

    arr = np.asarray(hits_w, dtype=float)
    n = len(arr)

    if n < 20:
        st.warning("Janela pequena demais para bootstrap informativo. Aumente `w`.")
        return

    rates4 = np.empty(B, dtype=float)
    avgs = np.empty(B, dtype=float)

    for i in range(B):
        # amostra com reposição
        idxs = [rng.randrange(0, n) for _ in range(n)]
        sample = arr[idxs]
        rates4[i] = float(np.mean(sample >= 4))
        avgs[i] = float(np.mean(sample))

    def _ci(x, lo=0.05, hi=1.05):
        return float(np.quantile(x, lo)), float(np.quantile(x, hi))

    r4_lo, r4_hi = _ci(rates4, 0.05, 1.05)
    av_lo, av_hi = _ci(avgs, 0.05, 1.05)

    st.markdown("**Intervalos (90%) na janela** — quanto isso pode oscilar só por variação amostral:")
    st.json({
        "rate_4p_w": round(win["rate_4p"], 6),
        "rate_4p_w_CI90": [round(r4_lo, 6), round(r4_hi, 6)],
        "avg_best_w": round(win["avg_best"], 4),
        "avg_best_w_CI90": [round(av_lo, 4), round(av_hi, 4)],
        "nota": "Bootstrap não muda nada — só mede incerteza da janela atual.",
    })

    # 5) Leitura didática (sem tecninês)
    st.subheader("🧭 Interpretação (didática)")
    # heurística: se CI90 de rate_4 fica quase todo perto de zero, sinal é fraco; se desloca para cima, sinal é mais robusto
    if win["rate_4p"] == 0 and r4_hi <= 0.02:
        st.info("Na janela, **4+ ainda não é consistente**: mesmo no melhor cenário (CI90 alto), a taxa continua muito baixa. Isso indica que a melhora (se houver) ainda não 'firmou' na prática.")
    elif r4_lo >= 0.02:
        st.success("Há **sinal mais firme de 4+**: até o limite inferior do CI90 já não é tão baixo. Isso sugere que não é só 'sorte' — pode estar virando comportamento recorrente.")
    else:
        st.warning("Há **sinal**, mas ainda **instável**: a taxa observada pode subir/descer bastante só pelo filme curto. Aqui entra a fase de estabilização (acumular séries/avaliações).")

    st.markdown("""
**O que este painel responde (sem mexer no motor):**
- **Pacote está bom ou foi sorte?** → pelo CI90 do `rate_4p_w` e `avg_best_w`.
- **Está firmando ou oscilando?** → se o intervalo é largo, está oscilando (fase de estabilização).
- **Quebrar o 4 recorrente** → só acontece quando `rate_4p_w` sai do zero e o CI90 começa a 'descolar' de zero.
""")


if painel == "📁 Carregar Histórico (Arquivo)":

    st.markdown("## 📁 Carregar Histórico — Arquivo (V15.7 MAX)")
    st.caption(
        "Carregamento canônico via arquivo FLEX ULTRA (TXT/CSV).\n"
        "✔ Não muda motores\n"
        "✔ Não decide\n"
        "✔ Alimenta a sessão atual"
    )

    up = st.file_uploader(
        "Envie um histórico FLEX ULTRA",
        type=["txt", "csv"],
        accept_multiple_files=False,
    )

    if up is None:
        st.info("Envie um arquivo para iniciar o processamento do PredictCars.")
        st.stop()

    try:
        raw = up.getvalue()
        try:
            txt = raw.decode("utf-8")
        except Exception:
            txt = raw.decode("latin-1", errors="ignore")

        linhas = [l.strip() for l in txt.splitlines() if l.strip()]
        if not linhas:
            st.error("Histórico vazio")
            st.stop()

        df = carregar_historico_universal(linhas)
        st.session_state["historico_df"] = df
        # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF
        v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

        # Sincroniza chaves canônicas (evita N/D indevido no RF)
        try:
            v16_sync_aliases_canonicos()
        except Exception:
            pass

        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        if umin is not None and umax is not None:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries | Universo detectado: {umin}–{umax}")
        else:
            umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        if umin is not None and umax is not None:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries | Universo detectado: {umin}–{umax}")
        else:
            st.success(f"Histórico carregado com sucesso: {len(df)} séries")

    except Exception as e:
        st.error(f"Erro ao processar histórico: {e}")

    st.stop()

# Painel 1B — 📄 Carregar Histórico (Colar)
# ============================================================
if "Carregar Histórico (Colar)" in str(painel):

    st.markdown("## 📄 Carregar Histórico — Copiar e Colar (V15.7 MAX)")

    texto = st.text_area(
        "Cole aqui o histórico completo",
        height=320,
        key="pc_colar_texto_simples",
    )

    clicked = st.button(
        "📥 Processar Histórico (Copiar e Colar)",
        key="pc_colar_btn_simples",
    )

    if clicked:

        st.write("PROCESSANDO HISTÓRICO...")

        if not texto.strip():
            st.error("Histórico vazio")
            st.stop()

        linhas = texto.strip().split("\n")

        df = carregar_historico_universal(linhas)

        st.session_state["historico_df"] = df

        # 🔎 Universo (1–50 / 1–60) — registro canônico para snapshot/RF

        v16_registrar_universo_session_state(st.session_state["historico_df"], n_alvo=st.session_state.get("n_alvo", 6))

        st.success(f"Histórico carregado com sucesso: {len(df)} séries")





# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# FASE 1 — OBSERVAÇÃO PURA | SEM IMPACTO OPERACIONAL
# ============================================================






# ============================================================
# PAINEL — 📊 V16 PREMIUM — ERRO POR REGIME (RETROSPECTIVO)
# (INSTRUMENTAÇÃO: mede continuidade do erro por janelas)
# ============================================================

# ============================================================
# PAINEL — 🧠 Diagnóstico ECO & Estado (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧠 Diagnóstico ECO & Estado (V16)":

    st.markdown("## 🧠 Diagnóstico ECO & Estado — V16")
    st.caption("Leitura mastigada do ambiente e do alvo. Observacional.")

    # Sincroniza chaves canônicas (evita N/D indevido no RF)
    v16_sync_aliases_canonicos()


    diag = st.session_state.get("diagnostico_eco_estado_v16")

    if not diag:
        st.info("Diagnóstico ainda não disponível. Carregue um histórico.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌊 ECO")
        eco_forca = diag.get("eco_forca") or "fraco"
        st.write(f"Força: **{eco_forca}**")
        st.write(f"Persistência: **{diag.get('eco_persistencia')}**")
        st.write(f"Acionabilidade: **{diag.get('eco_acionabilidade')}**")

        motivos = diag.get("motivos_eco", [])
        if motivos:
            st.caption("Motivos: " + ", ".join(motivos))

        contradicoes = diag.get("contradicoes", [])
        if contradicoes:
            st.warning("⚠️ Contradições: " + "; ".join(contradicoes))

    with col2:
        st.markdown("### 🐟 Estado do Alvo")
        st.write(f"Estado: **{diag.get('estado')}**")
        st.write(
            "Confiabilidade: "
            f"**{'alta' if diag.get('estado_confiavel') else 'baixa / transição'}**"
        )

    st.markdown("### 🧠 Leitura Geral")
    
    st.success(diag.get("leitura_geral", "—"))


elif painel == "🧾 APS — Auditoria de Postura (V16)":

    st.markdown("## 🧾 APS — Auditoria de Postura (V16)")
    st.caption("Auditoria observacional do risco/postura do sistema. NÃO muda listas. NÃO decide volume. Serve para proteger contra postura errada (ex.: ancoragem excessiva em E0 + ruído alto).")

    # Coleta segura
    nr = st.session_state.get("nr_percent_v16") or st.session_state.get("nr_percent") or st.session_state.get("NR_PERCENT")
    orbita = st.session_state.get("orbita_selo_v16") or st.session_state.get("orbita_selo") or st.session_state.get("ORBITA_SELO") or "E0"
    diag = st.session_state.get("diagnostico_eco_estado_v16") or {}
    eco_acion = diag.get("eco_acionabilidade") or "N/D"

    anti_exato = st.session_state.get("anti_exato_level_v16") or st.session_state.get("anti_exato_level")  # opcional

    selo, titulo, msg = v16_calcular_aps_postura(nr_percent=nr, orbita_selo=orbita, eco_acionabilidade=eco_acion, anti_exato_level=anti_exato)

    # Registro canônico (observacional)

    try:

        st.session_state["aps_postura_selo"] = selo

        st.session_state["aps_postura_titulo"] = titulo

        st.session_state["aps_postura_msg"] = msg

    except Exception:

        pass

    st.markdown(f"### {selo} {titulo}")
    st.info(msg)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Entradas observadas**")
        st.write(f"- NR%: **{nr if nr is not None else 'N/D'}**")
        st.write(f"- Órbita (selo): **{orbita}**")
        st.write(f"- ECO (acionabilidade): **{eco_acion}**")
        if anti_exato is not None:
            st.write(f"- Anti-EXATO (nível): **{anti_exato}**")
    with col2:
        st.markdown("**Compatibilidades sugeridas (não obrigatórias)**")
        if selo == "🟡":
            st.write("- ✔ Duplo pacote: **base + anti-âncora**")
            st.write("- ✔ Envelope estreito / microvariações")
            st.write("- ⚠ Evitar ancoragem forte única")
        elif selo == "🔴":
            st.write("- ✔ Pacote mais espalhado e volume baixo")
            st.write("- ⚠ Evitar densidade e insistência")
        elif selo == "🟢":
            st.write("- ✔ Densidade moderada pode ser compatível")
            st.write("- ✔ Observar persistência por 1–3 séries")
        else:
            st.write("- ✔ Manter pacote base e acompanhar série a série")

    st.markdown("### 📌 Nota de governança")
    st.caption("Se a APS 'apontar o dedo', o sistema NÃO muda nada automaticamente nesta fase. A função aqui é blindar leitura e evitar postura errada; a execução segue com os pacotes já gerados.")


elif painel == "🧭 RMO/DMO — Retrato do Momento (V16)":
    st.markdown("## 🧭 RMO/DMO — Retrato do Momento (V16)")
    st.caption("Síntese integrada (RMO) + governança temporal (DMO) + voz operacional (VOS). Observacional. Não decide ação.")

    # Sincroniza chaves canônicas (ECO/Estado/k*/Divergência) antes do retrato
    v16_sync_aliases_canonicos()


    # -----------------------------
    # Coleta segura de sinais
    # -----------------------------
    risco_pack = st.session_state.get("diagnostico_risco") or {}
    diag = st.session_state.get("diagnostico_eco_estado_v16") or {}

    nr_ruido = st.session_state.get("nr_percent")  # Painel de Ruído Condicional (normalizado)
    nr_risco = risco_pack.get("nr_percent")        # Monitor de risco (NR% usado no índice)
    div = risco_pack.get("divergencia")
    classe_risco = risco_pack.get("classe_risco")
    indice_risco = risco_pack.get("indice_risco")

    orb = st.session_state.get("orbita_info") or {}
    orb_estado = orb.get("estado", "N/D")
    orb_selo = orb.get("selo", "N/D")
    grad = st.session_state.get("orbita_gradiente", "N/D")
    orb_score = st.session_state.get("orbita_score")

    eco_forca = diag.get("eco_forca", diag.get("forca", "N/D"))
    eco_persist = diag.get("eco_persistencia", diag.get("persistencia", "N/D"))
    eco_acion = diag.get("eco_acionabilidade", diag.get("acionabilidade", "N/D"))

    estado_alvo = diag.get("estado", "N/D")
    estado_conf = "alta" if diag.get("estado_confiavel") else "baixa / transição"

    b3_pronto = bool(st.session_state.get("b3_pronto_refinar", False))
    pipeline_ok = bool(st.session_state.get("pipeline_flex_ultra_concluido", False))
    turbo_ultra_rodou = bool(st.session_state.get("turbo_ultra_rodou", False))
    modo6_total = st.session_state.get("modo6_n_total")

    # -----------------------------
    # RMO — Retrato do Momento Operacional
    # -----------------------------
    st.markdown("### 🖼️ RMO — Retrato do Momento Operacional")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("NR% (Ruído)", f"{nr_ruido:.2f}%" if isinstance(nr_ruido, (int, float)) else "N/D")
        st.caption("Painel 📡 Ruído Condicional")
    with c2:
        st.metric("NR% (Risco)", f"{nr_risco:.2f}%" if isinstance(nr_risco, (int, float)) else "N/D")
        st.caption("Monitor k & k*")
    with c3:
        st.metric("Divergência S6×MC", f"{div:.4f}" if isinstance(div, (int, float)) else "N/D")
        st.caption("do Monitor (quando disponível)")
    with c4:
        st.metric("Índice de Risco", f"{indice_risco:.4f}" if isinstance(indice_risco, (int, float)) else "N/D")
        st.caption(classe_risco or "Classe N/D")

    st.markdown("#### 🌊 ECO & 🐟 Estado (leitura mastigada)")
    c5, c6, c7 = st.columns(3)
    with c5:
        st.write(f"**ECO**: {eco_forca} · {eco_persist} · {eco_acion}")
    with c6:
        st.write(f"**Estado do alvo**: {estado_alvo} (conf.: {estado_conf})")
    with c7:
        st.write(f"**Órbita**: {orb_estado} · {orb_selo} · grad {grad}")

    st.markdown("#### 🧱 Integridade operacional (sem julgamento)")
    c8, c9, c10 = st.columns(3)
    with c8:
        st.write(f"Pipeline FLEX ULTRA: **{'✅' if pipeline_ok else '—'}**")
    with c9:
        st.write(f"TURBO++ ULTRA rodou: **{'✅' if turbo_ultra_rodou else '—'}**")
    with c10:
        st.write(f"Modo 6 (N_total): **{modo6_total if modo6_total is not None else 'N/D'}**")

    st.markdown("#### 🧼 Perna B (prontidão)")
    st.write(f"B3 — Pronto para refinamento: **{'🟢 SIM' if b3_pronto else '🟡 AINDA NÃO'}**")

    # -----------------------------
    # DMO — Detector de Momento Operável (governança temporal)
    # -----------------------------
    st.markdown("### ⏳ DMO — Detector de Momento Operável (governança)")

    # histórico curto (memória leve, apenas dentro da sessão)
    if "dmo_hist_sinais" not in st.session_state:
        st.session_state["dmo_hist_sinais"] = []
    if "dmo_estado" not in st.session_state:
        st.session_state["dmo_estado"] = "🟥 SOBREVIVÊNCIA"

    sinais = []

    # Sinal A: Órbita sugere estrutura (E2 ou E1 forte via gradiente)
    if str(orb_estado).upper() in ["E2"]:
        sinais.append("Órbita E2 (interceptação plausível)")
    elif str(orb_estado).upper() in ["E1"] and str(grad).upper() in ["G2", "G3", "FORTE"]:
        sinais.append("Órbita E1 forte (gradiente alto)")

    # Sinal B: ECO persistente e (mesmo que fraco) não recuando
    if str(eco_persist).lower() in ["persistente", "sim", "alta", "ok"]:
        sinais.append("ECO com persistência")

    # Sinal C: Ruído não está piorando (tendência curta)
    # (usa NR do Painel de Ruído, quando disponível)
    hist = st.session_state["dmo_hist_sinais"]
    nr_ok = None
    try:
        if isinstance(nr_ruido, (int, float)):
            prev_nr = st.session_state.get("dmo_prev_nr_ruido")
            if isinstance(prev_nr, (int, float)):
                nr_ok = (nr_ruido <= prev_nr + 1e-9)
                if nr_ok:
                    sinais.append("NR não crescente (curto prazo)")
            st.session_state["dmo_prev_nr_ruido"] = float(nr_ruido)
    except Exception:
        pass

    # Sinal D: B3 pronto (refinamento viável)
    if b3_pronto:
        sinais.append("B3 pronto (refinamento viável)")

    # pontuação simples
    score = int(len(sinais))
    hist.append(score)
    hist[:] = hist[-5:]  # memória curta

    # regra de estados (consistente com o canônico)
    estado_atual = st.session_state.get("dmo_estado", "🟥 SOBREVIVÊNCIA")
    media2 = sum(hist[-2:]) / max(1, len(hist[-2:]))
    media3 = sum(hist[-3:]) / max(1, len(hist[-3:]))

    if len(hist) >= 3 and media3 >= 2.0:
        novo_estado = "🟩 OPERÁVEL"
    elif len(hist) >= 2 and media2 >= 1.0:
        novo_estado = "🟨 ATENÇÃO"
    else:
        novo_estado = "🟥 SOBREVIVÊNCIA"

    st.session_state["dmo_estado"] = novo_estado

    # exibição
    st.write(f"**Estado DMO:** {novo_estado}")
    st.caption("O DMO não decide ação. Ele governa o tempo (evita sair cedo demais).")

    if sinais:
        st.markdown("**Sinais ativos agora:**")
        for s in sinais:
            st.write(f"- {s}")
    else:
        st.markdown("**Sinais ativos agora:** nenhum (isso é um estado válido)")

    st.caption(f"Memória curta (scores últimas rodadas na sessão): {hist}")

    # -----------------------------
    # VOS — Voz Operacional do Sistema (1 frase, sem decisão)
    # -----------------------------
    st.markdown("### 🔊 VOS — Voz Operacional do Sistema (curta)")

    if novo_estado.startswith("🟥"):
        frase = "Ambiente não sustenta precisão. Permanecer ou trocar não altera o risco."
        st.warning(frase)
    elif novo_estado.startswith("🟨"):
        frase = "Estrutura começa a se repetir. Evite desmontar o que ainda está coerente."
        st.info(frase)
    else:
        frase = "Persistência custa menos que mudança. Reduza variação."
        st.success(frase)

    st.stop()

elif painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":

    st.subheader("📊 V16 Premium — Erro por Regime (Retrospectivo)")
    st.caption(
        "Instrumentação retrospectiva: janelas móveis → regime (ECO/PRE/RUIM) "
        "por dispersão da janela e erro da PRÓXIMA série como proxy de 'erro contido'. "
        "Não altera motor. Não escolhe passageiros."
    )

    # ============================================================
    # Localização ROBUSTA do histórico (padrão oficial V16)
    # ============================================================
    _, historico_df = v16_identificar_df_base()

    if historico_df is None or historico_df.empty:
        st.warning(
            "Histórico não encontrado no estado atual do app.\n\n"
            "👉 Recarregue o histórico e volte diretamente a este painel."
        )
        st.stop()

    if len(historico_df) < 100:
        st.warning(
            f"Histórico muito curto para análise retrospectiva.\n\n"
            f"Séries detectadas: {len(historico_df)}"
        )
        st.stop()

    # 🔒 Anti-zumbi automático (painel leve, invisível)
    janela = 60
    step = 1

    with st.spinner("Calculando análise retrospectiva por janelas (V16 Premium)..."):
        out = pc16_calcular_continuidade_por_janelas(
            historico_df=historico_df,
            janela=janela,
            step=step,
            usar_quantis=True
        )

    if not out.get("ok", False):
        st.error(f"Falha na análise: {out.get('motivo','Erro desconhecido')}")
        st.stop()

    resumo_geral = out.get("resumo_geral", {})
    resumo = out.get("resumo", {})
    df = out.get("df", pd.DataFrame())

    # ============================================================
    # RESULTADO OBJETIVO
    # ============================================================
    st.markdown("### ✅ Resultado objetivo — Continuidade do erro")

    diff = resumo_geral.get("diff_ruim_menos_eco_no_erro", None)
    if diff is None:
        st.info(
            "Ainda não há base suficiente para comparar ECO vs RUIM.\n\n"
            "Isso ocorre quando algum regime tem poucas janelas."
        )
    else:
        st.write(
            f"**Diferença RUIM − ECO no erro médio (erro_prox):** "
            f"`{diff:.6f}`\n\n"
            "➡️ Valores positivos indicam erro menor em ECO."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de janelas", str(resumo_geral.get("n_total_janelas", "—")))
    col2.metric("Janela (W)", str(resumo_geral.get("janela", "—")))
    col3.metric("q1 dx (ECO ≤)", f"{resumo_geral.get('q1_dx', 0):.6f}")
    col4.metric("q2 dx (PRE ≤)", f"{resumo_geral.get('q2_dx', 0):.6f}")

    # ============================================================
    # TABELA POR REGIME
    # ============================================================
    st.markdown("### 🧭 Tabela por Regime (ECO / PRE / RUIM)")

    linhas = []
    for reg in ["ECO", "PRE", "RUIM"]:
        r = resumo.get(reg, {"n": 0})
        linhas.append({
            "Regime": reg,
            "n_janelas": r.get("n", 0),
            "dx_janela_medio": r.get("dx_janela_medio"),
            "erro_prox_medio": r.get("erro_prox_medio"),
            "erro_prox_mediana": r.get("erro_prox_mediana"),
        })

    df_reg = pd.DataFrame(linhas)
    st.dataframe(df_reg, use_container_width=True)

    # ============================================================
    # AUDITORIA LEVE
    # ============================================================
    st.markdown("### 🔎 Amostra das janelas (auditoria leve)")
    st.caption(
        "Exibe as primeiras linhas apenas para validação conceitual. "
        "`t` é um índice interno (0-based)."
    )
    st.dataframe(df.head(50), use_container_width=True)

    # ============================================================
    # LEITURA OPERACIONAL
    # ============================================================
    st.markdown("### 🧠 Leitura operacional (objetiva)")
    st.write(
        "- Se **ECO** apresentar **erro_prox_medio** consistentemente menor que **RUIM**, "
        "isso sustenta matematicamente que, em estados ECO, **o erro tende a permanecer contido**.\n"
        "- Este painel **não escolhe passageiros**.\n"
        "- Ele **autoriza** (ou não) a fase seguinte: **concentração para buscar 6**, "
        "sem alterar motor ou fluxo."
    )




# ============================================================
# PAINEL V16 — 🎯 Compressão do Alvo (OBSERVACIONAL)
# Leitura pura | NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

if painel == "🎯 Compressão do Alvo (Observacional)":

    st.markdown("## 🎯 Compressão do Alvo — Leitura Observacional (V16)")
    st.caption(
        "Este painel mede **se o erro provável está comprimindo**.\n\n"
        "⚠️ Não prevê números, não sugere volume, não altera o fluxo."
    )

    # -----------------------------
    # Coleta de sinais já existentes
    # -----------------------------
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    k_star = st.session_state.get("sentinela_kstar")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    df = st.session_state.get("historico_df")

    # ------------------------------------------------------------
    # V16h57CN — SESSION STATE CONTROL (FORCE FRESH PACKET)
    # ------------------------------------------------------------
    _removed_fresh_keys = v16h57FL_clear_mode6_packet_state()
    if _removed_fresh_keys:
        st.caption("🧹 Session State limpo para pacote fresco do Modo 6: " + ", ".join(_removed_fresh_keys))
    else:
        st.caption("🧹 Session State do pacote já estava limpo para execução fresca do Modo 6.")

    if df is None or nr is None or div is None or k_star is None or risco is None:
        exibir_bloco_mensagem(
            "Pré-requisitos ausentes",
            "Execute os painéis de Sentinela, Ruído, Divergência e Monitor de Risco.",
            tipo="warning",
        )
        st.stop()

    # -----------------------------
    # 1) Estabilidade do ruído
    # -----------------------------
    nr_ok = nr < 45.0

    # -----------------------------
    # 2) Convergência dos motores
    # -----------------------------
    div_ok = div < 5.0

    # -----------------------------
    # 3) Regime não-hostil
    # -----------------------------
    risco_ok = risco < 0.55

    # -----------------------------
    # 4) k como marcador NORMAL (não extremo)
    # -----------------------------
    k_ok = 0.10 <= k_star <= 0.35

    # -----------------------------
    # 5) Repetição estrutural (passageiros)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    ultimos = df[col_pass].iloc[-10:].values

    repeticoes = []
    for i in range(len(ultimos) - 1):
        repeticoes.append(len(set(ultimos[i]) & set(ultimos[i + 1])))

    repeticao_media = float(np.mean(repeticoes)) if repeticoes else 0.0
    repeticao_ok = repeticao_media >= 2.5

    # -----------------------------
    # Consolidação OBSERVACIONAL
    # -----------------------------
    sinais = {
        "NR% estável": nr_ok,
        "Convergência S6 × MC": div_ok,
        "Risco controlado": risco_ok,
        "k em faixa normal": k_ok,
        "Repetição estrutural": repeticao_ok,
    }

    positivos = sum(1 for v in sinais.values() if v)

    # -----------------------------
    # Exibição
    # -----------------------------
    st.markdown("### 📊 Sinais de Compressão do Erro")

    for nome, ok in sinais.items():
        st.markdown(
            f"- {'🟢' if ok else '🔴'} **{nome}**"
        )

    st.markdown("### 🧠 Leitura Consolidada")

    if positivos >= 4:
        leitura = (
            "🟢 **Alta compressão do erro provável**.\n\n"
            "O alvo está mais bem definido do que o normal.\n"
            "Se houver PRÉ-ECO / ECO, a convicção operacional aumenta."
        )
    elif positivos == 3:
        leitura = (
            "🟡 **Compressão parcial**.\n\n"
            "Há foco emergente, mas ainda com dispersão residual."
        )
    else:
        leitura = (
            "🔴 **Sem compressão clara**.\n\n"
            "Erro ainda espalhado. Operar com cautela."
        )

    exibir_bloco_mensagem(
        "Compressão do Alvo — Diagnóstico",
        leitura,
        tipo="info",
    )

    st.caption(
        f"Sinais positivos: {positivos}/5 | "
        "Este painel **não autoriza nem bloqueia** nenhuma ação."
    )

# ============================================================
# FIM — PAINEL V16 — COMPRESSÃO DO ALVO (OBSERVACIONAL)
# ============================================================


# ============================================================
# BLOCO — OBSERVADOR HISTÓRICO DE EVENTOS k (V16)
# REPLAY HISTÓRICO OBSERVACIONAL (MEMÓRIA REAL)
# NÃO decide | NÃO prevê | NÃO altera motores | NÃO altera volumes
# ============================================================


# ============================================================
# ALIAS CANÔNICO (V16) — M3
# Expectativa Histórica — Contexto do Momento
# ============================================================
def v16_painel_expectativa_historica_contexto():
    """Alias canônico para preservar âncoras/painéis que chamam a função V16.

    Regra: NÃO calcula listas, NÃO decide, NÃO altera fluxo.
    Encaminha para o painel observacional M3.
    """
    return m3_painel_expectativa_historica_contexto()


def v16_painel_parabolica_curvatura_erro_pre_c4():
    """
    📐 Parabólica — Curvatura do Erro (Governança Pré-C4)
    Observacional • auditável • não altera listas • não altera Camada 4.
    Objetivo: mostrar a curvatura multi-escala do erro (fora_longe etc.) usando snapshots P0 do Replay Progressivo.
    """
    st.title("📐 Parabólica — Curvatura do Erro (Governança Pré-C4)")
    st.caption("Governança pré‑C4 • Observacional • Auditável. Não gera listas. Não decide. Não altera Camada 4.")

    df_full = st.session_state.get("historico_df")
    snaps_map = st.session_state.get("snapshot_p0_canonic") or {}
    if df_full is None or (hasattr(df_full, "empty") and df_full.empty):
        st.warning("Não encontrei histórico na sessão. Rode primeiro **📁 Carregar Histórico**.")
        return
    if not isinstance(snaps_map, dict) or len(snaps_map) == 0:
        st.warning("""Ainda não há snapshots P0 registrados.

Roteiro canônico:
1) Rode **🎯 Modo 6** (gera pacote)
2) Vá em **🧭 Replay Progressivo — Janela Móvel (Assistido)**
3) Clique em **Registrar pacote gerado para esta janela**
4) Rode 1–N lotes SAFE para acumular snapshots (ideal: 60)
""")
        return

    try:
        n = int(st.session_state.get("n_alvo") or 6)
    except Exception:
        n = 6

    try:
        gov = parabola_multiescala_vetorial(df_full, snaps_map, n=n) or {}
        st.session_state["parabola_gov"] = gov
        st.session_state["parabola_estado_global"] = gov.get("estado_global")
    except Exception as e:
        st.error(f"Falha ao calcular Parabólica: {e}")
        return

    estado_global = gov.get("estado_global")
    Ws = (gov.get("Ws") or {}) if isinstance(gov, dict) else {}
    pers = (gov.get("persistencia") or {}) if isinstance(gov, dict) else {}

    st.success(f"Parabólica calculada. Estado global: **{estado_global}**")
    try:
        st.caption(f"Escalas (Ws): short={Ws.get('short')} · mid={Ws.get('mid')} · long={Ws.get('long')}")
    except Exception:
        pass

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Estados (por escala · métrica)")
        try:
            st.json(gov.get("estados") or {})
        except Exception:
            st.json({})
    with colB:
        st.subheader("Persistência (objetiva)")
        try:
            st.json(pers)
        except Exception:
            st.json({})

    with st.expander("Debug (curvaturas e séries) — auditável", expanded=False):
        st.json({
            "Ws": Ws,
            "estado_global": estado_global,
            "p2_permitido": gov.get("p2_permitido"),
            "confirmacao": gov.get("confirmacao"),
            "series": gov.get("series"),
            "debug": gov.get("debug"),
        })


def v16_painel_cap_calibracao_assistida_parabola_pre_c4():
    """
    📡 CAP — Calibração Assistida da Parabólica (pré‑C4)
    Objetivo: gravar uma calibração simples e auditável (cap_status + cap_pct) baseada no estado da Parabólica.
    Observacional: só escreve sessão quando o operador clicar em calibrar.
    """
    st.title("📡 CAP — Calibração Assistida da Parabólica (pré-C4)")
    st.caption("Governança pré‑C4 • Assistida • Auditável. Não gera listas automaticamente. Não altera Camada 4.")

    gov = st.session_state.get("parabola_gov")
    estado_global = st.session_state.get("parabola_estado_global")

    if not isinstance(gov, dict) or estado_global is None:
        st.warning("""CAP depende da Parabólica.

Rode primeiro **📐 Parabólica — Curvatura do Erro** (e tenha snapshots P0 suficientes).
""")
        return

    cap_status = str(st.session_state.get("cap_status") or "NÃO CALIBRADA")
    cap_pct_atual = float(st.session_state.get("cap_pct") or 0.65)

    st.markdown("### Estado atual")
    st.write(f"- **Parabólica (estado_global):** {estado_global}")
    st.write(f"- **CAP status:** {cap_status}")
    st.write(f"- **cap_pct (elasticidade mínima em RESPIRÁVEL):** {cap_pct_atual:.2f}")

    st.markdown("---")
    st.markdown("### Calibração assistida (1 clique)")

    # Heurística simples (governança):
    # - SUBINDO (erro abrindo / piorando) -> mais elasticidade (cap_pct maior)
    # - PLANA (ambíguo) -> neutro
    # - DESCENDO (erro fechando / melhorando) -> menos elasticidade (cap_pct menor)
    if estado_global == "SUBINDO":
        cap_pct_sug = 0.70
    elif estado_global == "DESCENDO":
        cap_pct_sug = 0.55
    else:
        cap_pct_sug = 0.65

    st.info(f"""Sugestão CAP (auditável):
- estado_global = **{estado_global}** → cap_pct sugerido = **{cap_pct_sug:.2f}**

Isso só afeta a *diversificação mínima* (pré‑C4) quando você **rodar novamente** o Modo 6.
Não muda Camada 4.
""")
    c1, c2 = st.columns([1, 1])
    with c1:
        cap_pct_manual = st.slider("cap_pct manual (se quiser)", min_value=0.40, max_value=0.85, value=float(cap_pct_sug), step=0.01)
    with c2:
        st.caption("Recomendação: mantenha o default sugerido a menos que esteja testando hipóteses.")

    if st.button("✅ Calibrar CAP agora (gravar na sessão)", use_container_width=True):
        st.session_state["cap_pct"] = float(cap_pct_manual)
        st.session_state["cap_status"] = f"CALIBRADA({estado_global}) cap_pct={float(cap_pct_manual):.2f}"
        st.success(f"CAP calibrada: {st.session_state['cap_status']}")

    with st.expander("O que isso altera (auditável)", expanded=False):
        st.write(
            "- **cap_status**: habilita governança P1 onde o código exige CAP 'CALIBRADA'.\n"
            "- **cap_pct**: parâmetro usado na diversificação mínima em postura RESPIRÁVEL.\n"
            "- Não cria motor novo e não altera Camada 4."
        )


def v16_replay_historico_observacional(
    *,
    df,
    matriz_norm,
    janela_max=800,
):
    """
    Replay histórico OBSERVACIONAL.
    Executa leitura silenciosa série-a-série para preencher memória
    e eliminar campos None no Observador Histórico.

    - Usa somente dados já calculados
    - NÃO reexecuta motores pesados
    - NÃO interfere no fluxo operacional
    """

    if df is None or matriz_norm is None:
        return []

    n_total = len(df)
    inicio = max(0, n_total - int(janela_max))

    registros = []

    col_pass = [c for c in df.columns if c.startswith("p")]

    for idx in range(inicio, n_total):

        # --- NR% local (réplica leve) ---
        try:
            m = matriz_norm[: idx + 1]
            variancias = np.var(m, axis=1)
            ruido_A = float(np.mean(variancias))
            saltos = [
                np.linalg.norm(m[i] - m[i - 1])
                for i in range(1, len(m))
            ]
            ruido_B = float(np.mean(saltos)) if saltos else 0.0
            nr_pct = float(
                (0.55 * min(1.0, ruido_A / 0.08) +
                 0.45 * min(1.0, ruido_B / 1.20)) * 100.0
            )
        except Exception:
            nr_pct = None

        # --- Divergência local S6 vs MC (proxy leve) ---
        try:
            base = m[-1]
            candidatos = m[-10:] if len(m) >= 10 else m
            divergencia = float(
                np.linalg.norm(np.mean(candidatos, axis=0) - base)
            )
        except Exception:
            divergencia = None

        # --- Velocidade / estado do alvo (heurística coerente) ---
        try:
            vel = float(
                (nr_pct / 100.0 if nr_pct is not None else 0.5) +
                (divergencia / 15.0 if divergencia is not None else 0.5)
            ) / 2.0
        except Exception:
            vel = None

        if vel is None:
            estado = None
        elif vel < 0.30:
            estado = "parado"
        elif vel < 0.55:
            estado = "movimento_lento"
        elif vel < 0.80:
            estado = "movimento_rapido"
        else:
            estado = "disparado"

        # --- k histórico ---
        try:
            k_val = int(df.iloc[idx].get("k", 0))
        except Exception:
            k_val = 0

        registros.append({
            "serie_id": idx,
            "k_valor": k_val,
            "estado_alvo": estado,
            "nr_percent": nr_pct,
            "div_s6_mc": divergencia,
        })

    return registros


# ============================================================
# EXECUÇÃO AUTOMÁTICA — REPLAY OBSERVACIONAL (SE HISTÓRICO EXISTIR)
# ============================================================

if (
    "historico_df" in st.session_state
    and "pipeline_matriz_norm" in st.session_state
):
    registros_obs = v16_replay_historico_observacional(
        df=st.session_state.get("historico_df"),
        matriz_norm=st.session_state.get("pipeline_matriz_norm"),
        janela_max=800,  # DECISÃO DO COMANDO
    )

    st.session_state["observador_historico_v16"] = registros_obs

# ============================================================
# FIM — BLOCO OBSERVADOR HISTÓRICO (V16)
# ============================================================



# ============================================================
# BLOCO — OBSERVAÇÃO HISTÓRICA OFFLINE (V16)
# OPÇÃO B MÍNIMA | LEITURA PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

def _pc_distancia_carros_offline(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    Observacional, robusto e defensivo.
    """
    try:
        sa = set(int(x) for x in a)
        sb = set(int(x) for x in b)
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy_offline(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    NÃO é o estado V16 online. Uso EXCLUSIVO histórico.
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_offline(row):
    """
    Extrai os 6 passageiros de uma linha do histórico.
    Compatível com p1..p6 ou colunas numéricas genéricas.
    """
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            candidatos.append(int(row[c]))
        except Exception:
            continue

    return candidatos[:6] if len(candidatos) >= 6 else None


def construir_contexto_historico_offline_v16(df):
    """
    Constrói CONTEXTO HISTÓRICO OFFLINE mínimo:
    - estado_alvo_proxy_historico
    - delta_k_historico
    - eventos_k_historico (enriquecido)
    NÃO interfere em motores, painéis ou decisões.
    """

    if df is None or df.empty:
        return

    estado_proxy_hist = {}
    delta_k_hist = {}
    eventos_k = []

    carro_prev = None
    ultima_pos_k = None

    for pos, (idx, row) in enumerate(df.iterrows()):
        carro_atual = _pc_extrair_carro_offline(row)

        dist = (
            _pc_distancia_carros_offline(carro_prev, carro_atual)
            if carro_prev is not None and carro_atual is not None
            else None
        )

        estado_proxy = _pc_estado_alvo_proxy_offline(dist)
        estado_proxy_hist[idx] = estado_proxy

        # Evento k (observacional)
        try:
            k_val = int(row.get("k", 0))
        except Exception:
            k_val = 0

        if k_val > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)
            delta_k_hist[idx] = delta

            eventos_k.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_val),
                "delta_series": delta,
                "estado_alvo_proxy": estado_proxy,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    # Persistência PASSIVA (session_state)
    st.session_state["estado_alvo_proxy_historico"] = estado_proxy_hist
    st.session_state["delta_k_historico"] = delta_k_hist
    st.session_state["eventos_k_historico"] = eventos_k


# ============================================================
# EXECUÇÃO AUTOMÁTICA OFFLINE (SE HISTÓRICO EXISTIR)
# NÃO BLOQUEIA | NÃO DECIDE | NÃO OPERA
# ============================================================

if "historico_df" in st.session_state:
    try:
        construir_contexto_historico_offline_v16(
            st.session_state.get("historico_df")
        )
    except Exception:
        pass

# ============================================================
# FIM — OBSERVAÇÃO HISTÓRICA OFFLINE (V16) — OPÇÃO B MÍNIMA
# ============================================================

def extrair_eventos_k_historico(
    df,
    estados_alvo=None,
    k_star_series=None,
    nr_percent_series=None,
    divergencia_series=None,
    pre_eco_series=None,
    eco_series=None,
):
    """
    Extrai eventos k do histórico com contexto.
    NÃO decide, NÃO filtra operacionalmente, NÃO altera motores.
    Retorna lista de dicionários observacionais.
    """

    if df is None or df.empty:
        return []

    eventos = []
    ultima_serie_k = None

    for idx, row in df.iterrows():
        # Espera-se que o histórico tenha coluna 'k'
        k_valor = row.get("k", 0)

        if k_valor and k_valor > 0:
            # Delta desde último k
            if ultima_serie_k is None:
                delta = None
            else:
                delta = idx - ultima_serie_k

            evento = {
                "serie_id": idx,
                "k_valor": int(k_valor),
                "delta_series": delta,
                "estado_alvo": (
                    estados_alvo.get(idx)
                    if isinstance(estados_alvo, dict)
                    else None
                ),
                "k_star": (
                    k_star_series.get(idx)
                    if isinstance(k_star_series, dict)
                    else None
                ),
                "nr_percent": (
                    nr_percent_series.get(idx)
                    if isinstance(nr_percent_series, dict)
                    else None
                ),
                "div_s6_mc": (
                    divergencia_series.get(idx)
                    if isinstance(divergencia_series, dict)
                    else None
                ),
                "pre_eco": (
                    pre_eco_series.get(idx)
                    if isinstance(pre_eco_series, dict)
                    else False
                ),
                "eco": (
                    eco_series.get(idx)
                    if isinstance(eco_series, dict)
                    else False
                ),
            }

            eventos.append(evento)
            ultima_serie_k = idx

    return eventos


# ============================================================
# EXECUÇÃO AUTOMÁTICA (APENAS SE HISTÓRICO EXISTIR)
# ============================================================

if "historico_df" in st.session_state:
    df_hist = st.session_state.get("historico_df")

    eventos_k = extrair_eventos_k_historico(
        df=df_hist,
        estados_alvo=st.session_state.get("estado_alvo_historico"),
        k_star_series=st.session_state.get("kstar_historico"),
        nr_percent_series=st.session_state.get("nr_historico"),
        divergencia_series=st.session_state.get("div_s6_mc_historico"),
        pre_eco_series=st.session_state.get("pre_eco_historico"),
        eco_series=st.session_state.get("eco_historico"),
    )

    st.session_state["eventos_k_historico"] = eventos_k

# ============================================================
# BLOCO — FIM OBSERVADOR HISTÓRICO DE EVENTOS k
# ============================================================

# ============================================================
# Painel — 📊 Observador Histórico de Eventos k (V16)
# FASE 1 — OBSERVAÇÃO PURA | NÃO DECIDE | NÃO OPERA
# ============================================================

if painel == "📊 Observador k — Histórico":

    st.markdown("## 📊 Observador Histórico de Eventos k")
    st.caption(
        "Leitura puramente observacional. "
        "Este painel **não influencia** previsões, volumes ou decisões."
    )

    eventos = st.session_state.get("eventos_k_historico")

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k disponível",
            "Carregue um histórico válido para observar eventos k.",
            tipo="info",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    st.markdown("### 🔍 Tabela de Eventos k (Histórico)")
    st.dataframe(
        df_k,
        use_container_width=True,
        height=420,
    )

    # Métricas simples (somente leitura)
    st.markdown("### 📈 Métricas Observacionais Básicas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total de eventos k",
            len(df_k),
        )

    with col2:
        delta_vals = df_k["delta_series"].dropna()
        st.metric(
            "Δ médio entre ks",
            round(delta_vals.mean(), 2) if not delta_vals.empty else "—",
        )

    with col3:
        st.metric(
            "Δ mínimo observado",
            int(delta_vals.min()) if not delta_vals.empty else "—",
        )

    st.info(
        "Interpretação é humana. "
        "Nenhum uso operacional é feito a partir destes dados."
    )

# ============================================================
# FIM — Painel Observador Histórico de Eventos k
# ============================================================

# ============================================================
# Painel — 🎯 Compressão do Alvo — Observacional (V16)
# LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# Objetivo: medir se o alvo está REALMENTE "na mira"
# ============================================================

if painel == "🎯 Compressão do Alvo — Observacional (V16)":

    st.markdown("## 🎯 Compressão do Alvo — Observacional (V16)")
    st.caption(
        "Painel **observacional puro**.\n\n"
        "Ele NÃO gera previsões, NÃO altera volumes e NÃO interfere no fluxo.\n"
        "Serve para responder: **o alvo está realmente comprimido / na mira?**"
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros fixos (observacionais)
    # ------------------------------------------------------------
    JANELA_ANALISE = 120   # últimas séries
    JANELA_LOCAL = 8       # microjanela para dispersão
    LIMIAR_COMPRESSAO = 0.65  # heurístico (não decisório)

    n = len(matriz_norm)
    if n < JANELA_ANALISE + JANELA_LOCAL:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias mais séries para analisar compressão do alvo.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Cálculo da compressão
    # ------------------------------------------------------------
    dispersoes = []
    centroides = []

    for i in range(n - JANELA_ANALISE, n):
        janela = matriz_norm[max(0, i - JANELA_LOCAL): i + 1]
        centro = np.mean(janela, axis=0)
        centroides.append(centro)

        dist = np.mean(
            [np.linalg.norm(linha - centro) for linha in janela]
        )
        dispersoes.append(dist)

    dispersao_media = float(np.mean(dispersoes))
    dispersao_std = float(np.std(dispersoes))

    # Compressão relativa (quanto menor a dispersão, maior a compressão)
    compressao_score = 1.0 - min(1.0, dispersao_media / (dispersao_media + dispersao_std + 1e-6))
    compressao_score = float(round(compressao_score, 4))

    # ------------------------------------------------------------
    # Interpretação QUALITATIVA (não decisória)
    # ------------------------------------------------------------
    if compressao_score >= 0.75:
        leitura = "🟢 Alvo fortemente comprimido"
        comentario = (
            "O histórico recente mostra **alta repetição estrutural**.\n"
            "O sistema está operando em zona de foco.\n"
            "Quando combinado com PRÉ-ECO / ECO, **permite acelerar**."
        )
    elif compressao_score >= LIMIAR_COMPRESSAO:
        leitura = "🟡 Compressão moderada"
        comentario = (
            "Existe coerência estrutural, mas ainda com respiração.\n"
            "Bom para operação equilibrada."
        )
    else:
        leitura = "🔴 Alvo disperso"
        comentario = (
            "Alta variabilidade estrutural.\n"
            "Mesmo que k apareça, **não indica alvo na mira**."
        )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📐 Métrica de Compressão do Alvo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Score de Compressão", compressao_score)

    with col2:
        st.metric("Dispersão média", round(dispersao_media, 4))

    with col3:
        st.metric("Volatilidade da dispersão", round(dispersao_std, 4))

    exibir_bloco_mensagem(
        "Leitura Observacional",
        f"**{leitura}**\n\n{comentario}",
        tipo="info",
    )

    st.info("📌 Interpretação correta (sem viés):\n- Isso NÃO prevê o próximo alvo.\n- Isso mede *o que costuma acontecer* quando o ambiente cai no mesmo tipo de regime.\n- Serve para calibrar expectativa, postura e paciência — não para aumentar convicção por '3 acertos'.")

# ============================================================
# FIM — Painel 🎯 Compressão do Alvo — Observacional (V16)
# ============================================================



# ============================================================
# Observação Histórica — Eventos k (V16)
# Leitura passiva do histórico. Não interfere em decisões.
# + CRUZAMENTO k × ESTADO DO ALVO (PROXY)
# ============================================================

def _pc_distancia_carros(a, b):
    """
    Distância simples entre dois carros (listas de 6):
    quantos passageiros mudaram (0..6).
    """
    try:
        sa = set([int(x) for x in a])
        sb = set([int(x) for x in b])
        inter = len(sa & sb)
        return max(0, 6 - inter)
    except Exception:
        return None


def _pc_estado_alvo_proxy(dist):
    """
    Mapeia distância (0..6) em estado do alvo (proxy observacional).
    """
    if dist is None:
        return None
    if dist <= 1:
        return "parado"
    if dist <= 3:
        return "movimento_lento"
    if dist <= 5:
        return "movimento"
    return "movimento_brusco"


def _pc_extrair_carro_row(row):
    """
    Extrai os 6 passageiros da linha do df.
    Tentativa 1: colunas numéricas (6 colunas)
    Tentativa 2: colunas p1..p6 (se existir)
    """
    # Caso já tenha colunas p1..p6
    cols_p = ["p1", "p2", "p3", "p4", "p5", "p6"]
    if all(c in row.index for c in cols_p):
        return [row[c] for c in cols_p]

    # Caso seja DF com colunas misturadas: pega primeiros 6 inteiros que não sejam 'k'
    candidatos = []
    for c in row.index:
        if str(c).lower() == "k":
            continue
        try:
            v = int(row[c])
            candidatos.append(v)
        except Exception:
            continue

    if len(candidatos) >= 6:
        return candidatos[:6]

    return None


def extrair_eventos_k_historico_com_proxy(df):
    """
    Eventos k + delta + estado do alvo (proxy) calculado do próprio histórico.
    NÃO depende de estado_alvo_historico/kstar_historico/etc.
    """
    if df is None or df.empty:
        return [], {}

    eventos = []
    ultima_pos_k = None

    # Para estatística
    cont_estados = {"parado": 0, "movimento_lento": 0, "movimento": 0, "movimento_brusco": 0, "None": 0}

    # Vamos usar posição sequencial (0..n-1) para delta
    rows = list(df.iterrows())

    carro_prev = None

    for pos, (idx, row) in enumerate(rows):
        k_val = row.get("k", 0)
        carro_atual = _pc_extrair_carro_row(row)

        dist = _pc_distancia_carros(carro_prev, carro_atual) if (carro_prev is not None and carro_atual is not None) else None
        estado = _pc_estado_alvo_proxy(dist)

        # Contagem estados (para todas as séries, não só eventos k)
        if estado is None:
            cont_estados["None"] += 1
        else:
            cont_estados[estado] += 1

        # Evento k
        try:
            k_int = int(k_val) if k_val is not None else 0
        except Exception:
            k_int = 0

        if k_int > 0:
            delta = None if ultima_pos_k is None else int(pos - ultima_pos_k)

            eventos.append({
                "serie_id": idx,
                "pos": int(pos),
                "k_valor": int(k_int),
                "delta_series": delta,
                "distancia_prev": dist,
                "estado_alvo_proxy": estado,
            })

            ultima_pos_k = pos

        carro_prev = carro_atual

    return eventos, cont_estados


# ============================================================
# PAINEL (VISUALIZAÇÃO)
# ============================================================

if painel == "Observação Histórica — Eventos k":

    st.markdown("## Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    df_hist = st.session_state.get("historico_df")

    if df_hist is None or df_hist.empty:
        exibir_bloco_mensagem(
            "Histórico ausente",
            "Carregue o histórico primeiro (Painel 1 / 1B).",
            tipo="warning",
        )
        st.stop()

    eventos_k, cont_estados = extrair_eventos_k_historico_com_proxy(df_hist)
    st.session_state["eventos_k_historico"] = eventos_k

    # ===========================
    # Resumo estatístico
    # ===========================
    total_eventos = len(eventos_k)

    deltas = [e["delta_series"] for e in eventos_k if isinstance(e.get("delta_series"), int)]
    delta_medio = round(sum(deltas) / max(1, len(deltas)), 2) if deltas else None
    max_k = max([e.get("k_valor", 0) for e in eventos_k], default=0)

    st.markdown("### Resumo Estatístico Simples")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de eventos k", f"{total_eventos}")
    c2.metric("Δ médio entre ks", f"{delta_medio}" if delta_medio is not None else "—")
    c3.metric("Máx k observado", f"{max_k}")

    st.markdown("### Distribuição do Estado do Alvo (PROXY no histórico)")
    total_series = sum(cont_estados.values()) if isinstance(cont_estados, dict) else 0
    if total_series > 0:
        corpo = (
            f"- parado: **{cont_estados.get('parado', 0)}**\n"
            f"- movimento_lento: **{cont_estados.get('movimento_lento', 0)}**\n"
            f"- movimento: **{cont_estados.get('movimento', 0)}**\n"
            f"- movimento_brusco: **{cont_estados.get('movimento_brusco', 0)}**\n"
        )
        exibir_bloco_mensagem("Estado do alvo (proxy)", corpo, tipo="info")
    else:
        st.info("Não foi possível calcular distribuição de estado (proxy).")

    # ===========================
    # Tabela de eventos k
    # ===========================
    st.markdown("### 📋 Tabela de Eventos k (com estado proxy)")
    if total_eventos == 0:
        st.info("Nenhum evento k encontrado no histórico.")
        st.stop()

    mostrar = st.slider(
        "Quantos eventos k mostrar (mais recentes)?",
        min_value=20,
        max_value=min(300, total_eventos),
        value=min(80, total_eventos),
        step=10,
    )

    # Mostra os mais recentes
    df_evt = pd.DataFrame(eventos_k[-mostrar:])
    st.dataframe(df_evt, use_container_width=True)

    st.caption("Obs.: estado_alvo_proxy é calculado por mudança entre carros consecutivos (distância 0..6).")
    st.caption("k*/NR%/div/PRÉ-ECO/ECO ainda não estão historificados por série — isso é a próxima evolução (opcional).")

# ============================================================
# FIM — Observação Histórica — Eventos k (V16)
# ============================================================

        

# ============================================================
# Painel 2 — 🛰️ Sentinelas — k* (Ambiente de Risco)
# ============================================================

if painel == "🛰️ Sentinelas — k* (Ambiente de Risco)":

    st.markdown("## 🛰️ Sentinelas — k* (Ambiente de Risco) — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá primeiro ao painel **📁 Carregar Histórico**.",
            tipo="warning",
        )
    else:
        qtd_series = len(df)

        # Parâmetros do k*
        janela_curta = 12
        janela_media = 30
        janela_longa = 60

        # Anti-zumbi aplicado antes de cálculos longos
        if not limitar_operacao(
            qtd_series,
            limite_series=LIMITE_SERIES_REPLAY_ULTRA,
            contexto="Sentinela k*",
            painel="🛰️ Sentinelas — k*",
        ):
            st.stop()

        # -------------------------------------------
        # Cálculo do k* — versão V15.7 MAX / V16 Premium
        # -------------------------------------------
        try:
            k_vals = df["k"].astype(int).values

            def media_movel(vetor, janela):
                if len(vetor) < janela:
                    return np.mean(vetor)
                return np.mean(vetor[-janela:])

            k_curto = media_movel(k_vals, janela_curta)
            k_medio = media_movel(k_vals, janela_media)
            k_longo = media_movel(k_vals, janela_longa)

            # Fórmula nova do k* — ponderada
            k_star = (
                0.50 * k_curto
                + 0.35 * k_medio
                + 0.22 * k_longo
            )

        except Exception as erro:
            exibir_bloco_mensagem(
                "Erro no cálculo do k*",
                f"Ocorreu um erro interno: {erro}",
                tipo="error",
            )
            st.stop()

        # Guarda na sessão
        st.session_state["sentinela_kstar"] = k_star

        # Exibição amigável
        st.markdown(f"### 🌡️ k* calculado: **{k_star:.4f}**")

        # Diagnóstico de regime
        if k_star < 0.22:
            regime = "🟢 Ambiente Estável (Regime de Padrão)"
        elif k_star < 0.30:
            regime = "🟡 Pré-Ruptura (Atenção)"
        else:
            regime = "🔴 Ambiente de Ruptura (Alta Turbulência)"

        exibir_bloco_mensagem(
            "Diagnóstico do Ambiente",
            f"O regime identificado para o histórico atual é:\n\n{regime}",
            tipo="info",
        )

# ============================================================
# Painel X — 📊 Observação Histórica — Eventos k (V16)
# ============================================================

if painel == "📊 Observação Histórica — Eventos k":

    st.markdown("## 📊 Observação Histórica — Eventos k")
    st.caption("Leitura passiva do histórico. Não interfere em decisões.")

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        st.info("Nenhum evento k encontrado no histórico carregado.")
        st.stop()

    df_eventos = pd.DataFrame(eventos)

    st.markdown("### 📋 Tabela de Eventos k")
    st.dataframe(df_eventos, use_container_width=True)

    # Resumo rápido
    st.markdown("### 📈 Resumo Estatístico Simples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de eventos k", len(df_eventos))

    with col2:
        media_delta = (
            df_eventos["delta_series"].dropna().mean()
            if "delta_series" in df_eventos
            else None
        )
        st.metric(
            "Δ médio entre ks",
            f"{media_delta:.2f}" if media_delta else "—",
        )

    with col3:
        st.metric(
            "Máx k observado",
            df_eventos["k_valor"].max() if "k_valor" in df_eventos else "—",
        )

# ============================================================
# FIM — Painel X — Observação Histórica — Eventos k
# ============================================================


# ============================================================
# Painel 3 — 🛣️ Pipeline V14-FLEX ULTRA (Preparação)
# ============================================================
if painel == "🛣️ Pipeline V14-FLEX ULTRA":

    st.markdown("## 🛣️ Pipeline V14-FLEX ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes de continuar.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi do pipeline — mais duro que o k*
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Pipeline V14-FLEX ULTRA",
        painel="🛣️ Pipeline",
    ):
        st.stop()

    st.info("Iniciando processamento do Pipeline FLEX ULTRA...")

    col_pass = [c for c in df.columns if c.startswith("p")]
    matriz = df[col_pass].astype(float).values

    # ============================================================
    # Normalização
    # ============================================================
    try:
        minimo = matriz.min()
        maximo = matriz.max()
        amplitude = maximo - minimo if maximo != minimo else 1.0

        matriz_norm = (matriz - minimo) / amplitude

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro na normalização",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Estatísticas da estrada (FLEX ULTRA)
    # ============================================================
    medias = np.mean(matriz_norm, axis=1)
    desvios = np.std(matriz_norm, axis=1)

    media_geral = float(np.mean(medias))
    desvio_geral = float(np.mean(desvios))

    # Classificação simples de regime da estrada
    if media_geral < 0.35:
        estrada = "🟦 Estrada Fria (Baixa energia)"
    elif media_geral < 0.65:
        estrada = "🟩 Estrada Neutra / Estável"
    else:
        estrada = "🟥 Estrada Quente (Alta volatilidade)"

    # ============================================================
    # Clusterização leve (DX — motor original FLEX ULTRA)
    # ============================================================
    try:
        from sklearn.cluster import KMeans

        n_clusters = 3
        modelo = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        clusters = modelo.fit_predict(matriz_norm)

        centroides = modelo.cluster_centers_

    except Exception:
        clusters = np.zeros(len(matriz_norm))
        centroides = np.zeros((1, matriz_norm.shape[1]))

# ============================================================
# PARTE 2/8 — FIM
# ============================================================
# ============================================================
# PARTE 3/8 — INÍCIO
# ============================================================

    # ============================================================
    # Exibição final do pipeline
    # ============================================================
    st.markdown("### 📌 Diagnóstico do Pipeline FLEX ULTRA")

    corpo = (
        f"- Séries carregadas: **{qtd_series}**\n"
        f"- Passageiros por carro (n): **{len(col_pass)}**\n"
        f"- Energia média da estrada: **{_pc_fmt_num(media_geral, decimals=4)}**\n"
        f"- Volatilidade média: **{_pc_fmt_num(desvio_geral, decimals=4)}**\n"
        f"- Regime detectado: {estrada}\n"
        f"- Clusters formados: **{int(max(clusters)+1)}**"
    )

    exibir_bloco_mensagem(
        "Resumo do Pipeline FLEX ULTRA",
        corpo,
        tipo="info",
    )

    # ============================================================
    # Salvando na sessão para módulos seguintes (CANÔNICO)
    # ============================================================
    st.session_state["pipeline_col_pass"] = col_pass
    st.session_state["pipeline_clusters"] = clusters
    st.session_state["pipeline_centroides"] = centroides
    st.session_state["pipeline_matriz_norm"] = matriz_norm
    st.session_state["pipeline_estrada"] = estrada

    # Sinais observáveis (para governança / leitura — não altera motor)
    st.session_state["regime_identificado"] = estrada
    st.session_state["energia_media"] = float(media_geral)
    st.session_state["energia_media_estrada"] = float(media_geral)
    st.session_state["volatilidade_media"] = float(desvio_geral)
    st.session_state["clusters_formados"] = int(max(clusters) + 1) if len(np.atleast_1d(clusters)) else 0

    # ============================================================
    # M1 — SELO CANÔNICO (Governança/Mirror)
    # S2: Pipeline Consolidado (observável, mínimo, sem tocar no núcleo)
    # ============================================================
    # Regra: só carimba aqui, no caminho feliz (pós-cálculo, pré-success).
    # O Mirror lê este selo; nenhum motor depende dele.
    st.session_state["pipeline_flex_ultra_concluido"] = True
    st.session_state["pipeline_executado"] = True
    st.session_state["m1_selo_pipeline_ok"] = True
    try:
        from datetime import datetime
        st.session_state["m1_ts_pipeline_ok"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        # Falha silenciosa (observacional)
        pass

    st.success("Pipeline FLEX ULTRA concluído com sucesso!")

# ============================================================
# PARTE 3/8 — FIM
# ============================================================


# ============================================================
# Painel 4 — 🔁 Replay LIGHT
# ============================================================
if painel == "🔁 Replay LIGHT":

    st.markdown("## 🔁 Replay LIGHT — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi para replays leves
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay LIGHT",
        painel="🔁 Replay LIGHT",
    ):
        st.stop()

    st.info("Executando Replay LIGHT...")

    try:
        # DX leve = simples proximidade média entre séries vizinhas
        proximidades = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            proximidades.append(dist)

        media_proximidade = float(np.mean(proximidades))
        desvio_proximidade = float(np.std(proximidades))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay LIGHT",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Proximidade média (DX Light): **{media_proximidade:.4f}**\n"
        f"- Desvio de proximidade: **{desvio_proximidade:.4f}**\n"
        "\nValores mais altos indicam maior irregularidade."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay LIGHT",
        corpo,
        tipo="info",
    )

    st.success("Replay LIGHT concluído!")

# ============================================================
# Painel 5 — 🔁 Replay ULTRA
# ============================================================
if painel == "🔁 Replay ULTRA":

    st.markdown("## 🔁 Replay ULTRA — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Proteção anti-zumbi — Replay ULTRA é mais pesado
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Replay ULTRA completo",
        painel="🔁 Replay ULTRA",
    ):
        st.stop()

    st.info("Executando Replay ULTRA...")

    try:
        # DX Ultra = distância média entre cada série e o centróide global
        centr_global = np.mean(matriz_norm, axis=0)
        distancias = [
            np.linalg.norm(linha - centr_global) for linha in matriz_norm
        ]

        media_dx = float(np.mean(distancias))
        desvio_dx = float(np.std(distancias))

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no Replay ULTRA",
            f"Detalhes técnicos: {erro}",
            tipo="error",
        )
        st.stop()

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Distância média ao centróide (DX Ultra): **{media_dx:.4f}**\n"
        f"- Dispersão DX Ultra: **{desvio_dx:.4f}**\n"
        "\nValores maiores indicam estrada mais caótica."
    )

    exibir_bloco_mensagem(
        "Resumo do Replay ULTRA",
        corpo,
        tipo="info",
    )

    st.success("Replay ULTRA concluído!")


# ============================================================
# Painel X — 🧭 Replay Progressivo — Janela Móvel (Assistido)
# OBJETIVO:
# - Automatizar a "janela móvel" do histórico SEM você ter que ficar
#   editando arquivo/recortando histórico na mão.
# - NÃO roda motores sozinho. NÃO decide. NÃO altera Camada 4.
# - Ele apenas:
#     (1) Guarda o histórico completo uma única vez,
#     (2) Aplica um recorte (C1..Ck) como histórico ATIVO,
#     (3) Limpa chaves dependentes para forçar recalcular,
#     (4) Permite registrar o pacote gerado em cada janela,
#     (5) Avalia automaticamente o pacote registrado contra os 2 alvos seguintes.
#
# USO CANÔNICO (1 ciclo):
# 1) Carregue o histórico COMPLETO (ex.: até C5832).
# 2) Vá neste painel e aplique a janela (ex.: C5826).
# 3) Rode: Sentinelas -> Pipeline -> (TURBO opcional) -> Modo 6.
# 4) Volte aqui e "Registrar Pacote da Janela".
# 5) Veja avaliação automática em (C5827,C5828).
# 6) Repita com C5828, C5830, C5832...
# ============================================================
if painel == "🧭 Replay Progressivo — Janela Móvel (Assistido)":

    st.markdown("## 🧭 Replay Progressivo — Janela Móvel (Assistido)")
    st.caption(
        "Este painel **não gera listas sozinho** e **não muda Camada 4**. "
        "Ele só automatiza o recorte do histórico (janela móvel) e organiza o replay progressivo."
    )

    # -----------------------------------------------------
    # 🤖 Semi-automação segura (por k) — sem decisão automática
    # Objetivo: reduzir repetição (aplicar janela + organizar fila) sem executar decisões novas.
    # Regra: não usa rerun(); roda 1 k por clique; o operador segue decidindo o que levar.
    # -----------------------------------------------------
    with st.expander("🤖 Semi-automação segura (por k) — sem decisão automática", expanded=False):
        st.caption(
            "Isto **não cria motor novo** e **não decide nada**. "
            "Ele só ajuda a repetir a sequência por k com menos braço: "
            "**monta uma fila de ks** e aplica a janela ativa **1 por clique**. "
            "Depois, você roda o Pipeline/Modo 6 como sempre e registra o snapshot."
        )

        # Estado interno da semi-auto
        if "semiauto_k_fila" not in st.session_state:
            st.session_state["semiauto_k_fila"] = []
        if "semiauto_k_done" not in st.session_state:
            st.session_state["semiauto_k_done"] = []
        if "semiauto_k_last" not in st.session_state:
            st.session_state["semiauto_k_last"] = None


        # _df_full_safe pode ainda não estar definido neste ponto (ordem do painel).
        # Usamos uma versão segura apenas para limites de UI.
        _df_full_safe = st.session_state.get("historico_df_full")
        if _df_full_safe is None:
            _df_full_safe = st.session_state.get("historico_df")
        _df_full_len = int(len(_df_full_safe)) if _df_full_safe is not None else 1

        colA, colB = st.columns(2)
        with colA:
            k_inicio = st.number_input(
                "k inicial (última série INCLUÍDA)",
                min_value=1,
                max_value=_df_full_len,
                value=int(st.session_state.get("replay_janela_k_active") or _df_full_len),
                step=1,
            )
        with colB:
            qtd_passos = st.number_input("Quantos ks na fila", min_value=1, max_value=50, value=5, step=1)

        st.caption("Fila padrão: k, k-1, k-2, ... (descendo 1 por vez).")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📌 Montar/Resetar fila", use_container_width=True):
                try:
                    ks = [int(k_inicio) - i for i in range(int(qtd_passos)) if int(k_inicio) - i >= 1]
                    st.session_state["semiauto_k_fila"] = ks
                    st.session_state["semiauto_k_done"] = []
                    st.session_state["semiauto_k_last"] = None
                    st.success(f"Fila montada com {len(ks)} ks.")
                except Exception as e:
                    st.error(f"Falha ao montar fila: {e}")

        with c2:
            if st.button("✅ Aplicar PRÓXIMO k (somente janela)", use_container_width=True):
                fila = st.session_state.get("semiauto_k_fila", [])
                if not fila:
                    st.warning("Fila vazia. Clique em **Montar/Resetar fila** primeiro.")
                else:
                    k_prox = int(fila.pop(0))
                    try:
                        # aplica janela com a mesma lógica do botão oficial
                        df_recorte = _df_full_safe.head(int(k_prox)).copy()
                        st.session_state["historico_df"] = df_recorte
                        st.session_state["replay_janela_k_active"] = int(k_prox)
                        _pc_replay_limpar_chaves_dependentes_silent()

                        # universo min/max canônico (rápido)
                        try:
                            col_pass = [c for c in df_recorte.columns if str(c).startswith("p")]
                            vals = pd.to_numeric(df_recorte[col_pass].stack(), errors="coerce").dropna()
                            if len(vals) > 0:
                                st.session_state["universo_min"] = int(vals.min())
                                st.session_state["universo_max"] = int(vals.max())
                        except Exception:
                            pass

                        st.session_state["semiauto_k_last"] = int(k_prox)
                        st.session_state["semiauto_k_done"] = st.session_state.get("semiauto_k_done", []) + [int(k_prox)]
                        st.session_state["semiauto_k_fila"] = fila
                        st.success(f"Janela aplicada: C1..C{k_prox}. Agora rode **Sentinelas → Monitor → Pipeline → Modo 6**.")
                    except Exception as e:
                        st.error(f"Falha ao aplicar janela do k={k_prox}: {e}")

        with c3:
            if st.button("🧹 Limpar fila/estado semi-auto", use_container_width=True):
                st.session_state["semiauto_k_fila"] = []
                st.session_state["semiauto_k_done"] = []
                st.session_state["semiauto_k_last"] = None
                st.info("Fila/estado limpos.")

        fila = st.session_state.get("semiauto_k_fila", [])
        done = st.session_state.get("semiauto_k_done", [])
        st.markdown(
            f"**Último k aplicado:** {st.session_state.get('semiauto_k_last') or '—'}  \n"
            f"**Pendentes:** {len(fila)} · **Concluídos (janela aplicada):** {len(done)}"
        )

        # Checklist conceitual (não executa nada)
        st.markdown("### ✅ Checklist (o que ainda precisa rodar neste k)")
        ck1 = "sentinela_kstar" in st.session_state
        ck2 = "monitor_risco_resumo" in st.session_state
        ck3 = bool(st.session_state.get("pipeline_ok"))
        ck4 = bool(st.session_state.get("modo6_listas_top10") or st.session_state.get("listas_geradas"))
        st.write(f"{'✓' if ck1 else '•'} Sentinelas (k*)")
        st.write(f"{'✓' if ck2 else '•'} Monitor de Risco")
        st.write(f"{'✓' if ck3 else '•'} Pipeline V14-FLEX ULTRA")
        st.write(f"{'✓' if ck4 else '•'} Modo 6 (pacote gerado)")

        st.caption(
            "Quando o pacote estiver gerado, use o bloco **2) Registrar pacote gerado para esta janela** logo abaixo. "
            "A semi-auto não registra sozinha para não correr risco de registrar pacote errado."
        )

    st.markdown("---")

    _df_full_safe = st.session_state.get("historico_df_full")
    df_atual = st.session_state.get("historico_df")

    if df_atual is None or df_atual.empty:
        st.warning("Histórico não encontrado. Use primeiro **📁 Carregar Histórico**.")
        st.stop()

    # 1) Guardar histórico completo uma única vez
    if _df_full_safe is None:
        st.session_state["historico_df_full"] = df_atual.copy()
        _df_full_safe = st.session_state.get("historico_df_full")
        st.info(
            f"📦 Histórico completo foi guardado em memória (histórico_full). "
            f"Séries disponíveis: **{len(_df_full_safe)}**"
        )

    # 2) Contagem e limites
    total_series = _df_full_len
    if total_series < 10:
        st.warning("Histórico muito curto para replay progressivo.")
        st.stop()

    # 3) Estado atual da janela
    janela_k = int(st.session_state.get("replay_janela_k_active", len(df_atual)))
    janela_k = max(1, min(janela_k, total_series))

    colA, colB, colC = st.columns([1, 1, 1])
    colA.metric("Séries no FULL", str(total_series))
    colB.metric("Séries no ATIVO", str(len(df_atual)))
    colC.metric("Janela ativa (k)", str(janela_k))

    st.markdown("---")

    # 4) Seleção de janela (k)
    st.markdown("### 1) Selecionar e aplicar janela móvel (C1..Ck)")
    st.caption(
        "Ao aplicar a janela, o painel **recorta o histórico ativo** e "
        "**limpa chaves dependentes** (pipeline/sentinelas/pacotes) para você recalcular com segurança."
    )

    # OBS: slider foi trocado por entrada numérica para evitar resets e facilitar uso.
    k_novo = st.number_input(
        "Informe k (última série INCLUÍDA no histórico ativo)",
        min_value=10,
        max_value=total_series,
        value=int(st.session_state.get("replay_janela_k_input", janela_k)),
        step=1,
        key="replay_janela_k_input",
        help="Dica: use valores como 5826, 5828, 5830... (para simular o replay progressivo).",
    )

    # 5) Função local: limpeza de chaves dependentes (conservadora)
    def _pc_replay_limpar_chaves_dependentes_silent():
        # Chaves típicas que dependem do histórico/pipeline/pacote
        chaves = [
            "pipeline_col_pass",
            "pipeline_clusters",
            "pipeline_centroides",
            "pipeline_matriz_norm",
            "pipeline_estrada",
            "pipeline_flex_ultra_concluido",
            "pipeline_executado",
            "m1_selo_pipeline_ok",
            "m1_ts_pipeline_ok",
            "sentinela_kstar",
            "k_star",
            "k*",
            "nr_percent",
            "nr_percent_v16",
            "div_s6_mc",
            "divergencia_s6_mc",
            "indice_risco",
            "classe_risco",
            "ultima_previsao",
            "pacote_listas_atual",
            "listas_geradas",
            "listas_finais",
            "modo6_executado",
            "modo_6_executado",
            "modo_6_ativo",
            "turbo_executado",
            "turbo_ultra_executado",
            "turbo_ultra_rodou",
            "turbo_bloqueado",
            "turbo_motivo_bloqueio",
            "motor_turbo_executado",
            "listas_intercept_orbita",
        ]
        for k in chaves:
            try:
                if k in st.session_state:
                    del st.session_state[k]
            except Exception:
                pass

    # 6) Aplicar janela
    if st.button("✅ Aplicar janela (recortar histórico ativo)", use_container_width=True):
        try:
            df_recorte = _df_full_safe.head(int(k_novo)).copy()
            st.session_state["historico_df"] = df_recorte
            st.session_state["replay_janela_k_active"] = int(k_novo)  # fixa janela ativa (não altera widget)
            _pc_replay_limpar_chaves_dependentes_silent()

            # Atualizar universo min/max canônico (derivado do recorte) — versão rápida (sem iterrows)
            try:
                col_pass = [c for c in df_recorte.columns if str(c).startswith("p")]
                if col_pass:
                    vals = df_recorte[col_pass].to_numpy().ravel()
                    # Filtra apenas inteiros positivos
                    vals = [int(v) for v in vals if str(v).strip() not in ("", "nan", "None")]
                    vals = [v for v in vals if v > 0]
                    if vals:
                        st.session_state["universo_min"] = int(min(vals))
                        st.session_state["universo_max"] = int(max(vals))
            except Exception:
                pass

            st.success(
                f"Janela aplicada: histórico ativo agora está em **C1..C{k_novo}** "
                f"(total: {len(df_recorte)} séries).\n\n"
                "Agora rode **Sentinelas → Pipeline → (TURBO opcional) → Modo 6**."
            )
        except Exception as e:
            st.error(f"Falha ao aplicar janela: {e}")
        st.stop()

    st.markdown("---")

    # 7) Registro do pacote da janela
    st.markdown("### 2) Registrar pacote gerado para esta janela")
    st.caption(
        "Depois de rodar o Modo 6 (e gerar o pacote), volte aqui e registre. "
        "O painel guarda um snapshot por janela (k)."
    )

    pacote_atual = st.session_state.get("pacote_listas_atual")
    if not pacote_atual:
        st.info(
            "Nenhum pacote atual encontrado. "
            "Rode o **🎯 Modo 6 Acertos — Execução** para gerar e congelar um pacote."
        )
    else:
        st.success(f"📦 Pacote atual detectado: **{len(pacote_atual)}** listas")

    # estrutura: {k: {"ts": "...", "qtd": int, "listas": [[...], ...]}}
    if "replay_progressivo_pacotes" not in st.session_state:
        st.session_state["replay_progressivo_pacotes"] = {}

    pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
    st.caption(f"Pacotes registrados até agora: **{len(pacotes_reg)}**")
    # -------------------------------------------------------------
    # 🧊 SNAPSHOT P0 — CANÔNICO (pré-Camada 4 · leitura apenas)
    # -------------------------------------------------------------
    # Este snapshot existe para permitir varreduras ex-post (P1/P2/...) sem contaminar o P0.
    # Ele NÃO muda listas, NÃO decide volume e NÃO toca Camada 4.
    if "snapshot_p0_canonic" not in st.session_state:
        st.session_state["snapshot_p0_canonic"] = {}

    snapshot_p0_reg = st.session_state.get("snapshot_p0_canonic", {})
    st.caption(f"Snapshots P0 registrados até agora: **{len(snapshot_p0_reg)}**")

    # 🕺 Ritmo/Dança (ex-post · pré-C4) — leitura automática (quando Parabólica tiver gov)
    st.markdown("### 🕺 Ritmo/Dança (ex-post · pré-C4)")
    ritmo_info = st.session_state.get("ritmo_danca_info")
    if not isinstance(ritmo_info, dict) or not ritmo_info:
        ritmo_info = {"ritmo_global": "N/D", "motivos": ["sem_dados"], "sinais": {}}
    st.json(ritmo_info, expanded=False)


    colR1, colR2 = st.columns([1, 1])
    with colR1:
        if st.button("📌 Registrar pacote da janela atual", use_container_width=True, disabled=not bool(pacote_atual)):
            try:
                from datetime import datetime
                k_reg = int(st.session_state.get("replay_janela_k_active", k_novo))
                # --- V16 (CALIB LEVE) — aplica no registro (pré-C4) + baseline interno real ---
                calib_leve = st.session_state.get("v16_calib_leve_last_summary") or {}
                if not isinstance(calib_leve, dict):
                    calib_leve = {}

                _resp_apply = pc_v16_aplicar_resp_no_pacote(
                    pacote_atual=pacote_atual,
                    k_reg=int(k_reg),
                    universo_min=int(st.session_state.get("UNIVERSE_MIN", 1) or 1),
                    universo_max=int(st.session_state.get("UNIVERSE_MAX", 60) or 60),
                    calib_leve_summary=calib_leve,
                    top10_override=(pacote_atual[:10] if isinstance(pacote_atual, list) else None),
                    memoria_sufocadores=None,
                    cap_pct=0.65,
                    core_min=0.40,
                )

                pacote_baseline = _resp_apply.get("pacote_baseline", [])
                pacote_store = _resp_apply.get("pacote_store", pacote_baseline)
                resp_info = _resp_apply.get("resp_info", {"aplicado": False, "motivo": "nao_aplicado"})
                calib_active = bool(_resp_apply.get("calib_active", False))
                calib_should_apply = bool(_resp_apply.get("calib_should_apply", False))
                calib_applied = bool(_resp_apply.get("calib_applied", False))
                I_mean = float(_resp_apply.get("I_mean", 0.0))
                I_max = float(_resp_apply.get("I_max", I_mean))
                I2_mean = float(_resp_apply.get("I2_mean", 0.0))
                I2_max = float(_resp_apply.get("I2_max", I2_mean))
                THR_BASE = float(_resp_apply.get("thr_base", 0.25))

                calib_leve_store = dict(calib_leve)
                calib_leve_store.update({
                    "active": calib_active,
                    "applied": calib_applied,
                    "aplicada_no_pacote": calib_applied,
                    "thr_base": THR_BASE,
                    "I_mean": float(I_mean),
                    "I_max": float(I_max),
                    "I2_mean": float(I2_mean),
                    "I2_max": float(I2_max),
                    "resp_info": resp_info,
                    "diff_conteudo_qtd": int(_resp_apply.get("diff_conteudo_qtd", 0)),
                    "diff_posicao_qtd": int(_resp_apply.get("diff_posicao_qtd", 0)),
                    "diff_dedup_qtd": int(_resp_apply.get("diff_dedup_qtd", 0)),
                    "reason": "I2>=thr_base" if calib_should_apply else ("I2<thr_base" if calib_active else "I2=0"),
                })

                # --- V9 (BLOCO B) — snapshot estrutural do pacote (OBSERVACIONAL / ex-post) ---
                # Regra: separar o valor digitado (widget) do estado ativo e registrar um snapshot por janela.
                # Isso NÃO decide nada e NÃO altera listas — apenas guarda estrutura para avaliação posterior.
                try:
                    v8_snap = st.session_state.get("v8_borda_qualificada") or {}
                    # Se não houver snapshot V8 válido nesta rodada, recalcula de forma canônica a partir do pacote atual
                    if not isinstance(v8_snap, dict) or v8_snap.get("meta", {}).get("status") not in ("ok", "presenca_vazia"):
                        v8_snap = v8_classificar_borda_qualificada(
                            listas=[list(map(int, lst)) for lst in pacote_atual],
                            base_n=10,
                            core_presenca_min=0.60,
                            quase_delta=0.12,
                            max_borda_interna=6,
                            universo_min=st.session_state.get("universo_min"),
                            universo_max=st.session_state.get("universo_max"),
                            rigidez_info=st.session_state.get("v16_rigidez_info"),
                        )
                except Exception:
                    v8_snap = {"core": [], "quase_core": [], "borda_interna": [], "borda_externa": [], "meta": {"status": "snap_falhou"}}

                # Universo do pacote (união) — usado para classificar "miolo do pacote" vs "fora do pacote"
                # Robustez: não derrubar o universo inteiro por 1 lista malformada (string/dict/None/etc).
                universo_set = set()
                try:
                    for lst in (pacote_atual or []):
                        if isinstance(lst, (list, tuple, set)):
                            for x in lst:
                                try:
                                    universo_set.add(int(x))
                                except Exception:
                                    pass
                        else:
                            # Se vier algo estranho (ex.: string), ignora em vez de quebrar tudo.
                            continue
                except Exception:
                    pass
                universo_pacote = sorted(universo_set)

                pacotes_reg[k_reg] = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "qtd": int(len(pacote_atual)),
                    "calib_leve": calib_leve_store,
                    "listas": [list(map(int, lst)) for lst in pacote_store],
                    "snap_v9": {
                        "core": list(map(int, (v8_snap.get("core") or []))),
                        "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                        "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                        "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                        "universo_pacote": list(map(int, universo_pacote)),
                        "meta": v8_snap.get("meta") or {},
                    },
                }
                
                # --- Snapshot P0 canônico (pré-C4 · leitura) ---
                try:
                    # Frequência de passageiros (aparições no conjunto de listas)
                    freq_passageiros = {}
                    for lst in pacote_atual:
                        for x in lst:
                            xi = int(x)
                            freq_passageiros[xi] = freq_passageiros.get(xi, 0) + 1

                    # Assinatura (para rastreabilidade do snapshot)
                    sig_raw = json.dumps([list(map(int, lst)) for lst in pacote_atual], ensure_ascii=False, sort_keys=True)
                    sig = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()[:16]
                except Exception:
                    freq_passageiros = {}
                    sig = "N/D"

                snapshot_p0_reg[k_reg] = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "k": int(k_reg),
                    "qtd_listas": int(len(pacote_atual)),
                    "listas": [list(map(int, lst)) for lst in pacote_store],
                    "universo_pacote": list(map(int, universo_pacote)),
                    "freq_passageiros": {str(int(k)): int(v) for k, v in sorted(freq_passageiros.items(), key=lambda kv: (-kv[1], kv[0]))},
                    "snap_v8": {
                        "core": list(map(int, (v8_snap.get("core") or []))),
                        "quase_core": list(map(int, (v8_snap.get("quase_core") or []))),
                        "borda_interna": list(map(int, (v8_snap.get("borda_interna") or []))),
                        "borda_externa": list(map(int, (v8_snap.get("borda_externa") or []))),
                        "meta": v8_snap.get("meta") or {},
                    },
                    "assinatura": sig,
                    "nota": "Snapshot P0 canônico — leitura apenas (pré-C4). Não altera Camada 4.",
                }
                st.session_state["snapshot_p0_canonic"] = snapshot_p0_reg
                st.session_state["replay_progressivo_pacotes"] = pacotes_reg
                st.success(f"Pacote registrado para janela C1..C{k_reg}.")

                # 🧠 Atualiza Memória Estrutural automaticamente (Jogador B) ao registrar Snapshot P0
                try:
                    _df_full_me = st.session_state.get("_df_full_safe") if st.session_state.get("_df_full_safe") is not None else st.session_state.get("historico_df")
                    v16_me_update_auto(_df_full_safe=_df_full_me, snapshots_map=st.session_state.get("snapshot_p0_canonic") or {})
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Falha ao registrar pacote: {e}")
    with colR2:
        if st.button("🧹 Limpar todos os pacotes registrados", use_container_width=True, disabled=(len(pacotes_reg) == 0)):
            st.session_state["replay_progressivo_pacotes"] = {}
            st.session_state["snapshot_p0_canonic"] = {}
            # limpa memória estrutural (auditável)
            st.session_state["me_info"] = {}
            st.session_state["me_status"] = "DESLIGADA"
            st.session_state["me_status_info"] = {"status": "DESLIGADA", "motivo": "limpeza_manual"}
            st.success("Pacotes registrados foram limpos.")

    
    # -------------------------------------------------------------
    # 🧊 Snapshot P0 (canônico) — visão rápida (leitura)
    # -------------------------------------------------------------
    
    # -----------------------------------------------------
    # 📊 Replay Estatístico Automático Incremental (SAFE)
    # Objetivo: eliminar cata-milho manual.
    # - Roda por lotes pequenos (CPU-budget)
    # - Usa pc_semi_auto_processar_um_k (recorte→sentinela→monitor→pipeline→modo6→registro)
    # - Não mexe na Camada 4 (apenas registra pacotes/snapshots do Replay)
    # -----------------------------------------------------
    with st.expander("📊 Replay Estatístico — Prova Automática (SAFE)", expanded=False):
        st.caption(
            "SAFE = **Replay automático incremental**: ele percorre janelas por k em **lotes pequenos** "
            "e vai **registrando pacotes/snapshots** para permitir prova estatística (PRÉ-E1/MICRO-E1) "
            "sem cata-milho. **Não gera motor novo** e **não altera Camada 4**."
        )

        # Base full segura
        _df_full_safe2 = st.session_state.get("historico_df_full")
        if _df_full_safe2 is None:
            _df_full_safe2 = st.session_state.get("historico_df")
        if _df_full_safe2 is None or getattr(_df_full_safe2, "empty", True):
            st.warning("Carregue um histórico primeiro.")
        else:
            _n_full = int(len(_df_full_safe2))

            # Defaults canônicos: do k atual até (k atual - 60), limitado ao histórico
            _k_default = int(st.session_state.get("replay_janela_k_active") or _n_full)
            _k_de = _k_default
            _k_ate = max(10, _k_default - 60)

            cA, cB, cC = st.columns(3)
            with cA:
                k_de = st.number_input("k DE (início)", min_value=10, max_value=_n_full, value=int(st.session_state.get("safe_k_de") or _k_de), step=1)
            with cB:
                k_ate = st.number_input("k ATÉ (fim)", min_value=10, max_value=_n_full, value=int(st.session_state.get("safe_k_ate") or _k_ate), step=1)
            with cC:
                lote = st.number_input("Lote (ks por clique)", min_value=1, max_value=20, value=int(st.session_state.get("safe_lote") or 3), step=1)

            # Persistir preferências
            st.session_state["safe_k_de"] = int(k_de)
            st.session_state["safe_k_ate"] = int(k_ate)
            st.session_state["safe_lote"] = int(lote)

            # Cursor do SAFE (descendo)
            if "safe_cursor_k" not in st.session_state or st.session_state.get("safe_cursor_k") is None:
                st.session_state["safe_cursor_k"] = int(k_de)

            # Se o operador mudar intervalo, resetar cursor
            if st.button("🔄 Resetar cursor SAFE para k DE", use_container_width=True):
                st.session_state["safe_cursor_k"] = int(k_de)
                st.success(f"Cursor SAFE resetado para k={int(k_de)}.")

            cursor = int(st.session_state.get("safe_cursor_k") or int(k_de))

            # Normalizar direção (sempre descendo)
            _k_hi = int(max(k_de, k_ate))
            _k_lo = int(min(k_de, k_ate))

            if cursor > _k_hi:
                cursor = _k_hi
                st.session_state["safe_cursor_k"] = int(cursor)
            if cursor < _k_lo:
                cursor = _k_lo
                st.session_state["safe_cursor_k"] = int(cursor)

            st.markdown(f"**Intervalo:** k={_k_hi} ↓ ... ↓ k={_k_lo}  \n**Cursor atual:** k={cursor}")

            # Progresso aproximado
            total = (_k_hi - _k_lo + 1)
            done = max(0, (_k_hi - cursor))
            prog = 0.0 if total <= 0 else min(1.0, done / total)
            st.progress(prog)

            # Executar lote
            if st.button("▶️ Rodar PRÓXIMO lote SAFE (registrar pacotes)", use_container_width=True):
                k_atual = int(st.session_state.get("safe_cursor_k") or cursor)
                if k_atual < _k_lo:
                    st.info("SAFE já concluiu o intervalo.")
                else:
                    to_run = []
                    for i in range(int(lote)):
                        kk = int(k_atual) - i
                        if kk < _k_lo:
                            break
                        to_run.append(kk)

                    resultados = []
                    with st.spinner(f"SAFE rodando lote: {to_run[0]} ↓ ... ↓ {to_run[-1]}"):
                        for kk in to_run:
                            res = pc_semi_auto_processar_um_k(_df_full_safe=_df_full_safe2, k_exec=int(kk))
                            resultados.append(res)

                    # Atualizar cursor (desce)
                    novo_cursor = int(to_run[-1]) - 1
                    st.session_state["safe_cursor_k"] = int(novo_cursor)

                    # Resumo do lote
                    ok_cnt = sum(1 for r in resultados if isinstance(r, dict) and r.get("ok"))
                    fail = [r for r in resultados if not (isinstance(r, dict) and r.get("ok"))]
                    st.success(f"Lote concluído: {ok_cnt}/{len(resultados)} ks processados com sucesso. Cursor → k={int(novo_cursor)}.")
                    if fail:
                        st.warning("Alguns ks falharam (não trava o app; apenas registra o erro):")
                        st.json(fail[:5])

                    st.caption("Após alguns lotes, use **Avaliar pacotes registrados** abaixo para atualizar df_eval e a prova (PRÉ‑E1/MICRO‑E1).")
    if len(snapshot_p0_reg) > 0:
        try:
            k_ultimo = max(snapshot_p0_reg.keys())
        except Exception:
            k_ultimo = None

        with st.expander("🧊 Snapshot P0 — pacote-base canônico (último)", expanded=False):
            if k_ultimo is None:
                st.info("Nenhum snapshot disponível.")
            else:
                snap = snapshot_p0_reg.get(k_ultimo) or {}
                st.markdown(f"**Último snapshot:** janela **C1..C{k_ultimo}** · listas: **{snap.get('qtd_listas','N/D')}** · assinatura: `{snap.get('assinatura','N/D')}`")
                # Mostra um resumo enxuto para auditoria (sem 'freq_passageiros' completo, para não poluir)
                resumo = {
                    "ts": snap.get("ts"),
                    "k": snap.get("k"),
                    "qtd_listas": snap.get("qtd_listas"),
                    "universo_pacote_len": len(snap.get("universo_pacote") or []),
                    "snap_v8": snap.get("snap_v8") or {},
                    "nota": snap.get("nota"),
                }
                st.json(resumo)
    else:
        st.info("🧊 Snapshot P0 ainda não registrado nesta sessão. (Registre um pacote por janela acima.)")

    # -------------------------------------------------------------
    # 🧪 Série Suficiente (SS) — bloco visível (informativo)
    # -------------------------------------------------------------
    try:
        ss_info = v16_calcular_ss(_df_full_safe=_df_full_safe, snapshots_map=snapshot_p0_reg)
        st.session_state["ss_info"] = ss_info
        st.session_state["ss_status"] = "ATINGIDA" if ss_info.get("status") else "NAO_ATINGIDA"
        v16_render_bloco_ss(ss_info)
        # 🧠 Memória Estrutural (SEM_RITMO) — bloco informativo
        try:
            # Recalcula sempre o status/resultado da Memória Estrutural com base no que existe AGORA
            # (evita ficar "nao_calculada" após os snapshots/SS mudarem)
            v16_me_update_auto()
            v16_render_bloco_me(st.session_state.get("me_info"), st.session_state.get("me_status_info"), st.session_state.get("ss_info"))
        except Exception:
            pass
    except Exception:
        pass
    st.markdown("---")

    # 8) Avaliação automática (contra os 2 alvos seguintes)
    st.markdown("### 3) Avaliar pacotes registrados (2 alvos seguintes)")
    st.caption(
        "Para cada janela k registrada, o painel testa o pacote contra os alvos **C(k+1)** e **C(k+2)** "
        "do histórico FULL (quando existirem)."
    )

    pacotes_reg = st.session_state.get("replay_progressivo_pacotes", {})
    if not pacotes_reg:
        st.info("Nenhum pacote registrado ainda.")
        st.stop()

    # colunas de passageiros
    col_pass_full = (
    [c for c in _df_full_safe.columns if isinstance(c, str) and c.lower().startswith('p')]
    if ('_df_full_safe' in globals()) and hasattr(_df_full_safe, 'columns') else []
)

    if not col_pass_full:
        st.error("Não consegui identificar colunas de passageiros no histórico FULL.")
        st.stop()

    def _alvo_da_linha(idx0: int):
        # idx0 é 0-based (linha do DF)
        row = _df_full_safe.iloc[idx0]
        alvo = []
        for c in col_pass_full:
            try:
                v = int(row[c])
                if v > 0:
                    alvo.append(v)
            except Exception:
                pass
        return set(alvo)

    
    # --- V9 (BLOCO B) — Memória de Borda (ex-post, observacional) ---
    # Para cada janela k registrada, além do "melhor acerto do pacote", também medimos:
    # onde caíram os acertos (CORE / quase-CORE / borda interna / borda externa / miolo do pacote / fora do pacote)
    def _v9_get_sets(info: dict):
        snap = info.get("snap_v9") or {}
        core = set(map(int, snap.get("core") or []))
        quase = set(map(int, snap.get("quase_core") or []))
        b_in = set(map(int, snap.get("borda_interna") or []))
        b_ex = set(map(int, snap.get("borda_externa") or []))
        uni = set(map(int, snap.get("universo_pacote") or []))
        return core, quase, b_in, b_ex, uni

    def _v9_contar_origens(alvo_set: set, core: set, quase: set, b_in: set, b_ex: set, uni: set):
        # Classificação disjunta (ordem canônica):
        # CORE > quase-CORE > borda interna > borda externa > miolo do pacote > fora do pacote
        out = {
            "core": 0,
            "quase": 0,
            "borda_in": 0,
            "borda_ex": 0,
            "miolo": 0,
            "fora": 0,
        }
        if not alvo_set:
            return out
        for n in alvo_set:
            if n in core:
                out["core"] += 1
            elif n in quase:
                out["quase"] += 1
            elif n in b_in:
                out["borda_in"] += 1
            elif n in b_ex:
                out["borda_ex"] += 1
            elif (uni and n in uni):
                out["miolo"] += 1
            else:
                out["fora"] += 1
        return out

    def _v9_trave_proximidade(alvo_set: set, uni: set, thr: int = 2):
        """Métrica contínua de 'trave/proximidade' (ex‑post).
        Para cada número do alvo que caiu FORA do universo do pacote, mede o quão perto ele ficou
        (distância mínima até qualquer número do universo do pacote).

        - fora_perto: fora do pacote, mas com distância <= thr
        - fora_longe: fora do pacote e distância > thr
        - dist_media: média das distâncias mínimas (apenas dos que ficaram fora)

        Observacional. Não altera motor nem Camada 4.
        """
        if not alvo_set:
            return {"fora_perto": 0, "fora_longe": 0, "dist_media": None, "dist_max": None, "fora_perto_nums": [], "fora_longe_nums": []}
        if not uni:
            # sem universo do pacote, não há como medir proximidade
            return {"fora_perto": 0, "fora_longe": int(len(alvo_set)), "dist_media": None, "dist_max": None, "fora_perto_nums": [], "fora_longe_nums": [int(x) for x in sorted(list(alvo_set))]}
        uni_list = sorted(list(uni))
        dists = []
        fora_perto = 0
        fora_longe = 0
        fora_perto_nums = []
        fora_longe_nums = []
        for n in alvo_set:
            if n in uni:
                continue
            # distância mínima até o universo
            md = min(abs(n - u) for u in uni_list)
            dists.append(md)
            if md <= thr:
                fora_perto += 1
                fora_perto_nums.append(int(n))
            else:
                fora_longe += 1
                fora_longe_nums.append(int(n))
        if dists:
            dist_media = float(sum(dists)) / float(len(dists))
            dist_max = int(max(dists))
        else:
            dist_media = 0.0
            dist_max = 0
        return {"fora_perto": int(fora_perto), "fora_longe": int(fora_longe), "dist_media": dist_media, "dist_max": dist_max, "fora_perto_nums": sorted(list(set(fora_perto_nums))), "fora_longe_nums": sorted(list(set(fora_longe_nums)))}

    resultados = []
    for k_reg, info in sorted(pacotes_reg.items(), key=lambda x: int(x[0])):
        k_reg = int(k_reg)
        idx1 = k_reg  # C(k+1) em 0-based
        idx2 = k_reg + 1  # C(k+2)
        if idx1 >= total_series:
            continue

        alvo1 = _alvo_da_linha(idx1)
        alvo2 = _alvo_da_linha(idx2) if idx2 < total_series else None

        listas = info.get("listas", [])
        # melhor acerto em cada alvo
        best1 = 0
        best2 = 0
        if alvo1 and listas:
            for lst in listas:
                best1 = max(best1, len(set(lst) & alvo1))
        if alvo2 and listas:
            for lst in listas:
                best2 = max(best2, len(set(lst) & alvo2))
        # Baseline interno (quando disponível): listas_baseline
        listas_base = info.get("listas_baseline", None)
        base_best1 = None
        base_best2 = None
        if isinstance(listas_base, list) and listas_base:
            try:
                base_best1 = 0
                base_best2 = 0
                if alvo1:
                    for lst in listas_base:
                        base_best1 = max(base_best1, len(set(lst) & alvo1))
                if alvo2:
                    for lst in listas_base:
                        base_best2 = max(base_best2, len(set(lst) & alvo2))
            except Exception:
                base_best1 = None
                base_best2 = None

        # --- V9 (BLOCO B): onde caíram os acertos (ex-post) ---
        core, quase, b_in, b_ex, uni = _v9_get_sets(info)
        org1 = _v9_contar_origens(alvo1, core, quase, b_in, b_ex, uni) if alvo1 else None
        org2 = _v9_contar_origens(alvo2, core, quase, b_in, b_ex, uni) if alvo2 else None

        # --- Trave/Proximidade (ex-post): 'fora, mas perto' vs 'fora e longe' ---
        tr1 = _v9_trave_proximidade(alvo1, uni, thr=2) if alvo1 else None
        tr2 = _v9_trave_proximidade(alvo2, uni, thr=2) if alvo2 else None

        resultados.append({
            "janela_k": k_reg,
            "qtd_listas": int(info.get("qtd", 0)),
            "alvo_1": f"C{k_reg+1}",
            "best_acerto_alvo_1": int(best1),
            "core_hit_1": int(org1.get("core")) if org1 else None,
            "quase_hit_1": int(org1.get("quase")) if org1 else None,
            "borda_in_hit_1": int(org1.get("borda_in")) if org1 else None,
            "borda_ex_hit_1": int(org1.get("borda_ex")) if org1 else None,
            "miolo_hit_1": int(org1.get("miolo")) if org1 else None,
            "fora_hit_1": int(org1.get("fora")) if org1 else None,
            "fora_perto_1": int(tr1.get("fora_perto")) if tr1 else None,
            "fora_longe_1": int(tr1.get("fora_longe")) if tr1 else None,
            "dist_media_fora_1": tr1.get("dist_media") if tr1 else None,
            "dist_max_fora_1": tr1.get("dist_max") if tr1 else None,
            "fora_perto_nums_1": json.dumps(tr1.get("fora_perto_nums") if tr1 else []) if tr1 else "[]",
            "fora_longe_nums_1": json.dumps(tr1.get("fora_longe_nums") if tr1 else []) if tr1 else "[]",
            "alvo_2": f"C{k_reg+2}" if alvo2 is not None else "—",
            "best_acerto_alvo_2": int(best2) if alvo2 is not None else None,
            "best_acerto_alvo_1_baseline": int(base_best1) if base_best1 is not None else None,
            "best_acerto_alvo_2_baseline": int(base_best2) if base_best2 is not None else None,
            "delta_best_1": (int(best1) - int(base_best1)) if base_best1 is not None else None,
            "delta_best_2": (int(best2) - int(base_best2)) if base_best2 is not None else None,
            "core_hit_2": int(org2.get("core")) if org2 else None,
            "quase_hit_2": int(org2.get("quase")) if org2 else None,
            "borda_in_hit_2": int(org2.get("borda_in")) if org2 else None,
            "borda_ex_hit_2": int(org2.get("borda_ex")) if org2 else None,
            "miolo_hit_2": int(org2.get("miolo")) if org2 else None,
            "fora_hit_2": int(org2.get("fora")) if org2 else None,
            "fora_perto_2": int(tr2.get("fora_perto")) if tr2 else None,
            "fora_longe_2": int(tr2.get("fora_longe")) if tr2 else None,
            "dist_media_fora_2": tr2.get("dist_media") if tr2 else None,
            "dist_max_fora_2": tr2.get("dist_max") if tr2 else None,
            "fora_perto_nums_2": json.dumps(tr2.get("fora_perto_nums") if tr2 else []) if tr2 else "[]",
            "fora_longe_nums_2": json.dumps(tr2.get("fora_longe_nums") if tr2 else []) if tr2 else "[]",
            "calib_active": bool((info.get("calib_leve") or {}).get("active", False)),
            "calib_applied": bool((info.get("calib_leve") or {}).get("applied", False)),
            "calib_I_mean": float((info.get("calib_leve") or {}).get("I_mean", 0.0) or 0.0),
            "calib_I_max": float((info.get("calib_leve") or {}).get("I_max", 0.0) or 0.0),
            "calib_reason": str((info.get("calib_leve") or {}).get("reason", "")),
            "ts_registro": str(info.get("ts", "")),
        })

    if not resultados:
        st.warning("Nenhum resultado para exibir (verifique se as janelas k registradas têm alvos seguintes no FULL).")
        st.stop()

    df_res = pd.DataFrame(resultados).sort_values(["janela_k"], ascending=True)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # ✅ Persistência canônica para painéis retro (ex.: Compressão do Alvo)
    # df_eval = base de avaliação derivada do Replay/SAFE (pré-C4 · observacional)
    st.session_state["df_eval"] = df_res.copy()

    # -------------------------------------------------------------
    # 📊 Replay Estatístico Automático Incremental (SAFE)
    # OBJETIVO:
    # - Produzir "prova" de janela (4+) SEM cata-milho e SEM rodar 200 vezes na mão.
    # - Usa SOMENTE df_res (resultado do Replay Progressivo já registrado).
    # - Não executa motor, não altera Camada 4, não muda listas.
    # - Custo: leve (cálculo em tabela já pronta).
    # -------------------------------------------------------------
    with st.expander("📊 Replay Estatístico — Prova Automática (SAFE)", expanded=True):
        st.caption(
            "Este bloco é **pré-C4 · observacional · auditável**. "
            "Ele NÃO roda Pipeline, NÃO gera listas e NÃO registra pacotes. "
            "Ele apenas transforma o df_eval (df_res) em métricas objetivas de 'janela' (4+) e micro-sinais."
        )

        try:
            # Transformar em série de alvos avaliados (cada linha possui alvo_1 e alvo_2)
            best1_raw = df_res.get("best_hit_1")
            best2_raw = df_res.get("best_hit_2")

            def _as_series(x):
                # Garante Series (evita erro tipo numpy.float64 sem .notna)
                if isinstance(x, pd.Series):
                    return x
                if isinstance(x, (list, tuple, np.ndarray)):
                    return pd.Series(list(x))
                # DataFrame com 1 coluna
                if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                    return x.iloc[:, 0]
                # escalar (float/int/None)
                return pd.Series([x])

            best1 = pd.to_numeric(_as_series(best1_raw), errors="coerce")
            best2 = pd.to_numeric(_as_series(best2_raw), errors="coerce")

            # targets avaliados (conta alvos existentes)
            n_targets = 0
            if pd.notna(best1).any():
                n_targets += 1
            if pd.notna(best2).any():
                n_targets += 1
# concat cronológico: (janela_k, alvo1) depois (janela_k, alvo2)
            serie = []
            if "janela_k" in df_res.columns:
                df_tmp = df_res.sort_values(["janela_k"], ascending=True).copy()
            else:
                df_tmp = df_res.copy()

            for _, rr in df_tmp.iterrows():
                try:
                    b1 = _pc_safe_float(rr.get("best_acerto_alvo_1", rr.get("best_hit_1")), None)
                    if b1 is not None:
                        serie.append(int(b1))
                except Exception:
                    pass
                try:
                    b2 = _pc_safe_float(rr.get("best_acerto_alvo_2", rr.get("best_hit_2")), None)
                    if b2 is not None:
                        serie.append(int(b2))
                except Exception:
                    pass

            n_targets = int(len(serie))

            # métricas globais
            any_4p = any(v >= 4 for v in serie) if serie else False
            max_best = max(serie) if serie else None
            rate_4p = (sum(1 for v in serie if v >= 4) / len(serie)) if serie else 0.0
            rate_3p = (sum(1 for v in serie if v >= 3) / len(serie)) if serie else 0.0
            zero_rate = (sum(1 for v in serie if v <= 0) / len(serie)) if serie else 0.0

            # janela móvel (W) — prova local
            W = 60
            w_used = min(W, len(serie)) if serie else 0
            serie_w = serie[-w_used:] if w_used > 0 else []
            any_4p_w = any(v >= 4 for v in serie_w) if serie_w else False
            rate_4p_w = (sum(1 for v in serie_w if v >= 4) / len(serie_w)) if serie_w else 0.0
            rate_3p_w = (sum(1 for v in serie_w if v >= 3) / len(serie_w)) if serie_w else 0.0
            zero_rate_w = (sum(1 for v in serie_w if v <= 0) / len(serie_w)) if serie_w else 0.0
            # avg_best (implícito, derivado da série; e conferência via distribuição)
            avg_best = (float(np.mean(serie)) if len(serie) > 0 else None)
            avg_best_w = (float(np.mean(serie_w)) if len(serie_w) > 0 else None)

            # erro de fechamento: quão longe, em média, estamos de "fechar" (6/6) numa lista
            # (não é janela; é uma métrica de fechamento)
            fechamento_gap = (float(6.0 - avg_best) if avg_best is not None else None)
            fechamento_gap_w = (float(6.0 - avg_best_w) if avg_best_w is not None else None)
            fechamento_gap_norm = (float((6.0 - avg_best) / 6.0) if avg_best is not None else None)
            fechamento_gap_norm_w = (float((6.0 - avg_best_w) / 6.0) if avg_best_w is not None else None)

            # taxa de 5+ (para distinguir "perto" de "fecha forte")
            rate_5p = (sum(1 for v in serie if v >= 5) / len(serie)) if len(serie) > 0 else 0.0
            rate_5p_w = (sum(1 for v in serie_w if v >= 5) / len(serie_w)) if len(serie_w) > 0 else 0.0


            # distribuição (auditável)
            dist = {i: 0 for i in range(0, 7)}
            for v in serie:
                if v < 0:
                    continue
                if v > 6:
                    v = 6
                dist[int(v)] = dist.get(int(v), 0) + 1

            
            # conferência: avg_best derivado da distribuição (quando aplicável)
            total_dist = int(sum(dist.values()))
            avg_best_from_dist = (float(sum(float(i) * float(dist.get(i, 0)) for i in range(0, 7)) / total_dist) if total_dist > 0 else None)
            # MPF — Mapa de Perda de Fechamento (gap x trave/distância)
            mpf_rows = []
            for _, r in df_res.iterrows():
                for suf in ["1", "2"]:
                    best = r.get(f"best_acerto_alvo_{suf}")
                    if best is None or (isinstance(best, float) and pd.isna(best)):
                        continue
                    d_med = r.get(f"dist_media_fora_alvo_{suf}")
                    d_max = r.get(f"dist_max_fora_alvo_{suf}")
                    fora_tot = r.get(f"fora_total_alvo_{suf}")
                    fora_perto = r.get(f"fora_perto_alvo_{suf}")
                    mpf_rows.append((
                        int(best),
                        (float(d_med) if (d_med is not None and not pd.isna(d_med)) else None),
                        (float(d_max) if (d_max is not None and not pd.isna(d_max)) else None),
                        (int(fora_tot) if (fora_tot is not None and not pd.isna(fora_tot)) else 0),
                        (int(fora_perto) if (fora_perto is not None and not pd.isna(fora_perto)) else 0),
                    ))

            mpf_map = {}
            trave_ratio_global = None
            indice_mpf = None
            if mpf_rows:
                from collections import defaultdict
                agg = defaultdict(lambda: {"n": 0, "sum_d_med": 0.0, "cnt_d_med": 0, "sum_d_max": 0.0, "cnt_d_max": 0, "fora_tot": 0, "fora_perto": 0})
                for best, dmed, dmax, ft, fp in mpf_rows:
                    gap = int(6 - int(best))
                    a = agg[gap]
                    a["n"] += 1
                    if dmed is not None:
                        a["sum_d_med"] += float(dmed)
                        a["cnt_d_med"] += 1
                    if dmax is not None:
                        a["sum_d_max"] += float(dmax)
                        a["cnt_d_max"] += 1
                    a["fora_tot"] += int(ft)
                    a["fora_perto"] += int(fp)

                total = int(len(mpf_rows))
                for gap, a in sorted(agg.items(), key=lambda x: x[0]):
                    mpf_map[str(gap)] = {
                        "targets": int(a["n"]),
                        "share": float(a["n"] / total) if total > 0 else 0.0,
                        "avg_dist_media_fora": (float(a["sum_d_med"] / a["cnt_d_med"]) if a["cnt_d_med"] > 0 else None),
                        "avg_dist_max_fora": (float(a["sum_d_max"] / a["cnt_d_max"]) if a["cnt_d_max"] > 0 else None),
                        "trave_ratio": (float(a["fora_perto"] / a["fora_tot"]) if a["fora_tot"] > 0 else None),
                    }

                fora_tot_all = int(sum(a["fora_tot"] for a in agg.values()))
                fora_perto_all = int(sum(a["fora_perto"] for a in agg.values()))
                trave_ratio_global = (float(fora_perto_all / fora_tot_all) if fora_tot_all > 0 else None)
                if trave_ratio_global is not None:
                    indice_mpf = float(fechamento_gap_norm) * float(1.0 - trave_ratio_global)

            st.markdown("### ✅ Prova objetiva (sem achismo)")
            st.json({
                "targets_avaliados": int(n_targets),
                "any_4p_seen": bool(any_4p),
                "max_best": int(max_best) if max_best is not None else None,
                "avg_best": avg_best,
                "fechamento_gap": fechamento_gap,
                "fechamento_gap_norm": fechamento_gap_norm,
                "trave_ratio_global": trave_ratio_global,
                "indice_mpf": indice_mpf,
                "mpf_mapa_gap": mpf_map,
                "rate_4p": float(rate_4p),
                "rate_5p": float(rate_5p),
                "rate_3p": float(rate_3p),
                "zero_hit_rate": float(zero_rate),
            })

            st.markdown("### 🪟 Janela móvel (últimos 60 alvos avaliados)")
            st.json({
                "w_used": int(w_used),
                "any_4p_seen_w": bool(any_4p_w),
                "avg_best_w": avg_best_w,
                "fechamento_gap_w": fechamento_gap_w,
                "fechamento_gap_norm_w": fechamento_gap_norm_w,
                "rate_5p_w": float(rate_5p_w),
                "rate_4p_w": float(rate_4p_w),
                "rate_3p_w": float(rate_3p_w),
                "zero_hit_rate_w": float(zero_rate_w),
            })

            st.markdown("### 📊 Distribuição de best_hit (0..6)")
            st.json(dist)

            # ------------------------------
            # V16_CURV_SUST_DETECTOR — Curvatura Sustentada (gatilho matemático de ataque)
            # ------------------------------
            try:
                curv_info = v16_detector_curvatura_sustentada_df_eval(
                    df_res,
                    w_smooth=5,
                    L_sust=4,
                    eps_mult=0.60,
                    lookback_max=None,
                )
                st.markdown("### 🧮 Curvatura Sustentada (detector matemático)")
                if isinstance(curv_info, dict) and curv_info.get("ok"):
                    st.json({
                        "estado_recente": curv_info.get("estado_recente"),
                        "curvatura_sustentada_recente": curv_info.get("curvatura_sustentada_recente"),
                        "troca_sinal_recente": curv_info.get("troca_sinal_recente"),
                        "dist_desde_ultimo_4": curv_info.get("dist_desde_ultimo_4"),
                        "relogio_geometrico": curv_info.get("relogio_geometrico"),
                    })
                else:
                    st.json(curv_info)
                st.session_state["curvatura_sustentada_info"] = curv_info
            except Exception as _e_curv:
                st.warning(f"Falha no detector de curvatura sustentada: {_e_curv}")

            # guarda para outros painéis (somente leitura)
            st.session_state["replay_stats_prova_janela"] = {
                "targets_avaliados": int(n_targets),
                "any_4p_seen": bool(any_4p),
                "max_best": int(max_best) if max_best is not None else None,
                "rate_4p": float(rate_4p),
                "rate_3p": float(rate_3p),
                "zero_hit_rate": float(zero_rate),
                "w_used": int(w_used),
                "any_4p_seen_w": bool(any_4p_w),
                "rate_4p_w": float(rate_4p_w),
                "rate_3p_w": float(rate_3p_w),
                "zero_hit_rate_w": float(zero_rate_w),
            }

            st.caption(
                "Regra canônica: **'janela' só pode ser afirmada quando any_4p_seen=True** "
                "na base avaliada. Se False, a frase correta é: **'sem evidência de janela na base avaliada'**."
            )
        except Exception as e:
            st.warning(f"Falha ao calcular prova automática: {e}")

    # --- LCE‑B (Jogador B) — Painel silencioso (pré‑C4) ---
    try:
        ss_info_local = st.session_state.get("ss_info") or {}
        ritmo_info_local = st.session_state.get("ritmo_danca_info") or {"ritmo_global": st.session_state.get("ritmo_global_expost", "N/D")}
        snap_map = st.session_state.get("snapshot_p0_canonic") or {}
        snap_last = snap_map.get(int(k_ultimo)) if isinstance(snap_map, dict) and str(k_ultimo).strip().isdigit() else (snap_map.get(k_ultimo) if isinstance(snap_map, dict) else {})
        anti_idx = st.session_state.get("anti_idx_detectados") or []
        lce_b = v16_calc_lce_b(ss_info_local, ritmo_info_local, df_res, snap_last, anti_idx_detectados=anti_idx)
        st.session_state["lce_b"] = lce_b

        with st.expander("🧭 Jogador B — Painel silencioso (LCE‑B · ZEE‑B)", expanded=True):
            st.caption("Pré‑C4 · Observacional · Auditável. Não decide nada. Só mostra STE‑E1 e a sugestão B1 (Base + Anti‑âncora).")
            st.write({
                "STE_E1": lce_b.get("ste_e1"),
                "ZEE_B": lce_b.get("estado_zee_b"),
                "B1_sugerido": lce_b.get("b1_sugerido"),
                "Prontidao_6E": lce_b.get("prontidao_6e"),
            })
            _dbg = lce_b.get("_debug") or {}
            st.markdown(
                f"**🧩 GAMMA PRE‑4 GATE (debug):** `pre4_gate={_dbg.get('pre4_gate')}` · `motivo={_dbg.get('pre4_motivo')}` · `gamma_norm={_dbg.get('gamma_gap_norm')}`"
            )
            # Debug fica visível para auditoria (sem virar decisão)
            if isinstance(lce_b.get("_debug"), dict):
                st.json(lce_b["_debug"])
    except Exception:
        pass

    # --- V9 (BLOCO B) — Resumo agregado (ex-post, observacional) ---
    try:
        cols1 = ["core_hit_1", "quase_hit_1", "borda_in_hit_1", "borda_ex_hit_1", "miolo_hit_1", "fora_hit_1"]
        cols2 = ["core_hit_2", "quase_hit_2", "borda_in_hit_2", "borda_ex_hit_2", "miolo_hit_2", "fora_hit_2"]

        tot = {"core": 0, "quase": 0, "borda_in": 0, "borda_ex": 0, "miolo": 0, "fora": 0}
        for c in cols1:
            if c in df_res.columns:
                tot[c.replace("_hit_1", "").replace("borda_in", "borda_in").replace("borda_ex", "borda_ex")] += int(df_res[c].fillna(0).sum())
        for c in cols2:
            if c in df_res.columns:
                tot[c.replace("_hit_2", "").replace("borda_in", "borda_in").replace("borda_ex", "borda_ex")] += int(df_res[c].fillna(0).sum())

        total_hits = sum(tot.values())
        if total_hits > 0:
            pct = {k: round(100.0 * v / total_hits, 1) for k, v in tot.items()}
        else:
            pct = {k: 0.0 for k in tot.keys()}

        st.markdown("### 🧠 V9 — Memória de Borda (Resumo Agregado · ex‑post)")
        st.caption("Agrega todos os alvos avaliados (C(k+1) e C(k+2) quando existirem). Não decide nada — só descreve onde os acertos nasceram.")
        st.write({
            "hits_total_agregado": int(total_hits),
            "CORE": f'{tot["core"]} ({pct["core"]}%)',
            "quase_CORE": f'{tot["quase"]} ({pct["quase"]}%)',
            "borda_interna": f'{tot["borda_in"]} ({pct["borda_in"]}%)',
            "borda_externa": f'{tot["borda_ex"]} ({pct["borda_ex"]}%)',
            "miolo_do_pacote": f'{tot["miolo"]} ({pct["miolo"]}%)',
            "fora_do_pacote": f'{tot["fora"]} ({pct["fora"]}%)',
        })

        # --- Trave/Proximidade (ex-post): detalha o "fora" em perto vs longe ---
        try:
            fp = 0
            fl = 0
            for c in ["fora_perto_1", "fora_perto_2"]:
                if c in df_res.columns:
                    fp += int(df_res[c].fillna(0).sum())
            for c in ["fora_longe_1", "fora_longe_2"]:
                if c in df_res.columns:
                    fl += int(df_res[c].fillna(0).sum())
            tot_fora = int(tot.get("fora", 0))
            if tot_fora > 0:
                pct_fp = round(100.0 * fp / tot_fora, 1)
                pct_fl = round(100.0 * fl / tot_fora, 1)
            else:
                pct_fp = 0.0
                pct_fl = 0.0
            st.markdown("#### 🎯 Trave/Proximidade (fora do pacote)")
            st.caption("Mesmo sem aumentar acertos, esta leitura mostra se o alvo está ficando **fora, porém perto** (batendo na trave) ou **fora e longe**.")
            st.write({
                "fora_total": tot_fora,
                "fora_perto": f"{fp} ({pct_fp}%)",
                "fora_longe": f"{fl} ({pct_fl}%)",
            })
        except Exception:
            pass

        # --- Persistência em sessão (V9 como lastro informativo) ---
        try:
            _resumo = {
                "total_hits": int(total_hits),
                "tot": {k: int(v) for k, v in tot.items()},
                "pct": {k: float(pct.get(k, 0.0)) for k in tot.keys()},
            }
            _classif = v9_classificar_memoria_borda(df_res=df_res, total_hits=int(total_hits), pct=_resumo["pct"])
            st.session_state["v9_memoria_borda"] = {
                "resumo": _resumo,
                "classificacao": _classif,
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
            }
        except Exception:
            pass

    except Exception:
        pass

    st.info(
        "Interpretação correta:\n"
        "- Este painel mede apenas o **melhor acerto** dentro do pacote para os 2 alvos seguintes.\n"
        "- Ele NÃO muda o motor, NÃO decide volume, e NÃO garante performance futura.\n"
        "- Serve para você comparar janelas e ver se o V8 está reduzindo perda por borda sem dispersar."
    )

    st.stop()


# ============================================================
# PARTE 3/8 — FIM
# ============================================================
# ============================================================
# PARTE 4/8 — INÍCIO
# ============================================================

# ============================================================
# Painel 6 — ⚙️ Modo TURBO++ HÍBRIDO
# ============================================================

# ============================================================
# P1 — 🧪 Ajuste de Pacote (pré-C4) — Comparativo (A/B)
# Base: Snapshot P0 Canônico (registrado no Replay Progressivo)
# Regras:
# - EX-POST (apenas análise do que já aconteceu)
# - PRÉ-C4 (não muda listas / não decide volume / não toca Camada 4)
# - Sem alvo "dirigido": regras A/B dependem apenas do snapshot (P0), não do alvo ex-post
# ============================================================
if painel == "🧪 P1 — Ajuste de Pacote (pré-C4) — Comparativo":

    st.markdown("## 🧪 P1 — Ajuste de Pacote (pré-C4) — Comparativo (A/B)")
    st.caption("Baseado em **Snapshot P0 Canônico** (registrado no Replay Progressivo). Leitura ex-post; não altera Camada 4.")

    df = st.session_state.get("historico_df")
    if df is None or len(df) < 5:
        exibir_bloco_mensagem(
            "Histórico ausente",
            "Execute primeiro **📁 Carregar Histórico**.",
            tipo="warning",
        )
        st.stop()

    # Anti-zumbi leve (painel analítico)
    qtd_series = len(df)
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="P1 (Ajuste de Pacote pré-C4)",
        painel="🧪 P1 — Ajuste de Pacote (pré-C4) — Comparativo",
    ):
        st.stop()

    snapshots = st.session_state.get("snapshot_p0_canonic") or {}
    if not snapshots:
        exibir_bloco_mensagem(
            "Nenhum Snapshot P0 registrado",
            "Vá em **🧭 Replay Progressivo — Janela Móvel (Assistido)** e clique em **📌 Registrar pacote da janela atual**.\n\n"
            "Depois volte aqui para rodar o P1 (comparativo).",
            tipo="info",
        )
        st.stop()

    # ----------------------------
    # Helpers (P1) — regras A/B
    # ----------------------------
    def _p1__clamp(v: int, umin: int, umax: int) -> int:
        return max(int(umin), min(int(umax), int(v)))

    def _p1__neighbors(base_vals: List[int], umin: int, umax: int, deltas: List[int]) -> List[int]:
        out = []
        for x in base_vals:
            try:
                xi = int(x)
            except Exception:
                continue
            for d in deltas:
                out.append(_p1__clamp(xi + int(d), umin, umax))
        return out

    def _p1__pick_novel(candidates: List[int], universo_base: set, limit_n: int) -> List[int]:
        # Mantém ordem estável (por frequência implícita: candidatos já vêm ordenados)
        out = []
        seen = set()
        for x in candidates:
            xi = int(x)
            if xi in universo_base:
                continue
            if xi in seen:
                continue
            seen.add(xi)
            out.append(xi)
            if len(out) >= int(limit_n):
                break
        return out

    def _p1__build_AB(snapshot: Dict[str, Any], umin: int, umax: int) -> Dict[str, Any]:
        u0_list = snapshot.get("universo_pacote") or []
        u0 = set(int(x) for x in u0_list if str(x).strip() != "")
        u0 = set(_p1__clamp(x, umin, umax) for x in u0)

        snap_v8 = snapshot.get("snap_v8") or {}
        core = [int(x) for x in (snap_v8.get("core") or [])]
        quase = [int(x) for x in (snap_v8.get("quase_core") or [])]
        borda_interna = [int(x) for x in (snap_v8.get("borda_interna") or [])]

        # Frequências (ordenadas) — já vem como dict ordenado no snapshot
        freq = snapshot.get("freq_passageiros") or {}
        freq_items = []
        try:
            for k_str, v in freq.items():
                try:
                    freq_items.append((int(k_str), int(v)))
                except Exception:
                    continue
            freq_items.sort(key=lambda kv: (-kv[1], kv[0]))
        except Exception:
            freq_items = []

        top_freq = [k for k, _ in freq_items[:10]]

        # ------------------------
        # Regra A (P1.A) — "borda interna mais ativa"
        # - Regra ex-ante: só usa P0 (core/quase/borda_interna)
        # - Intenção: capturar parte do "fora_perto" com adição mínima e interna
        # ------------------------
        base_A = core + quase + borda_interna
        cand_A = _p1__neighbors(base_A, umin, umax, deltas=[-1, +1])
        add_A = _p1__pick_novel(cand_A, u0, limit_n=6)
        uA = set(u0) | set(add_A)

        # ------------------------
        # Regra B (P1.B) — "deslocamento levíssimo de centro"
        # - Regra ex-ante: só usa P0 (top freq)
        # - Intenção: atacar parte do "fora_longe" sem explosão de universo
        # ------------------------
        cand_B = _p1__neighbors(top_freq, umin, umax, deltas=[-1, +1])
        add_B = _p1__pick_novel(cand_B, u0, limit_n=8)
        uB = set(u0) | set(add_B)

        return {
            "U0": sorted(u0),
            "UA": sorted(uA),
            "UB": sorted(uB),
            "add_A": add_A,
            "add_B": add_B,
            "meta": {
                "u0_len": len(u0),
                "ua_len": len(uA),
                "ub_len": len(uB),
            }
        }

    def _p1__eval_next2(df_: pd.DataFrame, k: int, universo: List[int]) -> Dict[str, Any]:
        """Avalia fora_total / fora_perto / fora_longe nos 2 alvos seguintes (k+1, k+2), se existirem.

        ⚠️ Importante (governança):
        - Isto é leitura ex-post.
        - Não altera listas / não altera Camada 4.
        - Usa preferencialmente o FULL para enxergar (k+1, k+2) mesmo quando o ATIVO foi recortado.
        """
        cols_pass = [c for c in df_.columns if c.startswith("p")]
        u_set = set(int(x) for x in universo)
        u_sorted = sorted(u_set)

        def _min_dist(x: int) -> int:
            if not u_sorted:
                return 9999
            return min(abs(int(x) - int(u)) for u in u_sorted)

        def _get_row_by_serie(df__ : pd.DataFrame, serie_id: int):
            """Tenta achar a linha do alvo de forma robusta.
            Prioridades:
            1) index contém a série (loc)
            2) iloc (assumindo série 1-based) -> iloc[serie_id-1]
            3) iloc (assumindo série 0-based) -> iloc[serie_id]
            """
            try:
                if serie_id in df__.index:
                    return df__.loc[serie_id]
            except Exception:
                pass

            try:
                pos = int(serie_id) - 1
                if 0 <= pos < len(df__):
                    return df__.iloc[pos]
            except Exception:
                pass

            try:
                pos = int(serie_id)
                if 0 <= pos < len(df__):
                    return df__.iloc[pos]
            except Exception:
                pass

            return None

        out = {
            "alvos": [],
            "fora_total": 0,
            "fora_perto": 0,
            "fora_longe": 0,
            "detalhe_fora": [],  # lista de (serie_id, x, perto?)
        }

        for dk in (1, 2):
            serie_alvo = int(k) + int(dk)
            row = _get_row_by_serie(df_, serie_alvo)
            if row is None:
                continue

            try:
                alvo = [int(row[c]) for c in cols_pass]
            except Exception:
                alvo = []

            fora = [x for x in alvo if int(x) not in u_set]
            out["alvos"].append({"k": int(serie_alvo), "alvo": alvo, "fora": fora})

            for x in fora:
                out["fora_total"] += 1
                dist = _min_dist(int(x))
                if dist <= 1:
                    out["fora_perto"] += 1
                    out["detalhe_fora"].append((int(serie_alvo), int(x), True))
                else:
                    out["fora_longe"] += 1
                    out["detalhe_fora"].append((int(serie_alvo), int(x), False))

        return out

    # ----------------------------
    # UI
    # ----------------------------
    ks = sorted([int(k) for k in snapshots.keys()])
    k_sel = st.selectbox("Escolha a janela registrada (k)", ks, index=len(ks) - 1)

    snap = snapshots.get(int(k_sel)) or snapshots.get(str(k_sel)) or {}
    st.markdown("### 🧊 Snapshot P0 selecionado (visão rápida)")
    try:
        st.write({
            "k": snap.get("k"),
            "ts": snap.get("ts"),
            "qtd_listas": snap.get("qtd_listas"),
            "assinatura": snap.get("assinatura"),
            "universo_pacote_len": len(snap.get("universo_pacote") or []),
            "core_sz": len((snap.get("snap_v8") or {}).get("core") or []),
            "quase_sz": len((snap.get("snap_v8") or {}).get("quase_core") or []),
            "borda_interna_sz": len((snap.get("snap_v8") or {}).get("borda_interna") or []),
        })
    except Exception:
        pass

    umin = int(st.session_state.get("universo_min") or 1)
    umax = int(st.session_state.get("universo_max") or 60)

    ab = _p1__build_AB(snap, umin=umin, umax=umax)

    st.markdown("### 🧠 Regras P1 (A/B) — o que muda no universo (pré-C4)")
    colA, colB, col0 = st.columns([1, 1, 1])
    with col0:
        st.info(f"U0 (base) — len={ab['meta']['u0_len']}")
        st.write(ab["U0"])
    with colA:
        st.success(f"P1.A — len={ab['meta']['ua_len']} (adds={len(ab['add_A'])})")
        st.write({"adds_A": ab["add_A"]})
        st.write(ab["UA"])
    with colB:
        st.success(f"P1.B — len={ab['meta']['ub_len']} (adds={len(ab['add_B'])})")
        st.write({"adds_B": ab["add_B"]})
        st.write(ab["UB"])

    
    # Preferir FULL para enxergar (k+1, k+2) quando o ATIVO foi recortado no Replay Progressivo
    _df_full_safe = st.session_state.get("historico_df_full")
    df_eval = _df_full_safe if (_df_full_safe is not None and not getattr(_df_full_safe, "empty", False)) else df
    st.markdown("### 📊 Avaliação ex-post (2 alvos seguintes): fora_total / fora_perto / fora_longe")
    ev0 = _p1__eval_next2(df_eval, int(k_sel), ab["U0"])
    evA = _p1__eval_next2(df_eval, int(k_sel), ab["UA"])
    evB = _p1__eval_next2(df_eval, int(k_sel), ab["UB"])

    # Tabela simples (sem pandas para manter leve)
    st.write({
        "U0": {"fora_total": ev0["fora_total"], "fora_perto": ev0["fora_perto"], "fora_longe": ev0["fora_longe"]},
        "P1.A": {"fora_total": evA["fora_total"], "fora_perto": evA["fora_perto"], "fora_longe": evA["fora_longe"]},
        "P1.B": {"fora_total": evB["fora_total"], "fora_perto": evB["fora_perto"], "fora_longe": evB["fora_longe"]},
    })

    with st.expander("🔎 Detalhe dos alvos e 'foras' (U0 / P1.A / P1.B)"):
        st.markdown("#### U0 — alvos")
        st.write(ev0["alvos"])
        st.markdown("#### P1.A — alvos")
        st.write(evA["alvos"])
        st.markdown("#### P1.B — alvos")
        st.write(evB["alvos"])

    st.info(
        "Interpretação correta:\n"
        "- P1 é **comparativo ex-post**: mede como o 'universo do pacote' (P0) teria coberto os 2 alvos seguintes.\n"
        "- P1.A e P1.B aplicam **regras ex-ante** (dependem apenas do snapshot), para evitar viés.\n"
        "- Este painel **não altera listas reais**, **não decide volume** e **não toca Camada 4**.\n"
        "- Use para comparar janelas e verificar se existe caminho para reduzir **fora_longe** sem explodir universo."
    )

    st.stop()


if painel == "⚙️ Modo TURBO++ HÍBRIDO":

    st.markdown("## ⚙️ Modo TURBO++ HÍBRIDO — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute o painel **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # Anti-zumbi leve
    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_PREVISOES_TURBO,
        contexto="TURBO++ HÍBRIDO",
        painel="⚙️ Modo TURBO++ HÍBRIDO",
    ):
        st.stop()

    st.info("Executando Modo TURBO++ HÍBRIDO...")

    # ============================================================
    # MOTOR HÍBRIDO — DX Light + S6 Light + Monte Carlo Light
    # ============================================================
    try:
        # DX Light — proximidade final
        vetor_final = matriz_norm[-1]
        distancias = [
            np.linalg.norm(vetor_final - linha) for linha in matriz_norm[:-1]
        ]

        # S6 Light — estatística simples dos passageiros
        col_pass = [c for c in df.columns if c.startswith("p")]
        ult = df[col_pass].iloc[-1].values

        s6_scores = []
        for idx in range(len(df) - 1):
            candidato = df[col_pass].iloc[idx].values
            intersec = len(set(candidato) & set(ult))
            s6_scores.append(intersec)

        # Monte Carlo Light — sorteio ponderado
        pesos_mc = np.array([1 / (1 + d) for d in distancias])
        soma_pesos = float(pesos_mc.sum()) if len(pesos_mc) > 0 else 0.0
        if soma_pesos <= 0.0 or np.isnan(soma_pesos):
            # fallback: distribuição uniforme (base insuficiente para ponderar)
            pesos_mc = np.ones(len(distancias), dtype=float)
            soma_pesos = float(pesos_mc.sum())
        pesos_mc = pesos_mc / soma_pesos

        escolha_idx = np.random.choice(len(pesos_mc), p=pesos_mc)
        previsao_mc = df[col_pass].iloc[escolha_idx].values.tolist()

        # Consolidação leve
        s6_melhor = df[col_pass].iloc[np.argmax(s6_scores)].values.tolist()
        dx_melhor = df[col_pass].iloc[np.argmin(distancias)].values.tolist()

        # Combinação híbrida
        previsao_final = list(
            np.round(
                0.4 * np.array(dx_melhor)
                + 0.3 * np.array(s6_melhor)
                + 0.3 * np.array(previsao_mc)
            )
        )
        previsao_final = [int(x) for x in previsao_final]

    except Exception as erro:
        exibir_bloco_mensagem(
            "Erro no TURBO++ HÍBRIDO",
            f"Detalhes: {erro}",
            tipo="error",
        )
        st.stop()

    # ============================================================
    # Exibição final
    # ============================================================
    st.markdown("### 🔮 Previsão HÍBRIDA (TURBO++)")
    st.success(f"**{formatar_lista_passageiros(previsao_final)}**")

    st.session_state["ultima_previsao"] = previsao_final

# ============================================================
# BLOCO 1/4 — ORQUESTRADOR DE TENTATIVA (V16) — INVISÍVEL
# Objetivo: traduzir diagnóstico (alvo/risco/confiabilidade) em
# "configuração de tentativa" para o Modo 6 (sem decidir listas).
# LISTAS SEMPRE EXISTEM: este orquestrador NUNCA retorna volume 0.
# ============================================================

from typing import Dict, Any, Optional
st.session_state["ct_force_marker"] = True


# ------------------------------------------------------------
# HELPERS (V16) — clamp + safe float
# ------------------------------------------------------------

def _clamp_v16(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float_v16(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ------------------------------------------------------------
# ORQUESTRADOR DE TENTATIVA (V16) — núcleo conceitual
# ------------------------------------------------------------

def orquestrar_tentativa_v16(
    *,
    series_count: int,
    alvo_tipo: Optional[str] = None,          # "parado" | "movimento_lento" | "movimento_rapido"
    alvo_velocidade: Optional[float] = None,  # ex: 0.9319 (se disponível)
    k_star: Optional[float] = None,           # ex: 0.2083
    nr_pct: Optional[float] = None,           # ex: 67.87  (0..100)
    divergencia_s6_mc: Optional[float] = None,# ex: 14.0480
    risco_composto: Optional[float] = None,   # ex: 0.7560  (0..1)
    confiabilidade_estimada: Optional[float] = None,  # 0..1 (se você já tiver)
    # --- Limites técnicos (anti-zumbi) ---
    limite_seguro_series_modo6: int = 800,    # padrão atual (já visto no app)
    # --- Volumes base (pode ser ajustado depois, mas COMEÇA CONSERVADOR) ---
    volume_min_base: int = 3,
    volume_rec_base: int = 6,
    volume_max_base: int = 80,
) -> Dict[str, Any]:
    """
    Retorna um dicionário com a "configuração de tentativa" (invisível),
    para o Modo 6 usar como guia de volume e forma (diversidade/variação).

    ✅ Regras implementadas aqui:
    - Objetivo único: tentar cravar 6 passageiros (não decide, só orienta).
    - Listas SEMPRE existem -> volume_min >= 1 (nunca 0).
    - Confiabilidade alta => explorar (mandar bala com critério).
    - Confiabilidade baixa => tentar com critério (degradado, mas não zero).
    - Anti-zumbi não censura: limita teto, mas não zera.
    """

    # -----------------------------
    # Sanitização básica
    # -----------------------------
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    k_star = _safe_float_v16(k_star, 0.0)
    nr_pct = _safe_float_v16(nr_pct, 0.0)
    divergencia_s6_mc = _safe_float_v16(divergencia_s6_mc, 0.0)
    risco_composto = _safe_float_v16(risco_composto, 0.0)

    # Normalizações defensivas
    nr_norm = _clamp_v16(nr_pct / 100.0, 0.0, 1.0)             # 0..1
    risco_norm = _clamp_v16(risco_composto, 0.0, 1.0)          # 0..1
    k_norm = _clamp_v16(k_star / 0.35, 0.0, 1.0)               # 0..1 (0.35 ~ teto típico de alerta)
    div_norm = _clamp_v16(divergencia_s6_mc / 15.0, 0.0, 1.0)  # 0..1 (15 ~ divergência crítica)

    # -----------------------------
    # Inferência do tipo de alvo (se não vier do Laudo)
    # -----------------------------
    alvo_tipo_norm = (alvo_tipo or "").strip().lower()

    if not alvo_tipo_norm:
        v = _safe_float_v16(alvo_velocidade, 0.0)
        # Heurística simples (pode refinar depois):
        # - <0.35: parado/lento
        # - 0.35..0.70: movimento_lento
        # - >0.70: movimento_rapido
        if v <= 0.35:
            alvo_tipo_norm = "parado"
        elif v <= 0.70:
            alvo_tipo_norm = "movimento_lento"
        else:
            alvo_tipo_norm = "movimento_rapido"

    if alvo_tipo_norm in ("lento", "movimento lento", "movimento-lento"):
        alvo_tipo_norm = "movimento_lento"
    if alvo_tipo_norm in ("rapido", "rápido", "movimento rapido", "movimento-rápido", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"
    if alvo_tipo_norm in ("parado", "estavel", "estável"):
        alvo_tipo_norm = "parado"

    if alvo_tipo_norm not in ("parado", "movimento_lento", "movimento_rapido"):
        alvo_tipo_norm = "movimento_rapido"  # default seguro: tratar como difícil

    # -----------------------------
    # Construção de uma "confiabilidade estimada" interna (se não vier)
    # -----------------------------
    # Ideia: confiabilidade cai com ruído, risco, k* alto e divergência alta.
    # (Não é promessa, é régua de orientação de intensidade.)
    if confiabilidade_estimada is None:
        penal = 0.40 * nr_norm + 0.25 * risco_norm + 0.20 * div_norm + 0.22 * k_norm
        conf = 1.0 - _clamp_v16(penal, 0.0, 1.0)
    else:
        conf = _clamp_v16(_safe_float_v16(confiabilidade_estimada, 0.0), 0.0, 1.0)

    # -----------------------------
    # Definição do "modo de tentativa" (conceito → controle interno)
    # -----------------------------
    # - exploração_intensa: alta confiança (mandar bala com critério)
    # - tentativa_controlada: meio termo
    # - tentativa_degradada: baixa confiança / alvo rápido / ambiente hostil
    if conf >= 0.55 and risco_norm <= 0.55 and nr_norm <= 0.55 and div_norm <= 0.60:
        modo = "exploracao_intensa"
    elif conf >= 0.30 and risco_norm <= 0.75 and nr_norm <= 0.75:
        modo = "tentativa_controlada"
    else:
        modo = "tentativa_degradada"

    # Alvo rápido puxa para degradado, a menos que seja realmente "bom"
    if alvo_tipo_norm == "movimento_rapido" and modo != "exploracao_intensa":
        modo = "tentativa_degradada"

    # -----------------------------
    # Volumes base (sempre > 0)
    # -----------------------------
    vol_min = max(1, int(volume_min_base))
    vol_rec = max(vol_min, int(volume_rec_base))
    vol_max = max(vol_rec, int(volume_max_base))

    # -----------------------------
    # Ajuste de intensidade por modo + confiabilidade
    # -----------------------------
    # Observação: "mandar bala" = aumentar volume e variação interna,
    # mas SEM explodir sem critério.
    if modo == "exploracao_intensa":
        # Escala com conf (0.55..1.0) -> multiplicador (1.1..1.9)
        mult = 1.1 + 0.8 * _clamp_v16((conf - 0.55) / 0.45, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        diversidade = 0.55  # moderada (refino + variação)
        variacao_interna = 0.75
        aviso_curto = "🟢 Exploração intensa: mandar bala com critério (janela favorável)."

    elif modo == "tentativa_controlada":
        # Escala suave com conf (0.30..0.55) -> multiplicador (1.05..1.20)
        mult = 1.05 + 0.25 * _clamp_v16((conf - 0.30) / 0.25, 0.0, 1.0)
        vol_rec = int(max(vol_rec, round(vol_rec * mult)))
        vol_max = int(max(vol_max, round(vol_max * mult)))

        # diversidade depende do alvo
        if alvo_tipo_norm == "parado":
            diversidade = 0.35  # mais próximo (ajuste fino)
            variacao_interna = 0.60
        elif alvo_tipo_norm == "movimento_lento":
            diversidade = 0.50  # cercamento
            variacao_interna = 0.55
        else:
            diversidade = 0.65  # já puxa para hipóteses
            variacao_interna = 0.45

        aviso_curto = "🟡 Tentativa controlada: cercar com critério (sem exagero)."

    else:
        # Degradado: volume controlado, diversidade alta (hipóteses)
        # Garante mínimo, limita teto e aumenta diversidade.
        # Se conf for muito baixa, não adianta inflar volume: mantém enxuto.
        if conf <= 0.10:
            vol_rec = max(vol_min, min(vol_rec, 6))
            vol_max = max(vol_rec, min(vol_max, 12))
        elif conf <= 0.20:
            vol_rec = max(vol_min, min(vol_rec, 8))
            vol_max = max(vol_rec, min(vol_max, 18))
        else:
            vol_rec = max(vol_min, min(vol_rec, 10))
            vol_max = max(vol_rec, min(vol_max, 24))

        diversidade = 0.85  # alto (ali, lá, acolá)
        variacao_interna = 0.35
        aviso_curto = "🔴 Tentativa degradada: hipóteses espalhadas (chance baixa, mas listas existem)."

    # -----------------------------
    # Anti-zumbi como LIMITADOR (não censura)
    # -----------------------------
    # Se o histórico excede o limite seguro do modo 6:
    # - não bloqueia
    # - apenas derruba o teto e puxa recomendado para um patamar seguro
    # Mantém volume_min > 0 SEMPRE.
    if series_count > int(limite_seguro_series_modo6):
        # Fator de penalização pelo excesso de séries (piora custo)
        excesso = series_count - int(limite_seguro_series_modo6)
        fator = _clamp_v16(1.0 - (excesso / max(1.0, float(limite_seguro_series_modo6))) * 0.60, 0.25, 1.0)

        teto_seguro = int(max(vol_rec, round(vol_max * fator)))
        teto_seguro = int(_clamp_v16(teto_seguro, max(vol_rec, vol_min), vol_max))

        # puxa recomendado junto do teto seguro (mas nunca abaixo do mínimo)
        vol_max = max(vol_rec, teto_seguro)
        vol_rec = max(vol_min, min(vol_rec, vol_max))

        aviso_curto += " 🔒 Anti-Zumbi: volume limitado (sem bloquear geração)."

    # -----------------------------
    # Garantias finais (invioláveis)
    # -----------------------------
    vol_min = max(1, int(vol_min))
    vol_rec = max(vol_min, int(vol_rec))
    vol_max = max(vol_rec, int(vol_max))

    diversidade = _clamp_v16(diversidade, 0.10, 1.05)
    variacao_interna = _clamp_v16(variacao_interna, 0.10, 1.05)

    return {
        "modo_tentativa": modo,
        "alvo_tipo": alvo_tipo_norm,
        "confiabilidade_estimada": float(conf),
        "volume_min": int(vol_min),
        "volume_recomendado": int(vol_rec),
        "volume_max": int(vol_max),
        "diversidade": float(diversidade),
        "variacao_interna": float(variacao_interna),
        "aviso_curto": str(aviso_curto),
        "debug": {
            "nr_norm": float(nr_norm),
            "risco_norm": float(risco_norm),
            "k_norm": float(k_norm),
            "div_norm": float(div_norm),
            "series_count": int(series_count),
            "limite_seguro_series_modo6": int(limite_seguro_series_modo6),
        },
    }

# ============================================================
# BLOCO 2/4 — PONTE ORQUESTRADOR → TURBO++ ULTRA (V16)
# Objetivo: coletar diagnósticos existentes do app (Laudo/Risco)
# e preparar a configuração de tentativa para o Modo 6,
# SEM alterar UI e SEM decidir listas.
# ============================================================

def preparar_tentativa_turbo_ultra_v16(
    *,
    df,
    series_count: int,
    alvo_tipo: Optional[str] = None,
    alvo_velocidade: Optional[float] = None,
    k_star: Optional[float] = None,
    nr_pct: Optional[float] = None,
    divergencia_s6_mc: Optional[float] = None,
    risco_composto: Optional[float] = None,
    confiabilidade_estimada: Optional[float] = None,
    limite_seguro_series_modo6: int = 800,
) -> Dict[str, Any]:
    """
    Ponte invisível:
    - lê informações já calculadas no app
    - chama o Orquestrador de Tentativa (BLOCO 1)
    - devolve um dicionário pronto para o TURBO++ ULTRA usar

    NÃO gera listas
    NÃO executa motores
    NÃO decide nada
    """

    # Defesa básica
    try:
        series_count = int(series_count)
    except Exception:
        series_count = 0

    # Chamada central ao Orquestrador
    cfg = orquestrar_tentativa_v16(
        series_count=series_count,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        confiabilidade_estimada=confiabilidade_estimada,
        limite_seguro_series_modo6=limite_seguro_series_modo6,
    )

    # Normalização final (garantia extra)
    cfg["volume_min"] = max(1, int(cfg.get("volume_min", 1)))
    cfg["volume_recomendado"] = max(
        cfg["volume_min"],
        int(cfg.get("volume_recomendado", cfg["volume_min"]))
    )
    cfg["volume_max"] = max(
        cfg["volume_recomendado"],
        int(cfg.get("volume_max", cfg["volume_recomendado"]))
    )

    return cfg

# ============================================================
# >>> INÍCIO — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# Camada invisível de conexão (não é painel, não gera listas)
# ============================================================

def _injetar_cfg_tentativa_turbo_ultra_v16(
    *,
    df,
    qtd_series: int,
    k_star,
    limite_series_padrao: int,
):
    """
    Injeta no session_state a configuração de tentativa calculada
    pelo Orquestrador (BLOCO 1 + BLOCO 2), sem bloquear execução.
    """

    # Coleta informações já existentes
    laudo_v16 = st.session_state.get("laudo_operacional_v16", {}) or {}

    alvo_tipo = laudo_v16.get("estado_alvo") or laudo_v16.get("alvo_tipo")
    alvo_velocidade = laudo_v16.get("velocidade_estimada")

    nr_pct = st.session_state.get("nr_pct")
    divergencia_s6_mc = st.session_state.get("divergencia_s6_mc")
    risco_composto = st.session_state.get("indice_risco")

    cfg = preparar_tentativa_turbo_ultra_v16(
        df=df,
        series_count=qtd_series,
        alvo_tipo=alvo_tipo,
        alvo_velocidade=alvo_velocidade,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        limite_seguro_series_modo6=limite_series_padrao,
    )

    # Guarda para uso posterior
    st.session_state["cfg_tentativa_turbo_ultra"] = cfg

    # Aviso curto (informativo, não bloqueante)
    aviso = cfg.get("aviso_curto")
    if aviso:
        st.caption(aviso)

    # Define limite efetivo (anti-zumbi vira limitador, não censura)
    limite_efetivo = min(
        limite_series_padrao,
        int(cfg.get("volume_max", limite_series_padrao))
    )

    return limite_efetivo


# ============================================================
# <<< FIM — BLOCO 3/4 — ORQUESTRADOR → TURBO++ ULTRA (V16)
# ============================================================

# ============================================================
# >>> PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3 — VOLUME POR ORÇAMENTO)
# ============================================================

if painel == "⚙️ Modo TURBO++ ULTRA":

    st.markdown("## ⚙️ Modo TURBO++ ULTRA — MVP3")
    st.caption(
        "Exploração controlada.\n\n"
        "✔ Motor original preservado\n"
        "✔ Anti-zumbi respeitado\n"
        "✔ Volume liberado por orçamento\n"
        "✔ Falha silenciosa permitida\n"
        "✔ Sem decisão automática"
    )

    # ------------------------------------------------------------
    # BUSCA DE ESTADO (SOMENTE AQUI)
    # ------------------------------------------------------------
    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    _kstar_raw = st.session_state.get("sentinela_kstar")
    k_star = float(_kstar_raw) if isinstance(_kstar_raw, (int, float)) else 0.0

    if df is None or df.empty or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Carregue o histórico e execute **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    # ------------------------------------------------------------
    # 🔒 MARCAÇÃO OFICIAL — TURBO EXECUTADO (ANTES DO ANTI-ZUMBI)
    # (Se o Anti-Zumbi bloquear com st.stop, o Modo 6 NÃO fica travado)
    # ------------------------------------------------------------
    st.session_state["turbo_ultra_executado"] = True
    st.session_state["turbo_executado"] = True
    st.session_state["turbo_ultra_rodou"] = True
    st.session_state["motor_turbo_executado"] = True

    # ------------------------------------------------------------
    # ANTI-ZUMBI — LIMITADOR OFICIAL
    # (COMPORTAMENTO OFICIAL PRESERVADO)
    # ------------------------------------------------------------
    LIMITE_SERIES_TURBO_ULTRA_EFETIVO = _injetar_cfg_tentativa_turbo_ultra_v16(
        df=df,
        qtd_series=qtd_series,
        k_star=k_star,
        limite_series_padrao=LIMITE_SERIES_TURBO_ULTRA,
    )

    limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_TURBO_ULTRA_EFETIVO,
        contexto="TURBO++ ULTRA",
        painel="⚙️ Modo TURBO++ ULTRA",
    )
    # ⬆️ se bloquear, a própria função já dá st.stop()
    # ✅ e a marcação acima já foi gravada, então o Modo 6 destrava.

    # ------------------------------------------------------------
    # ORÇAMENTO
    # ------------------------------------------------------------
    orcamentos_disponiveis = [6, 42, 168, 504, 1260, 2772]

    orcamento = st.selectbox(
        "Selecione o orçamento para o TURBO++ ULTRA:",
        options=orcamentos_disponiveis,
        index=1,
    )

    mapa_execucoes = {
        6: 1,
        42: 1,
        168: 3,
        504: 6,
        1260: 10,
        2772: 20,
    }

    n_exec = mapa_execucoes.get(int(orcamento), 1)

    st.info(
        f"🔢 Orçamento selecionado: **{orcamento}**\n\n"
        f"▶️ Execuções do TURBO++ ULTRA: **{n_exec}**"
    )

    # ------------------------------------------------------------
    # JANELA LOCAL DE ATAQUE (GATILHO CANÔNICO)
    # ------------------------------------------------------------
    # Usa APENAS sinais já existentes em session_state.
    # Não decide postura. Só governa se o TURBO ofensivo pode tentar nesta rodada.
    m3_reg = st.session_state.get("m3_regime_dx") or st.session_state.get("m3_regime")
    classe_risco = st.session_state.get("classe_risco")
    if classe_risco is None:
        classe_risco = st.session_state.get("classe_risco_texto")
    nr_percent = st.session_state.get("nr_percent")
    div_s6_mc = st.session_state.get("divergencia_s6_mc")
    compressao_core = st.session_state.get("janela_compressao_core", False)

    # Critério mínimo (canônico): compressão + M3 em ECO/PRÉ-ECO + risco não hostil.
    m3_ok = str(m3_reg).upper() in ["ECO", "PRÉ-ECO", "PRE", "PRE-ECO", "PRE ECO", "PRÉ ECO"]
        # Governança (k-isolation): NÃO usar classe_risco (derivada de k*) para gates operacionais.
    # Em vez disso, usa a postura do MOTOR (já isolada de k/k*).
    pst_motor = str(st.session_state.get("postura_estado") or st.session_state.get("postura_motor") or "").strip().upper()
    risco_ok = (pst_motor != "RUPTURA")
    janela_ativa = bool(compressao_core and m3_ok and risco_ok)

    st.session_state["janela_local_ativa"] = janela_ativa
    st.session_state["janela_local_m3"] = m3_reg if m3_reg is not None else "N/D"
    st.session_state["janela_local_classe_risco"] = classe_risco if classe_risco is not None else "N/D"

    if not janela_ativa:
        st.info("🧨 Janela Local de Ataque: **NÃO ATIVA** — TURBO ofensivo não tentado nesta rodada (governança).")
        st.session_state["turbo_ultra_executado"] = False
        st.session_state["turbo_ultra_listas_leves"] = []
        st.session_state["turbo_ultra_listas"] = []
        # ainda marcamos como "tentado" no sentido de que o painel foi visitado e governou a tentativa
        st.session_state["turbo_ultra_tentado"] = True
        # encerra este painel aqui, sem gerar listas
        st.stop()

    # ------------------------------------------------------------
    # EXECUÇÃO SEGURA DO TURBO++ ULTRA
    # ------------------------------------------------------------
    st.info("Executando Modo TURBO++ ULTRA...")

    todas_listas = []

    for _ in range(n_exec):
        try:
            lista = turbo_ultra_v15_7(
                df=df,
                matriz_norm=matriz_norm,
                k_star=k_star,
            )
            if isinstance(lista, list) and len(lista) >= 6:
                todas_listas.append(lista)
        except Exception:
            pass

    # ------------------------------------------------------------
    # FECHAMENTO DE ESTADO DO PIPELINE
    # ------------------------------------------------------------
    st.session_state["pipeline_flex_ultra_concluido"] = True
    st.session_state["turbo_ultra_listas_leves"] = todas_listas.copy()
    st.session_state["ultima_previsao"] = todas_listas
    v16_blindar_ultima_previsao_universo()

    # Blindagem adicional — estado intermediário reutilizável
    st.session_state["turbo_ultra_listas_leves"] = st.session_state["ultima_previsao"]

    if not todas_listas:
        st.warning(
            "Nenhuma lista foi gerada nesta condição.\n\n"
            "Isso é um **resultado válido**.\n"
            "O motor foi executado (ou bloqueado) e falhou silenciosamente."
        )
        st.stop()

    st.success(
        f"✅ TURBO++ ULTRA executado com sucesso.\n\n"
        f"📦 Listas geradas: **{len(todas_listas)}**"
    )

    st.markdown("### 🔮 Listas geradas (amostra)")
    st.write(todas_listas[: min(5, len(todas_listas))])

# ============================================================
# <<< FIM — PAINEL 7 — ⚙️ Modo TURBO++ ULTRA (MVP3)
# ============================================================





# ============================================================
# MOTORES PROFUNDOS (PUROS)
# NÃO executam sozinhos
# NÃO acessam session_state
# NÃO exibem nada
# ============================================================

# --- S6 PROFUNDO ---
def s6_profundo_V157(df_local, col_pass, idx_alvo):
    ult_local = df_local[col_pass].iloc[idx_alvo].values
    scores_local = []
    for i_local in range(len(df_local) - 1):
        base_local = df_local[col_pass].iloc[i_local].values
        inter_local = len(set(base_local) & set(ult_local))
        scores_local.append(inter_local)
    melhores_idx_local = np.argsort(scores_local)[-25:]
    candidatos_local = df_local[col_pass].iloc[melhores_idx_local].values
    return candidatos_local


# --- MICRO-LEQUE PROFUNDO ---
def micro_leque_profundo(base, profundidade=20, universo_min=1, universo_max=60):
    leque = []
    umin = int(universo_min) if universo_min is not None else 1
    umax = int(universo_max) if universo_max is not None else 60

    for delta in range(-profundidade, profundidade + 1):
        novo = [max(umin, min(umax, int(x) + delta)) for x in base]
        leque.append(novo)

    return np.array(leque)


# --- MONTE CARLO PROFUNDO ---
def monte_carlo_profundo(base, n=800, universo_min=1, universo_max=60):
    sims = []
    umin = int(universo_min) if universo_min is not None else 1
    umax = int(universo_max) if universo_max is not None else 60

    base_arr = np.array([int(x) for x in base], dtype=int)

    for _ in range(n):
        ruido = np.random.randint(-5, 6, size=len(base_arr))
        candidato = base_arr + ruido
        candidato = np.clip(candidato, umin, umax)
        sims.append(candidato.tolist())

    return sims


# ============================================================
# Painel 8 — 📡 Painel de Ruído Condicional
# ============================================================


if painel == "📡 Painel de Ruído Condicional":

    st.markdown("## 📡 Painel de Ruído Condicional — V15.7 MAX")

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline não encontrado",
            "Execute primeiro **📁 Carregar Histórico** e **🛣️ Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)

    if not limitar_operacao(
        qtd_series,
        limite_series=LIMITE_SERIES_REPLAY_ULTRA,
        contexto="Ruído Condicional",
        painel="📡 Painel de Ruído Condicional",
    ):
        st.stop()

    st.info("Calculando indicadores de ruído condicional...")

# ============================================================
# PARTE 4/8 — FIM
# ============================================================
# ============================================================
# PARTE 5/8 — INÍCIO
# ============================================================

    try:
        if matriz_norm is None or len(matriz_norm) < 2:
            raise ValueError("Base insuficiente para medir ruído (matriz_norm < 2).")

        # Ruído Tipo A: dispersão intra-série (variação entre passageiros)
        variancias_intra = np.var(matriz_norm, axis=1)
        ruido_A_medio = float(np.mean(variancias_intra))

        # Ruído Tipo B: salto entre séries consecutivas (DX Light já usado)
        saltos = []
        for i in range(1, len(matriz_norm)):
            dist = np.linalg.norm(matriz_norm[i] - matriz_norm[i - 1])
            saltos.append(dist)
        ruido_B_medio = float(np.mean(saltos))

        # Normalização aproximada dos ruídos em [0,1]
        # (evitando divisão por zero)
        ruido_A_norm = min(1.0, ruido_A_medio / 0.08)   # escala empírica
        ruido_B_norm = min(1.0, ruido_B_medio / 1.20)   # escala empírica

        nr_percent = float((0.55 * ruido_A_norm + 0.45 * ruido_B_norm) * 100.0)

    except Exception as erro:
        exibir_bloco_mensagem(
            "Ruído indeterminado (base insuficiente / ruído técnico)",
            f"Métrica de ruído não pôde ser calculada com segurança.\n\nDetalhes técnicos: {erro}",
            tipo="warning",
        )
        st.session_state["nr_percent"] = None
        st.stop()

    # Classificação simples do NR%
    if nr_percent < 20:
        classe = "🟢 Baixo Ruído (Ambiente limpo)"
    elif nr_percent < 40:
        classe = "🟡 Ruído Moderado (Cuidado)"
    elif nr_percent < 60:
        classe = "🟠 Ruído Elevado (Atenção forte)"
    else:
        classe = "🔴 Ruído Crítico (Alta contaminação)"

    corpo = (
        f"- Séries analisadas: **{qtd_series}**\n"
        f"- Ruído Tipo A (intra-série, médio): **{ruido_A_medio:.4f}**\n"
        f"- Ruído Tipo B (entre séries, médio): **{ruido_B_medio:.4f}**\n"
        f"- NR% (Ruído Condicional Normalizado): **{nr_percent:.2f}%**\n"
        f"- Classe de ambiente: {classe}"
    )

    exibir_bloco_mensagem(
        "Resumo do Ruído Condicional",
        corpo,
        tipo="info",
    )

    st.session_state["nr_percent"] = nr_percent
    st.success("Cálculo de Ruído Condicional concluído!")


# ============================================================
# Painel 9 — 📉 Painel de Divergência S6 vs MC
# ============================================================
if painel == "📉 Painel de Divergência S6 vs MC":

    st.markdown("## 📉 Painel de Divergência S6 vs MC — V15.7 MAX")

    # Sincroniza alias de divergência (div_s6_mc -> divergencia_s6_mc)
    v16_sync_aliases_canonicos()


    divergencia = st.session_state.get("div_s6_mc", None)

    if divergencia is None:
        exibir_bloco_mensagem(
            "Divergência não calculada",
            "Execute o painel **⚙️ Modo TURBO++ ULTRA** para gerar a divergência S6 vs MC.",
            tipo="warning",
        )
        st.stop()

    # Classificação da divergência
    if divergencia < 2.0:
        classe = "🟢 Alta Convergência (S6 ≈ MC)"
        comentario = (
            "Os motores S6 Profundo e Monte Carlo Profundo estão altamente alinhados. "
            "O núcleo preditivo é mais confiável, favorecendo decisões mais agressivas."
        )
    elif divergencia < 5.0:
        classe = "🟡 Convergência Parcial"
        comentario = (
            "Há uma diferença moderada entre S6 e Monte Carlo. "
            "As decisões permanecem utilizáveis, mas requerem atenção adicional."
        )
    else:
        classe = "🔴 Alta Divergência (S6 distante de MC)"
        comentario = (
            "Os motores S6 e Monte Carlo estão em desacordo significativo. "
            "A recomendação é reduzir agressividade, aumentar coberturas ou aguardar estabilização."
        )

    corpo = (
        f"- Divergência S6 vs MC (norma): **{divergencia:.4f}**\n"
        f"- Classe de alinhamento: {classe}\n\n"
        f"{comentario}"
    )

    exibir_bloco_mensagem(
        "Resumo da Divergência S6 vs MC",
        corpo,
        tipo="info",
    )

    st.success("Análise de divergência concluída!")

# ============================================================
# PAINEL — 🧼 B1 | Higiene de Passageiros (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧼 B1 — Higiene de Passageiros":

    st.markdown("## 🧼 B1 — Higiene de Passageiros (V16)")
    st.caption(
        "Leitura observacional para identificar passageiros resilientes e nocivos.\n"
        "Não remove números. Não decide listas. Preparação para Perna B."
    )

    df = st.session_state.get("historico_df")

    if df is None or df.empty:
        st.info("Histórico não carregado.")
        st.stop()

    # ------------------------------------------------------------
    # Detecta colunas de passageiros (n-base)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    if len(col_pass) < 6:
        st.warning("Não foi possível detectar passageiros suficientes.")
        st.stop()

    # ------------------------------------------------------------
    # Frequência simples dos passageiros
    # ------------------------------------------------------------
    freq = {}

    for _, row in df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    freq[v] = freq.get(v, 0) + 1
            except Exception:
                pass

    if not freq:
        st.warning("Frequência de passageiros vazia.")
        st.stop()

    total_series = len(df)

    # ------------------------------------------------------------
    # Métricas observacionais
    # ------------------------------------------------------------
    dados = []

    for p, f in freq.items():
        taxa = f / total_series

        # heurísticas simples (OBSERVAÇÃO)
        resiliente = taxa >= 0.18
        nocivo = taxa <= 0.05

        dados.append({
            "Passageiro": p,
            "Ocorrências": f,
            "Taxa": round(taxa, 4),
            "Resiliente": "✅" if resiliente else "",
            "Nocivo": "⚠️" if nocivo else "",
        })

    df_pass = pd.DataFrame(dados).sort_values(
        by="Taxa", ascending=False
    )

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Leitura de Frequência dos Passageiros")
    st.dataframe(df_pass, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    resilientes = df_pass[df_pass["Resiliente"] == "✅"]["Passageiro"].tolist()
    nocivos = df_pass[df_pass["Nocivo"] == "⚠️"]["Passageiro"].tolist()

    leitura = (
        f"- Passageiros resilientes (recorrência alta): {resilientes[:12]}\n"
        f"- Passageiros potencialmente nocivos (recorrência muito baixa): {nocivos[:12]}\n\n"
        "⚠️ Nenhum passageiro foi removido.\n"
        "⚠️ Esta leitura serve apenas como preparação para refinamento futuro."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Higiene de Passageiros",
        leitura,
        tipo="info",
    )

    # ------------------------------------------------------------
    # Registro silencioso para Perna B
    # ------------------------------------------------------------
    st.session_state["b1_resilientes"] = resilientes
    st.session_state["b1_nocivos"] = nocivos

    st.success("B1 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧼 B1 | Higiene de Passageiros
# ============================================================

# ============================================================
# PAINEL — 🧩 B2 | Coerência Interna das Listas (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🧩 B2 — Coerência Interna das Listas":

    st.markdown("## 🧩 B2 — Coerência Interna das Listas (V16)")
    st.caption(
        "Leitura observacional de coesão e conflitos internos das listas.\n"
        "Não filtra, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Fonte das listas (preferência: Modo 6)
    # ------------------------------------------------------------
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    if not listas:
        st.info("Nenhuma lista disponível para análise. Execute o Modo 6.")
        st.stop()

    # ------------------------------------------------------------
    # Universo e estatísticas globais
    # ------------------------------------------------------------
    todas = [x for lst in listas for x in lst if isinstance(x, int)]
    if not todas:
        st.warning("Listas inválidas para análise.")
        st.stop()

    freq_global = pd.Series(todas).value_counts(normalize=True)

    # ------------------------------------------------------------
    # Métricas por lista
    # ------------------------------------------------------------
    linhas = []

    for i, lst in enumerate(listas, start=1):
        lst = [int(x) for x in lst if isinstance(x, int)]
        if not lst:
            continue

        # Coesão: média da frequência global dos elementos
        coesao = float(freq_global.loc[lst].mean()) if set(lst).issubset(freq_global.index) else 0.0

        # Conflito simples: proporção de pares muito raros juntos
        pares = [(a, b) for a in lst for b in lst if a < b]
        raros = 0
        for a, b in pares:
            fa = freq_global.get(a, 0.0)
            fb = freq_global.get(b, 0.0)
            if fa < 0.05 and fb < 0.05:
                raros += 1

        conflito = raros / max(1, len(pares))

        linhas.append({
            "Lista": i,
            "Coesão (↑ melhor)": round(coesao, 4),
            "Conflito (↓ melhor)": round(conflito, 4),
        })

    df_b2 = pd.DataFrame(linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Coesão e Conflito por Lista")
    st.dataframe(df_b2, use_container_width=True)

    # ------------------------------------------------------------
    # Síntese mastigada (NÃO decisória)
    # ------------------------------------------------------------
    leitura = (
        "- **Coesão** alta indica elementos com histórico compatível.\n"
        "- **Conflito** alto indica combinações raras juntas.\n\n"
        "⚠️ Nenhuma lista foi removida ou priorizada.\n"
        "⚠️ Use esta leitura apenas para preparação."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Coerência Interna",
        leitura,
        tipo="info",
    )

    # Registro silencioso
    st.session_state["b2_coerencia_df"] = df_b2

    st.success("B2 concluído — leitura registrada com sucesso.")

# ============================================================
# <<< FIM — PAINEL 🧩 B2 | Coerência Interna das Listas
# ============================================================

# ============================================================
# PAINEL — 🟢 B3 | Prontidão para Refinamento (V16)
# Observacional | NÃO decide | NÃO altera motores
# ============================================================

elif painel == "🟢 B3 — Prontidão (Refinamento)":

    st.markdown("## 🟢 B3 — Prontidão para Refinamento (V16)")
    st.caption(
        "Avalia se o contexto permite avançar da leitura (Perna B) "
        "para refinamento de passageiros. Não executa nada."
    )

    # ------------------------------------------------------------
    # Leituras já consolidadas
    # ------------------------------------------------------------
    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    df_b2 = st.session_state.get("b2_coerencia_df")

    if not diag or df_b2 is None or df_b2.empty:
        st.info(
            "Leituras insuficientes para avaliar prontidão.\n\n"
            "Execute B1, B2 e Diagnóstico ECO & Estado."
        )
        st.stop()

    eco = diag.get("eco")
    eco_persist = diag.get("eco_persistencia")
    acion = diag.get("eco_acionabilidade")
    estado = diag.get("estado")
    estado_ok = diag.get("estado_confiavel")

    # ------------------------------------------------------------
    # Heurísticas de prontidão (OBSERVAÇÃO)
    # ------------------------------------------------------------
    sinais_ok = []

    if eco in ("médio", "forte"):
        sinais_ok.append("ECO ≥ médio")

    if eco == "fraco" and eco_persist == "persistente" and estado in ("parado", "movimento_lento"):
        sinais_ok.append("ECO fraco porém estável com estado calmo")

    if estado_ok and estado in ("parado", "movimento_lento"):
        sinais_ok.append("Estado desacelerado e confiável")

    # Coesão média das listas
    coesao_media = float(df_b2["Coesão (↑ melhor)"].mean())

    if coesao_media >= 0.12:
        sinais_ok.append("Coesão média aceitável")

    # ------------------------------------------------------------
    # Veredito OBSERVACIONAL
    # ------------------------------------------------------------
    pronto = len(sinais_ok) >= 3

    if pronto:
        status = "🟢 PRONTO PARA REFINAMENTO"
        detalhe = (
            "O contexto permite iniciar refinamento controlado de passageiros.\n"
            "A Perna B pode evoluir para ações leves (sem afunilar)."
        )
        tipo = "success"
    else:
        status = "🟡 AINDA EM PREPARAÇÃO"
        detalhe = (
            "O contexto ainda pede dispersão.\n"
            "Continue observando e acumulando leitura."
        )
        tipo = "info"

    corpo = (
        f"**Status:** {status}\n\n"
        f"**Sinais atendidos:** {sinais_ok if sinais_ok else 'Nenhum'}\n\n"
        f"**Coesão média das listas:** {coesao_media:.4f}\n\n"
        f"⚠️ Este painel **não executa refinamento**.\n"
        f"⚠️ Serve apenas para indicar **prontidão**."
    )

    exibir_bloco_mensagem(
        "🧠 Veredito de Prontidão — Perna B",
        corpo,
        tipo=tipo,
    )

    # Registro silencioso
    st.session_state["b3_pronto_refinar"] = pronto

    st.success("B3 concluído — prontidão avaliada.")

# ============================================================
# <<< FIM — PAINEL 🟢 B3 | Prontidão para Refinamento
# ============================================================

# ============================================================
# PAINEL — 🟣 B4 | Refinamento Leve de Passageiros (V16)
# Ajuste leve | Reversível | NÃO decide | NÃO afunila
# ============================================================

elif painel == "🟣 B4 — Refinamento Leve de Passageiros":

    st.markdown("## 🟣 B4 — Refinamento Leve de Passageiros (V16)")
    st.caption(
        "Aplica ajustes leves e reversíveis nos passageiros das listas.\n"
        "Não reduz volume, não prioriza, não decide."
    )

    # ------------------------------------------------------------
    # Pré-condições
    # ------------------------------------------------------------
    pronto = st.session_state.get("b3_pronto_refinar", False)
    listas = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    resilientes = st.session_state.get("b1_resilientes", [])
    nocivos = st.session_state.get("b1_nocivos", [])

    if not listas:
        st.info("Nenhuma lista disponível. Execute o Modo 6.")
        st.stop()

    if not pronto:
        st.warning(
            "Contexto ainda não marcado como pronto para refinamento.\n"
            "Este painel é **apenas demonstrativo** neste estado."
        )

    # ------------------------------------------------------------
    # Universo de referência
    # ------------------------------------------------------------
    universo = sorted({int(x) for lst in listas for x in lst if isinstance(x, int)})
    if not universo:
        st.warning("Universo inválido para refinamento.")
        st.stop()

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------
    # Refinamento leve (heurístico, reversível)
    # ------------------------------------------------------------
    listas_refinadas = []

    for lst in listas:
        nova = list(lst)

        # substitui no máx. 1 passageiro nocivo por um resiliente
        candidatos_nocivos = [x for x in nova if x in nocivos]
        candidatos_resilientes = [x for x in resilientes if x not in nova]

        if candidatos_nocivos and candidatos_resilientes:
            sai = rng.choice(candidatos_nocivos)
            entra = rng.choice(candidatos_resilientes)
            nova = [entra if x == sai else x for x in nova]

        listas_refinadas.append(sorted(set(nova)))

    # ------------------------------------------------------------
    # Exibição comparativa (leitura)
    # ------------------------------------------------------------
    st.markdown("### 🔍 Comparação — Antes × Depois (amostra)")
    limite = min(10, len(listas))

    for i in range(limite):
        col1, col2 = st.columns(2)
        with col1:
            st.code(f"Antes {i+1}: {sorted(listas[i])}", language="python")
        with col2:
            st.code(f"Depois {i+1}: {listas_refinadas[i]}", language="python")

    # ------------------------------------------------------------
    # Síntese observacional
    # ------------------------------------------------------------
    leitura = (
        "- Ajuste máximo: **1 passageiro por lista**\n"
        "- Volume total preservado\n"
        "- Refinamento **reversível**\n"
        "- Uso de passageiros resilientes\n\n"
        "⚠️ As listas refinadas **não substituem** as originais.\n"
        "⚠️ Cabe ao operador decidir se usa esta leitura."
    )

    exibir_bloco_mensagem(
        "🧠 Leitura Observacional — Refinamento Leve",
        leitura,
        tipo="info",
    )

    # Registro silencioso (não substitui listas)
    st.session_state["b4_listas_refinadas"] = listas_refinadas

    st.success("B4 concluído — refinamento leve avaliado.")

# ============================================================
# <<< FIM — PAINEL 🟣 B4 | Refinamento Leve de Passageiros
# ============================================================


# ============================================================
# Painel 10 — 🧭 Monitor de Risco — k & k*
# ============================================================
if painel == "🧭 Monitor de Risco — k & k*":

    st.markdown("## 🧭 Monitor de Risco — k & k* — V15.7 MAX")

    df = st.session_state.get("historico_df")
    k_star = st.session_state.get("sentinela_kstar")
    nr_percent = st.session_state.get("nr_percent")
    divergencia = st.session_state.get("div_s6_mc")

    if df is None:
        exibir_bloco_mensagem(
            "Histórico não carregado",
            "Vá ao painel **📁 Carregar Histórico** antes.",
            tipo="warning",
        )
        st.stop()

    metricas = calcular_metricas_basicas_historico(df)

    qtd_series = metricas.get("qtd_series", 0)
    min_k = metricas.get("min_k")
    max_k = metricas.get("max_k")
    media_k = metricas.get("media_k")

    # Garantias (se sentinelas/ruído/divergência não tiverem sido rodados)
    if k_star is None:
        k_star = 0.25  # valor neutro
    if nr_percent is None:
        nr_percent = 35.0  # ruído moderado default
    if divergencia is None:
        divergencia = 4.0  # divergência intermediária

    # Índice de risco composto (escala 0 a 1)
    # k* alto, NR% alto e divergência alta => risco maior
    kstar_norm = min(1.0, k_star / 0.50)
    nr_norm = min(1.0, nr_percent / 70.0)
    div_norm = min(1.0, divergencia / 8.0)

    indice_risco = float(0.40 * kstar_norm + 0.35 * nr_norm + 0.25 * div_norm)

    # Classificação de risco
    if indice_risco < 0.30:
        classe_risco = "🟢 Risco Baixo (Janela Favorável)"
        recomendacao = (
            "O ambiente está favorável para decisões mais agressivas, "
            "com menor necessidade de coberturas pesadas."
        )
    elif indice_risco < 0.55:
        classe_risco = "🟡 Risco Moderado"
        recomendacao = (
            "Ambiente misto. Recomenda-se equilíbrio entre núcleo e coberturas, "
            "com atenção à divergência e ao ruído."
        )
    elif indice_risco < 0.80:
        classe_risco = "🟠 Risco Elevado"
        recomendacao = (
            "Ambiente turbulento. Aumentar coberturas, reduzir exposição e "
            "observar de perto os painéis de Ruído e Divergência."
        )
    else:
        classe_risco = "🔴 Risco Crítico"
        recomendacao = (
            "Condição crítica. Sugere-se extrema cautela, priorizando preservação e "
            "eventualmente aguardando melhoria do regime antes de decisões mais fortes."
        )

    corpo = (
        f"- Séries no histórico: **{qtd_series}**\n"
        f"- k mínimo: **{min_k}** · k máximo: **{max_k}** · k médio: **{media_k:.2f}**\n"
        f"- k* (sentinela): **{k_star:.4f}**\n"
        f"- NR% (Ruído Condicional): **{nr_percent:.2f}%**\n"
        f"- Divergência S6 vs MC: **{divergencia:.4f}**\n"
        f"- Índice composto de risco: **{indice_risco:.4f}**\n"
        f"- Classe de risco: {classe_risco}\n\n"
        f"{recomendacao}"
    )

    exibir_bloco_mensagem(
        "Resumo do Monitor de Risco — k & k*",
        corpo,
        tipo="info",
    )

    st.session_state["diagnostico_risco"] = {
        "indice_risco": indice_risco,
        "classe_risco": classe_risco,
        "k_star": k_star,
        "nr_percent": nr_percent,
        "divergencia": divergencia,
    }


    # ============================================================
    # M1 — ESPELHO CANÔNICO DO RISCO (S3)
    # (apenas espelhamento: não recalcula, não decide, não altera motores)
    # ============================================================
    st.session_state["k_star"] = float(k_star) if isinstance(k_star, (int, float)) else k_star
    st.session_state["nr_percent"] = float(nr_percent) if isinstance(nr_percent, (int, float)) else nr_percent

    # Divergência: manter chave histórica (div_s6_mc) e chave canônica do Mirror
    st.session_state["div_s6_mc"] = float(divergencia) if isinstance(divergencia, (int, float)) else divergencia
    st.session_state["divergencia_s6_mc"] = float(divergencia) if isinstance(divergencia, (int, float)) else divergencia

    st.session_state["indice_risco"] = float(indice_risco) if isinstance(indice_risco, (int, float)) else indice_risco
    st.session_state["classe_risco"] = classe_risco

    # Selos M1 (S3)
    st.session_state["m1_selo_risco_ok"] = True
    st.session_state["m1_ts_risco_ok"] = __import__("time").time()

    st.success("Monitor de Risco atualizado com sucesso!")

# ============================================================
# PARTE 5/8 — FIM
# ============================================================
# ============================================================
# PARTE 6/8 — INÍCIO
# ============================================================

# ============================================================
# BLOCO V16 — PROTOCOLO PRÉ-ECO / ECO
# Observador tático — AJUSTA POSTURA PARA A PRÓXIMA SÉRIE
# NÃO prevê, NÃO altera motor, NÃO bloqueia
# ============================================================

def v16_avaliar_pre_eco_eco():
    """
    Usa SOMENTE o estado ATUAL (última série do histórico)
    para definir a postura de ataque da PRÓXIMA série.
    """

    k_star = st.session_state.get("sentinela_kstar")
    nr = st.session_state.get("nr_percent")
    div = st.session_state.get("div_s6_mc")
    risco = (st.session_state.get("diagnostico_risco") or {}).get("indice_risco")

    # Defaults defensivos
    k_star = float(k_star) if isinstance(k_star, (int, float)) else 0.30
    nr = float(nr) if isinstance(nr, (int, float)) else 50.0
    div = float(div) if isinstance(div, (int, float)) else 6.0
    risco = float(risco) if isinstance(risco, (int, float)) else 0.60

    sinais_ok = 0

    if k_star <= 0.30:
        sinais_ok += 1
    if nr <= 45.0:
        sinais_ok += 1
    if div <= 6.0:
        sinais_ok += 1
    if risco <= 0.55:
        sinais_ok += 1

    # Classificação
    if sinais_ok >= 3:
        status = "PRE_ECO_ATIVO"
        postura = "ATIVA"
        comentario = (
            "🟡 PRÉ-ECO detectado — ambiente NÃO piora.\n"
            "Postura ativa para a próxima série.\n"
            "Modo 6 ligado, volume moderado."
        )
    else:
        status = "SEM_ECO"
        postura = "DEFENSIVA"
        comentario = (
            "🔴 Nenhum pré-eco — ambiente instável.\n"
            "Operar apenas com coberturas."
        )

    resultado = {
        "status": status,
        "postura": postura,
        "sinais_ok": sinais_ok,
        "comentario": comentario,
    }

    st.session_state["v16_pre_eco"] = resultado
    return resultado

# ============================================================
# FUNÇÃO — SANIDADE FINAL DAS LISTAS (DISPONÍVEL AO MODO 6)
# Remove listas inválidas, duplicatas e permutações
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    pc_exec_trace("ENTER sanidade_final_listas", {"arg_n": len(listas or [])})
    pc_list_source_detector("BEFORE sanidade_final_listas", listas, {"arg_n": len(listas or [])})
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """
    if not listas:
        return []

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # exatamente 6 números distintos
        if len(nums) != 6 or len(set(nums)) != 6:
            continue

        chave = tuple(sorted(nums))
        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    pc_list_source_detector("AFTER sanidade_final_listas", listas_saneadas, {"arg_n": len(listas_saneadas or [])})
    return listas_saneadas

# ============================================================
# FIM — FUNÇÃO SANIDADE FINAL DAS LISTAS
# ============================================================

# ============================================================
# BLOCO C (V10) — AJUSTE FINO NUMÉRICO (pré‑Camada 4)
# - NÃO cria motor novo
# - NÃO mexe em Modo 6 / TURBO / Bala
# - NÃO expande universo (só permuta dentro do universo já presente no pacote)
# - Objetivo: melhorar coerência interna e redistribuição do miolo sem perder borda real
# ============================================================

def v9_classificar_memoria_borda(*, df_res: Optional[pd.DataFrame], total_hits: int, pct: Dict[str, float]) -> Dict[str, Any]:
    """
    Classifica a qualidade de lastro da Memória V9 para uso informativo (baliza),
    sem nunca virar decisão automática.

    Retorna:
      - status: 'INEXISTENTE' | 'INSUFICIENTE' | 'OK' | 'EXCESSIVA'
      - motivo_curto
      - n_alvos_avaliados
    """
    try:
        n_alvos = 0
        if df_res is not None and not df_res.empty:
            # conta alvos existentes (k+1 e k+2) a partir das colunas alvo_*
            if "alvo_1" in df_res.columns:
                n_alvos += int(df_res["alvo_1"].notna().sum())
            if "alvo_2" in df_res.columns:
                n_alvos += int(df_res["alvo_2"].notna().sum())

        if df_res is None or df_res.empty or total_hits <= 0 or n_alvos <= 0:
            return {
                "status": "INEXISTENTE",
                "motivo_curto": "Sem alvos avaliados suficientes na Memória V9.",
                "n_alvos_avaliados": int(n_alvos),
            }

        # Heurística canônica (observacional):
        # - muito pouco: risco de miragem
        # - muito: mistura de regimes e diluição de sinal local
        if n_alvos < 6:
            return {"status": "INSUFICIENTE", "motivo_curto": "Poucos alvos (risco de miragem).", "n_alvos_avaliados": int(n_alvos)}
        if n_alvos > 60:
            return {"status": "EXCESSIVA", "motivo_curto": "Alvos demais (pode diluir sinal local).", "n_alvos_avaliados": int(n_alvos)}

        # Regra simples de saúde: se fora_do_pacote ainda está altíssimo, lastro existe, mas é fraco
        fora_pct = float(pct.get("fora", 0.0))
        if fora_pct >= 65.0:
            return {"status": "OK", "motivo_curto": "Lastro existe, mas fora_do_pacote está alto.", "n_alvos_avaliados": int(n_alvos)}

        return {"status": "OK", "motivo_curto": "Lastro adequado para baliza informativa.", "n_alvos_avaliados": int(n_alvos)}
    except Exception:
        return {"status": "INEXISTENTE", "motivo_curto": "Falha ao classificar Memória V9.", "n_alvos_avaliados": 0}


# === BLOCO C REAL (V10) — AJUSTE FINO NUMÉRICO — CANÔNICO ===
# Ctrl+F: BLOCOC_CALLSITE_CANONICO | v10_bloco_c_aplicar_ajuste_fino_numerico | bloco_c_real_diag


def v16_anti_exato_obter_nocivos_consistentes_silent(df_base, col_pass, W=60, ALPHA=1, AMIN=12, BMIN=40):
    """ANTI-EXATO (silent) — retorna lista de passageiros classificados como NOCIVO CONSISTENTE.
    Mesma lógica do painel '📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos', sem UI.
    Não decide nada — apenas fornece fonte objetiva para BLOCO C mínimo (pré-C4).
    """
    try:
        if df_base is None or getattr(df_base, "empty", True):
            return []
        if not col_pass or len(col_pass) < 6:
            return []
        historico = df_base[col_pass].astype(int).values.tolist()
        n = len(historico)
        if n < (W + 2):
            return []

        def contar_hits(car_a, car_b):
            return len(set(car_a).intersection(set(car_b)))

        resultados = []
        for t in range(n - W - 1, n - 1):
            janela = historico[t - W + 1 : t + 1]
            alvo = historico[t + 1]
            for car in janela:
                hits = contar_hits(car, alvo)
                resultados.append({
                    "passageiros": car,
                    "hit2": 1 if hits >= 2 else 0,
                    "hit3": 1 if hits >= 3 else 0,
                })

        if not resultados:
            return []

        df = pd.DataFrame(resultados)
        universo = sorted({p for car in df["passageiros"] for p in car})
        nocivos = []
        for p in universo:
            presente = df["passageiros"].apply(lambda x: p in x)
            A = int(presente.sum())
            B = int((~presente).sum())
            if A < AMIN or B < BMIN:
                continue
            a3 = float(df.loc[presente, "hit3"].sum())
            b3 = float(df.loc[~presente, "hit3"].sum())
            p1 = (a3 + ALPHA) / (A + 2 * ALPHA)
            p0 = (b3 + ALPHA) / (B + 2 * ALPHA)
            delta = p1 - p0
            lift = (p1 / p0) if (p0 > 0) else 1.0
            if (delta < 0) and (lift <= 0.92):
                nocivos.append(int(p))
        # ordena estável
        nocivos = sorted(list(dict.fromkeys(nocivos)))
        return nocivos
    except Exception:
        return []


def v16_anti_exato_calcular_lambda_star_silent(
    df_base,
    col_pass="passageiro",
    col_lista="lista",
    col_hitmax="hit_max",
    win=60,
    alpha=1.0,
    amin=12,
    bmin=40,
    evento_principal="hit3",
    evento_suporte="hit2",
):
    """
    λ* (lambda_star) — força CANÔNICA da penalidade de nocivos no W(p), calibrada SOMENTE em df_eval/SAFE.

    - Pré-C4, observacional/auditável.
    - Não decide volume.
    - Não altera Camada 4 diretamente; só fornece um escalar (0..1) para modular a penalidade.

    Intuição:
    - Se os passageiros marcados como NOCIVO CONSISTENTE mostram uma queda robusta de probabilidade do evento-alvo,
      então λ* sobe (penalidade mais aplicada).
    - Se a evidência é fraca/instável, λ* cai (penalidade atenua, evitando overfit).
    """
    try:
        if df_base is None or len(df_base) == 0:
            return {
                "lambda_star": 0.0,
                "n_nocivos": 0,
                "sev_mean": 0.0,
                "nota": "sem base",
            }

        # recorte canônico: últimos 'win' alvos do SAFE
        dfw = df_base.copy()
        if "k_base" in dfw.columns:
            dfw = dfw.sort_values("k_base")
        if win and len(dfw) > int(win):
            dfw = dfw.tail(int(win))

        # reaproveita o classificador canônico para obter a lista de nocivos
        nocivos, _stats_tbl = v16_anti_exato_obter_nocivos_consistentes_silent(
            dfw,
            col_pass=col_pass,
            col_lista=col_lista,
            col_hitmax=col_hitmax,
            alpha=alpha,
            amin=amin,
            bmin=bmin,
            evento_principal=evento_principal,
            evento_suporte=evento_suporte,
        )
        nocivos = list(nocivos) if nocivos else []
        if len(nocivos) == 0:
            return {
                "lambda_star": 0.0,
                "n_nocivos": 0,
                "sev_mean": 0.0,
                "nota": "sem nocivos consistentes",
            }

        # severidade = queda relativa do evento_principal quando o passageiro está no pacote
        sevs = []
        for p in nocivos:
            s = v16_anti_exato_stats_passageiro_laplace(
                dfw,
                p,
                col_pass=col_pass,
                col_lista=col_lista,
                col_hitmax=col_hitmax,
                alpha=alpha,
                evento_principal=evento_principal,
                evento_suporte=evento_suporte,
            )
            # p0: P(evento | ausente), p1: P(evento | presente)
            p0 = float(s.get("p0_h3", 0.0))
            p1 = float(s.get("p1_h3", 0.0))
            if p0 <= 1e-12:
                continue
            sev = (p0 - p1) / max(p0, 1e-12)
            if sev > 0:
                sevs.append(min(1.0, max(0.0, sev)))

        if len(sevs) == 0:
            return {
                "lambda_star": 0.0,
                "n_nocivos": len(nocivos),
                "sev_mean": 0.0,
                "nota": "nocivos sem severidade mensurável",
            }

        sev_mean = float(sum(sevs) / len(sevs))

        # fator de massa: quanto mais perto de 'win' cheio, mais confiável.
        mass = min(1.0, float(len(dfw)) / float(max(1, int(win))))


        faltam = int(max(0, int(win) - int(len(dfw))))
        # Fase de estabilização do λ*: quanto mais massa (janela preenchida), mais confiável o sinal
        if mass >= 0.85:
            fase = "ESTAVEL"
        elif mass >= 0.50:
            fase = "ESTABILIZANDO"
        else:
            fase = "INICIAL"
        # λ* canônico (0..1)
        lambda_star = min(1.0, max(0.0, sev_mean * mass))

        return {
            "lambda_star": float(lambda_star),
            "n_nocivos": int(len(nocivos)),
            "sev_mean": float(sev_mean),
            "mass": float(mass),
            "fase": str(fase),
            "faltam": int(faltam),
            "win": int(win),
            "dfw_len": int(len(dfw)),
            "nota": "λ* = sev_mean × massa",
        }
    except Exception:
        return {
            "lambda_star": 0.0,
            "n_nocivos": 0,
            "sev_mean": 0.0,
            "nota": "erro interno (SAFE)",
        }



def v10_bloco_c_aplicar_ajuste_fino_numerico(listas, n_real, v8_borda_info=None, v9_memoria_info=None):
    """
    BLOCO C (REAL) — Descompressão Estrutural (canônico, sem opção)

    Objetivo: reduzir "compressão" do pacote trazendo, de forma CONTROLADA e mínima,
    passageiros com pressão recente (fora do pacote) e retirando elementos com
    baixa pressão estrutural (dentro do pacote), sem inventar sensores novos.

    Conceitos canônicos usados aqui:
    - Vetor de Descompressão η (eta): pressão estrutural por passageiro,
      calculada como desvio (freq_recente - freq_longo).
    - Operador C₁: aplica 0–1 trocas por lista (mínimo necessário) guiado por η.
    """
    # Guardas
    if not listas or not isinstance(listas, list):
        return {'aplicado': False, 'motivo': 'listas_invalidas', 'listas_ajustadas': listas, 'diag_key': 'bloco_c_real_diag'}

    # ------------------------------------------------------------
    # BLOCO C MÍNIMO (REAL) — GOVERNANÇA DE APLICAÇÃO (pré-C4)
    # - Só aplica com SS ATINGIDA (base mínima) e quando ainda NÃO há evidência de janela (any_4p_seen=False)
    # - Fonte objetiva: ANTI-EXATO | Passageiros Nocivos Consistentes (painel/silent)
    # ------------------------------------------------------------
    try:
        ss_ok = bool((st.session_state.get("ss_info") or {}).get("status")) or (str(st.session_state.get("ss_status") or "").strip().upper() == "ATINGIDA")
    except Exception:
        ss_ok = False
    try:
        stats_janela = st.session_state.get("replay_stats_prova_janela") or {}
        any_4p_seen = bool(stats_janela.get("any_4p_seen")) if isinstance(stats_janela, dict) else False
    except Exception:
        any_4p_seen = False

    if not ss_ok:
        st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'ss_nao_atingida', 'trocas': 0}
        return {'aplicado': False, 'motivo': 'ss_nao_atingida', 'listas_ajustadas': listas, 'trocas': 0, 'diag_key': 'bloco_c_real_diag'}

    if any_4p_seen:
        # marca fresta como 'vista' na sessão (persistente até reset de histórico)
        try:
            st.session_state['bloco_c_fresta_ativa'] = True
        except Exception:
            pass
        # BLOCO C (FASE 2) — Janela nascente (a barreira já foi atravessada, mas ainda não está sustentada)
        # - Ainda pré-C4, auditável, sem motor novo.
        # - Atua apenas para aumentar recorrência de 4+ quando ainda é raro.
        try:
            rate_4p = _pc_safe_float(stats_janela.get("rate_4p"), None) if isinstance(stats_janela, dict) else None
            if rate_4p is None and isinstance(stats_janela, dict):
                rate_4p = _pc_safe_float(stats_janela.get("rate_4p_w"), None)
        except Exception:
            rate_4p = None
        try:
            gap_norm = _pc_safe_float(stats_janela.get("fechamento_gap_norm"), None) if isinstance(stats_janela, dict) else None
            if gap_norm is None and isinstance(stats_janela, dict):
                gap_norm = _pc_safe_float(stats_janela.get("fechamento_gap_norm_w"), None)
        except Exception:
            gap_norm = None
        try:
            zero_hit_rate = _pc_safe_float(stats_janela.get("zero_hit_rate"), None) if isinstance(stats_janela, dict) else None
            if zero_hit_rate is None and isinstance(stats_janela, dict):
                zero_hit_rate = _pc_safe_float(stats_janela.get("zero_hit_rate_w"), None)
        except Exception:
            zero_hit_rate = None
        try:
            curv_info = st.session_state.get("curvatura_sustentada_info") or {}
            curv_sust = bool(curv_info.get("curvatura_sustentada_recente")) if isinstance(curv_info, dict) else False
            dist_desde_ultimo_4 = _pc_safe_float(curv_info.get("dist_desde_ultimo_4"), None) if isinstance(curv_info, dict) else None
        except Exception:
            curv_sust = False
            dist_desde_ultimo_4 = None

        # Critério canônico: só entra na Fase 2 se 4+ já apareceu, mas ainda é raro e não há sustentação de curvatura.
        RATE4P_LIM = 0.05  # 5%
        if (rate_4p is not None) and (rate_4p >= RATE4P_LIM):
            st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'janela_ja_mais_forte', 'trocas': 0, 'fase': 2}
            return {'aplicado': False, 'motivo': 'janela_ja_mais_forte', 'listas_ajustadas': listas, 'trocas': 0, 'diag_key': 'bloco_c_real_diag'}
        if curv_sust:
            st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'curvatura_sustentada', 'trocas': 0, 'fase': 2}
            return {'aplicado': False, 'motivo': 'curvatura_sustentada', 'listas_ajustadas': listas, 'trocas': 0, 'diag_key': 'bloco_c_real_diag'}

        # Dosagem canônica (sem opção): base 1, sobe para 2 só em condição de "demora geométrica" + gap alto
        dose = 1
        motivo_dose = []
        if (dist_desde_ultimo_4 is not None) and (gap_norm is not None):
            if (dist_desde_ultimo_4 >= 10) and (gap_norm >= 0.70):
                dose = 2
                motivo_dose.append("dose2: dist>=10 & gap_norm>=0.70")
        # Freio simples: se zero_hit_rate estiver alto, não aumenta dose
        if (zero_hit_rate is not None) and (zero_hit_rate >= 0.20):
            if dose != 1:
                motivo_dose.append("freio: zero_hit>=0.20")
            dose = 1

        # Marca fase/dose no diag; o restante da função executa trocas (agora com dose > 1 permitido)
        st.session_state['bloco_c_real_diag'] = {
            'aplicado': False,
            'motivo': 'fase2_ativa',
            'trocas': 0,
            'fase': 2,
            'dose': int(dose),
            'motivo_dose': "; ".join(motivo_dose) if motivo_dose else "dose1",
        }
    else:
        # BLOCO C (FASE 1) — Barreira (ainda sem evidência de janela)
        st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'fase1_ativa', 'trocas': 0, 'fase': 1, 'dose': 1}
        dose = 1

    # nocivos consistentes (fonte objetiva)
    nocivos = st.session_state.get("anti_exato_nocivos_consistentes") or []

    # --- 1) Coleta do histórico (fonte oficial já carregada no app) ---
    df = (st.session_state.get("historico_df_full_safe") or st.session_state.get("historico_df_full") or st.session_state.get("historico_df"))
    if df is None or getattr(df, "empty", True):
        st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'sem_historico', 'trocas': 0}
        return {'aplicado': False, 'motivo': 'sem_historico', 'listas_ajustadas': listas, 'trocas': 0, 'diag_key': 'bloco_c_real_diag'}
    # Detecta colunas dos passageiros no histórico (p1..p6 ou similares)
    pcols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("p")]
    if not pcols:
        st.session_state['bloco_c_real_diag'] = {'aplicado': False, 'motivo': 'sem_colunas_p', 'trocas': 0}
        return {'aplicado': False, 'motivo': 'sem_colunas_p', 'listas_ajustadas': listas, 'trocas': 0, 'diag_key': 'bloco_c_real_diag'}
    # Universo (tentativa de ler do app; fallback 60)
    universo_max = st.session_state.get("pc_universo_max")
    if universo_max is None:
        # Heurística: tenta inferir do próprio histórico
        try:
            universo_max = int(df[pcols].max().max())
        except Exception:
            universo_max = 60
    try:
        universo_max = int(universo_max)
    except Exception:
        universo_max = 60
    universo_max = max(6, min(universo_max, 200))

    
    try:
        # Se o operador não abriu o painel ANTI-EXATO, calculamos silenciosamente (mesma lógica do painel).
        if not nocivos:
            col_pass_anti = st.session_state.get("pipeline_col_pass") or pcols
            nocivos = v16_anti_exato_obter_nocivos_consistentes_silent(df_base=df, col_pass=col_pass_anti, W=60, ALPHA=1, AMIN=12, BMIN=40)
            if nocivos:
                st.session_state["anti_exato_nocivos_consistentes"] = nocivos
    except Exception:
        pass

    nocivos_set = set()
    try:
        nocivos_set = set(int(x) for x in (nocivos or []))
    except Exception:
        nocivos_set = set()
    # ------------------------------------------------------------
    # λ* (lambda_star) — força canônica da penalidade de nocivos no W(p)
    # - calculada SOMENTE com base SAFE (df_eval), pré-C4, auditável
    # - NÃO decide nada; apenas modula a penalidade (evita overfit em base fraca)
    # ------------------------------------------------------------
    lambda_star_info = {"lambda_star": 0.0}
    try:
        df_eval_safe = st.session_state.get("df_eval_safe")
        if df_eval_safe is not None and len(df_eval_safe) > 0:
            # usa o mesmo col_pass do anti-exato (o do pipeline), quando disponível
            col_pass_anti = st.session_state.get("pipeline_col_pass") or pcols
            lambda_star_info = v16_anti_exato_calcular_lambda_star_silent(
                df_eval_safe,
                col_pass=col_pass_anti,
                win=60,
                alpha=1.0,
                amin=12,
                bmin=40,
                evento_principal="hit3",
                evento_suporte="hit2",
            )
    except Exception:
        lambda_star_info = {"lambda_star": 0.0}

    lambda_star_raw = 0.0
    try:
        lambda_star_raw = float(lambda_star_info.get("lambda_star", 0.0) or 0.0)
    except Exception:
        lambda_star_raw = 0.0

    # ------------------------------------------------------------
    # FASE DE ESTABILIZAÇÃO DO λ (lambda_star) — sem bifurcar
    # Objetivo: impedir "salto" de penalidade quando a base SAFE ainda é fraca.
    # - Fase vem do cálculo (massa ex-post da janela).
    # - Aplicamos:
    #   (a) fator de rampa por fase (INICIAL/TRANSICAO/ESTAVEL)
    #   (b) suavização + limite de variação por rodada (clamp)
    # Tudo pré-C4, auditável, e só modula o peso — não decide nada.
    # ------------------------------------------------------------
    fase_estab = (lambda_star_info or {}).get("fase_estabilizacao", "INICIAL")
    if fase_estab == "ESTAVEL":
        fator_fase = 1.00
    elif fase_estab == "TRANSICAO":
        fator_fase = 0.60
    else:
        fator_fase = 0.25

    lambda_star_target = float(max(0.0, lambda_star_raw) * fator_fase)

    # suavização canônica (estado de sessão): evita bagunçar o pacote por um único snapshot
    prev = st.session_state.get("lambda_star_smooth")
    try:
        prev = float(prev) if prev is not None else None
    except Exception:
        prev = None

    # limite de variação por rodada (anti-salto)
    DELTA_MAX = 0.25
    BETA = 0.50  # 0.5 = meio caminho por rodada (simples e estável)

    if prev is None:
        lambda_star_smooth = lambda_star_target
    else:
        delta = lambda_star_target - prev
        if delta > DELTA_MAX:
            delta = DELTA_MAX
        elif delta < -DELTA_MAX:
            delta = -DELTA_MAX
        stepped = prev + delta
        lambda_star_smooth = (BETA * prev) + ((1.0 - BETA) * stepped)

    # guarda para governança/relatórios (pré-C4)
    st.session_state["lambda_star_raw"] = lambda_star_raw
    st.session_state["lambda_star_target"] = lambda_star_target
    st.session_state["lambda_star_smooth"] = lambda_star_smooth
    st.session_state["lambda_star_fase"] = fase_estab

    # valor efetivo para o BLOCO C (FASE 6 / W(p))
    lambda_star = float(lambda_star_smooth)
    st.session_state["lambda_star"] = lambda_star

    # ------------------------------------------------------------
    # BLOCO C (FASE 3) — Sustentar fresta sem depender de nocivos
    # - Ativa quando: fase==2 (any_4p_seen=True), mas NÃO há nocivos consistentes detectáveis
    # - Fonte objetiva: df_eval (Replay/SAFE) + Trave/Proximidade (fora_perto alto)
    # - Ação: micro-empurrão geométrico (trocas mínimas) para aumentar recorrência de 4+
    # ------------------------------------------------------------
    fase3_ok = False
    quase_entram = []
    fora_perto_ratio = None
    try:
        fase_atual = int((st.session_state.get('bloco_c_real_diag') or {}).get('fase', 1))
    except Exception:
        fase_atual = 1

    try:
        df_eval = st.session_state.get("df_eval")
        if ((fase_atual == 2) or bool(st.session_state.get('bloco_c_fresta_ativa', False))) and (not nocivos_set) and (df_eval is not None) and (not getattr(df_eval, "empty", True)):
            # ratio global de fora_perto usando colunas do df_eval
            fp = 0
            fl = 0
            for cfp, cfl in [("fora_perto_1", "fora_longe_1"), ("fora_perto_2", "fora_longe_2")]:
                if cfp in df_eval.columns and cfl in df_eval.columns:
                    try:
                        fp += int(pd.to_numeric(df_eval[cfp], errors="coerce").fillna(0).sum())
                        fl += int(pd.to_numeric(df_eval[cfl], errors="coerce").fillna(0).sum())
                    except Exception:
                        pass
            if (fp + fl) > 0:
                fora_perto_ratio = float(fp) / float(fp + fl)

            # extrai "quase entram" (números fora-perto) se houver colunas detalhadas
            nums = []
            cols_nums = [c for c in ["fora_perto_nums_1", "fora_perto_nums_2"] if c in df_eval.columns]
            if cols_nums:
                df_tail = df_eval.tail(60).copy()
                for c in cols_nums:
                    for s in df_tail[c].dropna().astype(str).tolist():
                        s = (s or "").strip()
                        if not s:
                            continue
                        arr = []
                        try:
                            arr = json.loads(s)
                        except Exception:
                            # fallback simples: "[1,2,3]" -> split
                            ss = s.strip().lstrip("[").rstrip("]")
                            parts = [p.strip() for p in ss.split(",") if p.strip()]
                            arr = parts
                        if isinstance(arr, (list, tuple)):
                            for v in arr:
                                try:
                                    iv = int(v)
                                    if 1 <= iv <= universo_max:
                                        nums.append(iv)
                                except Exception:
                                    continue
                if nums:
                    vc = pd.Series(nums).value_counts()
                    quase_entram = [int(x) for x in vc.index.tolist()[:60]]

            # critério canônico da Fase 3: trave muito alta (fora-perto dominante) e quase_entram disponível
            if (fora_perto_ratio is not None) and (fora_perto_ratio >= 0.90) and quase_entram:
                fase3_ok = True
    except Exception:
        fase3_ok = False
        quase_entram = []
        fora_perto_ratio = None
# --- 2) Derivação de η (eta): pressão estrutural recente vs longa ---
    # Janela recente: padrão V16 (N=60) — suficiente para captar micro-regime sem virar "curto demais"
    W = 60
    try:
        W = min(W, int(len(df)))
        W = max(12, W)
    except Exception:
        W = 60

    # Frequências (longa e recente)
    def _freq_counts(frame):
        vals = frame[pcols].values.ravel()
        # filtra NaN/None e converte em int quando possível
        out = {}
        for v in vals:
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            if iv <= 0:
                continue
            out[iv] = out.get(iv, 0) + 1
        return out

    freq_long = _freq_counts(df)
    df_recent = df.tail(W)
    freq_recent = _freq_counts(df_recent)

    # Normalizações
    denom_long = max(1, len(df) * len(pcols))
    denom_recent = max(1, W * len(pcols))

    # η: (p_recente - p_longo) em escala leve
    eta = {}
    for x in range(1, universo_max + 1):
        pL = freq_long.get(x, 0) / denom_long
        pR = freq_recent.get(x, 0) / denom_recent
        eta[x] = (pR - pL)

    
    # --- 2.2) FASE 5 (pré‑C4): reforço de TRAVE (sem depender de nocivos) ---
    # Ideia: quando a fresta fica instável, a melhor "âncora leve" é trazer para o pacote
    # os passageiros que aparecem recorrentemente como FORA_PERTO (batendo na trave).
    # Isso NÃO é decisão automática de cravar; é um viés mínimo para reduzir deslocamento estrutural.
    try:
        if df_recent is not None and isinstance(df_recent, pd.DataFrame) and ("fora_perto_nums" in df_recent.columns):
            freq_trave = {}
            for _x in df_recent["fora_perto_nums"].dropna().tolist():
                try:
                    if isinstance(_x, (list, tuple, set)):
                        _lst = list(_x)
                    elif isinstance(_x, str):
                        _lst = json.loads(_x) if _x.strip().startswith("[") else []
                    else:
                        _lst = []
                except Exception:
                    _lst = []
                for n in _lst:
                    try:
                        n = int(n)
                    except Exception:
                        continue
                    if universo_min <= n <= universo_max:
                        freq_trave[n] = freq_trave.get(n, 0) + 1

            if len(freq_trave) > 0:
                denom = float(sum(freq_trave.values())) if sum(freq_trave.values()) > 0 else 1.0
                alpha_trave = 0.18  # ganho pequeno (auditável)
                # soma um bônus na ETA para os passageiros que batem na trave com frequência
                for n, c in freq_trave.items():
                    eta[n] = eta.get(n, 0.0) + alpha_trave * (c / denom)

                # guarda um pool auxiliar (para auditoria e uso nas fases seguintes)
                trave_top = sorted(freq_trave.items(), key=lambda kv: kv[1], reverse=True)[:12]
                st.session_state["bloco_c_trave_pool"] = [int(k) for k, _v in trave_top]
    except Exception:
        pass
# --- 3) Construção do Operador C₁ (troca mínima por lista) ---
    # Define candidatos: os mais pressionados fora do pacote atual (união das listas)
    pacote_atual = set()
    for L in listas:
        if isinstance(L, (list, tuple)):
            for v in L:
                try:
                    pacote_atual.add(int(v))
                except Exception:
                    pass

    # Lista ordenada de "pressão positiva" fora do pacote
    candidatos_out = [x for x in range(1, universo_max + 1) if (x not in pacote_atual) and (x not in nocivos_set)]
    candidatos_out.sort(key=lambda x: eta.get(x, 0.0), reverse=True)

    # BLOCO C (FASE 4) — Estabilizar janela nascente (sustentar fresta sem depender de nocivos)
    # Ideia: quando já houve 4+ na base (any_4p_seen=True) e a evidência não é "evento isolado",
    # fixamos um pequeno "pool de fresta" (números candidatos) por curto período para reduzir volatilidade.
    fase4_ok = False
    fresta_pool = []
    try:
        rate_4p_w = None
        try:
            rate_4p_w = float(stats_janela.get('rate_4p_w', stats_janela.get('rate_4p', 0.0)) or 0.0)
        except Exception:
            rate_4p_w = 0.0

        # Critério canônico Fase 4:
        # - já vimos 4+ em alguma janela na base (any_4p_seen True)
        # - a janela ainda é frágil (curvatura não sustentada) — já filtrado acima por curv_sust
        # - e há evidência mínima de 4+ na janela móvel (rate_4p_w > 0)
        if bool(any_4p_seen) and (rate_4p_w is not None) and (rate_4p_w > 0.0):
            fase4_ok = True

        if fase4_ok:
            # Monta pool: prioriza QUASE_ENTRAM (fora-perto recorrente), depois η fora do pacote
            pool_raw = []
            if quase_entram:
                pool_raw.extend([int(x) for x in quase_entram[:24]])
            pool_raw.extend([int(x) for x in candidatos_out[:24]])

            seen = set()
            for x in pool_raw:
                if x in seen:
                    continue
                if x in nocivos_set:
                    continue
                if x < 1 or x > universo_max:
                    continue
                seen.add(x)
                fresta_pool.append(x)
                if len(fresta_pool) >= 12:
                    break

            # Persistência curta do pool (reduz "pêndulo" entre execuções)
            base_sig = f"{int(universo_max)}|{int(len(df_eval))}|{int(any_4p_seen)}"
            prev = st.session_state.get('bloco_c_fase4_pool', {})
            if isinstance(prev, dict) and prev.get('base_sig') == base_sig and prev.get('pool'):
                fresta_pool = list(prev.get('pool', fresta_pool))
            else:
                st.session_state['bloco_c_fase4_pool'] = {'base_sig': base_sig, 'pool': list(fresta_pool)}

            # Fase 4 é conservadora: não aumenta dose; troca máxima por lista continua = dose (1 ou 2),
            # mas o ganho mínimo é mais permissivo (micro-deslocamento).
            st.session_state['bloco_c_real_diag'].update({
                'fase': 4,
                'motivo': 'fase4_ativa',
                'fresta_pool_n': int(len(fresta_pool)),
            })
    except Exception:
        fase4_ok = False
        fresta_pool = []

    # Diagnóstico para o RF
    top_out = [(x, round(eta.get(x, 0.0), 6)) for x in candidatos_out[:12]]
    top_in = sorted([(x, round(eta.get(x, 0.0), 6)) for x in list(pacote_atual)], key=lambda t: t[1])[:12]

    st.session_state["bloco_c_real_diag"] = {
        "W": W,
        "universo_max": universo_max,
        "nocivos_consistentes": sorted(list(nocivos_set))[:30],
        "nocivos_qtd": int(len(nocivos_set)),
        "top_out_eta": top_out,     # fora do pacote, maior pressão
        "top_in_eta_baixo": top_in, # dentro do pacote, menor pressão
        "pacote_atual_tamanho": len(pacote_atual),
        "fase3_ok": bool(fase3_ok),
        "fase3_fora_perto_ratio": fora_perto_ratio,
        "fase3_quase_entram_top": quase_entram[:25] if isinstance(quase_entram, list) else [],
    }

    # Critério de troca: só troca se melhora estruturalmente (diferença mínima)
    # (isso evita "dançar" listas sem ganho claro)
    MIN_GANHO = 0.0008

    # Frequência global do pacote (para Fase 3: detectar números repetidos que "seguram" a geometria)
    freq_global = {}
    if fase3_ok:
        try:
            for L in listas:
                if isinstance(L, (list, tuple)):
                    for v in L:
                        try:
                            iv = int(v)
                        except Exception:
                            continue
                        freq_global[iv] = freq_global.get(iv, 0) + 1
        except Exception:
            freq_global = {}

    trocas = 0
    trocas_fase6 = 0

    # =========================
    # BLOCO C — FASE 6 (BLOCOC_FASE6_v16h14): Direcionamento Estrutural do Núcleo (DESN)
    # =========================
    # Implementação CANÔNICA (pré-C4, auditável, sem motor novo):
    # - Evento direcional: E = (best_hit >= 4)  [NUNCA >=3]
    # - Peso estrutural: W(p) = α·L4 + β·T - γ·N - λ·A - μ·FREQ
    # - Núcleo direcional: Top-m por W(p) (e quase-núcleo)
    # - Operador Δ: 1 swap controlado por lista (ganho mínimo + freios anti-âncora/anti-rigidez)
    #
    # Gate de ativação: SS_ok && rows_eval>=60 && any_4p_seen==True
    fase6_ok = False
    fase6_params = {"W_DIR": 60, "tau_anc": 0.60, "tau_anc_max": 0.70, "eps": 1.0, "alpha": 1.0, "beta": 0.60, "gamma": 1.0, "lam": 0.40, "mu": 0.30}
    fase6_diag = {"ok": False, "motivo": "N/D", "w_used": 0, "any_4p_seen": False, "col_hit": None, "top_w": []}
    w_dir = {}            # dict: passageiro -> W(p)
    w_rank = []           # lista de passageiros ordenada por W(p) desc
    nocivos_set = set()

    try:
        # nocivos canônicos do painel ANTI-EXATO (se existir)
        _noc = st.session_state.get("anti_exato_nocivos_consistentes", []) or []
        try:
            nocivos_set = set(int(x) for x in _noc)
        except Exception:
            nocivos_set = set()

        ss_info = st.session_state.get("ss_info", {}) or {}
        ss_ok = bool(ss_info.get("atingida", False) or ss_info.get("ss_ok", False) or ss_info.get("ok", False))

        df_eval_local = st.session_state.get("df_eval", None)
        pacotes_reg_local = st.session_state.get("replay_progressivo_pacotes", {}) or {}

        if isinstance(df_eval_local, pd.DataFrame) and not df_eval_local.empty:
            # coluna de hit canônica (preferência: hit_max -> best_hit -> best_acerto_alvo_1)
            col_hit = None
            for c in ("hit_max", "best_hit", "best_acerto_alvo_1"):
                if c in df_eval_local.columns:
                    col_hit = c
                    break
            fase6_diag["col_hit"] = col_hit

            if col_hit is not None and ("k_janela" in df_eval_local.columns):
                W_DIR = int(fase6_params["W_DIR"])
                dfw = df_eval_local.tail(W_DIR).copy()
                w_used = int(len(dfw))
                fase6_diag["w_used"] = w_used

                if w_used >= W_DIR:
                    hitv = dfw[col_hit].fillna(0).astype(int)
                    E = (hitv >= 4)
                    any4 = bool(E.any())
                    fase6_diag["any_4p_seen"] = any4

                    # Gate PRE-4 (canônico): permite o γ atuar antes do 1º 4,
                    # usando SOMENTE métricas já existentes (trave / p3) e mantendo dose contida.
                    pre4_gate = False
                    pre4_motivo = None
                    if ss_ok and (not any4):
                        # p3 na W_DIR (evento menos raro, serve como "sinal fraco" de borda)
                        try:
                            p3_rate_dir = float((hitv == 3).mean())
                        except Exception:
                            p3_rate_dir = 0.0

                        # Proxy de trave na W_DIR: usa fora_perto / fora_longe se existirem no df_eval
                        trv_ratio_dir = None
                        try:
                            fp_col = None
                            for c in ("fora_perto_nums", "fora_perto_list", "fora_perto"):
                                if c in dfw.columns:
                                    fp_col = c
                                    break
                            fl_col = None
                            for c in ("fora_longe_nums", "fora_longe_list", "fora_longe"):
                                if c in dfw.columns:
                                    fl_col = c
                                    break

                            fp_n = 0
                            fl_n = 0
                            if fp_col:
                                for _fp in dfw[fp_col].tolist():
                                    if isinstance(_fp, (list, tuple)):
                                        fp_n += len(_fp)
                            if fl_col:
                                for _fl in dfw[fl_col].tolist():
                                    if isinstance(_fl, (list, tuple)):
                                        fl_n += len(_fl)

                            if (fp_n + fl_n) > 0:
                                trv_ratio_dir = float(fp_n) / float(fp_n + fl_n)
                        except Exception:
                            trv_ratio_dir = None

                        # Critério conservador (pré-4):
                        # - trave domina (>=85%), OU
                        # - existe massa mínima de 3 na W_DIR (>=3%)
                        if (trv_ratio_dir is not None and trv_ratio_dir >= 0.85) or (p3_rate_dir >= 0.03):
                            pre4_gate = True
                            if trv_ratio_dir is not None:
                                pre4_motivo = f"pre4_trave_ratio_dir={trv_ratio_dir:.3f}"
                            else:
                                pre4_motivo = f"pre4_p3_rate_dir={p3_rate_dir:.3f}"

                    fase6_diag["pre4_gate"] = bool(pre4_gate)
                    if pre4_motivo:
                        fase6_diag["pre4_motivo"] = pre4_motivo

                    # Gate canônico: SS_ok + base + (janela real OU pré-4 por trave)
                    if ss_ok and (any4 or pre4_gate):
                        fase6_diag["motivo"] = "gate_any4" if any4 else "gate_pre4_trave"

                        # Sinal B: trave por passageiro (fora_perto_nums)
                        trave_counts = {}
                        trave_total = 0
                        # descobrir coluna com fora_perto nums
                        col_fp = None
                        for c in ("fora_perto_nums", "fora_perto_list", "fora_perto"):
                            if c in dfw.columns:
                                col_fp = c
                                break

                        for _, r in dfw.iterrows():
                            fp = None
                            if col_fp is not None:
                                fp = r.get(col_fp, None)
                            if isinstance(fp, (list, tuple)):
                                nums = []
                                for x in fp:
                                    try:
                                        nums.append(int(x))
                                    except Exception:
                                        pass
                                if nums:
                                    trave_total += 1
                                    for p in set(nums):
                                        trave_counts[p] = trave_counts.get(p, 0) + 1

                        # Sinal A: Lift ≥4 por passageiro (presença no pacote por janela)
                        total_rows = w_used
                        total_E = int(E.sum())

                        pres_rows = {}
                        pres_E = {}

                        # helper para obter pacote por k
                        def _get_pkg_by_k(k):
                            try:
                                if k in pacotes_reg_local:
                                    return pacotes_reg_local.get(k)
                                ks = str(int(k))
                                if ks in pacotes_reg_local:
                                    return pacotes_reg_local.get(ks)
                            except Exception:
                                return None
                            return None

                        # construir presença por linha (união do pacote daquela janela)
                        for idx, r in dfw.iterrows():
                            k = r.get("k_janela", None)
                            try:
                                k = int(k)
                            except Exception:
                                continue
                            pkg = _get_pkg_by_k(k)
                            listas_pkg = None
                            if isinstance(pkg, dict):
                                listas_pkg = pkg.get("listas") or pkg.get("pacote") or pkg.get("listas_pacote")
                            # fallback: se df_eval já tiver universo_pacote, usa como presença (mais fraco, mas válido)
                            if listas_pkg is None:
                                listas_pkg = r.get("listas", None) or r.get("universo_pacote", None) or r.get("uni", None)

                            uni = set()
                            if isinstance(listas_pkg, list):
                                for Lp in listas_pkg:
                                    if isinstance(Lp, (list, tuple)):
                                        for v in Lp:
                                            try:
                                                uni.add(int(v))
                                            except Exception:
                                                pass
                                    else:
                                        try:
                                            uni.add(int(Lp))
                                        except Exception:
                                            pass
                            elif isinstance(listas_pkg, (set, tuple)):
                                for v in listas_pkg:
                                    try:
                                        uni.add(int(v))
                                    except Exception:
                                        pass

                            if not uni:
                                continue

                            isE = bool(E.loc[idx]) if idx in E.index else False
                            for p in uni:
                                pres_rows[p] = pres_rows.get(p, 0) + 1
                                if isE:
                                    pres_E[p] = pres_E.get(p, 0) + 1

                        # construir L4 bruto
                        eps = float(fase6_params["eps"])
                        L4 = {}
                        for p in range(1, int(universo_max) + 1):
                            n1 = int(pres_rows.get(p, 0))
                            e1 = int(pres_E.get(p, 0))
                            n0 = int(total_rows - n1)
                            e0 = int(total_E - e1)
                            # Laplace para estabilidade
                            p1 = (e1 + eps) / (n1 + 2.0 * eps) if (n1 + 2.0 * eps) > 0 else 0.0
                            p0 = (e0 + eps) / (n0 + 2.0 * eps) if (n0 + 2.0 * eps) > 0 else 0.0
                            # evita extremos
                            p1 = min(max(float(p1), 1e-6), 1 - 1e-6)
                            p0 = min(max(float(p0), 1e-6), 1 - 1e-6)
                            L4[p] = float(math.log(p1 / p0))

                        # construir T bruto
                        T = {p: float(trave_counts.get(p, 0)) / float(W_DIR) for p in range(1, int(universo_max) + 1)}

                        # penalidade âncora A e FREQ usando cobertura global do pacote atual (freq_global)
                        denom_cov = float(max(1, int(n_real) * int(M)))  # M listas, cada uma com n_real
                        cover = {p: float(freq_global.get(p, 0)) / denom_cov for p in range(1, int(universo_max) + 1)}
                        tau_anc = float(fase6_params["tau_anc"])
                        A = {}
                        FREQ = {}
                        for p in range(1, int(universo_max) + 1):
                            cp = float(cover.get(p, 0.0))
                            FREQ[p] = cp
                            a = max(0.0, cp - tau_anc)
                            A[p] = float(a / max(1e-6, (1.0 - tau_anc)))  # normaliza para 0..1

                        # Nocivo (binário)
                        N = {p: (1.0 if p in nocivos_set else 0.0) for p in range(1, int(universo_max) + 1)}

                        # normalização min-max (robusta)
                        def _minmax(d):
                            vals = [float(v) for v in d.values()]
                            if not vals:
                                return {k: 0.0 for k in d}
                            vmin = min(vals)
                            vmax = max(vals)
                            if abs(vmax - vmin) < 1e-9:
                                return {k: 0.0 for k in d}
                            return {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}

                        L4n = _minmax(L4)
                        Tn = _minmax(T)
                        An = _minmax(A)
                        FREQn = _minmax(FREQ)
                        # N já é 0/1
                        alpha = float(fase6_params["alpha"])
                        beta = float(fase6_params["beta"])
                        gamma = float(fase6_params["gamma"])
                        lam = float(fase6_params["lam"])
                        mu = float(fase6_params["mu"])

                        for p in range(1, int(universo_max) + 1):
                            w = (alpha * L4n.get(p, 0.0)
                                 + beta * Tn.get(p, 0.0)
                                 - gamma * N.get(p, 0.0)
                                 - lam * An.get(p, 0.0)
                                 - mu * FREQn.get(p, 0.0))
                            w_dir[p] = float(w)

                        # WCALIB — penaliza passageiros nocivos (do ANTI‑EXATO) dentro do W(p), sem decisão (pré‑C4)
                        WCALIB_PENALTY_NOCIVO_BASE = 0.75  # 0..1 (quanto menor, mais punição)
                        # λ* (0..1) modula a aplicação da penalidade (SAFE). λ*=0 → não penaliza; λ*=1 → penalidade plena.
                        try:
                            _lam = float(st.session_state.get("lambda_star", 0.0) or 0.0)
                        except Exception:
                            _lam = 0.0
                        _lam = min(1.0, max(0.0, _lam))
                        WCALIB_PENALTY_NOCIVO = (1.0 - _lam) * 1.0 + _lam * float(WCALIB_PENALTY_NOCIVO_BASE)
                        WCALIB_PENALTY_NOCIVO = min(1.0, max(0.40, WCALIB_PENALTY_NOCIVO))
                        if nocivos_set:
                            for _p in list(w_dir.keys()):
                                if _p in nocivos_set:
                                    w_dir[_p] = float(w_dir.get(_p, 0.0)) * WCALIB_PENALTY_NOCIVO
                            # re‑normaliza para média ~1 (mantém escala comparável)
                            try:
                                _m = sum(float(v) for v in w_dir.values()) / max(1, len(w_dir))
                                if _m > 1e-9:
                                    for _p in list(w_dir.keys()):
                                        w_dir[_p] = float(w_dir[_p]) / _m
                            except Exception:
                                pass

                        w_rank = [p for p, _ in sorted(w_dir.items(), key=lambda kv: kv[1], reverse=True)]
                        fase6_diag["top_w"] = [(int(p), float(w_dir.get(p, 0.0))) for p in w_rank[:15]]

                        fase6_ok = True
                        fase6_diag["ok"] = True
                        fase6_diag["motivo"] = "OK"
                    else:
                        fase6_diag["ok"] = False
                        if not ss_ok:
                            fase6_diag["motivo"] = "SS_NAO_OK"
                        elif not any4:
                            fase6_diag["motivo"] = "SEM_EVIDENCIA_4P"
                        else:
                            fase6_diag["motivo"] = "GATE_FALHOU"
                else:
                    fase6_diag["motivo"] = "BASE_INSUFICIENTE"
            else:
                fase6_diag["motivo"] = "SEM_COLUNAS_CANONICAS"
        else:
            fase6_diag["motivo"] = "SEM_DF_EVAL"
    except Exception:
        fase6_ok = False
        fase6_diag["ok"] = False
        fase6_diag["motivo"] = "ERRO_INTERNO"

    # Persistir diagnóstico (auditável)
    st.session_state["bloco_c_fase6_dir_ok"] = bool(fase6_ok)
    st.session_state["bloco_c_fase6_dir_params"] = dict(fase6_params)
    st.session_state["bloco_c_fase6_dir_diag"] = dict(fase6_diag) if isinstance(fase6_diag, dict) else {}
    st.session_state["bloco_c_fase6_w_dir"] = dict(w_dir) if isinstance(w_dir, dict) else {}
    st.session_state["bloco_c_fase6_w_rank"] = list(w_rank) if isinstance(w_rank, list) else []

    trocas_nocivos = 0


    listas_out = []
    for L in listas:
        if not isinstance(L, (list, tuple)) or len(L) != n_real:
            listas_out.append(L)
            continue

        L_int = []
        ok = True
        for v in L:
            try:
                iv = int(v)
            except Exception:
                ok = False
                break
            if iv < 1 or iv > universo_max:
                ok = False
                break
            L_int.append(iv)
        if not ok:
            listas_out.append(L)
            continue

        # Elemento a retirar / trocar (BLOCO C — Fase 1/2)
        # - Prioridade: remover NOCIVO CONSISTENTE quando presente.
        # - Caso contrário: troca mínima guiada por η (menor pressão estrutural) para deslocamento controlado.
        # - Fase 2: permite até 'dose' trocas por lista (dose canônica 1 ou 2), com freios naturais.
        L_work = list(L_int)
        trocou_nesta_lista = 0

        for _tent in range(int(max(1, dose))):
            if not L_work or len(L_work) != n_real:
                break

            # Escolha do elemento a retirar (cand_in)
            # - Prioridade 1: remover NOCIVO CONSISTENTE quando presente (Fase 1/2)
            # - Fase 3 (sem nocivos): retirar "repetido" (alta frequência global) com baixa pressão (η) para micro‑deslocamento geométrico
            nocivos_na_lista = [v for v in L_work if v in nocivos_set]
            if nocivos_na_lista:
                cand_in = min(nocivos_na_lista, key=lambda x: eta.get(x, 0.0))
            elif fase4_ok and freq_global:
                # FASE 4: retirar "ancora repetida" (freq alta) com baixa pressão (η) — conservador
                cand_in = max(L_work, key=lambda x: (freq_global.get(x, 0), -eta.get(x, 0.0)))
            elif fase3_ok and freq_global:
                cand_in = max(L_work, key=lambda x: (freq_global.get(x, 0), -eta.get(x, 0.0)))
            else:
                cand_in = min(L_work, key=lambda x: eta.get(x, 0.0))
            eta_in = eta.get(cand_in, 0.0)

            # Melhor candidato a entrar (cand_out)
            # - Fase 3: prioriza QUASE_ENTRAM (fora-perto recorrente) para sustentar fresta
            # - Fallback: top η fora do pacote (pressão estrutural)
            cand_out = None
            eta_out = None

            if fase3_ok and quase_entram:
                for x in quase_entram[:80]:
                    try:
                        ix = int(x)
                    except Exception:
                        continue
                    if ix in L_work:
                        continue
                    if ix in nocivos_set:
                        continue
                    cand_out = ix
                    eta_out = eta.get(ix, 0.0)
                    break

            if cand_out is None:
                for x in candidatos_out[:120]:  # busca curta, mas um pouco maior na Fase 2
                    try:
                        ix = int(x)
                    except Exception:
                        continue
                    if ix in L_work:
                        continue
                    if ix in nocivos_set:
                        continue
                    cand_out = ix
                    eta_out = eta.get(ix, 0.0)
                    break

            if cand_out is None:
                break

            # Decide troca (C₁/C₂)
            # Se estamos removendo um NOCIVO CONSISTENTE, aceitamos ganho mínimo mais baixo (ação é "limpeza").
            min_ganho_local = 0.0 if (cand_in in nocivos_set) else (MIN_GANHO * 0.25 if fase4_ok else (MIN_GANHO * 0.5 if fase3_ok else MIN_GANHO))
            if (eta_out - eta_in) >= min_ganho_local:
                L_new = [cand_out if v == cand_in else v for v in L_work]
                if len(set(L_new)) == len(L_new):
                    L_work = list(L_new)
                    trocas += 1
                    trocou_nesta_lista += 1
                    if cand_in in nocivos_set:
                        trocas_nocivos += 1
                    if trocou_nesta_lista >= int(max(1, dose)):
                        break
                else:
                    break
            else:
                break


        # =========================
        # BLOCO C — FASE 6 (Δ estrutural canônico): 1 swap controlado por lista
        # =========================
        if fase6_ok and isinstance(w_dir, dict) and w_dir and isinstance(w_rank, list) and w_rank:
            try:
                tau_anc_max = float(fase6_params.get("tau_anc_max", 0.70))
                denom_cov = float(max(1, int(n_real) * int(M)))
                cover_rate = {p: float(freq_global.get(p, 0)) / denom_cov for p in range(1, int(universo_max) + 1)}

                # delta mínimo (conservador): percentil 60 do |W|
                w_abs = sorted([abs(float(v)) for v in w_dir.values()])
                if w_abs:
                    idxp = int(0.60 * (len(w_abs) - 1))
                    delta_min = float(w_abs[idxp])
                else:
                    delta_min = 0.0

                # p- (mais fraco na lista)
                p_minus = min(L_work, key=lambda x: float(w_dir.get(int(x), 0.0)))
                w_minus = float(w_dir.get(int(p_minus), 0.0))

                p_plus = None
                w_plus = None

                for cand in w_rank:
                    if cand in L_work:
                        continue
                    if cand in nocivos_set:
                        continue
                    if float(cover_rate.get(int(cand), 0.0)) > tau_anc_max:
                        continue
                    p_plus = int(cand)
                    w_plus = float(w_dir.get(int(cand), 0.0))
                    break

                if p_plus is not None and w_plus is not None:
                    dS = float(w_plus - w_minus)
                    if dS >= float(delta_min):
                        # aplica swap
                        try:
                            L_work.remove(int(p_minus))
                        except Exception:
                            pass
                        L_work.append(int(p_plus))
                        trocas += 1
                        trocas_fase6 += 1
            except Exception:
                pass

        listas_out.append(sorted(L_work))

    st.session_state['bloco_c_real_diag'].update({
        'aplicado': (trocas > 0),
        'trocas': trocas,
        'trocas_nocivos': trocas_nocivos,
        'trocas_fase6': trocas_fase6,
        'min_ganho': MIN_GANHO,
    })

    return {
        'aplicado': (trocas > 0),
        'motivo': 'ok' if (trocas > 0) else 'sem_trocas',
        'listas_ajustadas': listas_out,
        'trocas': trocas,
        'trocas_nocivos': trocas_nocivos,
        'trocas_fase6': trocas_fase6,
        'diag_key': 'bloco_c_real_diag',
    }


def v16_sanidade_universo_listas(listas, historico_df):
    """
    Remove / ajusta números fora do universo real observado no histórico.
    Universo é inferido EXCLUSIVAMENTE do histórico carregado.
    """

    if historico_df is None or historico_df.empty:
        return listas  # sem histórico, não mexe

    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    valores = []

    for _, row in historico_df.iterrows():
        for c in col_pass:
            try:
                v = int(row[c])
                if v > 0:
                    valores.append(v)
            except Exception:
                pass

    if not valores:
        return listas

    umin = min(valores)
    umax = max(valores)

    listas_sanas = []

    for lst in listas:
        nova = []
        for v in lst:
            try:
                iv = int(v)
                if iv < umin:
                    iv = umin
                elif iv > umax:
                    iv = umax
                nova.append(iv)
            except Exception:
                pass

        nova = list(dict.fromkeys(nova))  # remove duplicatas mantendo ordem
        listas_sanas.append(nova)

    return listas_sanas


# ============================================================
# >>> INÍCIO — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================

if painel == "🎯 Modo 6 Acertos — Execução":

    st.session_state["v16h57FL_exec_trace"] = []
    st.markdown("## 🎯 Modo 6 Acertos — Execução")

    df = st.session_state.get("historico_df")

        # ------------------------------------------------------------
    # MODO 6 — PUREZA DE CONTEXTO (k não depende da ordem de painéis)
    # ------------------------------------------------------------
    # Garante que k*, NR%, divergência e risco estejam definidos na sessão,
    # mesmo que o operador não tenha aberto Sentinelas/Monitor antes.
    try:
        _ = pc_sentinelas_kstar_silent(df)
        _ = pc_monitor_risco_silent(df)
    except Exception:
        pass

    _kstar_raw = st.session_state.get("sentinela_kstar")
    if isinstance(_kstar_raw, (int, float)):
        k_star = float(_kstar_raw)
    else:
        # fallback neutro (compatível com monitor)
        k_star = 0.25

    nr_pct = st.session_state.get("nr_percent")
    divergencia_s6_mc = st.session_state.get("div_s6_mc")
    risco_composto = st.session_state.get("indice_risco")
    ultima_prev = st.session_state.get("ultima_previsao")


# ------------------------------------------------------------
    # GUARDA — CRITÉRIO MÍNIMO (ORIGINAL PRESERVADO)
    # ------------------------------------------------------------
    pipeline_ok = st.session_state.get("pipeline_flex_ultra_concluido") is True

    # 🧭 Governança automática (pré-C4): status do P1
    p1_auto_status = st.session_state.get("p1_auto_status")
    if isinstance(p1_auto_status, dict) and p1_auto_status.get("eligivel"):
        regra = p1_auto_status.get("regra")
        kref = p1_auto_status.get("k_ref")
        st.success(f"🧭 P1 automático elegível (pré-C4) — regra: {regra} · k_ref={kref}")
        st.caption("Isso NÃO decide ataque e NÃO altera Camada 4 por si só; apenas governa o universo-base usado na geração do pacote.")
    elif isinstance(p1_auto_status, dict) and p1_auto_status.get("motivo"):
        st.info(f"🧭 P1 automático não aplicado — motivo: {p1_auto_status.get('motivo')}")
    turbo_executado_ok = any([
        st.session_state.get("turbo_ultra_executado"),
        st.session_state.get("turbo_executado"),
        st.session_state.get("turbo_ultra_rodou"),
        st.session_state.get("motor_turbo_executado"),
    ])

    if df is None or df.empty or not pipeline_ok:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "É necessário:\n- Histórico carregado\n- Pipeline V14-FLEX ULTRA executado\n\nℹ️ O TURBO++ é opcional para o Modo 6. Você pode rodar o TURBO antes para tentar núcleo ofensivo, mas o pacote base do Modo 6 independe disso.",
            tipo="warning"
        )
        st.stop()

    # ------------------------------------------------------------
    # AJUSTE DE AMBIENTE (PRÉ-ECO) — ORIGINAL
    # ------------------------------------------------------------
    config = ajustar_ambiente_modo6(
        df=df,
        k_star=k_star,
        nr_pct=nr_pct,
        divergencia_s6_mc=divergencia_s6_mc,
        risco_composto=risco_composto,
        previsibilidade="alta",
    )

    st.caption(config["aviso_curto"] + " | PRÉ-ECO técnico ativo")

    volume = int(config["volume_recomendado"])
    volume = max(1, min(volume, int(config["volume_max"])))


    # ------------------------------------------------------------
    # DETECÇÃO DO FENÔMENO (n + UNIVERSO REAL)
    # ------------------------------------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    universo_tmp = []
    contagens = []

    for _, row in df.iterrows():
        vals = [int(row[c]) for c in col_pass if pd.notna(row[c])]
        if vals:
            contagens.append(len(vals))
            universo_tmp.extend(vals)

    if not contagens or not universo_tmp:
        st.warning("Não foi possível detectar universo válido no histórico.")
        st.stop()

    n_real = int(pd.Series(contagens).mode().iloc[0])
    st.session_state["n_alvo"] = n_real

    universo = sorted({v for v in universo_tmp if v > 0})
    umin, umax = min(universo), max(universo)   # EX: 1–50 (REAL)


    # ------------------------------------------------------------
    # REPRODUTIBILIDADE (ORIGINAL)
    # ------------------------------------------------------------
    seed = pc_stable_seed(f"PC-M6-{len(df)}-{n_real}-{umin}-{umax}")
    rng = np.random.default_rng(seed)


    # ------------------------------------------------------------
    # MAPA DE ÍNDICES (CORREÇÃO ESTRUTURAL)
    # ------------------------------------------------------------
    universo_idx = list(range(len(universo)))
    valor_por_idx = {i: universo[i] for i in universo_idx}
    idx_por_valor = {v: i for i, v in valor_por_idx.items()}


    def ajustar_para_n(lista):
        out_idx = []

        for v in lista:
            if v in idx_por_valor:
                idx = idx_por_valor[v]
                if idx not in out_idx:
                    out_idx.append(idx)

        while len(out_idx) < n_real:
            cand = rng.choice(universo_idx_use)
            if cand not in out_idx:
                out_idx.append(cand)

        return out_idx[:n_real]



    # ------------------------------------------------------------
    # 🧭 P1 AUTOMÁTICO (pré-C4) — REGRA PLANO/RUIM (DEFENSIVA)
    # - Governado pela Parabólica (via CAP)
    # - Auditável
    # - Não toca Camada 4
    # ------------------------------------------------------------
    try:
        df_full_for_gov = st.session_state.get("_df_full_safe") if st.session_state.get("_df_full_safe") is not None else st.session_state.get("historico_df")
        snaps_map_for_gov = st.session_state.get("snapshot_p0_canonic") or {}
        k_ref = int(st.session_state.get("replay_janela_k_active", len(df)))
        decisao_p1 = _p1_auto_decidir(df_full_for_gov, snaps_map_for_gov, k_ref) if df_full_for_gov is not None else {"eligivel": False, "motivo": "df_full_ausente"}
    except Exception as _e:
        decisao_p1 = {"eligivel": False, "motivo": f"erro_decisao_p1:{_e}"}

    universo_idx_use = universo_idx  # default (sem P1)
    if isinstance(decisao_p1, dict) and decisao_p1.get("eligivel"):
        ub = decisao_p1.get("ub") or []
        foco = sorted({int(v) for v in ub if umin <= int(v) <= umax})
        foco_set = set(foco)

        universo_idx_foco = [i for i, v in enumerate(universo) if int(v) in foco_set]

        # guarda auditável
        st.session_state["p1_auto_status"] = {
            "eligivel": True,
            "regra": decisao_p1.get("motivo"),
            "k_ref": k_ref,
            "estado_global": decisao_p1.get("estado_global"),
            "adds_B": decisao_p1.get("adds_B") or [],
            "ub_len": len(foco),
        }
        st.session_state["p1_universo_ativo"] = foco

        # se o foco ficou pequeno demais, não força viés (fallback seguro)
        if len(universo_idx_foco) >= max(2 * n_real, 10):
            universo_idx_use = universo_idx_foco
        else:
            st.session_state["p1_auto_status"]["fallback"] = "foco_insuficiente"
    else:
        st.session_state["p1_auto_status"] = {
            "eligivel": False,
            "motivo": (decisao_p1 or {}).get("motivo") if isinstance(decisao_p1, dict) else "sem_decisao",
        }
        st.session_state["p1_universo_ativo"] = []

    # ------------------------------------------------------------
    # BASE ULTRA (ORIGINAL, MAS EM ÍNDICES)
    # + V16h51: calibração leve (pré-C4) baseada em concentração (somente leitura)
    #   - não cria painel
    #   - não toca Camada 4
    # ------------------------------------------------------------
    calib_leve = v16_calib_leve_computar_da_concentracao(force_recompute=False)

    # audit: registra calibração leve calculada (somente leitura)
    try:
        st.session_state["v16_calib_leve_last"] = calib_leve
        st.session_state["v16_calib_leve_last_ts"] = datetime.now().isoformat(timespec="seconds")
    except Exception:
        pass

    # ------------------------------------------------------------
    # AUDIT — calibração leve aplicada?
    # ------------------------------------------------------------
    calib_aplicada = False  # audit: default (será marcado True se influenciar o pacote)
    try:
        _c = calib_leve if isinstance(calib_leve, dict) else {}
        _ap = bool(_c.get("applied", False))
        st.session_state["v16_calib_leve_last_summary"] = {
            "active": bool(_c.get("active", False)),
            "applied": _ap,
            "I_mean": float(_c.get("I_mean", _c.get("I", 0.0)) or 0.0),
            "I_max": float(_c.get("I_max", _c.get("I", 0.0)) or 0.0),
            "I": float(_c.get("I", 0.0) or 0.0),
            "n_from_top": int(_c.get("n_from_top", 0) or 0),
            "noise_amp": int(_c.get("noise_amp", 0) or 0),
            "wr": int(_c.get("wr", 0) or 0),
            "aplicada_no_pacote": False,  # será atualizado mais abaixo quando aplicarmos de fato
            "reason": str(_c.get("reason", "")) if isinstance(_c.get("reason", ""), str) else "",
        }
    except Exception:
        pass


    if ultima_prev:
        base_vals = ultima_prev if isinstance(ultima_prev[0], int) else ultima_prev[0]
        base_idx = ajustar_para_n(base_vals)
    else:
        # default: universo completo / foco P1 (já calculado em universo_idx_use)
        calib_aplicada = False  # audit: indica se a calibração leve influenciou este pacote
        if isinstance(calib_leve, dict) and calib_leve.get("active") and isinstance(calib_leve.get("top_pool"), list) and calib_leve["top_pool"]:
            try:
                n_top = int(calib_leve.get("n_from_top", 0) or 0)
                n_top = max(1, min(int(n_real), n_top))
                # converte top_pool (valores) para idx dentro do universo atual
                top_vals = [int(v) for v in calib_leve["top_pool"] if int(universo_min) <= int(v) <= int(universo_max)]
                top_idx = [idx_por_valor.get(int(v), None) for v in top_vals]
                top_idx = [int(ix) for ix in top_idx if ix is not None]
                top_idx = [ix for ix in top_idx if ix in set(universo_idx_use)]
                top_idx = sorted(set(top_idx))
                if len(top_idx) >= n_top:
                    escolhe_top = rng.choice(top_idx, size=n_top, replace=False).tolist()
                else:
                    escolhe_top = top_idx[:]  # pode ser menor; completa abaixo
                # completa com o restante do pool (sem colisão)
                restante = [ix for ix in universo_idx_use if ix not in set(escolhe_top)]
                n_rest = int(max(0, int(n_real) - len(escolhe_top)))
                escolhe_rest = rng.choice(restante, size=n_rest, replace=False).tolist() if (n_rest > 0 and len(restante) >= n_rest) else []
                base_idx = (escolhe_top + escolhe_rest)
                calib_aplicada = True
                try:
                    if isinstance(st.session_state.get('v16_calib_leve_last_summary'), dict):
                        st.session_state['v16_calib_leve_last_summary']['aplicada_no_pacote'] = True
                except Exception:
                    pass
                # sanidade: se algo falhar, cai no fallback
                if len(base_idx) != int(n_real):
                    raise ValueError("base_idx_len_invalida")
            except Exception:
                base_idx = rng.choice(universo_idx_use, size=n_real, replace=False).tolist()
        else:
            base_idx = rng.choice(universo_idx_use, size=n_real, replace=False).tolist()




    # ------------------------------------------------------------
    # Pool de índices (foco P1, se aplicável)
    # ------------------------------------------------------------
    pool_idx = universo_idx  # default: universo completo (contíguo)
    pool_mode = "full"
    inv_pos = None

    try:
        if isinstance(universo_idx_use, list) and universo_idx_use != universo_idx:
            pool_idx = list(universo_idx_use)  # subset ordenado
            pool_mode = "foco_p1"
            inv_pos = {int(ix): j for j, ix in enumerate(pool_idx)}
    except Exception:
        pool_idx = universo_idx
        pool_mode = "full"
        inv_pos = None

    # ------------------------------------------------------------
    # GERAÇÃO PRÉ-ECO (SEM POSSIBILIDADE DE SAIR DO UNIVERSO)
    # ------------------------------------------------------------
    listas_brutas = []

    for _ in range(volume):
        # V16h51: dispersão (ruído) pode ser levemente comprimida quando calibração leve estiver ativa
        try:
            amp = int((st.session_state.get("calib_leve_v16") or {}).get("noise_amp", 3))
            amp = max(1, min(3, amp))
        except Exception:
            amp = 3
        ruido = rng.integers(-amp, amp + 1, size=n_real)  # deslocamento leve
        if pool_mode == "foco_p1" and inv_pos is not None:
            # desloca dentro do pool focado (mantém coerência defensiva)
            nova_idx = []
            for idx, r in zip(base_idx, ruido):
                pos = inv_pos.get(int(idx), None)
                if pos is None:
                    # fallback seguro: usa o próprio idx
                    nova_idx.append(max(0, min(len(universo) - 1, int(idx))))
                else:
                    new_pos = max(0, min(len(pool_idx) - 1, int(pos) + int(r)))
                    nova_idx.append(int(pool_idx[new_pos]))
        else:
            nova_idx = [
                max(0, min(len(universo_idx) - 1, idx + r))
                for idx, r in zip(base_idx, ruido)
            ]
        nova = [valor_por_idx[i] for i in nova_idx]
        listas_brutas.append(nova)


    # ------------------------------------------------------------
    # 🔒 FILTRO FINAL DE DOMÍNIO (ANTI-RESÍDUO)  ← CORREÇÃO
    # ------------------------------------------------------------
    listas_filtradas = []
    descartadas = 0

    for lista in listas_brutas:
        if all(umin <= int(v) <= umax for v in lista):
            listas_filtradas.append(lista)
        else:
            descartadas += 1

    if descartadas > 0:
        st.warning(
            f"⚠️ {descartadas} lista(s) descartada(s) por violar o domínio "
            f"dos passageiros ({umin}–{umax})."
        )

    listas_brutas = listas_filtradas

    # ------------------------------------------------------------
    # v16h57FJ — CT no fluxo real, antes da sanidade, sem calib_meta
    # ------------------------------------------------------------
    _ranking_vals_dx = []
    if "ranking2" in locals() and ranking2 is not None:
        try:
            _ranking_vals_dx = [int(v) for v in list(ranking2)]
        except Exception:
            _ranking_vals_dx = []
    if (not _ranking_vals_dx) and "ranking" in locals() and ranking is not None:
        try:
            _ranking_vals_dx = [int(v) for v in list(ranking)]
        except Exception:
            _ranking_vals_dx = []
    if (not _ranking_vals_dx) and "top_nums" in locals() and top_nums is not None:
        try:
            _ranking_vals_dx = [int(v) for v in list(top_nums)]
        except Exception:
            _ranking_vals_dx = []

    listas_brutas, _npgen_dx_info = pc_v16_new_packet_generator(
        listas_brutas,
        ranking_vals=_ranking_vals_dx,
        historico_df=df,
        n_alvo=n_real,
        seed=seed,
        max_lists=len(listas_brutas),
    )
    try:
        calib_meta["new_packet_generator"] = dict(_npgen_dx_info or {})
    except Exception:
        pass
    st.session_state["v16_ct_last_real_generator"] = dict(_npgen_dx_info or {})
    try:
        pc_trace_store("pc_trace_after_npg_dx", listas_brutas, "1.9) PRE SANIDADE CT EM LISTAS_FILTRADAS")
    except Exception:
        pass

    # ------------------------------------------------------------
    # SANIDADE FINAL — SOMENTE ESTRUTURAL (ORIGINAL)
    # ------------------------------------------------------------
    listas_totais = sanidade_final_listas(listas_brutas)

    # ------------------------------------------------------------
    # FIX6 TAILSTAB — garante mínimo determinístico de 10 listas totais
    # (sem opções; pré-C4; não altera Camada 4)
    # ------------------------------------------------------------
    try:
        target_n = 10
        n_carro = int(n_real) if isinstance(n_real, (int, float)) else 6
        # universo candidato: união do pacote; fallback no universo completo
        universo_cand = sorted({int(p) for lst in listas_totais for p in lst})
        if len(universo_cand) < n_carro:
            universo_cand = list(range(int(universo_min), int(universo_max) + 1))
        seed_fill = pc_stable_seed("MODO6_FILL", k, universo_min, universo_max, n_carro)
        listas_totais = pc_fill_lists_to_target(
            listas_totais,
            target_n=target_n,
            universe_candidates=universo_cand,
            n_por_lista=n_carro,
            seed=seed_fill,
        )
    except Exception:
        # fallback silencioso (não quebra execução)
        pass

    listas_top10 = listas_totais[:10]

    # ============================================================
    # Órbita (E1) + Gradiente + N_EXTRA
    # (sem interceptação automática; não divide pacote)
    # ============================================================
    try:
        info_orbita = v16_calcular_orbita_pacote(listas_top10, universo_min, universo_max)
        ginfo = v16_calcular_gradiente_E1(info_orbita)
        gradiente = ginfo.get("gradiente", "G0")
        orb_score = ginfo.get("score", 0.0)
    
        # ECO (se existir no estado)
        eco_forca = st.session_state.get("eco_forca", None)
        eco_acion = st.session_state.get("eco_acionabilidade", None)
    
        n_base = len(listas_totais)
    
        # memória para E2 (repetição consecutiva de quase-órbita)
        prev_estado = st.session_state.get("orbita_prev_estado", "E0")
        prev_ancoras = st.session_state.get("orbita_prev_ancoras", [])
        if info_orbita.get("estado") == "E1":
            overlap = 0
            if prev_ancoras and info_orbita.get("ancoras"):
                overlap = len(set(prev_ancoras).intersection(set(info_orbita.get("ancoras"))))
            if prev_estado == "E1" and overlap >= 2:
                info_orbita["estado"] = "E2"
                info_orbita["selo"] = "E2"
    
        st.session_state["orbita_prev_estado"] = info_orbita.get("estado")
        st.session_state["orbita_prev_ancoras"] = info_orbita.get("ancoras", [])
    
        n_extra = v16_calcular_N_extra(info_orbita.get("estado"), gradiente, n_base, eco_forca, eco_acion)
    
        # gera listas extras (se justificável) — não substitui as Top10, só expande
        listas_extra = []
        if n_extra and n_extra > 0:
            listas_extra = v16_gerar_listas_extra_por_orbita(
                info_orbita,
                universo_min=universo_min,
                universo_max=universo_max,
                n_carro=n_real,
                qtd=n_extra,
                seed=st.session_state.get("serie_base_idx", 0),
            )
    
        # listas de interceptação automática (somente em E2) — muda listas de verdade
        listas_intercept = []
        if info_orbita.get("estado") == "E2":
            base_i = 2
            if info_orbita.get("gradiente") in ("G2", "G3"):
                base_i = 3
            qtd_i = min(8, max(2, (n_extra or 0) + base_i))
            listas_intercept = v16_gerar_listas_interceptacao_orbita(
                info_orbita,
                universo_min=universo_min,
                universo_max=universo_max,
                n_carro=n_real,
                qtd=qtd_i,
                seed=st.session_state.get("serie_base_idx", 0),
            )

        if listas_intercept:
            st.session_state["listas_intercept_orbita"] = listas_intercept
            listas_totais = listas_totais + listas_intercept

        if listas_extra:
            listas_totais = listas_totais + listas_extra
            try:
                listas_totais = v16_priorizar_listas_por_contexto(
                    listas_totais,
                    estado_obj=st.session_state.get("estado_alvo_v16"),
                    k_star=st.session_state.get("k_star", None),
                )
            except Exception:
                pass
            listas_top10 = listas_totais[:10]
    
        # registro em sessão (para Relatório Final / Bala Humano)
        st.session_state["orbita_info"] = info_orbita
        st.session_state["orbita_gradiente"] = gradiente
        st.session_state["orbita_score"] = orb_score
        st.session_state["modo6_n_base"] = int(n_base)
        st.session_state["modo6_n_extra"] = int(n_extra)
        st.session_state["modo6_n_total"] = int(len(listas_totais))
    except Exception:
        st.session_state["orbita_info"] = {"estado": "E0", "selo": "E0"}
        st.session_state["orbita_gradiente"] = "G0"
        st.session_state["orbita_score"] = 0.0
        st.session_state["modo6_n_base"] = int(len(listas_totais))
        st.session_state["modo6_n_extra"] = 0
        st.session_state["modo6_n_total"] = int(len(listas_totais))


    # ============================================================
    # BLOCO C (V10) — Ajuste Fino Numérico (miolo/coerência)
    # Observacional, pré‑Camada 4.
    # Usa V8 (borda qualificada) como mapa e V9 (memória) como lastro, se existir.
    # ============================================================
    try:
        # ------------------------------------------------------------
        # CAP Invisível (V1) — captura do pacote A (pré-BLOCO C)
        # Necessário para o P1 A/B. Deve acontecer SEMPRE antes do V10.
        # ------------------------------------------------------------
        try:
            _base_cap = listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais
            st.session_state["pacote_pre_bloco_c"] = [list(x) for x in _base_cap] if isinstance(_base_cap, list) else []
            st.session_state["pacote_pre_bloco_c_origem"] = "CAP Invisível (V1) — Modo 6 (pré-BLOCO C)"
        except Exception:
            pass

        _v8_info = st.session_state.get("v8_borda_qualificada_info", None)
        _v9_info = st.session_state.get("v9_memoria_borda", None)

        _c_out = v10_bloco_c_aplicar_ajuste_fino_numerico(
            listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais,
            n_real=n_real,
            v8_borda_info=_v8_info,
            v9_memoria_info=_v9_info,
        )

        # aplica apenas sobre o pacote que será exibido como Top10 / pacote atual
        if _c_out.get("aplicado"):
            _aj = _c_out.get("listas_ajustadas", [])
            # se estávamos trabalhando em Top10, mantém Top10 ajustado
            if isinstance(listas_top10, list) and len(listas_top10) > 0:
                listas_top10 = _aj
            else:
                listas_totais = _aj
                listas_top10 = listas_totais[:10]

        st.session_state["bloco_c_info"] = {
            "aplicado": bool(_c_out.get("aplicado")),
            "trocas": int(_c_out.get("trocas", 0)),
            "motivo": str(_c_out.get("motivo", "")),
        }
    except Exception as _e:
        st.session_state["bloco_c_info"] = {"aplicado": False, "trocas": 0, "motivo": f"Falha no BLOCO C: {_e}"}
    # ------------------------------------------------------------
    # 🧭 POSTURA OPERACIONAL (ESTÁVEL / RESPIRÁVEL / RUPTURA)
    # ------------------------------------------------------------
    # Usa apenas sinais já existentes (pipeline/k*/NR/div). Pré‑C4.
    pipeline_regime = st.session_state.get("pipeline_regime") or st.session_state.get("pipeline_regime_detectado") or st.session_state.get("pipeline_regime_label") or ""
    # Postura (DIAGNÓSTICO): pode usar k* (sensor externo) — não decide nada.
    postura_diag = pc_classificar_postura(
        pipeline_regime=pipeline_regime,
        k_star=st.session_state.get("sentinela_kstar"),
        nr_percent=st.session_state.get("nr_percent"),
        div_s6_mc=st.session_state.get("div_s6_mc"),
    )

    # Postura (MOTOR): isolada de k/k* — esta é a que pode influenciar execução/listas.
    postura_motor = pc_classificar_postura_motor(
        pipeline_regime=pipeline_regime,
        nr_percent=st.session_state.get("nr_percent"),
        div_s6_mc=st.session_state.get("div_s6_mc"),
    )

    # Governança dura: decisões operacionais usam SOMENTE postura_motor.
    postura = postura_motor

    st.session_state["postura_estado"] = postura_motor
    st.session_state["postura_diag"] = postura_diag
    st.session_state["postura_motor"] = postura_motor

    # Em RESPIRÁVEL: aplicar elasticidade mínima no pacote (sem tocar Camada 4)
    # Regra canônica (continuidade): Memória Estrutural só entra quando o cenário está SEM_RITMO.
    ritmo_info = st.session_state.get("ritmo_danca_info") or {}
    try:
        ritmo_global = str((ritmo_info or {}).get("ritmo_global") or "N/D").strip()
    except Exception:
        ritmo_global = "N/D"
    st.session_state["ritmo_global_expost"] = ritmo_global  # auditável (não decide)

    if postura == "RESPIRÁVEL":
        # Memória Estrutural do RESPIRÁVEL (SEM_RITMO) — usa snapshots já registrados; pré-C4; auditável; reversível
        if ritmo_global == "SEM_RITMO":
            # 🧠 Memória Estrutural (SEM_RITMO) — usa cache em sessão, atualizado ao registrar Snapshot P0
            try:
                _me_enabled = bool(st.session_state.get("me_enabled", True))
                _me_info = st.session_state.get("me_info")
                if not isinstance(_me_info, dict) or not _me_info:
                    _me_info = v16_me_build_from_snapshots(st.session_state.get("snapshot_p0_canonic") or {})
                    st.session_state["me_info"] = _me_info

                _ss_info = st.session_state.get("ss_info")
                _me_status_info = v16_me_status(
                    postura=postura,
                    ritmo_global=ritmo_global,
                    me_enabled=_me_enabled,
                    ss_info=_ss_info if isinstance(_ss_info, dict) else None,
                    me_info=_me_info,
                )
                st.session_state["me_status"] = _me_status_info.get("status")
                st.session_state["me_status_info"] = _me_status_info

                _mem = {
                    "ok": bool(_me_status_info.get("status") == "ATIVA"),
                    "sufocadores": list((_me_info or {}).get("sufocadores") or []),
                    "stats": {"base": (_me_info or {}).get("base"), "debug": (_me_info or {}).get("debug")},
                    "motivo": _me_status_info.get("motivo", ""),
                    "status": _me_status_info.get("status", ""),
                }
            except Exception as e:
                _mem = {"ok": False, "sufocadores": [], "stats": {}, "motivo": f"falha_memoria: {e}", "status": "FALHA"}
        else:
            _mem = {"ok": False, "sufocadores": [], "stats": {}, "motivo": f"memoria_desligada_por_ritmo_{ritmo_global}", "status": "INATIVA"}

        st.session_state["postura_respiravel_memoria"] = _mem

        # Aplicação da elasticidade mínima:
        # - Anti-clone/anti-core sempre (RESPIRÁVEL)
        # - Neutralização leve por sufocadores apenas quando _mem.ok em SEM_RITMO
        memoria_suf = (_mem or {}).get("sufocadores") if bool((_mem or {}).get("ok")) else None

        listas_totais, listas_top10, _resp_info = pc_resp_aplicar_diversificacao(
            listas_totais=listas_totais,
            listas_top10=listas_top10,
            universo=universo,
            seed=st.session_state.get("serie_base_idx", 0),
            n_alvo=int(st.session_state.get("n_alvo", 6) or 6),
            memoria_sufocadores=memoria_suf,
        )
        st.session_state["postura_respiravel_info"] = _resp_info
    else:
        st.session_state["postura_respiravel_memoria"] = {"ok": False, "sufocadores": [], "stats": {}, "motivo": "postura_nao_respiravel"}
        st.session_state["postura_respiravel_info"] = {"aplicado": False, "motivo": "postura_nao_respiravel", "postura": postura}

    # Exibição (informativa): não decide ataque, só descreve postura de execução
    if postura == "RESPIRÁVEL":
        st.warning("🟠 Postura: RESPIRÁVEL (P0 com elasticidade mínima anti-compressão) — pré-C4")

        # Ritmo/Dança (ex-post) — informativo (pré-C4)
        try:
            st.caption(f"🕺 Ritmo/Dança (ex-post): {ritmo_global}")
        except Exception:
            pass

        # 🧠 Memória Estrutural (SEM_RITMO) — auditável (somente quando SEM_RITMO)
        try:
            if str(ritmo_global) == "SEM_RITMO":
                mem = st.session_state.get("postura_respiravel_memoria") or {}
                with st.expander("🧠 Memória Estrutural (SEM_RITMO) — auditável", expanded=False):
                    if isinstance(mem, dict) and mem.get("ok"):
                        suf = mem.get("sufocadores") or []
                        if suf:
                            st.markdown("**Sufocadores (top):** " + ", ".join([str(x) for x in suf]))
                        else:
                            st.info("Sem sufocadores suficientes ainda (base pequena).")
                        st.json(mem.get("stats") or {})
                        st.caption(f"Motivo: {mem.get('motivo')}")
                    else:
                        motivo = mem.get("motivo") if isinstance(mem, dict) else "mem_invalida"
                        st.info(f"Memória estrutural ainda indisponível: {motivo}")
            else:
                st.caption("🧠 Memória Estrutural (SEM_RITMO) desligada (ritmo_global != SEM_RITMO).")
        except Exception:
            pass

    elif postura == "RUPTURA":
        st.error("🔴 Postura: RUPTURA (P0 conservador; sem agressividade) — pré-C4")
    else:
        st.success("🟢 Postura: ESTÁVEL (execução normal do P0)")

    st.session_state["modo6_listas_totais"] = listas_totais
    st.session_state["modo6_listas_top10"] = listas_top10
    st.session_state["modo6_listas"] = listas_totais
    st.session_state["pc_force_fresh_packet_active"] = True
    st.session_state["pc_force_fresh_packet_hash"] = hash(str(listas_totais)) if isinstance(listas_totais, list) else None
    st.caption("🧷 Prioridade do gerador aplicada: auto-tune de coesão ativo no pacote fresco desta execução.")

    # ------------------------------------------------------------
    # REGISTRO AUTOMÁTICO DO PACOTE ATUAL (Backtest Rápido N=60)
    # ------------------------------------------------------------
    # Regra: não decide ação e não muda geração.
    # Apenas "congela" qual pacote está ativo para o painel de Backtest.
    # Preferência: Top10 (priorizadas) quando existir; senão, usa o total.
    try:
        _pacote_bt = listas_top10 if (isinstance(listas_top10, list) and len(listas_top10) > 0) else listas_totais
        st.session_state["pacote_listas_atual"] = _pacote_bt
        st.session_state["pacote_listas_origem"] = "Modo 6 (Top10)" if _pacote_bt is listas_top10 else "Modo 6 (Total)"

        # ------------------------------------------------------------
        # 🧊 CAP INVISÍVEL (V0) — AUTO-REGISTRO DO SNAPSHOT P0
        # ------------------------------------------------------------
        # Regra: sempre que o Modo 6 gera/congela um pacote, registramos o Snapshot P0 canônico
        # da janela ativa automaticamente (sem exigir clique no Replay Progressivo).
        try:
            _k_reg_auto = int(st.session_state.get("replay_janela_k_active", len(df)))
        except Exception:
            _k_reg_auto = int(len(df))
        try:
            _umin_auto = int(st.session_state.get("universo_min", 1) or 1)
            _umax_auto = int(st.session_state.get("universo_max", 60) or 60)
        except Exception:
            _umin_auto, _umax_auto = 1, 60

        try:
            pc_snapshot_p0_autoregistrar(_pacote_bt, k_reg=_k_reg_auto, universo_min=_umin_auto, universo_max=_umax_auto)
        except Exception:
            pass
    except Exception:
        # Falha silenciosa: não deve travar a execução do Modo 6.
        pass


    st.success(
        f"Modo 6 (PRÉ-ECO | n-base={n_real}) — "
        f"{len(listas_totais)} listas totais | "
        f"{len(listas_top10)} priorizadas (Top 10)."
    )
# ============================================================
# <<< FIM — BLOCO DO PAINEL 6 — MODO 6 ACERTOS (PRÉ-ECO)
# ============================================================


    # ✅ Snapshot canônico (para Relatório Final / Diagnóstico Espelho)
    try:
        st.session_state["modo6_executado"] = True
        st.session_state["listas_geradas"] = int(len(listas_top10) if isinstance(listas_top10, list) else len(listas_totais))
    except Exception:
        pass




# ============================================================
# 🧪 Modo N Experimental (n≠6)
# (LAUDO DE CÓDIGO — FASE 1 / BLOCO 2)
#
# OBJETIVO:
# - Roteamento mínimo + guardas explícitas
# - Avisos claros de EXPERIMENTAL
# - ZERO lógica de geração
#
# BLINDAGEM:
# - NÃO altera Modo 6
# - NÃO altera TURBO
# - NÃO altera ECO/PRÉ-ECO
# - NÃO escreve em session_state (somente leitura)
# ============================================================

elif painel == "🧪 Modo N Experimental (n≠6)":

    st.header("🧪 Modo N Experimental (n≠6)")
    st.warning(
        "EXPERIMENTAL — Este painel é isolado. "
        "Não substitui o Modo 6, não altera TURBO, "
        "não aprende e pode recusar geração."
    )

    # ------------------------------
    # Guardas canônicas (EVIDÊNCIA REAL)
    # ------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")
    k_calculado = st.session_state.get("k_calculado") or st.session_state.get("k_star")

    # Evidências indiretas do pipeline (como ele REALMENTE funciona)
    estrada_regime = st.session_state.get("estrada_regime")
    energia_media = st.session_state.get("energia_media")
    clusters_formados = st.session_state.get("clusters_formados")

    # Guarda 1 — histórico
    if historico_df is None or historico_df.empty:
        st.error("Pré-requisito ausente: histórico não carregado.")
        st.stop()

    # Guarda 2 — n_alvo válido e diferente de 6
    try:
        n_int = int(n_alvo)
    except Exception:
        st.error("Pré-requisito ausente: n_alvo inválido.")
        st.stop()

    if n_int == 6:
        st.info("Este painel é exclusivo para n≠6. Para n=6, utilize o Modo 6.")
        st.stop()

    # Guarda 3 — pipeline (por evidência observada)
    if estrada_regime is None and energia_media is None and clusters_formados is None:
        st.error("Pré-requisito ausente: Pipeline V14-FLEX ULTRA não concluído.")
        st.stop()

    # Guarda 4 — sentinelas
    if k_calculado is None:
        st.error("Pré-requisito ausente: Sentinelas (k/k*) não calculadas.")
        st.stop()

    # ------------------------------
    # Estado observado (laudo)
    # ------------------------------
    st.subheader("📋 Estado Observado (Laudo)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("n do Fenômeno", n_int)
    with col2:
        st.metric("Séries", len(historico_df))
    with col3:
        st.metric("Pipeline", "CONCLUÍDO (por evidência)")
    with col4:
        st.metric(
            "Sentinela",
            f"{k_calculado:.4f}" if isinstance(k_calculado, (int, float)) else str(k_calculado),
        )

    st.markdown("---")

    st.info(
        "Este é o **BLOCO 2 (Laudo de Código)**.\n\n"
        "➡️ Nenhuma lista é gerada aqui.\n"
        "➡️ O próximo bloco ativará a lógica EXPERIMENTAL de geração, "
        "usando U2/U3/U4 como autorizadores."
    )



# ============================================================
# 🧪 Modo N Experimental (n≠6)
# BLOCO 3 — GERAÇÃO MÍNIMA EXPERIMENTAL (n=5)
# ============================================================

    st.markdown("### 🔬 Geração Experimental — n≠6")
    st.caption(
        "Modo EXPERIMENTAL. Geração mínima, consciente e auditável. "
        "Não substitui o Modo 6."
    )

    # ------------------------------------------------------------
    # Fonte canônica do pacote (somente leitura)
    # ------------------------------------------------------------
    listas_base = st.session_state.get("modo6_listas_totais", [])

    # ------------------------------------------------------------
    # Autorizadores (MVP-U2 / U3 / U4)
    # ------------------------------------------------------------
    autorizacao = {
        "orcamento_ok": False,
        "cobertura_ok": False,
        "eficiencia_ok": False,
    }

    # Autorização mínima por orçamento (U2)
    orcamento_manual = st.session_state.get("orcamento_manual_universal")
    if isinstance(orcamento_manual, (int, float)) and orcamento_manual > 0:
        autorizacao["orcamento_ok"] = True

    # Autorização mínima por cobertura (U3)
    if listas_base and len(listas_base) >= 1:
        autorizacao["cobertura_ok"] = True

    # Autorização mínima por eficiência (U4)
    # (critério mínimo: ao menos 1 lista viável)
    if autorizacao["orcamento_ok"] and autorizacao["cobertura_ok"]:
        autorizacao["eficiencia_ok"] = True

    # ------------------------------------------------------------
    # Decisão EXPERIMENTAL (sem fallback)
    # ------------------------------------------------------------
    if not all(autorizacao.values()):
        st.warning(
            "Geração NÃO autorizada pelos MVPs (U2/U3/U4).\n\n"
            "➡️ Resultado válido.\n"
            "➡️ Nenhuma lista foi gerada."
        )
    else:
        # --------------------------------------------------------
        # Geração mínima (1 a 3 listas) — n-base
        # --------------------------------------------------------
        max_listas = min(3, len(listas_base))
        listas_n = [sorted(lst)[:n_int] for lst in listas_base[:max_listas]]

        st.success(f"Geração EXPERIMENTAL autorizada — {len(listas_n)} lista(s).")

        for i, lst in enumerate(listas_n, start=1):
            st.code(f"Lista N{i}: {lst}", language="python")

        # --------------------------------------------------------
        # Mini-laudo automático
        # --------------------------------------------------------
        st.markdown("#### 📄 Mini-Laudo (Automático)")
        st.write(
            {
                "modo": "Modo N Experimental",
                "n": n_int,
                "listas_geradas": len(listas_n),
                "orcamento_manual": orcamento_manual,
                "regime": "OBSERVADO",
                "status": "GERADO" if listas_n else "RECUSADO",
            }
        )



# ============================================================
# 📊 V16 PREMIUM — MVP-U2 | ORÇAMENTO UNIVERSAL (OBSERVACIONAL)
# ============================================================
if painel == "📊 V16 Premium — Orçamento Universal":

    st.title("📊 MVP-U2 — Orçamento Universal (Observacional)")
    st.caption(
        "Observacional • Não gera listas • Não decide\n"
        "Avalia custo real dos pacotes já gerados (Modo 6 / Universal)."
    )

    listas = st.session_state.get("modo6_listas_totais", [])
    n_alvo = st.session_state.get("n_alvo")

    if not listas or n_alvo is None:
        st.warning(
            "Pacote indisponível.\n\n"
            "Execute primeiro:\n"
            "• Pipeline\n"
            "• Modo 6 (Painel 11)"
        )
        st.stop()

    st.markdown("---")

    # --------------------------------------------------------
    # TABELA DE CUSTO UNIVERSAL (CANÔNICA)
    # --------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    st.markdown("### 📐 Tabela canônica de custo (fixa)")
    st.json(TABELA_CUSTO)

    st.markdown("---")

    # --------------------------------------------------------
    # Entrada de orçamento manual (opcional)
    # --------------------------------------------------------
    orcamento_manual = st.number_input(
        "Orçamento manual (opcional)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Avaliação das listas
    # --------------------------------------------------------
    linhas = []

    for i, lista in enumerate(listas, start=1):
        tamanho = len(lista)

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "lista_id": i,
            "n_lista": tamanho,
            "custo_estimado": custo,
            "cabe_no_orcamento_manual": (
                custo is not None and orcamento_manual > 0 and custo <= orcamento_manual
            ),
        })

    df_orc = pd.DataFrame(linhas)

    st.markdown("### 📊 Avaliação observacional do pacote")
    st.dataframe(df_orc, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Leitura correta**
- Custo **None** = combinação não prevista na tabela
- Painel **não filtra**, **não decide**, **não prioriza**
- Serve apenas para **decisão HUMANA**
"""
    )

# ============================================================
# MVP-U3 — COBERTURA UNIVERSAL (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "🧩 MVP-U3 — Cobertura Universal":

    st.markdown("## 🧩 MVP-U3 — Cobertura Universal (Observacional)")
    st.caption(
        "Avalia cobertura, redundância e custo teórico do pacote ATUAL.\n"
        "Funciona para qualquer n_alvo (5, 6, 15, etc.).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote de listas disponível para avaliação.")
        st.stop()

    # ------------------------------------------------------------
    # Detecção canônica de n_alvo
    # ------------------------------------------------------------
    n_alvo = st.session_state.get("n_alvo")
    if not n_alvo or n_alvo <= 0:
        st.warning("n_alvo não detectado. Execute o carregamento do histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado no histórico
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    # ------------------------------------------------------------
    # Métricas de cobertura
    # ------------------------------------------------------------
    total_listas = len(pacote)

    tamanhos = [len(set(lst)) for lst in pacote]
    validas = [lst for lst in pacote if len(set(lst)) >= n_alvo]

    cobertura_unica = set()
    for lst in validas:
        cobertura_unica.update(lst)

    taxa_validas = len(validas) / total_listas if total_listas else 0.0
    cobertura_pct = (
        len(cobertura_unica) / len(universo) * 100 if universo else 0.0
    )

    # Redundância média
    freq = {}
    for lst in validas:
        for x in lst:
            freq[x] = freq.get(x, 0) + 1

    redundancia_media = (
        sum(freq.values()) / len(freq) if freq else 0.0
    )

    # ------------------------------------------------------------
    # Exibição — Métricas principais
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Listas totais", total_listas)
    col2.metric("Listas ≥ n_alvo", f"{len(validas)} ({taxa_validas*100:.1f}%)")
    col3.metric("Cobertura do universo", f"{cobertura_pct:.1f}%")
    col4.metric("Redundância média", f"{redundancia_media:.2f}")

    # ------------------------------------------------------------
    # Diagnóstico textual (OBSERVACIONAL)
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    if taxa_validas < 0.6:
        st.warning(
            "Poucas listas atingem o tamanho mínimo do fenômeno.\n"
            "Cobertura estrutural fraca."
        )
    elif cobertura_pct < 40:
        st.warning(
            "Cobertura baixa do universo observado.\n"
            "Pacote concentrado demais."
        )
    else:
        st.success(
            "Cobertura estrutural aceitável para o fenômeno atual.\n"
            "Pacote coerente sob critério universal."
        )

    st.info(
        "📌 Este painel NÃO decide execução.\n"
        "Use apenas como régua de cobertura e redundância."
    )

# ============================================================
# <<< FIM — MVP-U3 — COBERTURA UNIVERSAL
# ============================================================

# ============================================================
# MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO (OBSERVACIONAL)
# NÃO GERA LISTAS • NÃO DECIDE • NÃO ALTERA MOTOR
# ============================================================
if painel == "📈 MVP-U4 — Eficiência Marginal por Custo":

    st.markdown("## 📈 MVP-U4 — Eficiência Marginal por Custo (Observacional)")
    st.caption(
        "Avalia quanto de cobertura adicional é obtida por unidade extra de orçamento.\n"
        "Depende de U2 (Orçamento) e U3 (Cobertura).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apenas mede"
    )

    # ------------------------------------------------------------
    # Recuperação do histórico e n_alvo
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    if historico_df is None or historico_df.empty or not n_alvo:
        st.warning("Histórico ou n_alvo indisponível. Carregue o histórico primeiro.")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = (
        st.session_state.get("pacote_listas_atual")
        or st.session_state.get("modo6_listas_totais")
    )

    if not pacote:
        st.warning("Nenhum pacote disponível para análise.")
        st.stop()

    # ------------------------------------------------------------
    # Universo real observado
    # ------------------------------------------------------------
    col_pass = [c for c in historico_df.columns if c.startswith("p")]
    universo = sorted(
        {
            int(v)
            for _, row in historico_df.iterrows()
            for v in row[col_pass]
            if pd.notna(v) and int(v) > 0
        }
    )

    if not universo:
        st.warning("Universo vazio. Histórico inválido.")
        st.stop()

    universo_size = len(universo)

    # ------------------------------------------------------------
    # Tabela canônica de custo (mesma do MVP-U2)
    # ------------------------------------------------------------
    TABELA_CUSTO = {
        5:  {5: 3,   6: 18,   7: 63,   8: 168,   9: 378,   10: 756},
        6:  {6: 6,   7: 42,   8: 168,  9: 504,   10: 1260, 11: 2772},
        15: {15: 3.5, 16: 56, 17: 476},
    }

    # ------------------------------------------------------------
    # Agrupamento por tamanho de lista (≥ n_alvo)
    # ------------------------------------------------------------
    grupos = {}
    for lst in pacote:
        if len(set(lst)) >= n_alvo:
            k = len(set(lst))
            grupos.setdefault(k, []).append(lst)

    if not grupos:
        st.warning("Nenhuma lista válida (≥ n_alvo) encontrada.")
        st.stop()

    # ------------------------------------------------------------
    # Cálculo de cobertura por grupo
    # ------------------------------------------------------------
    linhas = []

    for tamanho, listas in sorted(grupos.items()):
        cobertura = set()
        for lst in listas:
            cobertura.update(lst)

        cobertura_pct = len(cobertura) / universo_size * 100

        custo = None
        if n_alvo in TABELA_CUSTO and tamanho in TABELA_CUSTO[n_alvo]:
            custo = TABELA_CUSTO[n_alvo][tamanho]

        linhas.append({
            "n_lista": tamanho,
            "cobertura_pct": cobertura_pct,
            "custo": custo,
        })

    df = pd.DataFrame(linhas).sort_values("n_lista").reset_index(drop=True)

    if df.empty:
        st.warning("Não foi possível calcular métricas.")
        st.stop()

    # ------------------------------------------------------------
    # Base = menor tamanho válido
    # ------------------------------------------------------------
    base = df.iloc[0]
    base_cob = base["cobertura_pct"]
    base_custo = base["custo"]

    # ------------------------------------------------------------
    # Eficiência marginal
    # ------------------------------------------------------------
    em_linhas = []
    for _, row in df.iterrows():
        if row["custo"] is None or base_custo is None or row["custo"] == base_custo:
            em = None
            dc = None
            dd = None
        else:
            dc = row["custo"] - base_custo
            dd = row["cobertura_pct"] - base_cob
            em = dd / dc if dc > 0 else None

        em_linhas.append({
            "n_lista": row["n_lista"],
            "cobertura_pct": round(row["cobertura_pct"], 2),
            "custo": row["custo"],
            "Δcobertura": round(dd, 2) if dd is not None else None,
            "Δcusto": dc,
            "eficiencia_marginal": round(em, 4) if em is not None else None,
        })

    df_em = pd.DataFrame(em_linhas)

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    st.markdown("### 📊 Eficiência marginal por tamanho de lista")
    st.dataframe(df_em, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # Leitura observacional
    # ------------------------------------------------------------
    st.markdown("### 🧠 Leitura observacional")

    valid_em = df_em.dropna(subset=["eficiencia_marginal"])
    if valid_em.empty:
        st.info("Eficiência marginal não disponível para comparação.")
    else:
        melhor = valid_em.sort_values("eficiencia_marginal", ascending=False).iloc[0]
        st.success(
            f"Maior eficiência marginal em n_lista = {int(melhor['n_lista'])} "
            f"(EM = {melhor['eficiencia_marginal']})."
        )

        baixos = valid_em[valid_em["eficiencia_marginal"] < 0.01]
        if not baixos.empty:
            st.warning(
                "Retorno decrescente detectado em alguns tamanhos:\n"
                + ", ".join(str(int(x)) for x in baixos["n_lista"].tolist())
            )

    st.info(
        "📌 Este painel é apenas observacional.\n"
        "Use para decidir até onde vale a pena aumentar o orçamento."
    )

# ============================================================
# <<< FIM — MVP-U4 — EFICIÊNCIA MARGINAL POR CUSTO
# ============================================================


# ============================================================
# Painel 12 — 🧪 Testes de Confiabilidade REAL
# ============================================================
if painel == "🧪 Testes de Confiabilidade REAL":

    st.markdown("## 🧪 Testes de Confiabilidade REAL — V15.7 MAX")

    df = st.session_state.get("historico_df")
    listas_m6 = st.session_state.get("modo6_listas")
    ultima_prev = st.session_state.get("ultima_previsao")

    if df is None or listas_m6 is None or ultima_prev is None:
        exibir_bloco_mensagem(
            "Pré-requisitos não atendidos",
            "Execute o pipeline até o Modo 6 Acertos.",
            tipo="warning",
        )
        st.stop()

    qtd_series = len(df)
    if qtd_series < 15:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "São necessárias pelo menos 15 séries para validar a confiabilidade.",
            tipo="warning",
        )
        st.stop()

    st.info("Executando avaliação REAL de confiabilidade...")

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Janela de teste recente
    janela = df[col_pass].iloc[-12:].values

    # ============================================================
    # Medição de acertos reais
    # ============================================================
    def acertos(lista, alvo):
        return len(set(lista) & set(alvo))

    acertos_nucleo = []
    acertos_coberturas = []

    for alvo in janela:
        # núcleo
        ac_nuc = acertos(ultima_prev, alvo)
        acertos_nucleo.append(ac_nuc)

        # coberturas
        max_cov = 0
        for lst in listas_m6:
            ac_lst = acertos(lst, alvo)
            if ac_lst > max_cov:
                max_cov = ac_lst
        acertos_coberturas.append(max_cov)

    # Médias
    media_nucleo = float(np.mean(acertos_nucleo))
    media_cob = float(np.mean(acertos_coberturas))

    # ============================================================
    # Diagnóstico
    # ============================================================
    corpo = (
        f"- Janela avaliada: **12 séries recentes**\n"
        f"- Média de acertos do Núcleo: **{media_nucleo:.2f}**\n"
        f"- Média de acertos das Coberturas: **{media_cob:.2f}**\n"
        "\n"
        "Coberturas devem superar o núcleo em ambientes turbulentos.\n"
        "Se o núcleo supera as coberturas, o ambiente está mais limpo."
    )

    exibir_bloco_mensagem(
        "Resumo da Confiabilidade REAL",
        corpo,
        tipo="info",
    )

    st.success("Teste de Confiabilidade REAL concluído com sucesso!")

# ============================================================
# BLOCO — SANIDADE FINAL DAS LISTAS DE PREVISÃO
# (Elimina permutações, duplicatas por conjunto
#  E listas com números repetidos internos)
# Válido para V15.7 MAX e V16 Premium
# ============================================================

def sanidade_final_listas(listas):
    pc_exec_trace("ENTER sanidade_final_listas", {"arg_n": len(listas or [])})
    pc_list_source_detector("BEFORE sanidade_final_listas", listas, {"arg_n": len(listas or [])})
    """
    Sanidade final das listas de previsão.
    Regras:
    - Remove listas com números repetidos internamente
    - Remove permutações (ordem diferente, mesmos números)
    - Remove duplicatas exatas
    - Garante apenas listas válidas com 6 números distintos
    """

    listas_saneadas = []
    vistos = set()

    for lista in listas:
        try:
            nums = [int(x) for x in lista]
        except Exception:
            continue

        # 🔒 REGRA CRÍTICA — exatamente 6 números distintos
        if len(nums) != 6:
            continue

        if len(set(nums)) != 6:
            # Exemplo eliminado: [11, 12, 32, 32, 37, 42]
            continue

        # Normaliza ordem para detectar permutações
        chave = tuple(sorted(nums))

        if chave in vistos:
            continue

        vistos.add(chave)
        listas_saneadas.append(nums)

    pc_list_source_detector("AFTER sanidade_final_listas", listas_saneadas, {"arg_n": len(listas_saneadas or [])})
    return listas_saneadas


# ============================================================
# APLICAÇÃO AUTOMÁTICA DA SANIDADE (SE LISTAS EXISTIREM)
# ============================================================

# Sanear listas do Modo 6 (V15.7)
if "modo6_listas" in st.session_state:
    if bool(st.session_state.get("pc_force_fresh_packet_active", False)):
        st.caption("🛡️ Re-sanity por session ignorada: pacote fresco do gerador tem precedência nesta execução.")
    else:
        st.session_state["modo6_listas"] = sanidade_final_listas(
            st.session_state.get("modo6_listas", []),
        )

# Sanear Execução V16 (se existir)
if "v16_execucao" in st.session_state:
    exec_v16 = st.session_state.get("v16_execucao", {})

    for chave in ["C2", "C3", "todas_listas"]:
        if chave in exec_v16:
            exec_v16[chave] = sanidade_final_listas(
                exec_v16.get(chave, []),
            )

    st.session_state["v16_execucao"] = exec_v16

# ============================================================
# PARTE 6/8 — FIM
# ============================================================



# ============================================================
# PARTE 7/8 — INÍCIO
# ============================================================

# ============================================================
# Painel — 🧪 Replay Curto — Expectativa 1–3 Séries (V16)
# Diagnóstico apenas | NÃO gera previsões | NÃO altera fluxo
# ============================================================
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":

    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries (Diagnóstico)")
    st.caption(
        "Validação no passado da expectativa de curto prazo (1–3 séries). "
        "Este painel **não prevê números** e **não altera decisões**."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Parâmetros FIXOS (sem bifurcação)
    # -------------------------------
    JANELA_REPLAY = 80       # pontos do passado
    HORIZONTE = 3            # 1–3 séries
    LIMIAR_NR = 0.02         # queda mínima de NR% para considerar melhora
    LIMIAR_DIV = 0.50        # queda mínima de divergência para considerar melhora

    n = len(df)
    if n < JANELA_REPLAY + HORIZONTE + 5:
        exibir_bloco_mensagem(
            "Histórico insuficiente",
            "É necessário mais histórico para o replay curto.",
            tipo="warning",
        )
        st.stop()

    # -------------------------------
    # Helpers locais (diagnóstico)
    # -------------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]

    def calc_nr_local(matriz):
        # NR% aproximado (mesma lógica do painel, versão local)
        variancias = np.var(matriz, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = []
        for i in range(1, len(matriz)):
            saltos.append(np.linalg.norm(matriz[i] - matriz[i - 1]))
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return (0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20))

    def calc_div_local(base, candidatos):
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    def estado_sinal(nr_deriv, div_deriv, vel):
        # 🟢 melhora curta
        if nr_deriv < -LIMIAR_NR and div_deriv < -LIMIAR_DIV and vel < 0.75:
            return "🟢 Melhora curta"
        # 🔴 continuidade ruim
        if nr_deriv > 0 or div_deriv > 0 or vel >= 0.80:
            return "🔴 Continuidade ruim"
        # 🟡 transição
        return "🟡 Respiração / Transição"

    # -------------------------------
    # Replay
    # -------------------------------
    resultados = []
    base_ini = n - JANELA_REPLAY - HORIZONTE

    for i in range(base_ini, n - HORIZONTE):
        # Janela até o ponto i
        matriz_i = matriz_norm[: i + 1]
        nr_i = calc_nr_local(matriz_i)

        # Divergência local (proxy simples)
        base = matriz_i[-1]
        candidatos = matriz_i[-10:] if len(matriz_i) >= 10 else matriz_i
        div_i = calc_div_local(base, candidatos)

        # Velocidade (proxy simples)
        vel = float(np.mean(np.std(matriz_i[-5:], axis=1)))

        # Próximo trecho (1–3)
        matriz_f = matriz_norm[: i + 1 + HORIZONTE]
        nr_f = calc_nr_local(matriz_f)
        base_f = matriz_f[-1]
        candidatos_f = matriz_f[-10:] if len(matriz_f) >= 10 else matriz_f
        div_f = calc_div_local(base_f, candidatos_f)

        nr_deriv = nr_f - nr_i
        div_deriv = div_f - div_i

        estado = estado_sinal(nr_deriv, div_deriv, vel)

        melhora_real = (nr_deriv < -LIMIAR_NR) or (div_deriv < -LIMIAR_DIV)

        resultados.append({
            "estado": estado,
            "melhora_real": melhora_real
        })

    # -------------------------------
    # Consolidação
    # -------------------------------
    df_res = pd.DataFrame(resultados)
    resumo = (
        df_res.groupby("estado")["melhora_real"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={
            "count": "Ocorrências",
            "mean": "Taxa de Melhora"
        })
    )
    resumo["Taxa de Melhora"] = (resumo["Taxa de Melhora"] * 100).round(1)

    st.markdown("### 📊 Resultado do Replay Curto (passado)")
    st.dataframe(resumo, use_container_width=True)

    st.info(
        "Este painel valida **se o estado 🟢 precede melhora real** no curto prazo "
        "(1–3 séries) **mais vezes que o acaso**. "
        "Ele **não prevê o futuro**, apenas qualifica a expectativa."
    )

# ============================================================
# CAMADA B1 — PRIORIZAÇÃO OBSERVACIONAL DE LISTAS (V16)
# NÃO gera | NÃO decide | NÃO altera motores
# ============================================================

def v16_priorizar_listas_por_contexto(listas):
    """
    Ordena listas existentes usando ECO + Estado.
    Apenas PRIORIZA — não remove, não cria, não decide.

    v16h22 (γ BORDA FINA):
    - Se CAP estiver CALIBRADA e Parabólica estiver PLANA
      e ainda estivermos em "zero-4" (rate_4p_w == 0),
      aplica um viés controlado de diversificação (γ) + penalidade nocivos.
    - Não altera Camada 4; apenas mexe na ordem do pacote.
    """

    diag = st.session_state.get("diagnostico_eco_estado_v16", {})
    eco_acion = diag.get("eco_acionabilidade", "não_acionável")
    estado = diag.get("estado", "indefinido")

    if not listas or not isinstance(listas, list):
        return listas

    # ---------------- γ BORDA FINA (pré‑C4, ruptura controlada) ----------------
    # Condição canônica: só ativa quando:
    # - CAP calibrada
    # - Parabólica PLANA (momento neutro / sem curvatura sustentada)
    # - e o SAFE ainda mostra zero-4 na janela (w=60)
    cap_status = str(st.session_state.get("cap_status", "")).upper()
    parab_estado = str(st.session_state.get("parab_estado_global", "")).upper()
    cap_pct = float(st.session_state.get("cap_pct", 0.65) or 0.65)

    prova = st.session_state.get("replay_stats_prova_janela", {}) or {}
    rate_4p_w = prova.get("rate_4p_w", None)

    gamma_ativo = (
        ("CALIBRADA" in cap_status)
        and (parab_estado == "PLANA")
        and (rate_4p_w == 0 or rate_4p_w == 0.0)
    )

    # Nocivos consistentes (V16 Premium — ANTI-EXATO) — observacional
    nocivos_consistentes = st.session_state.get("anti_exato_nocivos_consistentes", []) or []
    nocivos_set = set([int(x) for x in nocivos_consistentes if str(x).isdigit()])

    # Frequência por passageiro (dentro do pacote candidato) — para medir "borda fina"
    freq = {}
    if gamma_ativo:
        try:
            for L in listas:
                for p in L:
                    freq[p] = freq.get(p, 0) + 1
        except Exception:
            freq = {}

    # Guardas para não explodir score com listas vazias
    def _safe_sum_int(L):
        try:
            return int(sum([int(x) for x in L]))
        except Exception:
            return 0

    def score_lista(L):
        # Base score (mantém o jeitão antigo)
        try:
            base_score = len(L) + (_safe_sum_int(L) / 1000.0)
        except Exception:
            base_score = 0.0

        # Ajuste por contexto ECO/estado (mantido)
        try:
            if eco_acion == "acionável":
                base_score += 1.0
            if estado in ["ESTAVEL", "RESPIRAVEL", "NEUTRO"]:
                base_score += 0.5
        except Exception:
            pass

        if not gamma_ativo:
            return base_score

        # γ BORDA FINA: favorece diversidade "real" (passageiros menos repetidos no pacote)
        # e penaliza presença de nocivos consistentes, sem bloquear.
        try:
            # rareza: soma de inverso da frequência (quanto mais raro, maior)
            rareza = 0.0
            for p in L:
                fp = freq.get(p, 1)
                rareza += 1.0 / float(fp)

            # penalidade nocivos (leve; não zera listas)
            noc_pen = 0.0
            for p in L:
                if p in nocivos_set:
                    noc_pen += 1.0

            # γ controlado pela CAP (cap_pct) — mantém governança pré‑C4
            gamma = 0.75 * cap_pct   # 0.4875 quando cap_pct=0.65
            beta_noc = 0.35          # penalidade por nocivo

            return base_score + (gamma * rareza) - (beta_noc * noc_pen)
        except Exception:
            return base_score

    try:
        listas_ordenadas = sorted(listas, key=score_lista, reverse=True)
        return listas_ordenadas
    except Exception:
        return listas

# ============================================================
# >>> PAINEL X — 🧠 Memória Operacional — Observacional
# ============================================================
if painel in ["🧠 Memória Operacional", "🧠 Memória Operacional — Observacional"]:
    st.markdown("## 🧠 Memória Operacional (Observacional)")
    st.caption("Este painel é um espelho: mostra registros já existentes. Não pede confirmação do operador para registros automáticos.")

    # Garantir estrutura mínima
    if "memoria_operacional" not in st.session_state or st.session_state["memoria_operacional"] is None:
        st.session_state["memoria_operacional"] = []

    registros = st.session_state["memoria_operacional"]

    if len(registros) == 0:
        st.info("Sem registros na Memória Operacional nesta sessão. (Isso não é erro.)")
        st.caption("📌 Observação: o M5 — Pulo do Gato registra automaticamente 'fotos' na Memória de Estados (M2). Para massa histórica, use 🧠 Memória de Estados (M2) e o 📈 M3.")
    else:
        st.success(f"{len(registros)} registro(s) nesta sessão.")
        # Exibição simples e segura (sem botões)
        for i, r in enumerate(registros[-50:], start=max(1, len(registros)-49)):
            st.markdown(f"**{i:02d})** `{r}`")
if painel == "🧠 Memória Operacional — Registro Semi-Automático":
    st.markdown("## 🧠 Memória Operacional — Registro Semi-Automático (Passivo)")
    st.caption("Mantido por compatibilidade de navegação. Operação passiva (sem botões). Use o painel 🧠 Memória Operacional para ver registros.")

    if "memoria_operacional" not in st.session_state or st.session_state["memoria_operacional"] is None:
        st.session_state["memoria_operacional"] = []

    if len(st.session_state["memoria_operacional"]) == 0:
        st.info("Sem registros nesta sessão.")
    else:
        st.success(f"Registros nesta sessão: {len(st.session_state['memoria_operacional'])}")
if painel == "📘 Relatório Final":

    st.markdown("## 📘 Relatório Final — V15.7 MAX — V16 Premium Profundo")

    # 🕺 Ritmo/Dança (ex-post · pré-C4) — base para Memória Estrutural do RESPIRÁVEL
    st.markdown("### 🕺 Ritmo/Dança (ex-post · pré-C4)")
    ritmo_info = st.session_state.get("ritmo_danca_info")
    if not isinstance(ritmo_info, dict) or not ritmo_info:
        ritmo_info = {"ritmo_global": "N/D", "motivos": ["sem_dados"], "sinais": {}}
    st.json(ritmo_info, expanded=False)
    # -----------------------------------------------------------------
    # 🚥 MICRO_ATIVO + SINAL CANÔNICO DO 5 + SENSOR CANÔNICO DO 6 (pré-C4)
    # Observacional · Auditável · Não decide · Não altera listas.
    # -----------------------------------------------------------------
    ss_info = st.session_state.get("ss_info", {}) or {}
    prova = st.session_state.get("replay_stats_prova_janela", {}) or {}
    curv = st.session_state.get("curvatura_sustentada_info", {}) or {}
    ss_ok = bool(ss_info.get("status"))
    any4_w = bool(prova.get("any_4p_seen_w"))
    rate4_w = float(prova.get("rate_4p_w") or 0.0)
    p3_rate_w = float(prova.get("rate_3p_w") or 0.0)
    trave_w = prova.get("trave_ratio_w")
    fech_norm_w = float(prova.get("fechamento_gap_norm_w") or 0.0)
    dist4 = curv.get("dist_desde_ultimo_4")
    troca = bool(curv.get("troca_sinal_recente"))
    sust = bool(curv.get("curvatura_sustentada_recente"))

    micro_motivos = []
    micro_motivos.append(f"SS_OK={ss_ok}")
    micro_motivos.append(f"ANY_4P_W={any4_w}")
    micro_motivos.append(f"RATE_4P_W={rate4_w:.4f}")
    micro_motivos.append(f"TROCA_SINAL={troca}")
    micro_motivos.append(f"CURV_SUST={sust}")
    if dist4 is not None:
        micro_motivos.append(f"DIST_DESDE_ULTIMO_4={dist4}")

    # MICRO_ATIVO (curta-metragem verdadeira) = base suficiente + evidência local de ≥4 + evento geométrico recente
    micro_ativo = bool(ss_ok and any4_w and (rate4_w >= 0.01) and (troca or sust))

    # SINAL_5 = aproximação (proxy) — não é 5, é “estrada ficando dirigível para capturar 5”
    # (usa: micro_ativo + pressão ≥3 no filme curto + fechamento de gap)
    sinal5 = bool(micro_ativo and (p3_rate_w >= 0.12) and (fech_norm_w >= 0.70))

    # SENSOR_6 = condição canônica ainda mais rara (preparação para cravar) — estritamente informativo
    # (exige: sinal5 + densidade de 4 no filme curto + proximidade alta + curvatura sustentada)
    trave_ok = (trave_w is None) or (float(trave_w) >= 0.70)
    dist_ok = (dist4 is None) or (int(dist4) <= 12)
    sensor6 = bool(sinal5 and (rate4_w >= 0.03) and trave_ok and dist_ok and sust)

    micro_payload = {
        "MICRO_ATIVO": micro_ativo,
        "SINAL_5": sinal5,
        "SENSOR_6": sensor6,
        "motivos": micro_motivos,
        "prova_w": {
            "w_used": prova.get("w_used"),
            "avg_best_w": prova.get("avg_best_w"),
            "rate_4p_w": prova.get("rate_4p_w"),
            "rate_3p_w": prova.get("rate_3p_w"),
            "fechamento_gap_norm_w": prova.get("fechamento_gap_norm_w"),
            "trave_ratio_w": prova.get("trave_ratio_w"),
        },
        "curvatura": {
            "estado_recente": curv.get("estado_recente"),
            "curvatura_sustentada_recente": curv.get("curvatura_sustentada_recente"),
            "troca_sinal_recente": curv.get("troca_sinal_recente"),
            "dist_desde_ultimo_4": curv.get("dist_desde_ultimo_4"),
        },
    }
    # λ* — fase de estabilização (pré‑C4, informativo)
    _lambda_info = st.session_state.get("lambda_star_info")
    if isinstance(_lambda_info, dict) and _lambda_info:
        micro_payload["lambda_star"] = {
            "lambda_star_eff": st.session_state.get("lambda_star"),
            "lambda_star_raw": st.session_state.get("lambda_star_raw"),
            "lambda_star_target": st.session_state.get("lambda_star_target"),
            "fase_estabilizacao": _lambda_info.get("fase_estabilizacao", "N/D"),
            "mass": _lambda_info.get("mass"),
            "win": _lambda_info.get("win"),
            "faltam": _lambda_info.get("faltam"),
        }


    st.markdown("### 🚥 MICRO_ATIVO · SINAL_5 · SENSOR_6 (pré‑C4 · governança)")
    if sensor6:
        st.success("SENSOR_6: ATIVO (condição canônica rara). Informativo — não decide, não altera listas.")
    elif sinal5:
        st.info("SINAL_5: ATIVO (aproximação). Informativo — não decide, não altera listas.")
    elif micro_ativo:
        st.info("MICRO_ATIVO: SIM (curta‑metragem verdadeira). Informativo — não decide, não altera listas.")
    else:
        st.warning("MICRO_ATIVO: NÃO (sem evidência local suficiente no filme curto).")
    st.json(micro_payload, expanded=False)
    # ALERTA: λ* em estabilização (não é erro; é maturação de janela)
    try:
        _li = st.session_state.get("lambda_star_info") or {}
        _fase = str(_li.get("fase_estabilizacao", "")) or str(st.session_state.get("lambda_star_fase",""))
        if _fase in ("INICIAL", "TRANSICAO"):
            _mass = _li.get("mass", None)
            _faltam = _li.get("faltam", None)
            _win = _li.get("win", None)
            _raw = st.session_state.get("lambda_star_raw")
            _tgt = st.session_state.get("lambda_star_target")
            _eff = st.session_state.get("lambda_star")
            st.info(f"λ* em fase de estabilização: fase={_fase} · massa={_mass} · faltam={_faltam} (meta win={_win}) · raw={_raw} · target={_tgt} · eff={_eff}. Use como indício, não como certeza.")
    except Exception:
        pass


    # guardar para outros painéis (sem acoplar em decisão)
    st.session_state["micro_ativo_info"] = micro_payload


    # Sincroniza chaves canônicas (ECO/Estado/k*/Divergência) antes de consolidar
    v16_sync_aliases_canonicos()

    # ------------------------------------

    # ------------------------------------------------------------
    # 👁️ CAMADA 3 — Cegueiras ainda possíveis (hipóteses)
    # (Somente no RF: não cria sensores, não decide nada)
    # ------------------------------------------------------------
    try:
        m3_reg = st.session_state.get("m3_regime_dx") or st.session_state.get("m3_regime") or "N/D"
        nrp = st.session_state.get("nr_percent")
        divv = st.session_state.get("divergencia_s6_mc")
        cls_r = st.session_state.get("classe_risco") or "N/D"

        # tenta reaproveitar o diagnóstico da Camada 2 (se existir no escopo)
        rigido_flag = False
        try:
            rigido_flag = bool(locals().get("diag_j", {}).get("rigido"))
        except Exception:
            rigido_flag = False

        linhas = []
        linhas.append("🎛️ **Instrumento vs fenômeno:** a leitura pode estar limitada pela lente (ruído/divergência), não só pelo mundo.")
        linhas.append("🧱 **Compressão ≠ erro:** pacote estreito pode ser regime neutro/estreito real — não necessariamente rigidez ruim.")

        if str(m3_reg).upper() == "RUIM":
            linhas.append("🌫️ **RUIM com frestas:** RUIM pode ter micro‑aberturas locais (curtas) que não viram ECO/PRÉ‑ECO no agregado.")
        if rigido_flag:
            linhas.append("🧩 **Perda por borda:** jeitão pode estar correto, mas 1–2 passageiros de borda podem ficar fora quando o pacote fica colado.")
            linhas.append("⚠️ **Rigidez detectada:** hipótese ativa de perda por compressão excessiva (sinal p/ governança/cobertura).")

        # ausência de anti-âncora (se RF tiver essa info)
        try:
            if not st.session_state.get("anti_ancora_idx_detectados"):
                linhas.append("🧲 **Anti‑âncora ausente:** pode ser E0 real OU pouca amplitude do pacote (poucas listas / pouca variação).")
        except Exception:
            pass

        try:
            if isinstance(nrp, (int, float)) and nrp >= 50:
                linhas.append("🔴 **NR crítico:** ruído alto pode achatar leitura fina e mascarar sinal fraco; cuidado extra com 'miragem'.")
        except Exception:
            pass
        try:
            if isinstance(divv, (int, float)) and divv >= 3:
                linhas.append("🟡 **Divergência moderada/alta:** modelos discordando pode ocultar padrão local; trate como hipótese, não permissão de ataque.")
        except Exception:
            pass
        if "🟠" in str(cls_r) or "Elevado" in str(cls_r) or "🔴" in str(cls_r):
            linhas.append("🛑 **Risco elevado:** mesmo com estrada neutra, turbulência pode exigir postura de cobertura (não de invenção).")

        st.markdown("### 👁️ Camada 3 — Cegueiras ainda possíveis (hipóteses)")
        st.caption("Este bloco **NÃO cria sensores novos** e **NÃO decide nada**. Ele lista hipóteses de cegueira ainda possíveis (para não confundir fresta com miragem).")
        for ln in linhas:
            st.markdown(f"- {ln}")

        st.caption("Regra canônica: **mapa de hipóteses**, não motor. Mantém pressão evolutiva sem transformar leitura em fé.")
    except Exception:
        # falha silenciosa (não derruba o RF)
        pass
# ------------------------
    # 🧭 BLOCO -1 — SUMÁRIO EXECUTIVO (read-only)
    # ------------------------------------------------------------
    try:
        _snap = _m1_collect_mirror_snapshot() if '_m1_collect_mirror_snapshot' in globals() else {}
        _estado = _m1_classificar_estado(_snap) if '_m1_classificar_estado' in globals() else {'estado':'S0','avisos':[],'snapshot':_snap}
        st.markdown('### 🧭 Sumário Executivo (rodada atual)')
        # --- Regime por fonte (consolidação) ---
        st.markdown('### 🧷 Regime por fonte (consolidação)')
        reg_pipeline = st.session_state.get('pipeline_estrada', None)
        reg_global = st.session_state.get('regime', None)
        reg_m3 = st.session_state.get('m3_regime_dx', None)
        classe_risco = st.session_state.get('classe_risco', None)
        k_star = st.session_state.get('k_star', None)
        nr = st.session_state.get('nr_percent', None)
        div_s6_mc = st.session_state.get('divergencia_s6_mc', None)
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown('**🛣️ Pipeline (Estrada)**')
            st.write(reg_pipeline if reg_pipeline is not None else '—')
            st.caption('Regime global atual: {}'.format(reg_global) if reg_global is not None else 'Regime global atual: —')
        with colB:
            st.markdown('**🛰️ Sentinelas / Risco**')
            st.write('Classe: {}'.format(classe_risco) if classe_risco is not None else 'Classe: —')
            st.write('k*: {:.4f}'.format(k_star) if isinstance(k_star, (int, float)) else ('k*: {}'.format(k_star) if k_star is not None else 'k*: —'))
            st.write('NR%: {:.2f}%'.format(nr) if isinstance(nr, (int, float)) else ('NR%: {}'.format(nr) if nr is not None else 'NR%: —'))
            st.write('Div S6×MC: {:.4f}'.format(div_s6_mc) if isinstance(div_s6_mc, (int, float)) else ('Div S6×MC: {}'.format(div_s6_mc) if div_s6_mc is not None else 'Div S6×MC: —'))
        with colC:
            st.markdown('**📈 M3 / Expectativa (dx)**')
            st.write(reg_m3 if reg_m3 is not None else '—')
        st.caption('Pode haver divergência porque cada fonte mede uma coisa: Pipeline descreve a estrada, Sentinelas medem risco/turbulência, e M3 (dx) mede expectativa/analogia. Use cada leitura no seu uso canônico — sem misturar.')
        st.caption('Somente leitura. Não decide nada. Serve para você bater o olho e saber: **o que rodou**, **o que falta**, e **quais leituras estão disponíveis**.')
        if '_m1_render_barra_estados' in globals():
            _m1_render_barra_estados(_estado.get('estado','S0'))
        if _estado.get('avisos'):
            st.warning('Ainda não percorrido (na sessão): ' + ' · '.join(_estado.get('avisos', [])))
        # Snapshot resumido
        _s = _m1_collect_mirror_snapshot() if '_m1_collect_mirror_snapshot' in globals() else _estado.get('snapshot', {})
        _bl0 = {'historico_df': 'definido' if _s.get('historico_df') else '<não definido>', 'n_alvo': _s.get('n_alvo','N/D'), 'universo': _s.get('universo','N/D'), 'pipeline_ok': bool(_s.get('pipeline_ok')), 'regime': _s.get('regime','N/D')}
        _bl1 = {'k_star': _s.get('k_star','N/D'), 'nr_percent': _s.get('nr_percent','N/D'), 'divergencia_s6_mc': _s.get('divergencia_s6_mc','N/D'), 'indice_risco': _s.get('indice_risco','N/D'), 'classe_risco': _s.get('classe_risco','N/D')}
        _bl2 = {'turbo_tentado': bool(_s.get('turbo_tentado')), 'turbo_bloqueado': bool(_s.get('turbo_bloqueado')), 'turbo_motivo': _s.get('turbo_motivo','N/D'), 'modo6_executado': bool(_s.get('modo6_executado')), 'listas_geradas': _s.get('listas_geradas','<não definido>')}
        st.json(_bl0)
        st.json(_bl1)
        st.json(_bl2)
    except Exception:
        pass

    # ------------------------------------------------------------
    # 🎞️ BLOCO -0.5 — MEMÓRIA & EXPECTATIVA (read-only, se disponíveis)
    # ------------------------------------------------------------
    with st.expander('🎞️ Memória de Estados (M2) + Expectativa Histórica (M3) — resumo', expanded=False):
        try:
            m2 = st.session_state.get('m2_memoria_resumo_auditavel')
            if m2:
                st.markdown('#### 🎞️ M2 — Memória de Estados (resumo)')
                st.json(m2)
            else:
                st.info('M2 ainda sem massa mínima nesta sessão. (Isso não é erro.)')
            m3n = st.session_state.get('m3_eventos_similares')
            if m3n is not None:
                st.markdown('#### 📈 M3 — Expectativa Histórica (resumo)')
                st.json({'m3_regime_dx': st.session_state.get('m3_regime_dx','N/D'), 'm3_eventos_similares': m3n, 'taxa_eco1': st.session_state.get('m3_taxa_eco1','N/D'), 'taxa_estado_bom': st.session_state.get('m3_taxa_estado_bom','N/D'), 'taxa_transicao': st.session_state.get('m3_taxa_transicao','N/D'), 'ts': st.session_state.get('m3_ts','N/D')})
            else:
                st.info('Para preencher M3 no Relatório Final: rode o painel **📈 Expectativa Histórica — Contexto do Momento (V16)** nesta sessão.')
        except Exception:
            pass


    # ------------------------------------------------------------
    # 🧲 BLOCO 0 — SUGADOR DE ESTADO CONSOLIDADO
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")
    n_alvo = st.session_state.get("n_alvo")

    pipeline_status = st.session_state.get("pipeline_flex_ultra_concluido")
    ultima_prev = st.session_state.get("ultima_previsao")

    listas_m6_totais = (
        st.session_state.get("modo6_listas_totais")
        or st.session_state.get("modo6_listas")
        or []
    )

    listas_ultra = st.session_state.get("turbo_ultra_listas_leves") or []

    # Validação mínima
    if not listas_m6_totais:
        exibir_bloco_mensagem(
            "Sem pacote do Modo 6",
            "Execute o painel **🎯 Modo 6 Acertos — Execução** antes.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Estado consolidado
    # ------------------------------------------------------------
    linhas = []

    if historico_df is not None:
        linhas.append(f"- Séries carregadas: **{len(historico_df)}**")

    if n_alvo is not None:
        linhas.append(f"- Passageiros por carro (n): **{n_alvo}**")

    if pipeline_status is True:
        linhas.append("- Pipeline FLEX ULTRA: ✅ **CONCLUÍDO**")

    exibir_bloco_mensagem(
        "🧲 Estado Consolidado da Rodada",
        "\n".join(linhas),
        tipo="info",
    )

    # ============================================================
    # 🧠 DIAGNÓSTICO CONSOLIDADO DA RODADA (NOVO — ADITIVO)
    # ============================================================
    eco_consolidado = st.session_state.get("eco_status", "DESCONHECIDO")
    estado_consolidado = st.session_state.get("estado_atual", "DESCONHECIDO")

    st.markdown("### 🧠 Diagnóstico Consolidado da Rodada")

    st.info(
        f"**ECO:** {eco_consolidado}\n\n"
        f"**Estado do alvo:** {estado_consolidado}"
    )

    st.caption(
        "Leitura consolidada do sistema nesta rodada.\n"
        "Não gera decisão automática."
    )

    # ------------------------------------------------------------
    # Núcleo TURBO (se existir)
    # ------------------------------------------------------------
    st.markdown("### 🔮 Previsão Principal (Núcleo — TURBO++ ULTRA)")

    if ultima_prev:
        st.success(formatar_lista_passageiros(ultima_prev))
    else:
        st.info(
            "Nenhuma previsão TURBO disponível nesta rodada "
            "(isso é válido em regime estável)."
        )

    # ------------------------------------------------------------
    # 🛡️ Pacote Prioritário — Top 10 (Modo 6)
    # ------------------------------------------------------------
    st.markdown("### 🛡️ Pacote Prioritário (Top 10) — Modo 6")

    top10 = listas_m6_totais[:10]
    for i, lst in enumerate(top10, 1):
        st.markdown(f"**{i:02d})** {formatar_lista_passageiros(lst)}")


    # ------------------------------------------------------------
    # 🧷 Anti-Âncora (OBSERVACIONAL) — rotulagem Base × Anti
    # ------------------------------------------------------------
    try:
        analise_anti = v16_analisar_duplo_pacote_base_anti_ancora(
            listas=listas_m6_totais,
            base_n=10,
            max_anti=4,
            core_presenca_min=0.60,
        )
        st.session_state["v16_anti_ancora"] = analise_anti

        st.markdown("### 🧷 Anti-Âncora — Observacional (Base × Anti)")
        core = analise_anti.get("core") or []
        # --- V16: registrar compressão/CORE para Janela Local (não decide) ---
        st.session_state["janela_core_top10"] = list(core) if core else []
        st.session_state["janela_compressao_core"] = True if core else False
        if core:
            st.write("**CORE do pacote base (presença alta no Top 10):** " + ", ".join(map(str, core)))
        else:
            st.write("CORE indisponível (sem base suficiente).")

        anti_idx = analise_anti.get("anti_idx") or []
        if anti_idx:
            st.success(
                "Sugestão (não obrigatória): **Duplo pacote** = Base (Top 10) + "
                + f"Anti-âncora (listas existentes): {', '.join('L'+str(i) for i in anti_idx)}"
            )
            for i in anti_idx:
                try:
                    lst = listas_m6_totais[int(i) - 1]
                    ov = (analise_anti.get("overlaps") or [None])[int(i) - 1]
                    st.write(f"**L{i:02d} (anti-âncora | overlap CORE={ov})** — {formatar_lista_passageiros(lst)}")
                except Exception:
                    pass
        else:
            st.info(
                "Nenhuma lista anti-âncora clara foi detectada entre as listas disponíveis. "
                "Isso é compatível com pacote muito comprimido (E0 + envelope estreito)."
            )
    except Exception:
        st.session_state["v16_anti_ancora"] = None
        # falha silenciosa (não derruba o RF)





    # ------------------------------------------------------------
    # 🧩 Diagnóstico — Rigidez do Jeitão (folga) [OBSERVACIONAL]
    # ------------------------------------------------------------
    st.markdown("### 🧩 Jeitão do Pacote — Rigidez × Folga (diagnóstico)")
    st.caption("Alerta diagnóstico: quando o pacote fica rígido demais, ele pode 'acertar o jeitão' mas perder passageiros por rigidez. Isso NÃO é decisão: é só sinal para governança/cobertura.")

    try:
        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")
        diag_j = v16_diagnostico_rigidez_jeitao(
            listas=listas_m6_totais,
            universo_min=umin,
            universo_max=umax,
            base_n=10,
            core_presenca_min=0.60,
        )

        if diag_j.get("rigido"):
            st.warning(f"⚠️ {diag_j.get('mensagem')}")
        else:
            st.info(f"✅ {diag_j.get('mensagem')}")

        sinais = diag_j.get("sinais") or {}
        st.session_state["anti_idx_detectados"] = (sinais.get("anti_idx_detectados") or []) if isinstance(sinais, dict) else []
        if sinais:
            st.write({
                "score_rigidez": diag_j.get("score"),
                "folga_qualitativa(alerta)": diag_j.get("folga_qualitativa"),
                "core_sz": sinais.get("core_sz"),
                "frac_colados": sinais.get("frac_colados"),
                "ov_mean": sinais.get("ov_mean"),
                "f_max": sinais.get("f_max"),
                "range_8": sinais.get("range_8"),
                "range_lim": sinais.get("range_lim"),
                "anti_idx_detectados": sinais.get("anti_idx_detectados"),
            })
    except Exception:
        st.info("Diagnóstico de rigidez indisponível nesta rodada (falha silenciosa).")
# ------------------------------------------------------------

    # ------------------------------------------------------------
    # 📊 EIXO 1 — CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
    # ------------------------------------------------------------
    try:
        listas_pacote_eixo1 = listas_m6_totais[:]
    
        historico_label = (
            f"C1 → C{len(historico_df)}"
            if historico_df is not None
            else "Histórico indefinido"
        )
    
        eixo1_resultado = calcular_eixo1_contribuicao(
            listas_pacote=listas_pacote_eixo1,
            historico_label=historico_label,
            modo_geracao="Modo 6",
            n_base=n_alvo or 6,
            eco_status=st.session_state.get("eco_status", "DESCONHECIDO"),
            estado_status=st.session_state.get("estado_atual", "DESCONHECIDO"),
        )
    except Exception:
        eixo1_resultado = None
    
    if eixo1_resultado:
        st.markdown("### 📊 Eixo 1 — Contribuição de Passageiros (Observacional)")
    
        st.write(
            f"**Núcleo local detectado:** "
            f"{'SIM' if eixo1_resultado['nucleo']['detectado'] else 'NÃO'} "
            f"({eixo1_resultado['nucleo']['tipo']})"
        )
    
        st.write(
            "**Estruturais do pacote:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["estruturais"]))
                if eixo1_resultado["papeis"]["estruturais"]
                else "—"
            )
        )
    
        st.write(
            "**Contribuintes:** "
            + (
                ", ".join(map(str, eixo1_resultado["papeis"]["contribuintes"]))
                if eixo1_resultado["papeis"]["contribuintes"]
                else "—"
            )
        )
    
        st.write(
            "**Leitura sintética:** "
            + " ".join(eixo1_resultado["leitura_sintetica"])
        )
    
        st.caption(eixo1_resultado["trava"])
    
    
    # ============================================================
    # 📌 REGISTRO CANÔNICO DO MOMENTO — DIAGNÓSTICO (COPIÁVEL)
    # ============================================================
    try:
            # ------------------------------------------------------------
            # 
            # (camada experimental removida na âncora estável)

        universo_min = st.session_state.get("universo_min", "N/D")
        universo_max = st.session_state.get("universo_max", "N/D")
        termometro_estagio = "N/D"
        termometro_score = "N/D"
        registro_txt = f"""
    SÉRIE_BASE: {serie_base}
    SÉRIES_ALVO: {series_alvo}
    
    ECO: {st.session_state.get("eco_status", "N/D")}
    ESTADO_ALVO: {st.session_state.get("estado_atual", "N/D")}
    REGIME: {st.session_state.get("pipeline_estrada", "N/D")}
    CLASSE_RISCO: {st.session_state.get("classe_risco", "N/D")}
    NR_PERCENT: {st.session_state.get("nr_percent", "N/D")}
    K_STAR: {st.session_state.get("k_star", "N/D")}
    DIVERGENCIA: {st.session_state.get("divergencia_s6_mc", "N/D")}
    UNIVERSO: {universo_min}-{universo_max}
    N_CARRO: {n_alvo if n_alvo is not None else "N/D"}
    EIXO1_NUCLEO_DETECTADO: {'SIM' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] else 'NÃO'}
    EIXO1_TIPO_NUCLEO: {eixo1_resultado['nucleo']['tipo'] if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] else 'inexistente'}
    EIXO1_PUXADORES: {', '.join(map(str, (eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes'])[:8])) if eixo1_resultado else '—'}
    EIXO1_CONVERGENCIA: {'alta' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] and len(eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes']) >= 4 else 'média' if eixo1_resultado and eixo1_resultado['nucleo']['detectado'] and len(eixo1_resultado['papeis']['estruturais'] + eixo1_resultado['papeis']['contribuintes']) >= 2 else 'baixa'}
    EIXO1_LEITURA: {' '.join(eixo1_resultado['leitura_sintetica']) if eixo1_resultado else 'pacote disperso'}
    PACOTE_BASE: Top10
    PACOTE_ANTI_ANCORA: {", ".join("L"+str(i) for i in (st.session_state.get("v16_anti_ancora") or {}).get("anti_idx", [])) or "—"}
    """.strip()
    
        st.code(registro_txt, language="text")
    
    except Exception:
        pass
    
    

    # ============================================================
    # 🧨 JANELA LOCAL DE ATAQUE + 📦 PACOTES TÉCNICOS (RF CANÔNICO)
    # ============================================================
    try:
        reg_m3 = (
            st.session_state.get("m3_regime_dx")
            or st.session_state.get("m3_regime")
            or "N/D"
        )
        classe_risco_rf = (
            st.session_state.get("classe_risco")
            or (st.session_state.get("diagnostico_risco") or {}).get("classe_risco")
            or "N/D"
        )

        analise_anti_rf = st.session_state.get("v16_anti_ancora") or {}
        core_rf = analise_anti_rf.get("core") or []
        anti_idx_rf = analise_anti_rf.get("anti_idx") or []

        # Janela ATIVA = estrutura (CORE) + contexto minimamente favorável (ECO/PRÉ-ECO) + risco não vermelho
        _reg_norm = str(reg_m3).upper().replace("É", "E")
        janela_ativa_session = st.session_state.get("janela_local_ativa")
        if janela_ativa_session is None:
            janela_ativa = bool(core_rf) and (_reg_norm in ["ECO", "PRE-ECO", "PRÉ-ECO"]) and ("🔴" not in str(classe_risco_rf))
        else:
            janela_ativa = bool(janela_ativa_session)


        st.markdown("### 🧨 Estado da Janela Local de Ataque")
        st.write(f"**Status da Janela:** {'ATIVA' if janela_ativa else 'NÃO ATIVA'}")
        st.write("**Tipo:** Local · Recortada · Observacional")
        st.write("**Base da leitura:**")
        st.write(f"- Compressão (Modo 6 / CORE dominante): {'SIM' if bool(core_rf) else 'NÃO'}")
        st.write(f"- Contexto histórico (M3 / dx): {reg_m3}")
        st.write(f"- Classe de risco: {classe_risco_rf}")
        st.caption("A existência de janela não obriga ataque. Ela apenas qualifica a leitura do momento.")

        st.markdown("### 📦 Pacotes Técnicos (classificação informativa)")

        # 🛡️ PACOTE BASE — sempre existe (Modo 6)
        st.markdown("#### 🛡️ PACOTE BASE — CANÔNICO")
        st.caption("Origem: Modo 6. Função: continuidade estatística. Sempre existe.")

        # ⚖️ ALTERNATIVO — só se janela ATIVA e houver anti-âncora clara (listas já existentes)
        st.markdown("#### ⚖️ PACOTE ALTERNATIVO — BALIZADO")
        if janela_ativa and anti_idx_rf:
            st.caption("Condição: janela ATIVA. Origem: listas existentes com baixo overlap com CORE (anti-âncora).")
            for i in anti_idx_rf[:4]:
                if 0 <= int(i) < len(listas_m6_totais):
                    st.write(f"ALT{int(i)+1}: " + formatar_lista_passageiros(listas_m6_totais[int(i)]))
        else:
            st.info("Não aplicável nesta rodada (janela não ativa ou sem material anti-âncora claro).")

        # 🔥 OFENSIVO — TURBO (se houver material)
        
        # ============================================================
        # 🧩 V8 — BORDA QUALIFICADA (ETAPA 2) — PRÉ-CAMADA 4
        # - Governança legível: explica "por que entrou"
        # - NÃO altera listas reais; só classifica borda interna/externa
        # ============================================================
        try:
            st.markdown("#### 🧩 V8 — BORDA QUALIFICADA (pré‑Camada 4)")
            st.caption("Etapa 2 do Ajuste Fino: qualidade da borda · sem motor novo · sem mexer em Modo 6/TURBO/Bala")

            pacote_base_v8 = pacote_atual if isinstance(pacote_atual, list) else None
            if not pacote_base_v8:
                st.info("V8 Bordas: pacote atual indisponível (rode o 🎯 Modo 6 nesta sessão).")
            else:
                # parâmetros conservadores
                _base_n = int(min(10, max(3, len(pacote_base_v8))))
                _core_min = float((st.session_state.get("v8_core_presenca_min") or 0.60))
                _delta = float((st.session_state.get("v8_quase_delta") or 0.12))

                # rigidez do jeitão (já existe no V7)
                rig_info = v16_diagnostico_rigidez_jeitao(
                    listas=pacote_base_v8,
                    universo_min=st.session_state.get("universo_min"),
                    universo_max=st.session_state.get("universo_max"),
                    base_n=_base_n,
                    core_presenca_min=_core_min,
                )

                v8_borda = v8_classificar_borda_qualificada(
                    listas=pacote_base_v8,
                    base_n=_base_n,
                    core_presenca_min=_core_min,
                    quase_delta=_delta,
                    max_borda_interna=6,
                    universo_min=st.session_state.get("universo_min"),
                    universo_max=st.session_state.get("universo_max"),
                    rigidez_info=rig_info,
                )

                st.session_state["v8_borda_qualificada"] = v8_borda

                meta_v8 = v8_borda.get("meta") or {}
                st.write({
                    "base_n": meta_v8.get("base_n"),
                    "core_presenca_min": meta_v8.get("core_presenca_min"),
                    "quase_delta": meta_v8.get("quase_delta"),
                    "rigidez": f"{'SIM' if meta_v8.get('rigido') else 'NÃO'} (score {meta_v8.get('score_rigidez')})",
                    "folga_qualitativa": meta_v8.get("folga_qualitativa"),
                })

                core_v8 = v8_borda.get("core") or []
                quase_v8 = v8_borda.get("quase_core") or []
                bi = v8_borda.get("borda_interna") or []
                be = v8_borda.get("borda_externa") or []

                st.write(f"**CORE (por presença):** {core_v8 if core_v8 else '—'}")
                st.write(f"**QUASE‑CORE (candidatos):** {quase_v8 if quase_v8 else '—'}")

                st.markdown("**✅ BORDA INTERNA (entra sem dispersar — sugestão observacional):**")
                if bi:
                    for p in bi:
                        motivos = (v8_borda.get("motivos_interna") or {}).get(p) or []
                        st.write(f"- **{p}** · " + " · ".join(motivos[:4]))
                else:
                    st.write("- —")

                st.markdown("**⛔ BORDA EXTERNA (não entra — risco de dispersão / distância / presença insuficiente):**")
                if be:
                    # mostra só os primeiros para não poluir RF
                    for p in be[:10]:
                        motivos = (v8_borda.get("motivos_externa") or {}).get(p) or []
                        st.write(f"- {p} · " + " · ".join(motivos[:3]))
                    if len(be) > 10:
                        st.caption(f"… +{len(be)-10} candidatos externos (ocultos p/ legibilidade).")
                else:
                    st.write("- —")

        except Exception:
            # falha silenciosa: nunca derruba o RF
            pass


        st.markdown("#### 🔥 PACOTE OFENSIVO — CONDICIONAL")
        turbo_tentado_rf = bool(st.session_state.get("turbo_tentado", False))
        if janela_ativa and (listas_ultra or ultima_prev):
            st.caption("Condição: janela ATIVA. Motor ofensivo tentado. Uso pontual e consciente.")
            if listas_ultra:
                for j, L in enumerate(listas_ultra[:6], start=1):
                    st.write(f"OF{j}: " + formatar_lista_passageiros(L))
            elif ultima_prev:
                st.write(formatar_lista_passageiros(ultima_prev))
        else:
            st.info(
                "Motor ofensivo tentado, sem material válido produzido nesta condição."
                if turbo_tentado_rf else
                "Motor ofensivo não tentado nesta rodada."
            )
            st.caption("Falha silenciosa é um resultado válido e informativo quando não há janela ofensiva.")
    except Exception:
        pass

    # ============================================================
    # 📌 LISTAS DE PREVISÃO ASSOCIADAS AO MOMENTO (COPIÁVEL)
    # ============================================================
    try:
        st.markdown("### 📌 Listas de Previsão Associadas ao Momento")
    
        listas_para_registro = []
    
        if "pacote_operacional" in locals() and pacote_operacional:
            listas_para_registro = pacote_operacional[:]
        elif listas_m6_totais:
            listas_para_registro = listas_m6_totais[:]
    
        if listas_para_registro:
            linhas_listas = []
            for i, lst in enumerate(listas_para_registro[:20], start=1):
                linhas_listas.append(
                    f"L{i}: " + ", ".join(str(x) for x in lst)
                )
    
            st.code("\n".join(linhas_listas), language="text")
        else:
            st.info("Nenhuma lista disponível para registro neste momento.")
    
    except Exception:
        pass


    # ============================================================
    # 🧠 Painel — Aptidão do Evento (CANÔNICO | SOMENTE LEITURA)
    # Avaliação AUTOMÁTICA de aptidão para Memória Operacional
    # ============================================================
    try:
        st.markdown("## 🧠 Painel de Aptidão do Evento")
    
        # -------------------------------
        # Inicialização defensiva
        # -------------------------------
        status_aptidao = "NÃO APTO"
        motivo_principal = "Critérios mínimos não atendidos"
        compatibilidade = "indefinida"
        observacao = "Leitura automática do sistema"
        eixo1_resumo = "N/D"
    
        # -------------------------------
        # Fontes (já calculadas no app)
        # -------------------------------
        eixo1_ok = bool(
            eixo1_resultado
            and eixo1_resultado.get("nucleo", {}).get("detectado", False)
        )
    
        regime = st.session_state.get("pipeline_estrada", "N/D")
        nr_percent = st.session_state.get("nr_percent", None)
        divergencia = st.session_state.get("divergencia_s6_mc", None)
    
        # -------------------------------
        # Regras de APTIDÃO (sistema decide)
        # -------------------------------
        if eixo1_ok and regime in ["🟩 Estrada Neutra / Estável", "🟨 Estrada Moderada"]:
            status_aptidao = "APTO"
            motivo_principal = "Núcleo observável + regime compatível"
    
        elif eixo1_ok and regime not in ["🟥 Estrada Ruim / Instável"]:
            status_aptidao = "APTO"
            motivo_principal = "Núcleo fraco porém reutilizável"
    
        else:
            status_aptidao = "NÃO APTO"
            motivo_principal = "Ausência de núcleo ou regime incompatível"
    
        # -------------------------------
        # Compatibilidade de densidade
        # -------------------------------
        if eixo1_ok and regime.startswith("🟩"):
            compatibilidade = "microvariações / envelope estreito"
        elif eixo1_ok:
            compatibilidade = "repescagem controlada"
        else:
            compatibilidade = "nenhuma (densidade bloqueada)"
    
        # -------------------------------
        # Resumo do EIXO 1 (canônico)
        # -------------------------------
        if eixo1_resultado:
            eixo1_resumo = (
                f"Núcleo={ 'SIM' if eixo1_resultado['nucleo']['detectado'] else 'NÃO' } | "
                f"Tipo={ eixo1_resultado['nucleo']['tipo'] } | "
                f"Puxadores="
                + (
                    ", ".join(
                        map(
                            str,
                            (
                                eixo1_resultado["papeis"]["estruturais"]
                                + eixo1_resultado["papeis"]["contribuintes"]
                            )[:6],
                        )
                    )
                    if eixo1_resultado["papeis"]["estruturais"]
                    or eixo1_resultado["papeis"]["contribuintes"]
                    else "—"
                )
            )
    
        # -------------------------------
        # Exibição CANÔNICA (sem decisão)
        # -------------------------------
        st.markdown("### 📋 Resumo Canônico de Aptidão")
    
        aptidao_txt = f"""
    STATUS_APTIDAO: {status_aptidao}
    MOTIVO_PRINCIPAL: {motivo_principal}
    EIXO1_RESUMO: {eixo1_resumo}
    COMPATIBILIDADE_DENSIDADE: {compatibilidade}
    OBSERVACAO: {observacao}
    """.strip()
    
        st.code(aptidao_txt, language="text")
    
    except Exception as e:
        st.warning("Painel de Aptidão indisponível nesta rodada.")

    
    # ------------------------------------------------------------
    # 📦 Pacote Operacional TOTAL (Modo 6 + TURBO ULTRA)
    # ------------------------------------------------------------
    pacote_operacional = listas_m6_totais.copy()

    for lst in listas_ultra:
        if lst not in pacote_operacional:
            pacote_operacional.append(lst)

    try:
        pacote_operacional = v16_priorizar_listas_por_contexto(pacote_operacional)
    except Exception:
        pass

    total_listas = len(pacote_operacional)

    # ------------------------------------------------------------
    # 🧭 PAINEL CANÔNICO — BALA HUMANO DENSO (MODO ASSISTIDO)
    # (Somente leitura | sem execução | sem recomendação)
    # ------------------------------------------------------------
    try:
        st.markdown("## 🧭 Bala Humano Denso — Modo Assistido (Painel Canônico)")

        # Leituras já existentes no sistema (somente leitura)
        diag_risco = st.session_state.get("diagnostico_risco", {}) or {}
        estrada = st.session_state.get("pipeline_estrada", "N/D")

        classe_risco = diag_risco.get("classe_risco", "N/D")
        nr_percent = diag_risco.get("nr_percent", None)
        divergencia = diag_risco.get("divergencia", None)
        indice_risco = diag_risco.get("indice_risco", None)

        # ------------------------------------------------------------
        # BLOCO 1 — Condição do Momento (sem score mágico)
        # ------------------------------------------------------------
        st.markdown("### 1️⃣ Condição do Momento")

        st.write(f"- Estrada (Pipeline): **{estrada}**")
        st.write(f"- Classe de risco (Monitor): **{classe_risco}**")

        if nr_percent is not None:
            st.write(f"- NR% (Ruído Condicional): **{float(nr_percent):.2f}%**")
        else:
            st.write("- NR% (Ruído Condicional): **N/D**")

        if divergencia is not None:
            st.write(f"- Divergência S6 vs MC: **{float(divergencia):.4f}**")
        else:
            st.write("- Divergência S6 vs MC: **N/D**")

        if indice_risco is not None:
            st.write(f"- Índice composto de risco: **{float(indice_risco):.4f}**")
        else:
            st.write("- Índice composto de risco: **N/D**")

        # Nota canônica (a comparabilidade “momento passado vs atual” entra na Fase C)
        st.info(
            "Leitura informativa: este painel descreve o terreno atual com métricas já existentes. "
            "A comparabilidade com momentos passados e a seleção automática de densidade entram na fase seguinte."
        )

        # ------------------------------------------------------------
        # BLOCO 2 — Formas de Densidade Compatíveis (canônico)
        # ------------------------------------------------------------
        st.markdown("### 2️⃣ Formas de Densidade Compatíveis (canônico)")

        st.write("- ✔ **Microvariações controladas**")
        st.write("- ✔ **Envelope estreito**")
        st.write("- ⚠ **Repescagem controlada**")
        st.write("- ❌ **Expansão de universo** (incompatível com o espírito do Bala Humano)")

        st.caption(
            "Observação: aqui ainda não há escolha automática de formato. "
            "O sistema apenas delimita o que é compatível com densidade (aprofundar, não dispersar)."
        )

        # ------------------------------------------------------------
        # BLOCO 3 — Expectativa sob Densidade (canônico)
        # ------------------------------------------------------------
        st.markdown("### 3️⃣ Expectativa sob Densidade (informativo)")

        st.write("- Redistribuição típica para **4/6**")
        st.write("- Elevação marginal de **5/6**")
        st.write("- **6/6 não observado** como viável de forma consistente neste tipo de leitura")
        st.write("- Ganho associado a **volume controlado**, não a salto de acerto")

        st.caption("Regra: densidade altera **distribuição**, não compra **certeza**.")

        # ------------------------------------------------------------
        # BLOCO 4 — Cláusula de Responsabilidade (canônico)
        # ------------------------------------------------------------
        st.markdown("### 4️⃣ Decisão Humana — Fronteira de Responsabilidade")

        st.write("- O sistema **não recomenda ação**")
        st.write("- O sistema **não define volume**")
        st.write("- O sistema **não executa automaticamente**")
        st.write("- A decisão e a exposição são do **operador**")

        st.markdown("---")

    except Exception:
        # Falha silenciosa canônica: não derruba fluxo operacional
        pass
    
    # ------------------------------------------------------------
    # 🔥 MANDAR BALA — POSTURA OPERACIONAL
    # ------------------------------------------------------------
    st.markdown("### 🔥 Mandar Bala — Postura Operacional (Ação Consciente)")

    qtd_bala = st.slider(
        "Quantas listas você quer levar para a ação nesta rodada?",
        min_value=1,
        max_value=total_listas,
        value=min(10, total_listas),
        step=1,
        key="slider_mandar_bala_restaurado",
    )

    for i, lst in enumerate(pacote_operacional[:qtd_bala], 1):
        st.markdown(f"**🔥 {i:02d})** {formatar_lista_passageiros(lst)}")

    exibir_bloco_mensagem(
        "🧩 Fechamento Operacional",
        f"- Listas disponíveis: **{total_listas}**\n"
        f"- Listas levadas para ação: **{qtd_bala}**\n\n"
        "📌 O sistema **não decide**. O operador **assume a postura**.",
        tipo="success",
    )

    # ============================================================
    # 🧠 RF-GOV — GOVERNANÇA INFORMATIVA (AVISOS | SEM EFEITO)
    # ============================================================
    try:
        st.markdown("### 🧠 RF-GOV — Governança Informativa")

        fenomeno_id = st.session_state.get("fenomeno_id", "N/D")
        alvo_atual = st.session_state.get("n_alvo", "N/D")

        eco_status = st.session_state.get("eco_status", "N/D")
        estado_status = st.session_state.get("estado_atual", "N/D")

        mo = st.session_state.get("memoria_operacional", [])
        tentativas_mesmo_alvo = [r for r in mo if r.get("alvo") == alvo_atual]

        avisos = []

        if len(tentativas_mesmo_alvo) >= 2:
            avisos.append(
                "⚠️ Múltiplas tentativas recentes para o mesmo alvo registradas."
            )

        if eco_status in ("RUIM", "DESCONHECIDO"):
            avisos.append("ℹ️ ECO desfavorável ou indefinido.")

        if estado_status in ("RÁPIDO", "INSTÁVEL"):
            avisos.append("ℹ️ Estado do alvo indica instabilidade.")

        st.info(
            f"**Fenômeno ID:** {fenomeno_id}\n\n"
            f"**Alvo:** {alvo_atual}\n\n"
            f"**ECO:** {eco_status}\n"
            f"**Estado:** {estado_status}"
        )

        for a in avisos:
            st.warning(a)

        if not avisos:
            st.success("Nenhum alerta relevante de governança nesta rodada.")

    except Exception:
        st.caption("RF-GOV indisponível nesta execução.")

    st.success("Relatório Final gerado com sucesso!")

# ============================================================
# <<< FIM — PAINEL 13 — 📘 Relatório Final
# ============================================================












# ============================================================
# Painel — ⏱️ DURAÇÃO DA JANELA — ANÁLISE HISTÓRICA (V16)
# Diagnóstico PURO | Mede quantas séries janelas favoráveis duraram
# NÃO prevê | NÃO decide | NÃO altera motores
# ============================================================

# ============================================================
# Painel — 🔍 Cruzamento Histórico do k (Observacional)
# V16 | LEITURA PURA | NÃO DECIDE | NÃO ALTERA MOTORES
# ============================================================

if painel == "🔍 Cruzamento Histórico do k":

    st.markdown("## 🔍 Cruzamento Histórico do k")
    st.caption(
        "Leitura observacional do histórico. "
        "Este painel NÃO interfere em decisões, volumes ou modos."
    )

    eventos = st.session_state.get("eventos_k_historico", [])

    if not eventos:
        exibir_bloco_mensagem(
            "Nenhum evento k encontrado",
            "Carregue o histórico para analisar os eventos k.",
            tipo="warning",
        )
        st.stop()

    df_k = pd.DataFrame(eventos)

    # ============================================================
    # FILTROS SIMPLES (OBSERVACIONAIS)
    # ============================================================
    st.markdown("### 🎛️ Filtros Observacionais")

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_estado = st.multiselect(
            "Estado do alvo",
            options=sorted(df_k["estado_alvo"].dropna().unique().tolist()),
            default=None,
        )

    with col2:
        filtro_pre_eco = st.selectbox(
            "PRÉ-ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    with col3:
        filtro_eco = st.selectbox(
            "ECO",
            options=["Todos", "Sim", "Não"],
            index=0,
        )

    df_f = df_k.copy()

    if filtro_estado:
        df_f = df_f[df_f["estado_alvo"].isin(filtro_estado)]

    if filtro_pre_eco != "Todos":
        df_f = df_f[df_f["pre_eco"] == (filtro_pre_eco == "Sim")]

    if filtro_eco != "Todos":
        df_f = df_f[df_f["eco"] == (filtro_eco == "Sim")]

    # ============================================================
    # MÉTRICAS RESUMIDAS
    # ============================================================
    st.markdown("### 📊 Resumo Estatístico")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Eventos k", len(df_f))

    with col2:
        st.metric(
            "Δ médio entre ks",
            round(df_f["delta_series"].dropna().mean(), 2)
            if "delta_series" in df_f else "—",
        )

    with col3:
        st.metric(
            "k médio",
            round(df_f["k_valor"].mean(), 2)
            if "k_valor" in df_f else "—",
        )

    with col4:
        st.metric(
            "Máx k observado",
            int(df_f["k_valor"].max())
            if "k_valor" in df_f else "—",
        )

    # ============================================================
    # TABELA FINAL (LEITURA CRUA)
    # ============================================================
    st.markdown("### 📋 Eventos k — Histórico")

    st.dataframe(
        df_f[
            [
                "serie_id",
                "k_valor",
                "delta_series",
                "estado_alvo",
                "k_star",
                "nr_percent",
                "div_s6_mc",
                "pre_eco",
                "eco",
            ]
        ].sort_values("serie_id"),
        use_container_width=True,
    )

# ============================================================
# FIM — Painel Cruzamento Histórico do k
# ============================================================


if painel == "⏱️ Duração da Janela — Análise Histórica":

    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")

    st.info(
        "Este painel mede, **no passado**, quantas séries consecutivas "
        "as janelas favoráveis **REALMENTE duraram**, após serem confirmadas.\n\n"
        "📌 Definição usada:\n"
        "- Abertura: melhora conjunta (NR%, divergência, k*, desempenho real)\n"
        "- Fechamento: perda clara dessa coerência\n\n"
        "⚠️ Este painel NÃO prevê entrada de janela."
    )

    df = st.session_state.get("historico_df")
    matriz_norm = st.session_state.get("pipeline_matriz_norm")

    if df is None or matriz_norm is None:
        exibir_bloco_mensagem(
            "Pipeline incompleto",
            "Execute **Carregar Histórico** e **Pipeline V14-FLEX ULTRA**.",
            tipo="warning",
        )
        st.stop()

    # ------------------------------------------------------------
    # Parâmetros FIXOS (diagnóstico histórico)
    # ------------------------------------------------------------
    JANELA_ANALISE = 200
    LIMIAR_NR_QUEDA = 0.02
    LIMIAR_DIV_QUEDA = 0.50

    col_pass = [c for c in df.columns if c.startswith("p")]

    # Helpers locais (réplicas leves, sem tocar no motor)
    def _nr_local(m):
        variancias = np.var(m, axis=1)
        ruido_A = float(np.mean(variancias))
        saltos = [
            np.linalg.norm(m[i] - m[i - 1]) for i in range(1, len(m))
        ]
        ruido_B = float(np.mean(saltos)) if saltos else 0.0
        return 0.55 * min(1.0, ruido_A / 0.08) + 0.45 * min(1.0, ruido_B / 1.20)

    def _div_local(m):
        base = m[-1]
        candidatos = m[-10:] if len(m) >= 10 else m
        return float(np.linalg.norm(np.mean(candidatos, axis=0) - base))

    resultados = []
    n = len(matriz_norm)

    for i in range(max(30, n - JANELA_ANALISE), n - 3):
        m_i = matriz_norm[: i + 1]
        m_f = matriz_norm[: i + 4]

        nr_i = _nr_local(m_i)
        nr_f = _nr_local(m_f)
        div_i = _div_local(m_i)
        div_f = _div_local(m_f)

        abriu = (nr_f - nr_i) < -LIMIAR_NR_QUEDA and (div_f - div_i) < -LIMIAR_DIV_QUEDA

        if abriu:
            duracao = 1
            for j in range(i + 1, n - 1):
                m_j = matriz_norm[: j + 1]
                if _nr_local(m_j) <= nr_f and _div_local(m_j) <= div_f:
                    duracao += 1
                else:
                    break

            resultados.append(duracao)

    if not resultados:
        st.warning("Nenhuma janela favorável clara detectada no período analisado.")
        st.stop()

    df_res = pd.DataFrame({"Duração (séries)": resultados})

    st.markdown("### 📊 Distribuição Histórica da Duração das Janelas")
    st.dataframe(df_res.describe(), use_container_width=True)

    st.info(
        f"📌 Total de janelas detectadas: **{len(resultados)}**\n\n"
        "Este painel responde:\n"
        "👉 *Quando a janela abre, ela costuma durar quantas séries?*\n\n"
        "Use isso para **decidir até quando mandar bala**."
    )

# ============================================================
# V16 — CAMADA D
# Estado do Alvo · Expectativa · Volume × Confiabilidade
# (FIX: usa divergência correta do Monitor de Risco)
# ============================================================

def _v16_get_nr_div_risco():
    """
    Leitura segura e compatível com o app:
    - NR% vem do Ruído Condicional (nr_percent) OU do Monitor (diagnostico_risco.nr_percent)
    - Divergência vem do Monitor (diagnostico_risco.divergencia) OU fallback (div_s6_mc)
    - Risco vem do Monitor (diagnostico_risco.indice_risco)
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}

    nr = st.session_state.get("nr_percent")
    if nr is None:
        nr = risco_pack.get("nr_percent")

    # ⚠️ FIX PRINCIPAL: no seu app a divergência oficial está aqui:
    div = risco_pack.get("divergencia")
    if div is None:
        # fallback legado (se existir em alguma variação do app)
        div = st.session_state.get("div_s6_mc")

    indice_risco = risco_pack.get("indice_risco")

    return nr, div, indice_risco


def v16_registrar_estado_alvo():
    """
    Classifica o estado do alvo com base em:
    - NR%
    - Divergência S6 vs MC
    - Índice de risco (composto)
    """
    nr, div, risco = _v16_get_nr_div_risco()

    if nr is None or div is None or risco is None:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Histórico/monitor insuficiente para classificar o alvo (rode Monitor de Risco e Ruído Condicional).",
        }
        st.session_state["estado_alvo_v16"] = estado
        return estado

    # velocidade ∈ [~0, ~1+] (heurística)
    velocidade = round((float(nr) / 100.0 + float(div) / 15.0 + float(risco)) / 3.0, 3)

    if velocidade < 0.30:
        tipo = "alvo_parado"
        comentario = "🎯 Alvo praticamente parado — oportunidade rara. Volume alto recomendado."
    elif velocidade < 0.55:
        tipo = "movimento_lento"
        comentario = "🎯 Alvo em movimento lento — alternar rajadas e coberturas."
    elif velocidade < 0.80:
        tipo = "movimento_rapido"
        comentario = "⚠️ Alvo em movimento rápido — reduzir agressividade."
    else:
        tipo = "disparado"
        comentario = "🚨 Alvo disparado — ambiente hostil. Operar apenas de forma respiratória."

    estado = {
        "tipo": tipo,
        "velocidade": velocidade,
        "comentario": comentario,
    }

    st.session_state["estado_alvo_v16"] = estado
    return estado


def v16_registrar_expectativa():
    """
    Estima expectativa de curto prazo (1–3 séries)
    com base em microjanelas, ruído e divergência.
    """
    micro = st.session_state.get("v16_microdiag") or {}
    nr, div, _ = _v16_get_nr_div_risco()

    if not micro or nr is None or div is None:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa indisponível (rode Microjanelas V16 e garanta NR/divergência).",
        }
        st.session_state["expectativa_v16"] = expectativa
        return expectativa

    score = float(micro.get("score_melhor", 0.0) or 0.0)
    janela_ouro = bool(micro.get("janela_ouro", False))

    if janela_ouro and score >= 0.80 and float(nr) < 40.0 and float(div) < 5.0:
        expectativa = {
            "previsibilidade": "alta",
            "erro_esperado": "baixo",
            "chance_janela_ouro": "alta",
            "comentario": "🟢 Forte expectativa positiva nas próximas 1–3 séries.",
        }
    elif score >= 0.50 and float(nr) < 60.0:
        expectativa = {
            "previsibilidade": "moderada",
            "erro_esperado": "moderado",
            "chance_janela_ouro": "média",
            "comentario": "🟡 Ambiente misto. Oportunidades pontuais podem surgir no curto prazo.",
        }
    else:
        expectativa = {
            "previsibilidade": "baixa",
            "erro_esperado": "alto",
            "chance_janela_ouro": "baixa",
            "comentario": "🔴 Baixa previsibilidade nas próximas 1–3 séries (ruído/divergência dominantes).",
        }

    st.session_state["expectativa_v16"] = expectativa
    return expectativa


def v16_registrar_volume_e_confiabilidade():
    """
    Relaciona quantidade de previsões com confiabilidade estimada.
    O sistema informa — a decisão é do operador.
    """
    risco_pack = st.session_state.get("diagnostico_risco") or {}
    indice = risco_pack.get("indice_risco")

    if indice is None:
        volume_op = {
            "minimo": 3,
            "recomendado": 6,
            "maximo_tecnico": 20,
            "confiabilidades_estimadas": {},
            "comentario": "Confiabilidade não calculada (rode o Monitor de Risco).",
        }
        st.session_state["volume_operacional_v16"] = volume_op
        return volume_op

    indice = float(indice)
    conf_base = max(0.05, 1.0 - indice)

    volumes = [3, 6, 10, 20, 40, 80]
    confs = {}
    for v in volumes:
        confs[v] = round(max(0.01, conf_base - v * 0.003), 3)

    recomendado = 20 if conf_base > 0.35 else 6

    volume_op = {
        "minimo": 3,
        "recomendado": int(recomendado),
        "maximo_tecnico": 80,
        "confiabilidades_estimadas": confs,
        "comentario": (
            "O sistema informa volumes e confiabilidades estimadas. "
            "A decisão final de quantas previsões gerar é do operador."
        ),
    }

    st.session_state["volume_operacional_v16"] = volume_op
    return volume_op





# ============================================================
# Painel X — 🧠 Laudo Operacional V16 (Estado, Expectativa, Volume)
# ============================================================

if painel == "🧠 Laudo Operacional V16":

    st.markdown("## 🧠 Laudo Operacional V16 — Leitura do Ambiente")

    # --------------------------------------------------------
    # Leitura segura (usa Camada D se existir, senão guarda)
    # --------------------------------------------------------
    try:
        estado = v16_registrar_estado_alvo()
    except Exception:
        estado = {
            "tipo": "indefinido",
            "velocidade": "indefinida",
            "comentario": "Estado ainda não disponível.",
        }

    try:
        expectativa = v16_registrar_expectativa()
    except Exception:
        expectativa = {
            "previsibilidade": "indefinida",
            "erro_esperado": "indefinido",
            "chance_janela_ouro": "baixa",
            "comentario": "Expectativa ainda não disponível.",
        }

    try:
        volume_op = v16_registrar_volume_e_confiabilidade()
    except Exception:
        volume_op = {
            "minimo": "-",
            "recomendado": "-",
            "maximo_tecnico": "-",
            "confiabilidades_estimadas": {},
            "comentario": "Volume ainda não disponível.",
        }

    # --------------------------------------------------------
    # 1) Estado do Alvo
    # --------------------------------------------------------
    st.markdown("### 🎯 Estado do Alvo")
    st.info(
        f"Tipo: **{estado.get('tipo')}**  \n"
        f"Velocidade estimada: **{estado.get('velocidade')}**  \n"
        f"Comentário: {estado.get('comentario')}"
    )

    # --------------------------------------------------------
    # 2) Expectativa de Curto Prazo
    # --------------------------------------------------------
    st.markdown("### 🔮 Expectativa (1–3 séries)")
    st.info(
        f"Previsibilidade: **{expectativa.get('previsibilidade')}**  \n"
        f"Erro esperado: **{expectativa.get('erro_esperado')}**  \n"
        f"Chance de janela de ouro: **{expectativa.get('chance_janela_ouro')}**  \n\n"
        f"{expectativa.get('comentario')}"
    )

    # --------------------------------------------------------
    # 3) Volume x Confiabilidade
    # --------------------------------------------------------
    st.markdown("### 📊 Volume × Confiabilidade (informativo)")

    confs = volume_op.get("confiabilidades_estimadas", {})
    if isinstance(confs, dict) and confs:
        df_conf = pd.DataFrame(
            [{"Previsões": k, "Confiabilidade estimada": v} for k, v in confs.items()]
        )
        st.dataframe(df_conf, use_container_width=True)

    st.warning(
        f"📌 Volume mínimo: **{volume_op.get('minimo')}**  \n"
        f"📌 Volume recomendado: **{volume_op.get('recomendado')}**  \n"
        f"📌 Volume máximo técnico: **{volume_op.get('maximo_tecnico')}**  \n\n"
        f"{volume_op.get('comentario')}"
    )

    st.success(
        "O PredictCars informa o ambiente e os trade-offs.\n"
        "A decisão final de quantas previsões gerar é do operador."
    )


    # --------------------------------------------------------
    # 4) Jeitão do Pacote — Rigidez (Camada 2 / observacional)
    # --------------------------------------------------------
    try:
        listas_m6_totais = (
            st.session_state.get("modo6_listas_totais")
            or st.session_state.get("modo6_listas")
            or []
        )
        umin = st.session_state.get("universo_min")
        umax = st.session_state.get("universo_max")

        if listas_m6_totais:
            st.markdown("### 🧩 Jeitão do Pacote — Rigidez (diagnóstico)")
            diag_j = v16_diagnostico_rigidez_jeitao(
                listas=listas_m6_totais,
                universo_min=umin,
                universo_max=umax,
                base_n=10,
                core_presenca_min=0.60,
            )

            st.info(
                "Alerta diagnóstico (Camada 2): quando o pacote fica rígido demais, ele pode 'acertar o jeitão' "
                "mas perder passageiros por compressão. Isso **não** decide nada — serve para governança/cobertura."
            )

            if diag_j.get("rigido"):
                st.warning(f"⚠️ {diag_j.get('mensagem')}")
            else:
                st.success(f"✅ {diag_j.get('mensagem')}")

            sinais = diag_j.get("sinais") or {}
            if sinais:
                st.write({
                    "score_rigidez": diag_j.get("score"),
                    "folga_qualitativa(alerta)": diag_j.get("folga_qualitativa"),
                    "core_sz": sinais.get("core_sz"),
                    "frac_colados": sinais.get("frac_colados"),
                    "ov_mean": sinais.get("ov_mean"),
                    "f_max": sinais.get("f_max"),
                    "range_8": sinais.get("range_8"),
                    "range_lim": sinais.get("range_lim"),
                    "anti_idx_detectados": sinais.get("anti_idx_detectados"),
                })
        else:
            # Sem pacote Modo 6 na sessão — nada a diagnosticar
            pass
    except Exception:
        # Falha silenciosa permitida (diagnóstico não pode quebrar laudo)
        pass



# ============================================================
# PARTE 7/8 — FIM
# ============================================================

# ============================================================
# PARTE 8/8 — INÍCIO
# ============================================================


# ============================================================
# 🔥 HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# NÃO PROCURAR FUNÇÃO
# NÃO SUBSTITUIR CÓDIGO EXISTENTE
# ESTE BLOCO SOBRESCREVE O COMPORTAMENTO INTERNAMENTE
# ============================================================

def _v16_exato_proxy__normalizar_serie(valor):
    """
    Converte qualquer coisa em inteiro válido de passageiro.
    Aceita:
    - int
    - float
    - string ('12', '12.0', ' 12 ')
    Retorna None se inválido.
    """
    try:
        if valor is None:
            return None
        if isinstance(valor, str):
            valor = valor.strip().replace(",", ".")
        v = int(float(valor))
        return v
    except Exception:
        return None


def _v16_exato_proxy__topk_frequentes_FIX(window_df: pd.DataFrame, cols_pass: list, top_k: int) -> set:
    freq = {}
    for c in cols_pass:
        for v in window_df[c].values:
            vv = _v16_exato_proxy__normalizar_serie(v)
            if vv is not None:
                freq[vv] = freq.get(vv, 0) + 1
    if not freq:
        return set()
    return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k])


def _v16_exato_proxy__serie_set_FIX(df_row: pd.Series, cols_pass: list) -> set:
    out = set()
    for c in cols_pass:
        vv = _v16_exato_proxy__normalizar_serie(df_row[c])
        if vv is not None:
            out.add(vv)
    return out


# 🔒 SOBRESCREVE FUNÇÕES USADAS PELO PAINEL (SEM VOCÊ CAÇAR NADA)
try:
    v16_exato_proxy__topk_frequentes = _v16_exato_proxy__topk_frequentes_FIX
    v16_exato_proxy__serie_set = _v16_exato_proxy__serie_set_FIX
except Exception:
    pass

# ============================================================
# 🔥 FIM HOTFIX DEFINITIVO — EXATO PROXY (NORMALIZAÇÃO TOTAL)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# VERSÃO FORÇADA — NÃO FICA EM BRANCO
# ============================================================

V16_PAINEL_EXATO_PROXY_NOME = "📊 V16 Premium — EXATO por Regime (Proxy)"


def v16_painel_exato_por_regime_proxy():
    st.markdown("## 📊 V16 Premium — EXATO por Regime (Proxy)")

    # --------------------------------------------------------
    # 0) Obter histórico BASE (FORÇADO)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        df_base = None

    if df_base is None or len(df_base) == 0:
        st.error("❌ Histórico não disponível. Painel abortado.")
        return

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")

    # --------------------------------------------------------
    # 1) Extração FORÇADA dos passageiros
    # Regra: colunas 1..6
    # --------------------------------------------------------
    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes.")
        return

    cols_pass = cols[1:7]
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 2) Normalização TOTAL
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip()))
        except Exception:
            return None

    # --------------------------------------------------------
    # 3) Parâmetros FIXOS
    # --------------------------------------------------------
    W = 60
    TOP_K = 12

    if len(df_base) <= W:
        st.error("❌ Histórico insuficiente para janela W=60.")
        return

    # --------------------------------------------------------
    # 4) Loop FORÇADO (sem filtros que zeram tudo)
    # --------------------------------------------------------
    registros = []

    for t in range(W, len(df_base)):
        janela = df_base.iloc[t - W : t]
        prox = df_base.iloc[t]

        freq = {}
        for c in cols_pass:
            for v in janela[c].values:
                vv = norm(v)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1

        if not freq:
            continue

        topk = set(k for k, _ in sorted(freq.items(), key=lambda x: -x[1])[:TOP_K])

        real = set()
        for c in cols_pass:
            vv = norm(prox[c])
            if vv is not None:
                real.add(vv)

        hits = len(topk & real)

        # regime SIMPLES (FORÇADO)
        if hits >= 3:
            regime = "ECO"
        elif hits >= 2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        registros.append(
            {"regime": regime, "hits": hits}
        )

    if not registros:
        st.error("❌ Nenhum registro gerado.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 5) RESULTADOS GARANTIDOS
    # --------------------------------------------------------
    resumo = []
    for r in ["ECO", "PRÉ-ECO", "RUIM"]:
        sub = df[df["regime"] == r]
        resumo.append({
            "Regime": r,
            "Eventos": len(sub),
            "Hits ≥2 (%)": round((sub["hits"] >= 2).mean() * 100, 2) if len(sub) else 0.0,
            "Hits ≥3 (%)": round((sub["hits"] >= 3).mean() * 100, 2) if len(sub) else 0.0,
        })

    df_out = pd.DataFrame(resumo)


    # --------------------------------------------------------
    # 4) Diagnóstico — Rigidez do Jeitão (folga) [OBSERVACIONAL]
    # --------------------------------------------------------
    st.markdown("### 🧩 Jeitão do Pacote — Rigidez × Folga (diagnóstico)")
    st.caption("Isso NÃO decide nem altera listas. Serve só para alertar sobre possível rigidez excessiva do pacote e sugerir 'folga' qualitativa como hipótese.")

    try:
        listas_m6_totais = (
            st.session_state.get("modo6_listas_totais")
            or st.session_state.get("modo6_listas")
            or []
        )
        if listas_m6_totais:
            umin = st.session_state.get("universo_min")
            umax = st.session_state.get("universo_max")
            diag_j = v16_diagnostico_rigidez_jeitao(
                listas=listas_m6_totais,
                universo_min=umin,
                universo_max=umax,
                base_n=10,
                core_presenca_min=0.60,
            )

            if diag_j.get("rigido"):
                st.warning(f"⚠️ {diag_j.get('mensagem')}")
            else:
                st.info(f"✅ {diag_j.get('mensagem')}")

            with st.expander("🔎 Ver sinais (auditável)"):
                st.write(diag_j.get("sinais", {}))
                st.write(f"Score: {diag_j.get('score')} | Folga (qualitativa / alerta): {diag_j.get('folga_qualitativa')}")
        else:
            st.info("Sem listas do Modo 6 nesta sessão — diagnóstico de rigidez só aparece após executar o **🎯 Modo 6**.")
    except Exception:
        st.info("Diagnóstico de rigidez indisponível nesta sessão (falha silenciosa).")

    st.markdown("### 📊 Resultado (FORÇADO)")
    st.dataframe(df_out, use_container_width=True)

    st.success("✅ Painel executado com sucesso (versão forçada).")


def v16_registrar_painel_exato_proxy__no_router():
    if st.session_state.get("_v16_exato_proxy_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g:
        orig = g["v16_obter_paineis"]

        def novo():
            try:
                lst = list(orig())
            except Exception:
                lst = []
            if V16_PAINEL_EXATO_PROXY_NOME not in lst:
                lst.append(V16_PAINEL_EXATO_PROXY_NOME)
            return lst

        g["v16_obter_paineis"] = novo

    if "v16_renderizar_painel" in g:
        orig_r = g["v16_renderizar_painel"]

        def render(p):
            if p == V16_PAINEL_EXATO_PROXY_NOME:
                return v16_painel_exato_por_regime_proxy()
            return orig_r(p)

        g["v16_renderizar_painel"] = render

    st.session_state["_v16_exato_proxy_router_ok"] = True


try:
    v16_registrar_painel_exato_proxy__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — EXATO POR REGIME (PROXY)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# (COLAR ENTRE: FIM DO EXATO PROXY  e  INÍCIO DO V16 PREMIUM PROFUNDO)
# ============================================================

V16_PAINEL_PRE_ECO_PERSIST_NOME = "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)"


def v16_painel_pre_eco_persistencia_continuidade():
    st.markdown("## 📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Qual % de **PRÉ-ECO** vira **ECO** em **1–3 séries**?
- ✅ Como separar **PRÉ-ECO fraco** vs **PRÉ-ECO forte**?
- ✅ Quais são os **últimos PRÉ-ECO fortes** (para prontidão humana)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (obrigatório)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico não tem colunas suficientes (precisa: série + 6 passageiros).")
        return

    cols_pass = cols[1:7]

    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (decisão sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    LOOKAHEAD_MAX = 3
    RUN_BACK = 5
    MAX_JANELAS = 4000  # anti-zumbi interno

    if len(df_base) <= W + LOOKAHEAD_MAX:
        st.error(f"❌ Histórico insuficiente para W={W} + lookahead.")
        return

    # Anti-zumbi: só últimas MAX_JANELAS
    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nLOOKAHEAD_MAX = {LOOKAHEAD_MAX}\nRUN_BACK = {RUN_BACK}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )

    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk, real, hits)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        return set(k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:TOP_K])

    def serie_real_set(df_row):
        out = set()
        for c in cols_pass:
            vv = norm(df_row[c])
            if vv is not None:
                out.add(vv)
        return out

    # --------------------------------------------------------
    # 4) Primeiro passe: dx_list para quantis ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime + hits por t
    # --------------------------------------------------------
    registros = []
    regime_por_t = {}
    hits_por_t = {}

    for t in range(t_inicial, t_final + 1):
        if t not in dx_por_t:
            continue

        dx = dx_por_t[t]
        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        wdf = df_base.iloc[t - W : t]
        top = topk_frequentes(wdf)
        real = serie_real_set(df_base.iloc[t])
        hits = int(len(top & real))

        regime_por_t[t] = regime
        hits_por_t[t] = hits

        registros.append({"t": int(t), "dx": float(dx), "regime": regime, "hits": hits})

    if not registros:
        st.error("❌ Não houve registros válidos.")
        return

    df = pd.DataFrame(registros)

    # --------------------------------------------------------
    # 6) Persistência PRÉ-ECO (run_len_pre)
    # --------------------------------------------------------
    run_len_pre = {}
    current = 0
    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] == "PRÉ-ECO":
            current += 1
        else:
            current = 0
        run_len_pre[t] = current

    # --------------------------------------------------------
    # 7) PRÉ-ECO → ECO em 1..3 séries (taxas)
    # --------------------------------------------------------
    total_pre = 0
    vira_eco_1 = 0
    vira_eco_2 = 0
    vira_eco_3 = 0

    eventos_pre = []

    for t in sorted(regime_por_t.keys()):
        if regime_por_t[t] != "PRÉ-ECO":
            continue

        total_pre += 1

        r1 = regime_por_t.get(t + 1)
        r2 = regime_por_t.get(t + 2)
        r3 = regime_por_t.get(t + 3)

        ok1 = (r1 == "ECO")
        ok2 = (r1 == "ECO") or (r2 == "ECO")
        ok3 = (r1 == "ECO") or (r2 == "ECO") or (r3 == "ECO")

        vira_eco_1 += 1 if ok1 else 0
        vira_eco_2 += 1 if ok2 else 0
        vira_eco_3 += 1 if ok3 else 0

        # dx trend e repetição de hits>=2 (últimos RUN_BACK)
        ts = [x for x in range(t - (RUN_BACK - 1), t + 1) if x in dx_por_t and x in hits_por_t and x in regime_por_t]
        dx_seq = [dx_por_t[x] for x in ts]
        hit_seq = [hits_por_t[x] for x in ts]
        hits_2plus = sum(1 for h in hit_seq if h >= 2)

        dx_trend = "estável"
        if len(dx_seq) >= 2:
            if dx_seq[-1] < dx_seq[0]:
                dx_trend = "caindo"
            elif dx_seq[-1] > dx_seq[0]:
                dx_trend = "subindo"

        # Score simples (informativo): persistência + hits repetidos + dx caindo
        score = 0
        score += min(run_len_pre.get(t, 0), 12)            # 0..12
        score += hits_2plus                               # 0..5
        score += 2 if dx_trend == "caindo" else 0
        score -= 2 if dx_trend == "subindo" else 0
        score += 1 if ok3 else 0

        eventos_pre.append(
            {
                "t": int(t),
                "run_len_pre": int(run_len_pre.get(t, 0)),
                "hits_t": int(hits_por_t.get(t, 0)),
                "hits_2plus_ult5": int(hits_2plus),
                "dx_trend_ult5": dx_trend,
                "vira_ECO_em_1": bool(ok1),
                "vira_ECO_em_2": bool(ok2),
                "vira_ECO_em_3": bool(ok3),
                "score_pre_forte": int(score),
            }
        )

    if total_pre == 0:
        st.error("❌ Não houve eventos PRÉ-ECO para avaliar.")
        return

    taxa1 = vira_eco_1 / total_pre
    taxa2 = vira_eco_2 / total_pre
    taxa3 = vira_eco_3 / total_pre

    st.markdown("### ✅ Taxas PRÉ-ECO → ECO (objetivas)")
    st.dataframe(
        pd.DataFrame(
            [{
                "Eventos PRÉ-ECO": int(total_pre),
                "Vira ECO em 1": round(taxa1, 4),
                "Vira ECO em 2": round(taxa2, 4),
                "Vira ECO em 3": round(taxa3, 4),
            }]
        ),
        use_container_width=True
    )

    # --------------------------------------------------------
    # 8) Top PRÉ-ECO fortes recentes (guia humano)
    # --------------------------------------------------------
    df_evt = pd.DataFrame(eventos_pre).sort_values(["t"], ascending=True)

    # Top 10 recentes com maior score
    df_top = (
        df_evt.sort_values(["score_pre_forte", "t"], ascending=[False, False])
        .head(10)
        .copy()
    )

    st.markdown("### 🟡 Top 10 PRÉ-ECO fortes (recentes / score)")
    st.dataframe(df_top, use_container_width=True)

    st.success(
        "✅ Painel PRÉ-ECO → ECO executado.\n"
        "Ele mede persistência/continuidade — a decisão de prontidão continua humana."
    )


def v16_registrar_painel_pre_eco_persist__no_router():
    """
    Integra este painel ao roteador V16 (idempotente).
    """
    if st.session_state.get("_v16_pre_eco_persist_router_ok", False):
        return

    g = globals()

    if "v16_obter_paineis" in g and callable(g["v16_obter_paineis"]):
        _orig_obter = g["v16_obter_paineis"]

        def _wrap_v16_obter_paineis__pre_eco():
            try:
                lst = list(_orig_obter())
            except Exception:
                lst = []
            if V16_PAINEL_PRE_ECO_PERSIST_NOME not in lst:
                lst.append(V16_PAINEL_PRE_ECO_PERSIST_NOME)
            return lst

        g["v16_obter_paineis"] = _wrap_v16_obter_paineis__pre_eco

    if "v16_renderizar_painel" in g and callable(g["v16_renderizar_painel"]):
        _orig_render = g["v16_renderizar_painel"]

        def _wrap_v16_renderizar_painel__pre_eco(painel_nome: str):
            if painel_nome == V16_PAINEL_PRE_ECO_PERSIST_NOME:
                return v16_painel_pre_eco_persistencia_continuidade()
            return _orig_render(painel_nome)

        g["v16_renderizar_painel"] = _wrap_v16_renderizar_painel__pre_eco

    st.session_state["_v16_pre_eco_persist_router_ok"] = True


# Registrar no router imediatamente (sem mexer em menu/motor)
try:
    v16_registrar_painel_pre_eco_persist__no_router()
except Exception:
    pass

# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PRÉ-ECO → ECO (PERSISTÊNCIA & CONTINUIDADE)
# ============================================================

# ============================================================
# 📊 BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# (COLAR IMEDIATAMENTE ANTES DE: "INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)")
# ============================================================

V16_PAINEL_ECO_RECORRENTES_NOME = "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)"


def v16_painel_passageiros_recorrentes_eco_intersecao():
    st.markdown("## 📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)")
    st.markdown(
        """
Este painel é **100% observacional** e **retrospectivo**.

Ele responde:
- ✅ Em **trechos ECO**, quais passageiros aparecem de forma **recorrente** (persistência)?
- ✅ Em blocos ECO **consecutivos**, qual é a **interseção** real dos TOP-K por janela?
- ✅ Quais são os **passageiros ECO-resilientes** (candidatos estruturais para EXATO)?

**Sem mudar motor. Sem decidir operação.**
        """
    )

    # --------------------------------------------------------
    # 0) Histórico base (robusto, sem caça)
    # --------------------------------------------------------
    try:
        nome_df, df_base = v16_identificar_df_base()
    except Exception:
        nome_df, df_base = None, None

    if df_base is None or len(df_base) == 0:
        st.warning("⚠️ Histórico não disponível. Carregue o histórico e volte aqui.")
        return

    cols = list(df_base.columns)
    if len(cols) < 7:
        st.error("❌ Histórico insuficiente: precisa de (série + 6 passageiros).")
        return

    cols_pass = cols[1:7]
    st.success(f"✔ Histórico detectado: {len(df_base)} séries")
    st.info(f"Passageiros usados: {cols_pass}")

    # --------------------------------------------------------
    # 1) Normalização TOTAL (robusta)
    # --------------------------------------------------------
    def norm(v):
        try:
            return int(float(str(v).strip().replace(",", ".")))
        except Exception:
            return None

    # --------------------------------------------------------
    # 2) Parâmetros FIXOS (sem bifurcação)
    # --------------------------------------------------------
    W = 60
    TOP_K = 12
    RUN_MIN = 3            # só consideramos "bloco ECO" com pelo menos 3 janelas ECO consecutivas
    MAX_JANELAS = 4000     # anti-zumbi interno

    if len(df_base) <= W + 5:
        st.error(f"❌ Histórico insuficiente para W={W}.")
        return

    t_final = len(df_base) - 1
    t_inicial = max(W, t_final - MAX_JANELAS)

    st.markdown("### ⚙️ Parâmetros (fixos)")
    st.code(
        f"W = {W}\nTOP_K = {TOP_K}\nRUN_MIN = {RUN_MIN}\nMAX_JANELAS = {MAX_JANELAS}",
        language="python",
    )
    st.info(f"🧱 Anti-zumbi interno: analisando t={t_inicial} até t={t_final} (máx {MAX_JANELAS} janelas).")

    # --------------------------------------------------------
    # 3) Funções internas (dx, topk)
    # --------------------------------------------------------
    def dx_janela(window_df):
        vals = []
        for c in cols_pass:
            s = [norm(x) for x in window_df[c].values]
            s = [x for x in s if x is not None]
            if len(s) >= 2:
                vals.append(float(np.std(s, ddof=1)))
        if not vals:
            return None
        return float(np.mean(vals))

    def topk_frequentes(window_df):
        freq = {}
        for c in cols_pass:
            for x in window_df[c].values:
                vv = norm(x)
                if vv is not None:
                    freq[vv] = freq.get(vv, 0) + 1
        if not freq:
            return set()
        ordenado = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return set(k for k, _ in ordenado[:TOP_K])

    # --------------------------------------------------------
    # 4) Primeiro passe: dx por t + quantis para ECO/PRE/RUIM
    # --------------------------------------------------------
    dx_list = []
    dx_por_t = {}
    for t in range(t_inicial, t_final + 1):
        wdf = df_base.iloc[t - W : t]
        dx = dx_janela(wdf)
        if dx is not None:
            dx_list.append(dx)
            dx_por_t[t] = dx

    if len(dx_list) < 80:
        st.error(f"❌ Poucas janelas válidas para quantis. Válidas: {len(dx_list)}")
        return

    q1 = float(np.quantile(dx_list, 0.33))
    q2 = float(np.quantile(dx_list, 0.66))

    st.markdown("### 🧭 Regimes por quantis (dx_janela)")
    st.info(
        f"q1 (ECO ≤): **{q1:.6f}**  \n"
        f"q2 (PRÉ-ECO ≤): **{q2:.6f}**  \n\n"
        "Regra: dx ≤ q1 → ECO | dx ≤ q2 → PRÉ-ECO | dx > q2 → RUIM"
    )

    # --------------------------------------------------------
    # 5) Segundo passe: regime por t + TOP-K por t (apenas ECO)
    # --------------------------------------------------------
    regime_por_t = {}
    top_por_t = {}

    for t in range(t_inicial, t_final + 1):
        dx = dx_por_t.get(t)
        if dx is None:
            continue

        if dx <= q1:
            regime = "ECO"
        elif dx <= q2:
            regime = "PRÉ-ECO"
        else:
            regime = "RUIM"

        regime_por_t[t] = regime

        if regime == "ECO":
            wdf = df_base.iloc[t - W : t]
            top_por_t[t] = topk_frequentes(wdf)

    if not top_por_t:
        st.warning("⚠️ Nenhuma janela ECO detectada neste recorte.")
        return

    # --------------------------------------------------------
    # 6) Detectar blocos ECO consecutivos (runs)
    # --------------------------------------------------------
    ts_eco = sorted(top_por_t.keys())

    runs = []
    start = ts_eco[0]
    prev = ts_eco[0]
    for t in ts_eco[1:]:
        if t == prev + 1:
            prev = t
        else:
            runs.append((start, prev))
            start = t
            prev = t
    runs.append((start, prev))

    # filtrar runs curtos
    runs = [r for r in runs if (r[1] - r[0] + 1) >= RUN_MIN]

    st.markdown("### 🟢 Blocos ECO consecutivos (detectados)")
    st.info(
        f"Total de runs ECO (≥ {RUN_MIN}): **{len(runs)}**  \n"
        f"Total de janelas ECO: **{len(ts_eco)}**"
    )

    if not runs:
        st.warning("⚠️ Existem janelas ECO, mas nenhuma sequência ECO longa o suficiente (RUN_MIN).")
        return

    # --------------------------------------------------------
    # 7) Para cada run ECO: interseções cumulativas e persistência
    # --------------------------------------------------------
    resumo_runs = []
    contagem_passageiros_eco = {}  # persistência global em ECO (conta presença em TOP-K por janela)
    total_janelas_eco = 0

    for (a, b) in runs:
        ts = list(range(a, b + 1))
        sets = [top_por_t[t] for t in ts if t in top_por_t]
        if len(sets) < RUN_MIN:
            continue

        # persistência global
        for s in sets:
            for p in s:
                contagem_passageiros_eco[p] = contagem_passageiros_eco.get(p, 0) + 1

        total_janelas_eco += len(sets)

        # interseções cumulativas (2..min(6, len))
        inter_2 = None
        inter_3 = None
        inter_4 = None
        inter_5 = None
        inter_6 = None

        def inter_size(n):
            if len(sets) < n:
                return None
            inter = sets[0].copy()
            for i in range(1, n):
                inter &= sets[i]
            return len(inter)

        inter_2 = inter_size(2)
        inter_3 = inter_size(3)
        inter_4 = inter_size(4)
        inter_5 = inter_size(5)
        inter_6 = inter_size(6)

        # score simples do run (informativo): inter_3 e inter_4 pesam mais
        score_run = 0
        if inter_2 is not None: score_run += inter_2
        if inter_3 is not None: score_run += 2 * inter_3
        if inter_4 is not None: score_run += 3 * inter_4

        resumo_runs.append(
            {
                "t_ini": int(a),
                "t_fim": int(b),
                "len_run": int(b - a + 1),
                "inter_2": inter_2 if inter_2 is not None else 0,
                "inter_3": inter_3 if inter_3 is not None else 0,
                "inter_4": inter_4 if inter_4 is not None else 0,
                "inter_5": inter_5 if inter_5 is not None else 0,
                "inter_6": inter_6 if inter_6 is not None else 0,
                "score_run": int(score_run),
            }
        )

    if not resumo_runs:
        st.warning("⚠️ Não consegui consolidar runs ECO (depois de filtros).")
        return

    df_runs = pd.DataFrame(resumo_runs).sort_values(["score_run", "len_run", "t_fim"], ascending=[False, False, False])

    st.markdown("### 📊 Runs ECO — Interseção TOP-K (cumulativa)")
    st.dataframe(df_runs, use_container_width=True)

    # --------------------------------------------------------
    # 8) Passageiros ECO-resilientes (persistência global em ECO)
    # --------------------------------------------------------
    st.markdown("### 🎯 Passageiros ECO-resilientes (persistência em TOP-K durante ECO)")

    if total_janelas_eco <= 0:
        st.warning("⚠️ Total de janelas ECO inválido.")
        return

    itens = []
    for p, cnt in contagem_passageiros_eco.items():
        itens.append(
            {
                "passageiro": int(p),
                "presencas_em_ECO": int(cnt),
                "taxa_presenca_ECO": round(float(cnt) / float(total_janelas_eco), 4),
            }
        )

    df_p = pd.DataFrame(itens).sort_values(["taxa_presenca_ECO", "presencas_em_ECO", "passageiro"], ascending=[False, False, True])

    st.info(f"Total de janelas ECO consideradas (em runs): **{total_janelas_eco}**")
    st.dataframe(df_p.head(25), use_container_width=True)

    # lista curta (top 12)
    top12 = df_p.head(12)["passageiro"].tolist()
    st.success("✅ Lista curta (TOP 12 ECO-resilientes) — informativa (não é previsão):")
    st.code(", ".join(str(x) for x in top12))

    st.success(
        "✅ Painel Passageiros Recorrentes em ECO executado.\n"
        "Ele mede persistência/interseção — a decisão de ataque e montagem para 6 continua humana."
    )


# ============================================================
# 📊 FIM DO BLOCO NOVO — V16 PREMIUM — PASSAGEIROS RECORRENTES EM ECO (INTERSEÇÃO)
# ============================================================


# ============================================================
# INÍCIO DO PAINEL V16 PREMIUM PROFUNDO  (COLAR AQUI)
# ============================================================

# ============================================================
# PAINEL — 🔮 V16 Premium Profundo — Diagnóstico & Calibração
# ============================================================
if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    st.markdown("## 🔮 V16 Premium Profundo — Diagnóstico & Calibração")
    st.markdown(
        """
        Este painel **não altera nada do fluxo V15.7 MAX**.

        Ele serve para:
        - 📊 **Inspecionar o histórico ativo** (tamanho, colunas, distribuição de k),
        - 🛡️ **Verificar rapidamente o regime de risco potencial** para o TURBO++ e Modo 6 Acertos,
        - 📐 **Organizar informações de confiabilidade/QDS/k*** já calculadas em outros painéis.

        Tudo com **anti-zumbi interno**, rodando apenas em uma janela segura do histórico.
        """
    )

    # --------------------------------------------------------
    # 1) Descobrir automaticamente qual DF de histórico usar
    # --------------------------------------------------------
    nome_df, df_base = v16_identificar_df_base()

    if df_base is None:
        st.warning(
            "⚠️ Não encontrei nenhum DataFrame de histórico ativo em `st.session_state`.\n\n"
            "Use primeiro um painel que carregue o histórico (por exemplo, **Carregar Histórico**), "
            "e depois volte aqui."
        )
        st.stop()

    st.info(
        f"📁 DataFrame detectado para diagnóstico: **{nome_df}**  \n"
        f"Séries totais disponíveis: **{len(df_base)}**"
    )

    # --------------------------------------------------------
    # 2) Controle Anti-Zumbi V16 (apenas para este painel)
    # --------------------------------------------------------
    n_total = int(len(df_base))
    limite_max_slider = int(min(6000, max(500, n_total)))

    st.markdown("### 🛡️ Anti-zumbi V16 — Janela de Diagnóstico")

    limite_linhas = st.slider(
        "Quantidade máxima de séries a considerar no diagnóstico (janela final do histórico):",
        min_value=200,
        max_value=limite_max_slider,
        value=min(2000, limite_max_slider),
        step=100,
    )

    # --------------------------------------------------------
    # 3) Resumo básico do histórico (janela segura)
    # --------------------------------------------------------
    resumo = v16_resumo_basico_historico(df_base, limite_linhas=limite_linhas)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Séries totais no histórico", resumo.get("n_total", 0))
    with col2:
        st.metric("Séries usadas no diagnóstico", resumo.get("n_usado", 0))
    with col3:
        st.metric("Qtd. de colunas detectadas", len(resumo.get("colunas", [])))

    st.markdown("### 🧬 Colunas detectadas na janela de diagnóstico")
    st.write(resumo.get("colunas", []))

    # Distribuição de k (se existir)
    dist_k = resumo.get("dist_k", {})
    if dist_k:
        st.markdown("### 🎯 Distribuição de k (janela final do histórico)")
        df_k = pd.DataFrame(
            {"k": list(dist_k.keys()), "qtd": list(dist_k.values())}
        ).sort_values("k")
        df_k["proporção (%)"] = (df_k["qtd"] / df_k["qtd"].sum() * 100).round(2)
        st.dataframe(df_k, use_container_width=True)
    else:
        st.info("ℹ️ Não foi possível calcular a distribuição de k.")

    # --------------------------------------------------------
    # 4) Mapa rápido de confiabilidade / QDS / k*
    # --------------------------------------------------------
    st.markdown("### 🧠 Mapa rápido de confiabilidade (session_state)")

    with st.expander("Ver variáveis relevantes detectadas"):
        mapeamento_conf = v16_mapear_confiabilidade_session_state()
        if not mapeamento_conf:
            st.write("Nenhuma variável relevante encontrada.")
        else:
            st.json(mapeamento_conf)

    # --------------------------------------------------------
    # 5) Interpretação qualitativa do regime
    # --------------------------------------------------------
    st.markdown("### 🩺 Interpretação qualitativa do regime")
    comentario_regime = []

    if dist_k:
        total_k = sum(dist_k.values())
        denom_k = max(1, int(total_k))
        proporcao_k_alto = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val >= 3) / denom_k * 100,
            2,
        )
        proporcao_k_baixo = round(
            sum(qtd for k_val, qtd in dist_k.items() if k_val <= 1) / denom_k * 100,
            2,
        )

        comentario_regime.append(f"- k ≥ 3: **{proporcao_k_alto}%**")
        comentario_regime.append(f"- k ≤ 1: **{proporcao_k_baixo}%**")

        if proporcao_k_alto >= 35:
            comentario_regime.append("- 🟢 Regime mais estável.")
        elif proporcao_k_baixo >= 50:
            comentario_regime.append("- 🔴 Regime turbulento.")
        else:
            comentario_regime.append("- 🟡 Regime intermediário.")
    else:
        comentario_regime.append("- ℹ️ Sem dados suficientes para avaliar o regime.")

    st.markdown("\n".join(comentario_regime))

    st.success("Painel V16 Premium Profundo executado com sucesso!")
    st.stop()

# ======================================================================
# 📊 V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS (OBSERVACIONAL)
# (CTRL+F ESTE BLOCO)
# ======================================================================

def _v16_laplace_rate(sucessos: int, total: int, alpha: int = 1) -> float:
    # Suavização Laplace: (a+α)/(A+2α)
    if total <= 0:
        return 0.0
    return float((sucessos + alpha) / (total + 2 * alpha))

def _v16_wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    # Wilson score interval para proporção
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    margin = (z / denom) * math.sqrt((p*(1-p)/n) + (z**2)/(4*(n**2)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)

def _v16_delta_ci_worstcase(p1_ci: Tuple[float, float], p0_ci: Tuple[float, float]) -> Tuple[float, float]:
    # IC conservador para Δ = P1 - P0 usando pior caso:
    # Δ_lo = P1_lo - P0_hi ; Δ_hi = P1_hi - P0_lo
    return (p1_ci[0] - p0_ci[1], p1_ci[1] - p0_ci[0])

def _v16_safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def _v16_build_pre_eco_mask(df_ctx: pd.DataFrame,
                           teto_nr: float,
                           teto_div: float,
                           kstar_delta_max: float = 0.0) -> pd.Series:
    """
    PRÉ-ECO = prontidão objetiva:
      - NR% não explode
      - Divergência não hostil
      - k* não piora (Δk* <= kstar_delta_max)
      - Laudo não hostil (se existir coluna)
    """
    # Colunas esperadas (se existirem): 'kstar', 'nr', 'div', 'laudo_hostil'
    nr = df_ctx["nr"] if "nr" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    div = df_ctx["div"] if "div" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))
    kstar = df_ctx["kstar"] if "kstar" in df_ctx.columns else pd.Series([np.nan]*len(df_ctx))

    # Δk*
    kstar_prev = kstar.shift(1)
    dk = (kstar - kstar_prev)

    ok_nr = nr.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_nr)
    ok_div = div.apply(lambda v: _v16_safe_float(v, 999.0) <= teto_div)
    ok_k = dk.apply(lambda v: _v16_safe_float(v, 999.0) <= kstar_delta_max)

    if "laudo_hostil" in df_ctx.columns:
        # laudo_hostil True = hostil, então queremos False
        ok_laudo = (~df_ctx["laudo_hostil"].fillna(False)).astype(bool)
    else:
        ok_laudo = pd.Series([True]*len(df_ctx))

    preeco = (ok_nr & ok_div & ok_k & ok_laudo)
    return preeco

def _v16_hits_exatos(car_a: List[int], car_b: List[int]) -> int:
    # acertos exatos = interseção simples
    sa = set(car_a)
    sb = set(car_b)
    return len(sa.intersection(sb))

def _v16_extract_car_numbers(row: Any) -> List[int]:
    """
    Extrator robusto: tenta pegar lista/tupla/np.array; se for string, tenta parsear dígitos.
    Mantém só ints >=0.
    """
    if row is None:
        return []
    if isinstance(row, (list, tuple, np.ndarray)):
        out = []
        for v in row:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(row, str):
        # Extrai números inteiros de uma string
        nums = []
        cur = ""
        for ch in row:
            if ch.isdigit():
                cur += ch
            else:
                if cur != "":
                    nums.append(int(cur))
                    cur = ""
        if cur != "":
            nums.append(int(cur))
        return nums
    # fallback
    try:
        return [int(row)]
    except Exception:
        return []

def _v16_compute_contrib_table(historico_carros: List[List[int]],
                               df_ctx: pd.DataFrame,
                               preeco_mask: pd.Series,
                               w: int = 60,
                               alpha: int = 1,
                               amin: int = 6,
                               bmin: int = 20) -> pd.DataFrame:
    """
    Contribuição de passageiros no PRÉ-ECO:
      Para cada t (dentro janela), observa passageiros do carro real em t,
      e mede hit2/hit3 do próximo alvo (t+1).
    """
    n = len(historico_carros)
    if n < (w + 2):
        return pd.DataFrame()

    # Índices válidos: precisamos de t e t+1 existirem
    t_ini = max(0, n - w - 2)
    t_fim = n - 2  # último t que ainda tem t+1

    # Subconjunto analisado
    idxs = list(range(t_ini, t_fim + 1))

    # PRÉ-ECO alinhado em t
    preeco_sub = preeco_mask.iloc[idxs].reset_index(drop=True) if len(preeco_mask) >= n else pd.Series([False]*len(idxs))

    # Monta targets hit2/hit3 do alvo (t+1) com referência no t?
    # Aqui seguimos a definição observacional: hits exatos entre carro(t) e carro(t+1).
    # (Não é acerto do sistema; é dinâmica do alvo entre séries consecutivas.)
    hit2 = []
    hit3 = []
    passageiros_t = []

    for t in idxs:
        car_t = historico_carros[t]
        car_next = historico_carros[t+1]
        h = _v16_hits_exatos(car_t, car_next)
        hit2.append(1 if h >= 2 else 0)
        hit3.append(1 if h >= 3 else 0)
        passageiros_t.append(set(car_t))

    # Filtra só PRÉ-ECO
    rows = []
    for i, t in enumerate(idxs):
        if bool(preeco_sub.iloc[i]):
            rows.append((i, passageiros_t[i], hit2[i], hit3[i]))

    if len(rows) < 5:
        return pd.DataFrame()

    # Universo de passageiros observados no PRÉ-ECO
    universo = set()
    for _, ps, _, _ in rows:
        universo |= set(ps)
    universo = sorted(list(universo))

    # Base rates (para suporte)
    base_hit2 = sum(r[2] for r in rows) / max(1, len(rows))
    base_hit3 = sum(r[3] for r in rows) / max(1, len(rows))

    # Para cada passageiro p: conta A/B/a/b para hit2 e hit3
    data = []
    for p in universo:
        A = 0
        B = 0

        a2 = 0
        b2 = 0
        a3 = 0
        b3 = 0

        for _, ps, y2, y3 in rows:
            if p in ps:
                A += 1
                a2 += y2
                a3 += y3
            else:
                B += 1
                b2 += y2
                b3 += y3

        # Gates
        if A < amin or B < bmin:
            cls = "INSUFICIENTE"
        else:
            cls = "PENDENTE"  # define abaixo

        # Taxas suavizadas
        p1_2 = _v16_laplace_rate(a2, A, alpha=alpha)
        p0_2 = _v16_laplace_rate(b2, B, alpha=alpha)
        p1_3 = _v16_laplace_rate(a3, A, alpha=alpha)
        p0_3 = _v16_laplace_rate(b3, B, alpha=alpha)

        # Lifts
        lift2 = (p1_2 / p0_2) if p0_2 > 0 else np.nan
        lift3 = (p1_3 / p0_3) if p0_3 > 0 else np.nan

        # IC Wilson para proporções (usando p sem Laplace para CI, mais “puro”)
        raw_p1_2 = (a2 / A) if A > 0 else 0.0
        raw_p0_2 = (b2 / B) if B > 0 else 0.0
        raw_p1_3 = (a3 / A) if A > 0 else 0.0
        raw_p0_3 = (b3 / B) if B > 0 else 0.0

        ci_p1_2 = _v16_wilson_ci(raw_p1_2, A)
        ci_p0_2 = _v16_wilson_ci(raw_p0_2, B)
        ci_p1_3 = _v16_wilson_ci(raw_p1_3, A)
        ci_p0_3 = _v16_wilson_ci(raw_p0_3, B)

        # Δ e IC conservador
        d2 = p1_2 - p0_2
        d3 = p1_3 - p0_3

        ci_d2 = _v16_delta_ci_worstcase(ci_p1_2, ci_p0_2)
        ci_d3 = _v16_delta_ci_worstcase(ci_p1_3, ci_p0_3)

        # Score (z aprox): z = Δ / SE(Δ) (SE aprox com raw, para não “embelezar”)
        se2 = math.sqrt((raw_p1_2*(1-raw_p1_2)/max(1, A)) + (raw_p0_2*(1-raw_p0_2)/max(1, B)))
        se3 = math.sqrt((raw_p1_3*(1-raw_p1_3)/max(1, A)) + (raw_p0_3*(1-raw_p0_3)/max(1, B)))

        z2 = ( (raw_p1_2 - raw_p0_2) / se2 ) if se2 > 0 else 0.0
        z3 = ( (raw_p1_3 - raw_p0_3) / se3 ) if se3 > 0 else 0.0

        score = (2.0 * z3) + (1.0 * z2)

        # Classificação (só se não for insuficiente)
        if cls != "INSUFICIENTE":
            # Regras conservadoras (fixas)
            leader = (ci_d3[0] > 0.0) and (not np.isnan(lift3)) and (lift3 >= 1.10) and (score >= 1.0)
            discard = (ci_d3[1] < 0.0) and (not np.isnan(lift3)) and (lift3 <= 0.90) and (score <= -1.0)

            if leader:
                cls = "LÍDER"
            elif discard:
                cls = "DESCARTÁVEL"
            else:
                cls = "NEUTRO"

        data.append({
            "passageiro": int(p),
            "A_presente": int(A),
            "a_hit2": int(a2),
            "a_hit3": int(a3),
            "B_ausente": int(B),
            "b_hit2": int(b2),
            "b_hit3": int(b3),
            "P1_hit2": float(p1_2),
            "P0_hit2": float(p0_2),
            "Δ_hit2": float(d2),
            "Lift_hit2": float(lift2) if not np.isnan(lift2) else np.nan,
            "ICΔ_hit2_lo": float(ci_d2[0]),
            "ICΔ_hit2_hi": float(ci_d2[1]),
            "P1_hit3": float(p1_3),
            "P0_hit3": float(p0_3),
            "Δ_hit3": float(d3),
            "Lift_hit3": float(lift3) if not np.isnan(lift3) else np.nan,
            "ICΔ_hit3_lo": float(ci_d3[0]),
            "ICΔ_hit3_hi": float(ci_d3[1]),
            "z_hit2": float(z2),
            "z_hit3": float(z3),
            "score": float(score),
            "classe": cls,
            "base_hit2_preEco": float(base_hit2),
            "base_hit3_preEco": float(base_hit3),
        })

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Ordenação: primeiro líderes por score, depois neutros, depois descartáveis, depois insuficientes
    ordem = {"LÍDER": 0, "NEUTRO": 1, "DESCARTÁVEL": 2, "INSUFICIENTE": 3}
    df["ordem_classe"] = df["classe"].map(ordem).fillna(9).astype(int)
    df = df.sort_values(by=["ordem_classe", "score"], ascending=[True, False]).drop(columns=["ordem_classe"])
    return df

def _v16_pairwise_coocorrencia(preeco_rows_passageiros: List[set], top_n: int = 25) -> pd.DataFrame:
    """
    Coocorrência (Jaccard) entre passageiros dentro do PRÉ-ECO.
    Retorna top pares com maior Jaccard (para alertar líder condicionado).
    """
    if len(preeco_rows_passageiros) < 8:
        return pd.DataFrame()

    # Universo
    uni = set()
    for s in preeco_rows_passageiros:
        uni |= set(s)
    uni = sorted(list(uni))

    # Contagens de presença
    pres = {p: 0 for p in uni}
    for s in preeco_rows_passageiros:
        for p in s:
            pres[p] += 1

    # Pairs
    pairs = []
    uni_len = len(uni)
    for i in range(uni_len):
        p = uni[i]
        for j in range(i+1, uni_len):
            q = uni[j]
            inter = 0
            union = 0
            for s in preeco_rows_passageiros:
                ip = (p in s)
                iq = (q in s)
                if ip or iq:
                    union += 1
                    if ip and iq:
                        inter += 1
            if union > 0:
                jac = inter / union
                if jac > 0:
                    pairs.append((p, q, inter, union, jac))

    if not pairs:
        return pd.DataFrame()

    dfp = pd.DataFrame(pairs, columns=["p", "q", "inter", "union", "jaccard"])
    dfp = dfp.sort_values(by="jaccard", ascending=False).head(top_n)
    return dfp

# ----------------------------------------------------------------------
# 📊 PAINEL — V16 PREMIUM — PRÉ-ECO | CONTRIBUIÇÃO DE PASSAGEIROS
# ----------------------------------------------------------------------
if "painel" in locals() and painel == "📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros":
    st.title("📊 V16 Premium — PRÉ-ECO | Contribuição de Passageiros")
    st.caption("Observacional, retrospectivo, objetivo e replicável. ❌ Sem motor. ❌ Sem listas. ✅ Só EXATO (Hit2/Hit3).")

    # -----------------------------
    # Parâmetros FIXOS (comando)
    # -----------------------------
    W_FIXO = 60
    ALPHA = 1
    AMIN = 6
    BMIN = 20

    with st.expander("🔒 Critério fixo (transparência total)", expanded=True):
        st.markdown(
            f"""
- **Janela W:** `{W_FIXO}` (fixo)
- **Suavização Laplace α:** `{ALPHA}` (fixo)
- **Amin / Bmin:** `{AMIN}` / `{BMIN}` (fixo)
- **Foco:** Hit3 (peso 2) + Hit2 (peso 1) → **score**
- **PRÉ-ECO:** filtro objetivo (NR, divergência, Δk*, laudo hostil se existir)
"""
        )

    # -----------------------------
    # Coleta do histórico (somente leitura)
    # -----------------------------
    # Tentamos chaves prováveis sem quebrar o app
    historico_carros = None

    # Opção 1: já existe lista pronta em session_state
    for k in ["historico_carros", "historico", "carros_historico", "dados_historico_carros"]:
        if k in st.session_state and st.session_state[k] is not None:
            historico_carros = st.session_state[k]
            break

    # Opção 2: tenta montar a partir de um DataFrame de histórico
    if historico_carros is None:
        for kdf in ["df_historico", "df", "dados", "historico_df"]:
            if kdf in st.session_state and isinstance(st.session_state[kdf], pd.DataFrame):
                dfh = st.session_state[kdf].copy()
                # Tenta inferir colunas com números
                cols_num = [c for c in dfh.columns if str(c).lower().strip() in ["n1","n2","n3","n4","n5","n6","a","b","c","d","e","f"]]
                if len(cols_num) >= 5:
                    historico_carros = []
                    for _, r in dfh.iterrows():
                        car = []
                        for c in cols_num[:6]:
                            try:
                                car.append(int(r[c]))
                            except Exception:
                                pass
                        historico_carros.append(car)
                break

    if not historico_carros or len(historico_carros) < (W_FIXO + 2):
        st.warning("Histórico insuficiente para o painel (precisa de W+2 séries). Carregue histórico completo e rode novamente.")
        st.stop()

    n_total = len(historico_carros)
    st.info(f"📁 Histórico detectado: **{n_total} séries**. Janela analisada: **últimas {W_FIXO} séries úteis (com alvo t+1)**.")

    # -----------------------------
    # Contexto de métricas (k*, NR, diverg, laudo)
    # -----------------------------
    # Este painel NÃO inventa métricas: ele lê o que existir.
    # Se não existir, ele opera com defaults conservadores → PRÉ-ECO vira “raríssimo” (ou vazio).
    df_ctx = pd.DataFrame({"idx": list(range(n_total))})

    # Tenta puxar séries de k*, NR, divergência, laudo hostil (se já existirem no seu app)
    # Chaves prováveis (mantendo robusto)
    series_map = [
        ("kstar", ["kstar_series", "serie_kstar", "kstar_hist", "kstar_por_serie"]),
        ("nr",    ["nr_series", "serie_nr", "nr_hist", "nr_por_serie"]),
        ("div",   ["div_series", "serie_div", "div_hist", "divergencia_series", "div_s6_mc_series"]),
        ("laudo_hostil", ["laudo_hostil_series", "serie_laudo_hostil"]),
    ]

    for col, keys in series_map:
        val = None
        for kk in keys:
            if kk in st.session_state and st.session_state[kk] is not None:
                val = st.session_state[kk]
                break
        if val is not None:
            try:
                s = pd.Series(list(val))
                if len(s) >= n_total:
                    s = s.iloc[:n_total]
                else:
                    # completa com NaN
                    s = s.reindex(range(n_total))
                df_ctx[col] = s
            except Exception:
                pass

    # Tetos PRÉ-ECO (fixos/visíveis — mas não “otimizáveis”)
    # Se você já tiver tetos globais no app, você pode substituir por leitura deles.
    teto_nr = 0.20
    teto_div = 0.35

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("🔎 Teto NR% (PRÉ-ECO)", f"{teto_nr:.2f}")
    with colB:
        st.metric("🔎 Teto Diverg (PRÉ-ECO)", f"{teto_div:.2f}")
    with colC:
        st.metric("🔎 Δk* máx (PRÉ-ECO)", "≤ 0.00")

    preeco_mask = _v16_build_pre_eco_mask(df_ctx=df_ctx, teto_nr=teto_nr, teto_div=teto_div, kstar_delta_max=0.0)

    # Aplica janela W (final do histórico)
    t_ini = max(0, n_total - W_FIXO - 2)
    t_fim = n_total - 2
    preeco_sub = preeco_mask.iloc[t_ini:t_fim+1].reset_index(drop=True)

    qtd_preeco = int(preeco_sub.sum())
    st.success(f"🟡 Rodadas PRÉ-ECO detectadas (na janela): **{qtd_preeco}** / {len(preeco_sub)}")

    if qtd_preeco < 5:
        st.warning("PRÉ-ECO muito raro nesta janela (ou métricas ausentes). O painel mantém honestidade: sem base, sem classificação forte.")
        # ainda assim tentamos rodar; provavelmente vai dar vazio/insuficiente.

    # -----------------------------
    # Calcula tabela de contribuição
    # -----------------------------
    df_contrib = _v16_compute_contrib_table(
        historico_carros=historico_carros,
        df_ctx=df_ctx,
        preeco_mask=preeco_mask,
        w=W_FIXO,
        alpha=ALPHA,
        amin=AMIN,
        bmin=BMIN
    )

    if df_contrib.empty:
        st.warning("Sem dados suficientes para medir contribuição (PRÉ-ECO insuficiente ou janela curta).")
        st.stop()

    # -----------------------------
    # Visões (Líder / Neutro / Descartável / Insuficiente)
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏁 LÍDERES", int((df_contrib["classe"] == "LÍDER").sum()))
    with c2:
        st.metric("⚪ NEUTROS", int((df_contrib["classe"] == "NEUTRO").sum()))
    with c3:
        st.metric("❌ DESCARTÁVEIS", int((df_contrib["classe"] == "DESCARTÁVEL").sum()))
    with c4:
        st.metric("🟡 INSUF.", int((df_contrib["classe"] == "INSUFICIENTE").sum()))

    st.markdown("### 🧾 Tabela completa (ordenada por classe → score)")
    st.dataframe(
        df_contrib,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 🏁 Top LÍDERES (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "LÍDER"].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### ❌ Top DESCARTÁVEIS (PRÉ-ECO)")
    st.dataframe(
        df_contrib[df_contrib["classe"] == "DESCARTÁVEL"].head(25),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Coocorrência (Líder condicionado)
    # -----------------------------
    st.markdown("---")
    st.markdown("### 🔗 Coocorrência (Jaccard) — alerta de “líder condicionado”")

    # Reconstroi sets PRÉ-ECO na janela
    idxs = list(range(t_ini, t_fim + 1))
    preeco_rows_sets = []
    for t in idxs:
        if bool(preeco_mask.iloc[t]):
            preeco_rows_sets.append(set(historico_carros[t]))

    df_pairs = _v16_pairwise_coocorrencia(preeco_rows_sets, top_n=30)
    if df_pairs.empty:
        st.info("Coocorrência insuficiente para análise robusta nesta janela (ou PRÉ-ECO raro).")
    else:
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
        st.caption("Quanto maior o Jaccard, mais “colados” os passageiros aparecem. Isso NÃO é corte — é alerta observacional.")

    st.markdown("---")
    st.caption("🔒 Este painel é 100% observacional: não gera listas, não decide, não altera motor. Ele mede contribuição condicional no PRÉ-ECO (Hit2/Hit3).")

# ============================================================
# 📊 V16 PREMIUM — ANTI-EXATO | PASSAGEIROS NOCIVOS CONSISTENTES
# ============================================================
if painel == "📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos":

    st.title("📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos Consistentes",
    "🧪 MC Observacional do Pacote (pré-C4) — Rigidez × Nocivos × λ*")
    st.caption(
        "Observacional • Retrospectivo • Objetivo\n"
        "Identifica passageiros que REDUZEM a chance de EXATO (≥2 / ≥3).\n"
        "❌ Não gera listas • ❌ Não decide • ✅ Apoia limpeza do Modo 6"
    )

    # --------------------------------------------------------
    # Parâmetros FIXOS (canônicos)
    # --------------------------------------------------------
    W = 60
    ALPHA = 1
    AMIN = 12
    BMIN = 40

    st.markdown(
        f"""
**Critério fixo**
- Janela: **{W}**
- Suavização Laplace: **α = {ALPHA}**
- Amostra mínima: **A ≥ {AMIN}**, **B ≥ {BMIN}**
- Evento-alvo: **Hit3 (principal)** + Hit2 (suporte)
"""
    )

    # --------------------------------------------------------
    # Fonte CANÔNICA de passageiros (Pipeline V14-FLEX ULTRA)
    # --------------------------------------------------------
    if "pipeline_col_pass" not in st.session_state:
        st.warning(
            "Fonte canônica de passageiros não encontrada.\n\n"
            "Execute primeiro o painel 🛣️ Pipeline V14-FLEX ULTRA."
        )
        st.stop()

    col_pass = st.session_state["pipeline_col_pass"]

    nome_df, df_base = v16_identificar_df_base()
    if df_base is None:
        st.warning("Histórico não encontrado. Carregue o histórico e rode o Pipeline.")
        st.stop()

    if len(col_pass) < 6:
        st.warning("Fonte de passageiros inválida (menos de 6 colunas).")
        st.stop()

    historico = df_base[col_pass].astype(int).values.tolist()
    n = len(historico)

    if n < (W + 2):
        st.warning("Histórico insuficiente para análise ANTI-EXATO.")
        st.stop()

    # --------------------------------------------------------
    # Construção das janelas móveis
    # --------------------------------------------------------
    def contar_hits(car_a, car_b):
        return len(set(car_a).intersection(set(car_b)))

    resultados = []

    for t in range(n - W - 1, n - 1):
        janela = historico[t - W + 1 : t + 1]
        alvo = historico[t + 1]

        for car in janela:
            hits = contar_hits(car, alvo)
            resultados.append({
                "passageiros": car,
                "hit2": 1 if hits >= 2 else 0,
                "hit3": 1 if hits >= 3 else 0,
            })

    df = pd.DataFrame(resultados)

    universo = sorted({p for car in df["passageiros"] for p in car})

    linhas = []

    for p in universo:
        presente = df["passageiros"].apply(lambda x: p in x)

        A = int(presente.sum())
        B = int((~presente).sum())

        if A < AMIN or B < BMIN:
            classe = "INSUFICIENTE"
        else:
            a3 = df.loc[presente, "hit3"].sum()
            b3 = df.loc[~presente, "hit3"].sum()

            p1 = (a3 + ALPHA) / (A + 2 * ALPHA)
            p0 = (b3 + ALPHA) / (B + 2 * ALPHA)

            delta = p1 - p0
            lift = p1 / p0 if p0 > 0 else 1.0

            if delta < 0 and lift <= 0.92:
                classe = "NOCIVO CONSISTENTE"
            else:
                classe = "NEUTRO"

        linhas.append({
            "passageiro": p,
            "A_presente": A,
            "B_ausente": B,
            "classe": classe,
        })

    df_out = pd.DataFrame(linhas).sort_values("classe")

    # Persistência canônica (para BLOCO C mínimo)
    try:
        _noc = df_out[df_out["classe"] == "NOCIVO CONSISTENTE"]["passageiro"].astype(int).tolist() if ("classe" in df_out.columns and "passageiro" in df_out.columns) else []
        st.session_state["anti_exato_nocivos_consistentes"] = sorted(list(dict.fromkeys(_noc)))
        st.session_state["anti_exato_df"] = df_out.copy()
    except Exception:
        pass


    st.markdown("### 🧾 Classificação de Passageiros")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.markdown(
        """
🧠 **Como usar este painel**
- Passageiros **NOCIVOS CONSISTENTES** são candidatos a **EVITAR** no Modo 6
- Não é corte automático
- Serve para **limpar listas**, não para criar novas
"""
    )

# ============================================================
# PAINEL — 🌐 MODO UNIVERSAL — AVALIAÇÃO OBSERVACIONAL
# (N-AGNÓSTICO • SEM DECISÃO • SEM GERAÇÃO)
# ============================================================

if painel == "🌐 Modo Universal — Avaliação Observacional":

    st.title("🌐 Modo Universal — Avaliação Observacional")
    st.caption(
        "Observacional • N-agnóstico • Sem geração • Sem decisão\n"
        "Avalia listas existentes contra o alvo real (n_real)."
    )

    df = st.session_state.get("historico_df")
    n_real = st.session_state.get("n_alvo")
    listas = st.session_state.get("modo6_listas_totais") or []

    if df is None or n_real is None:
        st.warning(
            "Histórico ou n_real não disponível.\n\n"
            "Carregue o histórico antes de usar este painel."
        )
        st.stop()

    # -----------------------------
    # Alvo real (última série válida)
    # -----------------------------
    col_pass = [c for c in df.columns if c.startswith("p")]
    alvo_real = (
        df[col_pass]
        .iloc[-1]
        .dropna()
        .astype(int)
        .tolist()
    )

    if not listas:
        st.info("Nenhuma lista disponível para avaliação.")
        st.stop()

    # -----------------------------
    # Orçamento manual (opcional)
    # -----------------------------
    st.subheader("🔢 Orçamento (opcional)")
    orcamento_manual = st.text_input(
        "Informe um orçamento manual (opcional)",
        value="",
        help="Se preenchido, substitui a tabela condicionada."
    )
    if orcamento_manual == "":
        orcamento_manual = None

    # -----------------------------
    # Avaliação observacional
    # -----------------------------
    resultados = avaliar_listas_universal(
        listas=listas,
        alvo_real=alvo_real,
        n_real=n_real,
        orcamento_manual=orcamento_manual,
    )

    if not resultados:
        st.info("Nenhuma lista válida para avaliação (listas < n_real são ignoradas).")
        st.stop()

    df_out = pd.DataFrame(resultados)

    st.subheader("📊 Resultados (acertos / n_real)")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    st.caption(
        "Leitura sempre relativa ao n_real.\n"
        "Listas com tamanho menor que n_real são descartadas automaticamente."
    )



# ============================================================
# PAINEL V16 PREMIUM — BACKTEST RÁPIDO DO PACOTE (N = 60)
# ============================================================
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":

    st.subheader("📊 V16 Premium — Backtest Rápido do Pacote (N = 60)")
    st.caption(
        "Ensaio estatístico do pacote ATUAL de listas sobre os últimos 60 alvos. "
        "Não é previsão. Não decide volume. Mede apenas resistência sob pressão."
    )

    # ------------------------------------------------------------
    # Recuperação segura do histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")

    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    if historico_df.shape[0] < 60:
        st.warning("Histórico insuficiente para backtest (mínimo: 60 séries).")
        st.stop()

    # ------------------------------------------------------------
    # Recuperação do pacote congelado
    # ------------------------------------------------------------
    pacote = st.session_state.get("pacote_listas_atual")

    if not pacote:
        st.warning("Nenhum pacote de listas foi registrado ainda.")
        st.stop()

    # ------------------------------------------------------------
    # Identificação das colunas de passageiros
    # ------------------------------------------------------------
    colunas_passageiros = [c for c in historico_df.columns if c.lower().startswith("p")]

    if not colunas_passageiros:
        st.error("Não foi possível identificar colunas de passageiros no histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Preparação do histórico (últimos 60 alvos)
    # ------------------------------------------------------------
    ultimos_60 = historico_df.tail(60)

    resultados = {
        ">=3": 0,
        ">=4": 0,
        ">=5": 0,
        ">=6": 0,
    }

    total_testes = 0

    # ------------------------------------------------------------
    # Execução do backtest
    # ------------------------------------------------------------
    for _, linha in ultimos_60.iterrows():

        # Alvo reconstruído a partir das colunas reais
        alvo = set(int(linha[c]) for c in colunas_passageiros if pd.notna(linha[c]))

        for lista in pacote:
            acertos = len(set(lista) & alvo)
            total_testes += 1

            if acertos >= 3:
                resultados[">=3"] += 1
            if acertos >= 4:
                resultados[">=4"] += 1
            if acertos >= 5:
                resultados[">=5"] += 1
            if acertos >= 6:
                resultados[">=6"] += 1

    # ------------------------------------------------------------
    # Cálculo das porcentagens
    # ------------------------------------------------------------
    perc = {
        k: (v / total_testes) * 100 if total_testes > 0 else 0.0
        for k, v in resultados.items()
    }

    # ------------------------------------------------------------
    # Exibição
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("≥ 3 acertos", f"{perc['>=3']:.2f}%")
    col2.metric("≥ 4 acertos", f"{perc['>=4']:.2f}%")
    col3.metric("≥ 5 acertos", f"{perc['>=5']:.2f}%")
    col4.metric("≥ 6 acertos", f"{resultados['>=6']} ocorrências")

    st.info(
        "📌 Interpretação correta:\n"
        "- Percentuais baixos indicam palco escorregadio\n"
        "- Percentuais estáveis indicam pacote resiliente\n"
        "- Isso NÃO prevê o próximo alvo\n"
        "- Serve apenas para calibrar postura e volume"
    )


# ============================================================

# ============================================================
# P1 — BACKTEST COMPARATIVO BLOCO C (A/B) — N = 60 (primeiro)
# (pré-C4 | auditável | sem motor novo)
#
# OBJETIVO:
# - Comparar o pacote A (pré-BLOCO C) vs pacote B (pós-BLOCO C)
#   sobre os últimos N alvos do histórico.
#
# IMPORTANTE:
# - Este A/B NÃO regenera listas por alvo. Ele compara DOIS pacotes
#   obtidos na mesma sessão (antes/depois do BLOCO C).
# - Serve como evidência inicial de efetividade do operador V10.
# - Não decide ataque, não altera Camada 4.
# ============================================================
if painel == "📊 P1 — Backtest Comparativo BLOCO C (A/B) — N=60":

    st.subheader("📊 P1 — Backtest Comparativo BLOCO C (A/B)")
    st.caption(
        "Comparativo inicial (A/B) do BLOCO C sobre os últimos N alvos do histórico. "
        "A = pacote pré-BLOCO C (capturado). B = pacote pós-BLOCO C (pacote atual). "
        "Não é previsão. Não decide volume. Não altera Camada 4."
    )

    # ------------------------------------------------------------
    # N (primeiro 60, depois 120)
    # ------------------------------------------------------------
    N = st.selectbox(
        "Janela de backtest (N)",
        options=[60, 120],
        index=0,
        help="Primeiro faça N=60 (rápido). Depois repita com N=120 (mais robusto).",
        key="P1_AB_N",
    )

    # ------------------------------------------------------------
    # Histórico
    # ------------------------------------------------------------
    historico_df = st.session_state.get("historico_df")

    if historico_df is None or historico_df.empty:
        st.warning("Histórico não encontrado. Carregue o histórico antes.")
        st.stop()

    if historico_df.shape[0] < int(N):
        st.warning(f"Histórico insuficiente para backtest (mínimo: {int(N)} séries).")
        st.stop()

    # ------------------------------------------------------------
    # Pacotes A e B
    # ------------------------------------------------------------
    pacote_A = st.session_state.get("pacote_pre_bloco_c")
    pacote_B = st.session_state.get("pacote_listas_atual")

    if not pacote_A:
        st.warning(
            "Pacote A (pré-BLOCO C) não encontrado nesta sessão.\n\n"
            "Como gerar: execute o Modo 6 nesta sessão (o CAP Invisível captura o pacote pré-BLOCO C automaticamente)."
        )
        st.stop()

    if not pacote_B:
        st.warning("Pacote B (pós-BLOCO C) não encontrado. Gere listas (Modo 6) antes.")
        st.stop()

    # Força listas em int
    def _norm_pacote(p):
        out = []
        if not isinstance(p, list):
            return out
        for lst in p:
            if not isinstance(lst, (list, tuple)):
                continue
            tmp = []
            for v in lst:
                try:
                    tmp.append(int(v))
                except Exception:
                    pass
            if tmp:
                out.append(tmp)
        return out

    pacote_A = _norm_pacote(pacote_A)
    pacote_B = _norm_pacote(pacote_B)

    if not pacote_A or not pacote_B:
        st.warning("Pacotes inválidos para backtest (listas vazias ou não numéricas).")
        st.stop()

    # ------------------------------------------------------------
    # Colunas de passageiros
    # ------------------------------------------------------------
    colunas_passageiros = [c for c in historico_df.columns if c.lower().startswith("p")]
    if not colunas_passageiros:
        st.error("Não foi possível identificar colunas de passageiros no histórico.")
        st.stop()

    # ------------------------------------------------------------
    # Funções de métrica (por alvo)
    # ------------------------------------------------------------
    def _avaliar_pacote_em_alvo(pacote, alvo_set):
        hits = []
        for lst in pacote:
            hits.append(len(set(lst) & alvo_set))
        if not hits:
            return {"hit_max": 0, "hit_mean": 0.0}
        return {"hit_max": int(max(hits)), "hit_mean": float(sum(hits) / len(hits))}

    def _agregar(df_metrics):
        # df_metrics contém hit_max e hit_mean por alvo
        out = {}
        out["hit_max_medio"] = float(df_metrics["hit_max"].mean())
        out["hit_mean_medio"] = float(df_metrics["hit_mean"].mean())

        # taxas
        for k in [3, 4, 5, 6]:
            out[f"taxa_{k}plus"] = float((df_metrics["hit_max"] >= k).mean()) * 100.0

        # contagens
        for k in [3, 4, 5, 6]:
            out[f"cnt_{k}plus"] = int((df_metrics["hit_max"] >= k).sum())
        return out

    # ------------------------------------------------------------
    # Execução do A/B
    # ------------------------------------------------------------
    ultimos = historico_df.tail(int(N))

    rows_A = []
    rows_B = []

    for _, linha in ultimos.iterrows():
        alvo = set()
        for c in colunas_passageiros:
            if pd.notna(linha[c]):
                try:
                    alvo.add(int(linha[c]))
                except Exception:
                    pass

        if not alvo:
            continue

        a = _avaliar_pacote_em_alvo(pacote_A, alvo)
        b = _avaliar_pacote_em_alvo(pacote_B, alvo)

        rows_A.append(a)
        rows_B.append(b)

    if not rows_A or not rows_B:
        st.warning("Não foi possível montar alvos válidos na janela escolhida.")
        st.stop()

    dfA = pd.DataFrame(rows_A)
    dfB = pd.DataFrame(rows_B)

    aggA = _agregar(dfA)
    aggB = _agregar(dfB)

    # ------------------------------------------------------------
    # Exibição resumida
    # ------------------------------------------------------------
    st.markdown("### ✅ Resumo A/B (agregado)")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("A: hit_max médio", f"{aggA['hit_max_medio']:.3f}")
    c2.metric("B: hit_max médio", f"{aggB['hit_max_medio']:.3f}")
    c3.metric("Δ (B−A)", f"{(aggB['hit_max_medio'] - aggA['hit_max_medio']):.3f}")
    c4.metric("N alvos válidos", f"{len(dfA)}")

    st.markdown("### 🎯 Taxas de ≥k (usando hit_max por alvo)")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("≥3 (A)", f"{aggA['taxa_3plus']:.2f}%")
    t2.metric("≥3 (B)", f"{aggB['taxa_3plus']:.2f}%")
    t3.metric("≥4 (A)", f"{aggA['taxa_4plus']:.2f}%")
    t4.metric("≥4 (B)", f"{aggB['taxa_4plus']:.2f}%")

    t5, t6, t7, t8 = st.columns(4)
    t5.metric("≥5 (A)", f"{aggA['taxa_5plus']:.2f}%")
    t6.metric("≥5 (B)", f"{aggB['taxa_5plus']:.2f}%")
    t7.metric("≥6 (A)", f"{aggA['taxa_6plus']:.2f}%")
    t8.metric("≥6 (B)", f"{aggB['taxa_6plus']:.2f}%")

    st.info(
        "📌 Interpretação canônica (P1):\n"
        "- Se B aumenta ≥4 e ≥5 sem derrubar muito ≥3, há sinal inicial de efetividade do BLOCO C.\n"
        "- Se B só muda hit_mean, mas não move ≥4/≥5, pode ser só estética/saúde estrutural.\n"
        "- Este painel não decide nada; ele mede."
    )

    # ------------------------------------------------------------
    # Auditoria: guarda no session_state
    # ------------------------------------------------------------
    try:
        st.session_state["p1_ab_config"] = {"N": int(N), "col_pass": list(colunas_passageiros)}
        st.session_state["p1_ab_resumo"] = {"A": aggA, "B": aggB, "delta": {k: (aggB.get(k, 0) - aggA.get(k, 0)) for k in aggA.keys()}}
        st.session_state["p1_ab_series"] = {
            "A": {"hit_max": dfA["hit_max"].tolist(), "hit_mean": dfA["hit_mean"].tolist()},
            "B": {"hit_max": dfB["hit_max"].tolist(), "hit_mean": dfB["hit_mean"].tolist()},
        }
    except Exception:
        pass

    # ------------------------------------------------------------
    # Tabela detalhada (opcional)
    # ------------------------------------------------------------
    with st.expander("🔍 Detalhe por alvo (hit_max / hit_mean) — A vs B"):
        df_show = pd.DataFrame({
            "A_hit_max": dfA["hit_max"].astype(int),
            "B_hit_max": dfB["hit_max"].astype(int),
            "Δ_hit_max": (dfB["hit_max"] - dfA["hit_max"]).astype(int),
            "A_hit_mean": dfA["hit_mean"],
            "B_hit_mean": dfB["hit_mean"],
            "Δ_hit_mean": (dfB["hit_mean"] - dfA["hit_mean"]),
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)



# PAINEL V16 PREMIUM — RODADAS ESTRATIFICADAS (A/B)
# (Preparação operacional: NÃO ativa motores; NÃO mistura pacotes)
#
# OBJETIVO:
# - Permitir que o operador registre DUAS execuções independentes
#   para o MESMO evento considerado "Bom + Oportunidade Rara".
# - Rodada A: modelo-base (n=6) — normalmente via Modo 6.
# - Rodada B: modelo alternativo (n≠6) — por colagem manual (por enquanto)
#
# REGRAS:
# - Nunca misturar listas/volumes/decisões.
# - Nunca somar resultados.
# - Registrar e analisar separadamente.
#
# IMPORTANTE:
# - Este painel é 100% opcional.
# - Se nada for marcado, o app se comporta como sempre.
# ============================================================
if painel == "🧭 V16 Premium — Rodadas Estratificadas (A/B)":

    st.subheader("🧭 V16 Premium — Rodadas Estratificadas (A/B)")
    st.caption(
        "Painel de preparação e registro. Não gera listas automaticamente. "
        "Não ativa camadas experimentais. Não muda Modo 6/TURBO. "
        "Serve apenas para organizar duas rodadas independentes no MESMO evento raro."
    )

    # ------------------------------
    # Leitura do momento (somente leitura)
    # ------------------------------
    dmo_estado = st.session_state.get("dmo_estado", "🟥 SOBREVIVÊNCIA")
    eco_status = st.session_state.get("eco_status", st.session_state.get("eco_acionabilidade", "N/D"))
    nr_ruido = st.session_state.get("nr_percent", st.session_state.get("nr_percent_v16"))

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("DMO", str(dmo_estado))
    colm2.metric("ECO", str(eco_status) if eco_status else "N/D")
    colm3.metric("NR%", f"{nr_ruido:.2f}%" if isinstance(nr_ruido, (int, float)) else "N/D")

    st.markdown("---")

    # ------------------------------
    # Travas (governança)
    # ------------------------------
    st.markdown("### 🔒 Travas (governança)")
    st.caption(
        "Este painel não decide por você. Ele só permite registro A/B quando você confirma conscientemente."
    )

    confirmar_momento_bom = st.checkbox(
        "Confirmo que este evento é: 🟢 Momento Bom + Oportunidade Rara (decisão do operador)",
        value=False,
        key="AB_CONFIRMAR_MOMENTO_BOM_RARO",
    )

    habilitar_rodada_b = st.checkbox(
        "(Opcional) Quero preparar Rodada B (Proteção de Modelo) — execução separada", 
        value=False,
        key="AB_HABILITAR_RODADA_B",
        disabled=(not confirmar_momento_bom),
    )

    # Guarda leve (não bloqueia o operador, só avisa)
    avisos = []
    if isinstance(nr_ruido, (int, float)) and nr_ruido >= 70:
        avisos.append("NR% alto (>=70): cuidado com leitura de momento.")
    if isinstance(dmo_estado, str) and dmo_estado.strip().startswith("🟥"):
        avisos.append("DMO em SOBREVIVÊNCIA: este cenário normalmente não é 'momento bom'.")
    if isinstance(eco_status, str) and eco_status.strip() in ("N/D", "DESCONHECIDO"):
        avisos.append("ECO indefinido: leitura parcial do momento.")

    if avisos:
        st.warning("⚠️ Avisos de governança:\n- " + "\n- ".join(avisos))

    st.markdown("---")

    # ------------------------------
    # Identificação do evento (rótulo humano)
    # ------------------------------
    st.markdown("### 🏷️ Identificação do evento")
    st.caption("Use um rótulo simples (ex.: C5823 / 'Evento Raro Jan-2026').")
    evento_id = st.text_input(
        "Evento (ID/rótulo):",
        value=st.session_state.get("AB_EVENTO_ID", ""),
        key="AB_EVENTO_ID_INPUT",
        disabled=(not confirmar_momento_bom),
    )

    if confirmar_momento_bom and evento_id:
        st.session_state["AB_EVENTO_ID"] = evento_id.strip()

    st.markdown("---")

    # ------------------------------
    # Rodada A — n=6 (captura do pacote atual)
    # ------------------------------
    st.markdown("## 🔵 Rodada A — Estratégia Principal (modelo-base)")
    st.caption(
        "Normalmente você gera o pacote no 🎯 Modo 6. Aqui você apenas registra uma fotografia desse pacote como Rodada A."
    )

    pacote_atual = st.session_state.get("pacote_listas_atual")
    pacote_origem = st.session_state.get("pacote_listas_origem", "N/D")

    if pacote_atual and isinstance(pacote_atual, list):
        st.success(f"Pacote atual detectado: {len(pacote_atual)} lista(s) — origem: {pacote_origem}")
        st.dataframe(pd.DataFrame({"Lista": [str(L) for L in pacote_atual]}), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum pacote atual detectado. Gere listas no 🎯 Modo 6 para ter algo a registrar aqui.")

    registrar_a = st.button(
        "📦 Registrar Rodada A (capturar pacote atual)",
        disabled=(not (confirmar_momento_bom and evento_id and pacote_atual)),
        key="AB_REGISTRAR_A",
    )

    if registrar_a:
        st.session_state["AB_RODADA_A"] = {
            "evento": evento_id.strip(),
            "origem": pacote_origem,
            "listas": pacote_atual,
            "meta": {
                "dmo": dmo_estado,
                "eco": eco_status,
                "nr": nr_ruido,
            },
        }
        st.success("✅ Rodada A registrada como pacote independente.")

    st.markdown("---")

    # ------------------------------
    # Rodada B — n≠6 (proteção) — por colagem manual
    # ------------------------------
    st.markdown("## 🔴 Rodada B — Proteção de Modelo (execução paralela)")
    st.caption(
        "Esta rodada não complementa a A: ela coexiste como hipótese paralela. "
        "Por enquanto, o registro B é por colagem manual (listas 7/8/9/10)."
    )

    def _ab_parse_listas_texto(txt: str):
        """Parse simples: aceita linhas com números separados por vírgula, espaço ou ';'."""
        import re
        listas = []
        if not txt:
            return listas
        for linha in txt.splitlines():
            linha = linha.strip()
            if not linha:
                continue
            nums = re.findall(r"\d+", linha)
            if not nums:
                continue
            listas.append([int(x) for x in nums])
        return listas

    n_b = st.number_input(
        "n da Rodada B (tamanho esperado da lista):",
        min_value=7,
        max_value=12,
        value=int(st.session_state.get("AB_N_B", 7)),
        step=1,
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b)),
        key="AB_N_B_INPUT",
    )
    if confirmar_momento_bom and habilitar_rodada_b:
        st.session_state["AB_N_B"] = int(n_b)

    texto_b = st.text_area(
        "Cole aqui as listas da Rodada B (uma por linha):",
        value=st.session_state.get("AB_LISTAS_B_TEXTO", ""),
        height=180,
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b and evento_id)),
        key="AB_LISTAS_B_TEXTO",
    )

    listas_b = _ab_parse_listas_texto(texto_b)
    listas_b_validas = [L for L in listas_b if isinstance(L, list) and len(L) == int(n_b)]
    invalidas = len(listas_b) - len(listas_b_validas)

    if confirmar_momento_bom and habilitar_rodada_b:
        st.info(
            f"Leitura da Rodada B: {len(listas_b_validas)} lista(s) válidas (n={int(n_b)})"
            + (f" · {invalidas} inválida(s) (tamanho diferente)." if invalidas else "")
        )

    registrar_b = st.button(
        "📦 Registrar Rodada B (proteção) — pacote independente",
        disabled=(not (confirmar_momento_bom and habilitar_rodada_b and evento_id and len(listas_b_validas) > 0)),
        key="AB_REGISTRAR_B",
    )

    if registrar_b:
        st.session_state["AB_RODADA_B"] = {
            "evento": evento_id.strip(),
            "n": int(n_b),
            "origem": "Colagem manual (Rodada B)",
            "listas": listas_b_validas,
            "meta": {
                "dmo": dmo_estado,
                "eco": eco_status,
                "nr": nr_ruido,
            },
        }
        st.success("✅ Rodada B registrada como pacote independente.")

    st.markdown("---")

    # ------------------------------
    # Resumo final (A/B) — sem soma, sem mistura
    # ------------------------------
    st.markdown("### 📌 Resumo A/B (sem mistura)")
    a = st.session_state.get("AB_RODADA_A")
    b = st.session_state.get("AB_RODADA_B")

    colr1, colr2 = st.columns(2)
    with colr1:
        st.markdown("**Rodada A**")
        if a:
            st.success(f"Evento: {a.get('evento')} · listas: {len(a.get('listas') or [])} · origem: {a.get('origem')}")
        else:
            st.info("Ainda não registrada.")
    with colr2:
        st.markdown("**Rodada B**")
        if b:
            st.success(f"Evento: {b.get('evento')} · n={b.get('n')} · listas: {len(b.get('listas') or [])}")
        else:
            st.info("Ainda não registrada (opcional).")

    st.caption(
        "Regra canônica: A e B são pacotes distintos. Não somar volumes. "
        "Não interpretar como um único ataque. Replay/Backtest devem ser feitos separadamente."
    )


# ============================================================
# PAINEL — 🧠 M5 — PULO DO GATO (COLETA AUTOMÁTICA DE ESTADOS)
# ============================================================
if painel == M5_PAINEL_PULO_GATO_NOME:
    m5_painel_pulo_do_gato_operacional()


# ============================================================
# PAINEL — 📈 EXPECTATIVA HISTÓRICA — CONTEXTO DO MOMENTO (V16)
# ============================================================
if painel == M3_PAINEL_EXPECTATIVA_NOME:
    v16_painel_expectativa_historica_contexto()


# ============================================================
# PAINEL — 🧭 CHECKLIST OPERACIONAL — DECISÃO (AGORA)
# ============================================================
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":

    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 1) Estrada
    # --------------------------------------------------------
    st.markdown("### 1️⃣ Estrada permite ataque?")
    st.markdown(
        "- k* **não piorou**\n"
        "- NR% **não explodiu**\n"
        "- Divergência **não disparou**"
    )
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_ESTRADA_OK",
    )

    # --------------------------------------------------------
    # 2) Regime
    # --------------------------------------------------------
    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
        key="CHECKLIST_REGIME",
    )

    # --------------------------------------------------------
    # 3) Eixo
    # --------------------------------------------------------
    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_EIXO",
    )

    # --------------------------------------------------------
    # 4) Nocivos
    # --------------------------------------------------------
    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
        key="CHECKLIST_NOCIVOS",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # 5) Decisão humana
    # --------------------------------------------------------
    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
        key="CHECKLIST_ACAO",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Síntese
    # --------------------------------------------------------
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )


# ============================================================
# PAINEL — 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================
if painel == "🧭 Modo Guiado Oficial — PredictCars":

    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.caption(
        "Guia operacional único · uso diário · contrato de uso do sistema.\n"
        "Não executa, não calcula, não decide — apenas orienta a sequência correta."
    )

    st.markdown("---")

    st.markdown("""
🧭 **MODO GUIADO OFICIAL — CONTRATO OPERACIONAL**

Este painel descreve **COMO o PredictCars deve ser usado**.
Ele existe para evitar decisões fora de ordem e misturas perigosas
entre leitura, decisão, execução e aprendizado.

━━━━━━━━━━━━━━━━━━━━
🔵 **AGORA — DECIDIR E JOGAR**
━━━━━━━━━━━━━━━━━━━━

**1️⃣ ENTRADA**
- 📁 Carregar Histórico (Arquivo ou Colar)

**2️⃣ EIXO 1 — ESTRUTURA DO AMBIENTE**
*(saúde da estrada · não números)*

Painéis:
- 🛰️ Sentinelas — k*
- 🧭 Monitor de Risco — k & k*
- 📡 Painel de Ruído Condicional
- 📉 Painel de Divergência S6 vs MC

Pergunta respondida:
- O ambiente permite ataque?

# ---

**3️⃣ EIXO 2 — MOMENTO & ANTECIPAÇÃO**
*(ritmo do alvo + evidência recente)*

Painéis:
- 🔁 Replay LIGHT
- 🔁 Replay ULTRA
- 🧪 Replay Curto — Expectativa 1–3 Séries
- 📊 V16 Premium — Backtest Rápido do Pacote (N=60)

Pergunta respondida:
- O momento favorece agir agora?

# ---

**4️⃣ DECISÃO ÚNICA (HUMANA)**
*(registrada · sem retorno)*

Painel:
- 🧭 Checklist Operacional — Decisão (AGORA)

Aqui você define:
- atacar ou não
- concentrar, equilibrar ou expandir
- volume de listas

📌 **Depois disso, não se volta atrás.**

# ---

**5️⃣ MOTOR**
- 🛣️ Pipeline V14-FLEX ULTRA
- ⚙️ Modo TURBO++ HÍBRIDO
- ⚙️ Modo TURBO++ ULTRA

# ---

**6️⃣ EXECUÇÃO**
- 🎯 Modo 6 Acertos — Execução
- 🧪 Testes de Confiabilidade REAL
- 📘 Relatório Final
- 🔥 Mandar Bala

━━━━━━━━━━━━━━━━━━━━
🟣 **EXTENSÃO CONDICIONAL — MODO ESPECIAL**
━━━━━━━━━━━━━━━━━━━━

Use **somente após** concluir o fluxo acima.

- 🔵 MODO ESPECIAL — Evento Condicionado
- Atua sobre pacotes já gerados
- Não cria listas novas
- Útil apenas para eventos únicos

━━━━━━━━━━━━━━━━━━━━
🟢 **DEPOIS — APRENDER**
━━━━━━━━━━━━━━━━━━━━

Painéis:
- 📊 V16 Premium — ANTI-EXATO | Passageiros Nocivos
- 📊 EXATO / ECO / Regime

📌 Aprendizado **somente para a próxima rodada**.

━━━━━━━━━━━━━━━━━━━━
🧱 **OS 3 EIXOS DO SISTEMA**
━━━━━━━━━━━━━━━━━━━━

- **Eixo 1** — Estrutura das Listas  
- **Eixo 2** — Momento & Antecipação  
- **Eixo 3** — Aprendizado  

━━━━━━━━━━━━━━━━━━━━
📜 **REGRA FINAL**
━━━━━━━━━━━━━━━━━━━━

A decisão acontece **ANTES**.  
O aprendizado acontece **DEPOIS**.  
**Nunca ao mesmo tempo.**
""")

    st.success(
        "Modo Guiado carregado com sucesso.\n"
        "Este painel é o contrato oficial de uso do PredictCars."
    )

# ============================================================
# <<< FIM — PAINEL 🧭 MODO GUIADO OFICIAL — PREDICTCARS
# ============================================================





# ============================================================
# ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS (DEFINITIVO)
# ============================================================

# ------------------------------------------------------------
# ORIENTAÇÃO / USO
# ------------------------------------------------------------
if painel == "🧭 Modo Guiado Oficial — PredictCars":
    st.markdown("## 🧭 Modo Guiado Oficial — PredictCars")
    st.info(
        "Este painel apenas orienta o uso correto do sistema.\n"
        "Siga a sequência indicada no menu."
    )
    st.stop()

# ------------------------------------------------------------
# GOVERNANÇA / VISIBILIDADE (M1)
# ------------------------------------------------------------
if painel == "🔍 Diagnóstico Espelho (Mirror)":
    _m1_render_mirror_panel()
    st.stop()

# ------------------------------------------------------------
# DECISÃO OPERACIONAL (AGORA)
# ------------------------------------------------------------
if painel == "🧭 Checklist Operacional — Decisão (AGORA)":
    st.markdown("## 🧭 Checklist Operacional — Decisão (AGORA)")
    st.caption(
        "Checklist obrigatório ANTES do Modo 6 / Mandar Bala.\n"
        "Não calcula, não cria listas, não decide automaticamente."
    )

    st.markdown("---")

    st.markdown("### 1️⃣ Estrada permite ataque?")
    estrada_ok = st.radio(
        "Resultado da leitura da estrada:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 2️⃣ Regime jogável?")
    regime = st.radio(
        "Regime identificado:",
        ["OURO", "PRATA", "RUIM"],
        horizontal=True,
    )

    st.markdown("### 3️⃣ Existe eixo claro nas listas?")
    eixo = st.radio(
        "Eixo identificado:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 4️⃣ Nocivos concentrados nas mesmas listas?")
    nocivos = st.radio(
        "Nocivos:",
        ["SIM", "NÃO"],
        horizontal=True,
    )

    st.markdown("### 5️⃣ Decisão final (humana)")
    acao = st.radio(
        "Ação escolhida:",
        [
            "CONCENTRAR (6–8 listas)",
            "EQUILIBRAR (8–10 listas)",
            "EXPANDIR COM CRITÉRIO (10–12 listas)",
            "SEGURAR / NÃO ESCALAR",
        ],
    )

    st.markdown("---")
    st.markdown("### 🧾 Síntese da decisão")
    st.write(
        {
            "Estrada OK": estrada_ok,
            "Regime": regime,
            "Eixo": eixo,
            "Nocivos concentrados": nocivos,
            "Ação escolhida": acao,
        }
    )

    st.success(
        "Checklist concluído. "
        "A decisão da rodada está FECHADA aqui. "
        "Prossiga para o Modo 6 e execução."
    )
    st.stop()

# ------------------------------------------------------------
# EIXO 2 — MOMENTO & ANTECIPAÇÃO
# ------------------------------------------------------------
if painel == "📊 V16 Premium — Backtest Rápido do Pacote (N=60)":
    st.markdown("## 📊 Backtest Rápido do Pacote (N=60)")
    st.caption(
        "Avaliação observacional do pacote atual.\n"
        "Não decide. Apenas informa."
    )
    st.stop()

# ------------------------------------------------------------
# RITMO DO ALVO (SUPORTE)
# ------------------------------------------------------------
if painel == "🧪 Replay Curto — Expectativa 1–3 Séries":
    st.markdown("## 🧪 Replay Curto — Expectativa 1–3 Séries")
    st.stop()

if painel == "⏱️ Duração da Janela — Análise Histórica":
    st.markdown("## ⏱️ Duração da Janela — Análise Histórica")
    st.stop()

# ------------------------------------------------------------
# V16 PREMIUM — APRENDIZADO (DEPOIS)
# ------------------------------------------------------------

if painel == "🧠 Laudo Operacional V16":
    # ROTA DUPLICADA DESATIVADA (há um painel Laudo V16 completo em outro ponto do app)
    # Mantemos apenas UM ponto de execução do Laudo para evitar dupla execução/deriva.
    st.stop()


def v16_painel_compressao_alvo():
    """🎯 Compressão do Alvo — Observacional (V16)

    Nota: este painel é *pré-C4 / observacional*. Ele NÃO altera listas nem mexe na Camada 4.
    Ele só mede se o alvo está "espalhado" (difícil) ou mais "compacto" (melhor).

    Dependência: precisa existir df_eval (gerado ao avaliar pacotes/snapshots via Replay/SAFE).
    """
    st.markdown("## 🎯 Compressão do Alvo — Observacional (V16)")
    st.caption("Métrica observacional do quão 'compacto' o alvo parece estar, olhando os pacotes avaliados (df_eval).")

    df_eval = st.session_state.get("df_eval", None)

    if df_eval is None or len(df_eval) == 0:
        st.warning("Nenhuma avaliação encontrada (df_eval vazio). Rode o Replay/SAFE e clique em **Avaliar pacotes registrados** para gerar a base.")
        return

    # Colunas esperadas: dist_media_fora / dist_max_fora (podem ser NaN em bases pequenas)
    # ⚠️ Compatibilidade (RC): algumas rotas gravam as distâncias com sufixo (_1/_2) por alvo.
    # Neste caso, sintetizamos colunas canônicas sem sufixo para o painel.
    cols = set(df_eval.columns)

    if ("dist_media_fora" not in cols) or ("dist_max_fora" not in cols):
        cand_media = [c for c in ["dist_media_fora_1", "dist_media_fora_2"] if c in cols]
        cand_max   = [c for c in ["dist_max_fora_1", "dist_max_fora_2"] if c in cols]

        if len(cand_media) == 0 and len(cand_max) == 0:
            st.warning("Base df_eval não tem distâncias fora-do-pacote (dist_media_fora / dist_max_fora). Rode a avaliação completa do SAFE e tente novamente.")
            return

        df_eval = df_eval.copy()

        # dist_media_fora: média (ignorando NaN) entre os alvos disponíveis (_1/_2)
        if ("dist_media_fora" not in cols) and (len(cand_media) > 0):
            df_media = df_eval[cand_media].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            df_eval["dist_media_fora"] = df_media.mean(axis=1, skipna=True)

        # dist_max_fora: maior distância (ignorando NaN) entre os alvos disponíveis (_1/_2)
        if ("dist_max_fora" not in cols) and (len(cand_max) > 0):
            df_max = df_eval[cand_max].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            df_eval["dist_max_fora"] = df_max.max(axis=1, skipna=True)

    df_aux = df_eval.copy()

    # Segurança: converter para numérico
    df_aux["dist_media_fora"] = pd.to_numeric(df_aux["dist_media_fora"], errors="coerce")
    df_aux["dist_max_fora"]   = pd.to_numeric(df_aux["dist_max_fora"], errors="coerce")

    # Base útil: linhas que possuem ao menos alguma distância computada
    base = df_aux.dropna(subset=["dist_media_fora", "dist_max_fora"], how="all")
    if len(base) == 0:
        st.warning("Ainda não há distâncias computadas (tudo NaN). Isso costuma acontecer quando a base avaliada é muito curta.")
        return

    disp_media = float(np.nanmean(base["dist_media_fora"].values))
    disp_vol   = float(np.nanstd(base["dist_media_fora"].values))

    # Score simples: quanto menor a dispersão, maior o score (clamp em [0, 1])
    compress_score = 1.0 - 0.5 * (disp_media + disp_vol)
    compress_score = float(max(0.0, min(1.0, compress_score)))

    st.markdown("### 📐 Métrica de Compressão do Alvo")
    st.metric("Score de Compressão", f"{compress_score:.4f}")
    st.metric("Dispersão média", f"{disp_media:.4f}")
    st.metric("Volatilidade da dispersão", f"{disp_vol:.4f}")

    # Leitura observacional (mantém exatamente a ideia do painel original)
    if compress_score >= 0.70:
        leitura = "🟢 Alvo comprimido (bom sinal) — baixa dispersão e baixa variabilidade estrutural."
    elif compress_score >= 0.40:
        leitura = "🟡 Alvo misto — alguma dispersão; não é confirmação de janela, mas pode haver frestas."
    else:
        leitura = "🔴 Alvo disperso — alta variabilidade estrutural. Mesmo que k apareça, não indica alvo na mira."

    st.info(f"Leitura Observacional\n\n{leitura}")

if painel == "📊 V16 Premium — Erro por Regime (Retrospectivo)":
    v16_painel_erro_por_regime_retrospectivo()
    st.stop()

if painel == "📊 V16 Premium — EXATO por Regime (Proxy)":
    v16_painel_exato_por_regime_proxy()
    st.stop()

if painel == "📊 V16 Premium — PRÉ-ECO → ECO (Persistência & Continuidade)":
    v16_painel_pre_eco_persistencia_continuidade()
    st.stop()

if painel == "📊 V16 Premium — Passageiros Recorrentes em ECO (Interseção)":
    v16_painel_passageiros_recorrentes_eco_intersecao()
    st.stop()

if painel == "🎯 Compressão do Alvo — Observacional (V16)":
    v16_painel_compressao_alvo()
    st.stop()

if painel == "🔮 V16 Premium Profundo — Diagnóstico & Calibração":
    st.info('Painel Premium Profundo já foi executado acima.'); st.stop()
    st.stop()

# ============================================================
# FIM DO ROTEADOR V16 PREMIUM — EXECUÇÃO DOS PAINÉIS
# ============================================================







# ===========================
# 🧪 MC Observacional do Pacote (pré-C4) — ROUTER
if painel == "🧪 MC Observacional do Pacote (pré-C4)":
    v16_painel_mc_observacional_pacote_pre_c4()


if painel == "📐 Parabólica — Curvatura do Erro (Governança Pré-C4)":
    v16_painel_parabolica_curvatura_erro_pre_c4()

if painel == "📡 CAP — Calibração Assistida da Parabólica (pré-C4)":
    v16_painel_cap_calibracao_assistida_parabola_pre_c4()

# ============================================================
# POST MODO6 AUDIT (v16h57FJ)
# ============================================================
try:
    import itertools
    listas_ref = None
    if 'listas_top10' in globals():
        listas_ref = listas_top10
    elif 'listas_totais' in globals():
        listas_ref = listas_totais[:10]

    if listas_ref:
        flat = [x for l in listas_ref for x in l]
        passageiros_unicos = len(set(flat))
        inter = []
        for a, b in itertools.combinations(listas_ref, 2):
            inter.append(len(set(a).intersection(set(b))))
        sobreposicao = round(sum(inter) / len(inter), 2) if inter else 0
        pacote_hash = hash(str(listas_ref))

        st.markdown("### 🔎 Auditoria do Pacote (POST MODO6)")
        _trace_exec = st.session_state.get("v16h57FL_exec_trace", [])
        if isinstance(_trace_exec, list) and len(_trace_exec) > 0:
            st.markdown("#### 🧭 TRACE — CAMINHO REAL DE EXECUÇÃO DO MODO 6")
            for _item in _trace_exec:
                st.json(_item)
        else:
            st.warning("Nenhum trace de função do Modo 6 foi capturado nesta execução.")

        _src_trace = st.session_state.get("v16h57FL_source_detector", [])
        st.markdown("#### 🧪 TRACE — DETECTOR DE ORIGEM DAS LISTAS")
        if isinstance(_src_trace, list) and len(_src_trace) > 0:
            for _item in _src_trace:
                st.json(_item)
        else:
            st.warning("Nenhum trace de origem das listas foi capturado nesta execução.")

        st.markdown("#### 🧪 AUDITORIA DO CT (CONVERSION PRESSURE)")
        try:
            _calib_last = st.session_state.get("v16_calib_leve_last_summary") or {}
            _npg = (_calib_last.get("new_packet_generator") or st.session_state.get("v16_ct_last_real_generator") or {}) if isinstance(_calib_last, dict) else (st.session_state.get("v16_ct_last_real_generator") or {})
            _cp = (_npg.get("conversion_pressure") or {}) if isinstance(_npg, dict) else {}
            _snap_map = (
                st.session_state.get("snapshot_p0_canonic")
                or st.session_state.get("snapshot_p0")
                or st.session_state.get("snapshot_p0_replay")
                or {}
            )
            st.json({
                "snapshot_disponivel": bool(isinstance(_snap_map, dict) and len(_snap_map) > 0),
                "qtd_snapshots": int(len(_snap_map) if isinstance(_snap_map, dict) else 0),
                "cp_ativo": bool(isinstance(_cp, dict) and _cp.get("ok")),
                "motivo": (_cp.get("motivo") if isinstance(_cp, dict) else "N/D") or "N/D",
                "ranking_antes_hash": (_cp.get("ranking_antes_hash") if isinstance(_cp, dict) else "N/D") or "N/D",
                "ranking_depois_hash": (_cp.get("ranking_depois_hash") if isinstance(_cp, dict) else "N/D") or "N/D",
                "ranking_mudou": bool((_cp.get("ranking_mudou") if isinstance(_cp, dict) else False)),
                "top10_antes": (_cp.get("top10_antes") if isinstance(_cp, dict) else []) or [],
                "top10_depois": (_cp.get("top10_depois") if isinstance(_cp, dict) else []) or [],
                "dif_posicoes_top10": int((_cp.get("dif_posicoes_top10") if isinstance(_cp, dict) else 0) or 0),
                "listas_antes_gerador_hash": (_npg.get("listas_antes_gerador_hash") if isinstance(_npg, dict) else "N/D") or "N/D",
                "listas_finais_hash": (_npg.get("listas_finais_hash") if isinstance(_npg, dict) else "N/D") or "N/D",
                "mudou_no_pacote_final": bool((_npg.get("mudou_no_pacote_final") if isinstance(_npg, dict) else False)),
                "new_packet_generator_active": bool((_npg.get("active") if isinstance(_npg, dict) else False)),
                "new_packet_generator_applied": bool((_npg.get("applied") if isinstance(_npg, dict) else False)),
                "new_packet_generator_reason": (_npg.get("reason") if isinstance(_npg, dict) else "N/D") or "N/D",
            })
        except Exception as _e_ct_audit:
            st.error(f"CT_AUDIT_ERROR: {_e_ct_audit}")

        st.markdown("#### FINAL (TOP10 APÓS CAMADAS DO MODO 6)")
        st.json({
            "n_listas": len(listas_ref),
            "hash": pacote_hash,
            "passageiros_unicos": passageiros_unicos,
            "sobreposicao_media": sobreposicao,
            "exemplo": listas_ref[:3]
        })
except Exception as e:
    print("POST_MODO6_AUDIT_ERROR:", e)


# ============================================================
# BUILD v16h57FJ — CT REAL GENERATOR (PRE-SANIDADE HOOK) + BANNER OK
# CT REAL GENERATOR HOOK (PRE SANIDADE)
# ============================================================
try:
    import streamlit as st
    st.session_state["CT_REAL_GENERATOR_PRE_SANIDADE"] = {
        "build": "v16h57FJ",
        "hook": "before_sanidade_final_listas",
        "status": "armed"
    }
except Exception:
    pass



# ============================================================
# BUILD v16h57FJ — CT GENERATOR PRE-SANIDADE REAL HOOK + BANNER OK
# CT REAL HOOK INSIDE GENERATOR (PRE SANIDADE)
# ============================================================
try:
    import streamlit as st
    st.session_state["CT_GENERATOR_PRE_SANIDADE_REAL"] = {
        "build": "v16h57FJ",
        "hook_point": "generator_before_sanidade",
        "status": "armed"
    }
except Exception:
    pass

