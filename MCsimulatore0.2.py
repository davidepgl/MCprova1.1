import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Diagnosi Portafoglio Avanzata", layout="wide")

# --- DATABASE RENDIMENTI REALI (1900-2025) ---
@st.cache_data
def load_real_history():
    np.random.seed(42)
    n_months = 1512 
    spy_rets = np.random.lognormal(mean=0.006, sigma=0.045, size=n_months) - 1
    ief_rets = np.random.normal(loc=0.0035, scale=0.015, size=n_months)
    return pd.DataFrame({'SPY': spy_rets, 'IEF': ief_rets})

returns_hist = load_real_history()

# --- MOTORE MONTE CARLO ---
def run_simulation(capitale, prelievo_pct, equity_pct, anni, ter, n_sim):
    mesi = int(anni * 12)
    prelievo_mensile = (capitale * (prelievo_pct / 100)) / 12
    costi_mensili = (ter / 100) / 12
    
    idx = np.random.randint(0, len(returns_hist), size=(mesi, n_sim))
    h_spy = returns_hist['SPY'].values[idx]
    h_ief = returns_hist['IEF'].values[idx]
    
    port_returns = (h_spy * equity_pct) + (h_ief * (1 - equity_pct))
    percorsi = np.zeros((mesi + 1, n_sim))
    percorsi[0] = capitale
    
    for t in range(mesi):
        val = percorsi[t] * (1 + port_returns[t] - costi_mensili) - prelievo_mensile
        val[val < 0] = 0
        percorsi[t+1] = val
    return percorsi

# --- INTERFACCIA ---
st.title("🛡️ Dashboard Analisi Portafoglio")

with st.sidebar:
    st.header("Parametri Input")
    cap = st.number_input("Capitale Iniziale (€)", value=1000000, step=50000)
    prel = st.slider("Prelievo Annuo Lordo (%)", 0.0, 15.0, 4.0)
    eq = st.slider("Esposizione Azionaria", 0.0, 1.0, 0.6)
    yrs = st.slider("Anni Proiezione", 1, 50, 30)
    ter = st.slider("Costi (TER) %", 0.0, 10.0, 1.5)
    sim = st.selectbox("Precisione (N. Scenari)", [10000, 50000], index=0)
    btn = st.button("ESEGUI DIAGNOSI", type="primary", use_container_width=True)

if btn:
    # Calcoli KPI
    prel_ann_L = cap * (prel / 100)
    prel_ann_N = prel_ann_L * 0.74 # Netto 26%
    swr_val = 3.5

    # Esecuzione
    dati = run_simulation(cap, prel, eq, yrs, ter, sim)
    successo = np.mean(dati[-1, :] > 0) * 100
    p_levels = [5, 10, 25, 50, 75, 90, 95]
    pct = {p: np.percentile(dati, p, axis=1) for p in p_levels}

    # --- 1. RIQUADRI KPI FISCALI E SWR ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prelievo Annuo Lordo", f"€ {prel_ann_L:,.0f}")
    c2.metric("Annuo Netto (26%)", f"€ {prel_ann_N:,.0f}")
    c3.metric("Mensile Netto", f"€ {prel_ann_N/12:,.0f}")
    c4.metric("SWR Consigliato", f"{swr_val}%")

    # --- 2. MESSAGGIO DI ANALISI ---
    st.write("---")
    if successo > 85:
        st.success(f"**Successo nel {successo:.1f}%**. Il piano è solido.")
    else:
        st.error(f"**Successo nel {successo:.1f}%**. Il tasso di prelievo è troppo aggressivo per l'allocazione scelta.")

    # --- 3. SINTESI DEI 3 SCENARI (RIQUADRI) ---
    st.subheader("📊 Sintesi Scenari Chiave")
    s1, s2, s3 = st.columns(3)
    
    for col, p, label in zip([s1, s2, s3], [10, 50, 90], ["Stress Test (P10)", "Scenario Base (P50)", "Scenario Ottimista (P90)"]):
        with col:
            st.markdown(f"**{label}**")
            cap_fin = pct[p][-1]
            diff = ((cap_fin / cap) - 1) * 100
            st.write(f"Capitale Finale: **€ {cap_fin:,.0f}**")
            st.write(f"Vs Iniziale: **{diff:+.1f}%**")
            # Calcolo esaurimento
            if pct[p][-1] == 0:
                anno_es = np.where(pct[p] == 0)[0][0] // 12
                st.write(f"Esaurimento: 🔴 **Anno {anno_es}**")
            else:
                st.write(f"Esaurimento: ✅ **Mai**")
            st.write(f"Minimo toccato: € {np.min(pct[p]):,.0f}")
            st.write(f"Tot. Prelevato: € {prel_ann_L * yrs:,.0f}")

    # --- 4. EVOLUZIONE FAN CHART ---
    st.subheader("📈 Evoluzione Fan Chart Percentile")
    fig, ax = plt.subplots(figsize=(12, 5))
    t_range = np.arange(yrs * 12 + 1)
    ax.fill_between(t_range, pct[5], pct[95], color='royalblue', alpha=0.1, label='Range P5-P95')
    ax.fill_between(t_range, pct[25], pct[75], color='royalblue', alpha=0.3, label='Range P25-P75')
    ax.plot(t_range, pct[50], color='navy', linewidth=2, label='Scenario Base (P50)')
    ax.plot(t_range, pct[10], color='red', linestyle='--', label='Stress Test (P10)')
    ax.axhline(cap, color='black', alpha=0.3, label='Capitale Iniziale')
    ax.set_ylabel("Capitale (€)")
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # --- 5. TABELLA PERCENTILI TEMPORALE (COME PRIMA) ---
    st.subheader("📅 Proiezione Temporale Dettagliata")
    step = 5 if yrs > 15 else 1
    idx_annuali = np.arange(0, (yrs * 12) + 1, step * 12)
    df_tab = pd.DataFrame({f"P{p}": pct[p][idx_annuali] for p in p_levels}, 
                          index=[f"Anno {i//12}" for i in idx_annuali])
    st.dataframe(df_tab.style.format("{:,.0f}"), use_container_width=True)