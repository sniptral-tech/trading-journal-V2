"""
====================================================================
 CARNET DE TRADING INTERACTIF - Streamlit
====================================================================
Application de suivi de sessions de trading (money management
évolutif avec pourcentage de mise dynamique dans chaque formulaire),
avec sauvegarde persistante sur une base Supabase (Postgres gratuit).

Configuration requise avant lancement :
  - Un projet Supabase avec la table créée via supabase_setup.sql
  - Un fichier .streamlit/secrets.toml (en local) ou les "Secrets"
    de l'appli (sur Streamlit Cloud) contenant :
        [supabase]
        url = "https://xxxxx.supabase.co"
        key = "eyJ...."   # clé "anon public"

  ⚠️ SÉCURITÉ : vérifie les policies RLS (Row Level Security) de ta
  table "sessions" sur Supabase. Avec la clé "anon public", si les
  policies sont trop permissives, n'importe qui connaissant l'URL de
  l'appli pourrait insérer ou supprimer des données (le bouton
  "Réinitialiser la base" fait un DELETE complet). Restreins l'accès
  en écriture si l'appli est déployée publiquement.

Lancement :
    pip install -r requirements.txt
    streamlit run app.py
====================================================================
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client

# ====================================================================
# 1. CONFIGURATION GÉNÉRALE & CONSTANTES
# ====================================================================

st.set_page_config(
    page_title="Carnet de Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Chargement du fichier style.css
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Colonnes de l'historique
COLUMNS = [
    "ID",
    "Date",
    "Mode",
    "Capital_Initial",
    "Capital_Final",
    "Nb_Trades",
    "Trades_Gagnes",
    "Trades_Perdus",
    "Payout_Pct",
    "Win_Rate_Pct",
    "Profit_Net",
    "Rendement_Pct",
]


# ====================================================================
# 2. PERSISTANCE CLOUD (Supabase)
# ====================================================================

SUPABASE_TABLE = "sessions"

COL_TO_DB = {
    "Date": "session_date",
    "Mode": "mode",
    "Capital_Initial": "capital_initial",
    "Capital_Final": "capital_final",
    "Nb_Trades": "nb_trades",
    "Trades_Gagnes": "trades_gagnes",
    "Trades_Perdus": "trades_perdus",
    "Payout_Pct": "payout_pct",
    "Win_Rate_Pct": "win_rate_pct",
    "Profit_Net": "profit_net",
    "Rendement_Pct": "rendement_pct",
}
DB_TO_COL = {v: k for k, v in COL_TO_DB.items()}
DB_TO_COL["id"] = "ID"


@st.cache_resource
def get_supabase_client() -> "Client":
    """Crée le client Supabase une seule fois par session."""
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def load_history() -> pd.DataFrame:
    """Charge l'historique complet depuis Supabase."""
    try:
        client = get_supabase_client()
        response = client.table(SUPABASE_TABLE).select("*").order("id").execute()
        rows = response.data
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(rows).rename(columns=DB_TO_COL)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[COLUMNS]
    except Exception as e:
        st.error(f"⚠️ Impossible de charger l'historique depuis Supabase : {e}")
        return pd.DataFrame(columns=COLUMNS)


def append_session(row: dict) -> bool:
    """Ajoute une session dans Supabase. Retourne True si l'opération a réussi."""
    try:
        client = get_supabase_client()
        db_row = {COL_TO_DB[k]: v for k, v in row.items() if k in COL_TO_DB}
        client.table(SUPABASE_TABLE).insert(db_row).execute()
        st.session_state.history = load_history()
        return True
    except Exception as e:
        st.error(f"⚠️ Échec de l'enregistrement sur Supabase : {e}")
        return False


def reset_history() -> bool:
    """Supprime définitivement toutes les sessions enregistrées. Retourne True si succès."""
    try:
        client = get_supabase_client()
        client.table(SUPABASE_TABLE).delete().gte("id", 0).execute()
        st.session_state.history = pd.DataFrame(columns=COLUMNS)
        return True
    except Exception as e:
        st.error(f"⚠️ Échec de la réinitialisation sur Supabase : {e}")
        return False


# ====================================================================
# 3. LOGIQUE DE CALCUL (Money Management Dynamique)
# ====================================================================

def compute_capital_final(c_start: float, wins: int, losses: int, payout_pct: float, mise_pct: float = 0.04) -> float:
    """Capital final après W gains et L pertes, selon le % de mise saisi."""
    payout = payout_pct / 100.0
    facteur_gain = 1 + mise_pct * payout
    facteur_perte = 1 - mise_pct
    return c_start * (facteur_gain ** wins) * (facteur_perte ** losses)


def solve_wins_from_capital(c_start: float, c_end: float, n_trades: int, payout_pct: float, mise_pct: float = 0.04):
    """Résout W par passage au log selon le % de mise saisi."""
    if c_start <= 0 or c_end <= 0 or n_trades <= 0:
        raise ValueError("Capital initial, capital final et nombre de trades doivent être > 0.")

    payout = payout_pct / 100.0
    facteur_gain = 1 + mise_pct * payout
    facteur_perte = 1 - mise_pct

    if facteur_gain == facteur_perte:
        raise ValueError("Payout invalide : impossible de distinguer gains et pertes.")

    numerateur = np.log(c_end / c_start) - n_trades * np.log(facteur_perte)
    denominateur = np.log(facteur_gain / facteur_perte)
    w_brut = numerateur / denominateur

    w_arrondi = int(round(w_brut))
    w_arrondi = max(0, min(n_trades, w_arrondi))

    return w_arrondi, w_brut


def simulate_session_step_by_step(c_start: float, resultats: list, payout_pct: float, mise_pct: float = 0.04) -> pd.DataFrame:
    """Simule une session trade par trade selon le % de mise saisi."""
    payout = payout_pct / 100.0
    capital = c_start
    lignes = []
    for i, gagne in enumerate(resultats, start=1):
        mise = mise_pct * capital
        if gagne:
            gain = mise * payout
            capital_apres = capital + gain
            resultat_txt = "✅ Gagné"
            variation = gain
        else:
            capital_apres = capital - mise
            resultat_txt = "❌ Perdu"
            variation = -mise
        lignes.append({
            "Trade #": i,
            "Capital avant ($)": round(capital, 2),
            f"Mise ({round(mise_pct*100, 1)}%) ($)": round(mise, 2),
            "Résultat": resultat_txt,
            "Variation ($)": round(variation, 2),
            "Capital après ($)": round(capital_apres, 2),
        })
        capital = capital_apres
    return pd.DataFrame(lignes)


def build_session_row(date_val, mode: str, c_start: float, c_end: float,
                       n_trades: int, wins: int, losses: int, payout_pct: float) -> dict:
    """Construit le dictionnaire d'une ligne d'historique."""
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0
    profit_net = c_end - c_start
    rendement = (profit_net / c_start * 100) if c_start > 0 else 0.0
    return {
        "Date": str(date_val),
        "Mode": mode,
        "Capital_Initial": round(c_start, 2),
        "Capital_Final": round(c_end, 2),
        "Nb_Trades": int(n_trades),
        "Trades_Gagnes": int(wins),
        "Trades_Perdus": int(losses),
        "Payout_Pct": round(payout_pct, 2),
        "Win_Rate_Pct": round(win_rate, 2),
        "Profit_Net": round(profit_net, 2),
        "Rendement_Pct": round(rendement, 2),
    }


# ====================================================================
# 4. INITIALISATION DU SESSION STATE
# ====================================================================

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if "prefill" not in st.session_state:
    st.session_state.prefill = None

# Compteur utilisé pour forcer Streamlit à régénérer les widgets du
# formulaire Mode B avec de nouvelles valeurs par défaut quand un
# transfert depuis le Simulateur a lieu (sans ça, Streamlit garde la
# valeur déjà affichée à l'écran et ignore le nouveau "value=").
if "prefill_version" not in st.session_state:
    st.session_state.prefill_version = 0


# ====================================================================
# 5. SIDEBAR
# ====================================================================

with st.sidebar:
    st.title("📈 Carnet de Trading")
    st.caption("Money management évolutif — Cloud Supabase")

    df_hist = st.session_state.history
    if not df_hist.empty:
        capital_actuel = df_hist.iloc[-1]["Capital_Final"]
        capital_depart = df_hist.iloc[0]["Capital_Initial"]
        rendement_global = (capital_actuel - capital_depart) / capital_depart * 100 if capital_depart else 0
        st.metric("Capital actuel", f"{capital_actuel:,.2f} $")
        st.metric("Rendement global", f"{rendement_global:+.2f} %")
        st.caption(f"{len(df_hist)} session(s) enregistrée(s)")
    else:
        st.info("Aucune session enregistrée pour le moment.")

    st.divider()
    st.caption("☁️ Données stockées sur Supabase (cloud)")
    st.caption("Persistantes même après une mise en veille de l'appli.")


# ====================================================================
# 6. ONGLETS PRINCIPAUX
# ====================================================================

tab_dashboard, tab_new, tab_sim, tab_data = st.tabs(
    ["🏠 Dashboard", "➕ Nouvelle Session", "🧪 Simulateur", "🛠️ Gestion des données"]
)

# --------------------------------------------------------------------
# ONGLET 1 : DASHBOARD
# --------------------------------------------------------------------
with tab_dashboard:
    df = st.session_state.history

    if df.empty:
        st.info("👋 Aucune session enregistrée. Rends-toi dans l'onglet **➕ Nouvelle Session** pour commencer.")
    else:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        capital_actuel = df.iloc[-1]["Capital_Final"]
        capital_depart = df.iloc[0]["Capital_Initial"]
        rendement_total = (capital_actuel - capital_depart) / capital_depart * 100 if capital_depart else 0
        win_rate_global = df["Trades_Gagnes"].sum() / df["Nb_Trades"].sum() * 100 if df["Nb_Trades"].sum() else 0
        meilleure_session = df["Profit_Net"].max()
        pire_session = df["Profit_Net"].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Capital Actuel Total", f"{capital_actuel:,.2f} $")
        c2.metric("📊 Rendement Total", f"{rendement_total:+.2f} %")
        c3.metric("🎯 Win Rate Global", f"{win_rate_global:.2f} %")
        c4.metric("🏆 Meilleure / 📉 Pire session", f"{meilleure_session:+.2f} $ / {pire_session:+.2f} $")


        st.divider()

        g1, g2 = st.columns(2)

        with g1:
            st.subheader("Profit / Perte net par session")
            df_bar = df.copy()
            df_bar["Session"] = [f"#{i+1}" for i in range(len(df_bar))]
            df_bar["Couleur"] = np.where(df_bar["Profit_Net"] >= 0, "Gain", "Perte")
            fig_bar = px.bar(
                df_bar, x="Session", y="Profit_Net", color="Couleur",
                color_discrete_map={"Gain": "#22c55e", "Perte": "#ef4444"},
                labels={"Profit_Net": "Profit net ($)"},
                text_auto=".2f",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                showlegend=False, margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
            )
            fig_bar.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
            fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with g2:
            st.subheader("Répartition Trades Gagnés / Perdus")
            total_gagnes = int(df["Trades_Gagnes"].sum())
            total_perdus = int(df["Trades_Perdus"].sum())
            fig_pie = px.pie(
                names=["Gagnés", "Perdus"],
                values=[total_gagnes, total_perdus],
                color=["Gagnés", "Perdus"],
                color_discrete_map={"Gagnés": "#22c55e", "Perdus": "#ef4444"},
                hole=0.45,
                template="plotly_dark",
            )
            fig_pie.update_layout(
                margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                legend=dict(font=dict(color="#e5e7eb")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Progression du capital au fil des sessions")
        df_line = df.copy()
        df_line["Session"] = [f"#{i+1}" for i in range(len(df_line))]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_line["Session"], y=df_line["Capital_Final"],
            mode="lines+markers", name="Capital final",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=7, color="#818cf8", line=dict(width=1, color="#0b0f19")),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.08)",
        ))
        fig_line.update_layout(
            template="plotly_dark",
            xaxis_title="Session", yaxis_title="Capital ($)",
            margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        )
        fig_line.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_line.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_line, use_container_width=True)

        st.divider()

        st.subheader("📋 Historique des sessions")
        df_display = df.copy()
        df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")
        df_display = df_display.rename(columns={
            "Date": "Date", "Mode": "Mode", "Capital_Initial": "Capital Départ ($)",
            "Capital_Final": "Capital Final ($)", "Nb_Trades": "Nb Trades",
            "Trades_Gagnes": "Gagnés", "Trades_Perdus": "Perdus",
            "Payout_Pct": "Payout (%)", "Win_Rate_Pct": "Win Rate (%)",
            "Profit_Net": "Profit Net ($)", "Rendement_Pct": "Rendement (%)",
        })
        st.dataframe(
            df_display.drop(columns=["ID"]),
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------------------------
# ONGLET 2 : NOUVELLE SESSION (Mode A / Mode B)
# --------------------------------------------------------------------
with tab_new:
    st.subheader("Enregistrer une nouvelle session")

    mode_choice = st.radio(
        "Mode de saisie",
        ["Mode A — Reconstitution automatique (via Capital Final)", "Mode B — Saisie manuelle (W / L connus)"],
        horizontal=False,
    )

    prefill = st.session_state.prefill
    pv = st.session_state.prefill_version  # suffixe de clé pour forcer le refresh

    # ================= MODE A =================
    if mode_choice.startswith("Mode A"):
        st.caption(
            "Renseigne le capital de départ, le capital final observé, le nombre total de "
            "trades, le pourcentage de mise et le payout. L'application retrouve automatiquement le nombre de trades "
            "gagnés (W) et perdus (L) qui expliquent ce résultat."
        )
        with st.form("form_mode_a"):
            col1, col2 = st.columns(2)
            with col1:
                date_a = st.date_input("Date de la session", value=date.today())
                c_start_a = st.number_input("Capital Initial ($)", min_value=0.01, value=1000.0, step=10.0)
                n_a = st.number_input("Nombre total de trades (N)", min_value=1, value=20, step=1)
            with col2:
                mise_a_pct = st.number_input("Mise par trade (%)", min_value=1.0, max_value=50.0, value=4.0, step=0.5)
                c_end_a = st.number_input("Capital Final ($)", min_value=0.01, value=1100.0, step=10.0)
                payout_a = st.number_input("Payout (%)", min_value=1.0, max_value=500.0, value=85.0, step=1.0)

            submit_a = st.form_submit_button("🔎 Calculer W / L", use_container_width=True)

        if submit_a:
            try:
                mise_a = mise_a_pct / 100.0
                w_est, w_brut = solve_wins_from_capital(c_start_a, c_end_a, int(n_a), payout_a, mise_a)
                l_est = int(n_a) - w_est
                capital_theorique = compute_capital_final(c_start_a, w_est, l_est, payout_a, mise_a)
                erreur_pct = abs(capital_theorique - c_end_a) / c_end_a * 100

                st.session_state["mode_a_result"] = {
                    "date": date_a, "c_start": c_start_a, "c_end": c_end_a,
                    "n": int(n_a), "payout": payout_a,
                    "w": w_est, "l": l_est, "w_brut": w_brut,
                    "capital_theorique": capital_theorique, "erreur_pct": erreur_pct,
                }
            except ValueError as e:
                st.error(f"Erreur de calcul : {e}")
                st.session_state["mode_a_result"] = None

        res_a = st.session_state.get("mode_a_result")
        if res_a:
            st.success(
                f"✅ Résultat estimé : **{res_a['w']} trades gagnés** / **{res_a['l']} trades perdus** "
                f"sur {res_a['n']} trades (W brut avant arrondi : {res_a['w_brut']:.2f})"
            )
            if res_a["erreur_pct"] > 0.5:
                st.warning(
                    f"⚠️ Écart de {res_a['erreur_pct']:.2f}% entre le capital final théorique "
                    f"({res_a['capital_theorique']:.2f} $) et le capital final saisi ({res_a['c_end']:.2f} $). "
                    "Cela peut venir de l'arrondi de W ou d'un payout non constant sur la session."
                )
            cA, cB, cC = st.columns(3)
            cA.metric("Win Rate estimé", f"{res_a['w']/res_a['n']*100:.2f} %")
            cB.metric("Profit net", f"{res_a['c_end']-res_a['c_start']:+,.2f} $")
            cC.metric("Rendement session", f"{(res_a['c_end']-res_a['c_start'])/res_a['c_start']*100:+.2f} %")

            if st.button("💾 Enregistrer cette session", type="primary", use_container_width=True):
                row = build_session_row(
                    res_a["date"], "Mode A (auto)", res_a["c_start"], res_a["c_end"],
                    res_a["n"], res_a["w"], res_a["l"], res_a["payout"],
                )
                if append_session(row):
                    st.session_state["mode_a_result"] = None
                    st.success("Session enregistrée sur Supabase ✅")
                    st.rerun()

    # ================= MODE B =================
    else:
        st.caption(
            "Renseigne le capital de départ, le nombre exact de trades gagnés (ITM) et perdus (OTM), "
            "ainsi que le pourcentage de mise par trade et le payout."
        )
        default_c_start = prefill["c_start"] if prefill else 1000.0
        default_w = prefill["w"] if prefill else 10
        default_l = prefill["l"] if prefill else 5
        default_payout = prefill["payout"] if prefill else 85.0

        # Les `key` intègrent `pv` (prefill_version) : quand un transfert
        # depuis le Simulateur arrive, `pv` change et Streamlit recrée ces
        # widgets avec les nouvelles valeurs par défaut au lieu de garder
        # ce qui était déjà affiché à l'écran.
        with st.form("form_mode_b"):
            col1, col2 = st.columns(2)
            with col1:
                date_b = st.date_input("Date de la session", value=date.today(), key="date_b")
                c_start_b = st.number_input("Capital Initial ($)", min_value=0.01, value=float(default_c_start), step=10.0, key=f"c_start_b_{pv}")
                wins_b = st.number_input("Trades Gagnés (ITM)", min_value=0, value=int(default_w), step=1, key=f"wins_b_{pv}")
            with col2:
                mise_b_pct = st.number_input("Mise par trade (%)", min_value=1.0, max_value=50.0, value=4.0, step=0.5, key="mise_b_pct")
                payout_b = st.number_input("Payout (%)", min_value=1.0, max_value=500.0, value=float(default_payout), step=1.0, key=f"payout_b_{pv}")
                losses_b = st.number_input("Trades Perdus (OTM)", min_value=0, value=int(default_l), step=1, key=f"losses_b_{pv}")

            submit_b = st.form_submit_button("🔎 Calculer le Capital Final", use_container_width=True)

        if submit_b:
            n_b = int(wins_b) + int(losses_b)
            if n_b == 0:
                st.error("Il faut au moins un trade (gagné ou perdu).")
                st.session_state["mode_b_result"] = None
            else:
                mise_b = mise_b_pct / 100.0
                c_end_b = compute_capital_final(c_start_b, int(wins_b), int(losses_b), payout_b, mise_b)
                st.session_state["mode_b_result"] = {
                    "date": date_b, "c_start": c_start_b, "c_end": c_end_b,
                    "n": n_b, "payout": payout_b, "w": int(wins_b), "l": int(losses_b),
                }

        res_b = st.session_state.get("mode_b_result")
        if res_b:
            st.success(f"✅ Capital final calculé : **{res_b['c_end']:,.2f} $**")
            cA, cB, cC = st.columns(3)
            cA.metric("Win Rate", f"{res_b['w']/res_b['n']*100:.2f} %")
            cB.metric("Profit net", f"{res_b['c_end']-res_b['c_start']:+,.2f} $")
            cC.metric("Rendement session", f"{(res_b['c_end']-res_b['c_start'])/res_b['c_start']*100:+.2f} %")

            if st.button("💾 Enregistrer cette session", type="primary", use_container_width=True, key="save_b"):
                row = build_session_row(
                    res_b["date"], "Mode B (manuel)", res_b["c_start"], res_b["c_end"],
                    res_b["n"], res_b["w"], res_b["l"], res_b["payout"],
                )
                if append_session(row):
                    st.session_state["mode_b_result"] = None
                    st.session_state.prefill = None
                    st.success("Session enregistrée sur Supabase ✅")
                    st.rerun()

# --------------------------------------------------------------------
# ONGLET 3 : SIMULATEUR
# --------------------------------------------------------------------
with tab_sim:
    st.subheader("🧪 Simulateur de session (avant exécution réelle)")
    st.caption(
        "Teste l'impact théorique d'une session sans rien enregistrer. "
        "Ajuste le pourcentage de mise pour observer l'évolution de ton capital."
    )

    df_current = st.session_state.history
    capital_defaut = float(df_current.iloc[-1]["Capital_Final"]) if not df_current.empty else 1000.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        c_start_sim = st.number_input("Capital de départ ($)", min_value=0.01, value=capital_defaut, step=10.0, key="sim_cstart")
    with col2:
        n_sim = st.slider("Nombre de trades à simuler", min_value=1, max_value=30, value=5, key="sim_n")
    with col3:
        mise_sim_input = st.number_input("Mise par trade (%)", min_value=1.0, max_value=50.0, value=4.0, step=0.5, key="sim_mise")
    with col4:
        payout_sim = st.number_input("Payout (%)", min_value=1.0, max_value=500.0, value=85.0, step=1.0, key="sim_payout")

    mise_sim = mise_sim_input / 100.0

    win_rate_cible = st.slider("Win rate cible pour pré-remplir la séquence (%)", 0, 100, 60, key="sim_wr")

    nb_wins_defaut = int(round(n_sim * win_rate_cible / 100))
    sequence_defaut = ["✅ Gagné"] * nb_wins_defaut + ["❌ Perdu"] * (n_sim - nb_wins_defaut)

    df_seq = pd.DataFrame({
        "Trade #": list(range(1, n_sim + 1)),
        "Résultat": sequence_defaut,
    })

    st.markdown("**Ajuste manuellement le résultat de chaque trade si besoin :**")
    df_seq_edited = st.data_editor(
        df_seq,
        column_config={
            "Trade #": st.column_config.NumberColumn(disabled=True),
            "Résultat": st.column_config.SelectboxColumn(
                options=["✅ Gagné", "❌ Perdu"], required=True,
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="sim_editor",
    )

    if st.button("▶️ Lancer la simulation", type="primary", use_container_width=True):
        resultats_bool = [r == "✅ Gagné" for r in df_seq_edited["Résultat"]]
        df_sim = simulate_session_step_by_step(c_start_sim, resultats_bool, payout_sim, mise_sim)
        st.session_state["sim_result_df"] = df_sim
        st.session_state["sim_result_meta"] = {
            "c_start": c_start_sim, "payout": payout_sim, "mise_pct": mise_sim,
            "w": sum(resultats_bool), "l": len(resultats_bool) - sum(resultats_bool),
            "n": len(resultats_bool),
        }

    if "sim_result_df" in st.session_state and st.session_state["sim_result_df"] is not None:
        df_sim = st.session_state["sim_result_df"]
        meta = st.session_state["sim_result_meta"]
        c_end_sim = df_sim.iloc[-1]["Capital après ($)"]
        profit_sim = c_end_sim - meta["c_start"]

        best_case = compute_capital_final(meta["c_start"], meta["n"], 0, meta["payout"], meta["mise_pct"])
        worst_case = compute_capital_final(meta["c_start"], 0, meta["n"], meta["payout"], meta["mise_pct"])

        st.divider()
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Capital final simulé", f"{c_end_sim:,.2f} $")
        cB.metric("Profit net simulé", f"{profit_sim:+,.2f} $")
        cC.metric("Meilleur cas (100% gagné)", f"{best_case:,.2f} $")
        cD.metric("Pire cas (100% perdu)", f"{worst_case:,.2f} $")

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=df_sim["Trade #"], y=df_sim["Capital après ($)"],
            mode="lines+markers", name="Capital",
            line=dict(color="#a855f7", width=3),
            marker=dict(size=7, color="#c4b5fd", line=dict(width=1, color="#0b0f19")),
            fill="tozeroy", fillcolor="rgba(168, 85, 247, 0.08)",
        ))
        fig_sim.update_layout(
            template="plotly_dark",
            xaxis_title="Trade #", yaxis_title="Capital ($)", margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        )
        fig_sim.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_sim.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_sim, use_container_width=True)

        st.dataframe(df_sim, use_container_width=True, hide_index=True)

        if st.button("➡️ Utiliser ce résultat pour enregistrer une session réelle (Mode B)", use_container_width=True):
            st.session_state.prefill = {
                "c_start": meta["c_start"], "w": meta["w"], "l": meta["l"], "payout": meta["payout"],
            }
            st.session_state.prefill_version += 1  # force le refresh des champs du Mode B
            st.success("Valeurs transférées ! Va dans l'onglet **➕ Nouvelle Session** (Mode B) pour finaliser l'enregistrement.")

# --------------------------------------------------------------------
# ONGLET 4 : GESTION DES DONNÉES
# --------------------------------------------------------------------
with tab_data:
    st.subheader("🛠️ Gestion des données")
    st.caption("Toutes les données sont stockées dans ta base Supabase (cloud).")

    df_data = st.session_state.history

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 Télécharger l'historique")
        if df_data.empty:
            st.info("Aucune donnée à télécharger pour le moment.")
        else:
            csv_bytes = df_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Télécharger journal_trading.csv",
                data=csv_bytes,
                file_name="journal_trading.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col2:
        st.markdown("### 🗑️ Réinitialiser la base")
        st.warning("Cette action supprime **définitivement** toutes les sessions enregistrées sur Supabase.")

        if not st.session_state.confirm_reset:
            if st.button("🗑️ Supprimer toutes les données", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.error("⚠️ Confirmes-tu la suppression complète de l'historique ? Cette action est irréversible.")
            cconf1, cconf2 = st.columns(2)
            with cconf1:
                if st.button("✅ Oui, tout supprimer", type="primary", use_container_width=True):
                    if reset_history():
                        st.session_state.confirm_reset = False
                        st.success("Historique réinitialisé sur Supabase.")
                        st.rerun()
            with cconf2:
                if st.button("❌ Annuler", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

    st.divider()
    st.markdown("### 📄 Aperçu brut des données stockées")
    if df_data.empty:
        st.info("Aucune session enregistrée.")
    else:
        st.dataframe(df_data, use_container_width=True, hide_index=True)
     
