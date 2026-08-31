# =====================================================================
# Application Streamlit — GMAO Transformateurs
# Visualisation et prédiction de l'état des transformateurs
# =====================================================================
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

try:
    from scipy.stats import chi2_contingency  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for stats tab
    chi2_contingency = None

from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

# ---------------------------------------------------------------------
# Configuration générale de la page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="GMAO Transformateurs",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "propre": "#2DD4BF",
    "fuite": "#E5484D",
    "amber": "#F5A623",
    "muted": "#7C8AA3",
}

st.markdown(
    """
    <style>
    .metric-card{background:#111A2C;border:1px solid rgba(45,212,191,0.15);
        border-radius:6px;padding:14px 18px;}
    div[data-testid="stMetricValue"]{font-size:28px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Chargement et nettoyage des données
# ---------------------------------------------------------------------
DEFAULT_DATA_PATH = "C:\Users\KETSIA TSHILIKA\OneDrive\Desktop\gmao-transfo-app"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduit le nettoyage des noms de colonnes fait dans le notebook."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\(kva\)", "_kva", regex=True)
        .str.replace(r"\(v\)", "_v", regex=True)
        .str.replace(r"\(°c\)", "_c", regex=True)
        .str.replace(r"\(%\)", "", regex=True)
        .str.replace(r"[^a-z0-9_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def find_col(df: pd.DataFrame, keyword: str):
    for c in df.columns:
        if keyword in c:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df = clean_columns(df)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "buchings" in df.columns:
        df["buchings"] = df["buchings"].astype(str).str.strip().str.lower()

    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, features: list, target: str, cat_cols: list):
    """Entraîne un RandomForest comme dans le notebook, retourne tout le nécessaire."""
    X = df[features].copy()
    y = df[target].copy()

    X_encoded = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns])
    numerical = [c for c in features if c not in cat_cols and pd.api.types.is_numeric_dtype(X[c])]

    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    if numerical:
        X_scaled[numerical] = scaler.fit_transform(X_scaled[numerical])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=780, random_state=42)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "numerical": numerical,
        "encoded_columns": X_scaled.columns.tolist(),
        "cat_cols": cat_cols,
        "features": features,
    }


# ---------------------------------------------------------------------
# Barre latérale — chargement du fichier
# ---------------------------------------------------------------------
st.sidebar.title("🔌 GMAO Transformateurs")
st.sidebar.caption("Maintenance prédictive du parc de transformateurs")

uploaded = st.sidebar.file_uploader(
    "Charger le dataset (.xlsx)", type=["xlsx"], help="Le fichier exporté depuis ton pipeline de données."
)

data_source = uploaded if uploaded is not None else DEFAULT_DATA_PATH
try:
    df = load_data(data_source)
    if uploaded is None:
        st.sidebar.info("Aucun fichier chargé — utilisation du jeu de données d'exemple.")
except FileNotFoundError:
    st.error(
        f"Fichier introuvable : `{DEFAULT_DATA_PATH}`. "
        "Dépose ton fichier via le panneau de gauche, ou ajoute un exemple dans `data/`."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture du fichier : {e}")
    st.stop()

# Détection des colonnes clés (tolérant aux variations de nommage)
COL_ID = find_col(df, "transfo_id") or "transfo_id"
COL_DATE = "date" if "date" in df.columns else None
COL_ASPECT = find_col(df, "aspet_gen") or find_col(df, "aspect_gen")
COL_OIL = find_col(df, "niveau_huile")
COL_SILICA = find_col(df, "silicagel")
COL_RELAIS = find_col(df, "relais_buchh")
COL_TARGET = "buchings" if "buchings" in df.columns else None
COL_KVA = find_col(df, "puissance_kva")
COL_TEMP = find_col(df, "temp_huile")

required = {"ID transformateur": COL_ID, "date": COL_DATE, "buchings": COL_TARGET}
missing = [k for k, v in required.items() if v is None or v not in df.columns]
if missing:
    st.error(
        "Colonnes essentielles introuvables dans le fichier : "
        + ", ".join(missing)
        + ". Vérifie le format de ton dataset."
    )
    st.stop()

# Filtres sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")
all_ids = sorted(df[COL_ID].unique().tolist())
selected_ids = st.sidebar.multiselect("Transformateurs", all_ids, default=[])
if COL_DATE:
    dmin, dmax = df[COL_DATE].min(), df[COL_DATE].max()
    date_range = st.sidebar.date_input("Période", (dmin, dmax))
else:
    date_range = None

df_f = df.copy()
if selected_ids:
    df_f = df_f[df_f[COL_ID].isin(selected_ids)]
if COL_DATE and isinstance(date_range, tuple) and len(date_range) == 2:
    df_f = df_f[(df_f[COL_DATE] >= pd.Timestamp(date_range[0])) & (df_f[COL_DATE] <= pd.Timestamp(date_range[1]))]

st.sidebar.markdown("---")
st.sidebar.caption(f"{df_f[COL_ID].nunique()} transformateur(s) · {len(df_f)} inspection(s) affichée(s)")

# ---------------------------------------------------------------------
# Navigation par onglets
# ---------------------------------------------------------------------
st.title("Panneau de contrôle — Maintenance prédictive des transformateurs")

tab_overview, tab_stats, tab_model, tab_data = st.tabs(
    ["📊 Vue d'ensemble", "🔬 Analyse statistique", "🤖 Modèle & prédictions", "📄 Données brutes"]
)

# ============================== VUE D'ENSEMBLE ==============================
with tab_overview:
    total_transfos = df_f[COL_ID].nunique()
    total_inspections = len(df_f)
    total_fuite = int((df_f[COL_TARGET] == "fuite").sum())
    taux_fuite = round(100 * total_fuite / total_inspections, 1) if total_inspections else 0

    dernier_etat = df_f.sort_values(COL_DATE).groupby(COL_ID)[COL_TARGET].last()
    transfos_critiques = int((dernier_etat == "fuite").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transformateurs suivis", total_transfos)
    c2.metric("Inspections totales", total_inspections)
    c3.metric("Taux de fuite global", f"{taux_fuite}%")
    c4.metric("Unités en état critique", transfos_critiques)

    st.markdown("#### État actuel du parc")
    led_cols = st.columns(13)
    for i, (tid, status) in enumerate(dernier_etat.items()):
        color = PALETTE["fuite"] if status == "fuite" else PALETTE["propre"]
        with led_cols[i % 13]:
            st.markdown(
                f"""<div title="{tid} — {status}" style="background:{color};
                border-radius:4px;height:34px;display:flex;align-items:center;
                justify-content:center;font-size:9px;color:#0A0F1C;font-weight:700;
                margin-bottom:6px;">{tid}</div>""",
                unsafe_allow_html=True,
            )

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            values=[total_inspections - total_fuite, total_fuite],
            names=["Propre", "Fuite"],
            color_discrete_sequence=[PALETTE["propre"], PALETTE["fuite"]],
            hole=0.55,
            title="Répartition globale des Buchings",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        comptage = df_f.groupby([COL_ID, COL_TARGET]).size().unstack(fill_value=0)
        for c in ["fuite", "propre"]:
            if c not in comptage.columns:
                comptage[c] = 0
        comptage = comptage.sort_values("fuite", ascending=False).head(16)
        fig = go.Figure()
        fig.add_bar(x=comptage.index, y=comptage["fuite"], name="Fuite", marker_color=PALETTE["fuite"])
        fig.add_bar(x=comptage.index, y=comptage["propre"], name="Propre", marker_color="rgba(45,212,191,0.35)")
        fig.update_layout(barmode="stack", title="Top 16 — fuites par transformateur")
        st.plotly_chart(fig, use_container_width=True)

# ============================== ANALYSE STATISTIQUE ==============================
with tab_stats:
    col1, col2 = st.columns(2)

    with col1:
        if COL_ASPECT:
            st.markdown("#### Aspect général vs. Buchings")
            tab_cont = pd.crosstab(df_f[COL_ASPECT], df_f[COL_TARGET])
            st.dataframe(tab_cont.style.background_gradient(cmap="YlGnBu"), use_container_width=True)

            stat, p, dof, exp = chi2_contingency(tab_cont)
            sig = "✅ significatif" if p < 0.05 else "❌ non significatif"
            st.metric("Test Chi² — p-value", f"{p:.2e}", sig)
        else:
            st.info("Colonne 'aspect général' non détectée dans ce dataset.")

    with col2:
        if COL_RELAIS:
            st.markdown("#### Relais Buchholz vs. Buchings")
            tab_relais = pd.crosstab(df_f[COL_RELAIS], df_f[COL_TARGET])
            st.dataframe(tab_relais.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

            stat, p, dof, exp = chi2_contingency(tab_relais)
            sig = "✅ significatif" if p < 0.05 else "❌ non significatif"
            st.metric("Test Chi² — p-value", f"{p:.2e}", sig)
        else:
            st.info("Colonne 'relais Buchholz' non détectée dans ce dataset.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        if COL_SILICA:
            st.markdown("#### Dispersion du Silicagel par transformateur")
            sg = df_f.groupby(COL_ID)[COL_SILICA].agg(["mean", "std", "min", "max"]).sort_values(
                "std", ascending=False
            )
            fig = px.bar(sg.reset_index(), x=COL_ID, y="std", color="std",
                         color_continuous_scale="Oranges", title="Écart-type du Silicagel")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'silicagel' non détectée dans ce dataset.")

    with col4:
        if COL_OIL:
            st.markdown("#### Répartition du niveau d'huile")
            oil_counts = df_f[COL_OIL].value_counts()
            fig = px.bar(
                x=oil_counts.values, y=oil_counts.index, orientation="h",
                title="États du niveau d'huile", labels={"x": "Nombre d'inspections", "y": ""},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'niveau d'huile' non détectée dans ce dataset.")

    if COL_SILICA and COL_DATE:
        st.markdown("#### Carte thermique — Silicagel dans le temps")
        pivot = df_f.pivot_table(index=COL_ID, columns=COL_DATE, values=COL_SILICA)
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto",
                         labels=dict(color="Niveau (%)"))
        st.plotly_chart(fig, use_container_width=True)

# ============================== MODÈLE & PRÉDICTIONS ==============================
with tab_model:
    candidate_features = [c for c in [COL_KVA, find_col(df, "tension_pri"), find_col(df, "tension_sec"),
                                       COL_ASPECT, COL_OIL, COL_TEMP, COL_SILICA, COL_RELAIS] if c]
    candidate_features = [c for c in candidate_features if c in df.columns]

    if not candidate_features or df[COL_TARGET].nunique() < 2:
        st.warning("Pas assez de colonnes ou de variabilité pour entraîner un modèle sur ce dataset.")
    else:
        with st.spinner("Entraînement du modèle Random Forest..."):
            cat_cols = [c for c in [COL_ASPECT, COL_OIL, COL_RELAIS] if c in candidate_features]
            bundle = train_model(df, candidate_features, COL_TARGET, cat_cols)

        model = bundle["model"]
        acc = accuracy_score(bundle["y_test"], model.predict(bundle["X_test"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Type de modèle", "Random Forest")
        c2.metric("Précision (accuracy)", f"{acc*100:.1f}%")
        c3.metric("Échantillons de test", len(bundle["y_test"]))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Importance des variables")
            importances = pd.Series(model.feature_importances_, index=bundle["encoded_columns"])
            importances = importances.sort_values(ascending=True).tail(12)
            fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                         title="Variables les plus influentes")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Matrice de confusion")
            cm = confusion_matrix(bundle["y_test"], model.predict(bundle["X_test"]), labels=model.classes_)
            fig = px.imshow(cm, x=model.classes_, y=model.classes_, text_auto=True,
                             color_continuous_scale="Blues",
                             labels=dict(x="Prédit", y="Réel"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Prédiction à N jours")
        n_days = st.slider("Nombre de jours à prédire", 7, 60, 30)

        if st.button("Lancer la prédiction"):
            with st.spinner("Calcul des prédictions futures..."):
                last_date = df[COL_DATE].max()
                future_dates = [last_date + timedelta(days=d) for d in range(1, n_days + 1)]
                unique_ids = df[COL_ID].unique()

                rows = []
                for tid in unique_ids:
                    sub = df[df[COL_ID] == tid]
                    for fdate in future_dates:
                        row = {COL_ID: tid, COL_DATE: fdate}
                        for col in candidate_features:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                row[col] = sub[col].mean()
                            else:
                                mode = sub[col].mode()
                                row[col] = mode.iloc[0] if not mode.empty else sub[col].iloc[0]
                        rows.append(row)

                future_df = pd.DataFrame(rows)
                X_future = pd.get_dummies(future_df[candidate_features], columns=cat_cols)
                for c in bundle["encoded_columns"]:
                    if c not in X_future.columns:
                        X_future[c] = 0
                X_future = X_future[bundle["encoded_columns"]]
                if bundle["numerical"]:
                    X_future[bundle["numerical"]] = bundle["scaler"].transform(X_future[bundle["numerical"]])

                preds = model.predict(X_future)
                probs = model.predict_proba(X_future)
                fuite_idx = list(model.classes_).index("fuite") if "fuite" in model.classes_ else 0

                result = pd.DataFrame({
                    "Date": future_df[COL_DATE].dt.strftime("%d/%m/%Y"),
                    "Transformateur": future_df[COL_ID],
                    "Statut prédit": preds,
                    "Confiance fuite (%)": (probs[:, fuite_idx] * 100).round(1),
                })
                result = result.sort_values("Confiance fuite (%)", ascending=False)
                st.session_state["future_predictions"] = result

        if "future_predictions" in st.session_state:
            result = st.session_state["future_predictions"]
            st.dataframe(
                result.style.background_gradient(subset=["Confiance fuite (%)"], cmap="Reds"),
                use_container_width=True, height=420,
            )
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les prédictions (CSV)", csv, "predictions_futures.csv", "text/csv")

# ============================== DONNÉES BRUTES ==============================
with tab_data:
    st.markdown("#### Données filtrées")
    st.dataframe(df_f, use_container_width=True, height=500)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger (CSV)", csv, "donnees_filtrees.csv", "text/csv")

st.markdown("---")
st.caption("GMAO Transformateurs · Application Streamlit générée à partir du pipeline de maintenance prédictive")
