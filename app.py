import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_cleaning import load_and_clean, engineer_score, get_features

st.set_page_config(page_title="IPL Performance Predictor", page_icon="🏏", layout="wide")

# Train model
@st.cache_resource(show_spinner="Training model, please wait...")
def get_model():
    df = load_and_clean("chum.csv")
    df = engineer_score(df)
    features = get_features()
    X = df[features]
    y = df["performance_score"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, features

@st.cache_data
def get_data():
    df = load_and_clean("chum.csv")
    df = engineer_score(df)
    return df

# Sidebar
st.sidebar.title("🏏 IPL Performance Predictor")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔮 Predict", "📊 Explore Data", "📈 Visualisations"])

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏏 IPL Player Performance Score Predictor")
    st.markdown("""
    This app uses **Machine Learning** to predict a composite **Performance Score** for IPL players.

    | Component | Key Factors |
    |---|---|
    | 🏏 **Batting** | Runs, Average, Strike Rate, Centuries, Sixes |
    | 🎳 **Bowling** | Wickets, Economy Rate, 4/5-wicket hauls |
    | 🧤 **Fielding** | Catches, Stumpings |
    """)
    df = get_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Score", f"{df['performance_score'].mean():.1f}")
    c3.metric("Highest Score", f"{df['performance_score'].max():.1f}")
    st.markdown("### 🏆 All-Time Top 5")
    top5 = df[["Player_Name","Year","performance_score"]].sort_values("performance_score", ascending=False).head(5).reset_index(drop=True)
    top5.index += 1
    top5["Year"] = top5["Year"].astype(int)
    st.dataframe(top5, use_container_width=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Predict Player Performance")
    model, features = get_model()

    with st.form("form"):
        st.subheader("🏏 Batting")
        c1, c2, c3 = st.columns(3)
        matches_batted = c1.number_input("Matches Batted", 0, 20, 14)
        not_outs       = c2.number_input("Not Outs", 0, 20, 3)
        balls_faced    = c3.number_input("Balls Faced", 0, 1000, 350)
        batting_avg    = c1.number_input("Batting Average", 0.0, 200.0, 35.0)
        batting_sr     = c2.number_input("Strike Rate", 0.0, 300.0, 140.0)
        centuries      = c3.number_input("Centuries", 0, 10, 0)
        half_centuries = c1.number_input("Half Centuries", 0, 20, 2)
        fours          = c2.number_input("Fours", 0, 100, 25)
        sixes          = c3.number_input("Sixes", 0, 100, 10)

        st.subheader("🎳 Bowling")
        b1, b2, b3 = st.columns(3)
        matches_bowled = b1.number_input("Matches Bowled", 0, 20, 0)
        balls_bowled   = b2.number_input("Balls Bowled", 0, 500, 0)
        wickets        = b3.number_input("Wickets Taken", 0, 30, 0)
        economy        = b1.number_input("Economy Rate", 0.0, 20.0, 0.0)
        four_wkt       = b2.number_input("4-Wicket Hauls", 0, 5, 0)
        five_wkt       = b3.number_input("5-Wicket Hauls", 0, 5, 0)

        st.subheader("🧤 Fielding")
        f1, f2, f3 = st.columns(3)
        catches   = f1.number_input("Catches", 0, 30, 5)
        stumpings = f2.number_input("Stumpings", 0, 20, 0)
        year      = f3.number_input("Year", 2008, 2025, 2024)

        submitted = st.form_submit_button("🚀 Predict Score", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Matches_Batted": matches_batted, "Not_Outs": not_outs,
            "Balls_Faced": balls_faced, "Batting_Strike_Rate": batting_sr,
            "Batting_Average": batting_avg, "Centuries": centuries,
            "Half_Centuries": half_centuries, "Fours": fours, "Sixes": sixes,
            "Catches_Taken": catches, "Stumpings": stumpings,
            "Matches_Bowled": matches_bowled, "Balls_Bowled": balls_bowled,
            "Wickets_Taken": wickets, "Economy_Rate": economy,
            "Four_Wicket_Hauls": four_wkt, "Five_Wicket_Hauls": five_wkt,
            "Year": year,
        }])
        score = model.predict(row)[0]
        grade = "🌟 Elite" if score >= 800 else "🔥 Excellent" if score >= 500 else "✅ Good" if score >= 300 else "📊 Average" if score >= 150 else "📉 Below Avg"
        st.markdown("---")
        st.markdown(f"## Predicted Score: `{score:.1f}`")
        st.markdown(f"### Grade: {grade}")
        st.progress(min(score / 1400, 1.0))

# ── EXPLORE ───────────────────────────────────────────────────────────────────
elif page == "📊 Explore Data":
    st.title("📊 Explore Player Data")
    df = get_data()
    c1, c2 = st.columns(2)
    years = sorted(df["Year"].dropna().astype(int).unique(), reverse=True)
    year_filter = c1.selectbox("Filter by Year", ["All"] + [str(y) for y in years])
    top_n = c2.slider("Top N Players", 5, 50, 10)
    filtered = df if year_filter == "All" else df[df["Year"] == int(year_filter)]
    top = filtered[["Player_Name","Year","Runs_Scored","Wickets_Taken","performance_score"]].sort_values("performance_score", ascending=False).head(top_n).reset_index(drop=True)
    top.index += 1
    top["Year"] = top["Year"].astype(int)
    st.dataframe(top, use_container_width=True)

    st.markdown("---")
    name = st.text_input("🔍 Search a Player")
    if name:
        res = df[df["Player_Name"].str.contains(name, case=False, na=False)]
        if res.empty:
            st.warning("No player found.")
        else:
            res = res[["Player_Name","Year","Runs_Scored","Batting_Average","Wickets_Taken","performance_score"]].sort_values("Year", ascending=False)
            res["Year"] = res["Year"].astype(int)
            st.dataframe(res, use_container_width=True)

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
elif page == "📈 Visualisations":
    st.title("📈 Model Visualisations")
    plots = {
        "Score Distribution": "plots/score_distribution.png",
        "Model Comparison": "plots/model_comparison.png",
        "Actual vs Predicted": "plots/actual_vs_predicted.png",
        "Feature Importance": "plots/feature_importance.png",
    }
    for title, path in plots.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_column_width=True)
        else:
            st.info(f"Run `python model.py` to generate '{path}'")
