"""
data_cleaning.py
────────────────
Loads and cleans the IPL player stats CSV.
Returns a cleaned DataFrame with an engineered performance_score column.
"""

import pandas as pd
import numpy as np


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Replace placeholder strings
    df.replace("No stats", np.nan, inplace=True)

    # Drop rows with no Year
    df.dropna(subset=["Year"], inplace=True)

    # Remove not-out asterisk from Highest_Score
    df["Highest_Score"] = (
        df["Highest_Score"].astype(str).str.replace("*", "", regex=False)
    )

    # Convert all stat columns to numeric
    skip_cols = ["Player_Name", "Best_Bowling_Match"]
    numeric_cols = [c for c in df.columns if c not in skip_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing stats with 0
    stat_cols = [c for c in numeric_cols if c != "Year"]
    df[stat_cols] = df[stat_cols].fillna(0)

    df = df.reset_index(drop=True)
    return df


def engineer_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add batting, bowling, fielding sub-scores and composite performance_score."""
    df = df.copy()

    df["batting_score"] = (
        df["Runs_Scored"]         * 1.0
        + df["Batting_Average"]   * 2.0
        + df["Batting_Strike_Rate"] * 0.5
        + df["Centuries"]         * 20.0
        + df["Half_Centuries"]    * 8.0
        + df["Fours"]             * 0.5
        + df["Sixes"]             * 1.5
    )

    df["bowling_score"] = (
        df["Wickets_Taken"]     * 15.0
        + df["Four_Wicket_Hauls"] * 20.0
        + df["Five_Wicket_Hauls"] * 30.0
    )
    # Penalise high economy (only for players who bowled)
    df["bowling_score"] = df.apply(
        lambda r: r["bowling_score"] - (r["Economy_Rate"] * 5)
        if r["Balls_Bowled"] > 0 else r["bowling_score"],
        axis=1,
    )

    df["fielding_score"] = df["Catches_Taken"] * 3.0 + df["Stumpings"] * 5.0

    df["performance_score"] = (
        df["batting_score"] + df["bowling_score"] + df["fielding_score"]
    )

    return df


def get_features() -> list:
    return [
        "Matches_Batted", "Not_Outs", "Balls_Faced", "Batting_Strike_Rate",
        "Batting_Average", "Centuries", "Half_Centuries", "Fours", "Sixes",
        "Catches_Taken", "Stumpings", "Matches_Bowled", "Balls_Bowled",
        "Wickets_Taken", "Economy_Rate", "Four_Wicket_Hauls", "Five_Wicket_Hauls",
        "Year",
    ]


if __name__ == "__main__":
    df = load_and_clean("chum.csv")
    df = engineer_score(df)
    print(f"Shape      : {df.shape}")
    print(f"Score range: {df['performance_score'].min():.1f} – {df['performance_score'].max():.1f}")
    print("\nTop 5 performers:")
    print(
        df[["Player_Name", "Year", "performance_score"]]
        .sort_values("performance_score", ascending=False)
        .head(5)
        .to_string(index=False)
    )
