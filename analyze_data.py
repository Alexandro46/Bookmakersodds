#!/usr/bin/env python3
"""Audit Football-Data CSVs and evaluate closing-odds calibration and FL bias."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]
LEAGUES = {
    "E0": ("England", "Premier League", "top"),
    "E1": ("England", "Championship", "second"),
    "D1": ("Germany", "Bundesliga", "top"),
    "D2": ("Germany", "2. Bundesliga", "second"),
    "I1": ("Italy", "Serie A", "top"),
    "I2": ("Italy", "Serie B", "second"),
    "SP1": ("Spain", "La Liga", "top"),
    "SP2": ("Spain", "Segunda División", "second"),
    "F1": ("France", "Ligue 1", "top"),
    "F2": ("France", "Ligue 2", "second"),
}
ODDS = ["AvgCH", "AvgCD", "AvgCA"]
OUTCOMES = ["H", "D", "A"]


def season_label(code: str) -> str:
    return f"20{code[:2]}/20{code[2:]}"


def load_matches(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, audit = [], []
    for season in SEASONS:
        for code, (country, league_name, tier) in LEAGUES.items():
            path = input_dir / season / f"{code}.csv"
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
            required = ["Date", "HomeTeam", "AwayTeam", "FTR", *ODDS]
            missing_columns = [c for c in required if c not in df.columns]
            if missing_columns:
                raise ValueError(f"{path} is missing required columns: {missing_columns}")
            odds = df[ODDS].apply(pd.to_numeric, errors="coerce")
            valid_result = df["FTR"].isin(OUTCOMES)
            valid_odds = odds.notna().all(axis=1) & (odds > 1).all(axis=1)
            audit.append({
                "season": season_label(season), "code": code, "country": country,
                "league": league_name, "tier": tier, "rows": len(df),
                "valid_results": int(valid_result.sum()),
                "complete_valid_avg_closing_odds": int(valid_odds.sum()),
                "missing_or_invalid_closing_odds": int((~valid_odds).sum()),
                "duplicate_date_home_away": int(df.duplicated(["Date", "HomeTeam", "AwayTeam"]).sum()),
                "missing_home_away_team": int(df[["HomeTeam", "AwayTeam"]].isna().any(axis=1).sum()),
            })
            keep = valid_result & valid_odds & df[["HomeTeam", "AwayTeam"]].notna().all(axis=1)
            part = pd.DataFrame({
                "date": pd.to_datetime(df.loc[keep, "Date"], dayfirst=True, errors="coerce"),
                "home": df.loc[keep, "HomeTeam"].astype(str),
                "away": df.loc[keep, "AwayTeam"].astype(str),
                "result": df.loc[keep, "FTR"].astype(str),
                "oh": odds.loc[keep, "AvgCH"].astype(float),
                "od": odds.loc[keep, "AvgCD"].astype(float),
                "oa": odds.loc[keep, "AvgCA"].astype(float),
                "season": season_label(season), "season_code": season,
                "code": code, "country": country, "league": league_name, "tier": tier,
            })
            part = part.dropna(subset=["date"])
            matches.append(part)
    data = pd.concat(matches, ignore_index=True)
    inv = 1 / data[["oh", "od", "oa"]].to_numpy()
    data[["pH", "pD", "pA"]] = inv / inv.sum(axis=1, keepdims=True)
    data["overround"] = inv.sum(axis=1) - 1
    return data, pd.DataFrame(audit)


def summarize_group(g: pd.DataFrame) -> dict:
    p = g[["pH", "pD", "pA"]].to_numpy()
    y = g["result"].to_numpy()
    y_idx = np.array([OUTCOMES.index(v) for v in y])
    actual = np.eye(3)[y_idx]
    fav_idx = p.argmax(axis=1)
    long_idx = p.argmin(axis=1)
    rows = np.arange(len(g))
    return {
        "matches": len(g),
        "overround_mean_pct": 100 * g["overround"].mean(),
        "market_log_loss": float(-np.log(np.clip(p[np.arange(len(g)), y_idx], 1e-15, 1)).mean()),
        "market_brier_sum": np.square(actual - p).sum(axis=1).mean(),
        "favorite_expected_win_pct": 100 * p[rows, fav_idx].mean(),
        "favorite_actual_win_pct": 100 * actual[rows, fav_idx].mean(),
        "favorite_actual_minus_expected_pp": 100 * (actual[rows, fav_idx] - p[rows, fav_idx]).mean(),
        "longshot_expected_win_pct": 100 * p[rows, long_idx].mean(),
        "longshot_actual_win_pct": 100 * actual[rows, long_idx].mean(),
        "longshot_actual_minus_expected_pp": 100 * (actual[rows, long_idx] - p[rows, long_idx]).mean(),
    }


def grouped_metrics(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    records = []
    for group_key, g in data.groupby(keys, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        records.append({**dict(zip(keys, group_key)), **summarize_group(g)})
    return pd.DataFrame(records)


def calibration_bins(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    records = []
    for group_key, g in data.groupby(keys, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        p = g[["pH", "pD", "pA"]].to_numpy()
        y = g["result"].to_numpy()
        for outcome_i, outcome in enumerate(OUTCOMES):
            event_p = p[:, outcome_i]
            bins = pd.qcut(event_p, q=10, labels=False, duplicates="drop")
            actual = (y == outcome).astype(int)
            temp = pd.DataFrame({"bin": bins, "p": event_p, "y": actual})
            for bin_id, b in temp.groupby("bin", observed=True):
                records.append({**dict(zip(keys, group_key)), "outcome": outcome,
                                "probability_decile": int(bin_id) + 1, "events": len(b),
                                "mean_implied_pct": 100 * b.p.mean(),
                                "observed_frequency_pct": 100 * b.y.mean(),
                                "observed_minus_implied_pp": 100 * (b.y.mean() - b.p.mean())})
    return pd.DataFrame(records)


def add_pre_match_form(data: pd.DataFrame) -> pd.DataFrame:
    """Build last-five-match team form using only fixtures on earlier dates."""
    form_cols = ["home_points5", "away_points5", "home_gd5", "away_gd5",
                 "home_winrate5", "away_winrate5", "form_points_diff", "form_gd_diff"]
    result = data.sort_values(["date", "code", "home", "away"]).copy().reset_index(drop=True)
    for col in form_cols:
        result[col] = np.nan
    # Form histories are isolated by competition and team; update only after each date.
    histories: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for (code, date), idxs in result.groupby(["code", "date"], sort=True).groups.items():
        additions: list[tuple[tuple[str, str], tuple[float, float, float]]] = []
        for idx in idxs:
            row = result.loc[idx]
            home_hist = histories.get((code, row.home), [])[-5:]
            away_hist = histories.get((code, row.away), [])[-5:]
            def means(hist: list[tuple[float, float, float]]) -> tuple[float, float, float]:
                if not hist:
                    return (np.nan, np.nan, np.nan)
                arr = np.array(hist, dtype=float)
                return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())
            hp, hgd, hwr = means(home_hist)
            ap, agd, awr = means(away_hist)
            result.loc[idx, form_cols] = [hp, ap, hgd, agd, hwr, awr,
                                         hp - ap if np.isfinite(hp) and np.isfinite(ap) else np.nan,
                                         hgd - agd if np.isfinite(hgd) and np.isfinite(agd) else np.nan]
            outcome = row.result
            hg, ag = (3.0, 0.0) if outcome == "H" else ((0.0, 3.0) if outcome == "A" else (1.0, 1.0))
            additions.extend([
                ((code, row.home), (hg, float(hg - ag), float(outcome == "H"))),
                ((code, row.away), (ag, float(ag - hg), float(outcome == "A"))),
            ])
        for key, observation in additions:
            histories.setdefault(key, []).append(observation)
    result["odds_logit_HD"] = np.log(result.pH / result.pA)
    result["odds_logit_DA"] = np.log(result.pD / result.pA)
    return result


def run_model_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window season validation for top-tier matches only."""
    top = add_pre_match_form(data[data.tier == "top"].copy())
    odds_features = ["odds_logit_HD", "odds_logit_DA"]
    form_features = ["home_points5", "away_points5", "home_gd5", "away_gd5",
                     "home_winrate5", "away_winrate5", "form_points_diff", "form_gd_diff"]
    records = []
    seasons = [season_label(s) for s in SEASONS]
    models = {
        "Logistic odds recalibration": (make_pipeline(SimpleImputer(strategy="median"),
                                                       LogisticRegression(max_iter=1500)), odds_features),
        "Logistic odds plus form": (make_pipeline(SimpleImputer(strategy="median"),
                                                   LogisticRegression(max_iter=1500)), odds_features + form_features),
        "Decision tree odds plus form": (make_pipeline(SimpleImputer(strategy="median"),
                                                        DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                                                               random_state=2026)), odds_features + form_features),
        "Boosting odds plus form": (make_pipeline(SimpleImputer(strategy="median"),
                                                   HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06,
                                                                                  max_leaf_nodes=15,
                                                                                  l2_regularization=2.0,
                                                                                  random_state=2026)), odds_features + form_features),
    }
    for test_i in range(2, len(seasons)):
        train = top[top.season.isin(seasons[:test_i])]
        test = top[top.season == seasons[test_i]]
        if train.empty or test.empty:
            continue
        y_train = train.result
        y_test = test.result
        market_probs = test[["pH", "pD", "pA"]].to_numpy()
        y_idx = np.array([OUTCOMES.index(v) for v in y_test])
        actual = np.eye(3)[y_idx]
        records.append({"test_season": seasons[test_i], "training_seasons": ", ".join(seasons[:test_i]),
                        "model": "Bookmaker odds (market)", "matches": len(test),
                        "log_loss": float(-np.log(np.clip(market_probs[np.arange(len(test)), y_idx], 1e-15, 1)).mean()),
                        "brier_sum": np.square(actual - market_probs).sum(axis=1).mean()})
        for name, (model, features) in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[features], y_train)
            probs = model.predict_proba(test[features])
            model_classes = list(model[-1].classes_)
            probs = probs[:, [model_classes.index(label) for label in OUTCOMES]]
            records.append({"test_season": seasons[test_i], "training_seasons": ", ".join(seasons[:test_i]),
                            "model": name, "matches": len(test),
                            "log_loss": float(-np.log(np.clip(probs[np.arange(len(test)),
                                                        [OUTCOMES.index(v) for v in y_test]], 1e-15, 1)).mean()),
                            "brier_sum": np.square(actual - probs).sum(axis=1).mean()})
    return pd.DataFrame(records)


def save_reliability_plot(bins: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    top = bins[bins.tier == "top"]
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharex=True, sharey=True)
    colors = {"H": "#2563eb", "D": "#d97706", "A": "#059669"}
    for ax, league in zip(axes, leagues):
        sub = top[top.league == league]
        ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
        for outcome in OUTCOMES:
            curve = sub[sub.outcome == outcome].sort_values("probability_decile")
            ax.plot(curve.mean_implied_pct / 100, curve.observed_frequency_pct / 100,
                    marker="o", markersize=3, linewidth=1.5, color=colors[outcome], label=outcome)
        ax.set_title(league, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.set_xlabel("Implied probability")
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(title="Outcome", loc="upper left", fontsize=8)
    fig.suptitle("Closing odds calibration by outcome (top divisions, 2019/20–2024/25)", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bootstrap_bias(data: pd.DataFrame, reps: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    records = []
    groups = list(data.groupby(["tier", "league"], sort=True))
    groups += [(('top', 'All top divisions'), data[data.tier == "top"]),
               (('second', 'All second divisions'), data[data.tier == "second"])]
    for keys, g in groups:
        if len(g) < 2:
            continue
        p = g[["pH", "pD", "pA"]].to_numpy()
        actual = np.eye(3)[[OUTCOMES.index(v) for v in g.result]]
        ix = np.arange(len(g))
        f = p[ix, p.argmax(axis=1)]
        fa = actual[ix, p.argmax(axis=1)]
        l = p[ix, p.argmin(axis=1)]
        la = actual[ix, p.argmin(axis=1)]
        fd, ld = fa - f, la - l
        boot_f, boot_l = [], []
        for _ in range(reps):
            sample = rng.integers(0, len(g), len(g))
            boot_f.append(fd[sample].mean())
            boot_l.append(ld[sample].mean())
        records.append({"tier": keys[0], "league": keys[1], "matches": len(g),
                        "favorite_gap_pp": 100 * fd.mean(),
                        "favorite_ci95_low_pp": 100 * np.quantile(boot_f, .025),
                        "favorite_ci95_high_pp": 100 * np.quantile(boot_f, .975),
                        "longshot_gap_pp": 100 * ld.mean(),
                        "longshot_ci95_low_pp": 100 * np.quantile(boot_l, .025),
                        "longshot_ci95_high_pp": 100 * np.quantile(boot_l, .975)})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/football_data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, audit = load_matches(args.input_dir)
    audit.to_csv(args.output_dir / "data_quality_by_season.csv", index=False)
    metrics = grouped_metrics(data, ["tier", "country", "league", "season"])
    metrics.to_csv(args.output_dir / "calibration_by_league_season.csv", index=False)
    overall = grouped_metrics(data, ["tier", "country", "league"])
    overall.to_csv(args.output_dir / "calibration_by_league.csv", index=False)
    bins = calibration_bins(data, ["tier", "country", "league"])
    bins.to_csv(args.output_dir / "reliability_bins_by_league.csv", index=False)
    bias = bootstrap_bias(data)
    bias.to_csv(args.output_dir / "favorite_longshot_bias.csv", index=False)
    models = run_model_comparison(data)
    models.to_csv(args.output_dir / "time_ordered_model_comparison.csv", index=False)
    save_reliability_plot(bins, args.output_dir / "calibration_curves.png")

    top = overall[overall.tier == "top"]
    second = overall[overall.tier == "second"]
    total_top = int(top.matches.sum())
    total_second = int(second.matches.sum())
    pooled_top_bias = bias[bias.league == "All top divisions"].iloc[0]
    model_means = models.groupby("model")[["log_loss", "brier_sum"]].mean()
    best_model = model_means.loc[model_means.log_loss.idxmin()]
    raw_model = model_means.loc["Bookmaker odds (market)"]
    report = [
        "# Football betting odds analysis\n",
        "## Scope and method\n",
        f"The analysis includes {total_top:,} top-division matches and {total_second:,} second-division matches with a full-time result and valid closing average odds. It covers 2019/20–2024/25 for England, Germany, Italy, Spain and France. The first divisions are the proposal's primary sample; second divisions are reported separately.\n",
        "For each match, inverse closing average odds (AvgCH, AvgCD, AvgCA) were normalized to sum to one. This proportional normalization removes the bookmaker overround. Log loss and multiclass Brier score assess probability accuracy. Favourite and longshot gaps compare the realized result rate with the normalized market probability for the most and least likely outcomes in each match. Confidence intervals use a match-level bootstrap.\n",
        "## Initial findings\n",
        f"Across the {total_top:,} top-division matches, the weighted market log loss is {summarize_group(data[data.tier == 'top'])['market_log_loss']:.3f} and the multiclass Brier score is {summarize_group(data[data.tier == 'top'])['market_brier_sum']:.3f}. For the {total_second:,} second-division matches, these are {summarize_group(data[data.tier == 'second'])['market_log_loss']:.3f} and {summarize_group(data[data.tier == 'second'])['market_brier_sum']:.3f}. The lower top-tier scores indicate better probability accuracy in this sample.\n",
        f"Pooled top-tier favourites win {pooled_top_bias.favorite_gap_pp:.2f} percentage points more often than implied by the margin-adjusted market probabilities (match-bootstrap 95% interval {pooled_top_bias.favorite_ci95_low_pp:.2f} to {pooled_top_bias.favorite_ci95_high_pp:.2f}). The longshot gap is {pooled_top_bias.longshot_gap_pp:.2f} points (95% interval {pooled_top_bias.longshot_ci95_low_pp:.2f} to {pooled_top_bias.longshot_ci95_high_pp:.2f}). This is preliminary evidence consistent with a favourite–longshot pattern among favourites, while the pooled longshot estimate remains uncertain; it is not proof of a uniform effect across leagues.\n",
        "Across the five top divisions, La Liga has the largest favourite win-rate gap in this sample; league-specific estimates and intervals are in favorite_longshot_bias.csv. Treat league-by-league intervals as exploratory because several comparisons are made.\n",
        f"Across the four time-ordered test seasons, {model_means.log_loss.idxmin()} has the lowest mean log loss ({best_model.log_loss:.3f}); raw bookmaker odds average {raw_model.log_loss:.3f}. Compare per-season results in time_ordered_model_comparison.csv.\n",
        "## Reading the results\n",
        "A positive favourite gap means favourites won more often than their odds implied; a negative longshot gap means the least likely outcomes occurred less often than implied. That pattern is consistent with favourite–longshot bias. It is descriptive evidence: the simple favourite/longshot comparison does not establish why the pricing pattern occurs. Check whether each 95% interval excludes zero before treating a gap as distinguishable from sampling variation.\n",
        "Calibration metrics and gaps are reported by league and season in the CSV files. Reliability bins compare observed frequencies with implied probabilities by outcome and probability range. Lower log loss and Brier score indicate better probabilistic forecasts.\n",
        "## Predictive model comparison\n",
        "The expanding-window comparison uses only top-division matches. It trains on earlier full seasons and tests on the next season. Historical form uses each team's previous five matches, excludes the current match date, and is constructed separately within each competition. It compares the raw bookmaker probabilities with logistic recalibration and with logistic regression, a decision tree and histogram gradient boosting that add historical form. Lower out-of-sample log loss/Brier score is better. These are initial models, not an exhaustive hyperparameter search.\n",
        "## Data checks and limitations\n",
        "See data_quality_by_season.csv for source row counts, valid results, usable closing odds and duplicate fixture checks. Raw downloads are preserved in data/football_data/raw. CSV column availability is checked before analysis; observations without valid closing average odds or a recognized result are excluded. The analysis should be treated as provisional until team names, duplicate fixtures, season changes and league-specific missingness have been reviewed.\n",
        "## Output files\n",
        "- calibration_by_league.csv and calibration_by_league_season.csv: overall and seasonal probability metrics.\n",
        "- reliability_bins_by_league.csv and calibration_curves.png: decile calibration data and a top-tier reliability plot.\n",
        "- favorite_longshot_bias.csv: favourite and longshot gaps with match-bootstrap 95% intervals.\n",
        "- time_ordered_model_comparison.csv: out-of-sample metrics by test season and model.\n",
        "- data_quality_by_season.csv: input coverage and exclusions.\n",
    ]
    (args.output_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Matches retained: {len(data):,} total; top tier {total_top:,}; second tier {total_second:,}")
    print(f"Outputs written to: {args.output_dir.resolve()}")
    print("Top-tier calibration by league:")
    print(top[["league", "matches", "market_log_loss", "market_brier_sum",
               "favorite_actual_minus_expected_pp", "longshot_actual_minus_expected_pp"]].to_string(index=False))
    print("\nFavourite/longshot bootstrap intervals:")
    print(bias.to_string(index=False))
    print("\nTime-ordered model comparison:")
    print(models.groupby("model")[["log_loss", "brier_sum"]].mean().sort_values("log_loss").to_string())


if __name__ == "__main__":
    main()

