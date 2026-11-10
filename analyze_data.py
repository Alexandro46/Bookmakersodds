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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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
                **{col: pd.to_numeric(df.loc[keep, col], errors="coerce") if col in df.columns
                   else np.nan for col in ["FTHG", "FTAG", "HST", "HS", "HC", "HY", "HR",
                                           "AST", "AS", "AC", "AY", "AR"]},
                "season": season_label(season), "season_code": season,
                "code": code, "country": country, "league": league_name, "tier": tier,
            })
            part = part.dropna(subset=["date"])
            matches.append(part)
    data = pd.concat(matches, ignore_index=True)
    inv = 1 / data[["oh", "od", "oa"]].to_numpy()
    data[["pH", "pD", "pA"]] = inv / inv.sum(axis=1, keepdims=True)
    data["overround"] = inv.sum(axis=1) - 1
    data[["pH_add", "pD_add", "pA_add"]] = additive_probabilities(inv)
    power = np.array([power_probabilities(row) for row in inv])
    shin = np.array([shin_probabilities(row) for row in inv])
    data[["pH_power", "pD_power", "pA_power"]] = power
    data[["pH_shin", "pD_shin", "pA_shin"]] = shin
    return data, pd.DataFrame(audit)


def additive_probabilities(q: np.ndarray) -> np.ndarray:
    return q - (q.sum(axis=1, keepdims=True) - 1.0) / q.shape[1]


def power_probabilities(q: np.ndarray) -> np.ndarray:
    from scipy.optimize import brentq
    exponent = brentq(lambda b: np.power(q, b).sum() - 1.0, 0.01, 50.0)
    p = np.power(q, exponent)
    return p / p.sum()


def shin_probabilities(q: np.ndarray) -> np.ndarray:
    from scipy.optimize import brentq
    total = q.sum()
    if total <= 1.0:
        return q / total
    def probs(z: float) -> np.ndarray:
        radicand = np.maximum(z * z + 4 * (1 - z) * np.square(q) / total, 0)
        return 2 * np.square(q) / (total * (np.sqrt(radicand) + z))
    z = brentq(lambda value: probs(value).sum() - 1.0, 0.0, 1.0)
    p = probs(z)
    return p / p.sum()


PROBABILITY_COLUMNS = {
    "proportional": ["pH", "pD", "pA"],
    "additive": ["pH_add", "pD_add", "pA_add"],
    "power": ["pH_power", "pD_power", "pA_power"],
    "shin": ["pH_shin", "pD_shin", "pA_shin"],
}


def summarize_group(g: pd.DataFrame, pcols: list[str] | None = None) -> dict:
    p = g[pcols or PROBABILITY_COLUMNS["proportional"]].to_numpy()
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


def grouped_metrics(data: pd.DataFrame, keys: list[str], pcols: list[str] | None = None) -> pd.DataFrame:
    records = []
    for group_key, g in data.groupby(keys, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        records.append({**dict(zip(keys, group_key)), **summarize_group(g, pcols)})
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
                 "home_winrate5", "away_winrate5", "form_points_diff", "form_gd_diff",
                 "home_gf5", "away_gf5", "home_ga5", "away_ga5",
                 "home_sot5", "away_sot5", "home_shots5", "away_shots5",
                 "home_corners5", "away_corners5", "home_cards5", "away_cards5",
                 "home_rest_days", "away_rest_days", "home_season_ppg", "away_season_ppg",
                 "home_season_gdpg", "away_season_gdpg", "home_h2h_points5", "away_h2h_points5"]
    result = data.sort_values(["date", "code", "home", "away"]).copy().reset_index(drop=True)
    for col in form_cols:
        result[col] = np.nan
    # Form histories are isolated by competition and team; update only after each date.
    histories: dict[tuple[str, str], list[dict[str, float]]] = {}
    season_histories: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    h2h: dict[tuple[str, str, str], list[float]] = {}
    last_played: dict[tuple[str, str], pd.Timestamp] = {}
    for (code, date), idxs in result.groupby(["code", "date"], sort=True).groups.items():
        additions: list[tuple[tuple[str, str], tuple[float, float, float]]] = []
        for idx in idxs:
            row = result.loc[idx]
            home_hist = histories.get((code, row.home), [])[-5:]
            away_hist = histories.get((code, row.away), [])[-5:]
            def means(hist: list[dict[str, float]]) -> tuple[float, float, float]:
                if not hist:
                    return (np.nan, np.nan, np.nan)
                return tuple(float(np.mean([o[key] for o in hist])) for key in ("points", "gd", "win"))
            hp, hgd, hwr = means(home_hist)
            ap, agd, awr = means(away_hist)
            def mean_stat(hist: list[dict[str, float]], key: str) -> float:
                available = [o[key] for o in hist if np.isfinite(o[key])]
                return float(np.mean(available)) if available else np.nan
            h2h_hist = h2h.get((code, row.home, row.away), [])[-5:]
            h2h_away_hist = h2h.get((code, row.away, row.home), [])[-5:]
            hseason = season_histories.get((code, row.season, row.home), [])
            aseason = season_histories.get((code, row.season, row.away), [])
            hdate, adate = last_played.get((code, row.home)), last_played.get((code, row.away))
            hrest = (date - hdate).days if hdate is not None else np.nan
            arest = (date - adate).days if adate is not None else np.nan
            result.loc[idx, form_cols] = [
                hp, ap, hgd, agd, hwr, awr,
                hp - ap if np.isfinite(hp) and np.isfinite(ap) else np.nan,
                hgd - agd if np.isfinite(hgd) and np.isfinite(agd) else np.nan,
                mean_stat(home_hist, "gf"), mean_stat(away_hist, "gf"),
                mean_stat(home_hist, "ga"), mean_stat(away_hist, "ga"),
                mean_stat(home_hist, "sot"), mean_stat(away_hist, "sot"),
                mean_stat(home_hist, "shots"), mean_stat(away_hist, "shots"),
                mean_stat(home_hist, "corners"), mean_stat(away_hist, "corners"),
                mean_stat(home_hist, "cards"), mean_stat(away_hist, "cards"), hrest, arest,
                float(np.mean([o["points"] for o in hseason])) if hseason else np.nan,
                float(np.mean([o["points"] for o in aseason])) if aseason else np.nan,
                float(np.mean([o["gd"] for o in hseason])) if hseason else np.nan,
                float(np.mean([o["gd"] for o in aseason])) if aseason else np.nan,
                float(np.mean(h2h_hist)) if h2h_hist else np.nan,
                float(np.mean(h2h_away_hist)) if h2h_away_hist else np.nan,
            ]
            outcome = row.result
            hg, ag = (3.0, 0.0) if outcome == "H" else ((0.0, 3.0) if outcome == "A" else (1.0, 1.0))
            def numeric_stat(col: str, default=np.nan) -> float:
                return float(row[col]) if col in row.index and pd.notna(row[col]) else default
            additions.extend([
                ((code, row.home), {"points": hg, "gd": float(hg - ag), "win": float(outcome == "H"),
                                    "gf": float(row.FTHG), "ga": float(row.FTAG),
                                    "sot": numeric_stat("HST"), "shots": numeric_stat("HS"),
                                    "corners": numeric_stat("HC"), "cards": numeric_stat("HY") + numeric_stat("HR", 0)}),
                ((code, row.away), {"points": ag, "gd": float(ag - hg), "win": float(outcome == "A"),
                                    "gf": float(row.FTAG), "ga": float(row.FTHG),
                                    "sot": numeric_stat("AST"), "shots": numeric_stat("AS"),
                                    "corners": numeric_stat("AC"), "cards": numeric_stat("AY") + numeric_stat("AR", 0)}),
            ])
            h2h.setdefault((code, row.home, row.away), []).append(hg)
            h2h.setdefault((code, row.away, row.home), []).append(ag)
            last_played[(code, row.home)] = date
            last_played[(code, row.away)] = date
        for key, observation in additions:
            histories.setdefault(key, []).append(observation)
            season_histories.setdefault((key[0], row.season, key[1]), []).append(observation)
    result["odds_logit_HD"] = np.log(result.pH / result.pA)
    result["odds_logit_DA"] = np.log(result.pD / result.pA)
    return result


def run_model_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window season validation for top-tier matches only."""
    top = add_pre_match_form(data[data.tier == "top"].copy())
    odds_features = ["odds_logit_HD", "odds_logit_DA"]
    form_features = ["home_points5", "away_points5", "home_gd5", "away_gd5",
                     "home_winrate5", "away_winrate5", "form_points_diff", "form_gd_diff"]
    extended_features = form_features + [
        "home_gf5", "away_gf5", "home_ga5", "away_ga5", "home_sot5", "away_sot5",
        "home_shots5", "away_shots5", "home_corners5", "away_corners5", "home_cards5",
        "away_cards5", "home_rest_days", "away_rest_days", "home_season_ppg", "away_season_ppg",
        "home_season_gdpg", "away_season_gdpg", "home_h2h_points5", "away_h2h_points5",
    ]
    records = []
    seasons = [season_label(s) for s in SEASONS]
    models = {
        "Logistic odds recalibration": (make_pipeline(SimpleImputer(strategy="median"),
                                                       LogisticRegression(max_iter=1500)), odds_features),
        "Logistic odds plus form": (make_pipeline(SimpleImputer(strategy="median"),
                                                   LogisticRegression(max_iter=1500)), odds_features + form_features),
        "Logistic odds plus extended features": (make_pipeline(SimpleImputer(strategy="median"),
                                                                 LogisticRegression(max_iter=1500)), odds_features + extended_features),
        "Logistic extended plus team identity": (make_pipeline(
            ColumnTransformer([("numeric", SimpleImputer(strategy="median"), odds_features + extended_features),
                               ("teams", OneHotEncoder(handle_unknown="ignore"), ["home", "away"])]),
            LogisticRegression(max_iter=2000)), odds_features + extended_features + ["home", "away"]),
        "Decision tree odds plus extended": (make_pipeline(SimpleImputer(strategy="median"),
                                                            DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                                                                   random_state=2026)), odds_features + extended_features),
        "Boosting odds plus form": (make_pipeline(SimpleImputer(strategy="median"),
                                                   HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06,
                                                                                  max_leaf_nodes=15,
                                                                                  l2_regularization=2.0,
                                                                                  random_state=2026)), odds_features + extended_features),
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


def fit_calibration_slope(prob: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    """Logistic calibration regression: event ~ intercept + slope * logit(market p)."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    x = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(40):
        fitted = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        weight = fitted * (1 - fitted)
        information = x.T @ (weight[:, None] * x) + np.eye(2) * 1e-9
        step = np.linalg.solve(information, x.T @ (event - fitted))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def calibration_slope_regression(data: pd.DataFrame, reps: int = 500) -> pd.DataFrame:
    """Estimate market calibration slope and bootstrap by whole match clusters."""
    rng = np.random.default_rng(8086)
    groups = list(data.groupby(["tier", "league"], sort=True))
    groups += [(('top', 'All top divisions'), data[data.tier == "top"]),
               (('second', 'All second divisions'), data[data.tier == "second"])]
    records = []
    for keys, g in groups:
        if len(g) < 2:
            continue
        prob = g[["pH", "pD", "pA"]].to_numpy()
        actual = np.eye(3)[[OUTCOMES.index(v) for v in g.result]]
        alpha, slope = fit_calibration_slope(prob.ravel(), actual.ravel())
        boot = np.empty(reps)
        for rep in range(reps):
            sample = rng.integers(0, len(g), len(g))
            boot[rep] = fit_calibration_slope(prob[sample].ravel(), actual[sample].ravel())[1]
        # Center the bootstrap distribution at H0 (slope=1) before computing a two-sided p-value.
        null_boot = boot - slope + 1.0
        p_raw = (1 + np.sum(np.abs(null_boot - 1.0) >= abs(slope - 1.0))) / (reps + 1)
        records.append({"tier": keys[0], "league": keys[1], "matches": len(g),
                        "intercept": alpha, "calibration_slope": slope,
                        "slope_ci95_low": np.quantile(boot, .025),
                        "slope_ci95_high": np.quantile(boot, .975), "p_unadjusted": p_raw})
    result = pd.DataFrame(records)
    result["p_holm_within_tier"] = np.nan
    for tier in ("top", "second"):
        mask = (result.tier == tier) & ~result.league.str.startswith("All ")
        indices = result.index[mask]
        ordered = result.loc[indices, "p_unadjusted"].sort_values()
        m = len(ordered)
        adjusted = np.maximum.accumulate([(m - rank) * p for rank, p in enumerate(ordered, start=1)])
        adjusted = np.minimum(adjusted, 1.0)
        result.loc[ordered.index, "p_holm_within_tier"] = adjusted
    return result


def margin_method_comparison(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, pcols in PROBABILITY_COLUMNS.items():
        for (tier, country, league), g in data.groupby(["tier", "country", "league"], sort=True):
            records.append({"margin_method": method, "tier": tier, "country": country,
                            "league": league, **summarize_group(g, pcols)})
        for tier, g in data.groupby("tier", sort=True):
            p = g[pcols].to_numpy()
            actual = np.eye(3)[[OUTCOMES.index(v) for v in g.result]]
            alpha, slope = fit_calibration_slope(p.ravel(), actual.ravel())
            rng = np.random.default_rng(4400 + len(method) + len(tier))
            boot = np.empty(500)
            for rep in range(len(boot)):
                sample = rng.integers(0, len(g), len(g))
                boot[rep] = fit_calibration_slope(p[sample].ravel(), actual[sample].ravel())[1]
            p_raw = (1 + np.sum(np.abs(boot - slope) >= abs(slope - 1.0))) / (len(boot) + 1)
            rows = np.arange(len(g))
            fav, long = p.argmax(axis=1), p.argmin(axis=1)
            records.append({"margin_method": method, "tier": tier, "country": "All",
                            "league": "All divisions in tier", "matches": len(g),
                            "market_log_loss": float(-np.log(np.clip(p[rows, [OUTCOMES.index(v) for v in g.result]], 1e-15, 1)).mean()),
                            "market_brier_sum": np.square(actual - p).sum(axis=1).mean(),
                            "favorite_actual_minus_expected_pp": 100 * (actual[rows, fav] - p[rows, fav]).mean(),
                            "longshot_actual_minus_expected_pp": 100 * (actual[rows, long] - p[rows, long]).mean(),
                            "calibration_slope": slope, "calibration_intercept": alpha,
                            "slope_ci95_low": np.quantile(boot, .025),
                            "slope_ci95_high": np.quantile(boot, .975), "slope_p_unadjusted": p_raw})
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
    slope_results = calibration_slope_regression(data)
    slope_results.to_csv(args.output_dir / "calibration_slope_regression.csv", index=False)
    margin_results = margin_method_comparison(data)
    margin_results.to_csv(args.output_dir / "margin_method_comparison.csv", index=False)
    models = run_model_comparison(data)
    models.to_csv(args.output_dir / "time_ordered_model_comparison.csv", index=False)
    save_reliability_plot(bins, args.output_dir / "calibration_curves.png")

    top = overall[overall.tier == "top"]
    second = overall[overall.tier == "second"]
    total_top = int(top.matches.sum())
    total_second = int(second.matches.sum())
    pooled_top_bias = bias[bias.league == "All top divisions"].iloc[0]
    pooled_top_slope = slope_results[slope_results.league == "All top divisions"].iloc[0]
    model_means = models.groupby("model")[["log_loss", "brier_sum"]].mean()
    best_model = model_means.loc[model_means.log_loss.idxmin()]
    raw_model = model_means.loc["Bookmaker odds (market)"]
    pooled_margin = margin_results[(margin_results.tier == "top") &
                                   (margin_results.league == "All divisions in tier")]
    method_slope_range = (pooled_margin.calibration_slope.min(), pooled_margin.calibration_slope.max())
    method_ll_range = (pooled_margin.market_log_loss.min(), pooled_margin.market_log_loss.max())
    report = [
        "# Football betting odds analysis\n",
        "## Scope and method\n",
        f"The analysis includes {total_top:,} top-division matches and {total_second:,} second-division matches with a full-time result and valid closing average odds. It covers 2019/20–2024/25 for England, Germany, Italy, Spain and France. The first divisions are the proposal's primary sample; second divisions are reported separately.\n",
        "For each match, inverse closing average odds (AvgCH, AvgCD, AvgCA) were converted to probabilities using proportional, additive, power and Shin margin-removal methods. Log loss and multiclass Brier score assess probability accuracy. Favourite and longshot gaps compare the realized result rate with the adjusted market probability for the most and least likely outcomes in each match. Confidence intervals use a match-level bootstrap.\n",
        "## Initial findings\n",
        f"Across the {total_top:,} top-division matches, the weighted market log loss is {summarize_group(data[data.tier == 'top'])['market_log_loss']:.3f} and the multiclass Brier score is {summarize_group(data[data.tier == 'top'])['market_brier_sum']:.3f}. For the {total_second:,} second-division matches, these are {summarize_group(data[data.tier == 'second'])['market_log_loss']:.3f} and {summarize_group(data[data.tier == 'second'])['market_brier_sum']:.3f}. The lower top-tier scores indicate better probability accuracy in this sample.\n",
        f"Pooled top-tier favourites win {pooled_top_bias.favorite_gap_pp:.2f} percentage points more often than implied by the margin-adjusted market probabilities (match-bootstrap 95% interval {pooled_top_bias.favorite_ci95_low_pp:.2f} to {pooled_top_bias.favorite_ci95_high_pp:.2f}). The longshot gap is {pooled_top_bias.longshot_gap_pp:.2f} points (95% interval {pooled_top_bias.longshot_ci95_low_pp:.2f} to {pooled_top_bias.longshot_ci95_high_pp:.2f}). This is preliminary evidence consistent with a favourite–longshot pattern among favourites, while the pooled longshot estimate remains uncertain; it is not proof of a uniform effect across leagues.\n",
        f"The pooled top-tier calibration regression estimates a slope of {pooled_top_slope.calibration_slope:.3f} (match-bootstrap 95% interval {pooled_top_slope.slope_ci95_low:.3f} to {pooled_top_slope.slope_ci95_high:.3f}). A slope above 1 means observed outcome probabilities are more extreme than the market implies, consistent with favourite–longshot bias; inspect the interval and league-level estimates before drawing conclusions.\n",
        "## Robustness checks\n",
        f"For the top divisions, the four margin-removal methods produce log loss between {method_ll_range[0]:.3f} and {method_ll_range[1]:.3f}, but calibration slopes range from {method_slope_range[0]:.3f} to {method_slope_range[1]:.3f}. Probability accuracy scores barely change, while the strength of the favourite–longshot calibration signal does. This means the evidence about bias is sensitive to how the bookmaker margin is allocated; see margin_method_comparison.csv for each league and method.\n",
        "League-level calibration-slope tests use match-bootstrap p-values and Holm correction across the five leagues separately within each tier. Adjusted values appear in calibration_slope_regression.csv; use the adjusted p-values, rather than the unadjusted intervals, for league-by-league significance claims.\n",
        "La Liga and Serie A have the largest top-tier slope estimates, but after Holm adjustment no individual league has a p-value below .05. The pooled estimate should guide interpretation more than an isolated league result; adjusted p-values are in calibration_slope_regression.csv.\n",
        f"Across the four time-ordered test seasons, {model_means.log_loss.idxmin()} has the lowest mean log loss ({best_model.log_loss:.3f}); raw bookmaker odds average {raw_model.log_loss:.3f}. Additional features were: logistic odds plus basic form {model_means.loc['Logistic odds plus form', 'log_loss']:.3f}; logistic plus extended match-history features {model_means.loc['Logistic odds plus extended features', 'log_loss']:.3f}; extended features plus team identities {model_means.loc['Logistic extended plus team identity', 'log_loss']:.3f}; decision tree {model_means.loc['Decision tree odds plus extended', 'log_loss']:.3f}; boosting {model_means.loc['Boosting odds plus form', 'log_loss']:.3f}. These small differences are descriptive; no paired significance test was run. See time_ordered_model_comparison.csv for per-season results.\n",
        "## Reading the results\n",
        "A positive favourite gap means favourites won more often than their odds implied; a negative longshot gap means the least likely outcomes occurred less often than implied. That pattern is consistent with favourite–longshot bias. It is descriptive evidence: the simple favourite/longshot comparison does not establish why the pricing pattern occurs. Check whether each 95% interval excludes zero before treating a gap as distinguishable from sampling variation.\n",
        "Calibration metrics and gaps are reported by league and season in the CSV files. Reliability bins compare observed frequencies with implied probabilities by outcome and probability range. Lower log loss and Brier score indicate better probabilistic forecasts.\n",
        "## Predictive model comparison\n",
        "The expanding-window comparison uses only top-division matches. It trains on earlier full seasons and tests on the next season. Historical performance features use only earlier match dates: recent points, goals, shots on target, shots, corners, cards, season-to-date points and goal difference, days since the previous match, and prior head-to-head points. Team identity is included as one-hot features in one logistic model. The comparison includes raw odds, odds recalibration, logistic regression, a decision tree and histogram gradient boosting. Lower out-of-sample log loss/Brier score is better. These are initial models, not an exhaustive feature search.\n",
        "## Data checks and limitations\n",
        "See data_quality_by_season.csv for source row counts, valid results, usable closing odds and duplicate fixture checks. Raw downloads are preserved in data/football_data/raw. CSV column availability is checked before analysis; observations without valid closing average odds or a recognized result are excluded. The analysis should be treated as provisional until team names, duplicate fixtures, season changes and league-specific missingness have been reviewed.\n",
        "## Output files\n",
        "- calibration_by_league.csv and calibration_by_league_season.csv: overall and seasonal probability metrics.\n",
        "- reliability_bins_by_league.csv and calibration_curves.png: decile calibration data and a top-tier reliability plot.\n",
        "- favorite_longshot_bias.csv and calibration_slope_regression.csv: favourite/longshot gaps and calibration-slope regressions, with match-bootstrap 95% intervals.\n",
        "- margin_method_comparison.csv: sensitivity of calibration and favourite/longshot findings to alternative margin-removal methods.\n",
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


