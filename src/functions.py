import os
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from matplotlib.ticker import MultipleLocator
from scipy import stats
from matplotlib import pyplot as plt

from classes import StandardScalerVAR
from constants import CV_SPLITS, POLY_DEGREE_MAX, KNN_K_MAX, P_MAX, ALPHA, PREDICTION_INTERVAL, COST_GLUCOSE, \
    COST_GLUTAMINE, COST_VCD, CULTURE_VALUE, DOWNTIME_DAYS


def data_loading(data_dir):
    ic_path = os.path.join(data_dir, "initial_conditions.csv")
    df_ic_all = pd.read_csv(ic_path)

    file_pattern = os.path.join(data_dir, "batch_*.csv")
    file_list = sorted(glob.glob(file_pattern))

    df_list = []
    loaded_batch_names = []

    for filepath in file_list:
        df_batch = pd.read_csv(filepath)
        batch_name = os.path.basename(filepath)

        if "Batch ID" not in df_batch.columns:
            df_batch["Batch ID"] = batch_name

        df_list.append(df_batch)
        loaded_batch_names.append(batch_name)

    df_combined = pd.concat(df_list, ignore_index=True)

    df_ic_train = df_ic_all[df_ic_all["Batch ID"].isin(loaded_batch_names)].reset_index(drop=True)

    return df_combined, df_ic_all, df_ic_train


def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    for batch_id in batch_ids:
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(x_batch[p - i:-i])

        lagged_values = np.concatenate(lagged_values, axis=1)

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur


def nvar_models(x_lag, x_cur, b_cur, p):
    gkf = GroupKFold(n_splits=CV_SPLITS)

    poly_pipeline = make_pipeline(
        StandardScalerVAR(p=p),
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )

    param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, POLY_DEGREE_MAX + 1)])

    grid_poly = GridSearchCV(
        estimator=poly_pipeline, param_grid=param_grid_poly, cv=gkf, scoring="neg_root_mean_squared_error",
        return_train_score=True
    )
    grid_poly.fit(x_lag, x_cur, groups=b_cur)

    knn_pipeline = make_pipeline(
        StandardScalerVAR(p=p),
        KNeighborsRegressor()
    )

    param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, KNN_K_MAX + 1)],
                          kneighborsregressor__weights=["uniform", "distance"])

    grid_knn = GridSearchCV(
        estimator=knn_pipeline, param_grid=param_grid_knn, cv=gkf, scoring="neg_root_mean_squared_error",
        return_train_score=True
    )
    grid_knn.fit(x_lag, x_cur, groups=b_cur)

    return grid_poly, grid_knn


def calculate_and_plot_residuals(
        grid_model, x_lag, x_cur, features, feature_plot, t_cur, df, model_name, p
):

    x_left = np.min(df.loc[:, "Time [h]"]) - 1
    x_right = np.max(df.loc[:, "Time [h]"]) + 1

    x_cur_pred = grid_model.predict(x_lag)
    x_cur_res = x_cur_pred - x_cur
    idx_fp = features.index(feature_plot)

    res_mean = []
    res_pi_upper = []
    res_pi_lower = []

    t_unique = np.unique(t_cur)

    for t in t_unique:
        mask = t_cur == t
        x_cur_res_t = x_cur_res[mask, idx_fp]

        n_mean = np.mean(x_cur_res_t)
        n_sd = np.std(x_cur_res_t)

        q = ALPHA / 2
        dof = x_cur_res_t.shape[0] - 1
        pi_sd = stats.t.isf(q=q, df=dof)

        res_mean.append(n_mean)
        res_pi_upper.append(n_mean + pi_sd * n_sd)
        res_pi_lower.append(n_mean - pi_sd * n_sd)

    res_mean = np.array(res_mean)
    res_pi_upper = np.array(res_pi_upper)
    res_pi_lower = np.array(res_pi_lower)

    y_max = max(np.max(np.abs(res_mean)),
                np.max(np.abs(res_pi_upper)),
                np.max(np.abs(res_pi_lower)),
                np.max(np.abs(x_cur_res[:, idx_fp])))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300, layout="constrained")

    ax.scatter(t_cur, x_cur_res[:, idx_fp], label="Noise", color="tab:pink", s=12,
               edgecolors="black", linewidths=0.5, alpha=0.7, zorder=3)

    ax.plot(t_unique, res_mean, label="Mean", color="black", zorder=4)
    ax.fill_between(t_unique, res_pi_lower, res_pi_upper, label=f"{PREDICTION_INTERVAL:.0%} PI",
                    color="tab:blue", edgecolor="none", zorder=2, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel("Time [h]", fontsize=14)
    ax.set_ylabel(f"{feature_plot} (Residuals)", fontsize=14)
    ax.legend(loc="upper left", fontsize=12)

    ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(bottom=-1.05 * y_max, top=1.05 * y_max)

    fig.savefig(os.path.join("figures", f"residuals_{model_name}_p{p:02d}.png"))
    plt.close(fig)

    return x_cur_res


def nvar_hyperparams(df_combined, batch_ids, features):
    results = []

    for p in range(1, P_MAX + 1):
        x_lag, x_cur, t_cur, b_cur = generate_x_lag_cur_var(p=p, batch_ids=batch_ids, df=df_combined, features=features)

        grid_poly, grid_knn = nvar_models(x_lag=x_lag, x_cur=x_cur, b_cur=b_cur, p=p)

        res_poly = calculate_and_plot_residuals(grid_model=grid_poly, x_lag=x_lag, x_cur=x_cur,
                                                features=features, feature_plot="mAb [mg/L]",
                                                t_cur=t_cur, df=df_combined, model_name="poly", p=p)

        res_knn = calculate_and_plot_residuals(grid_model=grid_knn, x_lag=x_lag, x_cur=x_cur,
                                               features=features, feature_plot="mAb [mg/L]",
                                               t_cur=t_cur, df=df_combined, model_name="knn", p=p)

        results.append(
            {
                "p": p,
                "poly_rmse": -grid_poly.best_score_,
                "poly_params": grid_poly.best_params_,
                "knn_rmse": -grid_knn.best_score_,
                "knn_params": grid_knn.best_params_,
                "grid_poly": grid_poly,
                "grid_knn": grid_knn,
                "res_poly": res_poly,
                "res_knn": res_knn,
            }
        )

    return results


def space_time_profit(best_model, df_ic_all, features, p, res_residuals):
    rng = np.random.default_rng(0)
    n_simulations = 1000

    best_space_time_profit = -np.inf
    optimization_results = {}

    idx_g = features.index("Glucose [mM]")
    idx_a = features.index("Glutamine [mM]")
    idx_vcd = features.index("VCD [cells/mL]")
    idx_mab = features.index("mAb [mg/L]")

    candidate_records = df_ic_all.to_dict("records")

    for row in candidate_records:
        batch_id = row["Batch ID"]
        g0 = float(row["Glucose [mM]"])
        a0 = float(row["Glutamine [mM]"])
        v0 = float(row["VCD [cells/mL]"])

        culture_cost = COST_GLUCOSE * g0 + COST_GLUTAMINE * a0 + COST_VCD * v0

        # Initial state vector at t = 0
        x_init = np.zeros(len(features))
        x_init[idx_g] = g0
        x_init[idx_a] = a0
        x_init[idx_vcd] = v0

        if "TCD [cells/mL]" in features:
            x_init[features.index("TCD [cells/mL]")] = v0

        x_lag_current = np.tile(x_init, p).reshape(1, -1)

        # 30x8 = 240
        t_max_steps = 30
        time_hours = 0.0

        for step in range(1, t_max_steps + 1):
            time_hours += 8.0
            time_days = time_hours / 24.0

            x_pred = best_model.predict(x_lag_current)[0]
            mab_t = max(0.0, float(x_pred[idx_mab]))

            val_t = CULTURE_VALUE * mab_t
            profit_t = val_t - culture_cost
            stp_t = profit_t / (time_days + DOWNTIME_DAYS)

            if stp_t > best_space_time_profit:
                best_space_time_profit = stp_t
                optimization_results = {
                    "Batch ID": batch_id,
                    "Glucose [mM]": g0,
                    "Glutamine [mM]": a0,
                    "VCD [mM]": v0,
                    "Harvest Time [h]": time_hours,
                    "Harvest Time [d]": time_days,
                    "Harvest Step": step,
                    "Cost [$/L]": culture_cost,
                }

            n_feat = len(features)
            x_lag_current = np.hstack(
                [x_lag_current[:, n_feat:], x_pred.reshape(1, -1)]
            )
    g0 = optimization_results["Glucose [mM]"]
    a0 = optimization_results["Glutamine [mM]"]
    v0 = optimization_results["VCD [mM]"]
    target_step = optimization_results["Harvest Step"]
    culture_cost = optimization_results["Cost [$/L]"]
    harvest_days = optimization_results["Harvest Time [d]"]

    mab_forecast = []
    val_forecast = []
    stp_forecast = []
    sty_forecast = []

    for simulation_id in range(n_simulations):
        x_init = np.zeros(len(features))
        x_init[idx_g] = g0
        x_init[idx_a] = a0
        x_init[idx_vcd] = v0

        if "TCD [cells/mL]" in features:
            x_init[features.index("TCD [cells/mL]")] = v0

        x_lag_current = np.tile(x_init, p).reshape(1, -1)

        for step in range(1, target_step + 1):
            noise_idx = rng.choice(res_residuals.shape[0])
            white_noise = res_residuals[noise_idx]

            x_pred = best_model.predict(x_lag_current)[0] + white_noise
            x_pred = np.maximum(0.0, x_pred)

            n_feat = len(features)
            x_lag_current = np.hstack([x_lag_current[:, n_feat:], x_pred.reshape(1, -1)])

        mab_final = float(x_pred[idx_mab])
        val_final = CULTURE_VALUE * mab_final
        profit_final = val_final - culture_cost
        stp_final = profit_final / (harvest_days + DOWNTIME_DAYS)
        sty_final = mab_final / (harvest_days + DOWNTIME_DAYS)

        mab_forecast.append(mab_final)
        val_forecast.append(val_final)
        stp_forecast.append(stp_final)
        sty_forecast.append(sty_final)

    q_low = ALPHA / 2.0
    q_high = 1.0 - (ALPHA / 2.0)

    results = {
        "Batch ID": optimization_results["Batch ID"],
        "Optimal Initial Glucose [mM]": g0,
        "Optimal Initial Glutamine [mM]": a0,
        "Optimal Initial VCD [cells/mL]": v0,
        "Recommended Harvest Time [h]": optimization_results["Harvest Time [h]"],
        "Harvest Time [d]": harvest_days,
        "Cell Culture Cost [$/L]": culture_cost,
        # mAb Concentration [mg/L]
        "mAb Median": np.median(mab_forecast),
        "mAb 95% PI Lower": np.quantile(mab_forecast, q_low),
        "mAb 95% PI Upper": np.quantile(mab_forecast, q_high),
        # Culture Value [$/L]
        "Value Median": np.median(val_forecast),
        "Value 95% PI Lower": np.quantile(val_forecast, q_low),
        "Value 95% PI Upper": np.quantile(val_forecast, q_high),
        # Space-Time Profit [$/(L*d)]
        "STP Median": np.median(stp_forecast),
        "STP 95% PI Lower": np.quantile(stp_forecast, q_low),
        "STP 95% PI Upper": np.quantile(stp_forecast, q_high),
        # Space-Time Yield [mg/(L*d)]
        "STY Median": np.median(sty_forecast),
        "STY 95% PI Lower": np.quantile(sty_forecast, q_low),
        "STY 95% PI Upper": np.quantile(sty_forecast, q_high),
    }

    return pd.DataFrame([results])
