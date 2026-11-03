import os
import pandas as pd
import numpy as np

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.constants import *  # This imports all constants from constants.py


def main():
    df_combined, df_ic_all, df_ic_train = sf.data_loading(DATA_DIR)

    batch_ids = np.unique(df_combined.loc[:, "Batch ID"])
    features = [col for col in df_combined.columns if col not in ["Batch ID", "Time [h]"]]

    eval_results = sf.nvar_hyperparams(
        df_combined=df_combined,
        batch_ids=batch_ids,
        features=features,
    )

    data_summary = []
    for res in eval_results:
        knn_k = res["knn_params"]["kneighborsregressor__n_neighbors"]
        knn_w = res["knn_params"]["kneighborsregressor__weights"]

        data_summary.append({
            "p": res["p"],
            "Poly RMSE": res["poly_rmse"],
            "Poly Degree": res["poly_params"]["polynomialfeatures__degree"],
            "kNN RMSE": res["knn_rmse"],
            "kNN Weights": f"k={knn_k}, {knn_w}",
        })

    df_summary = pd.DataFrame(data_summary)
    print("\n=======================================================")
    print("      NVAR Model Hyperparameter Search Summary        ")
    print("=======================================================")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join("tables", "nvar_model_hyperparameter_summary.csv"), index=False)

    best_res = min(eval_results, key=lambda r: min(r["poly_rmse"], r["knn_rmse"]))

    if best_res["knn_rmse"] < best_res["poly_rmse"]:
        win_type = "kNN-NVAR"
        win_rmse = best_res["knn_rmse"]
        win_params = f"k={best_res['knn_params']['kneighborsregressor__n_neighbors']}, weight = {best_res['knn_params']['kneighborsregressor__weights']}"
        winning_model = best_res["grid_knn"].best_estimator_
        winning_residuals = best_res["res_knn"]
    else:
        win_type = "Polynomial-NVAR"
        win_rmse = best_res["poly_rmse"]
        win_params = f"degree={best_res['poly_params']['polynomialfeatures__degree']}"
        winning_model = best_res["grid_poly"].best_estimator_
        winning_residuals = best_res["res_poly"]

    print("\n==========================================================")
    print(f"OVERALL WINNING MODEL: {win_type}")
    print(f"* Lag Order(p): {best_res['p']}")
    print(f"* Parameters: {win_params}")
    print(f"* Validation RMSE: {win_rmse:.4f}")

    df_optimal = sf.space_time_profit(
        best_model=winning_model,
        df_ic_all=df_ic_all,
        features=features,
        p=best_res["p"],
        res_residuals=winning_residuals,
    )

    print("=== Space-Time Profit Optimization Results ===")
    print(df_optimal.to_string(index=False))

    df_optimal.to_csv(os.path.join("tables", "optimal_mAb_production_parameters.csv"), index=False)


if __name__ == "__main__":
    main()
