import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
"""
Create exponentially decaying weights based on sample position.
More recent samples (higher indices) get higher weights.
decay_factor controls the rate of decay (0.95 = 5% decay per time unit)
"""
def create_time_weights(n_samples, decay_factor = 0.95):
    positions = np.arange(n_samples)
    normalized_positions = positions / (n_samples - 1)
    weights = decay_factor ** (1 - normalized_positions)
    weights = weights * n_samples / weights.sum()

train = pd.read_parquet()
test = pd.read_parquet()
sample = pd.read_csv()
selected_features = [
    "X344", "X598", "X863", "X862", "X856", "X137", "X174", "X425", "X612", "X167",
    "X852", "X168", "X27", "X422", "X342", "X427", "X532", "X178", "X539", "X881",
    "X889", "X421", "X341", "X875", "X465", "X97", "X603", "X138", "X855", "X572",
    "X338", "X890", "X95", "X161", "X533", "X271", "X861", "X279", "X424", "X888",
    "X866", "X169", "X879", "X283", "X332", "X854", "X574", "X28", "X281", "X757",
    "X754", "X445", "X180", "X94", "X88", "X525", "X285", "X181", "X429", "X343",
    "X688", "X692", "X680", "X832", "X755", "X860", "X695", "X345", "X611", "X689",
    "X387", "X588", "X686", "X140", "X530", "X878", "X753", "X98", "X24", "X880",
    "X756", "X540", "X531", "X340", "X383", "X331", "X873", "X385", "X277", "X602",
    "X136", "X586", "X786", "X887", "X300", "X284", "X91", "X379", "X685", "X177",
    'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'bid_ask_interaction',
    'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction', 'ask_sell_interaction', 'buy_sell_interaction',
    'spread_indicator',
    'volume_weighted_buy', 'volume_weighted_sell', 'volume_weighted_bid', 'volume_weighted_ask',
    'buy_sell_ratio', 'bid_ask_ratio',
    'order_flow_imbalance',
    'buying_pressure', 'selling_pressure',
    'total_liquidity', 'liquidity_imbalance', 'relative_spread',
    'trade_intensity', 'avg_trade_size', 'net_trade_flow',
    'depth_ratio', 'volume_participation', 'market_activity',
    'effective_spread_proxy', 'realized_volatility_proxy',
    'normalized_buy_volume', 'normalized_sell_volume',
    'liquidity_adjusted_imbalance', 'pressure_spread_interaction', 
    'trade_direction_ratio', 'net_buy_volume', 'bid_skew' , 'ask_skew'
]




train = train[selected_features + ["label"]]
test = test[selected_features]
RMV = ["label"]
FEATURES = [c for c in train.columns if c not in RMV]

FOLDS = 5
kf = KFold(n_splits = FOLDS, shuffle = True, random_state = 42)

xgb_params = {
    "tree_method": "gpu_hist",
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5625888953382505,
    "learning_rate": 0.029312951475451557,
    "min_child_samples": 63,
    "min_child_weight": 0.11456572852335424,
    "n_estimators": 126,
    "n_jobs": -1,
    "num_leaves": 37,
    "random_state": 42,
    "reg_alpha": 85.2476527854083,
    "reg_lambda": 99.38305361388907,
    "subsample": 0.450669817684892,
    "verbose": -1
}
#Create massive for predictions
oof_preds_model1 = np.zeros(len(train))
test_preds_model1 = np.zeros(len(test))
oof_preds_model2 = np.zeros(len(train))
test_preds_model2 = np.zeros(len(test))
oof_preds_model3 = np.zeros(len(train))  
test_preds_model3 = np.zeros(len(test))   
oof_preds_model3_lb = np.zeros(len(train))
test_preds_model3_lb = np.zeros(len(test))

sample_weights_full = create_time_weights(len(train), decay_factor = 0.95)
print(f"\nModel 1 - Full data sample weights range: [{sample_weights_full.min():.4f}, {sample_weights_full.max():.4f}]")
print(f"Model 1 - Full data sample weights mean: {sample_weights_full.mean():.4f}")

cutoff_idx_75 = int(len(train) * 0.25)
print(f"\nModel 2 - Using most recent {len(train) - cutoff_idx_75} samples (75% of data)")
cutoff_idx_50 = int(len(train) * 0.50)
print(f"\nModel 3 - Using most recent {len(train) - cutoff_idx_50} samples (50% of data)")

for i, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"Fold {i + 1}")
    print("\n--- Model 1: Full Data with Time Weights ---")

    X_train_m1 = train.iloc[train_idx][FEATURES]
    y_train_m1 = train.iloc[train_idx]['label']
    X_val = train.iloc[val_idx][FEATURES]
    y_val = train.iloc[val_idx]['label']
    X_test = test.iloc[FEATURES]

    train_weights_m1 = sample_weights_full[train_idx]
    model1 = XGBRegressor(**xgb_params)
    model1.fit(
        X_train_m1, y_train_m1,
        sample_weights = train_weights_m1,
        eval_set = [(X_val, y_val)],
        early_stopping_rounds = 25,
        verbose = 200
    )

    oof_preds_model1[val_idx] = model1.predict(X_val)
    test_preds_model1 += model1.predict(X_test)

    print("\n--- Model 2: 75% Most Recent Data ---")

    train_idx_recent_75 = train_idx[train_idx >= cutoff_idx_75]
    train_idx_recent_adjusted_75 = train_idx_recent_75 - cutoff_idx_75

    train_recent_75 = train.iloc[cutoff_idx_75:].reset_index(drop=True)
    
    X_train_m2 = train_recent_75.iloc[train_idx_recent_adjusted_75][FEATURES]
    y_train_m2 = train_recent_75.iloc[train_idx_recent_adjusted_75]["label"]

    sample_weights_recent_75 = create_time_weights(len(train_recent_75), decay_factor=0.95)
    train_weights_m2 = sample_weights_recent_75[train_idx_recent_adjusted_75]
    
    model2 = XGBRegressor(**xgb_params)
    model2.fit(
        X_train_m2, y_train_m2,
        sample_weight=train_weights_m2,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=25,
        verbose=200
    )

    valid_idx_in_range_75 = val_idx[val_idx >= cutoff_idx_75]
    if len(valid_idx_in_range_75) > 0: # check length
        X_valid_m2 = train.iloc[valid_idx_in_range_75][FEATURES]
        oof_preds_model2[valid_idx_in_range_75] = model2.predict(X_valid_m2)
    
    valid_idx_out_range_75 = val_idx[val_idx < cutoff_idx_75]
    if len(valid_idx_out_range_75) > 0:
        oof_preds_model2[valid_idx_out_range_75] = oof_preds_model1[valid_idx_out_range_75]
    
    test_preds_model2 += model2.predict(X_test)