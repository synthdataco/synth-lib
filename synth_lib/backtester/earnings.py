"""Reward-weight -> USD attribution for the earnings charts."""

from __future__ import annotations

import warnings

import pandas as pd

from synth.validator.competition_config import SMOOTHED_SCORE_COEFFICIENT

from synth_lib.backtester.config import _COMBINED_EMPTY_COLS, EMISSION_NORMALIZATION_FACTOR



def _compute_grand_total_weights(
    combined_by_profile: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-profile combined frames into one grand-total reward_weight frame.

    At each union timestamp, forward-fill each profile's per-miner reward_weight
    from its most recent round and sum across profiles. Drops timestamps before
    all profiles have produced their first round (otherwise the grand-total
    weights wouldn't sum to 1.0 across miners).

    Returns DataFrame with columns: updated_at, miner_uid, reward_weight,
    new_smoothed_score. Empty if any profile frame is empty or the input dict is
    empty (single-profile input also returns empty, since grand-total requires
    all profiles active).
    """
    if not combined_by_profile or any(df.empty for df in combined_by_profile.values()):
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if len(combined_by_profile) < 2:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Union of timestamps and miners across profiles
    all_miners: set[int] = set()
    all_timestamps: set[pd.Timestamp] = set()
    profile_starts: dict[str, pd.Timestamp] = {}
    for label, df in combined_by_profile.items():
        df_ts = pd.to_datetime(df["updated_at"])
        all_timestamps.update(df_ts.unique())
        all_miners.update(df["miner_uid"].unique())
        profile_starts[label] = df_ts.min()

    # Drop timestamps before ALL profiles have started
    latest_start = max(profile_starts.values())
    union_ts = sorted(t for t in all_timestamps if t >= latest_start)
    if not union_ts:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Forward-fill each profile's reward_weight onto the union grid per miner.
    # Reindex onto the union of ALL profile timestamps first (including those
    # before latest_start), so ffill can carry values forward from earlier rounds
    # of a profile whose first round pre-dates the other profile's first round.
    # Then restrict to union_ts.
    miners = sorted(all_miners)
    full_index = sorted(all_timestamps)
    ff_frames: list[pd.DataFrame] = []
    for _label, df in combined_by_profile.items():
        df_local = df.copy()
        df_local["updated_at"] = pd.to_datetime(df_local["updated_at"])
        pivot = (
            df_local.pivot_table(
                index="updated_at",
                columns="miner_uid",
                values="reward_weight",
                aggfunc="last",
            )
            .reindex(full_index)
            .ffill()
            .reindex(index=union_ts)
            .reindex(columns=miners, fill_value=0.0)
        )
        pivot = pivot.fillna(0.0)
        ff_frames.append(pivot)

    # Sum forward-filled weights across profiles
    total = sum(ff_frames)
    long = total.reset_index().melt(
        id_vars="updated_at", var_name="miner_uid", value_name="reward_weight"
    )
    long["new_smoothed_score"] = (
        0.0  # grand-total doesn't have a meaningful smoothed score
    )
    return (
        long[_COMBINED_EMPTY_COLS]
        .sort_values(["updated_at", "miner_uid"])
        .reset_index(drop=True)
    )


def _compute_earnings_df(
    combined: pd.DataFrame,
    miner_id: int,
    daily_pool_usd: pd.Series,
    *,
    share_denominator: float = SMOOTHED_SCORE_COEFFICIENT,
    emission_factor: float = EMISSION_NORMALIZATION_FACTOR,
) -> pd.DataFrame:
    """Compute per-round estimated USD earnings for a single miner.

    Formula per round:
      share_of_round = reward_weight / share_denominator
      usd_per_round  = reward_weight * daily_pool_usd[date] / rounds_per_day[date]
                       * emission_factor

    `reward_weight` is already the miner's share of the FULL on-chain weight
    attributable to this competition: synth.validator.moving_average.combine_moving_averages
    sums per-competition reward_weights, and each competition's softmax
    contributes up to SMOOTHED_SCORE_COEFFICIENT (1/3), the three summing to
    1.0 on-chain. So multiplying reward_weight by the subnet-wide daily pool
    yields the competition's realistic USD contribution — summing the three
    competition charts gives the miner's total expected daily earnings.

    Per-competition callers divide share_of_round by 1/3 (the default); the
    grand-total caller passes share_denominator=1.0 because grand-total
    reward_weight already sums to ~1.0. `emission_factor` scales USD to account
    for the validator's appended owner miner (see EMISSION_NORMALIZATION_FACTOR).

    `share_of_round` is kept as a display-friendly intermediate ("share of this
    competition's pool") used by the chart's top panel.

    Rounds on dates missing from `daily_pool_usd` (API gap / recent day not yet
    populated) are dropped with a stdout warning. The returned totals reflect
    only days with pool data.

    Returns DataFrame (sorted by updated_at) with columns:
      updated_at, miner_uid, reward_weight, share_of_round, date,
      usd_per_round, usd_cumulative.
    Empty if the miner is absent from `combined` or every round lacks pool data.
    """
    our = (
        combined.loc[combined["miner_uid"] == miner_id].sort_values("updated_at").copy()
    )
    if our.empty:
        return our

    our["updated_at"] = pd.to_datetime(our["updated_at"])
    our["share_of_round"] = our["reward_weight"] / share_denominator
    our["date"] = our["updated_at"].dt.floor("D")

    missing_mask = ~our["date"].isin(daily_pool_usd.index)
    if missing_mask.any():
        missing_dates = sorted(
            {d.date().isoformat() for d in our.loc[missing_mask, "date"]}
        )
        print(
            f"  Warning: dropping {int(missing_mask.sum())} round(s) on "
            f"{len(missing_dates)} day(s) with no /v1/miners/rewards/pool data: "
            f"{missing_dates}"
        )
        our = our.loc[~missing_mask].copy()
        if our.empty:
            return our

    rounds_per_day = our.groupby("date").size()
    our["usd_per_round"] = (
        our["reward_weight"]
        * our["date"].map(daily_pool_usd)
        / our["date"].map(rounds_per_day)
        * emission_factor
    )
    our["usd_cumulative"] = our["usd_per_round"].cumsum()
    return our
