"""In-memory MinerDataHandler so the real validator smoothing code runs without a DB."""

from __future__ import annotations

from sqlalchemy import create_engine

from synth.validator.miner_data_handler import MinerDataHandler



class _BacktestMinerDataHandler(MinerDataHandler):
    """MinerDataHandler that maps miner_uid == miner_id without a real DB."""

    def __init__(self) -> None:
        # Pass a dummy SQLite engine to avoid connecting to PostgreSQL
        self.engine = create_engine("sqlite://")

    def populate_miner_uid_in_miner_data(self, miner_data: list[dict]) -> list[dict]:
        for row in miner_data:
            row["miner_uid"] = row["miner_id"]
        return miner_data


_BACKTEST_MDH = _BacktestMinerDataHandler()
