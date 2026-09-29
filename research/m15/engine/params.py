from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Params:
    setups: tuple[str, ...] = ("A", "B", "C")
    directions: tuple[str, ...] = ("long", "short")

    # Setup A — pullback trendline break
    a_lookback: int = 60
    a_min_swings: int = 3
    a_min_r2: float = 0.80
    a_max_slope_atr: float = -0.03
    a_touch_tol_atr: float = 0.25
    a_break_atr: float = 0.20
    a_stop_buffer_atr: float = 0.20

    # Setup B — horizontal level break
    b_m15_lookback: int = 200
    b_h1_lookback: int = 100
    b_cluster_atr: float = 0.30
    b_min_touches: int = 2
    b_max_age: int = 100
    b_break_atr: float = 0.25
    b_min_body_atr: float = 0.60
    b_limit_offset_atr: float = 0.10
    b_limit_bars: int = 6
    b_entry: str = "limit"  # "limit" | "market"

    # Setup C — compression breakout
    c_lookback: int = 40
    c_min_span: int = 15
    c_min_touches: int = 5
    c_min_r2: float = 0.75
    c_min_height_atr: float = 1.5
    c_break_atr: float = 0.20
    c_target_cap_atr: float = 4.0

    # Candle quality
    min_body_frac: float = 0.50
    min_close_pos: float = 0.70
    max_range_atr: float = 2.5
    max_extension_atr: float = 1.5

    # Stops
    min_stop_atr: float = 0.8
    max_stop_atr: float = 2.0

    # Filters
    use_h1_bias: bool = True
    use_session: bool = True
    use_news: bool = True
    max_spread_frac: float = 0.10
    vol_min: float = 0.70
    vol_high: float = 1.80
    min_room_r: float = 1.5
    min_score: float = 60.0
    level_range_atr: float = 3.0

    # Targets & management
    tp1_r: float = 1.0
    tp1_frac: float = 0.5
    tp2_max_r: float = 3.0
    be_buffer_atr: float = 0.05
    trail_atr: float = 2.5
    failed_break_bars: int = 3
    failed_break_atr: float = 0.20
    time_stop_bars: int = 12
    time_stop_r: float = 0.5
    max_hold_bars: int = 96

    # Account
    risk_pct: float = 0.75
    max_open: int = 2
    cooldown_bars: int = 3
    zone_lock_atr: float = 1.0
    zone_lock_bars: int = 16
    daily_loss_r: float = -2.0
    max_consec_losses_day: int = 3
    weekly_loss_r: float = -5.0

    # Costs (backtest)
    slippage_atr: float = 0.05

    extra: dict[str, Any] = field(default_factory=dict)

    def with_(self, **kw: Any) -> "Params":
        d = asdict(self)
        d.update(kw)
        return Params(**d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        names = {f.name for f in fields(cls)}
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**clean)
