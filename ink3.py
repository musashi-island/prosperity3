from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json
import math
import statistics

# Logger
logger = None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[str, List[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for trades in state.own_trades.values() for t in trades],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for trades in state.market_trades.values() for t in trades],
            state.position,
            [
                state.observations.plainValueObservations,
                {p: [co.bidPrice, co.askPrice, co.transportFees, co.exportTariff,
                     co.importTariff, co.sugarPrice, co.sunlightIndex]
                 for p, co in state.observations.conversionObservations.items()}
            ]
        ]

    def compress_orders(self, orders: Dict[str, List[Order]]) -> list[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for orders_list in orders.values() for o in orders_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# Constants (if you want to use these globally, though the Trader class now uses its own thresholds)
VOUCHERS = {}

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK" 

SPREAD_HISTORY_LIMIT = 100
MIN_HOLD_TICKS = 10
POSITION_LIMIT = 50

class Trader:
    def __init__(self):
        self.squid_mid_history = []
        self.max_history = 50
        self.order_qty = 10  # Default order quantity
        self.LIMIT = {
            Product.SQUID_INK: 50,
            Product.KELP: 50
        }
        self.spread_position = 0  # 1 for long, -1 for short
        # New thresholds for zscore-based strategy:
        self.z_entry_threshold = 1.5   # Lower threshold for initiating trades
        self.z_exit_threshold = 0.5    # Exit when zscore reverts close to zero

    def get_swmid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0  # Not used here.
        traderData = ""  # No persistence needed for now.

        squid_depth = state.order_depths.get(Product.SQUID_INK)
        if squid_depth:
            squid_mid = self.get_swmid(squid_depth)
            if squid_mid is None:
                logger.print("No mid price available for SQUID_INK")
                logger.flush(state, result, conversions, traderData)
                return result, conversions, traderData

            # Append current mid price and maintain history (window size = self.max_history)
            self.squid_mid_history.append(squid_mid)
            if len(self.squid_mid_history) > self.max_history:
                self.squid_mid_history.pop(0)

            # Compute Bollinger Bands statistics if we have enough data; otherwise default to current price.
            if len(self.squid_mid_history) >= 20:
                rolling_mean = np.mean(self.squid_mid_history)
                rolling_std = np.std(self.squid_mid_history)
                upper_band = rolling_mean + 2 * rolling_std
                lower_band = rolling_mean - 2 * rolling_std
            else:
                rolling_mean = squid_mid
                rolling_std = 0.0
                upper_band = squid_mid
                lower_band = squid_mid

            current_pos = state.position.get(Product.SQUID_INK, 0)
            logger.print(
                f"SQUID_INK: mid={squid_mid:.2f}, mean={rolling_mean:.2f}, upper={upper_band:.2f}, lower={lower_band:.2f}, pos={current_pos}"
            )

            squid_orders: List[Order] = []
            remaining_capacity = self.LIMIT[Product.SQUID_INK] - abs(current_pos)

            # ----- Entry Conditions when flat -----
            if current_pos == 0 and remaining_capacity > 0:
                # If price is overextended high (above upper band), expect a downward reversion → go short.
                if squid_mid > upper_band and squid_depth.buy_orders:
                    best_bid = max(squid_depth.buy_orders.keys())
                    qty = min(self.order_qty, remaining_capacity)
                    squid_orders.append(Order(Product.SQUID_INK, best_bid, -qty))
                    logger.print(f"SHORT ENTRY: {-qty} @ {best_bid}")
                # If price is overextended low (below lower band), expect an upward reversion → go long.
                elif squid_mid < lower_band and squid_depth.sell_orders:
                    best_ask = min(squid_depth.sell_orders.keys())
                    qty = min(self.order_qty, remaining_capacity)
                    squid_orders.append(Order(Product.SQUID_INK, best_ask, qty))
                    logger.print(f"LONG ENTRY: {qty} @ {best_ask}")

            # ----- Exit Conditions when in a position -----
            else:
                # For a long position: exit when price reverts to or above the rolling mean.
                if current_pos > 0 and squid_mid >= rolling_mean and squid_depth.buy_orders:
                    best_bid = max(squid_depth.buy_orders.keys())
                    squid_orders.append(Order(Product.SQUID_INK, best_bid, -current_pos))
                    logger.print(f"LONG EXIT: {-current_pos} @ {best_bid}")
                # For a short position: exit when price reverts to or below the rolling mean.
                elif current_pos < 0 and squid_mid <= rolling_mean and squid_depth.sell_orders:
                    best_ask = min(squid_depth.sell_orders.keys())
                    squid_orders.append(Order(Product.SQUID_INK, best_ask, -current_pos))
                    logger.print(f"SHORT EXIT: {-current_pos} @ {best_ask}")

            if squid_orders:
                result[Product.SQUID_INK] = squid_orders

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
