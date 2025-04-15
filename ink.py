from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json

from typing import List, Dict, Tuple, Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Dict, List, Tuple, Any
from json import JSONEncoder
import jsonpickle
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

# Constants
VOUCHERS = {
    #"VOLCANIC_ROCK_VOUCHER_9500": 9500,
    #"VOLCANIC_ROCK_VOUCHER_9750": 9750,
    #"VOLCANIC_ROCK_VOUCHER_10000": 10000,
    #"VOLCANIC_ROCK_VOUCHER_10250": 10250,
    #"VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

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
Z_ENTRY = 3.5
Z_EXIT = 0.5
MIN_HOLD_TICKS = 10
POSITION_LIMIT = 50

class Trader:
    def __init__(self):
        self.squid_mid_history = []
        self.max_history = 50
        self.z_change_entry = 0.3
        self.z_score_data = []
        self.zscore_diff_history = []
        self.z_exit = 0.7
        self.min_hold_ticks = 10
        self.order_qty = 10  # Define default order quantity
        self.LIMIT = {
            Product.SQUID_INK: 50,
            Product.KELP: 50
        }

        # Track position in spread trade
        self.spread_position = 0  # 1 for long INK, -1 for short INK
        self.holding_ticks = 0

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
        conversions = 0  # Not used here
        traderData = ""  # No persistence needed for testing

        squid_depth = state.order_depths.get(Product.SQUID_INK)

        if squid_depth:
            squid_mid = self.get_swmid(squid_depth)
            
            # Skip if no mid price available
            if squid_mid is None:
                logger.print("No mid price available for SQUID_INK")
                logger.flush(state, result, conversions, traderData)
                return result, conversions, traderData
                
            self.squid_mid_history.append(squid_mid)
            
            # Limit history length
            if len(self.squid_mid_history) > self.max_history:
                self.squid_mid_history.pop(0)

            # Calculate z-score if we have enough data
            zscore = 0
            avg_zscore_diff = 0
            
            if len(self.squid_mid_history) >= self.max_history:
                mean = np.mean(self.squid_mid_history)
                std = np.std(self.squid_mid_history)
                
                # Avoid division by zero
                if std > 0:
                    zscore = (squid_mid - mean) / std
                    
                    # Calculate z-score difference
                    if self.z_score_data:
                        z_diff = zscore - self.z_score_data[-1]
                        self.zscore_diff_history.append(z_diff)
                        
                        # Keep history limited
                        if len(self.zscore_diff_history) > 2:
                            self.zscore_diff_history.pop(0)

                        # Calculate average z-score difference
                        if len(self.zscore_diff_history) == 2:
                            avg_zscore_diff = sum(self.zscore_diff_history) / 2
                        else:
                            avg_zscore_diff = z_diff
                    else:
                        avg_zscore_diff = 0
                        
            # Always append the current z-score to history
            self.z_score_data.append(zscore)
            
            # Keep z-score data history limited
            if len(self.z_score_data) > self.max_history:
                self.z_score_data.pop(0)
            
            # Log key metrics for debugging
            logger.print(f"SQUID_INK: mid={squid_mid:.2f}, zscore={zscore:.2f}, avg_diff={avg_zscore_diff:.2f}, " +
                         f"pos={state.position.get(Product.SQUID_INK, 0)}, spread_pos={self.spread_position}, " +
                         f"holding_ticks={self.holding_ticks}")
            
            # Trading logic begins here
            squid_orders: List[Order] = []
            squid_pos = state.position.get(Product.SQUID_INK, 0)

            if self.spread_position == 0 and avg_zscore_diff != 0:
                # ENTRY conditions
                if avg_zscore_diff > self.z_change_entry and squid_pos > -self.LIMIT[Product.SQUID_INK]:
                    # Short entry (positive z-score change indicates price moving up, we want to sell)
                    best_bid = max(squid_depth.buy_orders.keys())
                    squid_orders.append(Order(Product.SQUID_INK, best_bid, -self.order_qty))
                    self.spread_position = -1
                    self.holding_ticks = 0
                    logger.print(f"SHORT ENTRY: {-self.order_qty} @ {best_bid}")
                    
                elif avg_zscore_diff < -self.z_change_entry and squid_pos < self.LIMIT[Product.SQUID_INK]:
                    # Long entry (negative z-score change indicates price moving down, we want to buy)
                    best_ask = min(squid_depth.sell_orders.keys())
                    squid_orders.append(Order(Product.SQUID_INK, best_ask, self.order_qty))
                    self.spread_position = 1
                    self.holding_ticks = 0
                    logger.print(f"LONG ENTRY: {self.order_qty} @ {best_ask}")
            else:
                # EXIT conditions (based on holding time)
                self.holding_ticks += 1
                if self.holding_ticks >= self.min_hold_ticks:
                    if self.spread_position == 1 and squid_pos > 0:
                        # Exit long position
                        best_bid = max(squid_depth.buy_orders.keys())
                        squid_orders.append(Order(Product.SQUID_INK, best_bid, -squid_pos))
                        logger.print(f"LONG EXIT: {-squid_pos} @ {best_bid}")
                        
                    elif self.spread_position == -1 and squid_pos < 0:
                        # Exit short position
                        best_ask = min(squid_depth.sell_orders.keys())
                        squid_orders.append(Order(Product.SQUID_INK, best_ask, -squid_pos))
                        logger.print(f"SHORT EXIT: {-squid_pos} @ {best_ask}")
                        
                    self.spread_position = 0
                    self.holding_ticks = 0

            if squid_orders:
                result[Product.SQUID_INK] = squid_orders

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData