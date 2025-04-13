from datamodel import Order, OrderDepth, TradingState
import numpy as np
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json

from typing import List, Dict, Tuple,Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Dict, List, Tuple, Any
from json import JSONEncoder
import jsonpickle
import statistics

logger = None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]],
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

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for orders_list in orders.values() for o in orders_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# Constants
VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}
POSITION_LIMIT = 200
Z_ENTRY = 1.8
Z_EXIT = 0.2
FAIR_VALUES = {
    "VOLCANIC_ROCK_VOUCHER_9500": 666.5,
    "VOLCANIC_ROCK_VOUCHER_9750": 416.5,
    "VOLCANIC_ROCK_VOUCHER_10000": 166.5,
    "VOLCANIC_ROCK_VOUCHER_10250": 0.016,
    "VOLCANIC_ROCK_VOUCHER_10500": 0.0,
}

class Trader:
    def __init__(self):
        self.spread_history = {v: [] for v in VOUCHERS}

    def get_mid(self, order_depth: OrderDepth):
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def voucher_orders(self, product, order_depth, position, traderObject):
        orders = []
        fair_value = FAIR_VALUES[product]
        mid_price = self.get_mid(order_depth)
        if mid_price is None:
            return orders

        # Track spread
        spread = mid_price - fair_value
        history = traderObject.setdefault("spread_history", {}).setdefault(product, [])
        history.append(spread)
        if len(history) > 50:
            history.pop(0)

        std = np.std(history) if len(history) > 10 else 1  # fallback to 1 to avoid zero-division
        zscore = spread / std

        # Determine trade logic
        if zscore > Z_ENTRY and position > -POSITION_LIMIT:
            # Overpriced: short
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(product, best_bid, -min(10, POSITION_LIMIT + position)))
        elif zscore < -Z_ENTRY and position < POSITION_LIMIT:
            # Underpriced: buy
            best_ask = min(order_depth.sell_orders.keys())
            orders.append(Order(product, best_ask, min(10, POSITION_LIMIT - position)))
        elif abs(zscore) < Z_EXIT and position != 0:
            # Mean reversion: unwind
            if position > 0:
                best_bid = max(order_depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -position))
            elif position < 0:
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order(product, best_ask, -position))

        return orders

    def run(self, state: TradingState):
        result = {}
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        for product in VOUCHERS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                orders = self.voucher_orders(product, state.order_depths[product], position, traderObject)
                if orders:
                    result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
