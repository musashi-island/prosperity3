from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List
import jsonpickle
import numpy as np
import json

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
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}
POSITION_LIMIT = 200
Z_ENTRY = 1.8
Z_EXIT = 0.2

class Trader:
    def __init__(self):
        pass

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Synthetic weighted mid-price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def voucher_orders(self, product, order_depth, position, traderObject):
        orders = []
        mid_price = self.get_swmid(order_depth)
        if mid_price is None:
            return orders

        # Maintain rolling history of mid prices
        history = traderObject.setdefault("mid_history", {}).setdefault(product, [])
        history.append(mid_price)
        if len(history) > 50:
            history.pop(0)

        mean = np.mean(history)
        std = np.std(history) if len(history) > 10 else 1  # fallback to avoid division by zero
        zscore = (mid_price - mean) / std

        # Trading logic based on z-score reversion
        if zscore > Z_ENTRY and position > -POSITION_LIMIT:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(product, best_bid, -min(200, POSITION_LIMIT + position)))
        elif zscore < -Z_ENTRY and position < POSITION_LIMIT:
            best_ask = min(order_depth.sell_orders.keys())
            orders.append(Order(product, best_ask, min(200, POSITION_LIMIT - position)))
        elif abs(zscore) < Z_EXIT and position != 0:
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
