from datamodel import Observation, Order, OrderDepth, TradingState
from typing import List, Dict, Any
import jsonpickle
import numpy as np
import math

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

    def compress_state(self, state: TradingState, trader_data: str) -> list:
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

    def compress_orders(self, orders: Dict[str, List[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity]
                for orders_list in orders.values()
                for o in orders_list]

    def to_json(self, value: Any) -> str:
        import json
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

# instantiate
logger = Logger()

# === product constant ===
class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

# === main trader ===
class Trader:
    def __init__(self):
        # how many macarons to sell each tick
        self.symbol      = Product.MACARONS
        self.limit       = 10
        # track conversions across ticks if you like
        self.total_conversions = 0

    def macarons_strat(self, state: TradingState, obs: Observation) -> (List[Order], int):
        """
        Returns (orders, conversions) for the macaron strategy.
        We always sell up to `self.limit` at a small markup if we have fresh data.
        """
        orders: List[Order] = []
        conv = 0

        # 1) if we hold any macarons, "convert back" so we're flat
        pos = state.position.get(self.symbol, 0)
        if pos != 0:
            conv += abs(pos)

        # 2) if no conversion data, do nothing
        if obs is None:
            return orders, conv

        # 3) compute our prices
        buy_price  = obs.askPrice + obs.transportFees + obs.importTariff
        sell_price = max(int(obs.bidPrice - 0.5), int(buy_price + 1))

        # 4) place a sell order of size `limit`
        orders.append(Order(self.symbol, sell_price, -self.limit))

        return orders, conv

    def run(self, state: TradingState):

        if Product.MACARONS in state.order_depths:
            # get the latest conversion observation (might be None)
            obs = state.observations.conversionObservations.get(self.symbol)

            orders_list, conversions = self.macarons_strat(state, obs)

            orders: Dict[str, List[Order]] = {
                self.symbol: orders_list
            }
            self.total_conversions += conversions

            trader_data = jsonpickle.encode({
                "total_conversions": self.total_conversions
            })

        
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data