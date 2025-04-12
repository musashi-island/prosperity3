from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Dict, Tuple, Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Dict, List, Tuple, Any
from json import JSONEncoder
import jsonpickle
#from sklearn.linear_model import LinearRegression
import statistics

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

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }
        self.spread_history: List[float] = []
        self.max_history = 100  # Maximum history size for calculating z-score
        self.z_entry = 1 # Entry threshold for z-score
        self.z_exit = 0.2  # Exit threshold for z-score

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate the synthetic weighted midpoint (SWMID)"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
        }

        od = state.order_depths
        pos = state.position

        # Get the mid prices for the products and baskets
        b1_mid = self.get_swmid(od[Product.PICNIC_BASKET1])
        b2_mid = self.get_swmid(od[Product.PICNIC_BASKET2])
        croissant_mid = self.get_swmid(od[Product.CROISSANTS])
        jam_mid = self.get_swmid(od[Product.JAMS])
        djembe_mid = self.get_swmid(od[Product.DJEMBES])

        # If any of the mid prices are None, we cannot compute the spread
        if None in [b1_mid, b2_mid, croissant_mid, jam_mid, djembe_mid]:
            return {}, 0, ""

        # Calculate the synthetic spreads
        left = 2 * b2_mid - 1 * b1_mid
        right = 2 * croissant_mid + 1 * jam_mid - 1 * djembe_mid
        spread = left - right

        # Append the current spread to the history
        self.spread_history.append(spread)
        if len(self.spread_history) > self.max_history:
            self.spread_history.pop(0)

        # Calculate z-score based on historical spreads
        if len(self.spread_history) >= 10:
            mean = statistics.mean(self.spread_history)
            stdev = statistics.stdev(self.spread_history)
            z = (spread - mean) / stdev if stdev > 0 else 0
        else:
            z = 0

        # Define order quantity for simplicity
        qty = 1

        # Entry condition: If z-score exceeds the entry threshold, we enter a trade
        if z > self.z_entry:
            orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), -2 * qty))
            orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), qty))
            orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, int(croissant_mid), 2 * qty))
            orders[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), qty))
            orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), -qty))

        # Exit condition: If z-score falls below the exit threshold, we exit the trade
        elif z < -self.z_entry:
            orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), 2 * qty))
            orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), -qty))
            orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, int(croissant_mid), -2 * qty))
            orders[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), -qty))
            orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), qty))

        # Empty trader data to flush logs
        traderData = ""
        logger.flush(state, orders, 0, traderData)
        conversions = 1
        return orders, conversions, traderData

