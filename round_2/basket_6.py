from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Listing, Observation, ProsperityEncoder, Symbol, Trade

from typing import List, Dict, Tuple, Any
import string
import jsonpickle
import numpy as np
import math
import json

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

PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 48.76243,
        "default_spread_std": 76.07966,
        "spread_std_window": 25,
        "target_position": 33,
        "adaptive_base": 5,
        "vol_factor": 0.35,
    },
}

BASKET_WEIGHTS = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.JAMS: 350,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
        }

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_adaptive_threshold(self, spread_std: float) -> float:
        base = self.params[Product.SPREAD].get("adaptive_base", 5)
        vol_factor = self.params[Product.SPREAD].get("vol_factor", 0.3)
        return base + vol_factor * spread_std

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        J, C, D = BASKET_WEIGHTS[Product.JAMS], BASKET_WEIGHTS[Product.CROISSANTS], BASKET_WEIGHTS[Product.DJEMBES]
        synthetic_order_price = OrderDepth()

        J_bid = max(order_depths[Product.JAMS].buy_orders.keys(), default=0)
        J_ask = min(order_depths[Product.JAMS].sell_orders.keys(), default=float("inf"))
        C_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=0)
        C_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=float("inf"))
        D_bid = max(order_depths[Product.DJEMBES].buy_orders.keys(), default=0)
        D_ask = min(order_depths[Product.DJEMBES].sell_orders.keys(), default=float("inf"))

        implied_bid = J_bid * J + C_bid * C + D_bid * D
        implied_ask = J_ask * J + C_ask * C + D_ask * D

        if implied_bid > 0:
            J_vol = order_depths[Product.JAMS].buy_orders.get(J_bid, 0) // J
            C_vol = order_depths[Product.CROISSANTS].buy_orders.get(C_bid, 0) // C
            D_vol = order_depths[Product.DJEMBES].buy_orders.get(D_bid, 0) // D
            synthetic_order_price.buy_orders[implied_bid] = min(J_vol, C_vol, D_vol)

        if implied_ask < float("inf"):
            J_vol = -order_depths[Product.JAMS].sell_orders.get(J_ask, 0) // J
            C_vol = -order_depths[Product.CROISSANTS].sell_orders.get(C_ask, 0) // C
            D_vol = -order_depths[Product.DJEMBES].sell_orders.get(D_ask, 0) // D
            synthetic_order_price.sell_orders[implied_ask] = -min(J_vol, C_vol, D_vol)

        return synthetic_order_price

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask = min(basket_order_depth.sell_orders.keys())
            synthetic_bid = max(synthetic_order_depth.buy_orders.keys())
            volume = min(abs(basket_order_depth.sell_orders[basket_ask]), abs(synthetic_order_depth.buy_orders[synthetic_bid]), target_quantity)
            return {
                Product.PICNIC_BASKET1: [Order(Product.PICNIC_BASKET1, basket_ask, volume)],
                Product.JAMS: [Order(Product.JAMS, min(order_depths[Product.JAMS].sell_orders.keys()), volume * BASKET_WEIGHTS[Product.JAMS])],
                Product.CROISSANTS: [Order(Product.CROISSANTS, min(order_depths[Product.CROISSANTS].sell_orders.keys()), volume * BASKET_WEIGHTS[Product.CROISSANTS])],
                Product.DJEMBES: [Order(Product.DJEMBES, min(order_depths[Product.DJEMBES].sell_orders.keys()), volume * BASKET_WEIGHTS[Product.DJEMBES])],
            }
        else:
            basket_bid = max(basket_order_depth.buy_orders.keys())
            synthetic_ask = min(synthetic_order_depth.sell_orders.keys())
            volume = min(abs(basket_order_depth.buy_orders[basket_bid]), abs(synthetic_order_depth.sell_orders[synthetic_ask]), target_quantity)
            return {
                Product.PICNIC_BASKET1: [Order(Product.PICNIC_BASKET1, basket_bid, -volume)],
                Product.JAMS: [Order(Product.JAMS, max(order_depths[Product.JAMS].buy_orders.keys()), -volume * BASKET_WEIGHTS[Product.JAMS])],
                Product.CROISSANTS: [Order(Product.CROISSANTS, max(order_depths[Product.CROISSANTS].buy_orders.keys()), -volume * BASKET_WEIGHTS[Product.CROISSANTS])],
                Product.DJEMBES: [Order(Product.DJEMBES, max(order_depths[Product.DJEMBES].buy_orders.keys()), -volume * BASKET_WEIGHTS[Product.DJEMBES])],
            }

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]):
        if Product.PICNIC_BASKET1 not in order_depths:
            return None

        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_od)
        synthetic_swmid = self.get_swmid(synthetic_od)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - self.params[Product.SPREAD]["default_spread_mean"]) / spread_std
        adaptive_threshold = self.get_adaptive_threshold(spread_std)

        logger.print(f"Z-score: {zscore:.2f}, Adaptive Threshold: {adaptive_threshold:.2f}, Spread STD: {spread_std:.2f}")

        if zscore >= adaptive_threshold:
            return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        elif zscore <= -adaptive_threshold:
            return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderObject[Product.SPREAD])
        if spread_orders:
            for prod, orders in spread_orders.items():
                result[prod] = orders

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData