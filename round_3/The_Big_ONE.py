#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:12:42 2025

@author: Jules
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List
import jsonpickle
import numpy as np
import json


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
        return [[o.symbol, o.price, o.quantity]
                for orders_list in orders.values()
                for o in orders_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# ==================== Constants ====================

VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}
POSITION_LIMIT = 200
POSITION_LIMIT_VOLCANIC_ROCK = 400
Z_ENTRY = 1.8
Z_EXIT = 0

# We will treat VOLCANIC_ROCK as a separate product (not a voucher).
# Let's define a list of all products to iterate over in our logic.
ALL_PRODUCTS = list(VOUCHERS.keys()) + ["VOLCANIC_ROCK"]

class Trader:
    def __init__(self):
        pass

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """
        Synthetic weighted mid-price.
        Weights the best bid and best ask by their volumes.
        Returns None if data is insufficient.
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])

        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def mean_reversion_orders(
        self, product: str, order_depth: OrderDepth, position: int, trader_object: dict
    ) -> List[Order]:
        """
        Returns a list of Orders for a single product, either a voucher or VOLCANIC_ROCK,
        using the same mean-reversion logic based on z-score.
        """
        orders = []
        mid_price = self.get_swmid(order_depth)
        if mid_price is None:
            return orders  # No trades if we don't have both sides

        # Maintain a rolling history of mid prices for each product
        history = trader_object.setdefault("mid_history", {}).setdefault(product, [])
        history.append(mid_price)

        # Limit rolling window to 50
        if len(history) > 50:
            history.pop(0)

        mean = np.mean(history)
        # Use fallback std = 1 if too few data points to avoid div-by-zero
        std = np.std(history) if len(history) > 10 else 1
        zscore = (mid_price - mean) / std
        
        if product == "VOLCANIC_ROCK":
            POSITION_LIMIT = 400  
            amount = 400
        else : 
            POSITION_LIMIT = 200  
            amount = 200

        # Determine trade action based on zscore
        if zscore > Z_ENTRY and position > -POSITION_LIMIT:
            # Sell at best_bid
            best_bid = max(order_depth.buy_orders.keys())
            # Sell up to 100 units or whatever is allowed to go short
            quantity = -min(amount, POSITION_LIMIT + position)
            orders.append(Order(product, best_bid, quantity))

        elif zscore < -Z_ENTRY and position < POSITION_LIMIT:
            # Buy at best_ask
            best_ask = min(order_depth.sell_orders.keys())
            # Buy up to 100 units or whatever is allowed to go long
            quantity = min(amount, POSITION_LIMIT - position)
            orders.append(Order(product, best_ask, quantity))

        # If z-score is exactly Z_EXIT (i.e. 0) and we have a position, attempt to flatten
        # NOTE: floating-point comparisons to 0 can be tricky. This logic is from original code.
        if abs(zscore) == Z_EXIT and position != 0:
            if position > 0:
                # Sell everything at best_bid
                best_bid = max(order_depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -position))
            elif position < 0:
                # Buy everything back at best_ask
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order(product, best_ask, -position))

        return orders

    def run(self, state: TradingState):
        """
        Main loop for generating orders each time the exchange calls our trader.
        """
        result = {}
        trader_object = jsonpickle.decode(state.traderData) if state.traderData else {}

        # Iterate over all products, including vouchers and VOLCANIC_ROCK
        for product in ALL_PRODUCTS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                order_depth = state.order_depths[product]

                orders = self.mean_reversion_orders(
                    product,
                    order_depth,
                    position,
                    trader_object
                )
                if orders:
                    result[product] = orders

        # Encode updated trader state
        trader_data = jsonpickle.encode(trader_object)
        conversions = 0

        # Flush logs
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
