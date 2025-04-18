#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any
import jsonpickle
import numpy as np

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

logger = Logger()

class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

class Trader:
    def __init__(self):
        self.CSI = 26.06
        self.position_limit = 75
        self.conversion_limit = 10
        self.history_window = 100

    def get_best_bid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders:
            return None
        best_bid_price = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid_price])
        return best_bid_price, best_bid_vol

    def get_best_ask(self, order_depth: OrderDepth):
        if not order_depth.sell_orders:
            return None
        best_ask_price = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask_price])
        return best_ask_price, best_ask_vol

    def get_swmid(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid, bid_vol = self.get_best_bid(order_depth)
        best_ask, ask_vol = self.get_best_ask(order_depth)
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def run(self, state: TradingState):
        traderData = jsonpickle.decode(state.traderData) if state.traderData else {}
        orders = {}
        product = Product.MACARONS
        position = state.position.get(product, 0)

        order_depth = state.order_depths.get(product, 0)
        obs = state.observations.conversionObservations.get(product)
        #order_depth = OrderDepth

        sunlight = obs.sunlightIndex
        swmid = self.get_swmid(order_depth)
        #if swmid is None:
            #logger.print("SWMID is None at timestamp", state.timestamp)
            #logger.flush(state, {}, 0, state.traderData)
        #    return orders, 0, state.traderData

        history = traderData.get("history", {}).get(product, [])
        history.append(swmid)
        if len(history) > self.history_window:
            history.pop(0)

        traderData["history"] = traderData.get("history", {})
        traderData["history"][product] = history

        product_orders = []
        panic = sunlight < self.CSI
        #logger.print(f"Timestamp: {state.timestamp}, Sunlight: {sunlight:.2f}, Panic Mode: {panic}")

        if panic:
            best_ask = self.get_best_ask(order_depth)
            if best_ask is not None:
                price, volume = best_ask
                buy_qty = min(volume, self.position_limit - position)
                if buy_qty > 0:
                    #logger.print(f"Panic BUY {buy_qty} @ {price}")
                    product_orders.append(Order(product, price, buy_qty))
        else:
            if len(history) >= 20:
                mean = np.mean(history)
                std = np.std(history)
                z = (swmid - mean) / std if std > 0 else 0
               # logger.print(f"Z-score: {z:.2f}, Mean: {mean:.2f}, STD: {std:.2f}, SWMID: {swmid:.2f}")

                if z > 1.5 and position > -self.position_limit:
                    best_bid = self.get_best_bid(order_depth)
                    if best_bid is not None:
                        price, volume = best_bid
                        sell_qty = min(volume, self.position_limit + position)
                        if sell_qty > 0:
                            #logger.print(f"Sell {sell_qty} @ {price} due to high Z")
                            product_orders.append(Order(product, price, -sell_qty))

                elif z < -1.5 and position < self.position_limit:
                    best_ask = self.get_best_ask(order_depth)
                    if best_ask is not None:
                        price, volume = best_ask
                        buy_qty = min(volume, self.position_limit - position)
                        if buy_qty > 0:
                            #logger.print(f"Buy {buy_qty} @ {price} due to low Z")
                            product_orders.append(Order(product, price, buy_qty))

        if product_orders:
            orders[product] = product_orders

        new_traderData = jsonpickle.encode(traderData)
        logger.flush(state, orders, 0, new_traderData)
        return orders, 0, new_traderData