#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import List, Dict, Any
import jsonpickle
import numpy as np
import math
import json
from collections import deque

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
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

class Strategy:
    def init(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, args) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def save(self):
        return None

    def load(self, data,args) -> None:
        self.args = args
        pass

    
#NEED TO COMPUTE THE NEW CSI AND A AND B VALUES
# fair_value = 7.9888 * sugarPrice + -872.1104 for data < CSI = 44
#fair_value = 8.3533 * sugarPrice + 0.7749 * sunlightIndex + -977.6104   for data < CSI = 44
#fair_value = 3.4464* sugarPrice + -51.3392
#Model (SI > 44): fair_value = 5.4239 * sugarPrice + -2.4598 * sunlightIndex + -285.6516

class MacronStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().init(symbol, limit)
        self.lower_threshold = 43
        self.higher_threshold = 44
        self.fair_a = 3.4464
        self.fair_b = 0 
        self.fair_c = -51.3392

        #self.fair_a = 5.4239 #8.3533         # sugarPrice coefficient
        #self.fair_b = -2.4598 #0.7749        # sunlightIndex coefficient
        #self.fair_c = -285.6516 #-977.6104      # intercept

    
        self.previous_sunlightIndex = None
        self.window = deque()
        self.window_size = 10
        self.soft_position_limit = 0.3 #0.5 for 35k
        self.price_alt = 2
        self.entered = False

    def get_fair_value(self, observation: ConversionObservation) -> float:
        #return self.fair_a * observation.sugarPrice + self.fair_b
        return self.fair_a * observation.sugarPrice + self.fair_b * observation.sunlightIndex + self.fair_c

    def act(self, state: TradingState) -> None:
        obs = state.observations.conversionObservations[self.symbol]
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        csi = obs.sunlightIndex
        true_value = self.get_fair_value(obs)
        self.price_alt = max(1, int(0.05 * true_value))

        if self.previous_sunlightIndex is None:
            self.previous_sunlightIndex = csi
            return

        if csi >= self.lower_threshold and csi > self.previous_sunlightIndex and self.entered:    #csi going up, i close my short
            to_buy = self.limit - position
            sell_orders = sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []
            for price, volume in sell_orders:
                if to_buy > 0:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity
            if to_buy == 0:
                self.entered = False

        elif (csi >= self.higher_threshold and csi <= self.previous_sunlightIndex) or (csi > self.lower_threshold and csi > self.previous_sunlightIndex):
            # get out of a big position (buy or sell) when the trend is strong or changing
            
            # Track whether we’re stuck at the position limit
            self.window.append(abs(position) == self.limit)
            if len(self.window) > self.window_size:
                self.window.popleft()

            # Define soft and hard liquidation triggers
            soft_liquidate = (
                len(self.window) == self.window_size and
                sum(self.window) >= self.window_size / 2 and
                self.window[-1]
            )
            hard_liquidate = (
                len(self.window) == self.window_size and
                all(self.window)
            )

            # Adjust buy/sell price threshold based on how risky your position is
            max_buy_price = (
                true_value - self.price_alt if position > self.limit * self.soft_position_limit else true_value
            )
            min_sell_price = (
                true_value + self.price_alt if position < self.limit * -self.soft_position_limit else true_value
            )

            # Sort buy/sell orders by price
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True) if order_depth.buy_orders else []
            sell_orders = sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []

            # Compute how much we’re allowed to buy/sell without breaching limits
            to_buy = self.limit - position
            to_sell = self.limit + position

            # STEP 1 — try to BUY below max_buy_price
            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity

            # STEP 2 — fallback buy at popular price (up to 4 units)
            if to_buy > 0 and sell_orders:
                popular_price = max(sell_orders, key=lambda tup: tup[1])[0]
                if popular_price <= max_buy_price:
                    self.buy(popular_price, min(to_buy, 4))
                    to_buy -= min(to_buy, 4)

            # STEP 3 — forced buy at fair value (hard liquidation)
            if to_buy > 0 and hard_liquidate:
                quantity = to_buy // 2
                self.buy(true_value, quantity)
                to_buy -= quantity

            # STEP 4 — forced buy below fair value (soft liquidation)
            if to_buy > 0 and soft_liquidate:
                quantity = to_buy // 2
                self.buy(true_value - 2, quantity)
                to_buy -= quantity

            # STEP 5 — final buy fallback near popular price
            if to_buy > 0 and buy_orders:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                self.buy(price, to_buy)

            # STEP 6 — try to SELL above min_sell_price
            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    quantity = min(to_sell, volume)
                    self.sell(price, quantity)
                    to_sell -= quantity

            # STEP 7 — fallback sell at popular price (up to 4 units)
            if to_sell > 0 and buy_orders:
                popular_price = max(buy_orders, key=lambda tup: tup[1])[0]
                if popular_price >= min_sell_price:
                    self.sell(popular_price, min(to_sell, 4))
                    to_sell -= min(to_sell, 4)

            # STEP 8 — forced sell at fair value (hard liquidation)
            if to_sell > 0 and hard_liquidate:
                quantity = to_sell // 2
                self.sell(true_value, quantity)
                to_sell -= quantity

            # STEP 9 — forced sell above fair value (soft liquidation)
            if to_sell > 0 and soft_liquidate:
                quantity = to_sell // 2
                self.sell(true_value + 2, quantity)
                to_sell -= quantity

            # STEP 10 — final sell fallback near popular price
            if to_sell > 0 and sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(price, to_sell)

        else:
            self.entered = True
            if self.previous_sunlightIndex > csi:     #if csi going down i buy
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders)
                    buy_qty = self.limit - position
                    if buy_qty > 0:
                        self.buy(best_ask, buy_qty)
            elif self.previous_sunlightIndex < csi:    #if csi going up i sell
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders)
                    sell_qty = self.limit + position
                    if sell_qty > 0:
                        self.sell(best_bid, sell_qty)

        self.previous_sunlightIndex = csi

    def save(self):
        return {
            "previous_sunlightIndex": self.previous_sunlightIndex,
            "window": list(self.window),
            "entered": self.entered
        }

    def load(self, data, *args) -> None:
        self.args = args
        self.previous_sunlightIndex = data.get("previous_sunlightIndex", None)
        self.window = deque(data.get("window", []), maxlen=self.window_size)
        self.entered = data.get("entered", False)

class Trader:
    def __init__(self):
        self.strategies = {
            Product.MACARONS: MacronStrategy(Product.MACARONS, 75)
        }

    def run(self, state: TradingState):
        conversions = 0
        old_data = json.loads(state.traderData) if state.traderData else {}
        new_data = {}
        orders = {}

        if Product.MACARONS in state.order_depths:
            position = state.position.get(Product.MACARONS, 0)
            order_depth = state.order_depths[Product.MACARONS]

            if Product.MACARONS in old_data:
                self.strategies[Product.MACARONS].load(old_data[Product.MACARONS])
            macaron_orders = self.strategies[Product.MACARONS].run(state, args=None)
            if macaron_orders:
                orders[Product.MACARONS] = macaron_orders
            new_data[Product.MACARONS] = self.strategies[Product.MACARONS].save()

        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
