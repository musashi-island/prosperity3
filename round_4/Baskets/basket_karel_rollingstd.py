#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 01:45:06 2025

@author: Jules
"""
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

#110k for #1.4 and #100

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

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
        self.max_history = 30  # Maximum history size for calculating z-score
        self.z_entry = 1  # Entry threshold for z-score
        self.z_exit = 0.2  # Exit threshold for z-score

    def get_best_bid(self, order_depth: OrderDepth) -> Tuple[int, int]:
        """Return the best bid price and volume from the order book"""
        if not order_depth.buy_orders:
            return None  # No buy orders available
        best_bid_price = max(order_depth.buy_orders.keys())  # Highest bid price
        best_bid_vol = abs(order_depth.buy_orders[best_bid_price])  # The volume at that price
        return best_bid_price, best_bid_vol

    def get_best_ask(self, order_depth: OrderDepth) -> Tuple[int, int]:
        """Return the best ask price and volume from the order book"""
        if not order_depth.sell_orders:
            return None  # No sell orders available
        best_ask_price = min(order_depth.sell_orders.keys())  # Lowest ask price
        best_ask_vol = abs(order_depth.sell_orders[best_ask_price])  # The volume at that price
        return best_ask_price, best_ask_vol
    
    def get_midprice(self, order_depth: OrderDepth) -> float:
        """Compute the mid-price from an order depth (used for RAINFOREST_RESIN only)."""
        if not order_depth or (not order_depth.buy_orders) or (not order_depth.sell_orders):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate the synthetic weighted midpoint (SWMID)"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def short_spread(self, result: Dict[str, List[Order]], b1_pos: int, b2_pos: int, dj_pos: int,
                      od_b1: OrderDepth, od_b2: OrderDepth, od_dj: OrderDepth):
        # Short spread: SELL 2 Basket1, BUY 3 Basket2, BUY 2 DJEMBES
        limit_short_b1 = b1_pos + self.LIMIT["PICNIC_BASKET1"]
        best_bid_b1 = self.get_best_bid(od_b1)
        if best_bid_b1 and limit_short_b1 >= 2:
            b1_bid_price, b1_bid_vol = best_bid_b1
            trade_size_b1 = min(2, b1_bid_vol, limit_short_b1)
            if trade_size_b1 > 0:
                result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_bid_price), -trade_size_b1))
                limit_buy_b2 = self.LIMIT["PICNIC_BASKET2"] - b2_pos
                best_ask_b2 = self.get_best_ask(od_b2)
                if best_ask_b2 and limit_buy_b2 >= 3:
                    b2_ask_price, b2_ask_vol = best_ask_b2
                    needed_b2 = 3 * trade_size_b1
                    trade_size_b2 = min(needed_b2, b2_ask_vol, limit_buy_b2)
                    if trade_size_b2 > 0:
                        result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_ask_price), trade_size_b2))
                limit_buy_dj = self.LIMIT["DJEMBES"] - dj_pos
                best_ask_dj = self.get_best_ask(od_dj)
                if best_ask_dj and limit_buy_dj >= 2:
                    dj_ask_price, dj_ask_vol = best_ask_dj
                    needed_dj = 2 * trade_size_b1
                    trade_size_dj = min(needed_dj, dj_ask_vol, limit_buy_dj)
                    if trade_size_dj > 0:
                        result[Product.DJEMBES].append(Order(Product.DJEMBES, int(dj_ask_price), trade_size_dj))

    def long_spread(self, result: Dict[str, List[Order]], b1_pos: int, b2_pos: int, dj_pos: int,
                    od_b1: OrderDepth, od_b2: OrderDepth, od_dj: OrderDepth):
        # Long spread: BUY 2 Basket1, SELL 3 Basket2, SELL 2 DJEMBES
        limit_buy_b1 = self.LIMIT["PICNIC_BASKET1"] - b1_pos
        best_ask_b1 = self.get_best_ask(od_b1)
        if best_ask_b1 and limit_buy_b1 >= 2:
            b1_ask_price, b1_ask_vol = best_ask_b1
            trade_size_b1 = min(2, b1_ask_vol, limit_buy_b1)
            if trade_size_b1 > 0:
                result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_ask_price), trade_size_b1))
                limit_sell_b2 = b2_pos + self.LIMIT["PICNIC_BASKET2"]
                best_bid_b2 = self.get_best_bid(od_b2)
                if best_bid_b2 and limit_sell_b2 >= 3:
                    b2_bid_price, b2_bid_vol = best_bid_b2
                    needed_b2 = 3 * trade_size_b1
                    trade_size_b2 = min(needed_b2, b2_bid_vol, limit_sell_b2)
                    if trade_size_b2 > 0:
                        result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_bid_price), -trade_size_b2))
                limit_sell_dj = dj_pos + self.LIMIT["DJEMBES"]
                best_bid_dj = self.get_best_bid(od_dj)
                if best_bid_dj and limit_sell_dj >= 2:
                    dj_bid_price, dj_bid_vol = best_bid_dj
                    needed_dj = 2 * trade_size_b1
                    trade_size_dj = min(needed_dj, dj_bid_vol, limit_sell_dj)
                    if trade_size_dj > 0:
                        result[Product.DJEMBES].append(Order(Product.DJEMBES, int(dj_bid_price), -trade_size_dj))

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {
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

        #b1_mid = self.get_midprice(od[Product.PICNIC_BASKET1])
        #b2_mid = self.get_midprice(od[Product.PICNIC_BASKET2])
        #croissant_mid = self.get_midprice(od[Product.CROISSANTS])
        #jam_mid = self.get_midprice(od[Product.JAMS])
        #djembe_mid = self.get_midprice(od[Product.DJEMBES])

        # If any of the mid prices are None, we cannot compute the spread
        if None in [b1_mid, b2_mid, croissant_mid, jam_mid, djembe_mid]:
            return {}, 0, ""

        # Calculate the synthetic spreads
        left = 2 * b1_mid - 3 * b2_mid  # 2 * Basket1 - 3 * Basket2
        right = 2 * djembe_mid  # 2 * DJEMBE
        spread = left - right

        # Append the current spread to the history   
        self.spread_history.append(spread)
        if len(self.spread_history) > self.max_history:
            self.spread_history.pop(0)
            mean = -32.34
            #stdev = 119.13
            stdev = statistics.stdev(self.spread_history)
            z = (spread - mean) / stdev if stdev > 0 else (spread - mean)

        # Calculate z-score based on historical spreads
        #if len(self.spread_history) >= 30:
        #    mean = statistics.mean(self.spread_history)
        #    stdev = statistics.stdev(self.spread_history)
        #    z = (spread - mean) / stdev if stdev > 0 else (spread - mean)
        else:
            z = 0
        


        # Position for Basket1, Basket2, and DJEMBES
        b1_pos = pos.get(Product.PICNIC_BASKET1, 0)
        b2_pos = pos.get(Product.PICNIC_BASKET2, 0)
        dj_pos = pos.get(Product.DJEMBES, 0)

        # Use the short_spread or long_spread function based on z-score
        if z > self.z_entry:
            self.short_spread(orders, b1_pos, b2_pos, dj_pos, od[Product.PICNIC_BASKET1], od[Product.PICNIC_BASKET2], od[Product.DJEMBES])
        elif z < -self.z_entry:
            self.long_spread(orders, b1_pos, b2_pos, dj_pos, od[Product.PICNIC_BASKET1], od[Product.PICNIC_BASKET2], od[Product.DJEMBES])

        # Prepare trader data
        traderObject = {}
        traderData = jsonpickle.encode(traderObject)
        
        # Final result to match the expected output format
        result = {
            Product.DJEMBES: orders[Product.DJEMBES],
            Product.PICNIC_BASKET1: orders[Product.PICNIC_BASKET1
                                           ],
            Product.PICNIC_BASKET2: orders[Product.PICNIC_BASKET2],
        }

        # Return result, conversions, and trader data
        conversions = 1  # Placeholder value for conversions
        logger.flush(state, orders, 0, traderData)
        return result, conversions, traderData