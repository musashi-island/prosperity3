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
        self.max_history = 100  # Maximum history size for calculating z-score
        self.z_entry = 0.8  # Entry threshold for z-score
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
    
    def short_basket2(self, result: Dict[str, List[Order]], b2_pos: int, cro_pos: int, jam_pos: int,
                    od_b2: OrderDepth, od_cro: OrderDepth, od_jam: OrderDepth):
    # Short BSK2, Long 4 CRO, Long 2 JAM
        limit_sell_b2 = b2_pos + self.LIMIT[Product.PICNIC_BASKET2]
        best_bid_b2 = self.get_best_bid(od_b2)
        if best_bid_b2 and limit_sell_b2 > 0:
            b2_price, b2_vol = best_bid_b2
            trade_size = min(b2_vol, limit_sell_b2)
            if trade_size > 0:
                result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, b2_price, -trade_size))

                limit_buy_cro = self.LIMIT[Product.CROISSANTS] - cro_pos
                best_ask_cro = self.get_best_ask(od_cro)
                if best_ask_cro and limit_buy_cro >= 4 * trade_size:
                    cro_price, cro_vol = best_ask_cro
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, cro_price, 4 * trade_size))

                limit_buy_jam = self.LIMIT[Product.JAMS] - jam_pos
                best_ask_jam = self.get_best_ask(od_jam)
                if best_ask_jam and limit_buy_jam >= 2 * trade_size:
                    jam_price, jam_vol = best_ask_jam
                    result[Product.JAMS].append(Order(Product.JAMS, jam_price, 2 * trade_size))

    def long_basket2(self, result: Dict[str, List[Order]], b2_pos: int, cro_pos: int, jam_pos: int,
                 od_b2: OrderDepth, od_cro: OrderDepth, od_jam: OrderDepth):
    # Long BSK2, Short 4 CRO, Short 2 JAM
        limit_buy_b2 = self.LIMIT[Product.PICNIC_BASKET2] - b2_pos
        best_ask_b2 = self.get_best_ask(od_b2)
        if best_ask_b2 and limit_buy_b2 > 0:
            b2_price, b2_vol = best_ask_b2
            trade_size = min(b2_vol, limit_buy_b2)
            if trade_size > 0:
                result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, b2_price, trade_size))

                limit_sell_cro = cro_pos + self.LIMIT[Product.CROISSANTS]
                best_bid_cro = self.get_best_bid(od_cro)
                if best_bid_cro and limit_sell_cro >= 4 * trade_size:
                    cro_price, cro_vol = best_bid_cro
                    result[Product.CROISSANTS].append(Order(Product.CROISSANTS, cro_price, -4 * trade_size))

                limit_sell_jam = jam_pos + self.LIMIT[Product.JAMS]
                best_bid_jam = self.get_best_bid(od_jam)
                if best_bid_jam and limit_sell_jam >= 2 * trade_size:
                    jam_price, jam_vol = best_bid_jam
                    result[Product.JAMS].append(Order(Product.JAMS, jam_price, -2 * trade_size))

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {
            Product.PICNIC_BASKET2: [],
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        od = state.order_depths
        pos = state.position

        # Get SWMID prices
        b2_mid = self.get_swmid(od[Product.PICNIC_BASKET2])
        croissant_mid = self.get_swmid(od[Product.CROISSANTS])
        jam_mid = self.get_swmid(od[Product.JAMS])

        if None in [b2_mid, croissant_mid, jam_mid]:
            return {}, 0, ""

        # Synthetic price for Basket2 = 4 * CROISSANTS + 2 * JAMS
        b2_synthetic = 4 * croissant_mid + 2 * jam_mid
        spread = b2_mid - b2_synthetic

        # Update spread history
        self.spread_history.append(spread)
        if len(self.spread_history) > self.max_history:
            self.spread_history.pop(0)

        # Compute z-score
        #if len(self.spread_history) >= 30:
        #    mean = statistics.mean(self.spread_history)
        #    stdev = statistics.stdev(self.spread_history)
        #    z = (spread - mean) / stdev if stdev > 0 else 0
        #else:
        #    z = 0

        mean = 40.31
        stdev = 58.58
        z = (spread - mean) / stdev if stdev > 0 else 0


        b2_pos = pos.get(Product.PICNIC_BASKET2, 0)
        cro_pos = pos.get(Product.CROISSANTS, 0)
        jam_pos = pos.get(Product.JAMS, 0)
        

        # Entry logic based on z-score
        if z > self.z_entry:
            self.short_basket2(orders, b2_pos, cro_pos, jam_pos,
                            od[Product.PICNIC_BASKET2], od[Product.CROISSANTS], od[Product.JAMS])
        elif z < -self.z_entry:
            self.long_basket2(orders, b2_pos, cro_pos, jam_pos,
                            od[Product.PICNIC_BASKET2], od[Product.CROISSANTS], od[Product.JAMS])

        # Final packaging
        traderObject = {}
        traderData = jsonpickle.encode(traderObject)

        result = {
            Product.PICNIC_BASKET2: orders[Product.PICNIC_BASKET2],
            Product.CROISSANTS: orders[Product.CROISSANTS],
            Product.JAMS: orders[Product.JAMS],
        }

        logger.flush(state, orders, 0, traderData)
        return result, 0, traderData
