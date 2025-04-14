#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:09:16 2025

@author: Jules
"""

from datamodel import Order, OrderDepth, TradingState
import numpy as np
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json
from json import JSONEncoder
import statistics
from math import log, sqrt, exp
from statistics import NormalDist

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
Z_ENTRY = 2.2
T = 6/ 7   # So corresponds to day 2, when uploading put 5/7 for day 3

# FAIR_VALUES = {
#     "VOLCANIC_ROCK_VOUCHER_9500": 666.5,
#     "VOLCANIC_ROCK_VOUCHER_9750": 416.5,
#     "VOLCANIC_ROCK_VOUCHER_10000": 166.5,
#     "VOLCANIC_ROCK_VOUCHER_10250": 0.016,
#     "VOLCANIC_ROCK_VOUCHER_10500": 0.0,
# }


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if volatility == 0:
            return max(spot - strike, 0)
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        if volatility == 0:
            return 0.0
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if volatility == 0:
            return 0
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self):
        self.spread_history = {v: [] for v in VOUCHERS}

        # Define position limits for each product
        self.LIMIT = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate the synthetic weighted midpoint (SWMID)"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    

    def get_volatility(self, product, coupon_price, traderObject):
        past_prices = traderObject.setdefault("past_coupon_vol", [])
        past_prices.append(coupon_price)
        if len(past_prices) > 50:
            past_prices.pop(0)

        # Calculate volatility (std deviation of log returns)
        if len(past_prices) > 1:
            log_returns = np.diff(np.log(past_prices))
            volatility = np.std(log_returns)
        else:
            volatility = 0  # Fallback to 0 if not enough data
        return volatility

    def voucher_orders(self, product, order_depth, position, traderObject, underlying_price):
        orders = []
        
        # Retrieve the strike price from the VOUCHERS dictionary
        strike_price = VOUCHERS[product]
    
        # Use Black-Scholes to compute the fair value of the voucher
        # Assuming volatility is dynamically calculated or provided in traderObject
        volatility = self.get_volatility(product, underlying_price, traderObject)  # dynamically calculated volatility
        time_to_expiry = T  # Time to expiry (could be dynamic)
    
        # Compute the fair value of the voucher using the Black-Scholes call option pricing model
        fair_value = BlackScholes.black_scholes_call(underlying_price, strike_price, time_to_expiry, volatility)
        
    
        mid_price = self.get_swmid(order_depth)
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
        if zscore > Z_ENTRY and position > -self.LIMIT[product]:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(product, best_bid, -min(100, self.LIMIT[product] + position)))
        elif zscore < -Z_ENTRY and position < self.LIMIT[product]:
            best_ask = min(order_depth.sell_orders.keys())
            orders.append(Order(product, best_ask, min(100, self.LIMIT[product] - position)))
        # elif abs(zscore) < 0.2 and position != 0:  # Removed Z_EXIT for no exit logic
        #     if position > 0:
        #         best_bid = max(order_depth.buy_orders.keys())
        #         orders.append(Order(product, best_bid, -position))
        #     elif position < 0:
        #         best_ask = min(order_depth.sell_orders.keys())
        #         orders.append(Order(product, best_ask, -position))

        # Store delta for hedging
        strike = VOUCHERS[product]
        volatility = self.get_volatility(product, mid_price, traderObject)
        delta = BlackScholes.delta(underlying_price, strike, T, volatility)
        traderObject.setdefault("voucher_deltas", {})[product] = delta * position

        return orders

    def hedge_delta(self, traderObject, order_depth, position):
        orders = []
        net_delta = sum(traderObject.get("voucher_deltas", {}).values())

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders

        hedge_quantity = int(round(-net_delta))
        hedge_quantity = max(-self.LIMIT["VOLCANIC_ROCK"] - position, min(self.LIMIT["VOLCANIC_ROCK"] - position, hedge_quantity))

        if hedge_quantity > 0:
            orders.append(Order("VOLCANIC_ROCK", best_ask, hedge_quantity))
        elif hedge_quantity < 0:
            orders.append(Order("VOLCANIC_ROCK", best_bid, hedge_quantity))

        return orders

    def calculate_hedge(self, traderObject, total_exposure, volcanic_rock_mid, orders):
        position_limit = self.LIMIT["VOLCANIC_ROCK"]

        # If exposure exceeds the position limit, scale it down
        scaling_factor = min(position_limit / abs(total_exposure), 1) if abs(total_exposure) > 0 else 1

        # Adjusted exposure within position limits
        adjusted_exposure = total_exposure * scaling_factor
        
        if "VOLCANIC_ROCK" not in orders:
            orders["VOLCANIC_ROCK"] = []

        # If the exposure is positive, short the Volcanic Rock (hedge long position in coupons)
        if adjusted_exposure > 0:
            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", int(volcanic_rock_mid), -int(adjusted_exposure)))

        # If the exposure is negative, long the Volcanic Rock (hedge short position in coupons)
        elif adjusted_exposure < 0:
            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", int(volcanic_rock_mid), -int(adjusted_exposure)))

        return orders

    def run(self, state: TradingState):
        result = {}
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        underlying_price = self.get_swmid(state.order_depths.get("VOLCANIC_ROCK", OrderDepth()))
        if underlying_price is None:
            return {}, 0, jsonpickle.encode(traderObject)  # Can't proceed without it

        for product in VOUCHERS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                orders = self.voucher_orders(product, state.order_depths[product], position, traderObject, underlying_price)
                if orders:
                    result[product] = orders

        # Delta hedge
        rock_position = state.position.get("VOLCANIC_ROCK", 0)
        hedge_orders = self.hedge_delta(traderObject, state.order_depths.get("VOLCANIC_ROCK", OrderDepth()), rock_position)
        if hedge_orders:
            result["VOLCANIC_ROCK"] = hedge_orders

        # Hedge adjustment based on position and delta
        total_exposure = sum(traderObject.get("voucher_deltas", {}).values())
        volcanic_rock_mid = self.get_swmid(state.order_depths.get("VOLCANIC_ROCK", OrderDepth()))
        result = self.calculate_hedge(traderObject, total_exposure, volcanic_rock_mid, result)

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
