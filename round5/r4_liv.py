#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:50:13 2025

@author: Jules
"""
from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List
import jsonpickle
import numpy as np
import json

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

class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        sign = 1 if x >= 0 else -1
        x = abs(x) / _sqrt2
        t = 1.0 / (1.0 + _p * x)
        y = 1.0 - (((((_a5 * t + _a4) * t) + _a3) * t + _a2) * t + _a1) * t * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def norm_pdf(x):
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    @staticmethod
    def black_scholes_call(S, K, T, sigma, r=0):
        if T <= 1e-9 or sigma <= 1e-9:
            return max(0.0, S - K * math.exp(-r * T))
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * BlackScholes.norm_cdf(d1) - K * math.exp(-r * T) * BlackScholes.norm_cdf(d2)

    @staticmethod
    def delta(S, K, T, sigma, r=0):
        if T <= 1e-9 or sigma <= 1e-9:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return BlackScholes.norm_cdf(d1)

    @staticmethod
    def vega(S, K, T, sigma, r=0):
        if T <= 1e-9 or sigma <= 1e-9:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return S * BlackScholes.norm_pdf(d1) * math.sqrt(T) / 100

    @staticmethod
    def gamma(S, K, T, sigma, r=0):
        if T <= 1e-9 or sigma <= 1e-9:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return BlackScholes.norm_pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def vanna(S, K, T, sigma, r=0):
        if T <= 1e-9 or sigma <= 1e-9:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return BlackScholes.norm_pdf(d1) * d2 / sigma

    @staticmethod
    def implied_volatility(C, S, K, T, r=0, max_iterations=100, tolerance=1e-5):
        if C <= max(0.0, S - K * math.exp(-r * T)):
            return 0.0
        if C >= S:
            return 5.0
        low, high = 1e-4, 5.0
        vol = 0.16
        for _ in range(max_iterations):
            price = BlackScholes.black_scholes_call(S, K, max(T,1e-9), max(vol,1e-9), r)
            vga = BlackScholes.vega(S, K, T, vol, r) * 100
            diff = price - C
            if abs(diff) < tolerance:
                return max(vol,1e-4)
            if abs(vga) < 1e-6:
                if diff > 0: high = vol
                else: low = vol
            else:
                if diff > 0: high = vol
                else: low = vol
            vol = (low + high) / 2
            if vol < 1e-4: return 1e-4
            if vol > 5.0: return 5.0
        return max(vol,1e-4)


# ==================== Constants ====================

VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

ATM_VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
}

POSITION_LIMIT = 200
Z_ENTRY = 1.8
Z_EXIT = 0.2
AVERAGING_LENGTH = 200
SHORT_WINDOW = 100
LONG_WINDOW = 200
IV_DEVIATION_THRESHOLD = {
                        "VOLCANIC_ROCK_VOUCHER_9750" : 0.004,  
                        "VOLCANIC_ROCK_VOUCHER_10000": 0.003,
                        "VOLCANIC_ROCK_VOUCHER_10250": 0.0005,
                          }

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    VOLCANIC_ROCK                = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500   = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750   = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000  = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250  = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500  = "VOLCANIC_ROCK_VOUCHER_10500"
    MACARONS = "MAGNIFICENT_MACARONS"

_p = 0.3275911
_a1 = 0.254829592
_a2 = -0.284496736
_a3 = 1.421413741
_a4 = -1.453152027
_a5 = 1.061405429
_sqrt2 = math.sqrt(2.0)

GAMMA_THRESHOLD = 0.005
MIN_IV_STD_DEV_THRESHOLD = 0.0001
IV_STD_DEV_MULTIPLIER = 0.3
GAMMA_MEAN_FACTOR = 0.2
VANNA_MEAN_FACTOR = 0.2
MIN_POSITION_CHANGE = 0.001

PARAMS_OPTIONS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500:  {"strike": 9500,  "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_9750:  {"strike": 9750,  "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10000: {"strike": 10000, "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10250: {"strike": 10250, "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10500: {"strike": 10500, "vol_window": 100},
}

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0
    },
    Product.KELP: {
        "take_width": 1.5,            # 1.5
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,       # 15
        "reversion_beta": -0.229,    # -0.229
        "disregard_edge": 1,        # 1
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "fair_value" : 2000,
        "averaging_length" : 350,
        "trigger_price" : 10,
        "take_width": 5, #5
        "clear_width": 1, #1
        "prevent_adverse": False, #False
        "adverse_volume": 18, #16
        "reversion_beta":-0.3, #-0.3
        "disregard_edge": 1, #1
        "join_edge": 1, #1
        "default_edge": 1, #1
        "max_order_quantity": 35,
        "volatility_threshold": 3,
        "z_trigger" : 3.75
    },    Product.VOLCANIC_ROCK_VOUCHER_9500:  {"strike": 9500,  "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_9750:  {"strike": 9750,  "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10000: {"strike": 10000, "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10250: {"strike": 10250, "vol_window": 100},
    Product.VOLCANIC_ROCK_VOUCHER_10500: {"strike": 10500, "vol_window": 100},
}

# We will treat VOLCANIC_ROCK as a separate product (not a voucher).
# Let's define a list of all products to iterate over in our logic.
ALL_PRODUCTS = list(VOUCHERS.keys()) + ["VOLCANIC_ROCK"]

def smile(x):
    return 0.20600028*x**2 + 0.01064571*x + 0.18074582

class Trader:
    def __init__(self, params=None):
        self.params_opt = PARAMS_OPTIONS
        self.LIMIT_OPT  = {
            Product.VOLCANIC_ROCK:                400,
            Product.VOLCANIC_ROCK_VOUCHER_9500:   200,
            Product.VOLCANIC_ROCK_VOUCHER_9750:   200,
            Product.VOLCANIC_ROCK_VOUCHER_10000:  200,
            Product.VOLCANIC_ROCK_VOUCHER_10250:  200,
            Product.VOLCANIC_ROCK_VOUCHER_10500:  200,
        }
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK:                400,
            Product.VOLCANIC_ROCK_VOUCHER_9500:   200,
            Product.VOLCANIC_ROCK_VOUCHER_9750:   200,
            Product.VOLCANIC_ROCK_VOUCHER_10000:  200,
            Product.VOLCANIC_ROCK_VOUCHER_10250:  200,
            Product.VOLCANIC_ROCK_VOUCHER_10500:  200,
        }
        self.position = []
        # Store mid-price history for RAINFOREST_RESIN if desired
        self.resin_price_history = {Product.RAINFOREST_RESIN: []}
        self.spread_history: List[float] = []
        self.max_history = 200  # Maximum history size for calculating z-score
        self.z_entry = 1 # Entry threshold for z-score for 2bsk2 shit
        #self.z_exit = 0.2  # Exit threshold for z-score
        self.spread_history2: List[float] = []
        self.max_history2 = 100  # Maximum history size for calculating z-score
        self.z_entry2 = 1  # Entry threshold for z-score for croissant shit
        self.position_limit = 75
        self.conversion_limit = 10
        self.liv_trade = False
        self.take_profit = {
            "SQUID_INK": 65,  # 1% price movement for SQUID_INK
            "CROISSANTS": 5000,  # 3% price movement for CROISSANTS
        }
        self.entry_positions_olivia_trading = {}

    def determine_arbitrage_opportunity(self, state: TradingState):
        # 1) Current inventory
        current_pos = state.position.get('MAGNIFICENT_MACARONS', 0)

        # 2) South ask + fees
        conv = state.observations.conversionObservations['MAGNIFICENT_MACARONS']
        south_ask = conv.askPrice
        transport = conv.transportFees
        tariff    = conv.importTariff

        # 3) Available capacity
        free_capacity = 75 - abs(current_pos)
        trade_volume  = min(10, free_capacity)

        # 4) Compute prices
        # sell price on our island = ceil(south price + fees)
        sell_price_unit = math.ceil(south_ask + transport + tariff)
        # buy cost (south) = exact south price + fees
        buy_cost_unit   = south_ask + transport + tariff

        # 5) Profit check
        total_revenue = sell_price_unit * trade_volume
        total_cost    = buy_cost_unit * trade_volume
        is_arb = (trade_volume > 0) and (total_revenue > total_cost)

        return is_arb, sell_price_unit, trade_volume
    
    def get_midprice(self, order_depth: OrderDepth) -> float:
        """Compute the mid-price from an order depth (used for RAINFOREST_RESIN only)."""
        if not order_depth or (not order_depth.buy_orders) or (not order_depth.sell_orders):
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0
    
    def get_mmmid(self, od: OrderDepth, last_price=None) -> float:
        if od.sell_orders and od.buy_orders:
            ba, bb = min(od.sell_orders), max(od.buy_orders)
            fa = [p for p,v in od.sell_orders.items() if abs(v)>=10]
            fb = [p for p,v in od.buy_orders.items()  if abs(v)>=10]
            if fa and fb:
                return (min(fa)+max(fb))/2
            return last_price if last_price is not None else (ba+bb)/2
        return 0.0

    def fast_base_iv(self, m, v, timestamp, refit_interval=10):
        self.smile_fit_buffer.append((m, v))
        if len(self.smile_fit_buffer) < 5:
            return None

        if timestamp - self.last_fit_time >= refit_interval:
            m_arr, v_arr = zip(*self.smile_fit_buffer)
            coeffs = np.polyfit(m_arr, v_arr, 2)
            self.last_fit_poly = np.poly1d(coeffs)
            self.last_fit_time = timestamp

        return self.last_fit_poly(0) if self.last_fit_poly is not None else None

    def get_swmid_v(self, order_depth: OrderDepth) -> float:
        """Synthetic weighted mid-price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def ATM_voucher_orders(self, product, order_depth, position, traderObject, timestamp):
        orders = []

        # Time to expiry (TTE)
        tte = 4/365 - timestamp / (365 * 1e6)

        call_price = self.get_swmid_v(order_depth[product])
        spot = self.get_swmid_v(order_depth[Product.VOLCANIC_ROCK])

        if call_price is None or spot is None:
            return orders

        K = ATM_VOUCHERS[product]

        if product not in traderObject["IVs"]:
            traderObject["IVs"][product] = []

        # Calculate current implied volatility
        IV = BlackScholes.implied_volatility(call_price, spot, K, tte)
        traderObject["IVs"][product].append(IV)

        if len(traderObject["IVs"][product]) < LONG_WINDOW:
            return orders

        # Maintain the rolling window
        if len(traderObject["IVs"][product]) > LONG_WINDOW:
            traderObject["IVs"][product].pop(0)

        # Compute short and long moving averages
        short_ma = np.mean(traderObject["IVs"][product][-SHORT_WINDOW:])
        long_ma = np.mean(traderObject["IVs"][product])

        diff = IV - short_ma
        print(f"IV: {IV:.5f}, SMA100: {short_ma:.5f}, SMA200: {long_ma:.5f}, Spread: {diff:.5f}")

        trade_signal = 0

        # Mean Reversion Strategy
        if diff > IV_DEVIATION_THRESHOLD[product]:
            print("IV above long MA — SELL signal")
            trade_signal = -1
        elif diff < -IV_DEVIATION_THRESHOLD[product]:
            print("IV below long MA — BUY signal")
            trade_signal = 1

        if trade_signal:
            if len(order_depth[product].buy_orders.keys()) == 0 or len(order_depth[product].sell_orders.keys()) == 0:
                return orders

            best_bid = max(order_depth[product].buy_orders.keys())
            best_ask = min(order_depth[product].sell_orders.keys())
            best_bid_vol = order_depth[product].buy_orders[best_bid]
            best_ask_vol = order_depth[product].sell_orders[best_ask]

            price, vol = ((best_ask, -best_ask_vol) if trade_signal == 1 else (best_bid, -best_bid_vol))
            print(f"Placing order: {'BUY' if trade_signal == 1 else 'SELL'} {abs(vol)} @ {price}")
            orders.append(Order(product, price, vol))
            self.position += [vol]

            
        return orders
    

    def create_orders(self, product: str, pos: int, tgt: int, od: OrderDepth):
        if tgt == pos:
            return None
        qty = abs(tgt-pos)
        if tgt > pos:
            pa = min(od.sell_orders)
            avail = abs(od.sell_orders[pa])
            q = min(qty, avail)
            return [Order(product, pa+1, q)]
        else:
            pb = max(od.buy_orders)
            avail = abs(od.buy_orders[pb])
            q = min(qty, avail)
            return [Order(product, pb-1, -q)]

    def pair_orders(self, p1: str, p2: str, od: Dict[str,OrderDepth], pos: Dict[str,int], tobj: Dict[str,Any]):
        th = 10
        if p1 not in od or p2 not in od:
            return None
        last1 = tobj.get(f"{p1}_last_price", None)
        last2 = tobj.get(f"{p2}_last_price", None)
        m1 = (self.get_mmmid(od[p1], last1)-2000)/5
        m2 = -(self.get_mmmid(od[p2], last2)-2000)
        tobj[f"{p1}_last_price"] = m1
        tobj[f"{p2}_last_price"] = m2
        spread = m1 - m2
        z = (spread - 18.0)/1.0
        res = {}
        if z >= th:
            if pos.get(p1,0) != -50:
                res[p1] = self.create_orders(p1, pos.get(p1,0), -50, od[p1])
        elif z <= -th:
            if pos.get(p1,0) != 50:
                res[p1] = self.create_orders(p1, pos.get(p1,0), 50, od[p1])
        return res or None

    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        # Sell side
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Buy side
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """(Unchanged) Symmetrical market making. Works for all products."""
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int,int):
        """(Unchanged) For KELP/SQUID_INK. RAINFOREST_RESIN calls different code below."""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Filter out orders that do not meet the 'adverse_volume' threshold
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]

            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # If we can't find both a "meaningful" ask and bid, fallback
            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # Incorporate reversion logic if we have a previous price
            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None
        
    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if 'squid_ink_mm_midprices' not in traderObject:
            traderObject['squid_ink_mm_midprices'] = []
        if 'squid_ink_min_spreads' not in traderObject:
            traderObject['squid_ink_min_spreads'] = []
        if 'squid_ink_max_spreads' not in traderObject:
            traderObject['squid_ink_max_spreads'] = []
        
        # Calculate current market maker mid price
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            min_spread = best_ask - best_bid
            
            # Calculate max spread (widest spread in the book)
            max_ask = max(order_depth.sell_orders.keys())
            min_bid = min(order_depth.buy_orders.keys())
            max_spread = max_ask - min_bid
            
            # Filter for large orders (potential market makers)
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("squid_ink_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid-1) // 2 
                else:
                    mmmid_price = traderObject["squid_ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid-1) // 2
            
            # Store spreads
            traderObject['squid_ink_min_spreads'].append(min_spread)
            traderObject['squid_ink_max_spreads'].append(max_spread)
            
            # Keep only last 15 values
            if len(traderObject['squid_ink_min_spreads']) > 4:
                traderObject['squid_ink_min_spreads'] = traderObject['squid_ink_min_spreads'][-4:]
            if len(traderObject['squid_ink_max_spreads']) > 4:
                traderObject['squid_ink_max_spreads'] = traderObject['squid_ink_max_spreads'][-4:]
            
            # Store mm mid price
            traderObject['squid_ink_mm_midprices'].append(mmmid_price)
            if len(traderObject['squid_ink_mm_midprices']) > 4:
                traderObject['squid_ink_mm_midprices'] = traderObject['squid_ink_mm_midprices'][-4:]
            
            # Calculate average spreads
            avg_min_spread = sum(traderObject['squid_ink_min_spreads']) / len(traderObject['squid_ink_min_spreads'])
            avg_max_spread = sum(traderObject['squid_ink_max_spreads']) / len(traderObject['squid_ink_max_spreads'])
            traderObject['avg_min_spread'] = avg_min_spread
            traderObject['avg_max_spread'] = avg_max_spread
            if len(traderObject['squid_ink_mm_midprices']) >= 3:
                x = list(range(len(traderObject['squid_ink_mm_midprices'])))
                y = traderObject['squid_ink_mm_midprices']
                x_mean = sum(x) / len(x)
                y_mean = sum(y) / len(y)
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
                
                if denominator != 0:
                    slope = numerator / denominator
                else:
                    slope = 0
                
                # Update normalization to allow for both positive and negative slopes
                # Keep the sign of the slope, but normalize its magnitude
                slope_sign = 1 if slope > 0 else -1
                slope_magnitude = abs(slope)
                # Scale magnitude to a reasonable range (0.02 to 0.3)
                normalized_magnitude = min(0.3, max(0.02, 0.1 * (slope_magnitude / 100)))
                # Apply the original sign to the normalized magnitude
                norm_slope = slope_sign * normalized_magnitude
                
                traderObject['dynamic_reversion_beta'] = norm_slope
                if traderObject.get("squid_ink_last_price", None) is not None:
                    last_price = traderObject["squid_ink_last_price"]
                    last_returns = (mmmid_price - last_price) / last_price if last_price > 0 else 0
                    pred_returns = last_returns * norm_slope
                    fair = mmmid_price + (mmmid_price * pred_returns)
                else:
                    fair = mmmid_price
            else:
                if traderObject.get("squid_ink_last_price", None) is not None:
                    last_price = traderObject["squid_ink_last_price"]
                    last_returns = (mmmid_price - last_price) / last_price if last_price > 0 else 0
                    pred_returns = last_returns * -0.1
                    fair = mmmid_price + (mmmid_price * pred_returns)
                else:
                    fair = mmmid_price
                    
            # Store last price for next iteration
            traderObject["squid_ink_last_price"] = mmmid_price
            return fair

    def sma(self, order_depth: OrderDepth, traderObject, position: int):
        orders: List[Order] = []
        product = Product.SQUID_INK
        position_limit = self.LIMIT[product]

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        params = self.params[product]
        take_width = params["take_width"]
        av_len = params["averaging_length"]
        trigger_price = params["trigger_price"]
        max_quantity = params["max_order_quantity"]
        volatility_threshold = params["volatility_threshold"]

        if "squid_prices" not in traderObject:
            traderObject["squid_prices"] = []
        traderObject["squid_prices"].append(mid_price)
        if len(traderObject["squid_prices"]) > av_len:
            traderObject["squid_prices"].pop(0)

        if len(traderObject["squid_prices"]) < 3:
            return orders

        mean_price = sum(traderObject["squid_prices"]) / len(traderObject["squid_prices"])
        squared_diffs = [(p - mean_price) ** 2 for p in traderObject["squid_prices"]]
        volatility = (sum(squared_diffs) / len(traderObject["squid_prices"])) ** 0.5

        bid_volume = abs(order_depth.buy_orders[best_bid])
        ask_volume = abs(order_depth.sell_orders[best_ask])
        avg_volume = (bid_volume + ask_volume) / 2
        quantity = min(int(avg_volume), max_quantity)

        def safe_order(price, qty):
            new_position = position + sum(order.quantity for order in orders) + qty
            if abs(new_position) > position_limit:
                allowed_qty = position_limit - abs(position) if qty > 0 else -(position + sum(o.quantity for o in orders))
                qty = max(min(qty, allowed_qty), -allowed_qty)
            if qty != 0:
                orders.append(Order(product, price, qty))

        # Mean-Reversion if volatility is high
        if volatility > volatility_threshold:
            sma_val = mean_price
            diff = mid_price - sma_val
            if diff >= trigger_price and position > -position_limit:
                safe_order(best_bid - take_width, -quantity)
            elif diff <= -trigger_price and position < position_limit:
                safe_order(best_ask + take_width, quantity)

        else:
            # Market making
            print("MARKETMAKING")
            squid_ink_fair_value = self.squid_ink_fair_value(order_depth, traderObject)
            take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                order_depth,
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["take_width"],
                position,
                self.params[Product.SQUID_INK]["prevent_adverse"],
                self.params[Product.SQUID_INK]["adverse_volume"],
            )
            clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                order_depth,
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["clear_width"],
                position,
                buy_order_volume,
                sell_order_volume,
            )
            make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                order_depth,
                squid_ink_fair_value,
                position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
                traderObject=traderObject
            )
            for o in take_orders + clear_orders + make_orders:
                safe_order(o.price, o.quantity)

        return orders

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):

        if product == Product.RAINFOREST_RESIN:
            orders = []
            buy_order_volume = 0
            sell_order_volume = 0

            # Attempt to buy if best ask is significantly below fair_value
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -order_depth.sell_orders[best_ask]
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, self.LIMIT[product] - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity

            # Attempt to sell if best bid is significantly above fair_value
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, self.LIMIT[product] + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity

            return orders, buy_order_volume, sell_order_volume

        else:
            orders: List[Order] = []
            buy_order_volume = 0
            sell_order_volume = 0

            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                prevent_adverse,
                adverse_volume,
            )
            return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):

        if product == Product.RAINFOREST_RESIN:
            orders = []
            position_after_take = position + buy_order_volume - sell_order_volume
            buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
            sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

            if position_after_take > 0:
                # net long => see if we can sell near fair_value
                bids_to_hit = {
                    price: vol for price, vol in order_depth.buy_orders.items()
                    if price >= fair_value - clear_width
                }
                if bids_to_hit:
                    best_clear_bid = max(bids_to_hit.keys())
                    available_volume = abs(bids_to_hit[best_clear_bid])
                    qty_to_clear = min(abs(position_after_take), available_volume)
                    sent_quantity = min(sell_quantity, qty_to_clear)
                    if sent_quantity > 0:
                        orders.append(Order(product, best_clear_bid, -sent_quantity))
                        sell_order_volume += sent_quantity

            if position_after_take < 0:
                # net short => see if we can buy near fair_value
                asks_to_hit = {
                    price: vol for price, vol in order_depth.sell_orders.items()
                    if price <= fair_value
                }
                if asks_to_hit:
                    best_clear_ask = min(asks_to_hit.keys())
                    available_volume = abs(asks_to_hit[best_clear_ask])
                    qty_to_clear = min(abs(position_after_take), available_volume)
                    sent_quantity = min(buy_quantity, qty_to_clear)
                    if sent_quantity > 0:
                        orders.append(Order(product, best_clear_ask, sent_quantity))
                        buy_order_volume += sent_quantity

            return orders, buy_order_volume, sell_order_volume

        else:
            orders: List[Order] = []
            buy_order_volume, sell_order_volume = self.clear_position_order(
                product,
                fair_value,
                int(clear_width),
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
            return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  
        join_edge: float,       
        default_edge: float,    
        manage_position: bool = False,
        soft_position_limit: int = 0,
        traderObject: dict = None,
    ) -> (List[Order], int, int):
        """
        For RAINFOREST_RESIN: new "market-making" approach from your updated code.
        For KELP/SQUID_INK: original approach from your old code.
        """
        if product == Product.RAINFOREST_RESIN:
            orders = []
            volume_limit = self.params[product]["volume_limit"]

            # best ask above fair_value+1
            asks_above = [
                price for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
            best_ask_above_fair = min(asks_above) if asks_above else (fair_value + 1)

            # best bid below fair_value-1
            bids_below = [
                price for price in order_depth.buy_orders.keys()
                if price < fair_value - 1
            ]
            best_bid_below_fair = max(bids_below) if bids_below else (fair_value - 1)

            # If best ask is too close, push higher (if position within limit)
            if best_ask_above_fair <= fair_value + 2:
                if position <= volume_limit:
                    best_ask_above_fair = fair_value + 3

            # If best bid is too close, push lower (if position within limit)
            if best_bid_below_fair >= fair_value - 2:
                if position >= -volume_limit:
                    best_bid_below_fair = fair_value - 3

            # We use your usual 'market_make' for symmetrical orders:
            bid_price = best_bid_below_fair + 1
            ask_price = best_ask_above_fair - 1
            buy_order_volume, sell_order_volume = self.market_make(
                product,
                orders,
                bid_price,
                ask_price,
                position,
                buy_order_volume,
                sell_order_volume,
            )
            return orders, buy_order_volume, sell_order_volume

        else:
            orders: List[Order] = []

            # For SQUID_INK, check max_spread condition and adjust bid/ask based on slope sign
            if product == Product.SQUID_INK and traderObject is not None:
                # Get current max_spread and average max_spread
                current_max_spread = traderObject.get('squid_ink_max_spreads', [])[-1] if traderObject.get('squid_ink_max_spreads', []) else 0
                avg_max_spread = traderObject.get('avg_max_spread', 0)
                
                # Get current min_spread
                current_min_spread = traderObject.get('squid_ink_min_spreads', [])[-1] if traderObject.get('squid_ink_min_spreads', []) else 0
                
                # Get slope sign
                slope = traderObject.get('dynamic_reversion_beta', 0)
                positive_slope = slope > 0
                
                # Skip market making if max_spread is too large compared to average
                if current_max_spread > avg_max_spread + 2:
                    return [], buy_order_volume, sell_order_volume

            asks_above_fair = [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + disregard_edge
            ]
            bids_below_fair = [
                price
                for price in order_depth.buy_orders.keys()
                if price < fair_value - disregard_edge
            ]

            best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
            best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

            ask = round(fair_value + default_edge)
            if best_ask_above_fair is not None:
                if abs(best_ask_above_fair - fair_value) <= join_edge:
                    ask = best_ask_above_fair
                else:
                    ask = best_ask_above_fair - 1

            bid = round(fair_value - default_edge)
            if best_bid_below_fair is not None:
                if abs(fair_value - best_bid_below_fair) <= join_edge:
                    bid = best_bid_below_fair
                else:
                    bid = best_bid_below_fair + 1

            # SQUID_INK additional check
            if product == Product.SQUID_INK and traderObject is not None and best_bid_below_fair is not None and best_ask_above_fair is not None:
                current_min_spread = traderObject.get('squid_ink_min_spreads', [])[-1] if traderObject.get('squid_ink_min_spreads', []) else 0
                slope = traderObject.get('dynamic_reversion_beta', 0)
                
                if current_min_spread >= 2:
                    if slope > 0:  # Positive slope
                        if abs(best_ask_above_fair - fair_value) <= join_edge:
                            ask = best_ask_above_fair
                        else:
                            ask = best_ask_above_fair
                        if abs(fair_value - best_bid_below_fair) <= join_edge:
                            bid = best_bid_below_fair
                        else:
                            bid = best_bid_below_fair + 1
                    elif slope < 0:  # Negative slope
                        if abs(best_ask_above_fair - fair_value) <= join_edge:
                            ask = best_ask_above_fair
                        else:
                            ask = best_ask_above_fair -1
                        if abs(fair_value - best_bid_below_fair) <= join_edge:
                            bid = best_bid_below_fair
                        else:
                            bid = best_bid_below_fair 
                else:
                    return [], buy_order_volume, sell_order_volume


            if manage_position:
                if position > soft_position_limit:
                    ask -= 1
                elif position < -1 * soft_position_limit:
                    bid += 1

            buy_order_volume, sell_order_volume = self.market_make(
                product,
                orders,
                bid,
                ask,
                position,
                buy_order_volume,
                sell_order_volume,
            )
            return orders, buy_order_volume, sell_order_volume

    def squid_ink_orders(self, order_depth: OrderDepth, traderObject, position: int) -> List[Order]:
        orders: List[Order] = []
        product = Product.SQUID_INK
        params = PARAMS[product]
        position_limit = self.LIMIT[product]

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        # --- Market Data ---
        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        mid_price = (best_ask + best_bid) / 2
        fair_value = params["fair_value"]

        # --- Parameters ---
        take_width = params["take_width"]
        av_len = params["averaging_length"]
        max_quantity = params["max_order_quantity"]
        volatility_threshold = params["volatility_threshold"]
        z_trigger = params["z_trigger"]
        min_quantity = max(1, max_quantity // 4)

        # --- Historical Tracking ---
        if "squid_prices" not in traderObject:
            traderObject["squid_prices"] = []

        prices = traderObject["squid_prices"]
        prices.append(mid_price)
        if len(prices) > av_len:
            prices.pop(0)
        if len(prices) < av_len:
            return orders
        

        # --- Volatility & Z-Score ---
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        volatility = variance ** 0.5
        zscore = (mid_price - mean_price) / volatility if volatility else 0

        # --- Volume Estimation ---
        bid_volume = abs(order_depth.buy_orders[best_bid])
        ask_volume = abs(order_depth.sell_orders[best_ask])
        avg_volume = (bid_volume + ask_volume) / 2
        base_quantity = min(int(avg_volume), max_quantity)

        # --- Adaptive Position Control ---
        distance_from_anchor = abs(mid_price - fair_value)
        anchor_proximity = max(0, 1 - (distance_from_anchor / 100))  # Closer to 1 near 2000
        adjusted_limit = int(position_limit * anchor_proximity)
        adjusted_limit = max(adjusted_limit, 10)  # Always allow small trades

        # --- Helper: Clamped Order ---
        def safe_order(price: float, qty: int):
            new_position = position + sum(o.quantity for o in orders) + qty
            if abs(new_position) > position_limit:
                allowed_qty = position_limit - abs(position) if qty > 0 else -(position + sum(o.quantity for o in orders))
                qty = max(min(qty, allowed_qty), -allowed_qty)
            if qty != 0:
                orders.append(Order(product, price, qty))

        # --- Mean Reversion Trading ---
        if  abs(zscore) >= z_trigger:
            print("MEAN REVERSION")
            scale = min(abs(zscore), 3)
            dynamic_qty = max(min(int(scale * min(base_quantity, adjusted_limit)), max_quantity), min_quantity)
            price_offset = round(take_width * scale)

            if zscore > 0 and position > -adjusted_limit:
                safe_order(best_bid - price_offset, -dynamic_qty)
            elif zscore < 0 and position < adjusted_limit:
                safe_order(best_ask + price_offset, dynamic_qty)

        # --- Low-Volatility: Market Making Mode ---
        else:
            print("MARKETMAKING")
            fv = self.squid_ink_fair_value(order_depth, traderObject)

            take_orders, bvol, svol = self.take_orders(
                product, order_depth, fv,
                self.params[product]["take_width"], position,
                self.params[product]["prevent_adverse"],
                self.params[product]["adverse_volume"]
            )
            clear_orders, bvol, svol = self.clear_orders(
                product, order_depth, fv,
                self.params[product]["clear_width"], position,
                bvol, svol
            )
            make_orders, _, _ = self.make_orders(
                product, order_depth, fv, position, bvol, svol,
                self.params[product]["disregard_edge"],
                self.params[product]["join_edge"],
                self.params[product]["default_edge"],
                traderObject=traderObject
            )
            orders +=  take_orders + clear_orders + make_orders
        return orders

    ##########################################################################
    # Finally, the run method: same overall structure, new RAINFOREST_RESIN logic
    ##########################################################################

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

        #if len(history) > 100:
        #    history.pop(0)
        #mean = np.mean(history)
        #std = np.std(history) if len(history) > 10 else 1
        #zscore = (mid_price - mean) / std
        
        if len(history) > 100:
            history.pop(0)
        if len(history) > 40:
            mean = np.mean(history)
            std = np.std(history)
            zscore = (mid_price - mean) / std if std > 0 else (mid_price - mean)
        
        if len(history) <= 40: zscore = 0


        if product == "VOLCANIC_ROCK":
            POSITION_LIMIT = 400  
            amount = 400
        else : 
            POSITION_LIMIT = 200  
            amount = 200

        # Determine trade action based on zscore
        if zscore > 1.86 and position > -POSITION_LIMIT:
            # Sell at best_bid
            best_bid = max(order_depth.buy_orders.keys())
            # Sell up to 100 units or whatever is allowed to go short
            quantity = -min(amount, POSITION_LIMIT + position)
            orders.append(Order(product, best_bid, quantity))

        elif zscore < -1.86 and position < POSITION_LIMIT:
            # Buy at best_ask
            best_ask = min(order_depth.sell_orders.keys())
            # Buy up to 100 units or whatever is allowed to go long
            quantity = min(amount, POSITION_LIMIT - position)
            orders.append(Order(product, best_ask, quantity))

        # If z-score is exactly Z_EXIT (i.e. 0) and we have a position, attempt to flatten
        # NOTE: floating-point comparisons to 0 can be tricky. This logic is from original code.
        if abs(zscore) == 0 and position != 0:
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
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "IVs" not in traderObject:
            traderObject["IVs"] = {}

        result: Dict[str, List[Order]] = {}

        # --- ATM Voucher Orders ---
        for product in ATM_VOUCHERS:
            if product in state.order_depths:
                orders = self.ATM_voucher_orders(
                    product,
                    state.order_depths,
                    state.position.get(product, 0),
                    traderObject,
                    state.timestamp,
                )
                if orders:
                    result[product] = orders

        # --- KELP Strategy ---
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)

            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # --- RAINFOREST_RESIN Strategy ---
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            rainforest_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                rainforest_position,
            )
            rainforest_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
            )
            rainforest_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
                disregard_edge=0,
                join_edge=0,
                default_edge=0,
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_take_orders + rainforest_clear_orders + rainforest_make_orders
            )

        local_pos = { sym: state.position.get(sym, 0)
                    for sym in state.order_depths }

        # Shadow Olivia’s market trades
        for symbol, trades in state.market_trades.items():
            od = state.order_depths.get(symbol)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            best_ask = min(od.sell_orders.keys())
            best_bid = max(od.buy_orders.keys())
            limit    = self.LIMIT.get(symbol, 0)
            pos      = local_pos[symbol]
            take_profit_threshold = self.take_profit[symbol] if symbol in self.take_profit.keys() else 0
            
            for t in trades:
                # If Olivia bought, go max long
                if t.buyer == "Olivia" and symbol != "KELP":
                    self.liv_trade = True
                    qty = limit - pos
                    if qty > 0:
                        logger.print(f"Olivia BOUGHT {symbol}@{t.price}, going long {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_ask, qty)
                        )
                        pos += qty
                        self.entry_positions_olivia_trading[symbol] = t.price

                # If Olivia sold, go max short
                elif t.seller == "Olivia" and symbol != "KELP":
                    self.liv_trade = True
                    qty = limit + pos
                    if qty > 0:
                        logger.print(f"Olivia SOLD {symbol}@{t.price}, going short {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_bid, -qty)
                        )
                        pos -= qty
                        self.entry_positions_olivia_trading[symbol] = t.price
                        
                # Take profit logic: Check if price has moved enough to exit the position
                if symbol in self.entry_positions_olivia_trading:
                    entry_price = self.entry_positions_olivia_trading[symbol]

                    # For long positions (Olivia bought), take profit if price increases by the threshold
                    if pos > 0 and t.price >= entry_price  + take_profit_threshold:
                        self.liv_trade = False
                        qty_to_sell = pos
                        logger.print(f"Take profit triggered for {symbol} at {t.price}, selling {qty_to_sell} units")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_bid, -qty_to_sell)
                        )
                        pos -= qty_to_sell  # Reset the position after taking profit

                    # For short positions (Olivia sold), take profit if price decreases by the threshold
                    elif pos < 0 and t.price <= entry_price - take_profit_threshold:
                        self.liv_trade = False
                        qty_to_buy = -pos
                        logger.print(f"Take profit triggered for {symbol} at {t.price}, buying {qty_to_buy} units")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_ask, qty_to_buy)
                        )
                        pos += qty_to_buy  # Reset the position after taking profit

            local_pos[symbol] = pos        
        
        # --- SQUID_INK Strategy ---
        if self.liv_trade:
            pass
        else:
            spot = traderObject.setdefault("spot", {})
            pair = self.pair_orders(Product.SQUID_INK, Product.KELP,
                                    state.order_depths, state.position, spot)
            if pair and Product.SQUID_INK in pair:
                result[Product.SQUID_INK] = pair[Product.SQUID_INK]


        # --- BASKET Spread Strategy ---
        orders: Dict[str, List[Order]] = {
            Product.JAMS: [],
            Product.DJEMBES: [],
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
        }

        od = state.order_depths
        pos = state.position

        # Get the mid prices for the products and baskets
        required_basket = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.DJEMBES, Product.CROISSANTS, Product.JAMS]
        if all(p in od for p in required_basket):
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
            #mean = 36.323
            #stdev = 199.2
            #z = (spread - mean) / stdev if stdev > 0 else 0
            # Define order quantity for simplicity
            qty = 1

            # Entry condition: If z-score exceeds the entry threshold, we enter a trade
            if z > self.z_entry:
                orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), -2 * qty))
                orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), 1*qty))
                orders[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), qty))
                orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), -qty))

            # Exit condition: If z-score falls below the exit threshold, we exit the trade
            elif z < -self.z_entry:
                orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), 2 * qty))
                orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), -qty))
                orders[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), -qty))
                orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), qty))
            
            result[Product.PICNIC_BASKET2] = orders[Product.PICNIC_BASKET2]
            result[Product.PICNIC_BASKET1] = orders[Product.PICNIC_BASKET1]
            result[Product.JAMS] = orders[Product.JAMS]
            result[Product.DJEMBES] = orders[Product.DJEMBES]


        if Product.VOLCANIC_ROCK in state.order_depths:
            vr_position = state.position.get(Product.VOLCANIC_ROCK, 0)
            vr_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            vr_orders = self.mean_reversion_orders(
                Product.VOLCANIC_ROCK,
                vr_order_depth,
                vr_position,
                traderObject
            )
            if vr_orders:
                result[Product.VOLCANIC_ROCK] = vr_orders
            vr_position = state.position.get(Product.VOLCANIC_ROCK_VOUCHER_9500, 0)
            vr_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_9500]
            vr_orders = self.mean_reversion_orders(
                Product.VOLCANIC_ROCK_VOUCHER_9500,
                vr_order_depth,
                vr_position,
                traderObject
            )
            if vr_orders:
                result[Product.VOLCANIC_ROCK_VOUCHER_9500] = vr_orders

        if Product.MACARONS in state.order_depths:
            is_arb, sell_price, vol = self.determine_arbitrage_opportunity(state)
            od_local       = state.order_depths[Product.MACARONS]
            best_bid_local = max(od_local.buy_orders.keys(), default=0)

            if is_arb and vol > 0:
                price = best_bid_local if best_bid_local > sell_price else sell_price
                # append into result (will create the list if missing)
                result.setdefault(Product.MACARONS, []).append(
                    Order(Product.MACARONS, price, -vol)
                )
        conversions = 1
        # now proceed with your existing flush & return
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData