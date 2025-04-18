from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List
import jsonpickle
import numpy as np
import json
from math import log, sqrt, exp
from statistics import NormalDist
from logs import Logger
logger = Logger()

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
ATM_VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
}
OTM_VOUCHERS = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
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
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
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

def smile(x):
    return 0.20600028*x**2 + 0.01064571*x + 0.18074582

class Trader:
    def __init__(self):
        self.position = []
        pass

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Synthetic weighted mid-price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    def get_mid_price(self, od: OrderDepth) -> float:
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return 0.5 * (best_bid + best_ask)
    # Constants you might want to define globally
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
    
    def ATM_voucher_orders(self, product, order_depth, position, traderObject, timestamp):
        orders = []

        # Time to expiry (TTE)
        tte = 4/365 - timestamp / (365 * 1e6)

        call_price = self.get_swmid(order_depth[product])
        spot = self.get_swmid(order_depth[Product.VOLCANIC_ROCK])

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


    def run(self, state: TradingState):
        result = {}
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "IVs" not in traderObject:
            traderObject["IVs"] = {}

        for product in ATM_VOUCHERS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                orders = self.ATM_voucher_orders(product, state.order_depths, position, traderObject, state.timestamp)
                if orders:
                    result[product] = orders
        '''
        for product in OTM_VOUCHERS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                orders = self.OTM_voucher_orders(product, state.order_depths, position, traderObject, state.timestamp)
                if orders:
                    result[product] = orders
        '''

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
