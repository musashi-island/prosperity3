from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List
import jsonpickle
import numpy as np
import json
from math import log, sqrt, exp
from statistics import NormalDist


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
VOUCHERS = {"VOLCANIC_ROCK_VOUCHER_9500" : 9500,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
            "VOLCANIC_ROCK_VOUCHER_9750" : 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250}

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
AVERAGING_LENGTH = 200
SHORT_WINDOW = 65
IV_DEVIATION_THRESHOLD = {
                        "VOLCANIC_ROCK_VOUCHER_9750" : 0.004,  
                        "VOLCANIC_ROCK_VOUCHER_10000": 0.003,
                          }  
DELTA_PERIOD = 200
DAY = 3

class Trader:
    def __init__(self):
        self.position = {
            "VOLCANIC_ROCK" : 0,
            "VOLCANIC_ROCK_VOUCHER_9500" : 0,
            "VOLCANIC_ROCK_VOUCHER_10500": 0,
            "VOLCANIC_ROCK_VOUCHER_9750" : 0,
            "VOLCANIC_ROCK_VOUCHER_10000": 0,
            "VOLCANIC_ROCK_VOUCHER_10250": 0}
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
    
    def ATM_voucher_orders(self, product, order_depth, position, traderObject, timestamp):
        orders = []
        underlying = Product.VOLCANIC_ROCK

        # Key checks to avoid KeyErrors
        if product not in order_depth or \
        Product.VOLCANIC_ROCK not in order_depth or \
        product not in ATM_VOUCHERS or \
        product not in IV_DEVIATION_THRESHOLD:
            return orders

        tte = DAY / 365 - timestamp / (365 * 1e6)

        call_price = self.get_swmid(order_depth[product])
        spot = self.get_swmid(order_depth[Product.VOLCANIC_ROCK])

        if call_price is None or spot is None:
            return orders

        K = ATM_VOUCHERS[product]

        if "IVs" not in traderObject:
            traderObject["IVs"] = {}

        if product not in traderObject["IVs"]:
            traderObject["IVs"][product] = []

        IV = BlackScholes.implied_volatility(call_price, spot, K, tte)
        traderObject["IVs"][product].append(IV)
        if len(traderObject["IVs"][product]) < SHORT_WINDOW:
            return orders

        if len(traderObject["IVs"][product]) > SHORT_WINDOW:
            traderObject["IVs"][product].pop(0)
        delta = BlackScholes.delta(spot,K,tte,IV)
        print(f'The delta of {product} is: {delta}')
        short_ma = np.mean(traderObject["IVs"][product])

        diff = IV - short_ma

        trade_signal = 0

        if diff > IV_DEVIATION_THRESHOLD[product]:
            trade_signal = -1
        elif diff < -IV_DEVIATION_THRESHOLD[product]:
            trade_signal = 1

        if trade_signal:
            buy_orders = order_depth[product].buy_orders if product in order_depth else {}
            sell_orders = order_depth[product].sell_orders if product in order_depth else {}

            if not buy_orders or not sell_orders:
                return orders

            best_bid = max(buy_orders.keys())
            best_ask = min(sell_orders.keys())
            best_bid_vol = buy_orders[best_bid]
            best_ask_vol = sell_orders[best_ask]

            price, vol = ((best_ask, -best_ask_vol) if trade_signal == 1 else (best_bid, -best_bid_vol))
            orders.append(Order(product, price, vol))
            
            maximum = POSITION_LIMIT if position+vol>0 else -POSITION_LIMIT
            if abs(position+vol) > POSITION_LIMIT:
                self.position[product] = maximum
            else:
                self.position[product] = position+vol

        return orders


    def delta_neutral(self, order_depth, timestamp, position):
        orders = []
        portfolio_del = 0

        if Product.VOLCANIC_ROCK not in order_depth:
            return orders

        spot = self.get_swmid(order_depth[Product.VOLCANIC_ROCK])
        tte = DAY / 365 - timestamp / (365 * 1e6)

        if spot is None:
            return orders


        for voucher in VOUCHERS:
            if voucher not in order_depth:
                continue

            current_voucher_order = self.position.get(voucher, 0)
            k = VOUCHERS[voucher]
            call_price = self.get_swmid(order_depth[voucher])

            if call_price is None:
                continue

            IV = BlackScholes.implied_volatility(call_price, spot, k, tte)
            delta = BlackScholes.delta(spot, k, tte, IV)
            portfolio_del += delta * current_voucher_order

        print("total delta", portfolio_del)
        buy_orders = order_depth[Product.VOLCANIC_ROCK].buy_orders
        sell_orders = order_depth[Product.VOLCANIC_ROCK].sell_orders

        if not buy_orders or not sell_orders:
            return orders

        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())

        vol = round(-portfolio_del) - position
        print("Hedge volume", vol)
        price = best_ask if vol>0 else best_bid

        orders.append(Order(Product.VOLCANIC_ROCK, price, vol))
        return orders


    def run(self, state: TradingState):
        voucher_result = {}
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "IVs" not in traderObject or "delta_time" not in traderObject:
            traderObject["IVs"] = {}
            traderObject["delta_time"] = DELTA_PERIOD

        for product in ATM_VOUCHERS:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                orders = self.ATM_voucher_orders(product, state.order_depths, position, traderObject, state.timestamp)
                if orders:
                    voucher_result[product] = orders
        
        if not traderObject["delta_time"]:
            position = state.position.get(Product.VOLCANIC_ROCK, 0)
            delta_orders = self.delta_neutral( state.order_depths, state.timestamp, position)
            if delta_orders:
                voucher_result[Product.VOLCANIC_ROCK] = delta_orders
            traderObject["delta_time"] = DELTA_PERIOD
        else:
            traderObject["delta_time"] -= 1
        
        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        return voucher_result, conversions, traderData

