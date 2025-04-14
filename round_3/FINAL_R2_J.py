from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json

from typing import List, Dict, Tuple,Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Dict, List, Tuple, Any
from json import JSONEncoder
import jsonpickle
import statistics

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

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK" 

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0
    },
    Product.KELP: {
        "take_width": 2,            # 1.5
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,       # 15
        "reversion_beta": -0.25,    # -0.229
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
    },
}

class Trader:
    def __init__(self, params=None):
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
        }
        # Store mid-price history for RAINFOREST_RESIN if desired
        self.resin_price_history = {Product.RAINFOREST_RESIN: []}
        self.spread_history: List[float] = []
        self.max_history = 100  # Maximum history size for calculating z-score
        self.z_entry = 1.1 # Entry threshold for z-score
        #self.z_exit = 0.2  # Exit threshold for z-score
        self.spread_history2: List[float] = []
        self.max_history2 = 100  # Maximum history size for calculating z-score
        self.z_entry2 = 1  # Entry threshold for z-score
        
    def get_midprice(self, order_depth: OrderDepth) -> Optional[float]:
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
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # RAINFOREST RESIN
        
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            # Store mid-price if available
            mid_price = self.get_midprice(order_depth)
            if mid_price is not None:
                self.resin_price_history[Product.RAINFOREST_RESIN].append(mid_price)

            # 1) Take orders
            rainforest_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                rainforest_position,
            )
            # 2) Clear orders
            rainforest_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
            )
            # 3) Market-making
            rainforest_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                Product.RAINFOREST_RESIN,
                order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
                # We pass these but they aren't used in the new logic:
                disregard_edge=0,  
                join_edge=0,
                default_edge=0,
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_take_orders + rainforest_clear_orders + rainforest_make_orders
            )

        # KELP

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


        # SQUID INK

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
        # SQUID INK
            squid_position = (
                    state.position[Product.SQUID_INK]
                    if Product.SQUID_INK in state.position
                    else 0
                )
            result[Product.SQUID_INK] = self.squid_ink_orders(state.order_depths[Product.SQUID_INK], traderObject, squid_position)


        #BASKET ARBITRAGE

        orders: Dict[str, List[Order]] = {
            #Product.CROISSANTS: [],
            #Product.JAMS: [],
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

        # Calculate z-score based on historical spreads
        if len(self.spread_history) >= 10:
            #mean = statistics.mean(self.spread_history)
            mean = 27.14675
            stdev = statistics.stdev(self.spread_history)
            z = (spread - mean) / stdev if stdev > 0 else 0
        else:
            z = 0

        # Define order quantity for simplicity
        qty = 1

        # Entry condition: If z-score exceeds the entry threshold, we enter a trade
        if z > self.z_entry:
            orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), 3 * qty))
            orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), -2 * qty))
            orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), 2 * qty))

        # Exit condition: If z-score falls below the exit threshold, we exit the trade
        elif z < -self.z_entry:
            orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, int(b2_mid), -3 * qty))
            orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, int(b1_mid), 2 * qty))
            orders[Product.DJEMBES].append(Order(Product.DJEMBES, int(djembe_mid), -2 * qty))

        # Prepare trader data
        for product in orders:
            if product not in result:
                result[product] = []
            result[product].extend(orders[product])

        ### CROISSANTS JAMS
        orders2: Dict[str, List[Order]] = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        od = state.order_depths
        pos = state.position

        # Get the mid prices for Croissants and Jams
        croissant_mid = self.get_swmid(od[Product.CROISSANTS])
        jam_mid = self.get_swmid(od[Product.JAMS])

        # If any of the mid prices are None, we cannot compute the spread
        if None in [croissant_mid, jam_mid]:
            return {}, 0, ""

        # Calculate the spread between Croissants and Jams
        spread = croissant_mid - jam_mid

        # Append the current spread to the history
        
        self.spread_history2.append(spread)
        if len(self.spread_history2) > self.max_history2:
            self.spread_history2.pop(0)

        # Calculate z-score based on historical spreads
        if len(self.spread_history2) >= 10:
            #mean = statistics.mean(self.spread_history2)
            mean = -2284.4621125
            stdev = statistics.stdev(self.spread_history2)
            z = (spread - mean) / stdev if stdev > 0 else 0
        else:
            z = 0

        # Define order quantity for simplicity
        qty = 1

        # Entry condition: If z-score exceeds the entry threshold, we enter a trade
        if z > self.z_entry2:
            # Short Croissants and Long Jams
            orders2[Product.CROISSANTS].append(Order(Product.CROISSANTS, int(croissant_mid), -1 * qty)) #2
            orders2[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), 1 * qty)) #2

        # Exit condition: If z-score falls below the exit threshold, we exit the trade
        elif z < -self.z_entry2:
            # Exit by reversing the position
            orders2[Product.CROISSANTS].append(Order(Product.CROISSANTS, int(croissant_mid), 1 * qty)) #2
            orders2[Product.JAMS].append(Order(Product.JAMS, int(jam_mid), -1 * qty)) #2

        # Prepare trader data
        for product in orders2:
            if product not in result:
                result[product] = []
            result[product].extend(orders2[product])
        # Final result to match the expected output format


        traderData = jsonpickle.encode(traderObject)
        conversions = 1  # same as your original
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
