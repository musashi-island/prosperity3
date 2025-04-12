from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK" 
    
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1, # 1
        "clear_width": 0, # 0
        "prevent_adverse": True,
        "adverse_volume": 15, # 15
        "reversion_beta": -0.229, # -0.229
        "disregard_edge": 1, # 1
        "join_edge": 0, # 0
        "default_edge": 1, # 1
    },
    Product.SQUID_INK: {
        "averaging_length" : 300,
        "trigger_price" : 10,
        "take_width": 5, #5
        "clear_width": 1, #1
        "prevent_adverse": False, #False
        "adverse_volume": 18, #16
        "reversion_beta":-0.3, #-0.3
        "disregard_edge": 1, #1
        "join_edge": 1, #1
        "default_edge": 1, #1
        "max_order_quantity": 40,
        "volatility_threshold": 2.5
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK : 50}
        
    def take_best_orders(
            self,
            product : str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0
            ) -> (int,int):
        ''' Will check for the maximal profit order in both buy and sell,
        with a certain margin, respecting the position limit and update the volume 
        after we have made this trade'''
        
        position_limit = self.LIMIT[product]
        
        if len(order_depth.sell_orders) != 0:
            
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(
                            best_ask_amount, position_limit - position
                            )
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                           del order_depth.sell_orders[best_ask]
                               
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
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
            # Aggregate volume from all sell orders with price lower than fair_for_bid
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
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            # If we can't find both a "meaningful" ask and bid, fallback
            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    # fallback to simple midpoint of best bid/ask
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
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
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
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        traderObject: dict = None,
        # will penny all other levels with higher edge
    ):
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
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

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
    def sma(self, order_depth: OrderDepth, traderObject, position: int):
        orders: List[Order] = []
        product = Product.SQUID_INK
        position_limit = self.LIMIT[product]

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        # Extract parameters
        params = PARAMS[product]
        take_width = params["take_width"]
        av_len = params["averaging_length"]
        trigger_price = params["trigger_price"]
        max_quantity = params["max_order_quantity"]
        volatility_threshold = params["volatility_threshold"]

        # Initialize memory if needed
        if "squid_prices" not in traderObject:
            traderObject["squid_prices"] = []

        squid_prices = traderObject["squid_prices"]
        squid_prices.append(mid_price)
        if len(squid_prices) > av_len:
            squid_prices.pop(0)

        if len(squid_prices) < 3:
            return orders

        # --- Volatility Calculation ---
        mean_price = sum(squid_prices) / len(squid_prices)
        squared_diffs = [(p - mean_price) ** 2 for p in squid_prices]
        volatility = (sum(squared_diffs) / len(squid_prices)) ** 0.5

        # Estimate market volume
        bid_volume = abs(order_depth.buy_orders[best_bid])
        ask_volume = abs(order_depth.sell_orders[best_ask])
        avg_volume = (bid_volume + ask_volume) / 2
        quantity = min(int(avg_volume), max_quantity)

        # --- Helper to clamp position within limits ---
        def safe_order(price, qty):
            new_position = position + sum(order.quantity for order in orders) + qty
            if abs(new_position) > position_limit:
                allowed_qty = position_limit - abs(position) if qty > 0 else -(position + sum(o.quantity for o in orders))
                qty = max(min(qty, allowed_qty), -allowed_qty)
            if qty != 0:
                orders.append(Order(product, price, qty))

        # --- Mean Reversion Mode ---
        if volatility > volatility_threshold:
            sma = mean_price
            diff = mid_price - sma

            if diff >= trigger_price and position > -position_limit:
                safe_order(best_bid - take_width, -quantity)

            elif diff <= -trigger_price and position < position_limit:
                safe_order(best_ask + take_width, quantity)

        # --- Market Making Mode ---
        else:
            print("MARKETMAKING")
            squid_ink_fair_value = self.squid_ink_fair_value(
                order_depth, traderObject
            )
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
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
    
        
        # RAINFOREST RESIN
        
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_position = (
                state.position[Product.RAINFOREST_RESIN ]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_position,
                )
            )
            rainforest_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_take_orders + rainforest_clear_orders + rainforest_make_orders
            )

        # KELP

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
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
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )
        
        
        
        # SQUID INK

        squid_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
        result[Product.SQUID_INK] = self.sma(state.order_depths[Product.SQUID_INK], traderObject, squid_position)
        trader_data = "SAMPLE" 
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
    
        return result, conversions, traderData
        