import json
import jsonpickle
import math
import statistics
import copy
import collections
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

#############################
# Logger
#############################
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        # Use state.traderData if available
        trader_data_state = self.truncate(state.traderData, max_item_length) if hasattr(state, 'traderData') and state.traderData is not None else ""
        print(
            self.to_json(
                [
                    self.compress_state(state, trader_data_state),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            # Access attributes instead of subscripting
            compressed.append([
                listing.symbol,
                listing.product,
                listing.denomination
            ])
        return compressed


    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        # Note: Depending on your observation structure, you may need to adjust these fields.
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,      # For RAINFOREST_RESIN
                observation.sunlightIndex,   # For RAINFOREST_RESIN
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

# Global logger instance
logger = Logger()

#############################
# Product Definition
#############################
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

#############################
# Trader Class: Trades SQUID_INK, KELP, and RAINFOREST_RESIN
#############################
class Trader:
    def __init__(self):
        # Limits for each product
        self.LIMIT = {
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
        }

        #############################
        # SQUID_INK parameters (from first code block)
        #############################
        self.squid_params = {
            "take_width": 5,
            "clear_width": 1,
            "prevent_adverse": False,
            "adverse_volume": 18,
            "reversion_beta": -0.3,
            "disregard_edge": 1,
            "join_edge": 1,
            "default_edge": 1,
        }

        #############################
        # KELP parameters (from second code block)
        #############################
        self.kelp_timespan = 10       # Number of past fair value estimates to smooth over
        self.kelp_take_width = 1
        self.kelp_clear_width = 2
        self.kelp_prices: List[float] = []
        self.kelp_vwap: List[Dict[str, float]] = []

        #############################
        # RAINFOREST_RESIN parameters (from second code block)
        #############################
        self.take_width = 1           # Used in resin order-taking
        self.clear_width = 0          # Used in resin order-clearing (0 means no offset)
        self.fair_value = 10000       # Hard-coded fair value for resin
        self.disregard_edge = 1
        self.join_edge = 2
        self.default_edge = 4
        self.soft_position_limit = 50

    #########################################
    # SQUID_INK Trading Methods (First Code Block)
    #########################################
    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        # Update the order depth
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
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
    ):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
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
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject: dict) -> float:
        if 'squid_ink_mm_midprices' not in traderObject:
            traderObject['squid_ink_mm_midprices'] = []
        if 'squid_ink_min_spreads' not in traderObject:
            traderObject['squid_ink_min_spreads'] = []
        if 'squid_ink_max_spreads' not in traderObject:
            traderObject['squid_ink_max_spreads'] = []

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            min_spread = best_ask - best_bid

            max_ask = max(order_depth.sell_orders.keys())
            min_bid = min(order_depth.buy_orders.keys())
            max_spread = max_ask - min_bid

            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.squid_params["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.squid_params["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("squid_ink_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid - 1) // 2
                else:
                    mmmid_price = traderObject["squid_ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid - 1) // 2

            traderObject['squid_ink_min_spreads'].append(min_spread)
            traderObject['squid_ink_max_spreads'].append(max_spread)
            if len(traderObject['squid_ink_min_spreads']) > 4:
                traderObject['squid_ink_min_spreads'] = traderObject['squid_ink_min_spreads'][-4:]
            if len(traderObject['squid_ink_max_spreads']) > 4:
                traderObject['squid_ink_max_spreads'] = traderObject['squid_ink_max_spreads'][-4:]
            traderObject['squid_ink_mm_midprices'].append(mmmid_price)
            if len(traderObject['squid_ink_mm_midprices']) > 4:
                traderObject['squid_ink_mm_midprices'] = traderObject['squid_ink_mm_midprices'][-4:]
            
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
                slope = numerator / denominator if denominator != 0 else 0

                slope_sign = 1 if slope > 0 else -1
                slope_magnitude = abs(slope)
                normalized_magnitude = min(0.3, max(0.02, 0.1 * (slope_magnitude / 100)))
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
    ):
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
    ):
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
        product: str,
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
        traderObject: dict = None
    ):
        orders: List[Order] = []
        if product == Product.SQUID_INK and traderObject is not None:
            current_max_spread = traderObject.get('squid_ink_max_spreads', [])[-1] if traderObject.get('squid_ink_max_spreads', []) else 0
            avg_max_spread = traderObject.get('avg_max_spread', 0)
            current_min_spread = traderObject.get('squid_ink_min_spreads', [])[-1] if traderObject.get('squid_ink_min_spreads', []) else 0
            slope = traderObject.get('dynamic_reversion_beta', 0)
            if current_max_spread > avg_max_spread + 3:
                return [], buy_order_volume, sell_order_volume

        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair + 1

        if product == Product.SQUID_INK and traderObject is not None and best_bid_below_fair is not None and best_ask_above_fair is not None:
            current_min_spread = traderObject.get('squid_ink_min_spreads', [])[-1] if traderObject.get('squid_ink_min_spreads', []) else 0
            slope = traderObject.get('dynamic_reversion_beta', 0)
            if current_min_spread >= 2:
                if slope > 0:
                    ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair
                    bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair + 1
                elif slope < 0:
                    ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair - 1
                    bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair
            else:
                return [], buy_order_volume, sell_order_volume

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    #########################################
    # KELP Trading Methods (Second Code Block)
    #########################################
    def kelp_fair_value(self, order_depth: OrderDepth, timespan: int) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid
            mmmid_price = (mm_ask + mm_bid) / 2
            self.kelp_prices.append(mmmid_price)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-order_depth.sell_orders[best_ask]) + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
            return mmmid_price
        return None

    def kelp_take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                               order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def kelp_take_best_orders_with_adverse(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                                            order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
                                            adverse_volume: int) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def kelp_take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                         position: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.kelp_take_best_orders_with_adverse(
                product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, adverse_volume
            )
        else:
            buy_order_volume, sell_order_volume = self.kelp_take_best_orders(
                product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume
            )
        return orders, buy_order_volume, sell_order_volume

    def kelp_clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order],
                                  order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def kelp_clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                          position: int, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.kelp_clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def kelp_market_make(self, product: str, orders: List[Order], bid: int, ask: int,
                           position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
        return buy_order_volume, sell_order_volume

    def make_kelp_orders(self, order_depth: OrderDepth, fair_value: float, position: int,
                         buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        ask_prices = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bid_prices = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        best_ask = min(ask_prices) if ask_prices else fair_value + 2
        best_bid = max(bid_prices) if bid_prices else fair_value - 2
        buy_order_volume, sell_order_volume = self.kelp_market_make(
            Product.KELP, orders, best_bid + 1, best_ask - 1, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    #########################################
    # RAINFOREST_RESIN Trading Methods (Second Code Block)
    #########################################
    def resin_take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                               order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def resin_clear_position_order(self, product: str, fair_value: float, width: float, orders: List[Order],
                                   order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def advanced_market_make(self, product: str, fair_value: float, position: int, buy_order_volume: int,
                               sell_order_volume: int, order_depth: OrderDepth, disregard_edge: float,
                               join_edge: float, default_edge: float, soft_position_limit: int) -> List[Order]:
        orders: List[Order] = []
        position_limit = self.LIMIT[product]
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
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
        if position > soft_position_limit:
            ask -= 1  # More aggressive selling
        elif position < -soft_position_limit:
            bid += 1  # More aggressive buying
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
        return orders

    def resin_orders(self, order_depth: OrderDepth, fair_value: float, position: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        # 1) TAKE orders for resin
        buy_order_volume, sell_order_volume = self.resin_take_best_orders(
            product=Product.RAINFOREST_RESIN,
            fair_value=fair_value,
            take_width=self.take_width,
            orders=orders,
            order_depth=order_depth,
            position=position,
            buy_order_volume=buy_order_volume,
            sell_order_volume=sell_order_volume
        )
        # 2) CLEAR orders for resin
        buy_order_volume, sell_order_volume = self.resin_clear_position_order(
            product=Product.RAINFOREST_RESIN,
            fair_value=fair_value,
            width=self.clear_width,
            orders=orders,
            order_depth=order_depth,
            position=position,
            buy_order_volume=buy_order_volume,
            sell_order_volume=sell_order_volume
        )
        # 3) ADVANCED MARKET MAKE for resin
        mm_orders = self.advanced_market_make(
            product=Product.RAINFOREST_RESIN,
            fair_value=fair_value,
            position=position,
            buy_order_volume=buy_order_volume,
            sell_order_volume=sell_order_volume,
            order_depth=order_depth,
            disregard_edge=self.disregard_edge,
            join_edge=self.join_edge,
            default_edge=self.default_edge,
            soft_position_limit=self.soft_position_limit
        )
        orders.extend(mm_orders)
        return orders

    #########################################
    # Combined run() Method
    #########################################
    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}

        # --- Process SQUID_INK ---
        # traderObject is used to store state across calls for squid ink.
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            squid_ink_fair_value = self.squid_ink_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            squid_ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.squid_params["take_width"],
                squid_ink_position,
                self.squid_params["prevent_adverse"],
                self.squid_params["adverse_volume"],
            )
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.squid_params["clear_width"],
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
            )
            squid_ink_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
                self.squid_params["disregard_edge"],
                self.squid_params["join_edge"],
                self.squid_params["default_edge"],
                traderObject=traderObject
            )
            result[Product.SQUID_INK] = squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders

        # --- Process KELP ---
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            od_kelp = state.order_depths[Product.KELP]
            kelp_fair = self.kelp_fair_value(od_kelp, self.kelp_timespan)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.kelp_take_orders(
                Product.KELP, od_kelp, kelp_fair, self.kelp_take_width, kelp_position, True, 20
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.kelp_clear_orders(
                Product.KELP, od_kelp, kelp_fair, self.kelp_clear_width, kelp_position, buy_order_volume, sell_order_volume
            )
            kelp_make_orders, _, _ = self.make_kelp_orders(
                od_kelp, kelp_fair, kelp_position, buy_order_volume, sell_order_volume
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # --- Process RAINFOREST_RESIN ---
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            od_resin = state.order_depths[Product.RAINFOREST_RESIN]
            resin_final_orders = self.resin_orders(order_depth=od_resin, fair_value=self.fair_value, position=resin_position)
            result[Product.RAINFOREST_RESIN] = resin_final_orders

        conversions = 0
        # Optionally, update traderData with the squid ink traderObject state.
        trader_data = ''
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
