import json
from typing import Dict, List, Tuple, Any
from json import JSONEncoder
import jsonpickle
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation,  UserId, Order
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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



logger = Logger()


Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

class Product:
    DJEMBES = "DJEMBES"

##########################################
# Parameters for Djembes Trading
##########################################
PARAMS_DJEMBE = {
    "zscore_threshold": 3,      # Swing threshold: if abs(Z-score) > 4 => strong reversion signal
    "vol_window": 30,             # Rolling window size for mean/std
    "base_order_size": 15,         # Base size for orders
    "position_limit": 60,         # Max net position
    "volatility_threshold": 4,
    "take_width": 4
}
##########################################
# Trader for djembes
########################################## 
class Trader:
    def __init__(self):
        # You can store params or additional state if needed
        self.z_trigger = 3
        self.volatility_threshold = 3
        self.djembe_data = []
        self.position_ticker = 0  # track time in position
        
    

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):

        """
        Main strategy function, called each tick.

        Steps:
          1) Gather the mid-price from best bid/ask for SQUID_INK.
          2) Append to global history, compute rolling mean & std dev.
          3) Compute Z-score => (current_mid - mean) / std.
          4) If Z-score > +4 => place SELL order at best_bid (contrarian).
             If Z-score < -4 => place BUY order at best_ask (contrarian).
          5) Position-limited to ± 50.
        """
        result: Dict[str, List[Order]] = {}
        conversions = 1
        trader_data = ""



        # Ensure we have SQUID_INK data
        if Product.DJEMBES not in state.order_depths:
            return result, conversions, trader_data

        order_depth: OrderDepth = state.order_depths[Product.DJEMBES]

        # Compute mid-price from best bid & ask
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None and best_ask is None:
            # No market quotes => do nothing
            return result, conversions, trader_data

        if best_bid is not None and best_ask is not None:
            current_mid = 0.5 * (best_bid + best_ask)
        elif best_bid is not None:
            current_mid = float(best_bid)
        elif best_ask is not None:  # best_ask is not None
            current_mid = float(best_ask)
        else:
            return result, conversions, trader_data

        # Update our global price history
        self.djembe_data.append(current_mid)
        # Keep history from exploding
        if len(self.djembe_data) > 2000:
            self.djembe_data = self.djembe_data[-2000:]

        # If we don't yet have enough data, skip
        if len(self.djembe_data) < PARAMS_DJEMBE['vol_window']:
            return result, conversions, trader_data

        # Compute rolling mean & std dev over last vol_window points
        recent_prices = self.djembe_data[PARAMS_DJEMBE['vol_window']:]
        mean_price = np.mean(recent_prices)
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)if len(recent_prices) > 0 else 0
        volatility = variance ** 0.5

        
        bid_volume = sum(list(order_depth.buy_orders.values()))
        ask_volume = sum(list(order_depth.sell_orders.values()))
        total_volume = bid_volume + ask_volume

        
        current_position = state.position.get(Product.DJEMBES, 0)

        if current_position != 0:
            self.position_ticker += 1
        else:
            self.position_ticker = 0

        # --- Helper: Clamped Order ---
        def safe_order(price: float, qty: int):
            new_position = current_position + sum(o.quantity for o in orders) + qty
            if abs(new_position) > PARAMS_DJEMBE['position_limit']:
                allowed_qty = PARAMS_DJEMBE['position_limit'] - abs(current_position) if qty > 0 else -(current_position + sum(o.quantity for o in orders))
                qty = max(min(qty, allowed_qty), -allowed_qty)
            if qty != 0:
                orders.append(Order(Product.DJEMBES, price, qty))

        orders: List[Order] = []

            # Compute Z-score
        zscore = (current_mid - mean_price) / volatility if volatility else 0

        if volatility > self.volatility_threshold and abs(zscore) >= self.z_trigger:
            print("MEAN REVERSION")
            scale = min(abs(zscore), 3)
            dynamic_qty = PARAMS_DJEMBE['base_order_size']
            price_offset = round(PARAMS_DJEMBE['take_width'] * scale)

            if zscore > 0:
                safe_order(best_bid - price_offset, -dynamic_qty)
            elif zscore < 0:
                safe_order(best_ask + price_offset, dynamic_qty)

        exit_ready = abs(zscore) < 1.0 and self.position_ticker > 20
        if exit_ready:
            if current_position > 0 and best_bid:
                # Close long
                qty = min(current_position, order_depth.buy_orders.get(best_bid, current_position))
                if qty > 0:
                    orders.append(Order(Product.DJEMBES, best_bid, -qty))
                    self.position_ticker = 0
            elif current_position < 0 and best_ask:
                # Close short
                qty = min(-current_position, -order_depth.sell_orders.get(best_ask, -current_position))
                if qty > 0:
                    orders.append(Order(Product.DJEMBES, best_ask, qty))
                    self.position_ticker = 0

        #logger.print(f"SQUID_INK t={state.timestamp} mid={current_mid:.2f} mean={mean_price:.2f} std={std_price:.2f} zscore={zscore:.2f} pos={current_position} orders={orders}")
        logger.flush(state, result, conversions, trader_data)
        result[Product.DJEMBES] = orders
        return result, conversions, trader_data