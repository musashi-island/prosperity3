from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import *
import jsonpickle
import numpy as np
import json
from json import JSONEncoder
import statistics
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

    def compress_state(self, state: TradingState, trader_data: str) -> list:
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

    def compress_orders(self, orders: Dict[str, List[Order]]) -> list:
        return [[o.symbol, o.price, o.quantity]
                for orders_list in orders.values()
                for o in orders_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.LIMIT = {
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            # etc.
        }
        # Take-profit thresholds (as a percentage of the price movement)
        self.take_profit = {
            "SQUID_INK": 65,  # 1% price movement for SQUID_INK
            "CROISSANTS": 5000,  # 3% price movement for CROISSANTS
        }
        self.entry_positions_olivia_trading = {}

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # Debug header
        logger.print("=== Olivia Shadowing via market_trades ===")

        # Track our local positions so we don't exceed limits within this tick
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
            take_profit_threshold = self.take_profit[symbol] if symbol in self.take_profit.keys() else 0  # Default to 2% if not specified

            for t in trades:
                # If Olivia bought, go max long
                if t.buyer == "Olivia" and symbol != "KELP":
                    qty = limit - pos
                    if qty > 0:
                        logger.print(f"Olivia BOUGHT {symbol}@{t.price}, going long {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_ask, qty)
                        )
                        pos += qty
                        self.entry_positions_olivia_trading[symbol] = t.price  # Store the entry price for the long position

                # If Olivia sold, go max short
                elif t.seller == "Olivia" and symbol != "KELP":
                    qty = limit + pos
                    if qty > 0:
                        logger.print(f"Olivia SOLD {symbol}@{t.price}, going short {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_bid, -qty)
                        )
                        pos -= qty
                        self.entry_positions_olivia_trading[symbol] = t.price  # Store the entry price for the short position

                # Take profit logic: Check if price has moved enough to exit the position
                if symbol in self.entry_positions_olivia_trading:
                    entry_price = self.entry_positions_olivia_trading[symbol]

                    # For long positions (Olivia bought), take profit if price increases by the threshold
                    if pos > 0 and t.price >= entry_price  + take_profit_threshold:
                        qty_to_sell = pos
                        logger.print(f"Take profit triggered for {symbol} at {t.price}, selling {qty_to_sell} units")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_bid, -qty_to_sell)
                        )
                        pos -= qty_to_sell  # Reset the position after taking profit

                    # For short positions (Olivia sold), take profit if price decreases by the threshold
                    elif pos < 0 and t.price <= 1 - take_profit_threshold:
                        qty_to_buy = -pos
                        logger.print(f"Take profit triggered for {symbol} at {t.price}, buying {qty_to_buy} units")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_ask, qty_to_buy)
                        )
                        pos += qty_to_buy  # Reset the position after taking profit

            local_pos[symbol] = pos

        # (You can insert your other strategies here, they will not override Olivia shadow)

        # Persist and flush logs
        traderData = jsonpickle.encode(
            jsonpickle.decode(state.traderData) if state.traderData else {}
        )
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
