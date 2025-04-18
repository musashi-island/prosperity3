#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Any
import jsonpickle
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
            self.compress_state(state,
                                self.truncate(state.traderData, max_item_length)),
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
        import json
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

class Trader:
    def __init__(self):
        self.CSI = 26.06
        self.position_limit = 75
        self.base_size      = 10   # taille nominale des quotes
        self.tick           = 1    # granularité de prix

    def best_bid(self, od: OrderDepth):
        return (max(od.buy_orders), abs(od.buy_orders[max(od.buy_orders)])) \
               if od and od.buy_orders else (None, 0)

    def best_ask(self, od: OrderDepth):
        return (min(od.sell_orders), abs(od.sell_orders[min(od.sell_orders)])) \
               if od and od.sell_orders else (None, 0)

    def fair_value(self, conv_obs):
        import_cost  = conv_obs.askPrice + conv_obs.importTariff + conv_obs.transportFees
        export_value = conv_obs.bidPrice - conv_obs.exportTariff - conv_obs.transportFees
        return 0.5 * (import_cost + export_value)

    def run(self, state: TradingState):
        product   = Product.MACARONS
        od        = state.order_depths.get(product)
        conv      = state.observations.conversionObservations.get(product)
        if conv is None or od is None:
            logger.flush(state, {}, 0, state.traderData)
            return {}, 0, state.traderData

        sunlight  = conv.sunlightIndex
        position  = state.position.get(product, 0)

        orders: Dict[str, List[Order]] = {}
        conversions = 0

        if sunlight < self.CSI:
            bb_price, bb_vol = self.best_bid(od)
            ba_price, ba_vol = self.best_ask(od)

            if position > 0 and bb_price is not None:
                qty = min(position, bb_vol)
                orders[product] = [Order(product, bb_price, -qty)]

            elif position < 0 and ba_price is not None:
                qty = min(-position, ba_vol)
                orders[product] = [Order(product, ba_price, qty)]

            logger.print(f"[STOP] t={state.timestamp} sun={sunlight:.2f} pos={position}")

        else:
            fv = self.fair_value(conv)

            best_bid_px, _ = self.best_bid(od)
            best_ask_px, _ = self.best_ask(od)

            bid_price = (best_bid_px + self.tick) if best_bid_px is not None else math.floor(fv) - self.tick
            ask_price = (best_ask_px - self.tick) if best_ask_px is not None else math.ceil(fv) + self.tick

            skew = int(round(position / self.position_limit))  # -1, 0 ou 1
            bid_price -= skew
            ask_price -= skew

            if bid_price >= ask_price:
                bid_price = ask_price - self.tick

            bid_size = max(1, int(self.base_size * (1 - position / self.position_limit)))
            ask_size = max(1, int(self.base_size * (1 + position / self.position_limit)))

            orders[product] = [
                Order(product, bid_price,  bid_size),
                Order(product, ask_price, -ask_size),
            ]

            logger.print(
                f"[MM] t={state.timestamp} sun={sunlight:.2f} "
                f"FV={fv:.2f} • {bid_size}@{bid_price} / {ask_size}@{ask_price} • pos={position}"
            )

        new_trader_data = jsonpickle.encode({})  # pas d’état persistant pour l’instant
        logger.flush(state, orders, conversions, new_trader_data)
        return orders, conversions, new_trader_data
