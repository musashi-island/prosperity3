#!/usr/bin/env python3
# trader_shadow_debug.py

from datamodel import Order, TradingState, OrderDepth, Trade, Logger
import jsonpickle

# Instantiate a module‑level logger
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

            for t in trades:
                # If Olivia bought, go max long
                if t.buyer == "Olivia":
                    qty = limit - pos
                    if qty > 0:
                        logger.print(f"Olivia BOUGHT {symbol}@{t.price}, going long {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_ask, qty)
                        )
                        pos += qty

                # If Olivia sold, go max short
                elif t.seller == "Olivia":
                    qty = limit + pos
                    if qty > 0:
                        logger.print(f"Olivia SOLD {symbol}@{t.price}, going short {qty}")
                        result.setdefault(symbol, []).append(
                            Order(symbol, best_bid, -qty)
                        )
                        pos -= qty

            local_pos[symbol] = pos

        # (You can insert your other strategies here, they will not override Olivia shadow)

        # Persist and flush logs
        traderData = jsonpickle.encode(
            jsonpickle.decode(state.traderData) if state.traderData else {}
        )
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
