# 🏝️ Musashi's Island

> *"You must understand that there is more than one path to the top of the mountain."*  
> — **Miyamoto Musashi**

Welcome to **Musashi's Island**, our submission for the **IMC Prosperity Trading Challenge 2025**.

This project is named in honor of **Miyamoto Musashi** — the undefeated samurai, philosopher, and author of *The Book of Five Rings*. Like Musashi, we combine clarity of mind, adaptability, and discipline in the face of battle — or in our case, the fast-paced world of algorithmic trading.

> *"Perceive that which cannot be seen with the eye."*

Musashi lived by principles of constant improvement, strategy without hesitation, and seeing beyond the immediate. We carry that mindset into our code, our teamwork, and our approach to every market.

---

## 📈 Results

We finished 29th worldwide out of 12,620 teams, and 2nd in France.

![Result](result.png)

---

## 👨‍👩‍👧‍👦 Our Team

We are a group of five students from the Bachelor of Science program at École Polytechnique:

- **Jules Frealle**  
  [GitHub](https://github.com/julesFrealle) • [LinkedIn](https://www.linkedin.com/in/jules-frealle/)

- **Emeric Payer**  
  [GitHub](https://github.com/emeric7287) • [LinkedIn](https://www.linkedin.com/in/emeric-payer-754413265/)

- **Youssef Nakhla**  
  [GitHub](https://github.com/YoussefNakhla) • [LinkedIn](https://www.linkedin.com/in/youssef-nakhla-43963126b/)

- **Mael Kupperschmitt**  
  [GitHub](https://github.com/maelkupp) • [LinkedIn](https://www.linkedin.com/in/mael-kupperschmitt-84b40b294/)

- **Iyad El Khoury**  
  [GitHub](https://github.com/danido05)


---

## 🛠️ Usage

```bash
git clone git@gitlab.binets.fr:emeric.payer/musashi_island.git
code musashi_island
```

---

## 📖 Round Descriptions

### 1️⃣ Round 1

<details>
<summary><b>Round 1 — New Products and Strategy Overview</b></summary>

#### 📦 New Products
- **RAINFOREST_RESIN**: Stable commodity with predictable market behavior.
- **KELP**: Volatile asset showing strong mean-reversion tendencies.
- **SQUID_INK**: Extremely volatile asset characterized by sharp short-term trends.

#### 🛠️ Our Strategy
- **RAINFOREST_RESIN**: Used fixed fair-value market-making around 10,000. Implemented dynamic bid and ask spread adjustments based on live order-book data to manage risk and optimize trades.
- **KELP**: Deployed a mean-reversion strategy with a dynamic fair-value calculation based on order-book depth filtering. Safeguards were implemented to reduce adverse selection from large volume orders.
- **SQUID_INK**: Combined SMA (Simple Moving Averages) with volatility-triggered mean-reversion. Adapted trade aggressiveness dynamically based on short-term volatility and market conditions.

</details>

### 2️⃣ Round 2

<details>
<summary><b>Round 2 — Basket Strategies and Enhancements</b></summary>

#### 📦 New Products
- **CROISSANTS vs JAMS**: Statistical spread trading.
- **PICNIC_BASKET1 & PICNIC_BASKET2 vs DJEMBES**: Synthetic basket arbitrage.

#### 🛠️ Our Strategy
- Developed a z-score-based trading system for both basket strategies, triggering trades dynamically based on statistically significant spread deviations.

#### ✨ Changes to Existing Products
- **RAINFOREST_RESIN**: Refined market-making spreads for faster reaction to order-book shifts.
- **KELP**: Enhanced mean-reversion beta values and tightened adverse volume filtering.
- **SQUID_INK**: Improved volatility indicators and dynamic adjustments for better trend capture.

</details>

### 3️⃣ Round 3

<details>
<summary><b>Round 3 — Volatility Trading Introduction</b></summary>

#### 📦 New Products
- **VOLCANIC_ROCK & Vouchers**: Complex derivatives requiring advanced volatility modeling.

#### 🛠️ Our Strategy
- Built a robust volatility trading system utilizing the Black-Scholes model for implied volatility calculation. Implemented dynamic delta-neutral hedging to reduce exposure and optimize returns.

#### ✨ Changes to Existing Products
- Fine-tuned trading parameters across RAINFOREST_RESIN, KELP, SQUID_INK, and basket strategies for enhanced responsiveness and precision.

</details>

### 4️⃣ Round 4

<details>
<summary><b>Round 4 — Macaron Arbitrage</b></summary>

#### 📦 New Products
- **MAGNIFICENT_MACARONS**: Inter-island arbitrage opportunities.

#### 🛠️ Our Strategy
- Aggressively pursued arbitrage opportunities when transport, tariff, and conversion costs allowed profitable inter-island trades.

#### ✨ Changes to Existing Products
- Conducted further refinements on market-making parameters, volatility thresholds, and basket spread management strategies, significantly enhancing overall performance and risk management.

</details>

### 5️⃣ Round 5

<details>
<summary><b>Round 5 — Final Integrated Strategy</b></summary>

#### 🛠️ Our Final Strategy
- Unified all previously developed strategies into a cohesive, integrated trading system. Enhanced market-making, sophisticated mean-reversion trading, advanced basket arbitrages, volatility-driven options trading, and macaron arbitrage were all executed simultaneously with optimized coordination.

#### ✨ Final Adjustments
- Implemented comprehensive, real-time risk management and advanced position-sizing algorithms across the entire product portfolio, achieving maximum responsiveness and profitability.

</details>

---

> *"From one thing, know ten thousand things."*  

— With discipline and elegance,  
Team MUSASHI
