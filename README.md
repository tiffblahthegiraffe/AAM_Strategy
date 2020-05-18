# AAM_Trade
Asset Allocation with Momentum quant trading strategy design<br>

**Tribute the team effort to Jinkyu Paik, Jooheon Lee, Wonjun Jung and Henry Chang**

This repo is dedicated to document our designed thought process,back-testing results and the rationale behind the strategy.

### Business Rationale<br>
Our goal is to let risk-adverse/ risk-neutral investor have a strategy that guarantee increase on return in bullish markets and be defensive in berish markets.
- Bullish Market: Take advantage of the upward trend assets by taking momentum strategy
- Bearish Market: Use defense strategy to protect our portfolio by adjusting the weight of bonds investment

### Universe<br>
- Risky Assets (Max 70%)
  - iShares country EFTs excluding Asia countries (Value and Momentum Everywhere - Asness, Moskowitz and Pederson, 2012)
  - SDPR industry ETFs (Do Industries Explain Momentum - Moskowitz and Grinblatt, 1999)
- Safety Assets (min 30%)
  - iShares SHY
- Time Frame
  - 12/2005 - 12/2018

### Strategy Components<br>
By separating our main startegy into two components, we expect we can maximize our return in the bullish market, and we can provide a dense cusion for the portfolio in the bearish market and acquire a higher Sharpe Ratio. 
- Momentum Strategy<br>
  - Select risky assets ETFs classes where the momentum effect exists
  - Assign weight base on momentum score to risky assets ETFs and give the rest to bond ETFs
  - Reconfirm the degree of the momentum effect of each asset classes and allocate assets depending on the scale of the momentum
- Defense Stategy
  - As a base, set at least 30% weight in bond ETF, meaning at most 70% in risky assets ETFs
  - Adjust weight in bond ETF base on the momentum signal from risky assets ETFs


### Repo Assests Description
