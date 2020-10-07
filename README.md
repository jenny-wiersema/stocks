# stocks

The code is run from stock_factor_model.py, with utility functions in stock_utils.py. 

Repo contains code to evaluate risk an equal weighted portfolio, including:
- volatility plots
- portfolio risk envelope plots
- portfolio XSR breakdown

Sector based factors are created to evaluate the risk. 

Assumptions:
- Portfolio consists of assets contained in the SP500 index as of 10/5/20. This introduces a survivorship bias in the portfolio, as only stocks that exist in 2020 are used to construct portfolios starting in 2000. Since this code is used to illustrate a risk backtesting plot and not to estimate a stock trading strategy, this is reasonable. Ideally I would use an estimation universe that was based on what was available at the time, and not what is available now, but I was not able to find that information without it being behind a paywall. 
- Portfolio assumes equal weight of stocks, which assumes weekly rebalancing of positions. I am unable to find actual index weights that are not behind a paywall. This is not ideal but similar to the assumption above, since this code is to illustrate risk framework and not a stock trading strategy it is reasonable for this use case. 
- Specific risk is not included. This will be included in a future update. For a portfolio of this size, the specific risk will contribute a minimal amount. 
- Sector based information is static, as of 10/5/20. In reality sectors can be updated over time, but remain fairly static. I was not able to find this information historically. 