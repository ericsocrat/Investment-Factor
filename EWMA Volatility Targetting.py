#region Investment Factor Info and Imports:
# https://quantpedia.com/strategies/investment-factor/
#
# The investment universe consists of all NYSE, Amex, and NASDAQ stocks. Firstly, stocks are allocated to five Size groups (Small to Big) at the end of each June
# using NYSE market cap breakpoints. Stocks are allocated independently to five Investment (Inv) groups (Low to High) still using NYSE breakpoints.
# The intersections of the two sorts produce 25 Size-Inv portfolios. For portfolios formed in June of year t, Inv is the growth of total assets for
# the fiscal year ending in t-1 divided by total assets at the end of t-1. Long portfolio with the highest Size and simultaneously with the lowest
# Investment. Short portfolio with the highest Size and simultaneously with the highest Investment. The portfolios are value-weighted.
#
# QC implementation changes:
#   - Universe consists of top 3000 US stock by market cap from NYSE, AMEX and NASDAQ.
#   - Equally weighting is used
from AlgorithmImports import *
import numpy as np
#endregion

class InvestmentFactor(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2000, 1, 1)
        self.SetEndDate(2010, 1, 1)
        self.SetCash(10000)
        self.SetWarmUp(timedelta(days=252)) # Warmup period of trading days in a Quarter
        self.min_expected_return = 0.0001  # Minimum expected return above trading costs (0.01%)
        self.fixed_slippage = 0.00003
        self.custom_fee = 0.00003 
        self.target_volatility = 0.005 #0.5%

        self.take_profit = 1.10
        self.stop_loss = 0.95

        self.symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.coarse_count = 3000
        self.weight = {}
        self.last_year_data = {}

        self.selection_flag = False
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), self.TimeRules.AfterMarketOpen(self.symbol), self.Selection)
        self.debug = False
        self.debugger = Debugging(self, self.Debug)

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            security.SetLeverage(1)

    def CoarseSelectionFunction(self, coarse):
        if not self.selection_flag:
            return Universe.Unchanged

        selected = [x.Symbol for x in coarse if x.HasFundamentalData and x.Market == 'usa']
        return selected

    def FineSelectionFunction(self, fine):
        fine = [x for x in fine if x.MarketCap != 0 and x.FinancialStatements.BalanceSheet.TotalAssets.TwelveMonths != 0 and x.OperationRatios.TotalAssetsGrowth.OneYear and \
                    ((x.SecurityReference.ExchangeId == "NYS") or (x.SecurityReference.ExchangeId == "NAS") or (x.SecurityReference.ExchangeId == "ASE"))]
                    
        if len(fine) > self.coarse_count:
            sorted_by_market_cap = sorted(fine, key = lambda x: x.MarketCap, reverse=True)
            top_by_market_cap = sorted_by_market_cap[:self.coarse_count]
        else:
            top_by_market_cap = fine
            
        # Sorting by investment factor.
        sorted_by_inv_factor = sorted(top_by_market_cap, key = lambda x: (x.OperationRatios.TotalAssetsGrowth.OneYear / x.FinancialStatements.BalanceSheet.TotalAssets.TwelveMonths), reverse=True)
        
        if len(sorted_by_inv_factor) >= 5:
            quintile  = int(len(sorted_by_inv_factor) / 5)
            self.long = [x.Symbol for x in sorted_by_inv_factor[-quintile:]]
            self.short = [x.Symbol for x in sorted_by_inv_factor[:quintile]]
            self.Debug(f'Long list: {len(self.long)}, Short list: {len(self.short)}') # Debug the number of symbols selected in the long and short lists
        
        return self.long + self.short

    def expected_return(self, symbol):
        # Get 252 days of daily history data for the symbol
        history = self.History([symbol], 252, Resolution.Daily)

        # If no history data is returned, return 0.0
        if history.empty:
            return 0.0

        # Extract the symbol from the index of the history data
        mapped_symbol = history.index.get_level_values(0)[0]
        
        # Extract the closing price data from the history data
        symbol_history = history.loc[mapped_symbol].iloc[:, 3]

        # Compute the daily returns and drop any NaN values
        daily_returns = symbol_history.pct_change().dropna()

        # Compute the average daily return
        avg_return = daily_returns.mean()
        
        return avg_return

    def volatility_targeting(self, symbol, target_volatility):
        # Get historical price data for the symbol (past 252 days)
        history = self.History([symbol], 252, Resolution.Daily)

        if history.empty:
            return np.nan, np.nan, np.nan

        mapped_symbol = history.index.get_level_values(0)[0]
        symbol_history = history.loc[mapped_symbol].iloc[:, 3]

        # Calculate daily returns
        daily_returns = symbol_history.pct_change().dropna()

        if len(daily_returns) < 20:
            return np.nan, np.nan, np.nan

        # Calculate EWMA volatility with lambda = 0.9 (90%)
        lambda_ = 0.9
        ewma_volatility = daily_returns.ewm(alpha=1 - lambda_).std().iloc[-1]

        # Calculate expanding window average of historical 20-day volatility
        expanding_window_volatility = daily_returns.rolling(window=len(daily_returns)).std().iloc[-1]

        # Sets the target volatility for a specific position to the expanding window average of historical volatility (calculated over 20 days)
        target_volatility = max(expanding_window_volatility, self.target_volatility)

        # Calculate the leverage. The leverage is used to adjust the position size to achieve the desired target volatility.
        leverage = target_volatility / ewma_volatility

        # Limit the maximum leverage to 2.
        leverage = min(leverage, 2)

        # Calculate the weight for each symbol
        weight = 1.0 / (len(self.long) + len(self.short))

        # Calculate the position size by multiplying the leverage, weight, and portfolio value
        position_size = leverage * weight * self.Portfolio.TotalPortfolioValue

        if np.isnan(position_size):
            return np.nan, np.nan, np.nan
        else:
            return leverage, weight, position_size

    def OnOrderEvent(self, order_event):
        if order_event.Status == OrderStatus.Filled:
            # Update available_margin when an order is filled
            self.available_margin = self.Portfolio.MarginRemaining

    def OnData(self, data):
            if self.IsWarmingUp or not self.selection_flag or self.Time.weekday() > 4:
                return
            self.selection_flag = False

            stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]

            for symbol in stocks_invested:
                if symbol not in self.long + self.short:
                    self.Liquidate(symbol)

            available_margin = self.Portfolio.MarginRemaining

            for position_type, symbol_list in [('Long', self.long), ('Short', self.short)]:
                for symbol in symbol_list:
                    if symbol in data and data[symbol]:

                        exp_return = self.expected_return(symbol)

                        fee = data[symbol].Price * 1 * self.custom_fee
                        trading_cost = data[symbol].Price * self.fixed_slippage + fee

                        if exp_return - trading_cost < self.min_expected_return:
                            continue

                        leverage, weight, _ = self.volatility_targeting(symbol, self.target_volatility)
                        max_position_size = available_margin / data[symbol].Price
                        position_size = min(leverage * weight * self.Portfolio.TotalPortfolioValue, available_margin * 0.95, max_position_size)

                        if position_size <= 0 or math.isnan(position_size) or math.isinf(position_size) or abs(position_size) < 0.0000000001 or abs(position_size) > 1000000000:
                            continue

                        if available_margin < abs(position_size):
                            self.debugger.debug_margin(symbol, position_size, available_margin)
                            position_size = available_margin

                        # Set holdings if the position size is valid
                        self.SetHoldings(symbol, position_size if position_type == 'Long' else -position_size, liquidateExistingHoldings=False, tag=f'{position_type}: {symbol}')
                        available_margin -= abs(position_size)  # Update available margin after placing orders

                        invested_security = self.Portfolio[symbol]
                        if invested_security.Invested:
                            stop_loss = invested_security.AveragePrice * self.stop_loss
                            take_profit = invested_security.AveragePrice * self.take_profit
                            self.StopMarketOrder(symbol, -invested_security.Quantity, stop_loss)
                            self.LimitOrder(symbol, -invested_security.Quantity, take_profit)

            self.long.clear()
            self.short.clear()

    def Selection(self):
        self.selection_flag = True




class Debugging:
    def __init__(self, algorithm, debug):
        self.algorithm = algorithm
        self.debug = debug

    def debug_daily_returns(self, symbol, daily_returns):
        if self.debug:
            self.algorithm.Debug(f"Daily Returns for {symbol}: {daily_returns}")

    def debug_expected_return(self, symbol, avg_return, exp_return):
        if self.debug:
            self.algorithm.Debug(f"Calculated average daily return for {symbol}: {avg_return * 100}%")
            self.algorithm.Debug(f'Expected return for {symbol}: {exp_return}')

    def debug_volatility_targeting(self, symbol, expanding_window_volatility, ewma_volatility, target_volatility, leverage, weight, position_size):
        if self.debug:
            self.algorithm.Debug(f'Volatility targeting for {symbol}')
            self.algorithm.Debug(f'expanding_window_volatility: {expanding_window_volatility}')
            self.algorithm.Debug(f'ewma_volatility: {ewma_volatility}')
            self.algorithm.Debug(f'target_volatility: {target_volatility}')
            self.algorithm.Debug(f'Leverage: {leverage}')
            self.algorithm.Debug(f'Weight: {weight}')
            self.algorithm.Debug(f'Position Size: {position_size}')

    def debug_investment_factor(self):
        if self.debug:
            self.algorithm.Debug(f'Stocks invested: {[x.Key for x in self.algorithm.Portfolio if x.Value.Invested]}')
            self.algorithm.Debug(f'Long list: {len(self.algorithm.long)}, Short list: {len(self.algorithm.short)}')
            self.algorithm.Debug(f'Available Margin: {self.algorithm.Portfolio.MarginRemaining}')

    def debug_trading_cost(self, symbol, trading_cost, exp_return):
        if self.debug:
            self.algorithm.Debug(f'Trading Cost for {symbol}: {trading_cost}')
            self.algorithm.Debug(f'Difference between expected return and trading cost for {symbol}: {exp_return - trading_cost}')

    def debug_margin(self, symbol, position_size, available_margin):
        if self.debug:
            self.algorithm.Debug(f"Insufficient margin for {symbol}. Required: {position_size}, Available: {available_margin}")
            self.algorithm.Debug(f'Available Margin: {available_margin}')

    def debug_position_size(self, symbol, position_size):
        if self.debug:
            self.algorithm.Debug(f'Position Size for {symbol}: {position_size}')

    def debug_set_holdings(self, symbol, weight):
        if self.debug:
            self.algorithm.Debug(f'Set Holdings: {symbol}, Weight: {weight}')
        