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
        self.SetEndDate(2005, 1, 1)
        self.SetCash(100000)
        self.SetWarmUp(timedelta(days=252))

        self.symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.coarse_count = 2000
        self.weight = {}
        self.last_year_data = {}

        self.selection_flag = False
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.Schedule.On(self.DateRules.MonthEnd(self.symbol), self.TimeRules.AfterMarketOpen(self.symbol), self.Selection)

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            security.SetFeeModel(CustomFeeModel())
            security.SetLeverage(50)

    def CoarseSelectionFunction(self, coarse):
        if not self.selection_flag:
            return Universe.Unchanged

        selected = [x.Symbol for x in coarse if x.HasFundamentalData and x.Market == 'usa']
        return selected

    def FineSelectionFunction(self, fine):
        fine = [x for x in fine if x.MarketCap != 0 and x.FinancialStatements.BalanceSheet.TotalAssets.TwelveMonths != 0 and x.OperationRatios.TotalAssetsGrowth.OneYear and \
                    ((x.SecurityReference.ExchangeId == "NYS"))]
                    
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
        
        return self.long + self.short

    def volatility_targeting(self, symbol, target_volatility):
        # Get monthly historical price data for the symbol (past 12 months)
        history = self.History([symbol], 252, Resolution.Daily)
        
        if history.empty:
            return 0.0

        mapped_symbol = history.index.get_level_values(0)[0]
        symbol_history = history.loc[mapped_symbol].iloc[:, 3]

        # Resample daily data to monthly data
        symbol_history = symbol_history.resample('M').last()

        # Calculate monthly returns and their standard deviation (volatility)
        monthly_returns = symbol_history.pct_change().dropna()
        monthly_volatility = monthly_returns.std()

        if monthly_volatility == 0:
            return 0.0
        
        position_size = (target_volatility / monthly_volatility)

        if np.isnan(position_size):
            return 0.0
        else:
            return position_size

    def CalculateKellyFraction(self, symbol, win_probability, win_loss_ratio): 
        # Calculate the Kelly fraction using the win probability and win/loss ratio
        kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
        return kelly_fraction

    def GetWinProbabilityAndWinLossRatio(self, symbol):
        # Get monthly historical price data for the symbol (past 12 months)
        history = self.History([symbol], 252, Resolution.Daily)

        if history.empty:
            return

        # Get the mapped symbol from the history DataFrame multi-level index
        mapped_symbol = history.index.get_level_values(0)[0]

        # Get the 'close' prices, which are in the 4th column (index 3) of the history DataFrame Generally, historical price data is organized in the following order: open, high, low, close, and volume. With this structure, the 'close' prices would indeed be in the 4th column.
        symbol_history = history.loc[mapped_symbol].iloc[:, 3]

        # Resample daily data to monthly data
        symbol_history = symbol_history.resample('M').last()

        # Calculate monthly returns
        monthly_returns = symbol_history.pct_change().dropna()

        # Define a simple trading strategy
        long_trades = monthly_returns > 0
        short_trades = monthly_returns < 0

        # Calculate win probability and win-loss ratio
        total_trades = len(long_trades) + len(short_trades)
        winning_trades = long_trades.sum() + short_trades.sum()
        losing_trades = total_trades - winning_trades
        win_probability = winning_trades / total_trades

        average_win = monthly_returns[long_trades].mean()  # Calculate the average winning trade return for long trades
        average_short_win = -monthly_returns[short_trades].mean()  # Calculate the average winning trade return for short trades
        average_loss = -(monthly_returns[~long_trades & ~short_trades].mean())  # Calculate the average losing trade return

        win_loss_ratio = (average_win + average_short_win) / average_loss  # Calculate the combined win-loss ratio

        return win_probability, win_loss_ratio




    def OnData(self, data):

        # Return if the algorithm is warming up or the selection flag is not set
        if self.IsWarmingUp:
            return 
        if not self.selection_flag:
            return
        self.selection_flag = False

        # Trade execution
        long_count = len(self.long)
        short_count = len(self.short)

        # Get the list of invested symbols
        stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]

        # Liquidate positions that are not in the current long and short lists
        for symbol in stocks_invested:
            if symbol not in self.long + self.short:
                self.Liquidate(symbol)

        # Get the remaining margin in the portfolio
        available_margin = self.Portfolio.MarginRemaining


        max_positions = 5000      # Set the maximum number of positions allowed in the portfolio

        kelly_fraction_factor = 1     # Set the Kelly fraction factor for more or less conservative position sizing (1 for full Kelly)

        max_position_size_percentage = 0.2      # Set the maximum position size as a percentage of the total available funds (50% in this case)


        # LONG POSITIONS
        for i, symbol in enumerate(self.long):
            if i >= max_positions:
                break # Break if maximum positions reached

            if symbol in data and data[symbol]: # Check if the symbol has valid data available

                target_volatility = 1 # Set target volatility for the position
 
                win_probability, win_loss_ratio = self.GetWinProbabilityAndWinLossRatio(symbol) # Get win probability and win/loss ratio for the symbol based on historical data

                kelly_fraction = self.CalculateKellyFraction(symbol, win_probability, win_loss_ratio) * kelly_fraction_factor # Calculate the Kelly fraction for the symbol using win probability, win/loss ratio, and kelly_fraction_factor
 
                position_size = self.volatility_targeting(symbol, target_volatility) * kelly_fraction # Calculate the Kelly fraction for the symbol using win probability, win/loss ratio, and kelly_fraction_factor
 
                weight = position_size / (long_count + short_count) # Calculate the Kelly fraction for the symbol using win probability, win/loss ratio, and kelly_fraction_factor

                weight = min(weight, max_position_size_percentage) # Limit the weight of the position to the maximum position size percentage

                # Set holdings if the weight is valid
                if not math.isnan(weight) and not math.isinf(weight) and 0.0000000001 <= abs(weight) <= 1000000000:
                    self.SetHoldings(symbol, weight)


        # SHORT POSITIONS
        for i, symbol in enumerate(self.short):
            if i >= max_positions: 
                break # Break if maximum positions reached

            if symbol in data and data[symbol]:

                target_volatility = 1

                win_probability, win_loss_ratio = self.GetWinProbabilityAndWinLossRatio(symbol)

                kelly_fraction = self.CalculateKellyFraction(symbol, win_probability, win_loss_ratio) * kelly_fraction_factor

                position_size = self.volatility_targeting(symbol, target_volatility) * kelly_fraction

                weight = position_size / (long_count + short_count)

                weight = min(weight, max_position_size_percentage)

                if not math.isnan(weight) and not math.isinf(weight) and 0.0000000001 <= abs(weight) <= 1000000000:
                    self.SetHoldings(symbol, -weight)


        # Clear the long and short lists for the next round of selection
        self.long.clear()
        self.short.clear()




    def Selection(self):
        self.selection_flag = True




# Custom fee model.
class CustomFeeModel(FeeModel):
    def GetOrderFee(self, parameters):
        fee = parameters.Security.Price * parameters.Order.AbsoluteQuantity * 0.00005
        return OrderFee(CashAmount(fee, "USD"))