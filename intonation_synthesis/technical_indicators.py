import talib


def make_technical_indicators(df, periods=[5, 10, 50, 200]):

    source_cols = df.columns
    for column in source_cols:
        for period in periods:
            # STATISTICS

            # LINEARREG - Linear Regression
            df[str(column) + '_linreg_' + str(period)] = talib.LINEARREG(
                df[column], timeperiod=period)
            # LINEARREG_ANGLE - Linear Regression Angle
            df[str(column) + '_linreg_angle_' + str(period)] = \
                talib.LINEARREG_ANGLE(df[column], timeperiod=period)
            # LINEARREG_INTERCEPT - Linear Regression Intercept
            df[str(column) + '_linreg_intercept_' + str(period)] = \
                talib.LINEARREG_INTERCEPT(df[column], timeperiod=period)
            # LINEARREG_SLOPE - Linear Regression Slope
            df[str(column) + '_linreg_slope_' + str(period)] = \
                talib.LINEARREG_SLOPE(df[column], timeperiod=period)
            # STDDEV - Standard Deviation
            df[str(column) + '_stddev_' + str(period)] = talib.STDDEV(
                df[column], timeperiod=period, nbdev=1)
            # TSF - Time Series Forecast
            df[str(column) + '_tsf_' + str(period)] = talib.TSF(
                df[column], timeperiod=period)
            # VAR - Variance
            df[str(column) + '_var_' + str(period)] = talib.VAR(
                df[column], timeperiod=period, nbdev=1)

            # OVERLAP INDICATORS

            # BBANDS - Bollinger Bands
            (df[str(column) + '_upperband_' + str(period)],
             df[str(column) + '_middleband_' + str(period)],
             df[str(column) + '_lowerband_' + str(period)]) = talib.BBANDS(
                df[column], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
            # DEMA - Double Exponential Moving Average
            df[str(column) + '_dema_' + str(period)] = talib.DEMA(
                df[column], timeperiod=period)
            # EMA - Exponential Moving Average
            # NOTE: The EMA function has an unstable period.
            df[str(column) + '_ema_' + str(period)] = talib.EMA(
                df[column], timeperiod=period)
            # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
            # NOTE: The HT_TRENDLINE function has an unstable period.
            df[str(column) + '_ht_trend'] = talib.HT_TRENDLINE(df[column])
            # KAMA - Kaufman Adaptive Moving Average
            # NOTE: The KAMA function has an unstable period.
            df[str(column) + '_kama_' + str(period)] = talib.KAMA(
                df[column], timeperiod=period)
            # MA - Moving average
            df[str(column) + '_ma_' + str(period)] = talib.MA(
                df[column], timeperiod=period, matype=0)
            # SMA - Simple Moving Average
            df[str(column) + '_sma_' + str(period)] = talib.SMA(
                df[column], timeperiod=period)
            # T3 - Triple Exponential Moving Average (T3)
            # NOTE: The T3 function has an unstable period.
            df[str(column) + '_t3_' + str(period)] = talib.T3(
                df[column], timeperiod=period, vfactor=0)
            # TEMA - Triple Exponential Moving Average
            df[str(column) + '_tema_' + str(period)] = talib.TEMA(
                df[column], timeperiod=period)
            # TRIMA - Triangular Moving Average
            df[str(column) + '_trima_' + str(period)] = talib.TRIMA(
                df[column], timeperiod=period)
            # WMA - Weighted Moving Average
            df[str(column) + '_wma_' + str(period)] = talib.WMA(
                df[column], timeperiod=period)

            # MOMENTUM

            # APO - Absolute Price Oscillator
            df[str(column) + '_apo'] = talib.APO(
                df[column], fastperiod=10, slowperiod=25, matype=0)
            # CMO - Chande Momentum Oscillator
            # NOTE: The CMO function has an unstable period.
            df[str(column) + '_cmo_' + str(period)] = talib.CMO(
                df[column], timeperiod=period)
            # MACD - Moving Average Convergence/Divergence
            df[str(column) + '_macd'], df['macdsignal'], df['macdhist'] = \
                talib.MACD(
                    df[column], fastperiod=10, slowperiod=25, signalperiod=15)
            # MACDEXT - MACD with controllable MA type
            (df[str(column) + '_macdext'],
             df[str(column) + '_macdsignalext'],
             df[str(column) + '_macdhistext']) = talib.MACDEXT(
                df[column], fastperiod=10, fastmatype=0, slowperiod=25,
                slowmatype=0, signalperiod=15, signalmatype=0)
            # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
            (df[str(column) + '_macdfix'],
             df[str(column) + '_macdsignalfix'],
             df[str(column) + '_macdhistfix']) = talib.MACDFIX(
                df[column], signalperiod=15)
            # MOM - Momentum
            df[str(column) + '_mom_' + str(period)] = talib.MOM(
                df[column], timeperiod=period)
            # PPO - Percentage Price Oscillator
            df[str(column) + '_ppo'] = talib.PPO(
                df[column], fastperiod=10, slowperiod=25, matype=0)
            # ROC - Rate of change : ((price/prevPrice)-1)*100
            df[str(column) + '_roc_' + str(period)] = talib.ROC(
                df[column], timeperiod=period)
            # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
            df[str(column) + '_rocp_' + str(period)] = talib.ROCP(
                df[column], timeperiod=period)
            # ROCR - Rate of change ratio: (price/prevPrice)
            df[str(column) + '_rocr_' + str(period)] = talib.ROCR(
                df[column], timeperiod=period)
            # RSI - Relative Strength Index
            # NOTE: The RSI function has an unstable period.
            df[str(column) + '_rsi_' + str(period)] = talib.RSI(
                df[column], timeperiod=period)
            # STOCHRSI - Stochastic Relative Strength Index
            # NOTE: The STOCHRSI function has an unstable period.
            (df[str(column) + '_fastk_' + str(period)],
             df[str(column) + '_fastd_' + str(period)]) = talib.STOCHRSI(
                df[column], timeperiod=period, fastk_period=5, fastd_period=3,
                fastd_matype=0)
            # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            df[str(column) + '_trix_' + str(period)] = talib.TRIX(
                df[column], timeperiod=period)
            # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
            # NOTE: The HT_DCPERIOD function has an unstable period.
            df[str(column) + '_ht_dcperiod'] = talib.HT_DCPERIOD(df[column])
            # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
            # NOTE: The HT_DCPHASE function has an unstable period.
            df[str(column) + '_ht_dcphase'] = talib.HT_DCPHASE(df[column])
            # HT_PHASOR - Hilbert Transform - Phasor Components
            # NOTE: The HT_PHASOR function has an unstable period.
            df[str(column) + '_inphase'], df['quadrature'] = talib.HT_PHASOR(
                df[column])
            # HT_SINE - Hilbert Transform - SineWave
            # NOTE: The HT_SINE function has an unstable period.
            df[str(column) + '_sine'], df['leadsine'] = talib.HT_SINE(
                df[column])
            # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
            # NOTE: The HT_TRENDMODE function has an unstable period.
            df[str(column) + '_ht_trendmode'] = talib.HT_TRENDMODE(
                df[column])

    df.fillna(0, inplace=True)
    df = df.drop(source_cols, axis=1)

    return df
