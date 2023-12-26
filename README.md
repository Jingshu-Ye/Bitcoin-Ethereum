# Compare Forecasts of Prices of Two Cointegrated Cryptocurrencies with ARIMA Models
---
Title: "Compare Forecasts of Prices of Bitcoin and Ethereum with Two Independent Univariate ARIMA Models."
Output:
  html_document:
    df_print: paged
  pdf_document: default
---

# Comparison and analysis of prices of Bitcoin and Ethereum cryptocurrencies

### Prepared by  Paulina Nowakowska 

## Install and load packages

```{r results='hide', message=FALSE, warning=FALSE}
library("tidyverse")
library("urca")
library("forecast")
library("fUnitRoots")
library("tseries")
library("DMwR")
library("dynlm")
library("vars")

Sys.setlocale("LC_ALL","English")
```

## Data scraping

```{r results='hide', message=FALSE, warning=FALSE}
getCryptoHistoricalPrice <- function(x){
  library(tidyverse)
  # this function scraps the OHLC historical crypto prices from www.coinmarketcap.com
  paste0("https://coinmarketcap.com/currencies/",
         x,
         "/historical-data/?start=20160101&end=20200707") %>%
    xml2::read_html() %>%
    rvest::html_table(.) %>%
    .[[3]] %>% 
    as_tibble() %>%
    rename(Open  = `Open*`,
           Close = `Close**`,
           MarketCap = `Market Cap`) %>%
    mutate(Date = as.Date(Date, format = "%b %d, %Y")) %>%
    mutate_if(is.character, function(x) as.numeric(gsub(",", "", x))) %>%
    arrange(Date) %>%
    return()
}

btc <- getCryptoHistoricalPrice("bitcoin")
eth <- getCryptoHistoricalPrice("ethereum")
```

## Bitcoin and Ethereum historical prices

```{r  message=FALSE, warning=FALSE}
btc %>%
  ggplot(aes(Date, Close)) +
  geom_line(color = 'red') +
  geom_line(data = eth, color = 'blue') +
  labs(title = "Bitcoin vs Ethereum historical prices", subtitle = "Bitcoin in Red, Ehereum in Blue")
```

```{r results='hide', message=FALSE, warning=FALSE}
# Convert the data into ts
btc_ts <- ts(btc$Close,start=c(2016,1,1),frequency=365.25)
eth_ts <- ts(eth$Close,start=c(2016,1,1),frequency=365.25)
```


## Time series decomposition
```{r results = FALSE, message=FALSE, warning=FALSE}
btc_ts_decomposed <- decompose(btc_ts)
eth_ts_decomposed <- decompose(eth_ts)
```

### Bitcoin decomposition
```{r message=FALSE, warning=FALSE}
plot(btc_ts_decomposed)
``` 

### Ethereum decomposition
```{r message=FALSE, warning=FALSE}
plot(eth_ts_decomposed)
``` 

## ADF and KPSS tests to find the number of differences required for a stationary series
```{r message=FALSE, warning=FALSE}
btc_stationary <- diff(btc_ts, differences=1)
eth_stationary <- diff(eth_ts, differences=1)

adf.test(btc_ts) 
kpss.test(btc_ts) 
adf.test(btc_stationary) 
kpss.test(btc_stationary)

adf.test(eth_ts) 
kpss.test(eth_ts) 
adf.test(eth_stationary)
kpss.test(eth_stationary)
```

According to both tests, the original time series are not stationary. However, after differencing them once, they turn to be both stationary, which can also be visually confirmed based on the following graphs.

```{r message=FALSE, warning=FALSE}
plot(btc_ts)
plot(btc_stationary)
plot(eth_ts)
plot(eth_stationary)
```

## ACF and PACF plots
```{r message=FALSE, warning=FALSE}
acf(btc_stationary) 
pacf(btc_stationary) 

acf(eth_stationary)
pacf(eth_stationary)
```

ACF of both time series cut off at lag 0 meaning order of MA is 0 and PACF does not seem to be decaying, because of which order of AR component cannot be visually detected. Multiple values of AR will be tested and it will be chosen based on AIC, BIC and error measures.

## Train and test sets 80/20
```{r results='hide', message=FALSE, warning=FALSE}
# partition into train and test
train_btc_ts=btc_ts[1:1320] 
train_eth_ts=eth_ts[1:1320] 

test_btc_ts=btc_ts[1321:1650] 
test_eth_ts=eth_ts[1321:1650] 
```

## Models
### ARIMA
```{r results='hide', message=FALSE, warning=FALSE}
arimaModel_btc_1=arima(train_btc_ts, order=c(0,1,0))
arimaModel_btc_2=arima(train_btc_ts, order=c(1,1,0))
arimaModel_btc_3=arima(train_btc_ts, order=c(2,1,0))
arimaModel_btc_4 = auto.arima(train_btc_ts)

arimaModel_eth_1=arima(train_eth_ts, order=c(0,1,0))
arimaModel_eth_2=arima(train_eth_ts, order=c(1,1,0))
arimaModel_eth_3=arima(train_eth_ts, order=c(2,1,0))
arimaModel_eth_4 = auto.arima(train_eth_ts)
```

```{r message=FALSE, warning=FALSE}
print(arimaModel_btc_1);print(arimaModel_btc_2);print(arimaModel_btc_3) 
print(arimaModel_eth_1);print(arimaModel_eth_2);print(arimaModel_eth_3) 
```

Since AIC is not decreasing by adding higher orders of AR, simpler model is more desirable meaning AR(0). I will still test models with AR(1) and AR(2) to compare error measures.

#### Forecasts
```{r results='hide', message=FALSE, warning=FALSE}
forecast_btc_1=predict(arimaModel_btc_1, 330)
forecast_btc_2=predict(arimaModel_btc_2, 330)
forecast_btc_3=predict(arimaModel_btc_3, 330)
forecast_btc_4=predict(arimaModel_btc_4, 330)

forecast_eth_1=predict(arimaModel_eth_1, 330)
forecast_eth_2=predict(arimaModel_eth_2, 330)
forecast_eth_3=predict(arimaModel_eth_3, 330)
forecast_eth_4=predict(arimaModel_eth_4, 330)
```

#### Model accuracy
```{r message=FALSE, warning=FALSE}
accmeasures_btc_1=regr.eval(test_btc_ts, forecast_btc_1$pred)
accmeasures_btc_2=regr.eval(test_btc_ts, forecast_btc_2$pred)
accmeasures_btc_3=regr.eval(test_btc_ts, forecast_btc_3$pred)
accmeasures_btc_4=regr.eval(test_btc_ts, forecast_btc_4$pred)
accMeasure_btc=rbind(accmeasures_btc_1,accmeasures_btc_2,accmeasures_btc_3, accmeasures_btc_4)
print(accMeasure_btc) 

accmeasures_eth_1=regr.eval(test_eth_ts, forecast_eth_1$pred)
accmeasures_eth_2=regr.eval(test_eth_ts, forecast_eth_2$pred)
accmeasures_eth_3=regr.eval(test_eth_ts, forecast_eth_3$pred)
accmeasures_eth_4=regr.eval(test_eth_ts, forecast_eth_4$pred)
accMeasure_eth=rbind(accmeasures_eth_1,accmeasures_eth_2,accmeasures_eth_3, accmeasures_eth_4)
print(accMeasure_eth)
```

As I have already found that adding higher orders of AR does not have any added value based on AIC. I also confirmed the irrelevance of higher orders of AR based on close values of all four error measures regardless of the order. Same explanation applies to eth just like in the case of 'accMeasure_btc'.

## Granger causality
```{r message=FALSE, warning=FALSE}
data <- ts.union(btc_ts, eth_ts)
btc.d <- diff(btc_ts)[-1]
eth.d <- diff(eth_ts)[-1]
btc.eq <- lm(btc_ts ~ eth_ts, data = data)
error.ecm1 <- btc.eq$residuals[-1:-2]
btc.d1 <- diff(btc_ts)[-(length(btc_ts) - 1)]
eth.d1 <- diff(eth_ts[-(length(eth_ts) - 1)])

ecm.btc <- lm(btc.d ~ error.ecm1 + btc.d1 + eth.d1)
summary(ecm.btc) 
```

Speed of adjustment coefficient is positive and significant. So Bitcoin does all the work to get the two variables back towards the equilibrium path, which implies that there is Granger causality from Ethereum to Bitcoin and that it takes about 1/0.008861 periods to return to equilibrium.

## Cointegration
```{r message=FALSE, warning=FALSE}
cint1.dyn <- dynlm(btc_ts~eth_ts-1, data=data)
  ehat <- resid(cint1.dyn)
cint2.dyn <- dynlm(d(ehat)~L(ehat)-1)
summary(cint2.dyn)
```

Test rejects the null of no cointegration at 10% confidence level, meaning that the series are cointegrated.

## Forecast variance decomposition
```{r message=FALSE, warning=FALSE}
varmat <- as.matrix(cbind(btc_ts,eth_ts))
varfit <- VAR(varmat) 
summary(varfit)

plot(fevd(varfit)) 
```

Forecast variance decomposition estimates the contribution of a shock in each variable to the response in both variables. Graph shows that almost 100 percent of the variance in Bitcoin is caused by Bitcoin itself, while only about 70 percent of variance of Ethereum is caused by Ethereum and the rest is caused by Bitcoin.

## VAR model
```{r results=FALSE, message=FALSE, warning=FALSE}
# log return of cryptocurrencies
train_btc_log_return = log((as.vector(train_btc_ts))/(dplyr::lag(as.vector(train_btc_ts))))*100 
train_eth_log_return = log((as.vector(train_eth_ts))/(dplyr::lag(as.vector(train_eth_ts))))*100
train_btc_log_return[is.na(train_btc_log_return)] <- 0
```


```{r message=FALSE, warning=FALSE}
TSpread <- train_btc_ts - train_eth_ts
train_btc_ts_log_return <- ts(train_btc_log_return, start=c(2016,1,1),frequency=365.25)
TSpread <- ts(TSpread, start=c(2016,1,1),frequency=365.25)

VAR_data <- window(ts.union(train_btc_ts_log_return, TSpread))
VAR_est <- VAR(y = VAR_data, p = 2)

summary(VAR_est$varresult$train_btc_ts_log_return)$adj.r.squared
summary(VAR_est$varresult$TSpread)$adj.r.squared

forecasts <- forecast(VAR_est, 330)
regr.eval(test_btc_ts, forecasts$forecast$TSpread$mean)
forecasts %>%
  autoplot() + xlab("Year")
```

VAR model turns out to be performing better for Bitcoin than ARIMA models based on mape, 35% and 18% respectively.
