
#### KÜTÜPHANE IMPORTLARI VE SET OPTION AYARLARI ####

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler

#### VERİNİN HAZIRLANMASI ####

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()
df.head()
df.dtypes
df.shape
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
today_date = dt.datetime(2011, 12, 11)

#### LIFETIME VERİ YAPISININ HAZIRLANMASI ####

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate" : [lambda date: (date.max() - date.min()).days,
                                                lambda date: (today_date - date.min()).days],
                               "Invoice" : lambda num: num.nunique(),
                               "TotalPrice" : lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df.describe().T
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

#### BG-NBD MODELİNİN KURULMASI ####

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

## 1 ay içinde en çok satın alma beklenen  müşteri ##

cltv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency"],
                                                                                           cltv_df["T"]
                                                                                           )
## 6 ay içinde en çok satın alma beklenen  müşteri ##

cltv_df["expected_purc_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency"],
                                                                                           cltv_df["T"]
                                                                                           )
## 12 ay içinde en çok satın alma beklenen  müşteri ##

cltv_df["expected_purc_12_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*12,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency"],
                                                                                           cltv_df["T"]
                                                                                           )

#### TAHMİN SONUÇLARININ DEĞERLENDİRİLMESİ ####

plot_period_transactions(bgf)
plt.show(block=True)
plt.pause(5)

#### GAMMA-GAMMA MODELİNİN KURULMASI ####

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_avarage_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.head()

#### BG-NBD CE GAMMA-GAMMA İLE CLTV HESAPLANMASI ####

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time= 12,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values("clv", ascending=False).head(10)

#### MÜŞTERİ SEGMENTLERİNİN OLUŞTURULMASI ####

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.describe().T
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

cltv_final.to_excel("retail_cltv.xlsx")