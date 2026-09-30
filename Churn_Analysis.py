
# TELCO CUSTOMER CHURN & RETENTION ANALYSIS

# Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv



# 01. IMPORTS

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
import statistics
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Optional packages:
# import openpyxl
# import xgboost
# import lightgbm
# import imblearn

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 02. PATHS AND CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
#TABLE_DIR = OUTPUT_DIR / "tables"
#MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
#TABLE_DIR.mkdir(exist_ok=True)
#MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "Churn"
TARGET_FLAG = "Churn_Flag"
CUSTOMER_ID = "customerID"

# 03. DATA LOADING

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.tail())
print(df.sample(5, random_state=RANDOM_STATE))

# 04. INITIAL INSPECTION

print(df.info())
print(df.describe(include="all").T)
print(df.nunique().sort_values())
print(df.dtypes)
print(df.memory_usage(deep=True))
print(df.isna().sum().sort_values(ascending=False))
print(df.duplicated().sum())

# 05. DATA CLEANING

df = df.copy()

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype("string").str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

df = df.drop_duplicates()
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
df["SeniorCitizen_Label"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print("Cleaned shape:", df.shape)
print("Remaining missing values:")
print(df.isna().sum().sort_values(ascending=False))

# 06. FEATURE ENGINEERING

df["Tenure_Group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)

df["Tenure_Year"] = df["tenure"] / 12
df["MonthlyCharge_Group"] = pd.cut(
    df["MonthlyCharges"],
    bins=[-np.inf, 35, 60, 90, 120, np.inf],
    labels=["<=35", "36-60", "61-90", "91-120", ">120"]
)

df["Annual_Revenue"] = df["MonthlyCharges"] * 12
df["Revenue_Lost"] = np.where(
    df["Churn_Flag"].eq(1),
    df["MonthlyCharges"],
    0
)

df["Estimated_LTV"] = (
    df["MonthlyCharges"] *
    np.maximum(df["tenure"], 1)
)

df["Service_Count"] = (
    df["PhoneService"].eq("Yes").astype(int)
    + df["OnlineSecurity"].eq("Yes").astype(int)
    + df["OnlineBackup"].eq("Yes").astype(int)
    + df["DeviceProtection"].eq("Yes").astype(int)
    + df["TechSupport"].eq("Yes").astype(int)
    + df["StreamingTV"].eq("Yes").astype(int)
    + df["StreamingMovies"].eq("Yes").astype(int)
)

df["Support_Count"] = (
    df["OnlineSecurity"].eq("Yes").astype(int)
    + df["TechSupport"].eq("Yes").astype(int)
)

df["Streaming_Count"] = (
    df["StreamingTV"].eq("Yes").astype(int)
    + df["StreamingMovies"].eq("Yes").astype(int)
)

df["Is_Month_To_Month"] = df["Contract"].eq("Month-to-month").astype(int)
df["Has_Partner"] = df["Partner"].eq("Yes").astype(int)
df["Has_Dependents"] = df["Dependents"].eq("Yes").astype(int)
df["Paperless_Flag"] = df["PaperlessBilling"].eq("Yes").astype(int)
df["Electronic_Check_Flag"] = df["PaymentMethod"].eq("Electronic check").astype(int)

df["Customer_Value_Band"] = pd.qcut(
    df["Estimated_LTV"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
    duplicates="drop"
)

print(df.head())
print(df[["tenure", "Tenure_Group", "MonthlyCharges", "MonthlyCharge_Group",
          "Annual_Revenue", "Revenue_Lost", "Estimated_LTV", "Service_Count"]].head())

# ANALYSIS FUNCTION 001: Contract

def analyze_contract():
    result = (
        df.groupby("Contract", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_001 = analyze_contract()
print(result_001)

# ANALYSIS FUNCTION 002: Tenure_Group

def analyze_tenure_group():
    result = (
        df.groupby("Tenure_Group", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_002 = analyze_tenure_group()
print(result_002)

# ANALYSIS FUNCTION 003: MonthlyCharge_Group

def analyze_monthlycharge_group():
    result = (
        df.groupby("MonthlyCharge_Group", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_003 = analyze_monthlycharge_group()
print(result_003)

# ANALYSIS FUNCTION 004: PaymentMethod

def analyze_paymentmethod():
    result = (
        df.groupby("PaymentMethod", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_004 = analyze_paymentmethod()
print(result_004)

# ANALYSIS FUNCTION 005: InternetService

def analyze_internetservice():
    result = (
        df.groupby("InternetService", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_005 = analyze_internetservice()
print(result_005)

# ANALYSIS FUNCTION 006: OnlineSecurity

def analyze_onlinesecurity():
    result = (
        df.groupby("OnlineSecurity", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_006 = analyze_onlinesecurity()
print(result_006)

# ANALYSIS FUNCTION 007: OnlineBackup

def analyze_onlinebackup():
    result = (
        df.groupby("OnlineBackup", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_007 = analyze_onlinebackup()
print(result_007)

# ANALYSIS FUNCTION 008: DeviceProtection

def analyze_deviceprotection():
    result = (
        df.groupby("DeviceProtection", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_008 = analyze_deviceprotection()
print(result_008)

# ANALYSIS FUNCTION 009: TechSupport

def analyze_techsupport():
    result = (
        df.groupby("TechSupport", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_009 = analyze_techsupport()
print(result_009)

# ANALYSIS FUNCTION 010: StreamingTV

def analyze_streamingtv():
    result = (
        df.groupby("StreamingTV", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_010 = analyze_streamingtv()
print(result_010)

# ANALYSIS FUNCTION 011: StreamingMovies

def analyze_streamingmovies():
    result = (
        df.groupby("StreamingMovies", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_011 = analyze_streamingmovies()
print(result_011)

# ANALYSIS FUNCTION 012: PhoneService

def analyze_phoneservice():
    result = (
        df.groupby("PhoneService", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_012 = analyze_phoneservice()
print(result_012)

# ANALYSIS FUNCTION 013: MultipleLines

def analyze_multiplelines():
    result = (
        df.groupby("MultipleLines", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_013 = analyze_multiplelines()
print(result_013)

# ANALYSIS FUNCTION 014: PaperlessBilling

def analyze_paperlessbilling():
    result = (
        df.groupby("PaperlessBilling", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_014 = analyze_paperlessbilling()
print(result_014)

# ANALYSIS FUNCTION 015: Partner

def analyze_partner():
    result = (
        df.groupby("Partner", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_015 = analyze_partner()
print(result_015)

# ANALYSIS FUNCTION 016: Dependents

def analyze_dependents():
    result = (
        df.groupby("Dependents", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_016 = analyze_dependents()
print(result_016)

# ANALYSIS FUNCTION 017: gender

def analyze_gender():
    result = (
        df.groupby("gender", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_017 = analyze_gender()
print(result_017)

# ANALYSIS FUNCTION 018: SeniorCitizen_Label

def analyze_seniorcitizen_label():
    result = (
        df.groupby("SeniorCitizen_Label", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_018 = analyze_seniorcitizen_label()
print(result_018)

# ANALYSIS FUNCTION 019: Customer_Value_Band

def analyze_customer_value_band():
    result = (
        df.groupby("Customer_Value_Band", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_019 = analyze_customer_value_band()
print(result_019)

# ANALYSIS FUNCTION 020: Service_Count

def analyze_service_count():
    result = (
        df.groupby("Service_Count", dropna=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned_Customers=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum"),
            Annual_Revenue=("Annual_Revenue", "sum")
        )
        .reset_index()
    )
    result["Churn_Rate"] = result["Churn_Rate"] * 100
    result["Retention_Rate"] = 100 - result["Churn_Rate"]
    return result.sort_values("Churn_Rate", ascending=False)

result_020 = analyze_service_count()
print(result_020)

# NUMERIC STATISTICS 01: tenure

def stats_tenure():
    series = pd.to_numeric(df["tenure"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_01_result = stats_tenure()
print(stats_01_result)

# NUMERIC STATISTICS 02: MonthlyCharges

def stats_monthlycharges():
    series = pd.to_numeric(df["MonthlyCharges"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_02_result = stats_monthlycharges()
print(stats_02_result)

# NUMERIC STATISTICS 03: TotalCharges

def stats_totalcharges():
    series = pd.to_numeric(df["TotalCharges"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_03_result = stats_totalcharges()
print(stats_03_result)

# NUMERIC STATISTICS 04: Annual_Revenue


def stats_annual_revenue():
    series = pd.to_numeric(df["Annual_Revenue"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_04_result = stats_annual_revenue()
print(stats_04_result)

# NUMERIC STATISTICS 05: Revenue_Lost

def stats_revenue_lost():
    series = pd.to_numeric(df["Revenue_Lost"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_05_result = stats_revenue_lost()
print(stats_05_result)

# NUMERIC STATISTICS 06: Estimated_LTV

def stats_estimated_ltv():
    series = pd.to_numeric(df["Estimated_LTV"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_06_result = stats_estimated_ltv()
print(stats_06_result)

# NUMERIC STATISTICS 07: Service_Count

def stats_service_count():
    series = pd.to_numeric(df["Service_Count"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_07_result = stats_service_count()
print(stats_07_result)

# NUMERIC STATISTICS 08: Support_Count

def stats_support_count():
    series = pd.to_numeric(df["Support_Count"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_08_result = stats_support_count()
print(stats_08_result)

# NUMERIC STATISTICS 09: Streaming_Count

def stats_streaming_count():
    series = pd.to_numeric(df["Streaming_Count"], errors="coerce").dropna()
    return pd.Series({
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q2": series.quantile(0.50),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "range": series.max() - series.min(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis()
    })

stats_09_result = stats_streaming_count()
print(stats_09_result)

# CHART FUNCTION 001: Contract

def plot_churn_by_contract():
    temp = (
        df.groupby("Contract", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Contract")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 002: Tenure_Group

def plot_churn_by_tenure_group():
    temp = (
        df.groupby("Tenure_Group", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Tenure_Group")
    ax.set_xlabel("Tenure_Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_tenure_group.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 003: MonthlyCharge_Group

def plot_churn_by_monthlycharge_group():
    temp = (
        df.groupby("MonthlyCharge_Group", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by MonthlyCharge_Group")
    ax.set_xlabel("MonthlyCharge_Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_monthlycharge_group.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 004: PaymentMethod

def plot_churn_by_paymentmethod():
    temp = (
        df.groupby("PaymentMethod", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PaymentMethod")
    ax.set_xlabel("PaymentMethod")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_paymentmethod.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 005: InternetService

def plot_churn_by_internetservice():
    temp = (
        df.groupby("InternetService", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by InternetService")
    ax.set_xlabel("InternetService")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_internetservice.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 006: OnlineSecurity

def plot_churn_by_onlinesecurity():
    temp = (
        df.groupby("OnlineSecurity", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by OnlineSecurity")
    ax.set_xlabel("OnlineSecurity")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_onlinesecurity.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 007: OnlineBackup

def plot_churn_by_onlinebackup():
    temp = (
        df.groupby("OnlineBackup", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by OnlineBackup")
    ax.set_xlabel("OnlineBackup")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_onlinebackup.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 008: DeviceProtection

def plot_churn_by_deviceprotection():
    temp = (
        df.groupby("DeviceProtection", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by DeviceProtection")
    ax.set_xlabel("DeviceProtection")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_deviceprotection.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 009: TechSupport

def plot_churn_by_techsupport():
    temp = (
        df.groupby("TechSupport", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by TechSupport")
    ax.set_xlabel("TechSupport")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_techsupport.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 010: StreamingTV

def plot_churn_by_streamingtv():
    temp = (
        df.groupby("StreamingTV", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by StreamingTV")
    ax.set_xlabel("StreamingTV")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_streamingtv.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 011: StreamingMovies

def plot_churn_by_streamingmovies():
    temp = (
        df.groupby("StreamingMovies", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by StreamingMovies")
    ax.set_xlabel("StreamingMovies")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_streamingmovies.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 012: PhoneService

def plot_churn_by_phoneservice():
    temp = (
        df.groupby("PhoneService", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PhoneService")
    ax.set_xlabel("PhoneService")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_phoneservice.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 013: MultipleLines

def plot_churn_by_multiplelines():
    temp = (
        df.groupby("MultipleLines", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by MultipleLines")
    ax.set_xlabel("MultipleLines")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_multiplelines.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 014: PaperlessBilling

def plot_churn_by_paperlessbilling():
    temp = (
        df.groupby("PaperlessBilling", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PaperlessBilling")
    ax.set_xlabel("PaperlessBilling")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_paperlessbilling.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 015: Partner

def plot_churn_by_partner():
    temp = (
        df.groupby("Partner", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Partner")
    ax.set_xlabel("Partner")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_partner.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 016: Dependents

def plot_churn_by_dependents():
    temp = (
        df.groupby("Dependents", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Dependents")
    ax.set_xlabel("Dependents")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_dependents.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 017: gender

def plot_churn_by_gender():
    temp = (
        df.groupby("gender", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by gender")
    ax.set_xlabel("gender")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_gender.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 018: SeniorCitizen_Label

def plot_churn_by_seniorcitizen_label():
    temp = (
        df.groupby("SeniorCitizen_Label", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by SeniorCitizen_Label")
    ax.set_xlabel("SeniorCitizen_Label")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_seniorcitizen_label.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 019: Customer_Value_Band

def plot_churn_by_customer_value_band():
    temp = (
        df.groupby("Customer_Value_Band", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Customer_Value_Band")
    ax.set_xlabel("Customer_Value_Band")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_customer_value_band.png", dpi=150)
    plt.close(fig)
    return temp

# CHART FUNCTION 020: Service_Count

def plot_churn_by_service_count():
    temp = (
        df.groupby("Service_Count", dropna=False)[TARGET_FLAG]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    temp.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Service_Count")
    ax.set_xlabel("Service_Count")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "churn_by_service_count.png", dpi=150)
    plt.close(fig)
    return temp

# 07. CORE BUSINESS KPIs

def total_customers(data=df):
    return data[CUSTOMER_ID].nunique()

def churned_customers(data=df):
    return int(data[TARGET_FLAG].sum())

def retained_customers(data=df):
    return int((data[TARGET_FLAG] == 0).sum())

def churn_rate(data=df):
    return data[TARGET_FLAG].mean() * 100

def retention_rate(data=df):
    return 100 - churn_rate(data)

def total_monthly_revenue(data=df):
    return data["MonthlyCharges"].sum()

def total_annual_revenue(data=df):
    return data["Annual_Revenue"].sum()

def revenue_lost(data=df):
    return data["Revenue_Lost"].sum()

def average_monthly_charge(data=df):
    return data["MonthlyCharges"].mean()

def average_tenure(data=df):
    return data["tenure"].mean()

def average_ltv(data=df):
    return data["Estimated_LTV"].mean()

def arpu(data=df):
    return data["MonthlyCharges"].sum() / data[CUSTOMER_ID].nunique()

core_kpis = pd.Series({
    "Total Customers": total_customers(),
    "Churned Customers": churned_customers(),
    "Retained Customers": retained_customers(),
    "Churn Rate (%)": churn_rate(),
    "Retention Rate (%)": retention_rate(),
    "Monthly Revenue": total_monthly_revenue(),
    "Annual Revenue": total_annual_revenue(),
    "Revenue Lost from Churn": revenue_lost(),
    "Average Monthly Charge": average_monthly_charge(),
    "Average Tenure": average_tenure(),
    "Average Estimated LTV": average_ltv(),
    "ARPU": arpu()
})
print(core_kpis)

# 08. PROBABILITY ANALYSIS

p_churn = df[TARGET_FLAG].mean()
p_no_churn = 1 - p_churn

print("P(Churn):", p_churn)
print("P(No Churn):", p_no_churn)

contract_churn = pd.crosstab(
    df["Contract"],
    df[TARGET_FLAG],
    normalize="index"
)
print(contract_churn)

# Conditional probability:
# P(Churn | Month-to-month)
month_to_month = df["Contract"].eq("Month-to-month")
p_churn_given_m2m = df.loc[month_to_month, TARGET_FLAG].mean()
print("P(Churn | Month-to-month):", p_churn_given_m2m)

electronic = df["PaymentMethod"].eq("Electronic check")
p_churn_given_electronic = df.loc[electronic, TARGET_FLAG].mean()
print("P(Churn | Electronic check):", p_churn_given_electronic)

# Joint probability:
joint_m2m_churn = (
    df["Contract"].eq("Month-to-month") &
    df[TARGET_FLAG].eq(1)
).mean()
print("P(Month-to-month AND Churn):", joint_m2m_churn)

# Bayes-style calculation:
p_m2m = df["Contract"].eq("Month-to-month").mean()
p_churn_given_m2m = df.loc[month_to_month, TARGET_FLAG].mean()
bayes_joint = p_churn_given_m2m * p_m2m
print("Bayes reconstructed joint probability:", bayes_joint)

# HYPOTHESIS TEST 01: MonthlyCharges

churn_group_1 = df.loc[df[TARGET_FLAG] == 1, "MonthlyCharges"].dropna()
retain_group_1 = df.loc[df[TARGET_FLAG] == 0, "MonthlyCharges"].dropna()

t_stat_1, p_value_1 = ttest_ind(
    churn_group_1,
    retain_group_1,
    equal_var=False
)

u_stat_1, u_p_value_1 = mannwhitneyu(
    churn_group_1,
    retain_group_1,
    alternative="two-sided"
)

print("Feature:", "MonthlyCharges")
print("Welch t-test statistic:", t_stat_1)
print("Welch t-test p-value:", p_value_1)
print("Mann-Whitney statistic:", u_stat_1)
print("Mann-Whitney p-value:", u_p_value_1)

# HYPOTHESIS TEST 02: TotalCharges

churn_group_2 = df.loc[df[TARGET_FLAG] == 1, "TotalCharges"].dropna()
retain_group_2 = df.loc[df[TARGET_FLAG] == 0, "TotalCharges"].dropna()

t_stat_2, p_value_2 = ttest_ind(
    churn_group_2,
    retain_group_2,
    equal_var=False
)

u_stat_2, u_p_value_2 = mannwhitneyu(
    churn_group_2,
    retain_group_2,
    alternative="two-sided"
)

print("Feature:", "TotalCharges")
print("Welch t-test statistic:", t_stat_2)
print("Welch t-test p-value:", p_value_2)
print("Mann-Whitney statistic:", u_stat_2)
print("Mann-Whitney p-value:", u_p_value_2)

# HYPOTHESIS TEST 03: tenure

churn_group_3 = df.loc[df[TARGET_FLAG] == 1, "tenure"].dropna()
retain_group_3 = df.loc[df[TARGET_FLAG] == 0, "tenure"].dropna()

t_stat_3, p_value_3 = ttest_ind(
    churn_group_3,
    retain_group_3,
    equal_var=False
)

u_stat_3, u_p_value_3 = mannwhitneyu(
    churn_group_3,
    retain_group_3,
    alternative="two-sided"
)

print("Feature:", "tenure")
print("Welch t-test statistic:", t_stat_3)
print("Welch t-test p-value:", p_value_3)
print("Mann-Whitney statistic:", u_stat_3)
print("Mann-Whitney p-value:", u_p_value_3)

# HYPOTHESIS TEST 04: Estimated_LTV

churn_group_4 = df.loc[df[TARGET_FLAG] == 1, "Estimated_LTV"].dropna()
retain_group_4 = df.loc[df[TARGET_FLAG] == 0, "Estimated_LTV"].dropna()

t_stat_4, p_value_4 = ttest_ind(
    churn_group_4,
    retain_group_4,
    equal_var=False
)

u_stat_4, u_p_value_4 = mannwhitneyu(
    churn_group_4,
    retain_group_4,
    alternative="two-sided"
)

print("Feature:", "Estimated_LTV")
print("Welch t-test statistic:", t_stat_4)
print("Welch t-test p-value:", p_value_4)
print("Mann-Whitney statistic:", u_stat_4)
print("Mann-Whitney p-value:", u_p_value_4)

# HYPOTHESIS TEST 05: Service_Count

churn_group_5 = df.loc[df[TARGET_FLAG] == 1, "Service_Count"].dropna()
retain_group_5 = df.loc[df[TARGET_FLAG] == 0, "Service_Count"].dropna()

t_stat_5, p_value_5 = ttest_ind(
    churn_group_5,
    retain_group_5,
    equal_var=False
)

u_stat_5, u_p_value_5 = mannwhitneyu(
    churn_group_5,
    retain_group_5,
    alternative="two-sided"
)

print("Feature:", "Service_Count")
print("Welch t-test statistic:", t_stat_5)
print("Welch t-test p-value:", p_value_5)
print("Mann-Whitney statistic:", u_stat_5)
print("Mann-Whitney p-value:", u_p_value_5)

# CHI-SQUARE TEST 01: Contract

contingency_1 = pd.crosstab(df["Contract"], df[TARGET_FLAG])
chi2_1, p_1, dof_1, expected_1 = chi2_contingency(contingency_1)
print("Chi-square feature:", "Contract")
print("chi2:", chi2_1)
print("p-value:", p_1)
print("degrees of freedom:", dof_1)

# CHI-SQUARE TEST 02: Tenure_Group

contingency_2 = pd.crosstab(df["Tenure_Group"], df[TARGET_FLAG])
chi2_2, p_2, dof_2, expected_2 = chi2_contingency(contingency_2)
print("Chi-square feature:", "Tenure_Group")
print("chi2:", chi2_2)
print("p-value:", p_2)
print("degrees of freedom:", dof_2)

# CHI-SQUARE TEST 03: MonthlyCharge_Group

contingency_3 = pd.crosstab(df["MonthlyCharge_Group"], df[TARGET_FLAG])
chi2_3, p_3, dof_3, expected_3 = chi2_contingency(contingency_3)
print("Chi-square feature:", "MonthlyCharge_Group")
print("chi2:", chi2_3)
print("p-value:", p_3)
print("degrees of freedom:", dof_3)

# CHI-SQUARE TEST 04: PaymentMethod

contingency_4 = pd.crosstab(df["PaymentMethod"], df[TARGET_FLAG])
chi2_4, p_4, dof_4, expected_4 = chi2_contingency(contingency_4)
print("Chi-square feature:", "PaymentMethod")
print("chi2:", chi2_4)
print("p-value:", p_4)
print("degrees of freedom:", dof_4)

# CHI-SQUARE TEST 05: InternetService

contingency_5 = pd.crosstab(df["InternetService"], df[TARGET_FLAG])
chi2_5, p_5, dof_5, expected_5 = chi2_contingency(contingency_5)
print("Chi-square feature:", "InternetService")
print("chi2:", chi2_5)
print("p-value:", p_5)
print("degrees of freedom:", dof_5)

# CHI-SQUARE TEST 06: OnlineSecurity

contingency_6 = pd.crosstab(df["OnlineSecurity"], df[TARGET_FLAG])
chi2_6, p_6, dof_6, expected_6 = chi2_contingency(contingency_6)
print("Chi-square feature:", "OnlineSecurity")
print("chi2:", chi2_6)
print("p-value:", p_6)
print("degrees of freedom:", dof_6)

# CHI-SQUARE TEST 07: OnlineBackup

contingency_7 = pd.crosstab(df["OnlineBackup"], df[TARGET_FLAG])
chi2_7, p_7, dof_7, expected_7 = chi2_contingency(contingency_7)
print("Chi-square feature:", "OnlineBackup")
print("chi2:", chi2_7)
print("p-value:", p_7)
print("degrees of freedom:", dof_7)

# CHI-SQUARE TEST 08: DeviceProtection

contingency_8 = pd.crosstab(df["DeviceProtection"], df[TARGET_FLAG])
chi2_8, p_8, dof_8, expected_8 = chi2_contingency(contingency_8)
print("Chi-square feature:", "DeviceProtection")
print("chi2:", chi2_8)
print("p-value:", p_8)
print("degrees of freedom:", dof_8)

# CHI-SQUARE TEST 09: TechSupport

contingency_9 = pd.crosstab(df["TechSupport"], df[TARGET_FLAG])
chi2_9, p_9, dof_9, expected_9 = chi2_contingency(contingency_9)
print("Chi-square feature:", "TechSupport")
print("chi2:", chi2_9)
print("p-value:", p_9)
print("degrees of freedom:", dof_9)

# CHI-SQUARE TEST 10: StreamingTV

contingency_10 = pd.crosstab(df["StreamingTV"], df[TARGET_FLAG])
chi2_10, p_10, dof_10, expected_10 = chi2_contingency(contingency_10)
print("Chi-square feature:", "StreamingTV")
print("chi2:", chi2_10)
print("p-value:", p_10)
print("degrees of freedom:", dof_10)

# CHI-SQUARE TEST 11: StreamingMovies

contingency_11 = pd.crosstab(df["StreamingMovies"], df[TARGET_FLAG])
chi2_11, p_11, dof_11, expected_11 = chi2_contingency(contingency_11)
print("Chi-square feature:", "StreamingMovies")
print("chi2:", chi2_11)
print("p-value:", p_11)
print("degrees of freedom:", dof_11)

# CHI-SQUARE TEST 12: PhoneService

contingency_12 = pd.crosstab(df["PhoneService"], df[TARGET_FLAG])
chi2_12, p_12, dof_12, expected_12 = chi2_contingency(contingency_12)
print("Chi-square feature:", "PhoneService")
print("chi2:", chi2_12)
print("p-value:", p_12)
print("degrees of freedom:", dof_12)

# CHI-SQUARE TEST 13: MultipleLines

contingency_13 = pd.crosstab(df["MultipleLines"], df[TARGET_FLAG])
chi2_13, p_13, dof_13, expected_13 = chi2_contingency(contingency_13)
print("Chi-square feature:", "MultipleLines")
print("chi2:", chi2_13)
print("p-value:", p_13)
print("degrees of freedom:", dof_13)

# CHI-SQUARE TEST 14: PaperlessBilling

contingency_14 = pd.crosstab(df["PaperlessBilling"], df[TARGET_FLAG])
chi2_14, p_14, dof_14, expected_14 = chi2_contingency(contingency_14)
print("Chi-square feature:", "PaperlessBilling")
print("chi2:", chi2_14)
print("p-value:", p_14)
print("degrees of freedom:", dof_14)

# CHI-SQUARE TEST 15: Partner

contingency_15 = pd.crosstab(df["Partner"], df[TARGET_FLAG])
chi2_15, p_15, dof_15, expected_15 = chi2_contingency(contingency_15)
print("Chi-square feature:", "Partner")
print("chi2:", chi2_15)
print("p-value:", p_15)
print("degrees of freedom:", dof_15)

# CHI-SQUARE TEST 16: Dependents

contingency_16 = pd.crosstab(df["Dependents"], df[TARGET_FLAG])
chi2_16, p_16, dof_16, expected_16 = chi2_contingency(contingency_16)
print("Chi-square feature:", "Dependents")
print("chi2:", chi2_16)
print("p-value:", p_16)
print("degrees of freedom:", dof_16)

# CHI-SQUARE TEST 17: gender

contingency_17 = pd.crosstab(df["gender"], df[TARGET_FLAG])
chi2_17, p_17, dof_17, expected_17 = chi2_contingency(contingency_17)
print("Chi-square feature:", "gender")
print("chi2:", chi2_17)
print("p-value:", p_17)
print("degrees of freedom:", dof_17)

# CHI-SQUARE TEST 18: SeniorCitizen_Label

contingency_18 = pd.crosstab(df["SeniorCitizen_Label"], df[TARGET_FLAG])
chi2_18, p_18, dof_18, expected_18 = chi2_contingency(contingency_18)
print("Chi-square feature:", "SeniorCitizen_Label")
print("chi2:", chi2_18)
print("p-value:", p_18)
print("degrees of freedom:", dof_18)

# 09. CORRELATION ANALYSIS

numeric_df = df.select_dtypes(include=np.number)
pearson_matrix = numeric_df.corr(method="pearson")
spearman_matrix = numeric_df.corr(method="spearman")

print("Pearson correlation:")
print(pearson_matrix[TARGET_FLAG].sort_values(ascending=False))

print("Spearman correlation:")
print(spearman_matrix[TARGET_FLAG].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(pearson_matrix, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Numeric Feature Correlation Matrix")
fig.tight_layout()
fig.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)

# EDA BLOCK 01: Contract vs Churn

eda_01 = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 01 - Contract:")
print(eda_01)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
ax.set_title("Contract vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_01_Contract.png", dpi=150)
plt.close(fig)

# EDA BLOCK 02: Tenure_Group vs Churn

eda_02 = pd.crosstab(
    df["Tenure_Group"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 02 - Tenure_Group:")
print(eda_02)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="Tenure_Group", hue="Churn", ax=ax)
ax.set_title("Tenure_Group vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_02_Tenure_Group.png", dpi=150)
plt.close(fig)

# EDA BLOCK 03: PaymentMethod vs Churn

eda_03 = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 03 - PaymentMethod:")
print(eda_03)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax)
ax.set_title("PaymentMethod vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_03_PaymentMethod.png", dpi=150)
plt.close(fig)

# EDA BLOCK 04: InternetService vs Churn

eda_04 = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 04 - InternetService:")
print(eda_04)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax)
ax.set_title("InternetService vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_04_InternetService.png", dpi=150)
plt.close(fig)

# EDA BLOCK 05: OnlineSecurity vs Churn

eda_05 = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 05 - OnlineSecurity:")
print(eda_05)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="OnlineSecurity", hue="Churn", ax=ax)
ax.set_title("OnlineSecurity vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_05_OnlineSecurity.png", dpi=150)
plt.close(fig)

# EDA BLOCK 06: TechSupport vs Churn

eda_06 = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 06 - TechSupport:")
print(eda_06)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="TechSupport", hue="Churn", ax=ax)
ax.set_title("TechSupport vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_06_TechSupport.png", dpi=150)
plt.close(fig)

# EDA BLOCK 07: PaperlessBilling vs Churn

eda_07 = pd.crosstab(
    df["PaperlessBilling"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 07 - PaperlessBilling:")
print(eda_07)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="PaperlessBilling", hue="Churn", ax=ax)
ax.set_title("PaperlessBilling vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_07_PaperlessBilling.png", dpi=150)
plt.close(fig)

# EDA BLOCK 08: SeniorCitizen_Label vs Churn

eda_08 = pd.crosstab(
    df["SeniorCitizen_Label"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 08 - SeniorCitizen_Label:")
print(eda_08)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="SeniorCitizen_Label", hue="Churn", ax=ax)
ax.set_title("SeniorCitizen_Label vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_08_SeniorCitizen_Label.png", dpi=150)
plt.close(fig)

# EDA BLOCK 09: Partner vs Churn

eda_09 = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 09 - Partner:")
print(eda_09)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="Partner", hue="Churn", ax=ax)
ax.set_title("Partner vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_09_Partner.png", dpi=150)
plt.close(fig)

# EDA BLOCK 10: Dependents vs Churn

eda_10 = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 10 - Dependents:")
print(eda_10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="Dependents", hue="Churn", ax=ax)
ax.set_title("Dependents vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_10_Dependents.png", dpi=150)
plt.close(fig)

# EDA BLOCK 11: gender vs Churn

eda_11 = pd.crosstab(
    df["gender"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 11 - gender:")
print(eda_11)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="gender", hue="Churn", ax=ax)
ax.set_title("gender vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_11_gender.png", dpi=150)
plt.close(fig)

# EDA BLOCK 12: MonthlyCharge_Group vs Churn

eda_12 = pd.crosstab(
    df["MonthlyCharge_Group"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 12 - MonthlyCharge_Group:")
print(eda_12)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="MonthlyCharge_Group", hue="Churn", ax=ax)
ax.set_title("MonthlyCharge_Group vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_12_MonthlyCharge_Group.png", dpi=150)
plt.close(fig)

# EDA BLOCK 13: Customer_Value_Band vs Churn

eda_13 = pd.crosstab(
    df["Customer_Value_Band"],
    df["Churn"],
    normalize="index"
).mul(100)

print("EDA 13 - Customer_Value_Band:")
print(eda_13)

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="Customer_Value_Band", hue="Churn", ax=ax)
ax.set_title("Customer_Value_Band vs Churn")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(CHART_DIR / "eda_13_Customer_Value_Band.png", dpi=150)
plt.close(fig)

# 10. CUSTOMER SEGMENTATION

df["Tenure_Segment"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, 72],
    labels=["New", "Mid-Tenure", "Long-Term"]
)

df["Charge_Segment"] = pd.cut(
    df["MonthlyCharges"],
    bins=[-np.inf, 50, 90, np.inf],
    labels=["Low Charge", "Medium Charge", "High Charge"]
)

df["Risk_Segment"] = np.select(
    [
        (df["tenure"] <= 12) & (df["MonthlyCharges"] >= 70),
        (df["tenure"] <= 12) & (df["MonthlyCharges"] < 70),
        (df["tenure"] > 12) & (df["MonthlyCharges"] >= 90),
    ],
    [
        "High Risk: New + High Charge",
        "Medium Risk: New + Lower Charge",
        "Potential Risk: Existing + High Charge"
    ],
    default="Lower Observed Risk"
)

segment_columns = [
    "Tenure_Segment",
    "Charge_Segment",
    "Risk_Segment",
    "Customer_Value_Band"
]

for segment_column in segment_columns:
    segment_summary = (
        df.groupby(segment_column, observed=False)
        .agg(
            Customers=(CUSTOMER_ID, "nunique"),
            Churned=(TARGET_FLAG, "sum"),
            Churn_Rate=(TARGET_FLAG, "mean"),
            Avg_Monthly_Charge=("MonthlyCharges", "mean"),
            Avg_LTV=("Estimated_LTV", "mean"),
            Revenue_Lost=("Revenue_Lost", "sum")
        )
        .reset_index()
    )
    segment_summary["Churn_Rate"] *= 100
    segment_summary["Retention_Rate"] = 100 - segment_summary["Churn_Rate"]
    print(segment_column)
    print(segment_summary)

# 11. BUSINESS ANALYSIS OUTPUTS


# All analysis, visualization, machine learning, clustering, and PCA
# workflows are performed directly in Python.

# 12. MACHINE LEARNING PREPARATION

ml_df = df.copy()

drop_columns = [
    CUSTOMER_ID,
    TARGET,
    "Churn_Flag",
    "Revenue_Lost",
    "Annual_Revenue",
    "Estimated_LTV",
    "Customer_Value_Band",
    "Tenure_Group",
    "MonthlyCharge_Group",
    "SeniorCitizen_Label"
]

ml_df = ml_df.drop(columns=[c for c in drop_columns if c in ml_df.columns])

X = ml_df.copy()
y = df[TARGET_FLAG].copy()

categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Churn distribution:")
print(y.value_counts(normalize=True))

# 12. MACHINE LEARNING PREPARATION

ml_df = df.copy()
drop_columns = [
    CUSTOMER_ID, TARGET, TARGET_FLAG, "Revenue_Lost", "Annual_Revenue",
    "Estimated_LTV", "Customer_Value_Band", "Tenure_Group",
    "MonthlyCharge_Group", "SeniorCitizen_Label"
]
ml_df = ml_df.drop(columns=[c for c in drop_columns if c in ml_df.columns])
X = ml_df.copy()
y = df[TARGET_FLAG].copy()

categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# MODEL 01: LogisticRegression

model_01_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])
model_01_pipeline.fit(X_train, y_train)
pred_01 = model_01_pipeline.predict(X_test)
prob_01 = model_01_pipeline.predict_proba(X_test)[:, 1]

print("LogisticRegression")
print("Accuracy:", accuracy_score(y_test, pred_01))
print("Precision:", precision_score(y_test, pred_01, zero_division=0))
print("Recall:", recall_score(y_test, pred_01, zero_division=0))
print("F1:", f1_score(y_test, pred_01, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_01))
print(classification_report(y_test, pred_01, zero_division=0))

# MODEL 02: DecisionTree

model_02_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))
])
model_02_pipeline.fit(X_train, y_train)
pred_02 = model_02_pipeline.predict(X_test)
prob_02 = model_02_pipeline.predict_proba(X_test)[:, 1]

print("DecisionTree")
print("Accuracy:", accuracy_score(y_test, pred_02))
print("Precision:", precision_score(y_test, pred_02, zero_division=0))
print("Recall:", recall_score(y_test, pred_02, zero_division=0))
print("F1:", f1_score(y_test, pred_02, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_02))
print(classification_report(y_test, pred_02, zero_division=0))

# MODEL 03: RandomForest

model_03_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1))
])
model_03_pipeline.fit(X_train, y_train)
pred_03 = model_03_pipeline.predict(X_test)
prob_03 = model_03_pipeline.predict_proba(X_test)[:, 1]

print("RandomForest")
print("Accuracy:", accuracy_score(y_test, pred_03))
print("Precision:", precision_score(y_test, pred_03, zero_division=0))
print("Recall:", recall_score(y_test, pred_03, zero_division=0))
print("F1:", f1_score(y_test, pred_03, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_03))
print(classification_report(y_test, pred_03, zero_division=0))

# MODEL 04: GradientBoosting

model_04_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
])
model_04_pipeline.fit(X_train, y_train)
pred_04 = model_04_pipeline.predict(X_test)
prob_04 = model_04_pipeline.predict_proba(X_test)[:, 1]

print("GradientBoosting")
print("Accuracy:", accuracy_score(y_test, pred_04))
print("Precision:", precision_score(y_test, pred_04, zero_division=0))
print("Recall:", recall_score(y_test, pred_04, zero_division=0))
print("F1:", f1_score(y_test, pred_04, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_04))
print(classification_report(y_test, pred_04, zero_division=0))

# MODEL 05: ExtraTrees

model_05_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
])
model_05_pipeline.fit(X_train, y_train)
pred_05 = model_05_pipeline.predict(X_test)
prob_05 = model_05_pipeline.predict_proba(X_test)[:, 1]

print("ExtraTrees")
print("Accuracy:", accuracy_score(y_test, pred_05))
print("Precision:", precision_score(y_test, pred_05, zero_division=0))
print("Recall:", recall_score(y_test, pred_05, zero_division=0))
print("F1:", f1_score(y_test, pred_05, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, prob_05))
print(classification_report(y_test, pred_05, zero_division=0))

# 13. CROSS VALIDATION

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])
cv_accuracy = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
print("CV Accuracy:", cv_accuracy)
print("Mean CV Accuracy:", cv_accuracy.mean())
print("CV ROC-AUC:", cv_auc)
print("Mean CV ROC-AUC:", cv_auc.mean())

# 14. HYPERPARAMETER TUNING

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])
rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 8, 12, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__class_weight": [None, "balanced"]
}
rf_search = GridSearchCV(
    rf_pipeline, param_grid=rf_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
)

# 15. UNSUPERVISED LEARNING

cluster_features = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Service_Count", "Support_Count", "Streaming_Count"
]
cluster_data = df[cluster_features].fillna(df[cluster_features].median())
cluster_scaled = StandardScaler().fit_transform(cluster_data)
from sklearn.metrics import silhouette_score
inertias = []
silhouette_values = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(cluster_scaled, labels))

best_k = 3
kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["Customer_Cluster"] = kmeans_final.fit_predict(cluster_scaled)

cluster_summary = (
    df.groupby("Customer_Cluster")
    .agg(
        Customers=(CUSTOMER_ID, "nunique"),
        Avg_Tenure=("tenure", "mean"),
        Avg_MonthlyCharges=("MonthlyCharges", "mean"),
        Avg_TotalCharges=("TotalCharges", "mean"),
        Avg_Service_Count=("Service_Count", "mean"),
        Churn_Rate=(TARGET_FLAG, "mean"),
        Revenue_Lost=("Revenue_Lost", "sum")
    )
    .reset_index()
)
cluster_summary["Churn_Rate"] *= 100
print(cluster_summary)

# 16. PCA

pca_features = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Service_Count", "Support_Count", "Streaming_Count"
]
pca_data = df[pca_features].fillna(df[pca_features].median())
pca_scaled = StandardScaler().fit_transform(pca_data)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_result = pca.fit_transform(pca_scaled)
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]
print("PCA explained variance:", pca.explained_variance_ratio_)
print("PCA cumulative variance:", pca.explained_variance_ratio_.cumsum())

# 17. OUTLIER DETECTION

outlier_report = []
for feature in ["tenure", "MonthlyCharges", "TotalCharges", "Estimated_LTV"]:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[feature] < lower) | (df[feature] > upper)).sum()
    outlier_report.append({
        "Feature": feature, "Q1": q1, "Q3": q3, "IQR": iqr,
        "Lower_Bound": lower, "Upper_Bound": upper, "Outlier_Count": count
    })
outlier_report = pd.DataFrame(outlier_report)
print(outlier_report)

# BUSINESS QUERY 01: Contract

def business_query_contract():
    result = (
        df.groupby("Contract", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_01 = business_query_contract()
print(business_result_01)

# BUSINESS QUERY 02: PaymentMethod

def business_query_payment():
    result = (
        df.groupby("PaymentMethod", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_02 = business_query_payment()
print(business_result_02)

# BUSINESS QUERY 03: InternetService

def business_query_internet():
    result = (
        df.groupby("InternetService", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_03 = business_query_internet()
print(business_result_03)

# BUSINESS QUERY 04: OnlineSecurity

def business_query_security():
    result = (
        df.groupby("OnlineSecurity", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_04 = business_query_security()
print(business_result_04)

# BUSINESS QUERY 05: OnlineBackup

def business_query_backup():
    result = (
        df.groupby("OnlineBackup", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_05 = business_query_backup()
print(business_result_05)

# BUSINESS QUERY 06: DeviceProtection

def business_query_device():
    result = (
        df.groupby("DeviceProtection", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_06 = business_query_device()
print(business_result_06)

# BUSINESS QUERY 07: TechSupport

def business_query_support():
    result = (
        df.groupby("TechSupport", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_07 = business_query_support()
print(business_result_07)

# BUSINESS QUERY 08: StreamingTV

def business_query_stream_tv():
    result = (
        df.groupby("StreamingTV", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_08 = business_query_stream_tv()
print(business_result_08)

# BUSINESS QUERY 09: StreamingMovies

def business_query_stream_movies():
    result = (
        df.groupby("StreamingMovies", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_09 = business_query_stream_movies()
print(business_result_09)

# BUSINESS QUERY 10: PaperlessBilling

def business_query_paperless():
    result = (
        df.groupby("PaperlessBilling", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_10 = business_query_paperless()
print(business_result_10)

# BUSINESS QUERY 11: Partner

def business_query_partner():
    result = (
        df.groupby("Partner", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_11 = business_query_partner()
print(business_result_11)

# BUSINESS QUERY 12: Dependents

def business_query_dependents():
    result = (
        df.groupby("Dependents", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_12 = business_query_dependents()
print(business_result_12)

# BUSINESS QUERY 13: gender

def business_query_gender():
    result = (
        df.groupby("gender", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_13 = business_query_gender()
print(business_result_13)

# BUSINESS QUERY 14: SeniorCitizen_Label

def business_query_senior():
    result = (
        df.groupby("SeniorCitizen_Label", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_14 = business_query_senior()
print(business_result_14)

# BUSINESS QUERY 15: Tenure_Segment

def business_query_tenure_segment():
    result = (
        df.groupby("Tenure_Segment", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_15 = business_query_tenure_segment()
print(business_result_15)

# BUSINESS QUERY 16: Charge_Segment

def business_query_charge_segment():
    result = (
        df.groupby("Charge_Segment", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_16 = business_query_charge_segment()
print(business_result_16)

# BUSINESS QUERY 17: Risk_Segment

def business_query_risk_segment():
    result = (
        df.groupby("Risk_Segment", dropna=False)
        .agg(
            customer_count=(CUSTOMER_ID, "nunique"),
            churned_count=(TARGET_FLAG, "sum"),
            retained_count=(TARGET_FLAG, lambda s: int((s == 0).sum())),
            churn_rate=(TARGET_FLAG, "mean"),
            retention_rate=(TARGET_FLAG, lambda s: 1 - s.mean()),
            monthly_revenue=("MonthlyCharges", "sum"),
            annual_revenue=("Annual_Revenue", "sum"),
            revenue_lost=("Revenue_Lost", "sum"),
            average_tenure=("tenure", "mean"),
            average_monthly_charge=("MonthlyCharges", "mean"),
            average_ltv=("Estimated_LTV", "mean")
        )
        .reset_index()
    )
    result["churn_rate"] *= 100
    result["retention_rate"] *= 100
    return result.sort_values("churn_rate", ascending=False)

business_result_17 = business_query_risk_segment()
print(business_result_17)
# RECIPE 0001
recipe_0001_summary = (
    df.groupby("Contract", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0001_summary["share_of_total"] = (
    recipe_0001_summary["sum"] /
    recipe_0001_summary["sum"].sum()
)
recipe_0001_summary["metric_rank"] = (
    recipe_0001_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0002
recipe_0002_summary = (
    df.groupby("Contract", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0002_summary["share_of_total"] = (
    recipe_0002_summary["sum"] /
    recipe_0002_summary["sum"].sum()
)
recipe_0002_summary["metric_rank"] = (
    recipe_0002_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0003
recipe_0003_summary = (
    df.groupby("Contract", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0003_summary["share_of_total"] = (
    recipe_0003_summary["sum"] /
    recipe_0003_summary["sum"].sum()
)
recipe_0003_summary["metric_rank"] = (
    recipe_0003_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0004
recipe_0004_summary = (
    df.groupby("Contract", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0004_summary["share_of_total"] = (
    recipe_0004_summary["sum"] /
    recipe_0004_summary["sum"].sum()
)
recipe_0004_summary["metric_rank"] = (
    recipe_0004_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0005
recipe_0005_summary = (
    df.groupby("Contract", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0005_summary["share_of_total"] = (
    recipe_0005_summary["sum"] /
    recipe_0005_summary["sum"].sum()
)
recipe_0005_summary["metric_rank"] = (
    recipe_0005_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0006
recipe_0006_summary = (
    df.groupby("Contract", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0006_summary["share_of_total"] = (
    recipe_0006_summary["sum"] /
    recipe_0006_summary["sum"].sum()
)
recipe_0006_summary["metric_rank"] = (
    recipe_0006_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0007
recipe_0007_summary = (
    df.groupby("Contract", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0007_summary["share_of_total"] = (
    recipe_0007_summary["sum"] /
    recipe_0007_summary["sum"].sum()
)
recipe_0007_summary["metric_rank"] = (
    recipe_0007_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0008
recipe_0008_summary = (
    df.groupby("Contract", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0008_summary["share_of_total"] = (
    recipe_0008_summary["sum"] /
    recipe_0008_summary["sum"].sum()
)
recipe_0008_summary["metric_rank"] = (
    recipe_0008_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0009
recipe_0009_summary = (
    df.groupby("Contract", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0009_summary["share_of_total"] = (
    recipe_0009_summary["sum"] /
    recipe_0009_summary["sum"].sum()
)
recipe_0009_summary["metric_rank"] = (
    recipe_0009_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0010
recipe_0010_summary = (
    df.groupby("PaymentMethod", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0010_summary["share_of_total"] = (
    recipe_0010_summary["sum"] /
    recipe_0010_summary["sum"].sum()
)
recipe_0010_summary["metric_rank"] = (
    recipe_0010_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0011
recipe_0011_summary = (
    df.groupby("PaymentMethod", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0011_summary["share_of_total"] = (
    recipe_0011_summary["sum"] /
    recipe_0011_summary["sum"].sum()
)
recipe_0011_summary["metric_rank"] = (
    recipe_0011_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0012
recipe_0012_summary = (
    df.groupby("PaymentMethod", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0012_summary["share_of_total"] = (
    recipe_0012_summary["sum"] /
    recipe_0012_summary["sum"].sum()
)
recipe_0012_summary["metric_rank"] = (
    recipe_0012_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0013
recipe_0013_summary = (
    df.groupby("PaymentMethod", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0013_summary["share_of_total"] = (
    recipe_0013_summary["sum"] /
    recipe_0013_summary["sum"].sum()
)
recipe_0013_summary["metric_rank"] = (
    recipe_0013_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0014
recipe_0014_summary = (
    df.groupby("PaymentMethod", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0014_summary["share_of_total"] = (
    recipe_0014_summary["sum"] /
    recipe_0014_summary["sum"].sum()
)
recipe_0014_summary["metric_rank"] = (
    recipe_0014_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0015
recipe_0015_summary = (
    df.groupby("PaymentMethod", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0015_summary["share_of_total"] = (
    recipe_0015_summary["sum"] /
    recipe_0015_summary["sum"].sum()
)
recipe_0015_summary["metric_rank"] = (
    recipe_0015_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0016
recipe_0016_summary = (
    df.groupby("PaymentMethod", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0016_summary["share_of_total"] = (
    recipe_0016_summary["sum"] /
    recipe_0016_summary["sum"].sum()
)
recipe_0016_summary["metric_rank"] = (
    recipe_0016_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0017
recipe_0017_summary = (
    df.groupby("PaymentMethod", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0017_summary["share_of_total"] = (
    recipe_0017_summary["sum"] /
    recipe_0017_summary["sum"].sum()
)
recipe_0017_summary["metric_rank"] = (
    recipe_0017_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0018
recipe_0018_summary = (
    df.groupby("PaymentMethod", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0018_summary["share_of_total"] = (
    recipe_0018_summary["sum"] /
    recipe_0018_summary["sum"].sum()
)
recipe_0018_summary["metric_rank"] = (
    recipe_0018_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0019
recipe_0019_summary = (
    df.groupby("InternetService", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0019_summary["share_of_total"] = (
    recipe_0019_summary["sum"] /
    recipe_0019_summary["sum"].sum()
)
recipe_0019_summary["metric_rank"] = (
    recipe_0019_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0020
recipe_0020_summary = (
    df.groupby("InternetService", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0020_summary["share_of_total"] = (
    recipe_0020_summary["sum"] /
    recipe_0020_summary["sum"].sum()
)
recipe_0020_summary["metric_rank"] = (
    recipe_0020_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0021
recipe_0021_summary = (
    df.groupby("InternetService", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0021_summary["share_of_total"] = (
    recipe_0021_summary["sum"] /
    recipe_0021_summary["sum"].sum()
)
recipe_0021_summary["metric_rank"] = (
    recipe_0021_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0022
recipe_0022_summary = (
    df.groupby("InternetService", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0022_summary["share_of_total"] = (
    recipe_0022_summary["sum"] /
    recipe_0022_summary["sum"].sum()
)
recipe_0022_summary["metric_rank"] = (
    recipe_0022_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0023
recipe_0023_summary = (
    df.groupby("InternetService", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0023_summary["share_of_total"] = (
    recipe_0023_summary["sum"] /
    recipe_0023_summary["sum"].sum()
)
recipe_0023_summary["metric_rank"] = (
    recipe_0023_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0024
recipe_0024_summary = (
    df.groupby("InternetService", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0024_summary["share_of_total"] = (
    recipe_0024_summary["sum"] /
    recipe_0024_summary["sum"].sum()
)
recipe_0024_summary["metric_rank"] = (
    recipe_0024_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0025
recipe_0025_summary = (
    df.groupby("InternetService", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0025_summary["share_of_total"] = (
    recipe_0025_summary["sum"] /
    recipe_0025_summary["sum"].sum()
)
recipe_0025_summary["metric_rank"] = (
    recipe_0025_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0026
recipe_0026_summary = (
    df.groupby("InternetService", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0026_summary["share_of_total"] = (
    recipe_0026_summary["sum"] /
    recipe_0026_summary["sum"].sum()
)
recipe_0026_summary["metric_rank"] = (
    recipe_0026_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0027
recipe_0027_summary = (
    df.groupby("InternetService", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0027_summary["share_of_total"] = (
    recipe_0027_summary["sum"] /
    recipe_0027_summary["sum"].sum()
)
recipe_0027_summary["metric_rank"] = (
    recipe_0027_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0028
recipe_0028_summary = (
    df.groupby("OnlineSecurity", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0028_summary["share_of_total"] = (
    recipe_0028_summary["sum"] /
    recipe_0028_summary["sum"].sum()
)
recipe_0028_summary["metric_rank"] = (
    recipe_0028_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0029
recipe_0029_summary = (
    df.groupby("OnlineSecurity", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0029_summary["share_of_total"] = (
    recipe_0029_summary["sum"] /
    recipe_0029_summary["sum"].sum()
)
recipe_0029_summary["metric_rank"] = (
    recipe_0029_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0030
recipe_0030_summary = (
    df.groupby("OnlineSecurity", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0030_summary["share_of_total"] = (
    recipe_0030_summary["sum"] /
    recipe_0030_summary["sum"].sum()
)
recipe_0030_summary["metric_rank"] = (
    recipe_0030_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0031
recipe_0031_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0031_summary["share_of_total"] = (
    recipe_0031_summary["sum"] /
    recipe_0031_summary["sum"].sum()
)
recipe_0031_summary["metric_rank"] = (
    recipe_0031_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0032
recipe_0032_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0032_summary["share_of_total"] = (
    recipe_0032_summary["sum"] /
    recipe_0032_summary["sum"].sum()
)
recipe_0032_summary["metric_rank"] = (
    recipe_0032_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0033
recipe_0033_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0033_summary["share_of_total"] = (
    recipe_0033_summary["sum"] /
    recipe_0033_summary["sum"].sum()
)
recipe_0033_summary["metric_rank"] = (
    recipe_0033_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0034
recipe_0034_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0034_summary["share_of_total"] = (
    recipe_0034_summary["sum"] /
    recipe_0034_summary["sum"].sum()
)
recipe_0034_summary["metric_rank"] = (
    recipe_0034_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0035
recipe_0035_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0035_summary["share_of_total"] = (
    recipe_0035_summary["sum"] /
    recipe_0035_summary["sum"].sum()
)
recipe_0035_summary["metric_rank"] = (
    recipe_0035_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0036
recipe_0036_summary = (
    df.groupby("OnlineSecurity", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0036_summary["share_of_total"] = (
    recipe_0036_summary["sum"] /
    recipe_0036_summary["sum"].sum()
)
recipe_0036_summary["metric_rank"] = (
    recipe_0036_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0037
recipe_0037_summary = (
    df.groupby("OnlineBackup", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0037_summary["share_of_total"] = (
    recipe_0037_summary["sum"] /
    recipe_0037_summary["sum"].sum()
)
recipe_0037_summary["metric_rank"] = (
    recipe_0037_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0038
recipe_0038_summary = (
    df.groupby("OnlineBackup", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0038_summary["share_of_total"] = (
    recipe_0038_summary["sum"] /
    recipe_0038_summary["sum"].sum()
)
recipe_0038_summary["metric_rank"] = (
    recipe_0038_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0039
recipe_0039_summary = (
    df.groupby("OnlineBackup", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0039_summary["share_of_total"] = (
    recipe_0039_summary["sum"] /
    recipe_0039_summary["sum"].sum()
)
recipe_0039_summary["metric_rank"] = (
    recipe_0039_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0040
recipe_0040_summary = (
    df.groupby("OnlineBackup", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0040_summary["share_of_total"] = (
    recipe_0040_summary["sum"] /
    recipe_0040_summary["sum"].sum()
)
recipe_0040_summary["metric_rank"] = (
    recipe_0040_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0041
recipe_0041_summary = (
    df.groupby("OnlineBackup", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0041_summary["share_of_total"] = (
    recipe_0041_summary["sum"] /
    recipe_0041_summary["sum"].sum()
)
recipe_0041_summary["metric_rank"] = (
    recipe_0041_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0042
recipe_0042_summary = (
    df.groupby("OnlineBackup", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0042_summary["share_of_total"] = (
    recipe_0042_summary["sum"] /
    recipe_0042_summary["sum"].sum()
)
recipe_0042_summary["metric_rank"] = (
    recipe_0042_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0043
recipe_0043_summary = (
    df.groupby("OnlineBackup", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0043_summary["share_of_total"] = (
    recipe_0043_summary["sum"] /
    recipe_0043_summary["sum"].sum()
)
recipe_0043_summary["metric_rank"] = (
    recipe_0043_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0044
recipe_0044_summary = (
    df.groupby("OnlineBackup", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0044_summary["share_of_total"] = (
    recipe_0044_summary["sum"] /
    recipe_0044_summary["sum"].sum()
)
recipe_0044_summary["metric_rank"] = (
    recipe_0044_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0045
recipe_0045_summary = (
    df.groupby("OnlineBackup", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0045_summary["share_of_total"] = (
    recipe_0045_summary["sum"] /
    recipe_0045_summary["sum"].sum()
)
recipe_0045_summary["metric_rank"] = (
    recipe_0045_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0046
recipe_0046_summary = (
    df.groupby("DeviceProtection", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0046_summary["share_of_total"] = (
    recipe_0046_summary["sum"] /
    recipe_0046_summary["sum"].sum()
)
recipe_0046_summary["metric_rank"] = (
    recipe_0046_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0047
recipe_0047_summary = (
    df.groupby("DeviceProtection", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0047_summary["share_of_total"] = (
    recipe_0047_summary["sum"] /
    recipe_0047_summary["sum"].sum()
)
recipe_0047_summary["metric_rank"] = (
    recipe_0047_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0048
recipe_0048_summary = (
    df.groupby("DeviceProtection", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0048_summary["share_of_total"] = (
    recipe_0048_summary["sum"] /
    recipe_0048_summary["sum"].sum()
)
recipe_0048_summary["metric_rank"] = (
    recipe_0048_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0049
recipe_0049_summary = (
    df.groupby("DeviceProtection", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0049_summary["share_of_total"] = (
    recipe_0049_summary["sum"] /
    recipe_0049_summary["sum"].sum()
)
recipe_0049_summary["metric_rank"] = (
    recipe_0049_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0050
recipe_0050_summary = (
    df.groupby("DeviceProtection", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0050_summary["share_of_total"] = (
    recipe_0050_summary["sum"] /
    recipe_0050_summary["sum"].sum()
)
recipe_0050_summary["metric_rank"] = (
    recipe_0050_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0051
recipe_0051_summary = (
    df.groupby("DeviceProtection", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0051_summary["share_of_total"] = (
    recipe_0051_summary["sum"] /
    recipe_0051_summary["sum"].sum()
)
recipe_0051_summary["metric_rank"] = (
    recipe_0051_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0052
recipe_0052_summary = (
    df.groupby("DeviceProtection", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0052_summary["share_of_total"] = (
    recipe_0052_summary["sum"] /
    recipe_0052_summary["sum"].sum()
)
recipe_0052_summary["metric_rank"] = (
    recipe_0052_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0053
recipe_0053_summary = (
    df.groupby("DeviceProtection", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0053_summary["share_of_total"] = (
    recipe_0053_summary["sum"] /
    recipe_0053_summary["sum"].sum()
)
recipe_0053_summary["metric_rank"] = (
    recipe_0053_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0054
recipe_0054_summary = (
    df.groupby("DeviceProtection", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0054_summary["share_of_total"] = (
    recipe_0054_summary["sum"] /
    recipe_0054_summary["sum"].sum()
)
recipe_0054_summary["metric_rank"] = (
    recipe_0054_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0055
recipe_0055_summary = (
    df.groupby("TechSupport", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0055_summary["share_of_total"] = (
    recipe_0055_summary["sum"] /
    recipe_0055_summary["sum"].sum()
)
recipe_0055_summary["metric_rank"] = (
    recipe_0055_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0056
recipe_0056_summary = (
    df.groupby("TechSupport", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0056_summary["share_of_total"] = (
    recipe_0056_summary["sum"] /
    recipe_0056_summary["sum"].sum()
)
recipe_0056_summary["metric_rank"] = (
    recipe_0056_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0057
recipe_0057_summary = (
    df.groupby("TechSupport", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0057_summary["share_of_total"] = (
    recipe_0057_summary["sum"] /
    recipe_0057_summary["sum"].sum()
)
recipe_0057_summary["metric_rank"] = (
    recipe_0057_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0058
recipe_0058_summary = (
    df.groupby("TechSupport", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0058_summary["share_of_total"] = (
    recipe_0058_summary["sum"] /
    recipe_0058_summary["sum"].sum()
)
recipe_0058_summary["metric_rank"] = (
    recipe_0058_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0059
recipe_0059_summary = (
    df.groupby("TechSupport", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0059_summary["share_of_total"] = (
    recipe_0059_summary["sum"] /
    recipe_0059_summary["sum"].sum()
)
recipe_0059_summary["metric_rank"] = (
    recipe_0059_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0060
recipe_0060_summary = (
    df.groupby("TechSupport", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0060_summary["share_of_total"] = (
    recipe_0060_summary["sum"] /
    recipe_0060_summary["sum"].sum()
)
recipe_0060_summary["metric_rank"] = (
    recipe_0060_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0061
recipe_0061_summary = (
    df.groupby("TechSupport", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0061_summary["share_of_total"] = (
    recipe_0061_summary["sum"] /
    recipe_0061_summary["sum"].sum()
)
recipe_0061_summary["metric_rank"] = (
    recipe_0061_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0062
recipe_0062_summary = (
    df.groupby("TechSupport", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0062_summary["share_of_total"] = (
    recipe_0062_summary["sum"] /
    recipe_0062_summary["sum"].sum()
)
recipe_0062_summary["metric_rank"] = (
    recipe_0062_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0063
recipe_0063_summary = (
    df.groupby("TechSupport", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0063_summary["share_of_total"] = (
    recipe_0063_summary["sum"] /
    recipe_0063_summary["sum"].sum()
)
recipe_0063_summary["metric_rank"] = (
    recipe_0063_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0064
recipe_0064_summary = (
    df.groupby("StreamingTV", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0064_summary["share_of_total"] = (
    recipe_0064_summary["sum"] /
    recipe_0064_summary["sum"].sum()
)
recipe_0064_summary["metric_rank"] = (
    recipe_0064_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0065
recipe_0065_summary = (
    df.groupby("StreamingTV", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0065_summary["share_of_total"] = (
    recipe_0065_summary["sum"] /
    recipe_0065_summary["sum"].sum()
)
recipe_0065_summary["metric_rank"] = (
    recipe_0065_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0066
recipe_0066_summary = (
    df.groupby("StreamingTV", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0066_summary["share_of_total"] = (
    recipe_0066_summary["sum"] /
    recipe_0066_summary["sum"].sum()
)
recipe_0066_summary["metric_rank"] = (
    recipe_0066_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0067
recipe_0067_summary = (
    df.groupby("StreamingTV", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0067_summary["share_of_total"] = (
    recipe_0067_summary["sum"] /
    recipe_0067_summary["sum"].sum()
)
recipe_0067_summary["metric_rank"] = (
    recipe_0067_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0068
recipe_0068_summary = (
    df.groupby("StreamingTV", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0068_summary["share_of_total"] = (
    recipe_0068_summary["sum"] /
    recipe_0068_summary["sum"].sum()
)
recipe_0068_summary["metric_rank"] = (
    recipe_0068_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0069
recipe_0069_summary = (
    df.groupby("StreamingTV", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0069_summary["share_of_total"] = (
    recipe_0069_summary["sum"] /
    recipe_0069_summary["sum"].sum()
)
recipe_0069_summary["metric_rank"] = (
    recipe_0069_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0070
recipe_0070_summary = (
    df.groupby("StreamingTV", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0070_summary["share_of_total"] = (
    recipe_0070_summary["sum"] /
    recipe_0070_summary["sum"].sum()
)
recipe_0070_summary["metric_rank"] = (
    recipe_0070_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0071
recipe_0071_summary = (
    df.groupby("StreamingTV", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0071_summary["share_of_total"] = (
    recipe_0071_summary["sum"] /
    recipe_0071_summary["sum"].sum()
)
recipe_0071_summary["metric_rank"] = (
    recipe_0071_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0072
recipe_0072_summary = (
    df.groupby("StreamingTV", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0072_summary["share_of_total"] = (
    recipe_0072_summary["sum"] /
    recipe_0072_summary["sum"].sum()
)
recipe_0072_summary["metric_rank"] = (
    recipe_0072_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0073
recipe_0073_summary = (
    df.groupby("StreamingMovies", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0073_summary["share_of_total"] = (
    recipe_0073_summary["sum"] /
    recipe_0073_summary["sum"].sum()
)
recipe_0073_summary["metric_rank"] = (
    recipe_0073_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0074
recipe_0074_summary = (
    df.groupby("StreamingMovies", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0074_summary["share_of_total"] = (
    recipe_0074_summary["sum"] /
    recipe_0074_summary["sum"].sum()
)
recipe_0074_summary["metric_rank"] = (
    recipe_0074_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0075
recipe_0075_summary = (
    df.groupby("StreamingMovies", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0075_summary["share_of_total"] = (
    recipe_0075_summary["sum"] /
    recipe_0075_summary["sum"].sum()
)
recipe_0075_summary["metric_rank"] = (
    recipe_0075_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0076
recipe_0076_summary = (
    df.groupby("StreamingMovies", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0076_summary["share_of_total"] = (
    recipe_0076_summary["sum"] /
    recipe_0076_summary["sum"].sum()
)
recipe_0076_summary["metric_rank"] = (
    recipe_0076_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0077
recipe_0077_summary = (
    df.groupby("StreamingMovies", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0077_summary["share_of_total"] = (
    recipe_0077_summary["sum"] /
    recipe_0077_summary["sum"].sum()
)
recipe_0077_summary["metric_rank"] = (
    recipe_0077_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0078
recipe_0078_summary = (
    df.groupby("StreamingMovies", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0078_summary["share_of_total"] = (
    recipe_0078_summary["sum"] /
    recipe_0078_summary["sum"].sum()
)
recipe_0078_summary["metric_rank"] = (
    recipe_0078_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0079
recipe_0079_summary = (
    df.groupby("StreamingMovies", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0079_summary["share_of_total"] = (
    recipe_0079_summary["sum"] /
    recipe_0079_summary["sum"].sum()
)
recipe_0079_summary["metric_rank"] = (
    recipe_0079_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0080
recipe_0080_summary = (
    df.groupby("StreamingMovies", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0080_summary["share_of_total"] = (
    recipe_0080_summary["sum"] /
    recipe_0080_summary["sum"].sum()
)
recipe_0080_summary["metric_rank"] = (
    recipe_0080_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0081
recipe_0081_summary = (
    df.groupby("StreamingMovies", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0081_summary["share_of_total"] = (
    recipe_0081_summary["sum"] /
    recipe_0081_summary["sum"].sum()
)
recipe_0081_summary["metric_rank"] = (
    recipe_0081_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0082
recipe_0082_summary = (
    df.groupby("PhoneService", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0082_summary["share_of_total"] = (
    recipe_0082_summary["sum"] /
    recipe_0082_summary["sum"].sum()
)
recipe_0082_summary["metric_rank"] = (
    recipe_0082_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0083
recipe_0083_summary = (
    df.groupby("PhoneService", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0083_summary["share_of_total"] = (
    recipe_0083_summary["sum"] /
    recipe_0083_summary["sum"].sum()
)
recipe_0083_summary["metric_rank"] = (
    recipe_0083_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0084
recipe_0084_summary = (
    df.groupby("PhoneService", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0084_summary["share_of_total"] = (
    recipe_0084_summary["sum"] /
    recipe_0084_summary["sum"].sum()
)
recipe_0084_summary["metric_rank"] = (
    recipe_0084_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0085
recipe_0085_summary = (
    df.groupby("PhoneService", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0085_summary["share_of_total"] = (
    recipe_0085_summary["sum"] /
    recipe_0085_summary["sum"].sum()
)
recipe_0085_summary["metric_rank"] = (
    recipe_0085_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0086
recipe_0086_summary = (
    df.groupby("PhoneService", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0086_summary["share_of_total"] = (
    recipe_0086_summary["sum"] /
    recipe_0086_summary["sum"].sum()
)
recipe_0086_summary["metric_rank"] = (
    recipe_0086_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0087
recipe_0087_summary = (
    df.groupby("PhoneService", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0087_summary["share_of_total"] = (
    recipe_0087_summary["sum"] /
    recipe_0087_summary["sum"].sum()
)
recipe_0087_summary["metric_rank"] = (
    recipe_0087_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0088
recipe_0088_summary = (
    df.groupby("PhoneService", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0088_summary["share_of_total"] = (
    recipe_0088_summary["sum"] /
    recipe_0088_summary["sum"].sum()
)
recipe_0088_summary["metric_rank"] = (
    recipe_0088_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0089
recipe_0089_summary = (
    df.groupby("PhoneService", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0089_summary["share_of_total"] = (
    recipe_0089_summary["sum"] /
    recipe_0089_summary["sum"].sum()
)
recipe_0089_summary["metric_rank"] = (
    recipe_0089_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0090
recipe_0090_summary = (
    df.groupby("PhoneService", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0090_summary["share_of_total"] = (
    recipe_0090_summary["sum"] /
    recipe_0090_summary["sum"].sum()
)
recipe_0090_summary["metric_rank"] = (
    recipe_0090_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0091
recipe_0091_summary = (
    df.groupby("MultipleLines", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0091_summary["share_of_total"] = (
    recipe_0091_summary["sum"] /
    recipe_0091_summary["sum"].sum()
)
recipe_0091_summary["metric_rank"] = (
    recipe_0091_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0092
recipe_0092_summary = (
    df.groupby("MultipleLines", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0092_summary["share_of_total"] = (
    recipe_0092_summary["sum"] /
    recipe_0092_summary["sum"].sum()
)
recipe_0092_summary["metric_rank"] = (
    recipe_0092_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0093
recipe_0093_summary = (
    df.groupby("MultipleLines", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0093_summary["share_of_total"] = (
    recipe_0093_summary["sum"] /
    recipe_0093_summary["sum"].sum()
)
recipe_0093_summary["metric_rank"] = (
    recipe_0093_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0094
recipe_0094_summary = (
    df.groupby("MultipleLines", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0094_summary["share_of_total"] = (
    recipe_0094_summary["sum"] /
    recipe_0094_summary["sum"].sum()
)
recipe_0094_summary["metric_rank"] = (
    recipe_0094_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0095
recipe_0095_summary = (
    df.groupby("MultipleLines", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0095_summary["share_of_total"] = (
    recipe_0095_summary["sum"] /
    recipe_0095_summary["sum"].sum()
)
recipe_0095_summary["metric_rank"] = (
    recipe_0095_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0096
recipe_0096_summary = (
    df.groupby("MultipleLines", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0096_summary["share_of_total"] = (
    recipe_0096_summary["sum"] /
    recipe_0096_summary["sum"].sum()
)
recipe_0096_summary["metric_rank"] = (
    recipe_0096_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0097
recipe_0097_summary = (
    df.groupby("MultipleLines", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0097_summary["share_of_total"] = (
    recipe_0097_summary["sum"] /
    recipe_0097_summary["sum"].sum()
)
recipe_0097_summary["metric_rank"] = (
    recipe_0097_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0098
recipe_0098_summary = (
    df.groupby("MultipleLines", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0098_summary["share_of_total"] = (
    recipe_0098_summary["sum"] /
    recipe_0098_summary["sum"].sum()
)
recipe_0098_summary["metric_rank"] = (
    recipe_0098_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0099
recipe_0099_summary = (
    df.groupby("MultipleLines", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0099_summary["share_of_total"] = (
    recipe_0099_summary["sum"] /
    recipe_0099_summary["sum"].sum()
)
recipe_0099_summary["metric_rank"] = (
    recipe_0099_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0100
recipe_0100_summary = (
    df.groupby("PaperlessBilling", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0100_summary["share_of_total"] = (
    recipe_0100_summary["sum"] /
    recipe_0100_summary["sum"].sum()
)
recipe_0100_summary["metric_rank"] = (
    recipe_0100_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0101
recipe_0101_summary = (
    df.groupby("PaperlessBilling", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0101_summary["share_of_total"] = (
    recipe_0101_summary["sum"] /
    recipe_0101_summary["sum"].sum()
)
recipe_0101_summary["metric_rank"] = (
    recipe_0101_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0102
recipe_0102_summary = (
    df.groupby("PaperlessBilling", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0102_summary["share_of_total"] = (
    recipe_0102_summary["sum"] /
    recipe_0102_summary["sum"].sum()
)
recipe_0102_summary["metric_rank"] = (
    recipe_0102_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0103
recipe_0103_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0103_summary["share_of_total"] = (
    recipe_0103_summary["sum"] /
    recipe_0103_summary["sum"].sum()
)
recipe_0103_summary["metric_rank"] = (
    recipe_0103_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0104
recipe_0104_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0104_summary["share_of_total"] = (
    recipe_0104_summary["sum"] /
    recipe_0104_summary["sum"].sum()
)
recipe_0104_summary["metric_rank"] = (
    recipe_0104_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0105
recipe_0105_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0105_summary["share_of_total"] = (
    recipe_0105_summary["sum"] /
    recipe_0105_summary["sum"].sum()
)
recipe_0105_summary["metric_rank"] = (
    recipe_0105_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0106
recipe_0106_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0106_summary["share_of_total"] = (
    recipe_0106_summary["sum"] /
    recipe_0106_summary["sum"].sum()
)
recipe_0106_summary["metric_rank"] = (
    recipe_0106_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0107
recipe_0107_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0107_summary["share_of_total"] = (
    recipe_0107_summary["sum"] /
    recipe_0107_summary["sum"].sum()
)
recipe_0107_summary["metric_rank"] = (
    recipe_0107_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0108
recipe_0108_summary = (
    df.groupby("PaperlessBilling", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0108_summary["share_of_total"] = (
    recipe_0108_summary["sum"] /
    recipe_0108_summary["sum"].sum()
)
recipe_0108_summary["metric_rank"] = (
    recipe_0108_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0109
recipe_0109_summary = (
    df.groupby("Partner", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0109_summary["share_of_total"] = (
    recipe_0109_summary["sum"] /
    recipe_0109_summary["sum"].sum()
)
recipe_0109_summary["metric_rank"] = (
    recipe_0109_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0110
recipe_0110_summary = (
    df.groupby("Partner", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0110_summary["share_of_total"] = (
    recipe_0110_summary["sum"] /
    recipe_0110_summary["sum"].sum()
)
recipe_0110_summary["metric_rank"] = (
    recipe_0110_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0111
recipe_0111_summary = (
    df.groupby("Partner", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0111_summary["share_of_total"] = (
    recipe_0111_summary["sum"] /
    recipe_0111_summary["sum"].sum()
)
recipe_0111_summary["metric_rank"] = (
    recipe_0111_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0112
recipe_0112_summary = (
    df.groupby("Partner", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0112_summary["share_of_total"] = (
    recipe_0112_summary["sum"] /
    recipe_0112_summary["sum"].sum()
)
recipe_0112_summary["metric_rank"] = (
    recipe_0112_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0113
recipe_0113_summary = (
    df.groupby("Partner", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0113_summary["share_of_total"] = (
    recipe_0113_summary["sum"] /
    recipe_0113_summary["sum"].sum()
)
recipe_0113_summary["metric_rank"] = (
    recipe_0113_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0114
recipe_0114_summary = (
    df.groupby("Partner", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0114_summary["share_of_total"] = (
    recipe_0114_summary["sum"] /
    recipe_0114_summary["sum"].sum()
)
recipe_0114_summary["metric_rank"] = (
    recipe_0114_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0115
recipe_0115_summary = (
    df.groupby("Partner", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0115_summary["share_of_total"] = (
    recipe_0115_summary["sum"] /
    recipe_0115_summary["sum"].sum()
)
recipe_0115_summary["metric_rank"] = (
    recipe_0115_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0116
recipe_0116_summary = (
    df.groupby("Partner", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0116_summary["share_of_total"] = (
    recipe_0116_summary["sum"] /
    recipe_0116_summary["sum"].sum()
)
recipe_0116_summary["metric_rank"] = (
    recipe_0116_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0117
recipe_0117_summary = (
    df.groupby("Partner", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0117_summary["share_of_total"] = (
    recipe_0117_summary["sum"] /
    recipe_0117_summary["sum"].sum()
)
recipe_0117_summary["metric_rank"] = (
    recipe_0117_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0118
recipe_0118_summary = (
    df.groupby("Dependents", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0118_summary["share_of_total"] = (
    recipe_0118_summary["sum"] /
    recipe_0118_summary["sum"].sum()
)
recipe_0118_summary["metric_rank"] = (
    recipe_0118_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0119
recipe_0119_summary = (
    df.groupby("Dependents", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0119_summary["share_of_total"] = (
    recipe_0119_summary["sum"] /
    recipe_0119_summary["sum"].sum()
)
recipe_0119_summary["metric_rank"] = (
    recipe_0119_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0120
recipe_0120_summary = (
    df.groupby("Dependents", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0120_summary["share_of_total"] = (
    recipe_0120_summary["sum"] /
    recipe_0120_summary["sum"].sum()
)
recipe_0120_summary["metric_rank"] = (
    recipe_0120_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0121
recipe_0121_summary = (
    df.groupby("Dependents", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0121_summary["share_of_total"] = (
    recipe_0121_summary["sum"] /
    recipe_0121_summary["sum"].sum()
)
recipe_0121_summary["metric_rank"] = (
    recipe_0121_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0122
recipe_0122_summary = (
    df.groupby("Dependents", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0122_summary["share_of_total"] = (
    recipe_0122_summary["sum"] /
    recipe_0122_summary["sum"].sum()
)
recipe_0122_summary["metric_rank"] = (
    recipe_0122_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0123
recipe_0123_summary = (
    df.groupby("Dependents", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0123_summary["share_of_total"] = (
    recipe_0123_summary["sum"] /
    recipe_0123_summary["sum"].sum()
)
recipe_0123_summary["metric_rank"] = (
    recipe_0123_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0124
recipe_0124_summary = (
    df.groupby("Dependents", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0124_summary["share_of_total"] = (
    recipe_0124_summary["sum"] /
    recipe_0124_summary["sum"].sum()
)
recipe_0124_summary["metric_rank"] = (
    recipe_0124_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0125
recipe_0125_summary = (
    df.groupby("Dependents", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0125_summary["share_of_total"] = (
    recipe_0125_summary["sum"] /
    recipe_0125_summary["sum"].sum()
)
recipe_0125_summary["metric_rank"] = (
    recipe_0125_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0126
recipe_0126_summary = (
    df.groupby("Dependents", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0126_summary["share_of_total"] = (
    recipe_0126_summary["sum"] /
    recipe_0126_summary["sum"].sum()
)
recipe_0126_summary["metric_rank"] = (
    recipe_0126_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0127
recipe_0127_summary = (
    df.groupby("gender", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0127_summary["share_of_total"] = (
    recipe_0127_summary["sum"] /
    recipe_0127_summary["sum"].sum()
)
recipe_0127_summary["metric_rank"] = (
    recipe_0127_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0128
recipe_0128_summary = (
    df.groupby("gender", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0128_summary["share_of_total"] = (
    recipe_0128_summary["sum"] /
    recipe_0128_summary["sum"].sum()
)
recipe_0128_summary["metric_rank"] = (
    recipe_0128_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0129
recipe_0129_summary = (
    df.groupby("gender", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0129_summary["share_of_total"] = (
    recipe_0129_summary["sum"] /
    recipe_0129_summary["sum"].sum()
)
recipe_0129_summary["metric_rank"] = (
    recipe_0129_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0130
recipe_0130_summary = (
    df.groupby("gender", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0130_summary["share_of_total"] = (
    recipe_0130_summary["sum"] /
    recipe_0130_summary["sum"].sum()
)
recipe_0130_summary["metric_rank"] = (
    recipe_0130_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0131
recipe_0131_summary = (
    df.groupby("gender", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0131_summary["share_of_total"] = (
    recipe_0131_summary["sum"] /
    recipe_0131_summary["sum"].sum()
)
recipe_0131_summary["metric_rank"] = (
    recipe_0131_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0132
recipe_0132_summary = (
    df.groupby("gender", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0132_summary["share_of_total"] = (
    recipe_0132_summary["sum"] /
    recipe_0132_summary["sum"].sum()
)
recipe_0132_summary["metric_rank"] = (
    recipe_0132_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0133
recipe_0133_summary = (
    df.groupby("gender", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0133_summary["share_of_total"] = (
    recipe_0133_summary["sum"] /
    recipe_0133_summary["sum"].sum()
)
recipe_0133_summary["metric_rank"] = (
    recipe_0133_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0134
recipe_0134_summary = (
    df.groupby("gender", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0134_summary["share_of_total"] = (
    recipe_0134_summary["sum"] /
    recipe_0134_summary["sum"].sum()
)
recipe_0134_summary["metric_rank"] = (
    recipe_0134_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0135
recipe_0135_summary = (
    df.groupby("gender", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0135_summary["share_of_total"] = (
    recipe_0135_summary["sum"] /
    recipe_0135_summary["sum"].sum()
)
recipe_0135_summary["metric_rank"] = (
    recipe_0135_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0136
recipe_0136_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0136_summary["share_of_total"] = (
    recipe_0136_summary["sum"] /
    recipe_0136_summary["sum"].sum()
)
recipe_0136_summary["metric_rank"] = (
    recipe_0136_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0137
recipe_0137_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0137_summary["share_of_total"] = (
    recipe_0137_summary["sum"] /
    recipe_0137_summary["sum"].sum()
)
recipe_0137_summary["metric_rank"] = (
    recipe_0137_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0138
recipe_0138_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0138_summary["share_of_total"] = (
    recipe_0138_summary["sum"] /
    recipe_0138_summary["sum"].sum()
)
recipe_0138_summary["metric_rank"] = (
    recipe_0138_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0139
recipe_0139_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0139_summary["share_of_total"] = (
    recipe_0139_summary["sum"] /
    recipe_0139_summary["sum"].sum()
)
recipe_0139_summary["metric_rank"] = (
    recipe_0139_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0140
recipe_0140_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0140_summary["share_of_total"] = (
    recipe_0140_summary["sum"] /
    recipe_0140_summary["sum"].sum()
)
recipe_0140_summary["metric_rank"] = (
    recipe_0140_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0141
recipe_0141_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0141_summary["share_of_total"] = (
    recipe_0141_summary["sum"] /
    recipe_0141_summary["sum"].sum()
)
recipe_0141_summary["metric_rank"] = (
    recipe_0141_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0142
recipe_0142_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0142_summary["share_of_total"] = (
    recipe_0142_summary["sum"] /
    recipe_0142_summary["sum"].sum()
)
recipe_0142_summary["metric_rank"] = (
    recipe_0142_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0143
recipe_0143_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0143_summary["share_of_total"] = (
    recipe_0143_summary["sum"] /
    recipe_0143_summary["sum"].sum()
)
recipe_0143_summary["metric_rank"] = (
    recipe_0143_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0144
recipe_0144_summary = (
    df.groupby("SeniorCitizen_Label", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0144_summary["share_of_total"] = (
    recipe_0144_summary["sum"] /
    recipe_0144_summary["sum"].sum()
)
recipe_0144_summary["metric_rank"] = (
    recipe_0144_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0145
recipe_0145_summary = (
    df.groupby("Tenure_Group", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0145_summary["share_of_total"] = (
    recipe_0145_summary["sum"] /
    recipe_0145_summary["sum"].sum()
)
recipe_0145_summary["metric_rank"] = (
    recipe_0145_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0146
recipe_0146_summary = (
    df.groupby("Tenure_Group", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0146_summary["share_of_total"] = (
    recipe_0146_summary["sum"] /
    recipe_0146_summary["sum"].sum()
)
recipe_0146_summary["metric_rank"] = (
    recipe_0146_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0147
recipe_0147_summary = (
    df.groupby("Tenure_Group", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0147_summary["share_of_total"] = (
    recipe_0147_summary["sum"] /
    recipe_0147_summary["sum"].sum()
)
recipe_0147_summary["metric_rank"] = (
    recipe_0147_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0148
recipe_0148_summary = (
    df.groupby("Tenure_Group", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0148_summary["share_of_total"] = (
    recipe_0148_summary["sum"] /
    recipe_0148_summary["sum"].sum()
)
recipe_0148_summary["metric_rank"] = (
    recipe_0148_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0149
recipe_0149_summary = (
    df.groupby("Tenure_Group", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0149_summary["share_of_total"] = (
    recipe_0149_summary["sum"] /
    recipe_0149_summary["sum"].sum()
)
recipe_0149_summary["metric_rank"] = (
    recipe_0149_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0150
recipe_0150_summary = (
    df.groupby("Tenure_Group", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0150_summary["share_of_total"] = (
    recipe_0150_summary["sum"] /
    recipe_0150_summary["sum"].sum()
)
recipe_0150_summary["metric_rank"] = (
    recipe_0150_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0151
recipe_0151_summary = (
    df.groupby("Tenure_Group", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0151_summary["share_of_total"] = (
    recipe_0151_summary["sum"] /
    recipe_0151_summary["sum"].sum()
)
recipe_0151_summary["metric_rank"] = (
    recipe_0151_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0152
recipe_0152_summary = (
    df.groupby("Tenure_Group", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0152_summary["share_of_total"] = (
    recipe_0152_summary["sum"] /
    recipe_0152_summary["sum"].sum()
)
recipe_0152_summary["metric_rank"] = (
    recipe_0152_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0153
recipe_0153_summary = (
    df.groupby("Tenure_Group", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0153_summary["share_of_total"] = (
    recipe_0153_summary["sum"] /
    recipe_0153_summary["sum"].sum()
)
recipe_0153_summary["metric_rank"] = (
    recipe_0153_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0154
recipe_0154_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0154_summary["share_of_total"] = (
    recipe_0154_summary["sum"] /
    recipe_0154_summary["sum"].sum()
)
recipe_0154_summary["metric_rank"] = (
    recipe_0154_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0155
recipe_0155_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0155_summary["share_of_total"] = (
    recipe_0155_summary["sum"] /
    recipe_0155_summary["sum"].sum()
)
recipe_0155_summary["metric_rank"] = (
    recipe_0155_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0156
recipe_0156_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0156_summary["share_of_total"] = (
    recipe_0156_summary["sum"] /
    recipe_0156_summary["sum"].sum()
)
recipe_0156_summary["metric_rank"] = (
    recipe_0156_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0157
recipe_0157_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0157_summary["share_of_total"] = (
    recipe_0157_summary["sum"] /
    recipe_0157_summary["sum"].sum()
)
recipe_0157_summary["metric_rank"] = (
    recipe_0157_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0158
recipe_0158_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0158_summary["share_of_total"] = (
    recipe_0158_summary["sum"] /
    recipe_0158_summary["sum"].sum()
)
recipe_0158_summary["metric_rank"] = (
    recipe_0158_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0159
recipe_0159_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0159_summary["share_of_total"] = (
    recipe_0159_summary["sum"] /
    recipe_0159_summary["sum"].sum()
)
recipe_0159_summary["metric_rank"] = (
    recipe_0159_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0160
recipe_0160_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0160_summary["share_of_total"] = (
    recipe_0160_summary["sum"] /
    recipe_0160_summary["sum"].sum()
)
recipe_0160_summary["metric_rank"] = (
    recipe_0160_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0161
recipe_0161_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0161_summary["share_of_total"] = (
    recipe_0161_summary["sum"] /
    recipe_0161_summary["sum"].sum()
)
recipe_0161_summary["metric_rank"] = (
    recipe_0161_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0162
recipe_0162_summary = (
    df.groupby("MonthlyCharge_Group", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0162_summary["share_of_total"] = (
    recipe_0162_summary["sum"] /
    recipe_0162_summary["sum"].sum()
)
recipe_0162_summary["metric_rank"] = (
    recipe_0162_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0163
recipe_0163_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0163_summary["share_of_total"] = (
    recipe_0163_summary["sum"] /
    recipe_0163_summary["sum"].sum()
)
recipe_0163_summary["metric_rank"] = (
    recipe_0163_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0164
recipe_0164_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0164_summary["share_of_total"] = (
    recipe_0164_summary["sum"] /
    recipe_0164_summary["sum"].sum()
)
recipe_0164_summary["metric_rank"] = (
    recipe_0164_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0165
recipe_0165_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0165_summary["share_of_total"] = (
    recipe_0165_summary["sum"] /
    recipe_0165_summary["sum"].sum()
)
recipe_0165_summary["metric_rank"] = (
    recipe_0165_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0166
recipe_0166_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0166_summary["share_of_total"] = (
    recipe_0166_summary["sum"] /
    recipe_0166_summary["sum"].sum()
)
recipe_0166_summary["metric_rank"] = (
    recipe_0166_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0167
recipe_0167_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0167_summary["share_of_total"] = (
    recipe_0167_summary["sum"] /
    recipe_0167_summary["sum"].sum()
)
recipe_0167_summary["metric_rank"] = (
    recipe_0167_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0168
recipe_0168_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0168_summary["share_of_total"] = (
    recipe_0168_summary["sum"] /
    recipe_0168_summary["sum"].sum()
)
recipe_0168_summary["metric_rank"] = (
    recipe_0168_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0169
recipe_0169_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0169_summary["share_of_total"] = (
    recipe_0169_summary["sum"] /
    recipe_0169_summary["sum"].sum()
)
recipe_0169_summary["metric_rank"] = (
    recipe_0169_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0170
recipe_0170_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0170_summary["share_of_total"] = (
    recipe_0170_summary["sum"] /
    recipe_0170_summary["sum"].sum()
)
recipe_0170_summary["metric_rank"] = (
    recipe_0170_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0171
recipe_0171_summary = (
    df.groupby("Customer_Value_Band", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0171_summary["share_of_total"] = (
    recipe_0171_summary["sum"] /
    recipe_0171_summary["sum"].sum()
)
recipe_0171_summary["metric_rank"] = (
    recipe_0171_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0172
recipe_0172_summary = (
    df.groupby("Tenure_Segment", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0172_summary["share_of_total"] = (
    recipe_0172_summary["sum"] /
    recipe_0172_summary["sum"].sum()
)
recipe_0172_summary["metric_rank"] = (
    recipe_0172_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0173
recipe_0173_summary = (
    df.groupby("Tenure_Segment", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0173_summary["share_of_total"] = (
    recipe_0173_summary["sum"] /
    recipe_0173_summary["sum"].sum()
)
recipe_0173_summary["metric_rank"] = (
    recipe_0173_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0174
recipe_0174_summary = (
    df.groupby("Tenure_Segment", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0174_summary["share_of_total"] = (
    recipe_0174_summary["sum"] /
    recipe_0174_summary["sum"].sum()
)
recipe_0174_summary["metric_rank"] = (
    recipe_0174_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0175
recipe_0175_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0175_summary["share_of_total"] = (
    recipe_0175_summary["sum"] /
    recipe_0175_summary["sum"].sum()
)
recipe_0175_summary["metric_rank"] = (
    recipe_0175_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0176
recipe_0176_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0176_summary["share_of_total"] = (
    recipe_0176_summary["sum"] /
    recipe_0176_summary["sum"].sum()
)
recipe_0176_summary["metric_rank"] = (
    recipe_0176_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0177
recipe_0177_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0177_summary["share_of_total"] = (
    recipe_0177_summary["sum"] /
    recipe_0177_summary["sum"].sum()
)
recipe_0177_summary["metric_rank"] = (
    recipe_0177_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0178
recipe_0178_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0178_summary["share_of_total"] = (
    recipe_0178_summary["sum"] /
    recipe_0178_summary["sum"].sum()
)
recipe_0178_summary["metric_rank"] = (
    recipe_0178_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0179
recipe_0179_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0179_summary["share_of_total"] = (
    recipe_0179_summary["sum"] /
    recipe_0179_summary["sum"].sum()
)
recipe_0179_summary["metric_rank"] = (
    recipe_0179_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0180
recipe_0180_summary = (
    df.groupby("Tenure_Segment", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0180_summary["share_of_total"] = (
    recipe_0180_summary["sum"] /
    recipe_0180_summary["sum"].sum()
)
recipe_0180_summary["metric_rank"] = (
    recipe_0180_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0181
recipe_0181_summary = (
    df.groupby("Charge_Segment", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0181_summary["share_of_total"] = (
    recipe_0181_summary["sum"] /
    recipe_0181_summary["sum"].sum()
)
recipe_0181_summary["metric_rank"] = (
    recipe_0181_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0182
recipe_0182_summary = (
    df.groupby("Charge_Segment", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0182_summary["share_of_total"] = (
    recipe_0182_summary["sum"] /
    recipe_0182_summary["sum"].sum()
)
recipe_0182_summary["metric_rank"] = (
    recipe_0182_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0183
recipe_0183_summary = (
    df.groupby("Charge_Segment", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0183_summary["share_of_total"] = (
    recipe_0183_summary["sum"] /
    recipe_0183_summary["sum"].sum()
)
recipe_0183_summary["metric_rank"] = (
    recipe_0183_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0184
recipe_0184_summary = (
    df.groupby("Charge_Segment", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0184_summary["share_of_total"] = (
    recipe_0184_summary["sum"] /
    recipe_0184_summary["sum"].sum()
)
recipe_0184_summary["metric_rank"] = (
    recipe_0184_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0185
recipe_0185_summary = (
    df.groupby("Charge_Segment", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0185_summary["share_of_total"] = (
    recipe_0185_summary["sum"] /
    recipe_0185_summary["sum"].sum()
)
recipe_0185_summary["metric_rank"] = (
    recipe_0185_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0186
recipe_0186_summary = (
    df.groupby("Charge_Segment", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0186_summary["share_of_total"] = (
    recipe_0186_summary["sum"] /
    recipe_0186_summary["sum"].sum()
)
recipe_0186_summary["metric_rank"] = (
    recipe_0186_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0187
recipe_0187_summary = (
    df.groupby("Charge_Segment", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0187_summary["share_of_total"] = (
    recipe_0187_summary["sum"] /
    recipe_0187_summary["sum"].sum()
)
recipe_0187_summary["metric_rank"] = (
    recipe_0187_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0188
recipe_0188_summary = (
    df.groupby("Charge_Segment", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0188_summary["share_of_total"] = (
    recipe_0188_summary["sum"] /
    recipe_0188_summary["sum"].sum()
)
recipe_0188_summary["metric_rank"] = (
    recipe_0188_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0189
recipe_0189_summary = (
    df.groupby("Charge_Segment", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0189_summary["share_of_total"] = (
    recipe_0189_summary["sum"] /
    recipe_0189_summary["sum"].sum()
)
recipe_0189_summary["metric_rank"] = (
    recipe_0189_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0190
recipe_0190_summary = (
    df.groupby("Risk_Segment", dropna=False)["MonthlyCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0190_summary["share_of_total"] = (
    recipe_0190_summary["sum"] /
    recipe_0190_summary["sum"].sum()
)
recipe_0190_summary["metric_rank"] = (
    recipe_0190_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0191
recipe_0191_summary = (
    df.groupby("Risk_Segment", dropna=False)["TotalCharges"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0191_summary["share_of_total"] = (
    recipe_0191_summary["sum"] /
    recipe_0191_summary["sum"].sum()
)
recipe_0191_summary["metric_rank"] = (
    recipe_0191_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0192
recipe_0192_summary = (
    df.groupby("Risk_Segment", dropna=False)["tenure"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0192_summary["share_of_total"] = (
    recipe_0192_summary["sum"] /
    recipe_0192_summary["sum"].sum()
)
recipe_0192_summary["metric_rank"] = (
    recipe_0192_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0193
recipe_0193_summary = (
    df.groupby("Risk_Segment", dropna=False)["Annual_Revenue"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0193_summary["share_of_total"] = (
    recipe_0193_summary["sum"] /
    recipe_0193_summary["sum"].sum()
)
recipe_0193_summary["metric_rank"] = (
    recipe_0193_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0194
recipe_0194_summary = (
    df.groupby("Risk_Segment", dropna=False)["Revenue_Lost"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0194_summary["share_of_total"] = (
    recipe_0194_summary["sum"] /
    recipe_0194_summary["sum"].sum()
)
recipe_0194_summary["metric_rank"] = (
    recipe_0194_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0195
recipe_0195_summary = (
    df.groupby("Risk_Segment", dropna=False)["Estimated_LTV"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0195_summary["share_of_total"] = (
    recipe_0195_summary["sum"] /
    recipe_0195_summary["sum"].sum()
)
recipe_0195_summary["metric_rank"] = (
    recipe_0195_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0196
recipe_0196_summary = (
    df.groupby("Risk_Segment", dropna=False)["Service_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0196_summary["share_of_total"] = (
    recipe_0196_summary["sum"] /
    recipe_0196_summary["sum"].sum()
)
recipe_0196_summary["metric_rank"] = (
    recipe_0196_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0197
recipe_0197_summary = (
    df.groupby("Risk_Segment", dropna=False)["Support_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0197_summary["share_of_total"] = (
    recipe_0197_summary["sum"] /
    recipe_0197_summary["sum"].sum()
)
recipe_0197_summary["metric_rank"] = (
    recipe_0197_summary["mean"].rank(method="dense", ascending=False)
)
# RECIPE 0198
recipe_0198_summary = (
    df.groupby("Risk_Segment", dropna=False)["Streaming_Count"]
    .agg(["count", "sum", "mean", "median", "min", "max", "std"])
    .reset_index()
)
recipe_0198_summary["share_of_total"] = (
    recipe_0198_summary["sum"] /
    recipe_0198_summary["sum"].sum()
)
recipe_0198_summary["metric_rank"] = (
    recipe_0198_summary["mean"].rank(method="dense", ascending=False)
)
# CHART RECIPE 0199
def chart_recipe_0199():
    summary = (
        df.groupby("Contract", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Contract")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0199_contract.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0200
def chart_recipe_0200():
    summary = (
        df.groupby("PaymentMethod", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PaymentMethod")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0200_paymentmethod.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0201
def chart_recipe_0201():
    summary = (
        df.groupby("InternetService", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by InternetService")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0201_internetservice.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0202
def chart_recipe_0202():
    summary = (
        df.groupby("OnlineSecurity", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by OnlineSecurity")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0202_onlinesecurity.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0203
def chart_recipe_0203():
    summary = (
        df.groupby("OnlineBackup", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by OnlineBackup")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0203_onlinebackup.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0204
def chart_recipe_0204():
    summary = (
        df.groupby("DeviceProtection", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by DeviceProtection")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0204_deviceprotection.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0205
def chart_recipe_0205():
    summary = (
        df.groupby("TechSupport", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by TechSupport")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0205_techsupport.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0206
def chart_recipe_0206():
    summary = (
        df.groupby("StreamingTV", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by StreamingTV")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0206_streamingtv.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0207
def chart_recipe_0207():
    summary = (
        df.groupby("StreamingMovies", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by StreamingMovies")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0207_streamingmovies.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0208
def chart_recipe_0208():
    summary = (
        df.groupby("PhoneService", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PhoneService")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0208_phoneservice.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0209
def chart_recipe_0209():
    summary = (
        df.groupby("MultipleLines", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by MultipleLines")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0209_multiplelines.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0210
def chart_recipe_0210():
    summary = (
        df.groupby("PaperlessBilling", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by PaperlessBilling")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0210_paperlessbilling.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0211
def chart_recipe_0211():
    summary = (
        df.groupby("Partner", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Partner")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0211_partner.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0212
def chart_recipe_0212():
    summary = (
        df.groupby("Dependents", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Dependents")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0212_dependents.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0213
def chart_recipe_0213():
    summary = (
        df.groupby("gender", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by gender")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0213_gender.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0214
def chart_recipe_0214():
    summary = (
        df.groupby("SeniorCitizen_Label", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by SeniorCitizen_Label")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0214_seniorcitizen_label.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0215
def chart_recipe_0215():
    summary = (
        df.groupby("Tenure_Group", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by Tenure_Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0215_tenure_group.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary
# CHART RECIPE 0216
def chart_recipe_0216():
    summary = (
        df.groupby("MonthlyCharge_Group", dropna=False)[TARGET_FLAG]
        .mean().mul(100).sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Churn Rate by MonthlyCharge_Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_file = CHART_DIR / "recipe_0216_monthlycharge_group.png"
    fig.savefig(output_file, dpi=140)
    plt.close(fig)
    return summary

# 18. DATA QUALITY FUNCTIONS

def check_duplicates(data=df):
    return int(data.duplicated().sum())

def check_missing(data=df):
    return data.isna().sum().sort_values(ascending=False)

def check_unique_customers(data=df):
    return data[CUSTOMER_ID].nunique()

def check_negative_charges(data=df):
    return int((data["MonthlyCharges"] < 0).sum())

def check_negative_total_charges(data=df):
    return int((data["TotalCharges"] < 0).sum())

def check_invalid_churn(data=df):
    return sorted(set(data[TARGET].dropna()) - {"Yes", "No"})

print("Duplicates:", check_duplicates())
print("Missing values:", check_missing().head())
print("Unique customers:", check_unique_customers())
print("Invalid churn values:", check_invalid_churn())

# 19. AUTOMATED BUSINESS REPORT

def generate_business_report(data=df):
    report = {}
    report["customers"] = int(data[CUSTOMER_ID].nunique())
    report["churned"] = int(data[TARGET_FLAG].sum())
    report["retained"] = int((data[TARGET_FLAG] == 0).sum())
    report["churn_rate_pct"] = float(data[TARGET_FLAG].mean() * 100)
    report["retention_rate_pct"] = float((1 - data[TARGET_FLAG].mean()) * 100)
    report["monthly_revenue"] = float(data["MonthlyCharges"].sum())
    report["annual_revenue"] = float(data["Annual_Revenue"].sum())
    report["revenue_lost"] = float(data["Revenue_Lost"].sum())
    report["average_tenure"] = float(data["tenure"].mean())
    report["average_monthly_charge"] = float(data["MonthlyCharges"].mean())
    return report

business_report = generate_business_report()
print(json.dumps(business_report, indent=2, default=str))
