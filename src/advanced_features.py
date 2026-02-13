"""
Advanced Feature Engineering for Credit Risk Model
Feature Categories:
1. Post-Loan Affordability Stress
2. Delinquency Risk Signals (Time-Decay Weighted)
3. Credit Seeking Behavior (Desperation Index)
4. Income Stability Composite
5. Bureau Score Risk Transformation
6. Risk Interaction Features (Compound Risk)
7. Portfolio Leverage & Concentration
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# imputation logic based on mathematical relationships


def impute_monthly_income(df: pd.DataFrame) -> pd.Series:
    """
    Impute missing monthly_income using mathematical relationships.
    Formula A: monthly_installment / installment_to_income
    Formula B: total_monthly_obligation / debt_to_income
    Rule: Use average of both if they are close, else use the more reliable one.

    Returns:
        pd.Series: monthly_income with imputed values
    """
    income = df["monthly_income"].copy()
    missing_mask = income.isna()

    if missing_mask.sum() == 0:
        return income

    logger.info(
        f"Imputing {missing_mask.sum()} missing monthly_income values using "
        "mathematical relationships..."
    )

    # Formula A: monthly_installment / installment_to_income
    iti = df.loc[missing_mask, "installment_to_income"]
    installment = df.loc[missing_mask, "monthly_installment"]
    income_a = installment / iti.replace(0, np.nan)

    # Formula B: total_monthly_obligation / debt_to_income
    dti = df.loc[missing_mask, "debt_to_income"]
    obligation = df.loc[missing_mask, "total_monthly_obligation"]
    income_b = obligation / dti.replace(0, np.nan)

    # Apply imputation logic
    for idx in missing_mask[missing_mask].index:
        val_a = income_a.get(idx, np.nan)
        val_b = income_b.get(idx, np.nan)

        # Both valid: check if close
        if pd.notna(val_a) and pd.notna(val_b) and val_a > 0 and val_b > 0:
            ratio = abs(val_a - val_b) / max(val_a, val_b)
            if ratio <= 0.20:
                # Close enough, use average
                income.loc[idx] = (val_a + val_b) / 2
            else:
                # Not close, use Formula A (installment-based, more reliable for new loans)
                income.loc[idx] = val_a
        elif pd.notna(val_a) and val_a > 0:
            income.loc[idx] = val_a
        elif pd.notna(val_b) and val_b > 0:
            income.loc[idx] = val_b
        else:
            # Fallback to median if both formulas fail
            income.loc[idx] = df["monthly_income"].median()

    imputed_count = missing_mask.sum() - income[missing_mask].isna().sum()
    logger.info(f"Successfully imputed {imputed_count} monthly_income values")

    return income


def impute_income_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Impute missing income_volatility_proxy using business rules.
    Logic:
    - Base volatility for salary: 0.20 (stable)
    - Base volatility for self_emp: 0.40 (high variability)
    - Add 0.15 if income_missing_flag = 1 (higher risk due to missing data)
    - Cap at 0.80 maximum

    Returns:
        pd.Series: income_volatility_proxy with imputed values
    """
    volatility = df["income_volatility_proxy"].copy()
    missing_mask = volatility.isna()

    if missing_mask.sum() == 0:
        return volatility

    logger.info(
        f"Imputing {missing_mask.sum()} missing income_volatility_proxy values using "
        "business rules..."
    )

    for idx in missing_mask[missing_mask].index:
        # Base volatility based on income type
        income_type = df.loc[idx, "income_type"]
        if income_type == "self_emp":
            base_vol = 0.40  # Self-employed = high volatility
        else:
            base_vol = 0.20  # Salaried = low volatility

        # Add penalty for missing income flag
        income_missing = df.loc[idx, "income_missing_flag"]
        if income_missing == 1:
            base_vol += 0.15  # Higher risk due to missing data

        # Cap at 0.80
        volatility.loc[idx] = min(base_vol, 0.80)

    logger.info(
        "Imputed income_volatility: self_emp base=0.40, salary base=0.20, "
        "+0.15 if income_missing_flag=1"
    )

    return volatility


# Risk weight constants

# Bureau Score Risk Bands
BUREAU_RISK_BANDS = {
    (0, 550): (3.3, "very_high_risk"),  # 23.1% default
    (550, 600): (2.1, "high_risk"),  # 14.8% default
    (600, 650): (1.8, "medium_high_risk"),  # 12.6% default
    (650, 700): (1.4, "medium_risk"),  # 10.1% default
    (700, 750): (1.2, "low_risk"),  # 8.6% default
    (750, 900): (1.0, "very_low_risk"),  # 7.0% default
}

# DPD severity weights, last is more severe
DPD_TIME_DECAY = {
    "3m": 1.0,  # Most recent
    "6m": 0.7,  # Moderate
    "12m": 0.5,  # Older
    "24m": 0.3,  # Much older
}

# DPD level severity weights
DPD_SEVERITY = {
    0: 0,  # No DPD, no risk
    30: 1,  # 30DPD, minor
    60: 3,  # 60DPD, moderate (3x penalty)
    90: 10,  # 90DPD, severe (10x penalty)
}

# Inquiry Acceleration Thresholds
INQUIRY_VELOCITY_THRESHOLDS = {
    "low": (0, 0.3),  # Stable inquiry pattern
    "medium": (0.3, 0.5),  # Some acceleration
    "high": (0.5, 0.7),  # Concerning acceleration
    "very_high": (0.7, 1.1),  # Credit desperation
}


# 1. POST-LOAN AFFORDABILITY STRESS FEATURES


def create_affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-loan affordability features measuring the applicant's ability
    to handle the new loan payment on top of existing obligations.
    """
    logger.info("Creating affordability stress features...")

    features = pd.DataFrame(index=df.index)

    # Current total obligations
    current_obligation = df["total_monthly_obligation"].fillna(0)
    monthly_income = impute_monthly_income(df)  # Smart imputation using mathematical relationships
    new_installment = df["monthly_installment"].fillna(0)

    # 1.1 Post-Loan Debt-to-Income
    # Total debt burden AFTER taking the new loan
    total_obligation_post_loan = current_obligation + new_installment
    features["post_loan_dti"] = total_obligation_post_loan / monthly_income
    features["post_loan_dti"] = (
        features["post_loan_dti"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    features["post_loan_dti"] = features["post_loan_dti"].clip(0, 10)  # Cap at 1000%

    # 1.2 Residual Income (Absolute measure of affordability)
    # How much money is left after paying ALL obligations
    features["residual_income"] = monthly_income - total_obligation_post_loan
    features["residual_income_positive"] = (features["residual_income"] > 0).astype(int)

    # 1.3 Affordability Buffer (% of income remaining)
    features["affordability_buffer"] = features["residual_income"] / monthly_income
    features["affordability_buffer"] = (
        features["affordability_buffer"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    features["affordability_buffer"] = features["affordability_buffer"].clip(-5, 1)  # Cap extremes

    # 1.4 Distance to Critical ITI Threshold
    # How far are they from the danger zone (50% ITI where default rate = 24.7%)
    current_iti = df["installment_to_income"].fillna(0)
    features["iti_headroom"] = 0.50 - current_iti

    # 1.5 New Loan Relative Burden
    # How big is this new loan relative to existing debt
    features["new_loan_relative_burden"] = new_installment / (current_obligation + 1)
    features["new_loan_relative_burden"] = features["new_loan_relative_burden"].clip(0, 50)

    # 1.6 Loan Amount to Existing Debt Ratio
    features["loan_debt_stacking"] = df["requested_amount"] / (df["open_loans_total_amt"] + 1)
    features["loan_debt_stacking"] = features["loan_debt_stacking"].clip(0, 100)

    # 1.7 Income Coverage Ratio (Monthly income / new installment)
    # Higher- more cushion
    features["income_coverage_ratio"] = monthly_income / (new_installment + 1)
    features["income_coverage_ratio"] = features["income_coverage_ratio"].clip(0, 100)

    # 1.8 Affordability Risk Score
    # Combine DTI, ITI and buffer into single score
    dti_component = features["post_loan_dti"].clip(0, 1) * 0.4
    iti_component = current_iti.clip(0, 1) * 0.4
    buffer_component = (1 - features["affordability_buffer"].clip(0, 1)) * 0.2
    features["affordability_risk_score"] = dti_component + iti_component + buffer_component

    logger.info(f"Created {len(features.columns)} affordability features")
    return features


# 2. delinquency risk signals with time-decay weighting


def create_dpd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delinquency-based features with time-decay weighting.
    """
    logger.info("Creating DPD risk features...")

    features = pd.DataFrame(index=df.index)

    # 2.1 Time-Decay Weighted DPD Score
    # Recent delinquency weighted more heavily
    dpd_3m = df["max_dpd_last_3m"]
    dpd_6m = df["max_dpd_last_6m"]
    dpd_12m = df["max_dpd_last_12m"]
    dpd_24m = df["max_dpd_last_24m"]

    features["dpd_time_decay_score"] = (
        dpd_3m * DPD_TIME_DECAY["3m"]
        + dpd_6m * DPD_TIME_DECAY["6m"]
        + dpd_12m * DPD_TIME_DECAY["12m"]
        + dpd_24m * DPD_TIME_DECAY["24m"]
    )

    # 2.2 DPD Severity Score
    # ever_90dpd = 10 points, ever_60dpd = 3 points, ever_30dpd = 1 point
    features["dpd_severity_score"] = (
        df["ever_90dpd_flag"] * DPD_SEVERITY[90]
        + df["ever_60dpd_flag"] * DPD_SEVERITY[60]
        + df["ever_30dpd_flag"] * DPD_SEVERITY[30]
    )

    # 2.3 DPD Velocity
    # High velocity = concerning behavioral pattern
    features["dpd_velocity_3m_to_12m"] = dpd_3m / (dpd_12m + 1)
    features["dpd_velocity_3m_to_12m"] = features["dpd_velocity_3m_to_12m"].clip(0, 10)

    # 2.4 Recovery Score
    # Longer time clean + lower max DPD - better recovery
    months_since = df["months_since_last_dpd"].fillna(120)  # No DPD = long time
    max_dpd_ever = df[
        ["max_dpd_last_3m", "max_dpd_last_6m", "max_dpd_last_12m", "max_dpd_last_24m"]
    ].max(axis=1)

    features["dpd_recovery_score"] = months_since / (max_dpd_ever + 1)
    features["dpd_recovery_score"] = features["dpd_recovery_score"].clip(0, 120)

    # 2.5 Clean Record Indicator (No DPD in recent history)
    features["clean_record_3m"] = (dpd_3m == 0).astype(int)
    features["clean_record_6m"] = (dpd_6m == 0).astype(int)
    features["clean_record_12m"] = (dpd_12m == 0).astype(int)

    # 2.6 DPD Trend
    # Positive = improving (older DPD higher than recent)
    features["dpd_trend"] = dpd_24m - dpd_3m  # Positive = cleaning up

    # 2.7 Delinquency Recency
    # Transform to risk score: more recent = higher risk
    features["dpd_recency_risk"] = np.exp(-months_since / 12)

    # 2.8 Composite DPD Risk Score
    # Normalize and combine
    features["dpd_composite_risk"] = (
        features["dpd_time_decay_score"] / 90 * 0.4  # Normalized by max DPD
        + features["dpd_severity_score"] / 14 * 0.3  # Normalized by max severity
        + features["dpd_recency_risk"] * 0.3
    )

    logger.info(f"Created {len(features.columns)} DPD features")
    return features


# 3. CREDIT SEEKING BEHAVIOR (Desperation Index)


def create_inquiry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit seeking behavior features measuring desperation signals.
    """
    logger.info("Creating inquiry behavior features...")

    features = pd.DataFrame(index=df.index)

    inq_3m = df["inq_count_last_3m"]
    # inq_6m removed (unused variable)
    inq_12m = df["inq_count_last_12m"]
    inq_cash = df["inq_cash_last_12m"]
    inq_non_cash = df["inq_non_cash_last_12m"]

    # 3.1 Inquiry Velocity
    # High ratio - concentrated recent inquiries - concerning
    features["inquiry_velocity"] = inq_3m / (inq_12m + 1)
    features["inquiry_velocity"] = features["inquiry_velocity"].clip(0, 3)

    # 3.2 Inquiry Intensity (Monthly rate)
    features["inquiry_monthly_rate_3m"] = inq_3m / 3
    features["inquiry_monthly_rate_12m"] = inq_12m / 12

    # 3.3 Cash Inquiry Concentration
    # Higher proportion of cash inquiries = higher risk
    total_inq = inq_cash + inq_non_cash
    features["cash_inquiry_ratio"] = inq_cash / (total_inq + 1)

    # 3.4 Inquiry Trend (Recent vs Historical)
    # Positive = increasing inquiry activity
    features["inquiry_trend"] = inq_3m - (inq_12m - inq_3m) / 3  # Compare to rest of year

    # 3.5 Credit Desperation Index
    # Combine velocity, intensity, and cash focus
    velocity_score = features["inquiry_velocity"].clip(0, 1)
    intensity_score = (inq_3m / 6).clip(0, 1)

    features["credit_desperation_index"] = (
        velocity_score * 0.4 + intensity_score * 0.4 + features["cash_inquiry_ratio"] * 0.2
    )

    # 3.6 Multiple Lender Indicator
    features["multiple_lender_3m"] = (inq_3m >= 2).astype(int)
    features["heavy_inquiry_flag"] = (inq_3m >= 4).astype(int)

    # 3.7 Inquiry to Open Loan Ratio
    # Many inquiries relative to open loans = low approval rate elsewhere
    open_loans = df["open_loans_total_cnt"].fillna(0)
    features["inquiry_to_loan_ratio"] = inq_12m / (open_loans + 1)
    features["inquiry_to_loan_ratio"] = features["inquiry_to_loan_ratio"].clip(0, 20)

    logger.info(f"Created {len(features.columns)} inquiry features")
    return features


# 4. income stability composite features


def create_income_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Income stability features measuring employment and income reliability.
    """
    logger.info("Creating income stability features...")

    features = pd.DataFrame(index=df.index)

    job_tenure = df["job_tenure_months"]
    income_vol = impute_income_volatility(df)  # Smart imputation using business rules
    monthly_income = impute_monthly_income(df)  # Smart imputation using mathematical relationships
    age = df["age"].fillna(df["age"].median())

    # 4.1 Tenure Stability Score (Normalized)
    # Longer tenure = higher stability
    features["tenure_stability_score"] = np.log1p(job_tenure) / np.log1p(240)

    # 4.2 Career Maturity (Tenure relative to age)
    # Long tenure for age = stable career
    features["career_maturity"] = job_tenure / (age * 12 - 18 * 12 + 1)
    features["career_maturity"] = features["career_maturity"].clip(0, 1)

    # 4.3 Income Variability Risk
    # Transform volatility to risk score
    features["income_variability_risk"] = income_vol.clip(0, 1)

    # 4.4 Income Stability Index (Inverse of volatility)
    features["income_stability_index"] = 1 - features["income_variability_risk"]

    # 4.5 Employment Type Risk (Categorical encoding based on observed risk)
    # Self-employed and irregular payment = higher risk combinations
    is_self_emp = (df["income_type"] == "self_emp").astype(int)
    is_irregular = (df["salary_payment_frequency"] == "irregular").astype(int)
    # is_private removed (unused variable)

    features["self_emp_flag"] = is_self_emp
    features["irregular_payment_flag"] = is_irregular
    features["employment_risk_combo"] = is_self_emp + is_irregular * 0.5  # Weighted combo

    # 4.6 Income Adequacy (Income percentile proxy)
    # Higher income relative to median = more buffer for payments
    income_median = monthly_income.median()
    features["income_adequacy"] = monthly_income / income_median
    features["income_adequacy"] = features["income_adequacy"].clip(0, 10)

    # 4.7 New Hire Risk Indicator
    features["new_hire_risk"] = (job_tenure < 12).astype(int)  # less than 1 year
    features["very_new_hire_risk"] = (job_tenure < 6).astype(int)  # less than 6 months

    # 4.8 Income Stability Composite Score
    features["income_stability_composite"] = (
        features["tenure_stability_score"] * 0.4
        + features["income_stability_index"] * 0.3
        + (1 - features["employment_risk_combo"]) * 0.3
    )

    logger.info(f"Created {len(features.columns)} income stability features")
    return features


# 5. BUREAU SCORE RISK TRANSFORMATION


def create_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bureau score transformation with risk weights.
    """
    logger.info("Creating bureau score features...")

    features = pd.DataFrame(index=df.index)

    bureau_score = df["bureau_score_proxy"].fillna(df["bureau_score_proxy"].median())

    # 5.1 Bureau Risk Weight (Based on default rates)
    def get_risk_weight(score):
        for (lower, upper), (weight, _) in BUREAU_RISK_BANDS.items():
            if lower <= score < upper:
                return weight
        return 1.0  # Default

    features["bureau_risk_weight"] = bureau_score.apply(get_risk_weight)

    # 5.2 Bureau Score Percentile (Relative standing)
    features["bureau_percentile"] = bureau_score.rank(pct=True)

    # 5.3 Bureau Score Deviation from Mean
    bureau_mean = bureau_score.mean()
    bureau_std = bureau_score.std()
    features["bureau_z_score"] = (bureau_score - bureau_mean) / bureau_std

    # 5.4 Bureau Score Bands
    features["bureau_very_low"] = (bureau_score < 550).astype(int)
    features["bureau_low"] = ((bureau_score >= 550) & (bureau_score < 600)).astype(int)
    features["bureau_medium"] = ((bureau_score >= 600) & (bureau_score < 700)).astype(int)
    features["bureau_high"] = (bureau_score >= 700).astype(int)

    # 5.5 Non-linear Bureau Transformation
    # Map score to approximate log-odds of default
    features["bureau_logodds_proxy"] = np.log(
        (100 - bureau_score / 10) / (bureau_score / 10 + 1)
    ).clip(-3, 3)

    # 5.6 Bureau Score Squared (Capture non-linearity)
    features["bureau_score_squared"] = (bureau_score / 100) ** 2

    logger.info(f"Created {len(features.columns)} bureau features")
    return features


# 6. RISK INTERACTION FEATURES


def create_interaction_features(
    df: pd.DataFrame,
    affordability_features: pd.DataFrame,
    dpd_features: pd.DataFrame,
    inquiry_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Risk interaction features capturing compound risk scenarios.
    """
    logger.info("Creating risk interaction features...")

    features = pd.DataFrame(index=df.index)

    # Get underlying variables
    bureau_score = df["bureau_score_proxy"].fillna(df["bureau_score_proxy"].median())
    age = df["age"].fillna(df["age"].median())
    job_tenure = df["job_tenure_months"].fillna(0)

    # 6.1 High DTI + Low Bureau
    # Stretched borrower with poor credit history
    high_dti = (affordability_features["post_loan_dti"] > 0.5).astype(int)
    low_bureau = (bureau_score < 600).astype(int)
    features["toxic_dti_bureau"] = high_dti * low_bureau

    # 6.2 High Inquiries + Recent DPD
    high_inq = (inquiry_features["inquiry_velocity"] > 0.5).astype(int)
    recent_dpd = (dpd_features["clean_record_3m"] == 0).astype(int)
    features["credit_stress_signal"] = high_inq * recent_dpd

    # 6.3 Young + short tenure - unstable
    young = (age < 30).astype(int)
    short_tenure = (job_tenure < 24).astype(int)
    features["unstable_profile"] = young * short_tenure

    # 6.4 Low Bureau + High DPD Severity
    high_dpd_severity = (dpd_features["dpd_severity_score"] > 3).astype(int)
    features["double_red_flag"] = low_bureau * high_dpd_severity

    # 6.5 Affordability Stress + Credit Desperation
    affordability_stress = (affordability_features["affordability_risk_score"] > 0.5).astype(int)
    desperation = (inquiry_features["credit_desperation_index"] > 0.5).astype(int)
    features["danger_zone_flag"] = affordability_stress * desperation

    # 6.6 Multiple Risk Count
    # Count how many individual risk flags are triggered
    risk_flags = pd.DataFrame(
        {
            "high_dti": high_dti,
            "low_bureau": low_bureau,
            "recent_dpd": recent_dpd,
            "high_inquiry": high_inq,
            "short_tenure": short_tenure,
        }
    )
    features["risk_flag_count"] = risk_flags.sum(axis=1)

    # 6.7 Compound Risk Score
    # Normalize individual scores and multiply
    bureau_risk = 1 - (bureau_score - 400) / 450  # higher = more risk
    bureau_risk = bureau_risk.clip(0, 1)

    dti_risk = affordability_features["post_loan_dti"].clip(0, 1)
    dpd_risk = dpd_features["dpd_composite_risk"].clip(0, 1)
    inq_risk = inquiry_features["credit_desperation_index"].clip(0, 1)

    # Product of risks
    features["compound_risk_product"] = (
        bureau_risk * (1 + dti_risk) * (1 + dpd_risk) * (1 + inq_risk)
    )
    features["compound_risk_product"] = features["compound_risk_product"].clip(0, 10)

    # 6.8 Weighted Risk Score (linear combination)
    features["weighted_risk_score"] = (
        bureau_risk * 0.30 + dti_risk * 0.25 + dpd_risk * 0.30 + inq_risk * 0.15
    )

    logger.info(f"Created {len(features.columns)} interaction features")
    return features


# 7. portfolio leverage and concentration features


def create_portfolio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Portfolio-level features measuring existing debt composition and leverage.
    """
    logger.info("Creating portfolio features...")

    features = pd.DataFrame(index=df.index)

    open_total_cnt = df["open_loans_total_cnt"].fillna(0)
    open_total_amt = df["open_loans_total_amt"].fillna(0)
    # open_cash_cnt and open_cash_amt removed (unused variables)
    cash_share_cnt = df["cash_loan_share_cnt"].fillna(0)
    cash_share_amt = df["cash_loan_share_amt"].fillna(0)
    utilization = df["avg_utilization_proxy"].fillna(df["avg_utilization_proxy"].median())
    requested_amt = df["requested_amount"].fillna(0)

    # 7.1 Average Loan Size
    features["avg_loan_size"] = open_total_amt / (open_total_cnt + 1)
    features["avg_loan_size"] = features["avg_loan_size"].clip(0, 50000)

    # 7.2 New Loan as % of Portfolio
    features["new_loan_portfolio_pct"] = requested_amt / (open_total_amt + requested_amt + 1)

    # 7.3 Cash Loan Dominance
    features["cash_dominance_cnt"] = cash_share_cnt
    features["cash_dominance_amt"] = cash_share_amt

    # 7.4 Portfolio Diversification
    # More loan types = more diversified
    features["portfolio_diversification"] = 1 - cash_share_cnt.clip(0, 1)

    # 7.5 Debt Stacking Indicator
    # Multiple open loans = stacking debt
    features["debt_stacking_cnt"] = (open_total_cnt >= 3).astype(int)
    features["heavy_debt_load"] = (open_total_cnt >= 5).astype(int)

    # 7.6 Utilization Risk Bands
    features["util_low"] = (utilization < 0.3).astype(int)
    features["util_medium"] = ((utilization >= 0.3) & (utilization < 0.5)).astype(int)
    features["util_high"] = ((utilization >= 0.5) & (utilization < 0.7)).astype(int)
    features["util_very_high"] = (utilization >= 0.7).astype(int)

    # 7.7 Utilization Stability Proxy
    # Extreme utilization (very low or very high) may indicate instability
    features["utilization_extremity"] = np.abs(utilization - 0.5) * 2

    # 7.8 Portfolio Leverage Score
    monthly_income = impute_monthly_income(df)  # Smart imputation
    features["portfolio_leverage"] = open_total_amt / (monthly_income * 12)  # Years of income
    features["portfolio_leverage"] = features["portfolio_leverage"].clip(0, 20)

    logger.info(f"Created {len(features.columns)} portfolio features")
    return features


# 8. demographic and application-based features


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demographic and application-based features.

    Business Logic:
    - While demographics alone are weak predictors, they provide context
    - Application channel may indicate customer segment
    - Tenor and amount combinations reveal risk preferences
    """
    logger.info("Creating demographic features...")

    features = pd.DataFrame(index=df.index)

    age = df["age"].fillna(df["age"].median())
    dependents = df["dependents_cnt"].fillna(0)
    requested_amt = df["requested_amount"].fillna(0)
    tenor = df["tenor_months"].fillna(12)
    residency_years = df["residency_years"].fillna(df["residency_years"].median())

    # 8.1 Age Groups (Risk segments)
    features["age_young"] = (age < 25).astype(int)
    features["age_prime"] = ((age >= 25) & (age <= 45)).astype(int)
    features["age_mature"] = ((age > 45) & (age <= 55)).astype(int)
    features["age_senior"] = (age > 55).astype(int)

    # 8.2 Family Burden (Dependents relative to income)
    monthly_income = impute_monthly_income(df)  # Smart imputation
    features["income_per_dependent"] = monthly_income / (dependents + 1)
    features["income_per_dependent"] = features["income_per_dependent"].clip(0, 10000)

    # 8.3 High Dependents Flag
    features["high_dependents"] = (dependents >= 3).astype(int)

    # 8.4 Residential Stability
    features["residential_stability"] = np.log1p(residency_years) / np.log1p(30)

    # 8.5 Loan Characteristics
    features["amount_per_month"] = requested_amt / tenor
    features["long_tenor_flag"] = (tenor >= 36).astype(int)
    features["high_amount_flag"] = (requested_amt > requested_amt.quantile(0.75)).astype(int)

    # 8.6 Channel Risk (Digital vs Branch)
    features["digital_channel"] = (df["application_channel"] == "digital").astype(int)

    # 8.7 Region Risk
    features["baku_region"] = (df["region"] == "Baku").astype(int)

    # 8.8 Education Level (Ordinal encoding)
    education_map = {"secondary": 1, "bachelor": 2, "master": 3}
    features["education_ordinal"] = df["education_level"].map(education_map).fillna(1)

    # 8.9 Marital Status
    features["married_flag"] = (df["marital_status"] == "married").astype(int)

    logger.info(f"Created {len(features.columns)} demographic features")
    return features


# main function to generate all features and provide summary


def generate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to generate all advanced features.
    Args:
    df: Raw DataFrame with original features

    Returns:
    DataFrame with all new engineered features appended
    """

    logger.info("GENERATING ADVANCED FEATURES \n")

    # Generate feature groups
    affordability_feats = create_affordability_features(df)
    dpd_feats = create_dpd_features(df)
    inquiry_feats = create_inquiry_features(df)
    income_stability_feats = create_income_stability_features(df)
    bureau_feats = create_bureau_features(df)
    portfolio_feats = create_portfolio_features(df)
    demographic_feats = create_demographic_features(df)

    # Interaction features need other feature groups
    interaction_feats = create_interaction_features(
        df, affordability_feats, dpd_feats, inquiry_feats
    )

    # Combine all features
    all_new_features = pd.concat(
        [
            affordability_feats,
            dpd_feats,
            inquiry_feats,
            income_stability_feats,
            bureau_feats,
            portfolio_feats,
            demographic_feats,
            interaction_feats,
        ],
        axis=1,
    )

    # Combine with original DataFrame
    result = pd.concat([df, all_new_features], axis=1)

    logger.info("-" * 60)
    logger.info("feature generation completed. Summary: \n")
    logger.info(f"Original features: {len(df.columns)}")
    logger.info(f"New features: {len(all_new_features.columns)}")
    logger.info(f"Total features: {len(result.columns)}")
    logger.info("-" * 60)

    # Feature summary
    feature_summary = {
        "affordability": list(affordability_feats.columns),
        "dpd": list(dpd_feats.columns),
        "inquiry": list(inquiry_feats.columns),
        "income_stability": list(income_stability_feats.columns),
        "bureau": list(bureau_feats.columns),
        "portfolio": list(portfolio_feats.columns),
        "demographic": list(demographic_feats.columns),
        "interaction": list(interaction_feats.columns),
    }

    return result, feature_summary  # type: ignore


def get_feature_importance_groups() -> Dict[str, List[str]]:
    """Return feature grouping for analysis."""
    return {
        "affordability_stress": [
            "post_loan_dti",
            "residual_income",
            "affordability_buffer",
            "iti_headroom",
            "new_loan_relative_burden",
            "affordability_risk_score",
        ],
        "delinquency_risk": [
            "dpd_time_decay_score",
            "dpd_severity_score",
            "dpd_recovery_score",
            "dpd_composite_risk",
            "dpd_trend",
            "dpd_recency_risk",
        ],
        "credit_seeking": [
            "inquiry_velocity",
            "credit_desperation_index",
            "inquiry_to_loan_ratio",
        ],
        "income_stability": [
            "tenure_stability_score",
            "income_stability_composite",
            "career_maturity",
            "employment_risk_combo",
        ],
        "bureau_risk": ["bureau_risk_weight", "bureau_percentile", "bureau_z_score"],
        "compound_risk": [
            "toxic_dti_bureau",
            "credit_stress_signal",
            "unstable_profile",
            "double_red_flag",
            "danger_zone_flag",
            "risk_flag_count",
            "compound_risk_product",
            "weighted_risk_score",
        ],
    }


if __name__ == "__main__":
    # Test with sample data
    import io

    from src.utils.io import get_s3_client, load_config

    cfg = load_config()
    s3 = get_s3_client(cfg)

    obj = s3.get_object(Bucket=cfg["minio"]["bucket"], Key="raw/dataset.xlsx")
    df = pd.read_excel(io.BytesIO(obj["Body"].read()), engine="openpyxl")

    result, summary = generate_advanced_features(df)

    print("FEATURE GROUPS SUMMARY \n")
    for group, features in summary.items():  # type: ignore
        print(f"\n{group.upper()} ({len(features)} features):")
        for f in features:
            print(f"  - {f}")
