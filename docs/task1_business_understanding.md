# Task 1 — Business Understanding & Hypothesis Framing

## Client: PowerCo (Energy Provider)

## The Business Problem

PowerCo is experiencing significant customer churn. The client suspects that customers are becoming increasingly **price-sensitive** and switching to cheaper energy providers.

Our role as BCG data scientists is to:
1. Validate or disprove this hypothesis using data
2. Identify the true drivers of churn
3. Build a predictive model to flag at-risk customers

---

## Hypothesis

> **"Customer churn at PowerCo is primarily driven by price sensitivity."**

---

## Translating "Price Sensitivity" into Measurable Variables

To investigate this hypothesis, we need to convert the abstract concept of "price sensitivity" into concrete, measurable data variables:

| Concept | Measurable Variable |
|---|---|
| Price level | Price per kWh (electricity / gas) |
| Bill burden | Monthly bill amount |
| Price volatility | Price changes over time (month-on-month) |
| Pricing structure | Discounted vs non-discounted customers |
| Price elasticity | % change in consumption / % change in price |

**Price Elasticity of Demand formula:**
```
Price Elasticity = % change in quantity demanded / % change in price
```
If |elasticity| > 1 → the product is **elastic** (price strongly influences behaviour)

---

## Step-by-Step Investigation Framework

### Step 1 — Define Churn Groups
Split customers into:
- **Group 0:** Retained customers (churn = 0)
- **Group 1:** Churned customers (churn = 1)

### Step 2 — Compare Pricing Between Groups
Ask:
- Do churned customers pay more on average?
- Did their prices increase before they churned?
- Are they on less flexible tariffs?

### Step 3 — Analyse Price Changes Over Time
- Did churn spike after price increases?
- Are stable-price customers staying longer?

### Step 4 — Segment the Customer Base
Check if sensitivity varies by:
- Business size (small vs medium)
- Usage level (high vs low consumption)
- Customer age (new vs long-term)

### Step 5 — Model & Quantify
Use binary classification to predict churn, then use feature importances to quantify how much price contributes vs other factors.

---

## Data Requirements Identified

To conduct this analysis, we need three core datasets from the client:

1. **Customer Data** — industry, location, tenure, historical consumption
2. **Churn Data** — binary flag indicating whether a customer has churned
3. **Historical Pricing Data** — prices charged per customer for electricity and gas over time

Optional but useful:
- Contract and billing data (contract type, duration, discounts applied)

---

## Client Email Draft

**To:** Estelle Antonie (Senior Data Scientist, PowerCo)
**From:** BCG Data Science Team
**Subject:** Data Requirements & Analytical Approach — Customer Churn Investigation

---

Dear Estelle,

To test the hypothesis that customer churn is driven by price sensitivity, we propose to model customer churn probabilities and quantify the impact of pricing on churn behaviour.

To support this analysis, we would require the following data:

- **Customer data**: including characteristics such as industry, location, tenure, and historical electricity/gas consumption
- **Churn data**: indicating whether and when a customer has churned
- **Historical pricing data**: detailing the prices charged to each customer for electricity and gas over time at a sufficiently granular level
- **Contract and billing data** (if available): including contract type, duration, and any discounts or incentives applied

Once the data is obtained, our analytical approach will follow a structured process:

1. **Define and quantify price sensitivity**, for example by measuring changes in pricing relative to customer consumption and contract terms
2. **Data preparation and feature engineering**, including constructing variables such as price changes, average price levels, and customer tenure
3. **Exploratory data analysis** to identify patterns and differences between churned and retained customers
4. **Model development**, using binary classification techniques (e.g., Random Forest) to predict the likelihood of churn
5. **Model evaluation and selection**, based on performance metrics (accuracy, precision, recall, F1, ROC-AUC) as well as interpretability
6. **Insight generation**, using the selected model to quantify the extent to which price sensitivity contributes to churn

This approach will allow us to rigorously assess whether pricing is a primary driver of churn and support data-driven recommendations for improving customer retention.

Please let me know if you would like to refine any of these elements further.

Kind regards,
BCG Data Science Team

---

## Key Takeaway

If price sensitivity IS the driver:
> Churned customers will consistently show higher average prices and more frequent price increases compared to retained customers.

If price sensitivity is NOT the main driver:
> No significant pricing difference will be found — other factors (tenure, consumption, service quality) will dominate.

**→ We answer this question conclusively in Tasks 3 and 4 through EDA and feature importance analysis.**
