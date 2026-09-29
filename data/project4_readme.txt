# Customer Segmentation & Recommendation System

An end-to-end machine learning project that segments e-commerce customers using RFM analysis + KMeans clustering, and recommends products using item-based collaborative filtering — deployed as a live Flask web app.

🔗 **Live Demo:** [(https://customer-segmentation-recommendation-rx9j.onrender.com)]
(Free hosting — first load may take 30-50 seconds to wake up)

## Overview

Online retailers sit on huge amounts of transaction data but often don't know *who* their best customers are or *what* to recommend to whom. This project solves both problems:

1. **Customer Segmentation** — groups customers into meaningful segments (Champions, Loyal, New/Promising, At Risk) based on their purchase behavior, so a business can target each group differently.
2. **Product Recommendation** — suggests relevant products to a customer based on what similar customers have bought.

## Dataset

[Online Retail Dataset (UCI)](https://archive.ics.uci.edu/dataset/352/online+retail) — ~541,000 transactions from a UK-based online gift retailer (Dec 2010 – Dec 2011), covering 4,300+ customers across 38 countries.

## Approach

**1. Data Cleaning**
- Removed rows with missing CustomerID (guest orders)
- Removed cancelled orders and non-product entries (postage, manual charges)
- Removed extreme single-transaction outliers

**2. RFM Feature Engineering**
- Calculated Recency, Frequency, and Monetary value per customer
- Applied log transformation to handle heavy right-skew, then scaled with StandardScaler

**3. Customer Segmentation**
- Used the Elbow Method and Silhouette Score to choose k=4
- Applied KMeans clustering, labeled clusters based on their RFM profile:
  - **Champions** — recent, frequent, high spenders
  - **Loyal** — steady, consistent buyers
  - **New/Promising** — recent but low purchase history
  - **At Risk** — haven't purchased in a long time

**4. Recommendation System**
- Built a customer-item purchase matrix
- Computed item-item cosine similarity
- For a given customer, aggregated similarity scores across all their purchased items to recommend new, relevant products

**5. Deployment**
- Flask web app: enter a CustomerID → see their segment + top 5 recommended products
- Deployed live on Render

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Flask · Matplotlib/Seaborn · Gunicorn

## Project Structure