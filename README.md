[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://india-cpi-analysis-2013-25.streamlit.app/)

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fvrobbin3247%2Findia-CPI-analysis&label=views&icon=github&color=%23198754&message=&style=flat-square&tz=UTC)
![Python](https://img.shields.io/badge/Python-3.12.2-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42.1-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)
![GitHub Stars](https://img.shields.io/github/stars/vrobbin3247/india-CPI-analysis?style=social)
![Forks](https://img.shields.io/github/forks/vrobbin3247/india-CPI-analysis?style=social)
![Last Commit](https://img.shields.io/github/last-commit/vrobbin3247/india-CPI-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/vrobbin3247/india-CPI-analysis/blob/main/LICENSE)

# 🚀 Streamlit Web App for CPI Data Analysis

## 📌 Overview
This Streamlit web application allows users to explore and analyze the Consumer Price Index (CPI) data scraped from the Ministry of Statistics and Programme Implementation (MoSPI) India API. The data is available for different states, sectors, groups, and subgroups from 2013 to January 2025, with 2012 as the base year.

## 📊 Data Structure
<details>
  <summary>Click to expand</summary>

The CPI data is categorized into:
- **🗺️ States:** Individual states of India and an aggregated "All India" dataset.
- **🏙️ Sectors:** Urban, Rural, and Combined.
- **📦 Groups and Subgroups:**
  - **📌 General** (No subgroups)
  - **🍽️ Food and Beverages:**
    - 🌾 Cereals and Products
    - 🍖 Meat and Fish
    - 🥚 Egg
    - 🥛 Milk and Products
    - 🛢️ Oils and Fats
    - 🍎 Fruits
    - 🥕 Vegetables
    - 🌱 Pulses and Products
    - 🍬 Sugar and Confectionery
    - 🌶️ Spices
    - 🍱 Prepared Meals, Snacks, Sweets, etc.
    - 🥤 Non-alcoholic Beverages
  - **🚬 Pan, Tobacco and Intoxicants** (No subgroups)
  - **👕 Clothing and Footwear:**
    - 👗 Clothing
    - 👞 Footwear
  - **🏠 Housing** (No subgroups)
  - **🔥 Fuel and Light** (No subgroups)
  - **📌 Miscellaneous:**
    - 🏠 Household Goods and Services
    - 🏥 Health
    - 🚗 Transport and Communication
    - 🎭 Recreation and Amusement
    - 📚 Education
    - 💄 Personal Care and Effects
  - **🍛 Consumer Food Price** (No subgroups)
</details>

## 🌟 Features
### 1. 📈 CPI Change Metrics
<details>
  <summary>Click to expand</summary>

- Displays CPI change between the current month and the previous month along with the percentage change.
- Applied to the **General** category across Urban, Rural, and Combined sectors.
- Users can select a specific state to view its CPI change.

   ![CPI Change Metrics](./test%20scripts/metric.gif)
</details>

### 2. 📉 Line Chart Analysis
<details>
  <summary>Click to expand</summary>

- Users can compare data from multiple states, sectors, groups, and subgroups.
- Infinite possibilities for comparison using sidebar filters.
- A date slider allows users to select a specific time range for analysis.
- Users can enable a checkbox to view filtered data in tabular form.
- Filtered data can be downloaded as a CSV file.

  ![Line chart](./test%20scripts/analysis.gif)
</details>

### 3. 🌍 State-wise CPI Visualization
<details>
  <summary>Click to expand</summary>

- Users can visualize CPI for different states by selecting a sector (Rural, Urban, or Combined).
- Visualization options include:
  - 📋 Table
  - 📊 Bar Chart
  - 🗺️ Map of India (using Plotly and GeoJSON state boundary data)

     ![state analysis](./test%20scripts/maps.gif)
</details>

### 4. 🔮 Trend Forecasting
<details>
  <summary>Click to expand</summary>

- Forecast CPI values for the next 5 months using historical data.
- Users can select the value to be forecasted via a dropdown (e.g., "Milk and Products" index for Urban All India).
- Displays historical data and predicted values on a single line chart.

  ![forecasting](./test%20scripts/forecast.gif)
</details>

## 🛠️ Usage Instructions
<details>
  <summary>Click to expand</summary>

1. 🎯 Select a state from the dropdown to view CPI change metrics.
2. 📊 Use sidebar filters to analyze CPI data using line charts.
3. 📄 View data in table form and download it as a CSV if needed.
4. 📌 Select visualization type (Table, Bar Chart, or Map) to explore CPI across states.
5. 🔮 Forecast CPI trends for the next 5 months using the forecasting module.
</details>

## 💡 Technologies Used
<details>
  <summary>Click to expand</summary>

- **🎨 Streamlit**: Web application framework.
- **📊 Plotly**: Data visualization.
- **🗺️ GeoJSON**: Mapping Indian states.
- **📝 Pandas**: Data handling and manipulation.
</details>

This web app provides an interactive and comprehensive analysis of CPI trends, helping users gain insights into inflation and economic patterns across India. 🚀

