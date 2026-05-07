# 📊 MASA Hackathon 2026: R-Ignite

**Project:** Assessing Climate-Related Risks for Financial Resilience
**Team Name:** NextGenz
**Lead Researcher:** Nicole Ng Zhen Tiing

---

## 1. Executive Summary

This project develops a quantitative climate risk framework for a reinsurance firm, focusing specifically on Malaysia and the Philippines. By utilizing historical data from the World Bank and EM-DAT, we model greenhouse gas (GHG) emission trajectories and evaluate their correlation with natural disaster insurance claims. Our research aims to provide actionable insights for long-term financial resilience, stress-testing various emission scenarios to inform capital reserve requirements and parametric insurance pricing.

---

## 2. Team Information

### 🏫 General Information
* **University:** University of Technology Sarawak (UTS)
* **Team Name:** NextGenz

### 👥 Team Members
| Name | Role & Responsibility | Contact |
| :--- | :--- | :--- |
| **Nicole Ng Zhen Tiing** | Project Lead & Data Scientist | qpngjll607706@gmail.com |
| **CHAN XIN EN** | Report Formatting & Visual Optimization | xinen2811@gmail.com |
| **VEANN FOO WEI LING** | Copywriting & Data Collection | veann090909@gmail.com |
| **AUSTIN KHO QI ZHANG** | Literature Research & Risk Assessment | happy.family3266@gmail.com |

---

## 3. Project Repository Structure

```plaintext
.
├── NextGenZ_report.pdf                                 # Final 10-page research report
├── temp final.py                                       # Primary Python script for modeling
├── climate_risk_dashboard_sea 8 final.html             # Interactive visualization dashboard
├── climate_risk_dashboard_sea 8final.pbix              # Power BI
├── README.md                                           # Project documentation
└── Data/
    └── WB_WDI_WIDEF.csv                                # World Bank WDI raw dataset
```

---

## 4. Technical Requirements

### Environment

- **Python:** Version 3.9 or higher

### Key Libraries

| Library | Purpose |
|---|---|
| `pandas` | Robust data manipulation and cleaning |
| `scikit-learn` | Polynomial Regression, OLS, and SVR implementation |
| `matplotlib` | High-quality statistical visualizations |
| `seaborn` | High-quality statistical visualizations |

### Installation

Clone this repository and install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 5. Replication & Execution

To replicate the findings presented in our report:

1. **Data Setup:** Ensure `WB_WDI_WIDEF.csv` is placed in the root or `/Data` directory.

2. **Run Analysis:** Execute the Python script:

```bash
python climate_risk_analysis.py
```

3. **Outputs:**

   - **Cross-Validation:** The terminal will display the R² and MAE for the 5-fold cross-validation of the GHG model.
   - **Visuals:** The script generates:
     - `fig1_ghg_projection.png` — 2030 GHG projections
     - `fig2_insurance_analysis.png` — Correlation heatmap
   - **Interactive Dashboard:** Open `climate_risk_dashboard.html` in any modern web browser to simulate the 2030 stress-test scenarios.

---

## 6. Methodology

| Stage | Description |
|---|---|
| **Preprocessing** | Addressed missing values in the WDI dataset via interpolation; used Centered Year values to mitigate multi-collinearity in regression models |
| **Modeling** | Applied Polynomial Regression to capture the non-linear nature of historical GHG emissions |
| **Validation** | 5-fold cross-validation to ensure generalizability and prevent overfitting |
| **Stress Testing** | Simulated three scenarios — *Baseline*, *Moderate Mitigation*, and *High Risk* — to observe potential impacts on insurance claim volatility by 2030 |

---

## 7. Key Findings

> **Divergent Risk Profiles**
> While the Philippines shows acute risk from typhoons, Malaysia exhibits a structural increase in flood-related claims linked to urbanization and land-use changes.

> **Financial Impact**
> There is a statistically significant positive correlation between regional GHG intensity and the frequency of "secondary peril" claims.

> **Strategic Recommendation**
> Reinsurers should pivot toward **Parametric Reinsurance** triggers based on specific climatic thresholds rather than traditional indemnity models.

---

## 8. AI Disclosure

This project utilized generative AI *(Gemini 1.5 Flash)* for initial code scaffolding and document structure refinement. All statistical logic, data interpretations, and final model tuning were performed and verified manually by the NextGenz team to ensure accuracy and compliance with MASA Hackathon standards.

---

*© 2026 NextGenz — University of Technology Sarawak (UTS). All rights reserved.*
```
