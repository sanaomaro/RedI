# RedI
Telecom data coverage analysis - ReDi Project
Cellular Network Performance Analysis
Overview
This project analyzes cellular network performance data to identify patterns, 
optimize tower placement, and improve user experience. 
The dataset contains metrics like signal strength, SNR, call duration, Location type, Call types and durations and environmental factors affecting network quality.
****
**Key Features******
Data Cleaning: Handled invalid dates (e.g., 2022-02-29 → 2022-02-28)
Data Exploration: According to tower and environemnt and more

****Feature Engineering:**
**
Standardized environment names (open → rural, home → indoor)
Following telecom industry standards - log transformation for the explonential decay of the signal 

Added Location_Type (indoor/outdoor) classification
HotEncoding of categoridcal columns 

****Critical Insights:**

Avg signal strength: -85 dBm (📶 128 critical readings < -95 dBm)

Worst-performing tower: Tower 4

Best SNR environment: home (indoor)

**How to Run**
**Install requirements:**

bash
pip install pandas numpy matplotlib seaborn plotly kagglehub
Run the Jupyter notebook Cellular_network_performance.ipynb

Dataset Source
Kaggle: Cellular Network Performance Data

Contains 463 records

10 cellular towers

99 unique users

4 environment types

Recommendations for GitHub Enhancement
1. Add Visualizations
Include these plots to showcase findings:

python
# Signal strength distribution
sns.histplot(df['Signal Strength (dBm)'], kde=True)
plt.title("Signal Strength Distribution")

# Environment vs Call Duration
sns.boxplot(x='Environment', y='Call Duration (s)', data=df)

# Tower performance comparison
tower_perf = df.groupby('Tower ID')['Signal Strength (dBm)'].mean().sort_values()
tower_perf.plot(kind='bar', title='Avg Signal Strength by Tower')
2. Add Key Findings Section
markdown
## Key Findings
- 📉 27.6% of readings show critical signal strength (< -95 dBm)
- 📞 Voice calls last 22% longer than data sessions in rural areas
- 🏙️ Urban areas have strongest signals (-79.2 dBm avg) 
- 📡 Tower 7 handles most users (37 unique users)
3. Add Technical Debt Section
markdown
## Future Improvements
- [ ] Predict call drops using ML classification
- [ ] Optimize tower placement via geospatial analysis
- [ ] Analyze time-based patterns (hourly/daily trends)
- [ ] Investigate tower 4's underperformance
4. Repository Structure
Organize your project like this:

├── data/
│   └── train.csv             # Raw dataset
├── notebooks/
│   └── Cellular_network_performance.ipynb
├── outputs/                  # Generated plots/figures
├── .gitignore                # Exclude large data files
├── requirements.txt          # Dependencies
└── README.md
5. Add Analysis Highlights
markdown
## Critical Patterns
| Environment | Avg Signal (dBm) | Avg Call Duration |
|-------------|------------------|-------------------|
| Rural       | -92.4            | 412s              | 
| Indoor      | -81.7            | 738s              |
| Urban       | -79.2            | 857s              |
6. Add This to Your .gitignore
*.csv
*.ipynb_checkpoints
__pycache__/
.DS_Store

EDIT: 05 jUNE 2025
NOTE: YOU MIGHT FIND AN ATTEMPT TO SOLVE IT AS A CLASSIFICATION PROBLEM , **THIS IS NOT A CLASSIFICATION PROBLEM,** BUT OUT OF THE CURISITY WAS TRYING EXTRA MODELS,
ALSO IT SEEMS AS THIS DATA DOES NOT INCLUDE THE TRANSMITTED SIGNAL POWER FEATURE, SO IT WAS DIFFICULT TO PREDICT SNR NO MATTER THE MODEL IS.

