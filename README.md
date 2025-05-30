# 📘 README – 5004CMD: Data Science  
**Student Name:** Neha Ann Binoy  
**Student ID:** 14511550           
**Module:** 5004CMD – Data Science  
**Module Leader:** Dr. Katerina Stamou  
**Assessment:** Coursework – Report and Code Submission  
**Submission Date:** 17 April, 2025  

---

## 🎯 Project Aim  
To analyze national mobility trends using large-scale travel data through scalable processing methods, effective visualization, and predictive modelling techniques. The work demonstrates the ability to work with big datasets, apply efficient computation, and generate actionable insights using reproducible data science practices.

---


### Question 1 – Trend Analysis: Stay-at-Home vs. Not Staying at Home

- Filtered and cleaned both datasets to isolate **"National" level** data only.
- Aligned datasets using a **common date range** to ensure synchronized trend analysis.
- Computed **daily averages** using `.groupby('Date').mean()` to track mobility over time.
- Applied **parallel processing with Dask LocalCluster** (10 & 20 workers) and benchmarked against Pandas.
- Visualized trends using bar and line plots; computed and interpreted **population means** for each group.


---

### Question 2 – Set Operations on High-Volume Trip Days

- Identified and categorized dates where trips exceeded **10 million** for:
  - 10–25 mile range
  - 50–100 mile range
  - **Common dates** across both
- Used **set theory** for semantic segmentation and behavioral pattern discovery.
- Aggregated data **monthly** to analyze frequency spikes.
- Simulated **parallel filtering** using Dask and compared with sequential filtering.
- Visualized results using **bar chart** for enhanced readability.


---

### Question 3 – Parallel vs. Sequential Time Benchmarking

- Simulated **three configurations**: Pandas (sequential), Dask with 10 and 20 partitions.
- Timed and visualized operations: filtering, grouping, statistical summaries, and model training.
- Applied **Dask.distributed with LocalCluster** to emulate real-world scalable environments.
- Used consistent logic to ensure a **fair performance comparison** across all approaches.


---

### Question 4 – Predictive Modelling with Linear Regression

- Built a **Linear Regression model** to predict **short-range trips (5–10 miles)** using medium-range trips (25–50 miles).
- Cleaned, synchronized, and merged datasets on a common date range.
- Evaluated using **R², MAE, MSE**, achieving R² ≈ 0.98.
- Included **residual plot**, **prediction fit plot**, and **actual vs. predicted** scatter for explainability.
- Simulated **parallel training time** using Dask and compared with sequential.


---

### Question 5 – Distance-Based Trip Averages

- Analyzed **average number of trips by distance** category (e.g., <1 mile to >500 miles).
- Dropped irrelevant fields and selected only numeric data for summary.
- Used both **Dask and Pandas** to calculate and compare means across distance ranges.
- Visualized findings using a **sorted bar chart** to interpret travel patterns.
- Benchmarked timing under both **sequential and simulated parallel** execution modes.


---

## Tools & Technologies  

- Python 3.12.10
- **Dask** (`dask.dataframe`, `dask.distributed`)  
- **Pandas**, **NumPy**  
- **scikit-learn** (`LinearRegression`, evaluation metrics)  
- **Matplotlib** for visualization  
- **Jupyter Notebook** for development  
- **Git** for version control and reproducibility  

---

