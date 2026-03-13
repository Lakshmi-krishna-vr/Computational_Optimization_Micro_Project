# Computational_Optimization_Micro_Project
# 🏫 Campus Logistics Optimization System

An AI-driven logistics tool that uses **Mixed-Integer Linear Programming (MILP)** to determine the most cost-effective warehouse distribution strategy for a university campus. 

## 📌 Project Overview
The goal of this project is to minimize the total annual cost of distributing emergency supplies to critical campus facilities. The system balances fixed operational costs of warehouses with variable transportation costs while strictly adhering to financial and physical constraints.

### 🎯 Objectives
- **Optimal Siting:** Select exactly 2 warehouses from potential sites.
- **Cost Minimization:** Minimize the sum of amortized construction, daily operation, and per-unit shipping costs.
- **Constraint Satisfaction:** Ensure all facility demands are met without exceeding warehouse capacities or the **$1.5M annual budget**.

---

## 🛠️ Technical Stack
- **Language:** Python 3.x
- **Optimization Library:** [PuLP](https://coin-or.github.io/pulp/) (using CBC Solver)
- **Data Analysis:** Pandas
- **Visualization:** - **Folium:** Interactive geospatial dashboard.
  - **Matplotlib & NetworkX:** Static graph visualization of the supply chain network.

---

## 📊 Logic & Mathematical Model
The project implements a **Mixed-Integer Linear Programming (MILP)** model:

### 1. Decision Variables
- $X_i$: Binary variable (1 if Warehouse $i$ is opened, 0 otherwise).
- $Y_{ij}$: Continuous variable (Units shipped from Warehouse $i$ to Facility $j$).

### 2. Constraints
- **Demand:** $\sum_{i} Y_{ij} = \text{Demand}_j$ (Every building must be fully supplied).
- **Capacity:** $\sum_{j} Y_{ij} \leq \text{Capacity}_i \cdot X_i$ (Cannot ship from closed warehouses or exceed limits).
- **Policy:** $\sum X_i = 2$ (Exactly two warehouses must be operational for redundancy).
- **Budget:** Total Annual Cost $\leq \$1,500,000$.

---

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/Lakshmi-krishna-vr/Computational_Optimization_Micro_Project)
