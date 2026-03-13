import pulp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import folium
import os
import webbrowser

# --- 1. DATA GENERATION (As per Requirements 2.10.2, 2.10.3, 2.10.4) ---
def generate_project_data():
    """Generates the specific data provided in the Micro-Project brief."""
    # 2.10.2 Facilities Data
    facilities_data = {
        'facility_id': ['MED_CENTER', 'ENG_BUILDING', 'SCIENCE_HALL', 'DORM_A', 'DORM_B', 'LIBRARY'],
        'latitude': [40.8075, 40.8095, 40.8105, 40.8055, 40.8045, 40.8065],
        'longitude': [-73.9626, -73.9600, -73.9615, -73.9640, -73.9655, -73.9635],
        'daily_demand': [80, 30, 35, 55, 45, 25]
    }
    
    # 2.10.3 Warehouse Data
    warehouses_data = {
        'warehouse_id': ['WH_NORTH', 'WH_SOUTH', 'WH_EAST'],
        'latitude': [40.8120, 40.8030, 40.8090],
        'longitude': [-73.9620, -73.9645, -73.9580],
        'daily_capacity': [400, 350, 450],
        'construction_cost': [300000, 280000, 320000],
        'operational_cost_day': [800, 700, 900]
    }
    
    # 2.10.4 Transportation Costs (Simulating the pre-calculated matrix)
    # Using the project cost range: $3.68 - $5.03 per unit
    costs = []
    unit_rates = {'WH_NORTH': 3.85, 'WH_SOUTH': 4.10, 'WH_EAST': 4.50}
    for w in warehouses_data['warehouse_id']:
        for f in facilities_data['facility_id']:
            costs.append({'from_warehouse': w, 'to_facility': f, 'cost_per_unit': unit_rates[w]})

    return pd.DataFrame(facilities_data), pd.DataFrame(warehouses_data), pd.DataFrame(costs)

# --- 2. PRE-PROCESSING (Annualization & Amortization) ---
def pre_process_data(fac_df, wh_df, cost_df):
    facilities = fac_df['facility_id'].tolist()
    warehouses = wh_df['warehouse_id'].tolist()
    
    # 2.10.5 & 2.10.6: Annualize Demand and Capacity (365 days)
    annual_demand = {row['facility_id']: row['daily_demand'] * 365 for _, row in fac_df.iterrows()}
    annual_capacity = {row['warehouse_id']: row['daily_capacity'] * 365 for _, row in wh_df.iterrows()}
    
    # 2.10.5: Annual Fixed Cost = (Construction / 10) + (Daily Ops * 365)
    fixed_costs = {
        row['warehouse_id']: (row['construction_cost'] / 10) + (row['operational_cost_day'] * 365) 
        for _, row in wh_df.iterrows()
    }

    transport_costs = {(row['from_warehouse'], row['to_facility']): row['cost_per_unit'] for _, row in cost_df.iterrows()}
    
    locations = {row['facility_id']: (row['longitude'], row['latitude']) for _, row in fac_df.iterrows()}
    locations.update({row['warehouse_id']: (row['longitude'], row['latitude']) for _, row in wh_df.iterrows()})
    
    return facilities, warehouses, annual_demand, annual_capacity, fixed_costs, transport_costs, locations

# --- 3. OPTIMIZATION (MILP Formulation) ---
def solve_logistics(facilities, warehouses, annual_demand, annual_capacity, fixed_costs, transport_costs):
    prob = pulp.LpProblem("Campus_Logistics_Optimization", pulp.LpMinimize)
    
    # Decision Variables
    wh_vars = pulp.LpVariable.dicts("Open", warehouses, cat='Binary')
    flow_vars = pulp.LpVariable.dicts("Ship", (warehouses, facilities), lowBound=0, cat='Continuous')

    # Objective: Minimize Total Annual Costs
    total_cost_expr = (pulp.lpSum([fixed_costs[w] * wh_vars[w] for w in warehouses]) +
                       pulp.lpSum([transport_costs[(w, f)] * flow_vars[w][f] for w in warehouses for f in facilities]))
    prob += total_cost_expr

    # Constraints
    # 1. Demand Satisfaction
    for f in facilities:
        prob += pulp.lpSum([flow_vars[w][f] for w in warehouses]) == annual_demand[f]
    
    # 2. Capacity Limits
    for w in warehouses:
        prob += pulp.lpSum([flow_vars[w][f] for f in facilities]) <= annual_capacity[w] * wh_vars[w]
    
    # 3. Policy: Exactly 2 warehouses
    prob += pulp.lpSum([wh_vars[w] for w in warehouses]) == 2
    
    # 4. 2.10.6 Budget Limit: Total annual cost <= $1,500,000
    prob += total_cost_expr <= 1500000
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, wh_vars, flow_vars

# --- 4. REPORTING ---
def report_results(prob, facilities, warehouses, wh_vars, flow_vars, transport_costs, fixed_costs):
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("ERROR: No optimal solution found.")
        return

    # 1. Calculate Summary Totals first
    total_fixed = sum(fixed_costs[w] for w in warehouses if wh_vars[w].varValue == 1)
    total_transport = sum(flow_vars[w][f].varValue * transport_costs[(w, f)] for w in warehouses for f in facilities)
    
    # 2. Print the Header ONLY ONCE
    print("\n" + "="*65)
    print(" PROJECT 1: CAMPUS LOGISTICS OPTIMIZATION REPORT")
    print("="*65)
    print(f"STATUS:               {pulp.LpStatus[prob.status]}")
    print(f"TOTAL ANNUAL COST:    ${pulp.value(prob.objective):,.2f}")
    print(f"ANNUAL BUDGET:        $1,500,000.00")
    print(f"BUDGET UTILIZATION:   {(pulp.value(prob.objective)/1500000)*100:.2f}%")
    print("-" * 65)
    print(f"Fixed Costs (Amortized):  ${total_fixed:,.2f}")
    print(f"Transportation Costs:     ${total_transport:,.2f}")
    print("-" * 65)
    
    # 3. Print the Warehouse Status
    print("WAREHOUSE STATUS:")
    for w in warehouses:
        status = "OPEN (Active)" if wh_vars[w].varValue == 1 else "CLOSED"
        # Check if it's actually shipping or just a backup
        actual_shipping = sum(flow_vars[w][f].varValue for f in facilities)
        if status == "OPEN (Active)" and actual_shipping == 0:
            status = "OPEN (Standby/Backup)"
        print(f"- {w:<12}: {status}")
    print("-" * 65)

    # 4. Print the Route Table ONLY ONCE
    print(f"{'ACTIVE ROUTE':<30} | {'UNITS':<10} | {'ANNUAL COST':<15}")
    for w in warehouses:
        for f in facilities:
            qty = flow_vars[w][f].varValue
            if qty > 0: # This ensures we don't print empty/zero routes
                cost = qty * transport_costs[(w, f)]
                print(f"{w + ' -> ' + f:<30} | {int(qty):<10} | ${cost:,.2f}")
    print("="*65 + "\n")

# --- 5. VISUALIZATION (Static & Interactive) ---
def visualize(facilities, warehouses, wh_vars, flow_vars, locations):
    # Static Map
    plt.figure(figsize=(10, 6))
    G = nx.DiGraph()
    open_wh = [w for w in warehouses if wh_vars[w].varValue == 1]
    nx.draw_networkx_nodes(G, locations, nodelist=facilities, node_color='skyblue', node_size=500, label='Buildings')
    nx.draw_networkx_nodes(G, locations, nodelist=open_wh, node_color='green', node_shape='s', node_size=800, label='Open Warehouses')
    for w in warehouses:
        for f in facilities:
            if flow_vars[w][f].varValue > 0:
                nx.draw_networkx_edges(G, locations, edgelist=[(w, f)], width=2, edge_color='green', style='dashed')
    nx.draw_networkx_labels(G, locations, font_size=8, font_weight='bold')
    plt.title("Static Optimization Map")
    plt.show(block=False)

    # Interactive Map
    m = folium.Map(location=[40.8075, -73.9626], zoom_start=15, tiles='CartoDB positron')
    for w in warehouses:
        if wh_vars[w].varValue == 1:
            folium.Marker(locations[w][::-1], icon=folium.Icon(color='green', icon='home')).add_to(m)
            folium.Circle(locations[w][::-1], radius=350, color='green', fill=True, fill_opacity=0.05).add_to(m)
    for f in facilities:
        folium.CircleMarker(locations[f][::-1], radius=10, color='#00FFFF', fill=True, fill_opacity=0.8).add_to(m)
        for w in warehouses:
            if flow_vars[w][f].varValue > 0:
                folium.PolyLine([locations[w][::-1], locations[f][::-1]], color='#32CD32', weight=3, dash_array='10, 10').add_to(m)
    
    m.save("project_optimization_map.html")
    webbrowser.open("file://" + os.path.realpath("project_optimization_map.html"))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    fac_df, wh_df, cost_df = generate_project_data()
    f_list, w_list, a_dem, a_cap, f_cost, t_cost, locs = pre_process_data(fac_df, wh_df, cost_df)
    
    prob, wh_vars, flow_vars = solve_logistics(f_list, w_list, a_dem, a_cap, f_cost, t_cost)
    
    report_results(prob, f_list, w_list, wh_vars, flow_vars, t_cost, f_cost)
    if pulp.LpStatus[prob.status] == 'Optimal':
        visualize(f_list, w_list, wh_vars, flow_vars, locs)
        plt.show()
        