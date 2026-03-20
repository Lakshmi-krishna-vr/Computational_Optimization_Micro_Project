import pulp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import folium
import os
import webbrowser

# ---------------------------------------------------
# 1. LOAD DATA FROM CSV FILES
# ---------------------------------------------------
def load_data():

    facilities_df = pd.read_csv("facilities.csv")
    warehouses_df = pd.read_csv("warehouses.csv")
    transport_df = pd.read_csv("transportation_costs.csv")
    demand_df = pd.read_csv("demand.csv")
    geo_df = pd.read_csv("geographic_bounds.csv")

    return facilities_df, warehouses_df, transport_df, demand_df, geo_df


# ---------------------------------------------------
# 2. PREPROCESS DATA
# ---------------------------------------------------
def preprocess_data(fac_df, wh_df, cost_df, demand_df):

    facilities = fac_df["facility_id"].tolist()
    warehouses = wh_df["warehouse_id"].tolist()

    # Daily demand dictionary
    daily_demand = {
        row["facility_id"]: row["daily_demand"]
        for _, row in demand_df.iterrows()
    }

    # Annual demand
    annual_demand = {f: daily_demand[f] * 365 for f in facilities}

    # Annual warehouse capacity
    annual_capacity = {
        row["warehouse_id"]: row["capacity"] * 365
        for _, row in wh_df.iterrows()
    }

    # Annual fixed cost
    fixed_costs = {
        row["warehouse_id"]:
        (row["construction_cost"] / 10) + (row["operational_cost"] * 365)
        for _, row in wh_df.iterrows()
    }

    # Transportation costs
    transport_costs = {
        (row["from_warehouse"], row["to_facility"]): row["cost_per_unit"]
        for _, row in cost_df.iterrows()
    }

    # Locations for visualization
    locations = {
        row["facility_id"]: (row["longitude"], row["latitude"])
        for _, row in fac_df.iterrows()
    }

    locations.update({
        row["warehouse_id"]: (row["longitude"], row["latitude"])
        for _, row in wh_df.iterrows()
    })

    return facilities, warehouses, annual_demand, annual_capacity, fixed_costs, transport_costs, locations


# ---------------------------------------------------
# 3. OPTIMIZATION MODEL (MILP)
# ---------------------------------------------------
def solve_logistics(facilities, warehouses, annual_demand, annual_capacity, fixed_costs, transport_costs):

    prob = pulp.LpProblem("Campus_Logistics_Optimization", pulp.LpMinimize)

    # Decision variables
    warehouse_open = pulp.LpVariable.dicts("OpenWarehouse", warehouses, cat="Binary")

    shipment = pulp.LpVariable.dicts("Ship", (warehouses, facilities), lowBound=0)

    # Objective function       Z=   w∑​Fw​⋅yw ​+   w∑​f∑​Cwf​⋅xwf​
    total_cost = (
        pulp.lpSum(fixed_costs[w] * warehouse_open[w] for w in warehouses) +
        pulp.lpSum(transport_costs[(w, f)] * shipment[w][f] for w in warehouses for f in facilities)
    )

    prob += total_cost

    # Demand satisfaction constraint
    for f in facilities:
        prob += pulp.lpSum(shipment[w][f] for w in warehouses) == annual_demand[f]

    # Capacity constraint
    for w in warehouses:
        prob += pulp.lpSum(shipment[w][f] for f in facilities) <= annual_capacity[w] * warehouse_open[w]

    # Select exactly 2 warehouses
    prob += pulp.lpSum(warehouse_open[w] for w in warehouses) == 2

    # Budget constraint
    prob += total_cost <= 1500000

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return prob, warehouse_open, shipment


# ---------------------------------------------------
# 4. REPORT RESULTS
# ---------------------------------------------------
def report_results(prob, facilities, warehouses, wh_vars, flow_vars, transport_costs, fixed_costs):

    print("\n" + "="*60)
    print("CAMPUS EMERGENCY SUPPLY DISTRIBUTION REPORT")
    print("="*60)

    print("Status:", pulp.LpStatus[prob.status])
    print("Total Annual Cost:", pulp.value(prob.objective))

    total_fixed = sum(fixed_costs[w] for w in warehouses if wh_vars[w].varValue == 1)

    total_transport = sum(
        flow_vars[w][f].varValue * transport_costs[(w, f)]
        for w in warehouses for f in facilities
    )

    print("Fixed Cost:", total_fixed)
    print("Transportation Cost:", total_transport)

    print("\nWarehouse Status:")
    for w in warehouses:
        if wh_vars[w].varValue == 1:
            print(w, "OPEN")
        else:
            print(w, "CLOSED")

    print("\nActive Routes:")
    for w in warehouses:
        for f in facilities:
            qty = flow_vars[w][f].varValue
            if qty > 0:
                cost = qty * transport_costs[(w, f)]
                print(w, "->", f, "Units:", int(qty), "Cost:", cost)

    print("="*60)


# ---------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------
def visualize(facilities, warehouses, wh_vars, flow_vars, locations, geo_df):

    center_lat = geo_df["center_lat"][0]
    center_lon = geo_df["center_lon"][0]

    # Static graph
    plt.figure(figsize=(10,6))
    G = nx.DiGraph()

    open_wh = [w for w in warehouses if wh_vars[w].varValue == 1]

    nx.draw_networkx_nodes(G, locations, nodelist=facilities,
                           node_color="skyblue", node_size=600)

    nx.draw_networkx_nodes(G, locations, nodelist=open_wh,
                           node_color="green", node_shape="s", node_size=800)

    for w in warehouses:
        for f in facilities:
            if flow_vars[w][f].varValue > 0:
                nx.draw_networkx_edges(G, locations, edgelist=[(w,f)], width=2)

    nx.draw_networkx_labels(G, locations)
    plt.title("Campus Supply Distribution Network")
    plt.show(block=False)

    # Interactive map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    for w in warehouses:
        if wh_vars[w].varValue == 1:
            folium.Marker(locations[w][::-1],
                          icon=folium.Icon(color="green")).add_to(m)

    for f in facilities:
        folium.CircleMarker(locations[f][::-1],
                            radius=8,
                            color="blue",
                            fill=True).add_to(m)

        for w in warehouses:
            if flow_vars[w][f].varValue > 0:
                folium.PolyLine([locations[w][::-1], locations[f][::-1]],
                                color="green").add_to(m)

    m.save("campus_supply_map.html")
    webbrowser.open("file://" + os.path.realpath("campus_supply_map.html"))


# ---------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------
if __name__ == "__main__":

    fac_df, wh_df, cost_df, demand_df, geo_df = load_data()

    f_list, w_list, a_dem, a_cap, f_cost, t_cost, locs = preprocess_data(
        fac_df, wh_df, cost_df, demand_df
    )

    prob, wh_vars, flow_vars = solve_logistics(
        f_list, w_list, a_dem, a_cap, f_cost, t_cost
    )

    report_results(prob, f_list, w_list, wh_vars, flow_vars, t_cost, f_cost)

    if pulp.LpStatus[prob.status] == "Optimal":
        visualize(f_list, w_list, wh_vars, flow_vars, locs, geo_df)
        plt.show()