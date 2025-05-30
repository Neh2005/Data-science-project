# 8. Summary Interpretation

# --- 1. Mean number of people staying at home ---
mean_staying_home = grouped_national['Population Staying at Home'].mean()
print(f"\n On average, {mean_staying_home:,.0f} people were staying at home per day (national level).")

# --- 2. Mean number of people not staying at home ---
mean_not_staying_home = grouped_small['People Not Staying at Home'].mean()
print(f" On average, {mean_not_staying_home:,.0f} people were NOT staying at home per day (national level).")

# --- 3. How far are people traveling ---
distance_columns = [col for col in df2.columns if 'Trips' in col and ('<' in col or '-' in col or 'Miles' in col)]
if distance_columns:
    trip_means = df2[distance_columns].select_dtypes(include='number').mean()

    plt.figure(figsize=(10, 5))
    x_vals = np.arange(len(trip_means))
    y_vals = trip_means.values
    z3 = np.polyfit(x_vals, y_vals, 1)
    p3 = np.poly1d(z3)
if not trip_means.empty:
    print("\n Average number of trips by distance (when people didn't stay home):")
    for dist, avg in trip_means.items():
        print(f"  - {dist}: {avg:,.0f} trips per day")

