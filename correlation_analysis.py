def calculate_correlations(df):
    cols = [
        "median_house_value",
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "median_income"
    ]
    correlations = df[cols].corr()
    return correlations["median_house_value"].sort_values(ascending=False)