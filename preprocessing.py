def remove_capped_prices(df):
    return df[df["median_house_value"] < 500001]

def fill_missing_values(df):
    df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)
    return df