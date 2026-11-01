from src.analytics import load_data, build_forecast, segment_customers, inventory_recommendations

def test_load_data():
    sales, customers = load_data()
    assert not sales.empty
    assert not customers.empty

def test_forecast():
    sales, _ = load_data()
    result = build_forecast(sales, horizon=7)
    assert result.forecast.notna().sum() == 7

def test_segments():
    _, customers = load_data()
    result = segment_customers(customers)
    assert "segment" in result.columns

def test_inventory():
    sales, _ = load_data()
    result = inventory_recommendations(sales)
    assert "suggested_reorder" in result.columns
