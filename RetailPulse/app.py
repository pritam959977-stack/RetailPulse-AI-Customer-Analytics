import streamlit as st
import pandas as pd
import plotly.express as px
from src.analytics import load_data, build_forecast, train_churn_model, segment_customers, inventory_recommendations

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide")
st.title("📈 RetailPulse")
st.caption("AI-powered customer analytics & demand forecasting | Demo project")

sales, customers = load_data()
total_revenue = sales["revenue"].sum()
units = sales["quantity"].sum()
a,b,c,d = st.columns(4)
a.metric("Total revenue", f"₹{total_revenue:,.0f}")
b.metric("Units sold", f"{units:,.0f}")
c.metric("Customers", f"{customers.customer_id.nunique():,}")
c2 = customers["churn"].mean()*100
c.metric("Churn rate", f"{c2:.1f}%")
d.metric("Products", f"{sales.product_id.nunique()}")

tab1,tab2,tab3,tab4 = st.tabs(["Sales & forecast","Customer segments","Churn prediction","Inventory"])
with tab1:
    st.subheader("Sales performance")
    daily=sales.groupby("date",as_index=False).agg(revenue=("revenue","sum"),units=("quantity","sum"))
    daily["date"]=pd.to_datetime(daily["date"])
    st.plotly_chart(px.line(daily,x="date",y="revenue",title="Daily revenue"),use_container_width=True)
    product=sales.groupby("product_id",as_index=False).agg(revenue=("revenue","sum"),units=("quantity","sum"))
    st.plotly_chart(px.bar(product,x="product_id",y="units",title="Units sold by product"),use_container_width=True)
    horizon=st.slider("Forecast horizon (days)",7,60,30)
    fc=build_forecast(sales,horizon)
    st.plotly_chart(px.line(fc,x="date",y=["actual","forecast"],title="Demand forecast (P001)"),use_container_width=True)
with tab2:
    st.subheader("RFM-style customer segmentation")
    seg=segment_customers(customers)
    st.plotly_chart(px.scatter(seg,x="frequency",y="monetary_value",color="segment",hover_data=["customer_id"]),use_container_width=True)
    st.dataframe(seg.groupby("segment").agg(customers=("customer_id","count"),avg_spend=("monetary_value","mean")).round(2),use_container_width=True)
with tab3:
    st.subheader("Customer churn risk")
    model, scored, metrics=train_churn_model(customers)
    st.write(f"Holdout ROC-AUC: **{metrics['roc_auc']:.3f}**")
    st.dataframe(scored.sort_values("churn_probability",ascending=False).head(15),use_container_width=True)
with tab4:
    st.subheader("Inventory recommendations")
    stock=st.number_input("Assumed current stock for each product (units)",min_value=0,value=150)
    rec=inventory_recommendations(sales,stock)
    st.dataframe(rec,use_container_width=True)
    st.caption("Demo heuristic: 14-day average demand × 1.25 safety factor. Replace assumed stock with live inventory data in production.")

st.sidebar.info("Sample data is synthetic. Replace data/*.csv with validated business data.")
