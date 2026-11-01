from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    sales = pd.read_csv(ROOT/"data"/"sales.csv", parse_dates=["date"])
    customers = pd.read_csv(ROOT/"data"/"customers.csv")
    return sales, customers

def build_forecast(sales, horizon=30, product_id="P001"):
    daily=(sales[sales.product_id==product_id].groupby("date")["quantity"].sum()
           .asfreq("D",fill_value=0).rename("y").to_frame())
    daily["lag_1"]=daily.y.shift(1)
    daily["lag_7"]=daily.y.shift(7)
    daily["rolling_7"]=daily.y.shift(1).rolling(7).mean()
    daily["dow"]=daily.index.dayofweek
    train=daily.dropna()
    X=train[["lag_1","lag_7","rolling_7","dow"]]
    y=train["y"]
    model=RandomForestRegressor(n_estimators=250,random_state=42,min_samples_leaf=2)
    model.fit(X,y)
    history=daily.y.tolist()
    future=[]
    last_date=daily.index.max()
    for step in range(1,horizon+1):
        next_date=last_date+pd.Timedelta(days=step)
        features=pd.DataFrame([{
            "lag_1":history[-1],
            "lag_7":history[-7] if len(history)>=7 else history[-1],
            "rolling_7":float(np.mean(history[-7:])),
            "dow":next_date.dayofweek
        }])
        pred=max(0,float(model.predict(features)[0]))
        history.append(pred)
        future.append([next_date,pred])
    actual=daily[["y"]].tail(60).reset_index()
    actual.columns=["date","actual"]
    future_df=pd.DataFrame(future,columns=["date","forecast"])
    actual["forecast"]=np.nan
    future_df["actual"]=np.nan
    return pd.concat([actual[["date","actual","forecast"]],future_df[["date","actual","forecast"]]],ignore_index=True)

def segment_customers(customers, n_clusters=4):
    features=["recency_days","frequency","monetary_value","tenure_days"]
    data=customers.copy()
    scaler=StandardScaler()
    scaled=scaler.fit_transform(data[features])
    model=KMeans(n_clusters=min(n_clusters,len(data)),random_state=42,n_init=10)
    data["cluster"]=model.fit_predict(scaled)
    # Label clusters using simple, interpretable recency/frequency/spend scores.
    stats=data.groupby("cluster")[["recency_days","frequency","monetary_value"]].mean()
    score=(stats["frequency"].rank(pct=True)+stats["monetary_value"].rank(pct=True)-stats["recency_days"].rank(pct=True))
    labels={idx:("Champions" if val==score.max() else "At risk" if stats.loc[idx,"recency_days"]>stats.recency_days.median() else "Regular") for idx,val in score.items()}
    data["segment"]=data.cluster.map(labels)
    return data

def train_churn_model(customers):
    features=["recency_days","frequency","monetary_value","tenure_days"]
    X=customers[features]
    y=customers["churn"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=250,max_depth=5,class_weight="balanced",random_state=42)
    model.fit(X_train,y_train)
    probability=model.predict_proba(X_test)[:,1]
    scored=customers.loc[X_test.index,["customer_id","recency_days","frequency","monetary_value"]].copy()
    scored["churn_probability"]=probability
    auc=roc_auc_score(y_test,probability) if y_test.nunique()>1 else 0.0
    return model,scored,{"roc_auc":auc}

def inventory_recommendations(sales,current_stock=150):
    recent=sales[sales.date>=sales.date.max()-pd.Timedelta(days=14)]
    avg=recent.groupby("product_id")["quantity"].sum()/14
    out=avg.rename("avg_daily_demand").to_frame()
    out["recommended_stock"]=np.ceil(out.avg_daily_demand*14*1.25).astype(int)
    out["assumed_current_stock"]=int(current_stock)
    out["suggested_reorder"]=np.maximum(0,out.recommended_stock-out.assumed_current_stock)
    return out.reset_index().round(1)
