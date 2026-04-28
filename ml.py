import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor



def apply_ml(data):
    
    ml_df = data.copy()
    
    
    num_cols = ['sale_price', 'floor_area_sqft', 'satisfaction_score']
    for col in num_cols:
        if col in ml_df.columns:
            ml_df[col] = pd.to_numeric(ml_df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    

    if 'date_of_birth' in ml_df.columns:
        ml_df['date_of_birth'] = pd.to_datetime(ml_df['date_of_birth'], errors='coerce')
        ml_df['age'] = 2024 - ml_df['date_of_birth'].dt.year
        ml_df['age'] = ml_df['age'].fillna(ml_df['age'].median())


    cluster_features = ['age', 'sale_price', 'satisfaction_score']
    ml_df[cluster_features] = ml_df[cluster_features].fillna(ml_df[cluster_features].mean())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(ml_df[cluster_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    labels = {0: "💎 Premium Investors", 1: "🏠 Family Homebuyers", 
              2: "📈 Growth Seekers", 3: "🏢 Corporate Clients"}
    data['Buyer Segment'] = data['Cluster'].map(labels)
    data['age'] = ml_df['age']
    data['sale_price'] = ml_df['sale_price']
    data['floor_area_sqft'] = ml_df['floor_area_sqft']
    
    return data

def price_predictor_ui(data):
    """AI Price Predictor Tool"""
    st.markdown("---")
    st.subheader("🤖 AI Property Price Predictor")
    
    X = data[['age', 'floor_area_sqft', 'satisfaction_score']].fillna(0)
    y = data['sale_price'].fillna(data['sale_price'].mean())
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    c1, c2, c3 = st.columns(3)
    with c1: area = st.number_input("Area (sqft)", value=1200)
    with c2: age = st.slider("Property Age", 0, 50, 10)
    with c3: qual = st.slider("Quality Score", 1, 10, 7)

    if st.button("Predict Price"):
        pred = model.predict([[age, area, qual]])
        st.success(f"#### 💰 Estimated Market Value: ${pred[0]:,.2f}")


st.set_page_config(page_title="Parcl Real Estate Intelligence", layout="wide")
st.title("🏙️ Real Estate Buyer Intelligence")

st.sidebar.header("📤 Upload Files")
c_file = st.sidebar.file_uploader("Client CSV", type="csv")
p_file = st.sidebar.file_uploader("Properties CSV", type="csv")

if c_file and p_file:

    c_df = pd.read_csv(c_file)
    p_df = pd.read_csv(p_file)
    c_df.columns = c_df.columns.str.strip().str.lower()
    p_df.columns = p_df.columns.str.strip().str.lower()

    if 'client_id' in c_df.columns and 'client_ref' in p_df.columns:
        
        df_merged = pd.merge(c_df, p_df, left_on='client_id', right_on='client_ref', how='inner')
        
        
        final_df = apply_ml(df_merged)

        st.metric("Total Merged Records", len(final_df))
        
        
        tab1, tab2 = st.tabs(["📊 Segmentation Analysis", "🔮 Price Predictor"])
        
        with tab1:
            st.plotly_chart(px.pie(final_df, names='Buyer Segment', hole=0.4))
            st.dataframe(final_df)
        
        with tab2:

            price_predictor_ui(final_df)
    else:
        st.error("Missing linking columns: 'client_id' or 'client_ref'")
else:
    st.info("Please upload both CSV files in the sidebar to begin.")
