
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Housing Price Analysis",
    page_icon="🏠",
    layout="wide",
)

DATA_PATH = r"C:\Users\Asus\OneDrive\Desktop\City_Analysis\gurgaon_city.csv"


# --------------------------------------------------------------------------
# Data loading / caching
# --------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def clean_and_engineer(df_raw: pd.DataFrame):
    df = df_raw.copy()
    missing_before = df.isnull().sum()

    # Impute missing total_bedrooms with median
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

    # Feature engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    return df, missing_before, median_bedrooms


@st.cache_data
def encode_data(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    return df_encoded


@st.cache_resource
def train_models(df_encoded: pd.DataFrame):
    X = df_encoded.drop("median_house_value", axis=1)
    y = df_encoded["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    y_pred_lr = lin_reg.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    # Random Forest
    rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_scaled, y_train)
    y_pred_rf = rf_reg.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    results = {
        "X_columns": list(X.columns),
        "numerical_cols": list(numerical_cols),
        "scaler": scaler,
        "lin_reg": lin_reg,
        "rf_reg": rf_reg,
        "rmse_lr": rmse_lr,
        "r2_lr": r2_lr,
        "rmse_rf": rmse_rf,
        "r2_rf": r2_rf,
        "y_test": y_test,
        "y_pred_lr": y_pred_lr,
        "y_pred_rf": y_pred_rf,
    }
    return results


# --------------------------------------------------------------------------
# Load + process data once
# --------------------------------------------------------------------------
df_raw = load_raw_data()
df_clean, missing_before, median_bedrooms = clean_and_engineer(df_raw)
df_encoded = encode_data(df_clean)
models = train_models(df_encoded)

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🏠 Housing Price Analysis")
st.sidebar.caption("Data Science Mini Project")
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Cleaning",
        "Descriptive Statistics",
        "Feature Engineering",
        "Exploratory Data Analysis",
        "Correlation Analysis",
        "Categorical Encoding",
        "Model Training & Evaluation",
        "Predict a Price",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** California housing data "
    f"({df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns)"
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
if section == "Overview":
    st.title("🏠 Housing Price Analysis & Prediction")
    st.markdown(
        "This app explores a housing dataset, engineers new features, "
        "visualizes relationships between variables, and trains regression "
        "models to predict **median house value**."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df_raw.shape[0]:,}")
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Missing values", int(df_raw.isnull().sum().sum()))
    col4.metric("Ocean proximity classes", df_raw["ocean_proximity"].nunique())

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Column Info")
    buf = io.StringIO()
    df_raw.info(buf=buf)
    st.text(buf.getvalue())

# --------------------------------------------------------------------------
# Data Cleaning
# --------------------------------------------------------------------------
elif section == "Data Cleaning":
    st.title("🧹 Data Cleaning — Handling Missing Values")

    st.subheader("Missing Values Before Cleaning")
    st.dataframe(missing_before[missing_before > 0].rename("missing_count"))

    st.info(
        f"`total_bedrooms` had missing values, imputed with the **median** "
        f"= {median_bedrooms:.2f}"
    )

    st.subheader("Missing Values After Cleaning")
    st.dataframe(df_clean.isnull().sum().rename("missing_count"))

    st.subheader("Cleaned Data Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

# --------------------------------------------------------------------------
# Descriptive Statistics
# --------------------------------------------------------------------------
elif section == "Descriptive Statistics":
    st.title("📊 Descriptive Statistics")
    st.dataframe(df_clean.describe(), use_container_width=True)

    st.subheader("Ocean Proximity Value Counts")
    st.bar_chart(df_clean["ocean_proximity"].value_counts())

# --------------------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------------------
elif section == "Feature Engineering":
    st.title("🛠️ Feature Engineering")
    st.markdown(
        """
        Three new ratio-based features were created:
        - **rooms_per_household** = total_rooms / households
        - **bedrooms_per_room** = total_bedrooms / total_rooms
        - **population_per_household** = population / households
        """
    )
    st.dataframe(
        df_clean[
            [
                "total_rooms",
                "total_bedrooms",
                "households",
                "population",
                "rooms_per_household",
                "bedrooms_per_room",
                "population_per_household",
            ]
        ].head(10),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
elif section == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis (Live)")
    st.caption(
        "Every chart below is interactive and updates instantly as you change "
        "the filters — hover, zoom, and pan are all supported."
    )

    # ---- Live filters ----------------------------------------------------
    with st.container(border=True):
        st.markdown("**Filters**")
        f1, f2, f3 = st.columns(3)
        with f1:
            prox_options = sorted(df_clean["ocean_proximity"].unique())
            selected_prox = st.multiselect(
                "Ocean Proximity", prox_options, default=prox_options
            )
        with f2:
            income_range = st.slider(
                "Median Income range",
                float(df_clean.median_income.min()),
                float(df_clean.median_income.max()),
                (float(df_clean.median_income.min()), float(df_clean.median_income.max())),
            )
        with f3:
            age_range = st.slider(
                "Housing Median Age range",
                int(df_clean.housing_median_age.min()),
                int(df_clean.housing_median_age.max()),
                (int(df_clean.housing_median_age.min()), int(df_clean.housing_median_age.max())),
            )

    df_live = df_clean[
        (df_clean["ocean_proximity"].isin(selected_prox))
        & (df_clean["median_income"].between(*income_range))
        & (df_clean["housing_median_age"].between(*age_range))
    ]
    st.metric("Rows matching filters", f"{len(df_live):,} / {len(df_clean):,}")

    if df_live.empty:
        st.warning("No rows match the current filters — widen your ranges above.")
        st.stop()

    # ---- Live histogram ----------------------------------------------------
    st.subheader("Distribution Explorer")
    numeric_cols = df_live.select_dtypes(include=np.number).columns.tolist()
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        hist_feature = st.selectbox(
            "Feature to plot", numeric_cols, index=numeric_cols.index("median_house_value")
        )
    with hcol2:
        n_bins = st.slider("Bins", 10, 100, 50, key="hist_bins")
    fig_hist = px.histogram(
        df_live, x=hist_feature, nbins=n_bins, color="ocean_proximity",
        marginal="box", title=f"Distribution of {hist_feature}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---- Live scatter --------------------------------------------------
    st.subheader("Relationship Explorer")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        x_feature = st.selectbox("X axis", numeric_cols, index=numeric_cols.index("median_income"))
    with scol2:
        y_feature = st.selectbox(
            "Y axis", numeric_cols, index=numeric_cols.index("median_house_value")
        )
    with scol3:
        color_by = st.selectbox(
            "Color by", ["ocean_proximity"] + numeric_cols, index=0
        )
    fig_scatter = px.scatter(
        df_live, x=x_feature, y=y_feature, color=color_by, opacity=0.4,
        title=f"{x_feature} vs. {y_feature}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Live boxplot ----------------------------------------------------
    st.subheader("Median House Value by Ocean Proximity")
    fig_box = px.box(
        df_live, x="ocean_proximity", y="median_house_value", color="ocean_proximity",
        title="Median House Value by Ocean Proximity",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ---- Live map ----------------------------------------------------
    st.subheader("Geographic Distribution")
    st.caption("Color = median house value, size = population. Scroll to zoom, drag to pan.")
    map_sample = df_live.sample(min(4000, len(df_live)), random_state=42)
    fig_map = px.scatter_mapbox(
        map_sample,
        lat="latitude",
        lon="longitude",
        color="median_house_value",
        size="population",
        color_continuous_scale="Viridis",
        size_max=18,
        zoom=4.5,
        height=550,
        hover_data=["median_income", "ocean_proximity"],
        mapbox_style="carto-positron",
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
elif section == "Correlation Analysis":
    st.title("📈 Correlation Analysis")

    corr_matrix = df_clean.corr(numeric_only=True)

    st.subheader("Correlation with Median House Value")
    st.dataframe(
        corr_matrix["median_house_value"].sort_values(ascending=False).rename(
            "correlation"
        )
    )

    st.subheader("Correlation Heatmap (Interactive)")
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Matrix of Numerical Features",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Categorical Encoding
# --------------------------------------------------------------------------
elif section == "Categorical Encoding":
    st.title("🔤 Handling Categorical Features")
    st.markdown(
        "`ocean_proximity` is one-hot encoded (drop-first) so it can be used "
        "by machine learning models."
    )
    st.dataframe(df_encoded.head(10), use_container_width=True)
    st.caption(f"Resulting shape after encoding: {df_encoded.shape}")

# --------------------------------------------------------------------------
# Model Training & Evaluation
# --------------------------------------------------------------------------
elif section == "Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    st.markdown(
        "Data was split **80/20** into train/test sets, numerical features "
        "were standardized, and two regression models were trained:"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Regression")
        st.metric("R² Score", f"{models['r2_lr']:.3f}")
        st.metric("RMSE", f"${models['rmse_lr']:,.0f}")
    with col2:
        st.subheader("Random Forest Regressor")
        st.metric("R² Score", f"{models['r2_rf']:.3f}")
        st.metric("RMSE", f"${models['rmse_rf']:,.0f}")

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest"],
            "R2 Score": [models["r2_lr"], models["r2_rf"]],
            "RMSE": [models["rmse_lr"], models["rmse_rf"]],
        }
    ).set_index("Model")
    c1, c2 = st.columns(2)
    c1.bar_chart(comp_df["R2 Score"])
    c2.bar_chart(comp_df["RMSE"])

    better = "Random Forest" if models["r2_rf"] > models["r2_lr"] else "Linear Regression"
    st.success(f"🏆 **{better}** performed better on this test set.")

    st.subheader("Predicted vs Actual (Random Forest)")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(models["y_test"], models["y_pred_rf"], alpha=0.15)
    lims = [models["y_test"].min(), models["y_test"].max()]
    ax.plot(lims, lims, "r--", linewidth=2)
    ax.set_xlabel("Actual median_house_value")
    ax.set_ylabel("Predicted median_house_value")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------------------
# Predict a Price
# --------------------------------------------------------------------------
elif section == "Predict a Price":
    st.title("🔮 Predict Median House Value")
    st.markdown("Enter feature values below to get a live price prediction.")

    model_choice = st.selectbox(
        "Choose model", ["Random Forest Regressor", "Linear Regression"]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        longitude = st.slider(
            "Longitude", float(df_clean.longitude.min()), float(df_clean.longitude.max()),
            float(df_clean.longitude.median()),
        )
        latitude = st.slider(
            "Latitude", float(df_clean.latitude.min()), float(df_clean.latitude.max()),
            float(df_clean.latitude.median()),
        )
        housing_median_age = st.slider(
            "Housing Median Age", int(df_clean.housing_median_age.min()),
            int(df_clean.housing_median_age.max()), int(df_clean.housing_median_age.median()),
        )
    with c2:
        total_rooms = st.slider(
            "Total Rooms", int(df_clean.total_rooms.min()), int(df_clean.total_rooms.max()),
            int(df_clean.total_rooms.median()),
        )
        total_bedrooms = st.slider(
            "Total Bedrooms", int(df_clean.total_bedrooms.min()),
            int(df_clean.total_bedrooms.max()), int(df_clean.total_bedrooms.median()),
        )
        population = st.slider(
            "Population", int(df_clean.population.min()), int(df_clean.population.max()),
            int(df_clean.population.median()),
        )
    with c3:
        households = st.slider(
            "Households", int(df_clean.households.min()), int(df_clean.households.max()),
            int(df_clean.households.median()),
        )
        median_income = st.slider(
            "Median Income (10k USD)", float(df_clean.median_income.min()),
            float(df_clean.median_income.max()), float(df_clean.median_income.median()),
        )
        ocean_proximity = st.selectbox(
            "Ocean Proximity", sorted(df_clean.ocean_proximity.unique())
        )

    if st.button("Predict Price", type="primary"):
        rooms_per_household = total_rooms / households
        bedrooms_per_room = total_bedrooms / total_rooms
        population_per_household = population / households

        input_row = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
        }
        # one-hot encode ocean_proximity same as training (drop_first alphabetical)
        for col in models["X_columns"]:
            if col.startswith("ocean_proximity_"):
                cat = col.replace("ocean_proximity_", "")
                input_row[col] = 1 if ocean_proximity == cat else 0

        input_df = pd.DataFrame([input_row])[models["X_columns"]]
        input_scaled = input_df.copy()
        input_scaled[models["numerical_cols"]] = models["scaler"].transform(
            input_df[models["numerical_cols"]]
        )

        model = models["rf_reg"] if model_choice == "Random Forest Regressor" else models["lin_reg"]
        prediction = model.predict(input_scaled)[0]

        st.success(f"### Predicted Median House Value: **${prediction:,.0f}**")

st.sidebar.markdown("---")