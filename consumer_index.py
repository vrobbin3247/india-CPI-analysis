import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import json
import joblib
# from streamlit_theme import st_theme
import time
import tensorflow as tf
import numpy as np





st.title("Consumer Price Index (CPI) Analysis: 2013-2025")
# st.divider()

st.logo(
    'INDIA_CPI_ANALYSIS.png',
    link="https://github.com/vrobbin3247/india-CPI-analysis",
    size="large"
)
# File path
DATA_URL = "cpi_data_mospi_base12.csv"
# theme = st_theme()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # Convert Month names to numbers
    df['month'] = df['month'].str.strip()
    df['month'] = pd.to_datetime(df['month'], format='%B', errors='coerce').dt.month

    # Create Date column
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # Set Date as Index
    df.set_index('Date', inplace=True)

    df = df.sort_index()
    # Drop redundant columns
    df.drop(['year', 'month'], axis=1, inplace=True)

    return df


# Load data
df = load_data()

# 🔹 **Group-to-Subgroup Mapping**
group_to_subgroup = {
    "General": [],
    "Food and Beverages": ["Cereals and Products", "Meat and Fish", "Egg",
                           "Milk and Products", "Oils and Fats", "Fruits",
                           "Vegetables", "Pulses and Products", "Sugar and Confectionery",
                           "Spices", "Prepared Meals, Snacks, Sweets etc.",
                           "Non-alcoholic Beverages"],
    "Pan, Tobacco and Intoxicants": [],
    "Clothing and Footwear": ["Clothing", "Footwear"],
    "Housing": [],
    "Fuel and Light": [],
    "Miscellaneous": ["Household Goods and Services", "Health", "Transport and Communication",
                      "Recreation and Amusement", "Education", "Personal Care and Effects"],
    "Consumer Food Price": []
}

def line_chart():
    st.subheader("CPI Trends Over Time")
    st.caption("use sidebar for filtering")
    # # Sidebar filters
    st.sidebar.header("Filters")
    min_date, max_date = df.index.min(), df.index.max()

    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM",

    )

    # Convert date_range to pandas Timestamp
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # **Ensure correct index name**
    if df.index.name != "Date":
        df.index.name = "Date"  # Just in case it's renamed somewhere

    # **Filter Data by Date Range**
    filtered_df = df.loc[start_date:end_date]

    # **State Selection**
    selected_states = st.sidebar.multiselect("Select States", df['state'].unique(), default=["All India"])

    # **Sector Selection**
    selected_sectors = st.sidebar.multiselect("Select Sectors", df['sector'].unique(), default=["Urban"])

    # **Group Selection**
    selected_groups = st.sidebar.multiselect("Select Groups", df['group'].unique(), default=["Food and Beverages"])

    # **Subgroup Selection (Filtered)**
    available_subgroups = set()
    for group in selected_groups:
        available_subgroups.update(group_to_subgroup.get(group, []))

    # Only show subgroups if they exist
    if available_subgroups:
        selected_subgroups = st.sidebar.multiselect(
            "Select Subgroups",
            list(available_subgroups),
            default=list(available_subgroups)[:1]
        )
    else:
        selected_subgroups = []  # No selection for groups without subgroups
    st.sidebar.caption("Currently standalone groups and subgroups combinations are not working")

    # **Filter Data**
    if selected_subgroups:
        filtered_df = filtered_df[
            (filtered_df['state'].isin(selected_states)) &
            (filtered_df['sector'].isin(selected_sectors)) &
            (filtered_df['group'].isin(selected_groups)) &
            (filtered_df['subgroup'].isin(selected_subgroups))
            ]
    else:
        filtered_df = filtered_df[
            (filtered_df['state'].isin(selected_states)) &
            (filtered_df['sector'].isin(selected_sectors)) &
            (filtered_df['group'].isin(selected_groups))
            ]

    # **Plot CPI Trends**

    if not filtered_df.empty:
        # If group has no subgroups, remove 'subgroup' from pivot
        if not selected_subgroups:
            pivot_df = filtered_df.pivot_table(index=filtered_df.index,
                                               columns=['state', 'sector', 'group'],
                                               values='index')
        else:
            pivot_df = filtered_df.pivot_table(index=filtered_df.index,
                                               columns=['state', 'sector', 'group', 'subgroup'],
                                               values='index')

        # Flatten MultiIndex Columns
        pivot_df.columns = [' - '.join(col).strip() for col in pivot_df.columns.to_flat_index()]

        st.line_chart(pivot_df)  # Pass processed DataFrame

        # show checkbox
        agree = st.sidebar.checkbox("Show Filtered Data")
        if agree:
            st.write(pivot_df)

        #download
        st.sidebar.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv(index=True),
            file_name="filtered_cpi_data.csv",
            mime="text/csv",
        )
        # st.subheader("CPI Change Heatmap")
        #
        # heatmap_df = filtered_df.pivot_table(index="state", columns="sector", values="index", aggfunc="mean")
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, ax=ax)
        #
        # st.pyplot(fig)


    else:
        st.warning("No data available for the selected filters.")

def old_new_extractor(state, sector):
    data = (df['state'] == state) & (df["sector"] == sector ) & (df["group"] == "General")
    old_gi = df[data]['index'].values[-2]
    new_gi = df[data]['index'].values[-1]
    gi_date = df[data].index.values[-1]
    return old_gi, new_gi,gi_date

# selected_state = st.multiselect("Select States", df['state'].unique(), default=["All India"],max_selections=1)
def show_map(sector,latest_cpi):
    # Load the converted GeoJSON file
    with open("states_india.geojson", "r") as f:
        india_states = json.load(f)

    state_mapping = {
        "Andaman and Nicobar Islands": "Andaman and Nicobar Island",
        "Arunachal Pradesh": "Arunanchal Pradesh",
        "Dadra And Nagar Haveli": "Dadra and Nagar Haveli",
        "Daman And Diu": "Daman and Diu",
        "Delhi": "NCT of Delhi",
        "Jammu and Kashmir": "Jammu and Kashmir",
        "Odisha": "Odisha",  # Odisha is already in GeoJSON, but double-check
        "Uttarakhand": "Uttarakhand",  # Correct in both
    }

    # Apply mapping
    latest_cpi["state"] = latest_cpi["state"].replace(state_mapping)
    # st.write(latest_cpi)

    custom_colorscale = [
        (0.0, "#09AB3B"),  # Final Green
        (0.5, "#60C04B"),  # Yellowish-Green
        (0.75, "#FF9B4B"),  # Orange transition
        (1.0, "#FF4B4B"),  # Bright Red
    ]
    # Choropleth Map with Enhancements
    fig = px.choropleth(
        latest_cpi,
        geojson=india_states,
        locations="state",
        featureidkey="properties.st_nm",
        color="index",
        # color_continuous_scale="Viridis",  # Improved color scheme
        color_continuous_scale="PuBu",  # Improved color scheme
        hover_name="state",  # Show state name on hover
        hover_data={"index": True, "state": False},  # Show only index in hover
    )
    fig.update_layout(
        # width=900,  # Set overall figure width
        # height=600,  # Set overall figure height
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Remove extra margins
        geo=dict(
            center={"lat": 22, "lon": 80},  # Center the map on India
            projection_scale=6,  # Adjust zoom level (lower values zoom out)
        ),
        showlegend=False
    )

    fig.update_geos(
        visible=False,
        fitbounds="locations",  # Fit to India map
        showcountries=False,
        countrycolor="black",
        showframe=False,
        showcoastlines=False,
        # bgcolor=theme['backgroundColor'],
        bgcolor="#0E1117"

    )

    st.plotly_chart(fig, use_container_width=False)

def multi_chart():
    st.subheader("CPI of all States")
    colb1, colb2 = st.columns(2)
    sector = colb1.pills("Select Sector", ["Rural", "Urban", "Combined"], default="Rural")
    data_choice = colb2.pills("Select Visualization", ["Map", "Table", "Bar"], default="Bar")
    latest_cpi = df[(df["group"] == "General") & (df["sector"] == sector)].groupby("state")[
        "index"].last().reset_index()

    if data_choice == "Map":
        with st.spinner("Pls wait...", show_time=True):
            show_map(sector, latest_cpi)
    elif data_choice == "Table":
        st.write(latest_cpi.sort_values(by="index"))
        # st.table(latest_cpi)
    elif data_choice == "Bar":
        chart = alt.Chart(latest_cpi).mark_bar().encode(
            x=alt.X("state:N", title="State", sort="y"),
            y=alt.Y("index:Q", title="Index"),
            tooltip=["state", "index"]
        ).properties(
            width=600,
            height=400,
        )
        st.write(chart)


cola1,cola2 = st.columns([5,2.5],gap='small', vertical_alignment="center",border=False)
popover = cola2.popover("State")
selected_state = popover.pills(options=df['state'].unique(), default=["All India"], label=None)


rural_old_gi, rural_new_gi, rural_date = old_new_extractor(selected_state,"Rural")
urban_old_gi, urban_new_gi,urban_date = old_new_extractor(selected_state,"Urban")
rural_urban_old_gi, rural_urban_new_gi,combined_date = old_new_extractor(selected_state,"Combined")
gi_date = pd.Timestamp(rural_date)

cola1.subheader(f"CPI of {selected_state} for {gi_date.strftime('%B %Y')}")
col1, col2, col3 = st.columns(3)
col1.metric("Rural", value=rural_new_gi, delta=f"{round(((rural_new_gi - rural_old_gi) / rural_old_gi) * 100, 2)}%", delta_color="inverse")
col2.metric("Urban", value=urban_new_gi, delta=f"{round(((urban_new_gi - urban_old_gi) / urban_old_gi) * 100, 2)}%", delta_color="inverse" )
col3.metric("Combined", value=rural_urban_new_gi, delta=f"{round(((rural_urban_new_gi - rural_urban_old_gi) / rural_urban_old_gi) * 100, 2)}%", delta_color="inverse")

def predictions(state,sector,group,subgroup):
    # Combined Status Container
    with st.status("Initializing & Predicting... Please wait.", expanded=True) as status:
        # Step 1: Load Model
        st.write("Loading Model...")
        time.sleep(1)
        model = tf.keras.models.load_model("lstm_rural_model_12.keras")
        st.success("Model loaded successfully!")

        # Step 2: Load Scaler
        st.write("Loading Scaler...")
        time.sleep(1)
        scaler = joblib.load("scaler.pkl")
        st.success("Scaler loaded successfully!")

        # Step 3: Load Data
        st.write("Loading Data...")
        time.sleep(1)
        if subgroup:
            data2 = df[(df['state'] == state) & (df["sector"] == sector) & (df["group"] == group) & (df["subgroup"] == subgroup)]
            st.success(f"{state}-{sector}-{group}-{subgroup} Data loaded successfully!")
        else:
            data2 = df[(df['state'] == state) & (df["sector"] == sector) & (df["group"] == group)]
            st.success(f"{state}-{sector}-{group} Data loaded successfully!")

        last_sequence = data2['index'].values.reshape(-1, 1)
        data = np.array(last_sequence)

        # Step 4: Predict Future Values
        st.write("Predicting future values...")

        def predict_future(model, data, scaler, window_size=12, steps=12):
            """Predicts the next `steps` months using the last `window_size` data points."""

            # Scale the data
            scaled_data = scaler.transform(data.reshape(-1, 1))

            # Start with the last `window_size` values
            last_window = scaled_data[-window_size:].flatten()
            predictions = []

            # Streamlit progress bar
            progress_bar = st.progress(0)

            for step in range(steps):
                input_seq = last_window.reshape(1, -1)

                # Predict next value
                pred_scaled = model.predict(input_seq)[0][0]

                # Convert back to original scale
                pred_original = scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]
                predictions.append(pred_original)

                # Update last_window for next prediction
                last_window = np.append(last_window[1:], pred_scaled)

                # Update progress bar
                progress_bar.progress((step + 1) / steps)
                time.sleep(0.5)

            progress_bar.empty()
            return np.array(predictions)

        future_predictions = predict_future(model, data, scaler, window_size=12, steps=5)

        status.update(label="Initialization & Prediction Complete!", state="complete", expanded=False)

    # Generate time index starting from April 2013
    start_date = "2013-01-01"
    date_range = pd.date_range(start=start_date, periods=len(data), freq="M")
    future_dates = pd.date_range(start=date_range[-1] + pd.DateOffset(months=1), periods=len(future_predictions),
                                 freq="M")

    # Flatten the data
    data = data.flatten()

    # Insert the last actual data point at the beginning of future predictions
    future_predictions = np.insert(future_predictions, 0, data[-1])

    # Adjust future dates to match the length of future_predictions
    future_dates = pd.date_range(start=date_range[-1], periods=len(future_predictions), freq="M")

    # Prepare data for line chart
    df_chart = pd.DataFrame({
        "Date": np.concatenate([date_range, future_dates]),  # Now both arrays have matching lengths
        "Value": np.concatenate([data, future_predictions]),
        "Type": ["Historical"] * len(data) + ["Predicted"] * len(future_predictions)
    })

    # Plot using Streamlit's interactive chart
    if subgroup:
        st.subheader(f"Forecasted Trends for {sector} {state} {subgroup} index")
    else:
        st.subheader(f"Forecasted Trends for {sector} {state} {group} index")

    st.line_chart(df_chart.pivot(index="Date", columns="Type", values="Value"))
    # st.write(type(future_predictions))

line_chart()

multi_chart()
st.divider()


colc1,colc2 = st.columns([1.8,3], vertical_alignment="bottom")
popover = colc2.popover("select index to forcast")
selected_state = popover.selectbox("Select States", df['state'].unique())

# **Sector Selection**
selected_sector = popover.selectbox("Select Sectors", df['sector'].unique())

# **Group Selection**
selected_group = popover.selectbox("Select Groups", df['group'].unique())

available_subgroups = set()

available_subgroups.update(group_to_subgroup.get(selected_group, []))

# Only show subgroups if they exist
if available_subgroups:
    selected_subgroup = popover.selectbox(
        "Select Subgroup",
        list(available_subgroups)
    )
else:
    selected_subgroup = None  # No selection for groups without subgroups

if popover.button("Run Forecast"):
    predictions(selected_state,selected_sector,selected_group,selected_subgroup)












