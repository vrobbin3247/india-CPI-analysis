import streamlit as st
import pandas as pd
import numpy as np

# selected_state = None
# selected_sector = None
# selected_group = None
# selected_subgroup = None
@st.cache_data
def load_data():
    df = pd.read_csv('cpi_data_mospi_base12.csv')
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

df = load_data()

# def set_data(state, sector, group, subgroup):
#     global selected_state, selected_sector, selected_group, selected_subgroup  # Make them global
#     selected_state = state
#     selected_sector = sector
#     selected_group = group
#     selected_subgroup = subgroup
#
#     st.write(f"Forecasting for: {selected_state}, {selected_sector}, {selected_group}, {selected_subgroup}")
#     index_data = get_index_data()
#     train_model(index_data)

def get_index_data(state, sector, group, subgroup=None):
    df = load_data()
    filtered_df = df[
        (df['state'] == state) &
        (df['sector'] == sector) &
        (df['group'] == group)
    ]

    if subgroup:
        filtered_df = filtered_df[filtered_df['subgroup'] == subgroup]

    if not filtered_df.empty:
        return filtered_df['index'].to_numpy()  # Convert to NumPy array
    else:
        return np.array([])  # Return an empty array if no data

# index = get_index_data()
# st.write(index)
# model, predictor = train_model(index_data)




