import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Chess Cheater Prediction')

players = pd.read_csv("data/players.csv")


def interpolate(s):
    s = s.reindex(range(1, s.index.max() + 1))
    s_interpolated = s.interpolate(method='linear')
    return s_interpolated.values[20:55]


def get_data(column='move_accuracy', groupby='move'):
    data = df_moves.groupby(['username', groupby])[column].mean()
    count = df_moves.groupby(['username', groupby])[column].count()
    data = data[count >= 10]
    return data


selected_player_name = st.selectbox('Enter Player name', players['username'].values)

if st.button('Predict'):
    val = 0
    st.write(selected_player_name)
    df_moves = pd.read_csv("data/moves.csv")
    df_moves['use'] = (((df_moves['game_result'] > 0) & (~df_moves['did_flag']) | df_moves['was_flagged']) & (
                df_moves['move'] <= 70))
    df_moves = df_moves[df_moves['use']].copy().reset_index(drop=True)

    # Top 1 Move
    st.title('Top 1 Move')
    data = get_data(column='pv_rank_top1', groupby='move')
    diff = interpolate(data['MagnusCarlsen'].rolling(window=15, center=True).mean()) - interpolate(
        data[selected_player_name].rolling(window=15, center=True).mean())
    st.write(diff[diff < 0].sum())
    if diff[diff < 0].sum() < 0:
        val += 25
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data['MagnusCarlsen'].rolling(window=15, center=True).mean(), label='MagnusCarlsen', lw=4, ax=ax)
    sns.lineplot(data=data[selected_player_name].rolling(window=15, center=True).mean(),
                 label=f'{selected_player_name}', ax=ax)
    ax.set_xlabel('Move')
    ax.set_ylabel('Move Rank')
    ax.set_title('PV Rank (Top-1) Over Time')
    st.pyplot(fig)

    # Move Accuracy
    st.title('Move Accuracy')
    data = get_data(column='move_accuracy', groupby='move')
    diff = interpolate(data['MagnusCarlsen'].rolling(window=15, center=True).mean()) - interpolate(
        data[selected_player_name].rolling(window=15, center=True).mean())
    st.write(diff[diff < 0].sum())
    if diff[diff < 0].sum() < -0.1:
        val += 25
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data['MagnusCarlsen'].rolling(window=15, center=True).mean(), label='MagnusCarlsen', lw=4, ax=ax)
    sns.lineplot(data=data[selected_player_name].rolling(window=15, center=True).mean(),
                 label=f'{selected_player_name}', ax=ax)
    ax.set_xlabel('Move')
    ax.set_ylabel('Accuracy %')
    ax.set_title('Move Accuracy Over Time')
    st.pyplot(fig)

    # Blunder Ratio
    st.title('Blunder Ratio')
    data = get_data(column='is_blunder', groupby='move')
    diff = interpolate(data['MagnusCarlsen'].rolling(window=15, center=True).mean()) - interpolate(
        data[selected_player_name].rolling(window=15, center=True).mean())
    st.write(diff[diff < 0].sum())
    if diff[diff < 0].sum() > 0.007:
        val += 25
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data['MagnusCarlsen'].rolling(window=15, center=True).mean(), label='MagnusCarlsen', lw=4, ax=ax)
    sns.lineplot(data=data[selected_player_name].rolling(window=15, center=True).mean(),
                 label=f'{selected_player_name}', ax=ax)
    ax.set_xlabel('Move')
    ax.set_ylabel('Blunder Ratio %')
    ax.set_title('Blunder Ratio Over Time')
    st.pyplot(fig)

    # Top 3 Move
    st.title('Top 3 Move')
    data = get_data(column='is_top3', groupby='move')
    diff = interpolate(data['MagnusCarlsen'].rolling(window=15, center=True).mean()) - interpolate(
        data[selected_player_name].rolling(window=15, center=True).mean())
    st.write(diff[diff < 0].sum())
    if diff[diff < 0].sum() < 0:
        val += 25
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data['MagnusCarlsen'].rolling(window=15, center=True).mean(), label='MagnusCarlsen', lw=4, ax=ax)
    sns.lineplot(data=data[selected_player_name].rolling(window=15, center=True).mean(),
                 label=f'{selected_player_name}', ax=ax)
    ax.set_xlabel('Move')
    ax.set_ylabel('Top 3 Move Ratio %')
    ax.set_title('Top 3 Move Ratio Over Time')
    st.pyplot(fig)

    st.title(f'Cheater Percentage = {val} %')
