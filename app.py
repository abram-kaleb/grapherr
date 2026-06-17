import copy
import os  # Tambahkan library os untuk memisahkan nama file dan ekstensi
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    st.error(
        "Library 'plotly' is not installed. Please run 'pip install plotly' in your terminal."
    )
    st.stop()

# Set page configuration to wide mode
st.set_page_config(layout="wide")

st.title("Engine Data Monitoring and Summary Dashboard")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Second"] = df.index
    return df


def extract_stable_segments(df, column, window=100, threshold=5):
    """
    Function to detect stable horizontal (steady-state) data segments.
    """
    # Calculate rolling standard deviation
    rolling_std = df[column].rolling(window=window, center=True).std()

    # Filter data with low deviation
    stable_data = df[rolling_std < threshold].copy()

    if stable_data.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "Start Second",
                "End Second",
                f"Average {column}",
                "Duration (Seconds)",
            ]
        )

    # Group continuous stable data into distinct segments
    stable_data["Segment_ID"] = (stable_data["Second"].diff() > window).cumsum()

    # Aggregate data per segment
    summary = (
        stable_data.groupby("Segment_ID")
        .agg(
            Start_Second=("Second", "min"),
            End_Second=("Second", "max"),
            Average=(column, "mean"),
            Duration=("Second", "count"),
        )
        .reset_index(drop=True)
    )

    # Filter out segments that do not meet the minimum duration
    summary = summary[summary["Duration"] > window].reset_index(drop=True)
    summary["Average"] = summary["Average"].round(2)

    # Add index ID column
    summary.insert(0, "ID", summary.index + 1)
    summary.columns = [
        "ID",
        "Start Second",
        "End Second",
        f"Average {column}",
        "Duration (Seconds)",
    ]
    return summary


uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    all_columns = [col for col in df.columns if col != "Second"]

    # --- COLUMN CONFIGURATION ---
    st.write("### Column Settings")
    col_conf1, col_conf2 = st.columns(2)

    with col_conf1:
        primary_column = st.selectbox("Primary Column (Y-Axis):", options=all_columns)
    with col_conf2:
        comparison_columns = st.multiselect(
            "Comparison Columns:",
            options=[c for c in all_columns if c != primary_column],
        )

    selected_columns = [primary_column] + comparison_columns

    st.write("---")

    # --- FILTER SENSITIVITY CONFIGURATION (MAIN PAGE) ---
    st.write("### Filter Sensitivity Settings")
    col_sens1, col_sens2 = st.columns(2)

    with col_sens1:
        noise_tolerance = st.slider(
            "Noise Tolerance (Std Dev):",
            min_value=1.0,
            max_value=20.0,
            value=3.5,
            step=0.5,
        )
    with col_sens2:
        min_segment_length = st.slider(
            "Minimum Segment Length (Seconds):",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )

    st.write("---")

    # Extract summary table using the stable segments function
    df_summary = extract_stable_segments(
        df, primary_column, window=min_segment_length, threshold=noise_tolerance
    )

    # --- BASE CHART GENERATION (WITH CACHING) ---
    @st.cache_resource
    def create_base_chart(data, columns):
        fig_base = px.line(data, x="Second", y=columns, render_mode="webgl")
        fig_base.update_layout(
            hovermode="x unified",
            template="plotly_white",
            dragmode="pan",
            margin=dict(l=10, r=10, t=10, b=10),
            height=500,
        )
        return fig_base

    # --- INTERACTIVE DASHBOARD FRAGMENT ---
    @st.fragment
    def render_interactive_dashboard(df_source, df_sum, columns_to_plot):
        st.write("### Steady-State Summary Table (Click a row to highlight the chart)")

        selected_row = None

        if not df_sum.empty:
            # Interactive row selection table
            event = st.dataframe(
                df_sum,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            # Capture row selection event
            selected_rows_indices = event.get("selection", {}).get("rows", [])
            if selected_rows_indices:
                selected_idx = selected_rows_indices[0]
                selected_row = df_sum.iloc[selected_idx]

                start_sec = int(selected_row["Start Second"])
                end_sec = int(selected_row["End Second"])

                st.success(
                    f"Showing markers for range: Second {start_sec} to {end_sec}"
                )
        else:
            st.warning(
                "No stable data segments found with the current filter settings."
            )

        # --- CHART VISUALIZATION ---
        st.write("### Monitoring Chart")

        # Retrieve base chart from resource cache
        fig = create_base_chart(df_source, columns_to_plot)

        # Create a deep copy to prevent persisting dynamically added highlights
        fig_render = copy.deepcopy(fig)

        # Apply dynamic visual highlights if a row is selected
        if selected_row is not None:
            start_sec = selected_row["Start Second"]
            end_sec = selected_row["End Second"]

            fig_render.add_vline(
                x=start_sec,
                line_width=2,
                line_dash="dash",
                line_color="green",
                annotation_text="Start",
                annotation_position="top left",
            )
            fig_render.add_vline(
                x=end_sec,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text="End",
                annotation_position="top right",
            )
            fig_render.add_vrect(
                x0=start_sec,
                x1=end_sec,
                fillcolor="rgba(0, 255, 0, 0.1)",
                opacity=0.3,
                layer="below",
                line_width=0,
            )

        st.plotly_chart(fig_render, use_container_width=True, key="monitoring_chart")

    # Execute the dashboard fragment
    render_interactive_dashboard(df, df_summary, selected_columns)

    # --- MODIFIKASI FITUR DOWNLOAD CSV (FORMAT SAMA DENGAN MENTAH & NAMA DINAMIS) ---
    if not df_summary.empty:
        # 1. Definisikan batas batas segmen statis berdasarkan df_summary
        intervals = []
        for _, row in df_summary.iterrows():
            intervals.append(
                (int(row["Start Second"]), int(row["End Second"]), "stable")
            )

        # Urutkan interval
        intervals = sorted(intervals, key=lambda x: x[0])

        # 2. Gabungkan dengan bagian non-statis/transisi agar semua rentang waktu ter-cover secara berurutan
        full_segments = []
        current_idx = 0
        max_idx = df["Second"].max()

        for start, end, label in intervals:
            if start > current_idx:
                # Berarti ada gap kosong (transisi/tidak statis), masukkan rentang ini
                full_segments.append((current_idx, start - 1, "transition"))
            full_segments.append((start, end, "stable"))
            current_idx = end + 1

        if current_idx <= max_idx:
            full_segments.append((current_idx, max_idx, "transition"))

        # 3. Lakukan agregasi/rata-rata untuk setiap potongan segmen yang didapat
        summarized_rows = []
        # Mengambil list kolom file asli (tanpa kolom tambahan 'Second' jika aslinya tidak ada)
        original_columns = [c for c in df.columns if c != "Second"]

        for start, end, label in full_segments:
            # Filter potongan data berdasarkan detik start s/d end
            segment_data = df[(df["Second"] >= start) & (df["Second"] <= end)]
            if not segment_data.empty:
                # Hitung rata-rata untuk semua kolom original di segmen ini
                mean_values = segment_data[original_columns].mean().round(2)
                summarized_rows.append(mean_values)

        # Buat dataframe baru dengan format kolom persis file mentah asli
        df_condensed = pd.DataFrame(summarized_rows).reset_index(drop=True)

        # 4. MEMBUAT NAMA FILE DINAMIS SERSUAI FILE MENTAH ASLI
        # Memisahkan nama file dan ekstensi (contoh: "data_mesin.csv" -> "data_mesin", ".csv")
        file_base_name, file_extension = os.path.splitext(uploaded_file.name)
        # Menambahkan akhiran _filtered menjadi "data_mesin_filtered.csv"
        custom_file_name = f"{file_base_name}_filtered{file_extension}"

        # Sediakan tombol download dengan data yang sudah berkurang drastis barisnya
        csv_data = df_condensed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name=custom_file_name,  # Menggunakan nama dinamis di sini
            mime="text/csv",
        )
else:
    st.info(
        "Please upload a CSV file above to generate the monitoring charts and summary tables."
    )
