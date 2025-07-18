# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO # Required for in-memory file handling

# Import your core simulation functions from the other file
# Assuming your provided code is saved as simulation_core.py
# If you prefer to keep it all in one file, just copy-paste the functions here.
# For this example, I'll assume it's in a separate file for modularity.
# Note: The 'Copyright' line should be at the top of your original simulation_core.py
# and isn't typically repeated in the importing file unless specifically intended.

# --- Core Simulation Logic (Copied from your provided code for a self-contained app) ---
# Copyright (c) 2025 Mahdi Ramadhan
# All rights reserved.

def triangular_dist(low, mode, high, size=1):
    """
    Generates random numbers from a Triangular Distribution using the inverse CDF formula.
    """
    u = np.random.rand(size)
    c = (mode - low) / (high - low)
    
    x = np.where(u < c, 
                 low + np.sqrt(u * (high - low) * (mode - low)),
                 high - np.sqrt((1 - u) * (high - low) * (high - mode)))
    return x

def run_monte_carlo_simulation(phases_data, num_sims):
    """
    Runs the Monte Carlo simulation for project durations.
    """
    all_simulated_durations = []
    raw_sim_data = []

    for i in range(num_sims):
        current_project_duration = 0
        current_sim_row = {'Sim_ID': i + 1}

        for phase_name, data in phases_data.items():
            min_dur, mode_dur, max_dur = data
            sim_dur = triangular_dist(min_dur, mode_dur, max_dur, size=1)[0]
            current_project_duration += sim_dur
            current_sim_row[f'Duration {phase_name}'] = sim_dur
        
        all_simulated_durations.append(current_project_duration)
        current_sim_row['Total Project Duration'] = current_project_duration
        raw_sim_data.append(current_sim_row)

    return np.array(all_simulated_durations), pd.DataFrame(raw_sim_data)

def analyze_simulation_results(simulated_durations, target_dur, confidence_lvl):
    """
    Analyzes the project duration simulation results and returns a summary DataFrame.
    """
    # Basic Statistical Summary
    summary_data = {
        "Metric": ["Average", "Median", "Standard Deviation", "Min", "Max"],
        "Total Project Duration": [
            np.mean(simulated_durations),
            np.median(simulated_durations),
            np.std(simulated_durations),
            np.min(simulated_durations),
            np.max(simulated_durations)
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    # Duration Percentiles
    percentile_data = {"Percentile": [], "Total Project Duration": []}
    for p in [10, 20, 50, 80, 90]:
        percentile_val = np.percentile(simulated_durations, p)
        percentile_data["Percentile"].append(f"{p}%")
        percentile_data["Total Project Duration"].append(percentile_val)
    df_percentile = pd.DataFrame(percentile_data)

    # Probability of Completion within Target Duration
    prob_within_target = np.sum(simulated_durations <= target_dur) / len(simulated_durations)

    # Contingency Requirement Analysis
    duration_at_confidence = np.percentile(simulated_durations, confidence_lvl * 100)

    # Combine all summary results into a single DataFrame for display
    df_final_summary = pd.DataFrame(
        {
            "Metric": df_summary["Metric"].tolist() + df_percentile["Percentile"].tolist() + ["Prob. Complete <= Target Duration", f"Duration Needed ({confidence_lvl*100:.0f}% Confidence)"],
            "Total Project Duration": df_summary["Total Project Duration"].tolist() + df_percentile["Total Project Duration"].tolist() + [prob_within_target, duration_at_confidence]
        }
    )
    return df_final_summary, prob_within_target, duration_at_confidence

def create_histogram_plot(simulated_durations, mean_duration, target_dur):
    """
    Creates a Matplotlib histogram figure for project duration.
    Returns the figure object, so Streamlit can display it.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(simulated_durations, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Average: {mean_duration:.2f} units')
    ax.axvline(target_dur, color='green', linestyle='dashed', linewidth=1, label=f'Initial Target: {target_dur} units')
    ax.set_title('Probability Distribution of Project Duration')
    ax.set_xlabel('Project Duration (units)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    return fig

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="Probability Timeline Management with Monte Carlo Simulation")

st.title("Probability Timeline Management with Monte Carlo Simulation")
st.header("by Mahdi Ramadhan")

st.write("Upload an Excel file with your project phase durations (Min, Mode, Max).")
st.write("Example Excel format (sheet name can be anything):")
data = {
    "Project Task": ["Initiation", "Planning", "Execution"],
    "Min Duration": [8, 8, 7],
    "Mode Duration": [10, 9, 8],
    "Max Duration": [15, 10, 10]
}

# Buat DataFrame
df = pd.DataFrame(data)

# Tampilkan sebagai Streamlit DataFrame
st.dataframe(df,hide_index=True)

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type="xlsx")

    st.header("2. Simulation Settings")
    num_simulations_st = st.number_input(
        "Number of Simulations:",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Higher numbers increase accuracy but take longer."
    )
    target_duration_st = st.number_input(
        "Total Target Project Duration:",
        min_value=1,
        step=1,
        help="Your initial target completion time for the project."
    )
    confidence_level_st = st.slider(
        "Confidence Level for Contingency:",
        min_value=0.01,
        max_value=0.99,
        value=0.80,
        step=0.01,
        format="%.2f%%",
        help="The probability you want to achieve for project completion (e.g., 80% confident)."
    )

    run_simulation_button = st.button("Run Simulation")

st.markdown("---")

# --- Main Content Area for Results ---
if uploaded_file is not None:
    try:
        # Read the Excel file
        df_input = pd.read_excel(uploaded_file)
        
        # Validate columns
        required_cols = ['Project Task', 'Min Duration', 'Mode Duration', 'Max Duration']
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"Error: Excel file must contain columns: {', '.join(required_cols)}")
            st.stop() # Stop execution if columns are missing

        project_phases_data_st = {}
        for index, row in df_input.iterrows():
            phase_name = row['Project Task']
            # Ensure values are numeric and handle potential errors
            try:
                min_dur = float(row['Min Duration'])
                mode_dur = float(row['Mode Duration'])
                max_dur = float(row['Max Duration'])
                if not (min_dur <= mode_dur <= max_dur) or min_dur < 0:
                     st.error(f"Error in phase '{phase_name}': Durations must be non-negative and Min <= Mode <= Max.")
                     st.stop()
                project_phases_data_st[phase_name] = [min_dur, mode_dur, max_dur]
            except ValueError:
                st.error(f"Error in phase '{phase_name}': Durations must be valid numbers.")
                st.stop()

        st.subheader("Uploaded Project Phases:")
        st.dataframe(df_input)

        if run_simulation_button:
            st.info("Running simulation... Please wait.")
            
            # 1. Run the Simulation
            simulated_durations_array, df_raw_simulation_data = run_monte_carlo_simulation(
                project_phases_data_st, num_simulations_st
            )

            # 2. Analyze Simulation Results
            df_final_summary_results, prob_within_target_duration, duration_at_confidence_level = \
                analyze_simulation_results(
                    simulated_durations_array, target_duration_st, confidence_level_st
                )

            st.success("Simulation Complete!")

            st.subheader("Simulation Results Summary")
            st.dataframe(df_final_summary_results.set_index("Metric"))

            # --- Download Buttons for Tables ---
            st.markdown("---")
            st.subheader("Download Results")
            
            # Download Summary
            # Convert DataFrame to Excel in memory
            excel_buffer_summary = BytesIO()
            with pd.ExcelWriter(excel_buffer_summary, engine='openpyxl') as writer:
                df_final_summary_results.to_excel(writer, sheet_name="Summary", index=False)
            excel_buffer_summary.seek(0) # Rewind the buffer to the beginning
            st.download_button(
                label="Download Summary (.xlsx)",
                data=excel_buffer_summary,
                file_name="Project_Summary_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Download Raw Simulation Data
            # Convert DataFrame to Excel in memory
            excel_buffer_raw = BytesIO()
            with pd.ExcelWriter(excel_buffer_raw, engine='openpyxl') as writer:
                df_raw_simulation_data.to_excel(writer, sheet_name="Raw_Data", index=False)
            excel_buffer_raw.seek(0)
            st.download_button(
                label="Download Raw Simulation Data (.xlsx)",
                data=excel_buffer_raw,
                file_name="Project_Raw_Simulation_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("---")

            # 3. Visualize Results
            mean_sim_duration = np.mean(simulated_durations_array)
            fig_histogram = create_histogram_plot(simulated_durations_array, mean_sim_duration, target_duration_st)
            st.subheader("Project Duration Probability Distribution")
            st.pyplot(fig_histogram) # Display the Matplotlib figure in Streamlit

            st.markdown("---")
            st.info("Copyright (c) 2025 Mahdi Ramadhan. All rights reserved.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure your Excel file is correctly formatted with the specified columns.")

else:
    st.write("Awaiting Excel file upload...")