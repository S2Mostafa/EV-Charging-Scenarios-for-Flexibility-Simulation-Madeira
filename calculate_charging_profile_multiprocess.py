import os
import gc
import pandas as pd
import multiprocessing as mp
from datetime import date, datetime
import numpy as np
import json
import glob


def sample_brand(df):
    return np.random.choice(
        df["Brand"],
        p=df["adjusted_Percentage"] / df["adjusted_Percentage"].sum()
    )


def get_day_category(input_date):
    portugal_holidays = {
        date(2025, 1, 1),
        date(2025, 3, 4),
        date(2025, 4, 2),
        date(2025, 4, 18),
        date(2025, 4, 20),
        date(2025, 4, 25),
        date(2025, 5, 1),
        date(2025, 6, 10),
        date(2025, 6, 19),
        date(2025, 7, 1),
        date(2025, 8, 15),
        date(2025, 8, 21),
        date(2025, 10, 5),
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2025, 12, 8),
        date(2025, 12, 25),
        date(2025, 12, 26),
    }
    if input_date in portugal_holidays:
        return "Holiday"
    elif input_date.weekday() >= 5:
        return "Weekend"
    else:
        return "Weekday"


def load_ev_data():
    # Load EV models data from JSON file
    with open("ev_models_for_simulation.json", "r") as f:
        ev_models_data = json.load(f)
    
    # Load charging session data from CSV file
    df = pd.read_csv("charging_sessions_2025-03-30_to_2025-06-29.csv")

    # Clean the dataframe
    # Remove rows with NaN in any of the specified columns
    df_clear = df.dropna(subset=['max_electric_power', 'start_time', 'duration_minutes'])

    # Extract brands and percentages from JSON data
    Brand = list(ev_models_data.keys())
    adjusted_percentages = [ev_models_data[brand]["market_percentage"] for brand in Brand]
    
    df_old_total_percentages = pd.DataFrame({
        "Brand": Brand,
        "adjusted_Percentage": adjusted_percentages
    })

    # Prepare data containers
    SOC = {}
    GENERAL = {}

    # Process each brand from JSON data
    for brand in Brand:
        brand_data = ev_models_data[brand]
        
        # Create SOC dataframe from charging profile
        charging_profile = brand_data["charging_profile"]
        soc_values = list(range(0, 101))  # 0 to 100%
        speed_values = [charging_profile[str(soc)] for soc in soc_values]
        
        SOC[brand] = pd.DataFrame({
            'Time': np.full(101, np.nan),
            'SOC (%)': soc_values,
            'Speed (kW)': speed_values,
            'Energy Charged (kWh)': np.full(101, np.nan)
        })
        
        # Create GENERAL dataframe
        GENERAL[brand] = pd.DataFrame([{
            "model": brand_data["model_name"],
            "ev_type": brand_data["ev_type"],
            "battery_sizes": brand_data["battery_size_kwh"],
            "onboard AC charger KW": brand_data["onboard_ac_charger_kw"]
        }])

    return SOC, GENERAL, df_old_total_percentages, df_clear





def generate_soc_start_from_probability(SOC_start_probability_distribution):
    """Generate SOC_start based on probability distribution"""
    soc_values = list(SOC_start_probability_distribution.keys())
    probabilities = list(SOC_start_probability_distribution.values())
    
    # Normalize probabilities to ensure they sum to 1
    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]
    
    return np.random.choice(soc_values, p=normalized_probs)


def generate_soc_end_from_probability(SOC_end_probability_distribution):
    """Generate SOC_end based on probability distribution"""
    soc_values = list(SOC_end_probability_distribution.keys())
    probabilities = list(SOC_end_probability_distribution.values())
    
    # Normalize probabilities to ensure they sum to 1
    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]
    
    return np.random.choice(soc_values, p=normalized_probs)


def generate_charging_amount_from_probability(SOC_charging_probability_distribution):
    """Generate charging amount based on probability distribution"""
    charging_values = list(SOC_charging_probability_distribution.keys())
    probabilities = list(SOC_charging_probability_distribution.values())
    
    # Normalize probabilities to ensure they sum to 1
    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]
    
    return np.random.choice(charging_values, p=normalized_probs)


def process_SOC_pair_mode1(args):
    """Mode 1: Fixed SOC range for all sessions"""
    SOC_arrive, SOC_leaving, df_clear, SOC, GENERAL, df_old_total_percentages = args
    return process_SOC_pair_with_soc_range(SOC_arrive, SOC_leaving, df_clear, SOC, GENERAL, df_old_total_percentages)


def process_SOC_pair_mode2(args):
    """Mode 2: Variable SOC_start based on probability distribution"""
    df_clear, SOC, GENERAL, df_old_total_percentages, SOC_start_probability_distribution = args
    
    # Create a copy of df_clear for this simulation
    df_clear_copy = df_clear.copy()
    
    # Generate SOC_start for each session based on probability distribution
    soc_starts = []
    for _ in range(len(df_clear_copy)):
        soc_start = generate_soc_start_from_probability(SOC_start_probability_distribution)
        soc_starts.append(soc_start)
    
    # Add SOC_start column to dataframe
    df_clear_copy['SOC_start'] = soc_starts
    
    # Process each session with its assigned SOC_start, targeting 100% SOC
    return process_SOC_pair_with_variable_soc(df_clear_copy, 100, SOC, GENERAL, df_old_total_percentages)


def process_SOC_pair_mode3(args):
    """Mode 3: Variable SOC_start and charging amount based on probability distributions"""
    df_clear, SOC, GENERAL, df_old_total_percentages, SOC_start_probability_distribution, SOC_charging_probability_distribution = args
    
    # Create a copy of df_clear for this simulation
    df_clear_copy = df_clear.copy()
    
    # Generate SOC_start and charging amount for each session based on probability distributions
    soc_starts = []
    soc_ends = []
    for _ in range(len(df_clear_copy)):
        soc_start = generate_soc_start_from_probability(SOC_start_probability_distribution)
        charging_amount = generate_charging_amount_from_probability(SOC_charging_probability_distribution)
        
        # Calculate SOC_end by adding charging amount to SOC_start, capped at 100%
        soc_end = min(100, soc_start + charging_amount)
        
        soc_starts.append(soc_start)
        soc_ends.append(soc_end)
    
    # Add SOC_start and SOC_end columns to dataframe
    df_clear_copy['SOC_start'] = soc_starts
    df_clear_copy['SOC_end'] = soc_ends
    
    # Process each session with its assigned SOC_start and SOC_end
    return process_SOC_pair_with_variable_soc_start_end(df_clear_copy, SOC, GENERAL, df_old_total_percentages)


def process_SOC_pair_with_soc_range(SOC_arrive, SOC_leaving, df_clear, SOC, GENERAL, df_old_total_percentages):
    """Mode 1 processing function with fixed SOC range"""
    charging_efficiency = 0.90

    df_clear2 = df_clear.copy()
    df_clear2["start_time"] = pd.to_datetime(df_clear2["start_time"], errors='coerce')
    df_clear2["end_time"] = pd.to_datetime(df_clear2["end_time"], errors='coerce')

    power_delivered_list, charging_time_list = [], []
    session_id, ev_model_list = [], []
    start_time_day, end_time_day = [], []

    for idx, row in df_clear2.iterrows():
        ev_charging_time = 0
        ev_power_delivered = 0
        ev_occupy_time = row['duration_minutes']

        start_time = row['start_time'].replace(second=0, microsecond=0).tz_localize(None)
        end_time = row['end_time'].replace(second=0, microsecond=0).tz_localize(None)
        occupency_time_start = start_time

        full_time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        df_charging = pd.DataFrame(index=full_time_range)
        df_charging['charging_power'] = 0

        ev_model = sample_brand(df_old_total_percentages)

        for per in range(SOC_arrive, SOC_leaving):
            if ev_model not in SOC or ev_model not in GENERAL:
                continue

            general_df = GENERAL[ev_model]
            match_series = SOC[ev_model].loc[SOC[ev_model]["SOC (%)"] == per, "Speed (kW)"]
            if match_series.empty:
                continue

            power_charging = min(
                row['max_electric_power'] / 1000,
                float(general_df["onboard AC charger KW"].values[0]),
                float(match_series.iloc[0])
            ) * charging_efficiency

            battery_size = float(general_df["battery_sizes"].values[0])
            desired_power = battery_size / 100
            time_to_charge = (desired_power / power_charging) * 60

            actual_time = min(time_to_charge, ev_occupy_time - ev_charging_time)
            if actual_time <= 0:
                break

            ev_power_delivered += power_charging * (actual_time / 60)
            ev_charging_time += actual_time

            start_time_temp = occupency_time_start
            end_time_temp = occupency_time_start + pd.to_timedelta(actual_time, unit='m')

            minute_range = pd.date_range(start=start_time_temp.ceil('min'), end=end_time_temp.ceil('min'), freq='1min')
            for ts in minute_range:
                df_charging.loc[ts, 'charging_power'] = power_charging

            occupency_time_start = end_time_temp

            if ev_charging_time >= ev_occupy_time:
                break

        session_id.append(idx)
        ev_model_list.append(ev_model)
        power_delivered_list.append(ev_power_delivered)
        charging_time_list.append(ev_charging_time)
        start_time_day.append(get_day_category(start_time))
        end_time_day.append(get_day_category(end_time))

        folder_name = f"./data/outputs/SOC{SOC_arrive}to{SOC_leaving}"
        os.makedirs(folder_name, exist_ok=True)
        df_charging.to_csv(f"{folder_name}/instantaneous_power_sessionID_{idx}_SOC{SOC_arrive}to{SOC_leaving}.csv")

        del df_charging
        gc.collect()

    df_clear2["power_delivered"] = power_delivered_list
    df_clear2["charging_time"] = charging_time_list
    df_clear2["end_charging_time"] = df_clear2["start_time"] + pd.to_timedelta(df_clear2["charging_time"], unit='m')
    df_clear2["start_time_day"] = start_time_day
    df_clear2["end_time_day"] = end_time_day
    df_clear2["Flexibility"] = df_clear2["end_charging_time"].dt.floor("min") < df_clear2["end_time"].dt.floor("min")
    df_clear2["session_id"] = session_id
    df_clear2["Ev Model"] = ev_model_list

    os.makedirs("./data/outputs/charging_info", exist_ok=True)
    df_clear2.to_csv(f"{folder_name}/charging_info_SOC{SOC_arrive}to{SOC_leaving}.csv", index=False)
    df_clear2.to_csv(f"./data/outputs/charging_info/charging_info_SOC{SOC_arrive}to{SOC_leaving}.csv", index=False)

    del df_clear2
    gc.collect()


def process_SOC_pair_with_variable_soc(df_clear_with_soc, SOC_leaving, SOC, GENERAL, df_old_total_percentages):
    """Processing function with variable SOC_start for each session"""
    charging_efficiency = 0.90

    df_clear2 = df_clear_with_soc.copy()
    df_clear2["start_time"] = pd.to_datetime(df_clear2["start_time"], errors='coerce')
    df_clear2["end_time"] = pd.to_datetime(df_clear2["end_time"], errors='coerce')

    power_delivered_list, charging_time_list = [], []
    session_id, ev_model_list = [], []
    start_time_day, end_time_day = [], []
    soc_start_list = []

    for idx, row in df_clear2.iterrows():
        ev_charging_time = 0
        ev_power_delivered = 0
        ev_occupy_time = row['duration_minutes']
        SOC_arrive = int(row['SOC_start'])  # Get SOC_start for this session

        start_time = row['start_time'].replace(second=0, microsecond=0).tz_localize(None)
        end_time = row['end_time'].replace(second=0, microsecond=0).tz_localize(None)
        occupency_time_start = start_time

        full_time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        df_charging = pd.DataFrame(index=full_time_range)
        df_charging['charging_power'] = 0

        ev_model = sample_brand(df_old_total_percentages)

        for per in range(SOC_arrive, SOC_leaving):
            if ev_model not in SOC or ev_model not in GENERAL:
                continue

            general_df = GENERAL[ev_model]
            match_series = SOC[ev_model].loc[SOC[ev_model]["SOC (%)"] == per, "Speed (kW)"]
            if match_series.empty:
                continue

            power_charging = min(
                row['max_electric_power'] / 1000,
                float(general_df["onboard AC charger KW"].values[0]),
                float(match_series.iloc[0])
            ) * charging_efficiency

            battery_size = float(general_df["battery_sizes"].values[0])
            desired_power = battery_size / 100
            time_to_charge = (desired_power / power_charging) * 60

            actual_time = min(time_to_charge, ev_occupy_time - ev_charging_time)
            if actual_time <= 0:
                break

            ev_power_delivered += power_charging * (actual_time / 60)
            ev_charging_time += actual_time

            start_time_temp = occupency_time_start
            end_time_temp = occupency_time_start + pd.to_timedelta(actual_time, unit='m')

            minute_range = pd.date_range(start=start_time_temp.ceil('min'), end=end_time_temp.ceil('min'), freq='1min')
            for ts in minute_range:
                df_charging.loc[ts, 'charging_power'] = power_charging

            occupency_time_start = end_time_temp

            if ev_charging_time >= ev_occupy_time:
                break

        session_id.append(idx)
        ev_model_list.append(ev_model)
        power_delivered_list.append(ev_power_delivered)
        charging_time_list.append(ev_charging_time)
        start_time_day.append(get_day_category(start_time))
        end_time_day.append(get_day_category(end_time))
        soc_start_list.append(SOC_arrive)

        folder_name = f"./data/outputs/Mode2_SOC_distribution_to{SOC_leaving}"
        os.makedirs(folder_name, exist_ok=True)
        df_charging.to_csv(f"{folder_name}/instantaneous_power_sessionID_{idx}_SOC{SOC_arrive}to{SOC_leaving}.csv")

        del df_charging
        gc.collect()

    df_clear2["power_delivered"] = power_delivered_list
    df_clear2["charging_time"] = charging_time_list
    df_clear2["end_charging_time"] = df_clear2["start_time"] + pd.to_timedelta(df_clear2["charging_time"], unit='m')
    df_clear2["start_time_day"] = start_time_day
    df_clear2["end_time_day"] = end_time_day
    df_clear2["Flexibility"] = df_clear2["end_charging_time"].dt.floor("min") < df_clear2["end_time"].dt.floor("min")
    df_clear2["session_id"] = session_id
    df_clear2["Ev Model"] = ev_model_list
    df_clear2["SOC_start"] = soc_start_list

    os.makedirs("./data/outputs/charging_info", exist_ok=True)
    df_clear2.to_csv(f"{folder_name}/charging_info_VariableSOC_to{SOC_leaving}.csv", index=False)
    df_clear2.to_csv(f"./data/outputs/charging_info/charging_info_VariableSOC_to{SOC_leaving}.csv", index=False)

    del df_clear2
    gc.collect()


def process_SOC_pair_with_variable_soc_start_end(df_clear_with_soc, SOC, GENERAL, df_old_total_percentages):
    """Processing function with variable SOC_start and SOC_end for each session"""
    charging_efficiency = 0.90

    df_clear2 = df_clear_with_soc.copy()
    df_clear2["start_time"] = pd.to_datetime(df_clear2["start_time"], errors='coerce')
    df_clear2["end_time"] = pd.to_datetime(df_clear2["end_time"], errors='coerce')

    power_delivered_list, charging_time_list = [], []
    session_id, ev_model_list = [], []
    start_time_day, end_time_day = [], []
    soc_start_list = []
    soc_end_list = []

    for idx, row in df_clear2.iterrows():
        ev_charging_time = 0
        ev_power_delivered = 0
        ev_occupy_time = row['duration_minutes']
        SOC_arrive = int(row['SOC_start'])  # Get SOC_start for this session
        SOC_leaving = int(row['SOC_end'])  # Get SOC_end for this session

        start_time = row['start_time'].replace(second=0, microsecond=0).tz_localize(None)
        end_time = row['end_time'].replace(second=0, microsecond=0).tz_localize(None)
        occupency_time_start = start_time

        full_time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        df_charging = pd.DataFrame(index=full_time_range)
        df_charging['charging_power'] = 0

        ev_model = sample_brand(df_old_total_percentages)

        for per in range(SOC_arrive, SOC_leaving):
            if ev_model not in SOC or ev_model not in GENERAL:
                continue

            general_df = GENERAL[ev_model]
            match_series = SOC[ev_model].loc[SOC[ev_model]["SOC (%)"] == per, "Speed (kW)"]
            if match_series.empty:
                continue

            power_charging = min(
                row['max_electric_power'] / 1000,
                float(general_df["onboard AC charger KW"].values[0]),
                float(match_series.iloc[0])
            ) * charging_efficiency

            battery_size = float(general_df["battery_sizes"].values[0])
            desired_power = battery_size / 100
            time_to_charge = (desired_power / power_charging) * 60

            actual_time = min(time_to_charge, ev_occupy_time - ev_charging_time)
            if actual_time <= 0:
                break

            ev_power_delivered += power_charging * (actual_time / 60)
            ev_charging_time += actual_time

            start_time_temp = occupency_time_start
            end_time_temp = occupency_time_start + pd.to_timedelta(actual_time, unit='m')

            minute_range = pd.date_range(start=start_time_temp.ceil('min'), end=end_time_temp.ceil('min'), freq='1min')
            for ts in minute_range:
                df_charging.loc[ts, 'charging_power'] = power_charging

            occupency_time_start = end_time_temp

            if ev_charging_time >= ev_occupy_time:
                break

        session_id.append(idx)
        ev_model_list.append(ev_model)
        power_delivered_list.append(ev_power_delivered)
        charging_time_list.append(ev_charging_time)
        start_time_day.append(get_day_category(start_time))
        end_time_day.append(get_day_category(end_time))
        soc_start_list.append(SOC_arrive)
        soc_end_list.append(SOC_leaving)

        folder_name = f"./data/outputs/Mode3_SOC_distribution_start_end"
        os.makedirs(folder_name, exist_ok=True)
        df_charging.to_csv(f"{folder_name}/instantaneous_power_sessionID_{idx}_SOC{SOC_arrive}to{SOC_leaving}.csv")

        del df_charging
        gc.collect()

    df_clear2["power_delivered"] = power_delivered_list
    df_clear2["charging_time"] = charging_time_list
    df_clear2["end_charging_time"] = df_clear2["start_time"] + pd.to_timedelta(df_clear2["charging_time"], unit='m')
    df_clear2["start_time_day"] = start_time_day
    df_clear2["end_time_day"] = end_time_day
    df_clear2["Flexibility"] = df_clear2["end_charging_time"].dt.floor("min") < df_clear2["end_time"].dt.floor("min")
    df_clear2["session_id"] = session_id
    df_clear2["Ev Model"] = ev_model_list
    df_clear2["SOC_start"] = soc_start_list
    df_clear2["SOC_end"] = soc_end_list

    os.makedirs("./data/outputs/charging_info", exist_ok=True)
    df_clear2.to_csv(f"{folder_name}/charging_info_VariableSOC_start_end.csv", index=False)
    df_clear2.to_csv(f"./data/outputs/charging_info/charging_info_VariableSOC_start_end.csv", index=False)

    del df_clear2
    gc.collect()


def calculate_charging_profile(df_clear, SOC, GENERAL, df_old_total_percentages, cpu_count=4, mode=1, SOC_start_probability_distribution=None, SOC_charging_probability_distribution=None):
    """
    Calculate charging profile with three modes:
    
    Mode 1: Fixed SOC ranges (Mode 1 behavior)
    Mode 2: Variable SOC_start based on probability distribution - ONE PASS through data
    Mode 3: Variable SOC_start and charging amount based on probability distributions - ONE PASS through data
    
    Args:
        mode: 1 for fixed SOC ranges, 2 for variable SOC_start, 3 for variable SOC_start and charging amount
        SOC_start_probability_distribution: dict for modes 2&3, e.g., {20: 0.2, 25: 0.3, 30: 0.3, 35: 0.2}
        SOC_charging_probability_distribution: dict for mode 3, e.g., {20: 0.3, 25: 0.3, 30: 0.2, 35: 0.1, 40: 0.1}
    """
    
    if mode == 1:
        # Mode 1: Fixed SOC ranges
        SOC_sim_start = 25
        SOC_sim_end = 100
        SOC_increment = 5

        soc_pairs = [
            (SOC_arrive, SOC_leaving, df_clear, SOC, GENERAL, df_old_total_percentages)
            for SOC_arrive in range(SOC_sim_start, SOC_sim_end, SOC_increment)
            for SOC_leaving in range(SOC_sim_start+SOC_increment, 50, SOC_increment)
            if SOC_leaving > SOC_arrive
        ]

        with mp.Pool(processes=cpu_count) as pool:
            pool.map(process_SOC_pair_mode1, soc_pairs)
            
    elif mode == 2:
        # Mode 2: Variable SOC_start based on probability distribution - ONE PASS
        if SOC_start_probability_distribution is None:
            raise ValueError("SOC_start_probability_distribution must be provided for mode 2")
        
        # For mode 2, we do ONE PASS through the data
        # Each session gets a random SOC_start, target is always 100%
        soc_pairs = [
            (df_clear, SOC, GENERAL, df_old_total_percentages, SOC_start_probability_distribution)
        ]

        with mp.Pool(processes=cpu_count) as pool:
            pool.map(process_SOC_pair_mode2, soc_pairs)
    
    elif mode == 3:
        # Mode 3: Variable SOC_start and charging amount based on probability distributions
        if SOC_start_probability_distribution is None or SOC_charging_probability_distribution is None:
            raise ValueError("SOC_start_probability_distribution and SOC_charging_probability_distribution must be provided for mode 3")
        
        # For mode 3, we do ONE PASS through the data
        # Each session gets a random SOC_start and charging amount
        soc_pairs = [
            (df_clear, SOC, GENERAL, df_old_total_percentages, SOC_start_probability_distribution, SOC_charging_probability_distribution)
        ]

        with mp.Pool(processes=cpu_count) as pool:
            pool.map(process_SOC_pair_mode3, soc_pairs)
    
    else:
        raise ValueError("mode must be 1, 2, or 3")


if __name__ == "__main__":
    print("=" * 60)
    print("EV Charging Scenarios for Flexibility Simulation - Madeira")
    print("=" * 60)
    
    # Ask user for simulation mode
    while True:
        print("\nSelect simulation mode:")
        print("1. Mode 1: Fixed SOC ranges")
        print("2. Mode 2: Variable SOC_start based on probability distribution")
        print("3. Mode 3: Variable SOC_start and charging amount based on probability distributions")
        
        try:
            SIMULATION_MODE = int(input("\nEnter mode (1, 2, or 3): "))
            if SIMULATION_MODE in [1, 2, 3]:
                break
            else:
                print("❌ Invalid mode. Please enter 1, 2, or 3.")
        except ValueError:
            print("❌ Invalid input. Please enter a number (1, 2, or 3).")
    
    # Ask for charging sessions file path
    print("\n" + "-" * 40)
    print("Charging Sessions File")
    print("-" * 40)
    print("Default: charging_sessions_2025-03-30_to_2025-06-29.csv")
    
    sessions_file_choice = input("Use default file? (y/n): ").lower().strip()
    
    if sessions_file_choice == 'y' or sessions_file_choice == '':
        sessions_file_path = "charging_sessions_2025-03-30_to_2025-06-29.csv"
        print(f"Using default file: {sessions_file_path}")
    else:
        sessions_file_path = input("Enter path to charging sessions CSV file: ").strip()
        print(f"Using custom file: {sessions_file_path}")
    
    # Ask for EV models file path
    print("\n" + "-" * 40)
    print("EV Models File")
    print("-" * 40)
    print("Default: ev_models_for_simulation.json")
    
    models_file_choice = input("Use default file? (y/n): ").lower().strip()
    
    if models_file_choice == 'y' or models_file_choice == '':
        models_file_path = "ev_models_for_simulation.json"
        print(f"Using default file: {models_file_path}")
    else:
        models_file_path = input("Enter path to EV models JSON file: ").strip()
        print(f"Using custom file: {models_file_path}")
    
    # Load data with custom file paths
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    try:
        # Load EV models data from JSON file
        with open(models_file_path, "r") as f:
            ev_models_data = json.load(f)
        
        # Load charging session data from CSV file
        df = pd.read_csv(sessions_file_path)
        
        # Clean the dataframe
        df_clear = df.dropna(subset=['max_electric_power', 'start_time', 'duration_minutes'])
        
        # Extract brands and percentages from JSON data
        Brand = list(ev_models_data.keys())
        adjusted_percentages = [ev_models_data[brand]["market_percentage"] for brand in Brand]
        
        df_old_total_percentages = pd.DataFrame({
            "Brand": Brand,
            "adjusted_Percentage": adjusted_percentages
        })

        # Prepare data containers
        SOC = {}
        GENERAL = {}

        # Process each brand from JSON data
        for brand in Brand:
            brand_data = ev_models_data[brand]
            
            # Create SOC dataframe from charging profile
            charging_profile = brand_data["charging_profile"]
            soc_values = list(range(0, 101))  # 0 to 100%
            speed_values = [charging_profile[str(soc)] for soc in soc_values]
            
            SOC[brand] = pd.DataFrame({
                'Time': np.full(101, np.nan),
                'SOC (%)': soc_values,
                'Speed (kW)': speed_values,
                'Energy Charged (kWh)': np.full(101, np.nan)
            })
            
            # Create GENERAL dataframe
            GENERAL[brand] = pd.DataFrame([{
                "model": brand_data["model_name"],
                "ev_type": brand_data["ev_type"],
                "battery_sizes": brand_data["battery_size_kwh"],
                "onboard AC charger KW": brand_data["onboard_ac_charger_kw"]
            }])
        
        print(f"✅ Successfully loaded {len(df_clear)} charging sessions")
        print(f"✅ Successfully loaded {len(Brand)} EV models")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check the file paths and try again.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please check your files and try again.")
        exit(1)
    
    # Ask for CPU count
    print("\n" + "-" * 40)
    print("CPU Configuration")
    print("-" * 40)
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    cpu_choice = input("Use default CPU count (8)? (y/n): ").lower().strip()
    
    if cpu_choice == 'y' or cpu_choice == '':
        cpu_count = 8
        print(f"✅ Using {cpu_count} CPU cores")
    else:
        try:
            cpu_count = int(input("Enter number of CPU cores to use: "))
            if cpu_count <= 0 or cpu_count > mp.cpu_count():
                print(f"⚠️  Invalid CPU count. Using {mp.cpu_count()} cores instead.")
                cpu_count = mp.cpu_count()
            else:
                print(f"✅ Using {cpu_count} CPU cores")
        except ValueError:
            print(f"⚠️  Invalid input. Using {mp.cpu_count()} cores instead.")
            cpu_count = mp.cpu_count()
    
    # Run simulation based on selected mode
    print("\n" + "=" * 60)
    print("Starting Simulation...")
    print("=" * 60)
    
    if SIMULATION_MODE == 1:
        # Mode 1: Fixed SOC ranges
        print("Running Mode 1: Fixed SOC ranges")
        calculate_charging_profile(df_clear, SOC, GENERAL, df_old_total_percentages, cpu_count=cpu_count, mode=1)
        
    elif SIMULATION_MODE == 2:
        # Mode 2: Variable SOC_start based on probability distribution
        print("Running Mode 2: Variable SOC based on probability distribution")
        
        # Ask for SOC_start probability distribution
        print("\n" + "-" * 40)
        print("SOC Start Probability Distribution")
        print("-" * 40)
        print("Default distribution:")

        
        dist_choice = input("Use default distribution? (y/n): ").lower().strip()
        
        if dist_choice == 'y' or dist_choice == '':
            SOC_start_probability_distribution = {
            20: 0.25,
            25: 0.20,
            30: 0.15,
            35: 0.12,
            40: 0.10,
            45: 0.07,
            50: 0.05,
            55: 0.03,
            60: 0.02,
            65: 0.01 }
            print("✅ Using default SOC_start probability distribution")
        else:
            print("\nEnter custom probability distribution:")
            print("Format: SOC_percentage probability (e.g., '20 0.2')")
            print("Enter 'done' when finished")
            print("Note: Probabilities should sum to 1.0")
            
            SOC_start_probability_distribution = {}
            while True:
                user_input = input("Enter SOC% and probability (or 'done' to finish): ").strip()
                if user_input.lower() == 'done':
                    break
                elif user_input.lower() in ['exit', 'quit', 'n', 'no']:
                    print("💡 To finish entering values, type 'done'")
                    continue
                
                # Check if input is empty
                if not user_input:
                    print("❌ Empty input. Please enter values or 'done' to finish.")
                    continue
                
                # Split input and validate format
                parts = user_input.split()
                if len(parts) != 2:
                    print("❌ Invalid format. Use 'SOC% probability' (e.g., '20 0.2')")
                    print("💡 Type 'done' to finish entering values")
                    continue
                
                try:
                    # Validate SOC% is an integer
                    soc_str = parts[0]
                    if not soc_str.isdigit():
                        print("❌ SOC% must be an integer (e.g., 20, 25, 30)")
                        continue
                    
                    soc = int(soc_str)
                    
                    # Validate probability is a float
                    prob_str = parts[1]
                    try:
                        prob = float(prob_str)
                    except ValueError:
                        print("❌ Probability must be a number (e.g., 0.2, 0.5, 1.0)")
                        continue
                    
                    # Validate ranges
                    if not (0 <= soc <= 100):
                        print("❌ SOC% must be between 0 and 100")
                        continue
                        
                    if not (0 <= prob <= 1):
                        print("❌ Probability must be between 0 and 1")
                        continue
                    
                    # Check if SOC already exists
                    if soc in SOC_start_probability_distribution:
                        print(f"⚠️  SOC {soc}% already exists. Replacing with new value.")
                    
                    SOC_start_probability_distribution[soc] = prob
                    current_sum = sum(SOC_start_probability_distribution.values())
                    print(f"✅ Added: {soc}% SOC with {prob} probability")
                    print(f"📊 Current sum: {current_sum:.3f}")
                    
                    if current_sum > 1.0:
                        print("⚠️  Warning: Sum exceeds 1.0!")
                        print("💡 You can continue adding or type 'done' to finish and normalize")
                    elif abs(current_sum - 1.0) < 0.001:
                        print("🎯 Perfect! Sum equals 1.0")
                        print("💡 You can enter 'done' to finish or continue adding more entries.")
                    elif current_sum < 1.0:
                        remaining = 1.0 - current_sum
                        print(f"📝 Remaining probability needed: {remaining:.3f}")
                    
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    print("❌ Use format: 'SOC% probability' (e.g., '20 0.2')")
                    print("💡 Type 'done' to finish entering values")
            
            if not SOC_start_probability_distribution:
                print("⚠️  No valid distribution entered. Using default.")
                SOC_start_probability_distribution = {20: 0.2, 25: 0.3, 30: 0.3, 35: 0.2}
        
        # Validate probability distribution sum
        total_probability = sum(SOC_start_probability_distribution.values())
        print(f"\n📊 Final probability distribution:")
        for soc, prob in sorted(SOC_start_probability_distribution.items()):
            print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
        print(f"📊 Total sum: {total_probability:.3f}")
        
        if abs(total_probability - 1.0) < 0.001:  # Allow small floating point errors
            print("✅ Probability distribution is valid (sum = 1.0)")
        elif total_probability > 1.0:
            print(f"⚠️  Warning: Probability sum is {total_probability:.3f}, which exceeds 1.0")
            
            while True:
                normalize_choice = input("Do you want to normalize the distribution? (y/n): ").lower().strip()
                if normalize_choice in ['y', 'yes']:
                    # Normalize the distribution
                    normalized_distribution = {}
                    for soc, prob in SOC_start_probability_distribution.items():
                        normalized_distribution[soc] = prob / total_probability
                    
                    SOC_start_probability_distribution = normalized_distribution
                    print("✅ Distribution has been normalized:")
                    for soc, prob in sorted(SOC_start_probability_distribution.items()):
                        print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
                    print(f"📊 Normalized sum: {sum(SOC_start_probability_distribution.values()):.3f}")
                    break
                    
                elif normalize_choice in ['n', 'no']:
                    print("📝 You can edit the distribution:")
                    print("Enter 'edit' to modify entries, or 'continue' to proceed with current values")
                    
                    edit_choice = input("Edit distribution? (edit/continue): ").lower().strip()
                    if edit_choice == 'edit':
                        # Allow editing the distribution
                        while True:
                            print(f"\n📊 Current distribution (sum: {sum(SOC_start_probability_distribution.values()):.3f}):")
                            for soc, prob in sorted(SOC_start_probability_distribution.items()):
                                print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
                            
                            print("\nOptions:")
                            print("1. Enter 'remove SOC%' to remove an entry (e.g., 'remove 20')")
                            print("2. Enter 'add SOC% probability' to add/modify an entry (e.g., 'add 25 0.3')")
                            print("3. Enter 'done' to finish editing")
                            
                            edit_input = input("Enter command: ").strip()
                            
                            if edit_input.lower() == 'done':
                                break
                            elif edit_input.lower().startswith('remove '):
                                try:
                                    soc_to_remove = int(edit_input.split()[1])
                                    if soc_to_remove in SOC_start_probability_distribution:
                                        removed_prob = SOC_start_probability_distribution.pop(soc_to_remove)
                                        print(f"✅ Removed {soc_to_remove}% SOC (was {removed_prob:.3f})")
                                    else:
                                        print(f"❌ SOC {soc_to_remove}% not found in distribution")
                                except (ValueError, IndexError):
                                    print("❌ Invalid format. Use 'remove SOC%' (e.g., 'remove 20')")
                                    
                            elif edit_input.lower().startswith('add '):
                                try:
                                    parts = edit_input.split()
                                    if len(parts) != 3:
                                        print("❌ Invalid format. Use 'add SOC% probability' (e.g., 'add 25 0.3')")
                                        continue
                                    
                                    soc = int(parts[1])
                                    prob = float(parts[2])
                                    
                                    if not (0 <= soc <= 100):
                                        print("❌ SOC% must be between 0 and 100")
                                        continue
                                        
                                    if not (0 <= prob <= 1):
                                        print("❌ Probability must be between 0 and 1")
                                        continue
                                    
                                    SOC_start_probability_distribution[soc] = prob
                                    current_sum = sum(SOC_start_probability_distribution.values())
                                    print(f"✅ Added/Updated: {soc}% SOC with {prob} probability")
                                    print(f"📊 Current sum: {current_sum:.3f}")
                                    
                                except (ValueError, IndexError):
                                    print("❌ Invalid format. Use 'add SOC% probability' (e.g., 'add 25 0.3')")
                            else:
                                print("❌ Unknown command. Use 'remove', 'add', or 'done'")
                        
                        # Re-check sum after editing
                        total_probability = sum(SOC_start_probability_distribution.values())
                        print(f"\n📊 Final distribution after editing:")
                        for soc, prob in sorted(SOC_start_probability_distribution.items()):
                            print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
                        print(f"📊 Total sum: {total_probability:.3f}")
                        
                        if abs(total_probability - 1.0) < 0.001:
                            print("✅ Probability distribution is now valid (sum = 1.0)")
                        elif total_probability > 1.0:
                            print(f"⚠️  Sum still exceeds 1.0 ({total_probability:.3f}). Normalizing automatically.")
                            # Auto-normalize after editing
                            normalized_distribution = {}
                            for soc, prob in SOC_start_probability_distribution.items():
                                normalized_distribution[soc] = prob / total_probability
                            SOC_start_probability_distribution = normalized_distribution
                            print("✅ Distribution has been normalized.")
                        else:
                            print(f"⚠️  Sum is {total_probability:.3f}, not 1.0. Normalizing automatically.")
                            # Auto-normalize after editing
                            normalized_distribution = {}
                            for soc, prob in SOC_start_probability_distribution.items():
                                normalized_distribution[soc] = prob / total_probability
                            SOC_start_probability_distribution = normalized_distribution
                            print("✅ Distribution has been normalized.")
                        break
                        
                    elif edit_choice == 'continue':
                        print("⚠️  Proceeding with invalid distribution (sum ≠ 1.0). Normalizing automatically.")
                        # Auto-normalize
                        normalized_distribution = {}
                        for soc, prob in SOC_start_probability_distribution.items():
                            normalized_distribution[soc] = prob / total_probability
                        SOC_start_probability_distribution = normalized_distribution
                        print("✅ Distribution has been normalized.")
                        break
                    else:
                        print("❌ Invalid choice. Please enter 'edit' or 'continue'")
                else:
                    print("❌ Invalid choice. Please enter 'y' or 'n'")
        else:
            print(f"⚠️  Warning: Probability sum is {total_probability:.3f}, not 1.0")
            print("   The distribution will be automatically normalized.")
            
            # Normalize the distribution
            normalized_distribution = {}
            for soc, prob in SOC_start_probability_distribution.items():
                normalized_distribution[soc] = prob / total_probability
            
            SOC_start_probability_distribution = normalized_distribution
            print("✅ Distribution has been normalized:")
            for soc, prob in sorted(SOC_start_probability_distribution.items()):
                print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
            print(f"📊 Normalized sum: {sum(SOC_start_probability_distribution.values()):.3f}")
        
        calculate_charging_profile(
            df_clear, SOC, GENERAL, df_old_total_percentages, 
            cpu_count=cpu_count, mode=2, SOC_start_probability_distribution=SOC_start_probability_distribution
        )
        
    elif SIMULATION_MODE == 3:
        # Mode 3: Variable SOC_start and SOC_end based on probability distributions
        print("Running Mode 3: Variable SOC_start and SOC_end based on probability distributions")
        
        # Ask for SOC_start probability distribution
        print("\n" + "-" * 40)
        print("SOC Start Probability Distribution")
        print("-" * 40)
        print("Default distribution:")
        
        dist_choice_start = input("Use default SOC_start distribution? (y/n): ").lower().strip()
        
        if dist_choice_start == 'y' or dist_choice_start == '':
            SOC_start_probability_distribution = {
            20: 0.25,
            25: 0.20,
            30: 0.15,
            35: 0.12,
            40: 0.10,
            45: 0.07,
            50: 0.05,
            55: 0.03,
            60: 0.02,
            65: 0.01 }
            print("✅ Using default SOC_start probability distribution")
        else:
            print("\nEnter custom SOC_start probability distribution:")
            print("Format: SOC_percentage probability (e.g., '20 0.2')")
            print("Enter 'done' when finished")
            print("Note: Probabilities should sum to 1.0")
            
            SOC_start_probability_distribution = {}
            while True:
                user_input = input("Enter SOC% and probability (or 'done' to finish): ").strip()
                if user_input.lower() == 'done':
                    break
                elif user_input.lower() in ['exit', 'quit', 'n', 'no']:
                    print("💡 To finish entering values, type 'done'")
                    continue
                
                # Check if input is empty
                if not user_input:
                    print("❌ Empty input. Please enter values or 'done' to finish.")
                    continue
                
                # Split input and validate format
                parts = user_input.split()
                if len(parts) != 2:
                    print("❌ Invalid format. Use 'SOC% probability' (e.g., '20 0.2')")
                    print("💡 Type 'done' to finish entering values")
                    continue
                
                try:
                    # Validate SOC% is an integer
                    soc_str = parts[0]
                    if not soc_str.isdigit():
                        print("❌ SOC% must be an integer (e.g., 20, 25, 30)")
                        continue
                    
                    soc = int(soc_str)
                    
                    # Validate probability is a float
                    prob_str = parts[1]
                    try:
                        prob = float(prob_str)
                    except ValueError:
                        print("❌ Probability must be a number (e.g., 0.2, 0.5, 1.0)")
                        continue
                    
                    # Validate ranges
                    if not (0 <= soc <= 100):
                        print("❌ SOC% must be between 0 and 100")
                        continue
                        
                    if not (0 <= prob <= 1):
                        print("❌ Probability must be between 0 and 1")
                        continue
                    
                    # Check if SOC already exists
                    if soc in SOC_start_probability_distribution:
                        print(f"⚠️  SOC {soc}% already exists. Replacing with new value.")
                    
                    SOC_start_probability_distribution[soc] = prob
                    current_sum = sum(SOC_start_probability_distribution.values())
                    print(f"✅ Added: {soc}% SOC with {prob} probability")
                    print(f"📊 Current sum: {current_sum:.3f}")
                    
                    if current_sum > 1.0:
                        print("⚠️  Warning: Sum exceeds 1.0!")
                        print("💡 You can continue adding or type 'done' to finish and normalize")
                    elif abs(current_sum - 1.0) < 0.001:
                        print("🎯 Perfect! Sum equals 1.0")
                        print("💡 You can enter 'done' to finish or continue adding more entries.")
                    elif current_sum < 1.0:
                        remaining = 1.0 - current_sum
                        print(f"📝 Remaining probability needed: {remaining:.3f}")
                    
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    print("❌ Use format: 'SOC% probability' (e.g., '20 0.2')")
                    print("💡 Type 'done' to finish entering values")
            
            if not SOC_start_probability_distribution:
                print("⚠️  No valid distribution entered. Using default.")
                SOC_start_probability_distribution = {20: 0.2, 25: 0.3, 30: 0.3, 35: 0.2}
        
        # Validate SOC_start probability distribution sum
        total_probability_start = sum(SOC_start_probability_distribution.values())
        print(f"\n Final SOC_start probability distribution:")
        for soc, prob in sorted(SOC_start_probability_distribution.items()):
            print(f"  - {soc}% SOC: {prob:.3f} ({prob*100:.1f}%)")
        print(f" Total sum: {total_probability_start:.3f}")
        
        if abs(total_probability_start - 1.0) < 0.001:
            print(" SOC_start probability distribution is valid (sum = 1.0)")
        else:
            print(f"  Warning: Probability sum is {total_probability_start:.3f}, not 1.0. Normalizing automatically.")
            # Normalize the distribution
            normalized_distribution = {}
            for soc, prob in SOC_start_probability_distribution.items():
                normalized_distribution[soc] = prob / total_probability_start
            SOC_start_probability_distribution = normalized_distribution
            print(" SOC_start distribution has been normalized.")
        
        # Ask for SOC charging amount probability distribution
        print("\n" + "-" * 40)
        print("SOC Charging Amount Probability Distribution")
        print("-" * 40)
        print("This distribution determines how much SOC to add to the starting SOC.")
        print("Default distribution:")
        
        dist_choice_charging = input("Use default SOC charging amount distribution? (y/n): ").lower().strip()
        
        if dist_choice_charging == 'y' or dist_choice_charging == '':
            SOC_charging_probability_distribution = {
            20: 0.25,
            25: 0.30,
            30: 0.25,
            35: 0.15,
            40: 0.05 }
            print(" Using default SOC charging amount probability distribution")
        else:
            print("\nEnter custom SOC charging amount probability distribution:")
            print("Format: charging_amount probability (e.g., '25 0.3')")
            print("Enter 'done' when finished")
            print("Note: Probabilities should sum to 1.0")
            print("Note: Charging amount will be added to starting SOC (capped at 100%)")
            
            SOC_charging_probability_distribution = {}
            while True:
                user_input = input("Enter charging amount and probability (or 'done' to finish): ").strip()
                if user_input.lower() == 'done':
                    break
                elif user_input.lower() in ['exit', 'quit', 'n', 'no']:
                    print(" To finish entering values, type 'done'")
                    continue
                
                # Check if input is empty
                if not user_input:
                    print(" Empty input. Please enter values or 'done' to finish.")
                    continue
                
                # Split input and validate format
                parts = user_input.split()
                if len(parts) != 2:
                    print(" Invalid format. Use 'charging_amount probability' (e.g., '25 0.3')")
                    print(" Type 'done' to finish entering values")
                    continue
                
                try:
                    # Validate charging amount is an integer
                    charging_str = parts[0]
                    if not charging_str.isdigit():
                        print("❌ Charging amount must be an integer (e.g., 20, 25, 30)")
                        continue
                    
                    charging_amount = int(charging_str)
                    
                    # Validate probability is a float
                    prob_str = parts[1]
                    try:
                        prob = float(prob_str)
                    except ValueError:
                        print(" Probability must be a number (e.g., 0.2, 0.5, 1.0)")
                        continue
                    
                    # Validate ranges
                    if not (0 <= charging_amount <= 100):
                        print(" Charging amount must be between 0 and 100")
                        continue
                        
                    if not (0 <= prob <= 1):
                        print("❌ Probability must be between 0 and 1")
                        continue
                    
                    # Check if charging amount already exists
                    if charging_amount in SOC_charging_probability_distribution:
                        print(f"⚠️  Charging amount {charging_amount}% already exists. Replacing with new value.")
                    
                    SOC_charging_probability_distribution[charging_amount] = prob
                    current_sum = sum(SOC_charging_probability_distribution.values())
                    print(f"Added: {charging_amount}% charging amount with {prob} probability")
                    print(f"📊 Current sum: {current_sum:.3f}")
                    
                    if current_sum > 1.0:
                        print("⚠️  Warning: Sum exceeds 1.0!")
                        print("💡 You can continue adding or type 'done' to finish and normalize")
                    elif abs(current_sum - 1.0) < 0.001:
                        print("🎯 Perfect! Sum equals 1.0")
                        print("💡 You can enter 'done' to finish or continue adding more entries.")
                    elif current_sum < 1.0:
                        remaining = 1.0 - current_sum
                        print(f"📝 Remaining probability needed: {remaining:.3f}")
                    
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    print("❌ Use format: 'charging_amount probability' (e.g., '25 0.3')")
                    print("💡 Type 'done' to finish entering values")
            
            if not SOC_charging_probability_distribution:
                print("⚠️  No valid distribution entered. Using default.")
                SOC_charging_probability_distribution = {20: 0.25, 25: 0.30, 30: 0.25, 35: 0.15, 40: 0.05}
        
        # Validate SOC charging amount probability distribution sum
        total_probability_charging = sum(SOC_charging_probability_distribution.values())
        print(f"\n📊 Final SOC charging amount probability distribution:")
        for charging_amount, prob in sorted(SOC_charging_probability_distribution.items()):
            print(f"  - {charging_amount}% charging amount: {prob:.3f} ({prob*100:.1f}%)")
        print(f"📊 Total sum: {total_probability_charging:.3f}")
        
        if abs(total_probability_charging - 1.0) < 0.001:
            print("✅ SOC charging amount probability distribution is valid (sum = 1.0)")
        else:
            print(f"⚠️  Warning: Probability sum is {total_probability_charging:.3f}, not 1.0. Normalizing automatically.")
            # Normalize the distribution
            normalized_distribution = {}
            for charging_amount, prob in SOC_charging_probability_distribution.items():
                normalized_distribution[charging_amount] = prob / total_probability_charging
            SOC_charging_probability_distribution = normalized_distribution
            print("✅ SOC charging amount distribution has been normalized.")
        
        calculate_charging_profile(
            df_clear, SOC, GENERAL, df_old_total_percentages, 
            cpu_count=cpu_count, mode=3, 
            SOC_start_probability_distribution=SOC_start_probability_distribution,
            SOC_charging_probability_distribution=SOC_charging_probability_distribution
        )
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)

