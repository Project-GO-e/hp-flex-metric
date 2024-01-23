import pandas as pd
from typing import Optional, List, Dict
import numpy as np
from houseRVO import House
from heatpumpsystem import HeatpumpSystem
from datetime import datetime
from flexoptimizer import FlexOptimizer


class FlexProfileGenerator:
    def __init__(self):
        self.time_step: Optional[float] = None
        self.weather_profiles: Optional[pd.DataFrame] = None
        self.house_params: Optional[Dict] = None
        self.congestion_params: Optional[Dict] = None
        self.house: Optional[House] = None
        self.heatpumpsystem: Optional[HeatpumpSystem] = None

        self.solar_gain_profile: Optional[pd.Series] = None
        self.dhw_heat_profiles: Optional[pd.DataFrame] = None

        self.baseline_profiles: Optional[pd.DataFrame] = None
        self.state_temperatures: Optional[List[pd.DataFrame]] = None
        self.set_temperatures: Optional[List[float]] = None

        self.builder = None

    def generate(self, n_profiles=100):
        # Check whether the generator is initialized properly
        if self.house_params is None:
            raise ValueError(f"initialize physical system first")
        if self.congestion_params is None:
            raise ValueError(f"initialize congestion window first")

        # Check if there are more profiles requested than there are baseline profiles
        if n_profiles > len(self.state_temperatures):
            raise ValueError(f"n_profiles cannot be larger than {len(self.state_temperatures)}")

        window_start_id, window_end_id = self.get_window_indices()

        # Slice data applicable for all profiles
        baseline_profiles_w = self.baseline_profiles[window_start_id:window_end_id]
        dhw_heat_profiles_w = self.dhw_heat_profiles.iloc[window_start_id:window_end_id]
        weather_profiles_w = self.weather_profiles.iloc[window_start_id:window_end_id]
        solar_heat_profile_w = self.solar_heat_profile.iloc[window_start_id:window_end_id]

        flex_optimizer = FlexOptimizer(self.house, self.heatpumpsystem, self.congestion_params)
        flex_profiles = []
        for n in range(n_profiles):
            print(f"Flex Profile: {n + 1}")
            # Slice all profiles depending on the profile
            baseline_profile_w = baseline_profiles_w[baseline_profiles_w.columns[n]]
            dhw_heat_profile_w = dhw_heat_profiles_w[dhw_heat_profiles_w.columns[n]]
            start_state_temperatures_w = self.state_temperatures[n].iloc[window_start_id]  # we only need the first
            set_temperature = self.set_temperatures[n]

            # Create and solve optimization problem
            try:
                flex_optimizer.create_and_solve(baseline_profile_w,
                                                dhw_heat_profile_w,
                                                weather_profiles_w,
                                                solar_heat_profile_w,
                                                start_state_temperatures_w,
                                                set_temperature,
                                                self.time_step)

                solved_model = flex_optimizer.model
                flex_profile = flex_optimizer.get_result_array(solved_model.E_hp) / self.time_step
                flex_profiles.append(flex_profile)
            except:
                print(f"Creating profile resulted in an infeasible solution to the solver and will thus be skipped")

        # Create a df
        df_flex_profiles = pd.DataFrame(data=np.array(flex_profiles).T,
                                        index=self.baseline_profiles.index[window_start_id: window_end_id])
        df_flex_profiles = df_flex_profiles.astype(int)  # convert to int to reduce memory requirements
        return df_flex_profiles

    def get_window_indices(self):
        start_congestion_day = self.congestion_params['start_congestion_day']
        start_congestion_month = self.congestion_params['start_congestion_month']
        start_congestion_ptu = self.congestion_params['start_congestion_ptu']
        end_congestion_ptu = self.congestion_params['end_congestion_ptu']
        window_before_congestion_ptu = self.congestion_params['window_before_congestion_ptu']
        window_after_congestion_ptu = self.congestion_params['window_after_congestion_ptu']

        # Find index numbers giving the boundaries of the window under consideration
        # Find the index at which the window starts using dates
        start_window_ptu = start_congestion_ptu - window_before_congestion_ptu  # in [0, 95]
        day = start_congestion_day if start_window_ptu > 0 else start_congestion_day - 1

        window_start_date = datetime(year=2020, month=start_congestion_month, day=day,
                                     hour=start_window_ptu // 4,
                                     minute=(start_window_ptu % 4) * 15,
                                     second=0)

        window_start_id = list(self.baseline_profiles.index).index(window_start_date)  # in [0, 366 * 96]

        # Find the index at which the window ends using window start index and window length
        congestion_length = end_congestion_ptu - start_congestion_ptu
        congestion_length = congestion_length if end_congestion_ptu >= start_congestion_ptu else congestion_length + 96
        window_length = window_before_congestion_ptu + congestion_length + window_after_congestion_ptu
        window_end_id = window_start_id + window_length  # in [0, 366 * 96]
        return window_start_id, window_end_id

