import pandas as pd
from typing import List
import pickle
from houseRVO import House

from heatpumpsystem import HeatpumpSystem
from datetime import datetime
from flexprofilegenerator import FlexProfileGenerator


class FlexProfileGeneratorBuilder:
    def __init__(self):
        self.flex_profile_generator = FlexProfileGenerator()
        self.flex_profile_generator.builder = self

    def build_empty(self) -> FlexProfileGenerator:
        return self.flex_profile_generator

    def build_physical_attributes(self, house_params: dict) -> FlexProfileGenerator:
        self.set_physical_attributes(house_params)
        return self.flex_profile_generator

    def build_congestion_attributes(self, congestion_params: dict) -> FlexProfileGenerator:
        self.set_congestion_attributes(congestion_params)
        return self.flex_profile_generator

    def set_physical_attributes(self, house_params: dict):
        # Check user given house parameters first
        self.check_house_params(house_params)

        # Set physical attributes
        # Universal for all generators
        self.flex_profile_generator.time_step = 900
        self.flex_profile_generator.weather_profiles = pd.read_csv('weather_2020_15min.csv',
                                                                   index_col=0, parse_dates=True)

        # Input parameters
        self.flex_profile_generator.house_params = house_params

        # System objects
        self.flex_profile_generator.house = self.get_house(house_params)
        self.flex_profile_generator.heatpumpsystem = self.get_heatpumpsystem(house_params)

        # Heat fluxes
        self.flex_profile_generator.solar_heat_profile = self.get_solar_heat_profile()
        self.flex_profile_generator.dhw_heat_profiles = self.get_dhw_heat_profiles(house_params)

        # Results from baseline calculation:
        self.flex_profile_generator.baseline_profiles = self.get_baseline_profiles(house_params)
        self.flex_profile_generator.state_temperatures = self.get_state_temperatures(house_params)
        self.flex_profile_generator.set_temperatures = self.get_set_temperatures(house_params)

    def set_congestion_attributes(self, congestion_params):
        # Check user given house parameters first
        self.check_congestion_params(congestion_params)

        # Set attributes
        self.flex_profile_generator.congestion_params = congestion_params

    @staticmethod
    def check_house_params(house_params: dict):
        # Check whether a dictionary is provided
        if not type(house_params) is dict:
            raise TypeError(f'house_params should by of type: dict')

        # Check whether all necessary entries are there
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']

        # Check if values provided are from the allowed list of values
        if house_type not in ['vrijst', '2_1kap', 'hoek', 'tussen', 'maisonette', 'galerij', 'portiek', 'overig']:
            raise ValueError(f'house_type: {house_type} was not recognized')
        if house_year not in ['< 1946', '1946 - 1964', '1965 - 1974', '1975 - 1991', '1992 - 2005', '2006 - 2012',
                              '> 2012']:
            raise ValueError(f'house_year: {house_year} was not recognized')
        if residents_type not in ['one_person', 'two_person', 'family']:
            raise ValueError(f'house_type: {residents_type} was not recognized')

    @staticmethod
    def check_congestion_params(congestion_params: dict):
        # Check whether a dictionary is provided
        if not type(congestion_params) is dict:
            raise TypeError(f'congestion_params should by of type: dict')

        # Check whether all values are integers
        for key, value in congestion_params.items():
            if not isinstance(value, int):
                raise ValueError(f"The value for key '{key}' should be an integer.")

        # Check whether all necessary entries are there
        start_congestion_month = congestion_params['start_congestion_month']
        start_congestion_day = congestion_params['start_congestion_day']
        start_congestion_ptu = congestion_params['start_congestion_ptu']
        end_congestion_ptu = congestion_params['end_congestion_ptu']
        window_before_congestion_ptu = congestion_params['window_before_congestion_ptu']
        window_after_congestion_ptu = congestion_params['window_after_congestion_ptu']

        # Check whether a valid date in 2020 was provided
        try:
            date = datetime(2020, start_congestion_month, start_congestion_day)
        except ValueError as e:
            print(f'{start_congestion_day}-{start_congestion_month} is not a valid day in 2020')

        # Check on valid ptu numbers
        valid_ptus = [i for i in range(96)]
        for param_name in ['start_congestion_ptu', 'end_congestion_ptu',
                           'window_before_congestion_ptu', 'window_after_congestion_ptu']:
            if not congestion_params[param_name] in valid_ptus:
                raise ValueError(f'Value of {param_name} should be in range 0-96')

        # Check for congestion lengths that are 0 or too short
        congestion_length = end_congestion_ptu - start_congestion_ptu
        congestion_length = congestion_length if end_congestion_ptu >= start_congestion_ptu else congestion_length + 96

        if congestion_length == 0:
            raise ValueError("congestion length should be larger than 0")

        if congestion_length > 48:
            raise ValueError("congestion length should be smaller or equal to 48 ptus")

    @staticmethod
    def get_house(house_params: dict) -> House:
        # Build a House class from the properties in rvo_woningen.csv
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        # Read dataframe with all properties of the RVO houses
        df = pd.read_csv('rvo_woningen.csv', header=0, index_col=0, sep=';', decimal=',')
        properties = df.loc[house_type + ' ' + house_year]

        capacities = {'C_in': properties.C_in, 'C_out': properties.C_out}
        resistances = {'R_exch': 1.0 / properties.UA_exch,  # losses between the two nodes
                       'R_floor': 1.0 / properties.UA_floor,
                       'R_vent': 1.0 / properties.UA_vent,
                       'R_cond': 1.0 / properties.UA_transm - 1.0 / properties.UA_exch}
        window_area = properties.A_glass
        return House(capacities, resistances, window_area)

    def get_heatpumpsystem(self, house_params: dict) -> HeatpumpSystem:
        # Build a heat pump system with dimensions as provided in house_hp_map.pickle
        # And standard values given in the function
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']

        # Read the dictionary with nominal heat and tank masses corresponding to the house params
        # These parameters were tuned for all the different heat pump types
        with open('house_hp_map.pickle', 'rb') as file:
            house_hp_map = pickle.load(file)
            key = house_type + '+' + house_year.replace(" ", "") + '+' + residents_type
            hp_params = house_hp_map[key]

        # Set other heat pump properties
        # DON'T CHANGE THESE VALUES
        # Some standard parameters which could be changed
        minimal_relative_load = 0.3
        n_minimal_off_time = 1  # don't set smaller than 1!
        cop_element = 1.0
        heat_element = 3000

        # Create the SH tank properties
        house_tank_T_min_limit = 25.0 + 273.15
        house_tank_T_max_limit = 60 + 273.15
        house_tank_T_set = 40 + 273.15
        house_tank_T_init = 40 + 273.15

        house_tank_properties = {'house_tank_mass': hp_params['sh_tank_mass'],
                                 'house_tank_T_set': house_tank_T_set,
                                 'house_tank_T_min_limit': house_tank_T_min_limit,
                                 'house_tank_T_max_limit': house_tank_T_max_limit,
                                 'house_tank_T_init': house_tank_T_init}

        # Create  the DHW tank properties
        dhw_tank_T_min_limit = 25 + 273.15
        dhw_tank_T_max_limit = 85 + 273.15
        dhw_tank_T_set = 55 + 273.15
        dhw_tank_T_init = 50 + 273.15
        temperature_tap_water = 15 + 273.15

        dhw_tank_properties = {'dhw_tank_mass': hp_params['dhw_tank_mass'],
                               'dhw_tank_T_set': dhw_tank_T_set,
                               'dhw_tank_T_min_limit': dhw_tank_T_min_limit,
                               'dhw_tank_T_max_limit': dhw_tank_T_max_limit,
                               'dhw_tank_T_init': dhw_tank_T_init,
                               'temperature_tap_water': temperature_tap_water}

        heatpumpsystem = HeatpumpSystem(hp_params['nominal_heat'],
                                        minimal_relative_load,
                                        n_minimal_off_time,
                                        heat_element,
                                        cop_element,
                                        self.flex_profile_generator.house,
                                        house_tank_properties,
                                        dhw_tank_properties)
        return heatpumpsystem

    def get_solar_heat_profile(self) -> pd.Series:
        # Calculate solar gain profile from the house and precalculated irradiance profiles on the
        # front (east) and back (wall)
        house = self.flex_profile_generator.house
        irradiance_walls = pd.read_csv("irradiance_walls.csv", index_col=0, header=0)
        irradiance_front = irradiance_walls['irradiance_front']
        irradiance_back = irradiance_walls['irradiance_back']

        # shgc = solar heat gain coefficient
        # We assume half of the window area is on the back and the other half is on the front of the building
        solar_heat_profile = 0.5 * house.shgc * house.window_area * (irradiance_front + irradiance_back)
        solar_heat_profile = solar_heat_profile.loc['2020-01-01 00:00:00':]
        return solar_heat_profile

    def get_dhw_heat_profiles(self, house_params: dict) -> pd.DataFrame:
        # The DHW heat profile only depends on the residents type
        # The profiles are pre-calculated for all the residents types
        residents_type = house_params['residents_type']
        dhw_profiles = pd.read_csv("dhw_profiles_" + residents_type + ".csv", index_col=0)  # [L/s]
        # Convert water flow to the energy required to go tap water temperature -> dhw tank set temperature
        heatpumpsystem = self.flex_profile_generator.heatpumpsystem
        dhw_tank_T_set = heatpumpsystem.dhw_tank_properties['dhw_tank_T_set']
        T_tap = heatpumpsystem.dhw_tank_properties['temperature_tap_water']
        thermal_capacity_water = 4182.0  # [J/(kg K)]
        density_water = 0.997  # [kg/L]
        dhw_heat_profiles = dhw_profiles * density_water * thermal_capacity_water * (dhw_tank_T_set - T_tap)
        return dhw_heat_profiles

    @staticmethod
    def get_baseline_profiles(house_params: dict) -> pd.DataFrame:
        # Get the index of the dataframe of the baseline profiles
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']
        baseline_profiles = pd.read_csv('baselines+' + house_type + '+' +
                                        house_year.replace(" ", "").replace(">", "") + '+' +
                                        residents_type + '.csv', index_col=0, parse_dates=True)
        return baseline_profiles

    @staticmethod
    def get_state_temperatures(house_params: dict) -> List[pd.DataFrame]:
        # Get the temperatures in the house and the tanks from the baseline calculations
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']
        with open('states+' + house_type + '+' + house_year.replace(" ", "").replace(">", "") + '+'
                  + residents_type + '.pickle', 'rb') as file:
            # List with all dataframes containing the buffer and house temperatures. 1 for every baseline
            state_temperatures = pickle.load(file)  # [K]
        return state_temperatures

    @staticmethod
    def get_set_temperatures(house_params: dict) -> List:
        # Get the house set temperatures from the baseline calculations
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']
        with open('temps+' + house_type + '+' + house_year.replace(" ", "").replace(">", "") + '+'
                  + residents_type + '.pickle', 'rb') as file:
            # List of all set point temperatures. 1 for every baseline
            set_temperatures = pickle.load(file)  # [K]
        return set_temperatures