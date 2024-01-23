from houseRVO import House
from dataclasses import dataclass

'''
The full implementation of this class contains more which which are relevant 
for generating the baseline profiles. These will be added later. Now this class functions as a data class
'''


@dataclass
class HeatpumpSystem:
    def __init__(self, nominal_power: float,
                 minimal_relative_load: float,
                 n_minimal_off_time: int,
                 heat_element: float,
                 cop_element: float,
                 house: House,
                 house_tank_properties: dict,
                 dhw_tank_properties: dict):

        self.nominal_power = nominal_power
        self.minimal_relative_load = minimal_relative_load
        self.n_minimal_off_time = n_minimal_off_time
        self.heat_element = heat_element
        self.cop_element = cop_element
        self.house = house
        self.house_tank_properties = house_tank_properties
        self.dhw_tank_properties = dhw_tank_properties
