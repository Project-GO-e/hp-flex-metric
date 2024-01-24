import pyomo.environ as pyo
import numpy as np
import pandas as pd
from houseRVO import House
from heatpumpsystem import HeatpumpSystem
from pyomo.opt import SolverFactory


class FlexOptimizer:
    def __init__(self, house: House, heatpump: HeatpumpSystem, congestion_params: dict):
        self.start_congestion = congestion_params['window_before_congestion_ptu']
        start_congestion_ptu = congestion_params['start_congestion_ptu']
        end_congestion_ptu = congestion_params['end_congestion_ptu']
        congestion_length = end_congestion_ptu - start_congestion_ptu
        congestion_length = congestion_length if end_congestion_ptu >= start_congestion_ptu else congestion_length + 96
        self.end_congestion = self.start_congestion + congestion_length
        self.window_end = self.end_congestion + congestion_params['window_after_congestion_ptu']
        self.house = house
        self.heatpump = heatpump

        self.model = pyo.ConcreteModel()

    def create_and_solve(self,
                         baseline_profile: pd.Series,
                         dhw_profile: pd.Series,
                         weather_profiles: pd.DataFrame,
                         solar_gain: pd.Series,
                         start_states: pd.Series,
                         set_temperature: float,
                         time_step: float):

        self.create(baseline_profile,
                    dhw_profile,
                    weather_profiles,
                    solar_gain,
                    start_states,
                    set_temperature,
                    time_step)

        self.solve()

    def create(self,
               baseline_profile: pd.Series,
               dhw_profile: pd.Series,
               weather_profiles: pd.DataFrame,
               solar_gain: pd.Series,
               start_states: pd.Series,
               set_temperature: float,
               time_step: float):

        model = pyo.ConcreteModel()  # create new model
        model.time_index = pyo.RangeSet(0, self.window_end - 1)
        model.time_index_T = pyo.RangeSet(0, self.window_end)

        # Add all parameters
        # All house and heat pump properties
        model.time_step = pyo.Param(initialize=time_step, within=pyo.NonNegativeReals)  # [s]
        model.En = pyo.Param(initialize=self.heatpump.nominal_power * model.time_step, within=pyo.NonNegativeReals)  # [J]
        model.COP_sh = pyo.Param(initialize=4.0, within=pyo.NonNegativeReals)  # [-]
        model.COP_dhw = pyo.Param(initialize=2.5, within=pyo.NonNegativeReals)  # [-]
        model.COP_element = pyo.Param(initialize=self.heatpump.cop_element, within=pyo.NonNegativeReals)
        model.minrel = pyo.Param(initialize=self.heatpump.minimal_relative_load, within=pyo.NonNegativeReals)  # [-]
        model.E_element = pyo.Param(initialize=self.heatpump.heat_element * model.time_step, within=pyo.NonNegativeReals)

        house_tank_properties = self.heatpump.house_tank_properties
        dhw_tank_properties = self.heatpump.dhw_tank_properties
        model.C_house_tank = pyo.Param(initialize=house_tank_properties['house_tank_mass'] * 4183.0)  # [J/K]
        model.C_dhw_tank = pyo.Param(initialize=dhw_tank_properties['dhw_tank_mass'] * 4183.0)  # [J/K]
        model.minimal_off_time = pyo.Param(initialize=self.heatpump.n_minimal_off_time)  # [s]
        model.T_house_tank_min = pyo.Param(initialize=house_tank_properties['house_tank_T_min_limit'])
        model.T_house_tank_max = pyo.Param(initialize=house_tank_properties['house_tank_T_max_limit'])
        model.T_house_tank_hor = pyo.Param(initialize=house_tank_properties['house_tank_T_set'])
        model.T_house_tank_set = pyo.Param(initialize=house_tank_properties['house_tank_T_set'])
        model.T_house_tank_0 = pyo.Param(initialize=start_states.loc['house_tank_T'])
        model.T_dhw_tank_min = pyo.Param(initialize=dhw_tank_properties['dhw_tank_T_min_limit'])
        model.T_dhw_tank_max = pyo.Param(initialize=dhw_tank_properties['dhw_tank_T_max_limit'])
        model.T_dhw_tank_hor = pyo.Param(initialize=dhw_tank_properties['dhw_tank_T_set'])
        model.T_dhw_tank_set = pyo.Param(initialize=dhw_tank_properties['dhw_tank_T_set'])
        model.T_dhw_tank_0 = pyo.Param(initialize=start_states.loc['dhw_tank_T'])
        model.T_in_min = pyo.Param(initialize=set_temperature - 0.8)
        model.T_in_max = pyo.Param(initialize=set_temperature + 0.8)
        model.T_in_hor = pyo.Param(initialize=set_temperature)
        model.T_in_0 = pyo.Param(initialize=start_states.loc['T_in'])
        model.T_out_0 = pyo.Param(initialize=start_states.loc['T_out'])

        # Matrices
        matrix_size = len(self.house.K[0, :])
        matrix_index = pyo.RangeSet(1, matrix_size)
        model.K = pyo.Param(matrix_index, matrix_index, initialize=self.matrix2dict(self.house.K))
        model.K_amb = pyo.Param(matrix_index, matrix_index, initialize=self.matrix2dict(self.house.K_amb))
        model.C = pyo.Param(matrix_index, matrix_index, initialize=self.matrix2dict(self.house.C))

        # Add all variables
        # control variables
        model.E_house_tank = pyo.Var(model.time_index, bounds=(0, model.En))
        model.E_dhw_tank = pyo.Var(model.time_index, bounds=(0, model.En))
        model.on_house = pyo.Var(model.time_index, within=pyo.Binary, initialize=0)  # binary variable: 1: house, 0: dhw
        model.element_on = pyo.Var(model.time_index, bounds=(0, 1.0))  # binary variable: 1: element on
        model.E_house = pyo.Var(model.time_index, within=pyo.NonNegativeReals, initialize=0)  # om laatste None te voorkomen
        model.E_hp = pyo.Var(model.time_index, within=pyo.NonNegativeReals)
        model.on = pyo.Var(model.time_index, within=pyo.Binary, initialize=0)  # binary variable: 1-HP on, 0 HP off

        # state of charge
        model.T_in = pyo.Var(model.time_index_T, within=pyo.NonNegativeReals)
        model.T_out = pyo.Var(model.time_index_T, within=pyo.NonNegativeReals)
        model.T_house_tank = pyo.Var(model.time_index_T, within=pyo.NonNegativeReals)
        model.T_dhw_tank = pyo.Var(model.time_index_T, within=pyo.NonNegativeReals)
        model.T_in_slack = pyo.Var(model.time_index_T, within=pyo.NonNegativeReals)

        model.E_max_cong = pyo.Var(within=pyo.NonNegativeReals)  # variable to suppress the max energy use during congestion
        model.E_max = pyo.Var(within=pyo.NonNegativeReals)  # variable to suppress the max energy use


        # Add Constraints
        # Relations between decision variables
        model.cnstr_load = pyo.Constraint(model.time_index, rule=lambda m, t: m.E_hp[t] - m.element_on[t] * m.E_element / m.COP_element == (m.E_house_tank[t]/m.COP_sh + m.E_dhw_tank[t]/m.COP_sh))
        model.cnstr_max_energy = pyo.Constraint(model.time_index, rule=lambda m, t: m.E_house_tank[t] + m.E_dhw_tank[t] <= m.En)

        # Initial Conditions
        model.cnstr_T_house_tank0 = pyo.Constraint(rule=lambda m: m.T_house_tank[model.time_index.first()] == m.T_house_tank_0)
        model.cnstr_T_dhw_tank0 = pyo.Constraint(rule=lambda m: m.T_dhw_tank[model.time_index.first()] == m.T_dhw_tank_0)
        model.cnstr_T_in_0 = pyo.Constraint(rule=lambda m: m.T_in[model.time_index.first()] == m.T_in_0)
        model.cnstr_T_out_0 = pyo.Constraint(rule=lambda m: m.T_out[model.time_index.first()] == m.T_out_0)

        # Constraining state of charge
        model.cnstr_T_house_tank_min = pyo.Constraint(model.time_index,  rule=lambda m, t: m.T_house_tank[t] >= m.T_house_tank_min)
        model.cnstr_T_house_tank_max = pyo.Constraint(model.time_index, rule=lambda m, t: m.T_house_tank[t] <= m.T_house_tank_max)
        model.cnstr_T_house_tank_hor = pyo.Constraint(rule=lambda m: m.T_house_tank[model.time_index_T.last()] >= m.T_house_tank_hor)

        model.cnstr_T_dhw_tank_min = pyo.Constraint(model.time_index, rule=lambda m, t: m.T_dhw_tank[t] >= m.T_dhw_tank_min)
        model.cnstr_T_dhw_tank_max = pyo.Constraint(model.time_index, rule=lambda m, t: m.T_dhw_tank[t] <= m.T_dhw_tank_max)
        model.cnstr_T_dhw_tank_hor = pyo.Constraint(rule=lambda m: m.T_dhw_tank[model.time_index_T.last()] >= m.T_dhw_tank_hor)

        def constraint_min_T_in(m, t):
            start_cong = self.start_congestion
            end_cong = self.end_congestion
            discomfort_duration_wrt_cong = 4
            if (t > (start_cong - discomfort_duration_wrt_cong)) and (t < (end_cong + discomfort_duration_wrt_cong)):
                return m.T_in[t] >= m.T_in_min - m.T_in_slack[t]
            else:
                return pyo.Constraint.Skip

        def constraint_hor_T_in(m, t):
            start_cong = self.start_congestion
            end_cong = self.end_congestion
            discomfort_duration_wrt_cong = 4
            if (t <= (start_cong - discomfort_duration_wrt_cong)) or (t >= (end_cong + discomfort_duration_wrt_cong)):
                return m.T_in[t] >= m.T_in_hor - m.T_in_slack[t]
            else:
                return pyo.Constraint.Skip

        model.cnstr_T_in_min = pyo.Constraint(model.time_index, rule=constraint_min_T_in)
        model.cnstr_T_in_hor = pyo.Constraint(model.time_index, rule=constraint_hor_T_in)
        # Check if the T_in exceeds the bound due to natural heating already. if not, add the upper bound constraint
        too_hot_in_house = self.check_too_hot_in_house(weather_profiles, solar_gain, start_states, time_step, model.T_in_max)
        if not too_hot_in_house:
            model.cnstr_T_in_max = pyo.Constraint(model.time_index, rule=lambda m, t: m.T_in[t] <= m.T_in_max)

        # Updating state of charge
        model.cnstr_update_T_house_tank = pyo.Constraint(model.time_index, rule=lambda m, t:
        m.C_house_tank * m.T_house_tank[t + 1] == m.C_house_tank * m.T_house_tank[t] +
        m.E_house_tank[t] - m.E_house[t])

        dhw_values = dhw_profile.values

        model.cnstr_update_T_dhw_tank = pyo.Constraint(model.time_index, rule=lambda m, t:
        m.C_dhw_tank * m.T_dhw_tank[t + 1] == m.C_dhw_tank * m.T_dhw_tank[t] +
        m.E_dhw_tank[t] - dhw_values[t] * m.time_step + m.element_on[t] * m.E_element)

        T_airs = weather_profiles['T'].values
        T_floors = weather_profiles['TB4'].values
        Q_solar = solar_gain.values

        def constraint_update_T_in(m, t):
            return m.C[1, 1] * m.T_in[t + 1] == m.C[1, 1] * m.T_in[t] + m.time_step * (
                        - (m.K[1, 1] * m.T_in[t] + m.K[1, 2] * m.T_out[t])
                        + (m.K_amb[1, 1] * T_airs[t] + m.K_amb[1, 2] * T_floors[t])
                        + Q_solar[t] + m.E_house[t] / m.time_step
            )

        def constraint_update_T_out(m, t):
            return m.C[2, 2] * m.T_out[t + 1] == m.C[2, 2] * m.T_out[t] + m.time_step * (
                        - (m.K[2, 1] * m.T_in[t] + m.K[2, 2] * m.T_out[t])
                        + (m.K_amb[2, 1] * T_airs[t] + m.K_amb[2, 2] * T_floors[t])
            )

        model.cnstr_update_T_in = pyo.Constraint(model.time_index, rule=constraint_update_T_in)
        model.cnstr_update_T_out = pyo.Constraint(model.time_index, rule=constraint_update_T_out)

        # Constraint: Suppress max consumptions in the whole window
        model.cstr_E_max = pyo.Constraint(model.time_index, rule=lambda m, t: m.E_hp[t] <= m.E_max)

        def constraint_E_max_cong(m, t):
            start_cong = self.start_congestion
            end_cong = self.end_congestion
            if (t >= start_cong) and (t < end_cong):
                return m.E_hp[t] <= m.E_max_cong
            else:
                return pyo.Constraint.Skip
        model.cstr_E_max_cong = pyo.Constraint(model.time_index, rule=constraint_E_max_cong)

        # Constricting the load to be smaller than the one in the baseline
        def constraint_bl_reduction(m, t):
            start_cong = self.start_congestion
            end_cong = self.end_congestion
            if (t >= start_cong) and (t < end_cong):
                return m.E_hp[t] <= baseline_profile.values[t] * m.time_step
            else:
                return pyo.Constraint.Skip

        model.cnstr_bl_reduction = pyo.Constraint(model.time_index, rule=constraint_bl_reduction)

        # Cost function:
        def cost_function(m):
            start_cong = self.start_congestion
            end_cong = self.end_congestion

            # our index goes from 0, ..., N
            # if the end congestion is at point 3, we want all t = 0, 1, 2 in the OF
            congestion_term = sum(m.E_hp[t] for t in m.time_index if (t >= start_cong) and (t < end_cong))
            congestion_term += m.E_max
            window_term = m.E_max
            slack_term = sum(1.0e8 * m.T_in_slack[t] for t in m.time_index)  # For numerical stability
            return congestion_term + 0.01 * window_term + slack_term

        model.OF = pyo.Objective(expr=cost_function, sense=pyo.minimize)
        self.model = model

    def solve(self, tee_flg=False):
        # information on logging: https://www.gurobi.com/documentation/9.5/refman/simplex_logging.html
        opt = SolverFactory('appsi_highs')
        opt.options['mipgap'] = 5e-2
        # opt.options['TimeLimit'] = 240
        # opt.options['NoRelHeurTime'] = 600
        # opt.options['MIPFocus'] = 1
        status = opt.solve(self.model, tee=tee_flg)
        print(f"Solver status: {status.solver.termination_condition}")
        assert status.solver.termination_condition != 'infeasible', "solution status infeasible"

    def check_too_hot_in_house(self, weather_profiles, solar_gain, start_states, time_step, T_in_max) -> bool:
        T_in = start_states.loc['T_in']
        T_out = start_states.loc['T_out']

        K = self.house.K
        K_amb = self.house.K_amb
        C_inv = self.house.C_inv

        T = np.array([T_in, T_out])
        for t in range(self.window_end):
            T_ext = np.array([weather_profiles['T'].iloc[t], weather_profiles['TB4'].iloc[t]])
            Q_sol = solar_gain.iloc[t]
            rhs = -np.matmul(K, T) + np.matmul(K_amb, T_ext) + np.array([Q_sol, 0.0])
            T += time_step * np.matmul(C_inv, rhs)
            if T[0] > T_in_max:
                return True
        return False

    @staticmethod
    def get_result_array(variable) -> np.array:
        # variable is of the form model.X
        return np.array([variable[i].value for i in variable])

    @staticmethod
    def matrix2dict(matrix) -> dict:
        d = dict()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                d[(i + 1, j + 1)] = matrix[i, j]
        return d

    @staticmethod
    def array2dict(array) -> dict:
        d = dict()
        for i in range(len(array)):
            d[i + 1] = array[i]
        return d


