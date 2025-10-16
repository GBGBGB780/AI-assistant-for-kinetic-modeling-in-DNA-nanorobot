# coding=gb2312
# kinetics/nanorobot_solver.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import openpyxl
import math


class NanorobotSolver:
    def __init__(self, model_name, config_names_str, experimental_data_path_a):
        self.model_name = model_name
        self.num_configs = 14
        self.config_names = [f"State_{i}" for i in range(self.num_configs)]

        self.experimental_data_a = self._load_experimental_data(experimental_data_path_a)
        self.experimental_data_b = None
        self.parameters = None

        # FIX: Removed redundant default parameter dictionaries as they are now fully managed by the config.ini file.

    @staticmethod
    def _safe_int(val, default=0, min_val=None, max_val=None):
        try:
            if val is None: return int(default)
            if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)): return int(default)
            iv = int(val)
        except (ValueError, TypeError):
            try:
                iv = int(float(val))
            except (ValueError, TypeError):
                return int(default)
        if min_val is not None: iv = max(iv, min_val)
        if max_val is not None: iv = min(iv, max_val)
        return iv

    @staticmethod
    def _safe_float(val, default=0.0, min_val=None, max_val=None):
        try:
            if val is None:
                fv = float(default)
            else:
                fv = float(val)
                if np.isnan(fv) or np.isinf(fv): return float(default)
        except (ValueError, TypeError):
            return float(default)
        if min_val is not None: fv = max(fv, min_val)
        if max_val is not None: fv = min(fv, max_val)
        return fv

    @staticmethod
    def _sanitize_array(arr, nan_replacement=0.0, min_val=None, max_val=None):
        arr = np.array(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=nan_replacement, posinf=max_val if max_val is not None else 1e12,
                            neginf=min_val if min_val is not None else -1e12)
        if min_val is not None or max_val is not None:
            arr = np.clip(arr, a_min=min_val, a_max=max_val)
        return arr

    def _load_experimental_data(self, path):
        """
        Loads experimental data from either a .csv or .xlsx file.
        """
        try:
            # 检查文件扩展名，根据不同类型使用不同的pandas函数
            if path.lower().endswith('.csv'):
                data = pd.read_csv(path)
            elif path.lower().endswith('.xlsx'):
                data = pd.read_excel(path)
            else:
                # 如果文件格式不支持，则打印错误并返回None
                print(f"Error: Unsupported file format for '{path}'. Please use .csv or .xlsx.")
                return None

            # 加载数据后的处理逻辑保持不变
            if 'Time' not in data.columns and data.columns[0] != 'Time':
                data.rename(columns={data.columns[0]: 'Time'}, inplace=True)

            return data

        except FileNotFoundError:
            print(f"Error: The file was not found at path: '{path}'")
            return None
        except Exception as e:
            print(f"Error loading or processing data from '{path}': {e}")
            return None

    def set_parameters(self, params_dict):
        # FIX: Simplified the function to directly accept the complete parameter dictionary from main.py, removing redundant logic and performance-draining print statements.
        if not params_dict or len(params_dict) < 27:
            raise ValueError("set_parameters requires a complete dictionary with all 27 physical parameters.")
        self.parameters = params_dict

    def _calculate_free_energies(self):
        """
        Calculates the free energies and forces for all 14 configurations,
        perfectly matching the logic from the MATLAB script.
        """
        if self.parameters is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        # --- 1. 安全地获取所有需要的物理参数 ---
        p = self.parameters
        kBT = self._safe_float(p.get("kBT", 4.14), min_val=1e-6)
        lp_s = self._safe_float(p.get("lp_s", 0.75), min_val=1e-6)
        lc_s = self._safe_float(p.get("lc_s", 0.7), min_val=1e-6)
        lc_d = self._safe_float(p.get("lc_d", 0.34), min_val=1e-6)
        E_b = self._safe_float(p.get('E_b', -1.2))
        E_b_azo_trans = self._safe_float(p.get('E_b_azo_trans', -1.0))
        E_b_azo_cis = self._safe_float(p.get('E_b_azo_cis', -0.1))

        n_D1 = self._safe_int(p.get('n_D1', 10), min_val=0)
        n_D2 = self._safe_int(p.get('n_D2', 10), min_val=0)
        n_gray = self._safe_int(p.get('n_gray', 10), min_val=0)
        n_hairpin_1 = self._safe_int(p.get('n_hairpin_1', 8), min_val=1)
        n_hairpin_2 = self._safe_int(p.get('n_hairpin_2', 8), min_val=1)
        n_T_hairpin_1 = self._safe_int(p.get('n_T_hairpin_1', 3), min_val=0)
        n_T_hairpin_2 = self._safe_int(p.get('n_T_hairpin_2', 2), min_val=0)
        n_track_1 = self._safe_int(p.get('n_track_1', 15), min_val=1)
        n_track_2 = self._safe_int(p.get('n_track_2', 55), min_val=1)
        dE_TYE = self._safe_float(p.get('dE_TYE', -1.55))

        # --- 2. 计算6个基本构象的能量和力 ---
        E_config_t_base = np.zeros(6)
        E_config_c_base = np.zeros(6)
        f_config_t_base = np.zeros(6)
        f_config_c_base = np.zeros(6)

        # 2.1 计算 shearing foot 和 zipper foot 的能量
        E_shear_foot = 1000.0
        for i in range(n_D2 + 1):
            n_D2_detach = i
            E_b_shear = E_b * (n_D1 + n_D2 - n_D2_detach)
            denom = lc_s * (2 * n_D2_detach + n_D1)
            if abs(denom) < 1e-9:
                continue

            x = (n_track_1 * lc_d) / denom
            if 0 <= x < 1:
                try:
                    # 蠕虫状链 (Worm-like chain) 能量公式
                    E_shear = E_b_shear + denom * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                except (ValueError, ZeroDivisionError):
                    E_shear = 1000.0
            else:
                E_shear = 1000.0

            if E_shear_foot > E_shear:
                E_shear_foot = E_shear

        E_zipper_foot = E_b * (n_D1 + n_D2)

        # 状态 1-1, 1-2, 2-1, 2-2 的基本能量
        E_config_t_base[0] = E_zipper_foot
        E_config_t_base[1] = E_shear_foot
        E_config_c_base[0] = E_zipper_foot
        E_config_c_base[1] = E_shear_foot

        # 2.2 定义一个辅助函数来计算双足构象的能量 (逻辑与MATLAB中的for循环一致)
        def calculate_double_feet_energy(track_distance, E_foot1, E_foot2):
            E_state_min_t, f_state_min_t = 1000.0, 0.0
            E_state_min_c, f_state_min_c = 1000.0, 0.0

            for i in range(1, n_hairpin_1 + n_hairpin_2 + 1):
                n_hairpin_open = i
                if n_hairpin_open < n_hairpin_1:
                    x_denominator = n_hairpin_open * 2 * lc_s
                    n_chain = n_hairpin_open
                elif n_hairpin_1 <= n_hairpin_open < n_hairpin_1 + n_hairpin_2:
                    x_denominator = (n_hairpin_open + n_T_hairpin_1) * 2 * lc_s
                    n_chain = n_hairpin_open + n_T_hairpin_1
                else:
                    x_denominator = (n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2) * 2 * lc_s
                    n_chain = n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2

                if abs(x_denominator) < 1e-9:
                    continue

                x = track_distance / x_denominator

                if 0 <= x < 1:
                    try:
                        E_neck = 2 * (n_chain * 2 * lc_s / lp_s) * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                        f_state = 2 * kBT / lp_s * (x - 0.25 + 0.25 / ((1 - x) ** 2))
                    except (ValueError, ZeroDivisionError):
                        E_neck, f_state = 1000.0, 1000.0
                else:
                    E_neck, f_state = 1000.0, 1000.0

                E_state_t = E_neck + E_foot1 + E_foot2 - n_hairpin_open * E_b_azo_trans
                E_state_c = E_neck + E_foot1 + E_foot2 - n_hairpin_open * E_b_azo_cis

                if E_state_min_t > E_state_t:
                    E_state_min_t = E_state_t
                    f_state_min_t = f_state
                if E_state_min_c > E_state_c:
                    E_state_min_c = E_state_c
                    f_state_min_c = f_state

            return E_state_min_t, f_state_min_t, E_state_min_c, f_state_min_c

        # 2.3 计算状态 3, 4, 5, 6 的基本能量
        track_dist_3 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[2], f_config_t_base[2], E_config_c_base[2], f_config_c_base[2] = calculate_double_feet_energy(
            track_dist_3, E_zipper_foot, E_zipper_foot)

        track_dist_4 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[3], f_config_t_base[3], E_config_c_base[3], f_config_c_base[3] = calculate_double_feet_energy(
            track_dist_4, E_shear_foot, E_shear_foot)

        track_dist_5 = (n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[4], f_config_t_base[4], E_config_c_base[4], f_config_c_base[4] = calculate_double_feet_energy(
            track_dist_5, E_zipper_foot, E_shear_foot)

        track_dist_6 = (2 * n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[5], f_config_t_base[5], E_config_c_base[5], f_config_c_base[5] = calculate_double_feet_energy(
            track_dist_6, E_zipper_foot, E_shear_foot)

        # --- 3. 将6个基本能量映射到14个最终状态 ---
        E_config_t_final = np.zeros(self.num_configs)
        E_config_c_final = np.zeros(self.num_configs)
        f_config_t_final = np.zeros(self.num_configs)
        f_config_c_final = np.zeros(self.num_configs)

        # 能量映射
        map_indices = [(0, 3, 0), (3, 6, 1), (6, 8, 2), (8, 10, 3), (10, 12, 4), (12, 14, 5)]
        for start, end, base_idx in map_indices:
            E_config_t_final[start:end] = E_config_t_base[base_idx]
            E_config_c_final[start:end] = E_config_c_base[base_idx]
            f_config_t_final[start:end] = f_config_t_base[base_idx]
            f_config_c_final[start:end] = f_config_c_base[base_idx]

        # FIX: Removed the incorrect lines that overwrote cis-state energies and forces with trans-state values.
        # The mapping above now correctly handles both trans and cis states independently.

        # --- 4. 应用 dE_TYE 能量偏移 ---
        offset_indices = [0, 3, 6, 8, 10, 12]
        E_config_t_final[offset_indices] += dE_TYE
        E_config_c_final[offset_indices] += dE_TYE

        # --- 5. 清理最终结果，防止非法值 ---
        E_config_t_final = self._sanitize_array(E_config_t_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        E_config_c_final = self._sanitize_array(E_config_c_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        f_config_t_final = self._sanitize_array(f_config_t_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)
        f_config_c_final = self._sanitize_array(f_config_c_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)

        return E_config_t_final, f_config_t_final, E_config_c_final, f_config_c_final

    def _calculate_transition_rates(self, E_config_t, f_config_t, E_config_c, f_config_c):
        """
        Calculates the 14x14 transition rate matrices...
        """
        if self.parameters is None:
            raise ValueError("Parameters not set.")

        # --- 1. 安全地获取所有需要的动力学参数 ---
        p = self.parameters
        k0 = self._safe_float(p.get("k0", 0.000008), min_val=1e-12)
        k_mig = self._safe_float(p.get("k_mig", 0.05), min_val=0.0)
        drt_z = self._safe_float(p.get("drt_z", 0.5), min_val=1e-12)
        drt_s = self._safe_float(p.get("drt_s", 0.05), min_val=1e-12)
        kBT = self._safe_float(p.get("kBT", 4.14), min_val=1e-12)

        k_trans = np.zeros((self.num_configs, self.num_configs), dtype=np.float64)
        k_cis = np.zeros((self.num_configs, self.num_configs), dtype=np.float64)

        def safe_exp(val):
            return math.exp(np.clip(val, -100, 100))

        # FIX: Rewrote the entire transition rate calculation to be a 1-to-1 match with the MATLAB source,
        # ensuring all transitions are correctly implemented.
        try:
            # Helper function to populate both trans and cis matrices
            def populate_rates(k_matrix, E_config, f_config):
                # single-single
                k_matrix[3, 0] = k_mig;
                k_matrix[4, 1] = k_mig;
                k_matrix[5, 2] = k_mig
                k_matrix[0, 3] = k_matrix[3, 0] * safe_exp((E_config[0] - E_config[3]) / kBT)
                k_matrix[1, 4] = k_matrix[4, 1] * safe_exp((E_config[1] - E_config[4]) / kBT)
                k_matrix[2, 5] = k_matrix[5, 2] * safe_exp((E_config[2] - E_config[5]) / kBT)

                # single-double
                k_matrix[6, 0] = k0 * safe_exp(f_config[6] * drt_z / kBT);
                k_matrix[10, 0] = k0 * safe_exp(f_config[10] * drt_s / kBT)
                k_matrix[0, 6] = k_matrix[6, 0] * safe_exp((E_config[0] - E_config[6]) / kBT)
                k_matrix[0, 10] = k_matrix[10, 0] * safe_exp((E_config[0] - E_config[10]) / kBT)

                k_matrix[6, 1] = k0 * safe_exp(f_config[6] * drt_z / kBT);
                k_matrix[7, 1] = k0 * safe_exp(f_config[7] * drt_z / kBT)
                k_matrix[11, 1] = k0 * safe_exp(f_config[11] * drt_s / kBT);
                k_matrix[12, 1] = k0 * safe_exp(f_config[12] * drt_s / kBT)
                k_matrix[1, 6] = k_matrix[6, 1] * safe_exp((E_config[1] - E_config[6]) / kBT)
                k_matrix[1, 7] = k_matrix[7, 1] * safe_exp((E_config[1] - E_config[7]) / kBT)
                k_matrix[1, 11] = k_matrix[11, 1] * safe_exp((E_config[1] - E_config[11]) / kBT)
                k_matrix[1, 12] = k_matrix[12, 1] * safe_exp((E_config[1] - E_config[12]) / kBT)

                k_matrix[7, 2] = k0 * safe_exp(f_config[7] * drt_z / kBT);
                k_matrix[13, 2] = k0 * safe_exp(f_config[13] * drt_s / kBT)
                k_matrix[2, 7] = k_matrix[7, 2] * safe_exp((E_config[2] - E_config[7]) / kBT)
                k_matrix[2, 13] = k_matrix[13, 2] * safe_exp((E_config[2] - E_config[13]) / kBT)

                k_matrix[8, 3] = k0 * safe_exp(f_config[8] * drt_s / kBT);
                k_matrix[12, 3] = k0 * safe_exp(f_config[12] * drt_z / kBT)
                k_matrix[3, 8] = k_matrix[8, 3] * safe_exp((E_config[3] - E_config[8]) / kBT)
                k_matrix[3, 12] = k_matrix[12, 3] * safe_exp((E_config[3] - E_config[12]) / kBT)

                k_matrix[8, 4] = k0 * safe_exp(f_config[8] * drt_s / kBT);
                k_matrix[9, 4] = k0 * safe_exp(f_config[9] * drt_s / kBT)
                k_matrix[10, 4] = k0 * safe_exp(f_config[10] * drt_z / kBT);
                k_matrix[13, 4] = k0 * safe_exp(f_config[13] * drt_z / kBT)
                k_matrix[4, 8] = k_matrix[8, 4] * safe_exp((E_config[4] - E_config[8]) / kBT)
                k_matrix[4, 9] = k_matrix[9, 4] * safe_exp((E_config[4] - E_config[9]) / kBT)
                k_matrix[4, 10] = k_matrix[10, 4] * safe_exp((E_config[4] - E_config[10]) / kBT)
                k_matrix[4, 13] = k_matrix[13, 4] * safe_exp((E_config[4] - E_config[13]) / kBT)

                k_matrix[9, 5] = k0 * safe_exp(f_config[9] * drt_s / kBT);
                k_matrix[11, 5] = k0 * safe_exp(f_config[11] * drt_z / kBT)
                k_matrix[5, 9] = k_matrix[9, 5] * safe_exp((E_config[5] - E_config[9]) / kBT)
                k_matrix[5, 11] = k_matrix[11, 5] * safe_exp((E_config[5] - E_config[11]) / kBT)

                # double-double
                k_matrix[6, 10] = k_mig;
                k_matrix[12, 6] = k_mig
                k_matrix[10, 6] = k_matrix[6, 10] * safe_exp((E_config[10] - E_config[6]) / kBT)
                k_matrix[6, 12] = k_matrix[12, 6] * safe_exp((E_config[6] - E_config[12]) / kBT)

                k_matrix[7, 11] = k_mig;
                k_matrix[13, 7] = k_mig
                k_matrix[11, 7] = k_matrix[7, 11] * safe_exp((E_config[11] - E_config[7]) / kBT)
                k_matrix[7, 13] = k_matrix[13, 7] * safe_exp((E_config[7] - E_config[13]) / kBT)

                k_matrix[8, 10] = k_mig;
                k_matrix[12, 8] = k_mig
                k_matrix[10, 8] = k_matrix[8, 10] * safe_exp((E_config[10] - E_config[8]) / kBT)
                k_matrix[8, 12] = k_matrix[12, 8] * safe_exp((E_config[8] - E_config[12]) / kBT)

                k_matrix[9, 11] = k_mig;
                k_matrix[13, 9] = k_mig
                k_matrix[11, 9] = k_matrix[9, 11] * safe_exp((E_config[11] - E_config[9]) / kBT)
                k_matrix[9, 13] = k_matrix[13, 9] * safe_exp((E_config[9] - E_config[13]) / kBT)

            # Populate for trans and cis states
            populate_rates(k_trans, E_config_t, f_config_t)
            populate_rates(k_cis, E_config_c, f_config_c)

        except Exception as e:
            print(f"An error occurred during transition rate calculation: {e}")
            return np.zeros_like(k_trans), np.zeros_like(k_cis)

        return k_trans, k_cis

    def _ode_system(self, t, P, k_matrix, k_photo, light_on):
        """
        Defines the system of ordinary differential equations (ODEs).
        """
        dP_dt = np.zeros(self.num_configs)
        for i in range(self.num_configs):
            sum_val = 0
            for j in range(self.num_configs):
                if i != j:
                    sum_val += k_matrix[j, i] * P[j] - k_matrix[i, j] * P[i]
            dP_dt[i] = sum_val

        if light_on:
            # This part is a simplification. A more complex model might have state-specific photo-rates.
            # Assuming k_photo represents a global decay/transition rate affecting all states under light.
            pass  # Currently, k_photo is not used as per the ODE system structure.

        return dP_dt

    def run_simulation(self, P0, total_sim_time, light_schedule):
        """
        Runs the full simulation based on a flexible light schedule.

        Args:
            P0 (np.array): Initial probability distribution of the states.
            total_sim_time (float): The total duration of the simulation.
            light_schedule (list): A list of tuples, e.g., [(10, 'visible'), (20, 'uv'), ...],
                                     defining the end time and condition for each phase.
        """
        E_config_t, f_config_t, E_config_c, f_config_c = self._calculate_free_energies()
        k_trans, k_cis = self._calculate_transition_rates(E_config_t, f_config_t, E_config_c, f_config_c)
        k_photo = self._safe_float(self.parameters.get('k_photo', 0.0))

        current_P = np.array(P0, dtype=np.float64)
        current_time = 0.0

        all_times = [np.array([current_time])]
        all_probs = [current_P.reshape(-1, 1)]

        for end_time, light_condition in light_schedule:
            if current_time >= total_sim_time:
                break

            segment_end_time = min(end_time, total_sim_time)

            if segment_end_time <= current_time:
                continue

            if light_condition.lower() == 'uv':
                k_matrix = k_cis
                is_light_on = True
            else:
                k_matrix = k_trans
                is_light_on = False

            sol = solve_ivp(
                lambda t, P: self._ode_system(t, P, k_matrix, k_photo, is_light_on),
                (current_time, segment_end_time),
                current_P,
                method='RK45', dense_output=True, rtol=1e-6, atol=1e-9
            )

            if sol.success and len(sol.t) > 1:
                all_times.append(sol.t[1:])
                all_probs.append(sol.y[:, 1:])
                current_time = sol.t[-1]
                current_P = sol.y[:, -1]

        t_combined = np.concatenate(all_times)
        P_combined = np.hstack(all_probs)

        sim_df = pd.DataFrame(P_combined.T, columns=[f'P_{i}' for i in range(self.num_configs)])
        sim_df['Time'] = t_combined
        sim_df = sim_df[['Time'] + [f'P_{i}' for i in range(self.num_configs)]]

        return sim_df

    def evaluate_model(self, simulated_data_df, reward_flag=0):
        if reward_flag == 0:
            if self.experimental_data_a is None:
                print("Error: Experimental dataset 'a' failed to load. Cannot calculate reward.")
                return -1000.0

            p_unbind_track = self._safe_float(self.parameters.get('p_unbind_track', 0.09507))

            datasets = {'a': self.experimental_data_a}
            total_nmse = 0
            num_signals = 0

            for name, exp_df in datasets.items():
                if exp_df is None:
                    continue

                try:
                    exp_time = exp_df['Time'].values
                    exp_fam = exp_df['FAM/FAM T (+)'].values
                    exp_tye = exp_df['TYE/TYE T (-)'].values
                    exp_cy5 = exp_df['CY5/CY5 T (m)'].values

                    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
                    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

                    if len(exp_time) == 0:
                        continue

                    sim_time = simulated_data_df['Time'].values
                    if len(sim_time) < 2: return -1000.0

                    sim_fam = (simulated_data_df['P_0'] + simulated_data_df['P_1'] + simulated_data_df['P_3'] +
                               simulated_data_df['P_4'] + simulated_data_df['P_6'] + simulated_data_df['P_8'] +
                               simulated_data_df['P_10'] + simulated_data_df['P_12']).values
                    sim_tye = (simulated_data_df['P_1'] + simulated_data_df['P_2'] + simulated_data_df['P_4'] +
                               simulated_data_df['P_5'] + simulated_data_df['P_7'] + simulated_data_df['P_9'] +
                               simulated_data_df['P_11'] + simulated_data_df['P_13']).values
                    sim_cy5 = (simulated_data_df['P_0'] + simulated_data_df['P_2'] + simulated_data_df['P_3'] +
                               simulated_data_df['P_5']).values + p_unbind_track

                    interp_fam = interp1d(sim_time, self._sanitize_array(sim_fam), kind='linear',
                                          fill_value='extrapolate')(exp_time)
                    interp_tye = interp1d(sim_time, self._sanitize_array(sim_tye), kind='linear',
                                          fill_value='extrapolate')(exp_time)
                    interp_cy5 = interp1d(sim_time, self._sanitize_array(sim_cy5), kind='linear',
                                          fill_value='extrapolate')(exp_time)

                    mse_fam = mean_squared_error(exp_fam, interp_fam)
                    mse_tye = mean_squared_error(exp_tye, interp_tye)
                    mse_cy5 = mean_squared_error(exp_cy5, interp_cy5)

                    var_fam = np.var(exp_fam) + 1e-9
                    var_tye = np.var(exp_tye) + 1e-9
                    var_cy5 = np.var(exp_cy5) + 1e-9

                    nmse_fam = mse_fam / var_fam
                    nmse_tye = mse_tye / var_tye
                    nmse_cy5 = mse_cy5 / var_cy5

                    total_nmse += (nmse_fam + nmse_tye + nmse_cy5)
                    num_signals += 3

                except Exception as e:
                    print(f"An unexpected error during evaluation of dataset '{name}': {e}")
                    return -1000.0

            if num_signals == 0:
                return -1000.0

            average_nmse = total_nmse / num_signals
            reward = -average_nmse

            if not np.isfinite(reward):
                return -1000.0
            return float(reward)

        elif reward_flag == 1:
            try:
                last_state = simulated_data_df.iloc[-1]
                reward = float(last_state['P_12'] + last_state['P_13'])
                if not np.isfinite(reward):
                    return -1000.0
                return reward
            except (IndexError, KeyError) as e:
                print(f"Error during reward_flag=1 evaluation: {e}")
                return -1000.0

        else:
            # FIX: Changed the return value for an unsupported flag to a large negative number to provide a clear failure signal.
            print(f"Warning: Unsupported reward_flag value: {reward_flag}")
            return -1000.0