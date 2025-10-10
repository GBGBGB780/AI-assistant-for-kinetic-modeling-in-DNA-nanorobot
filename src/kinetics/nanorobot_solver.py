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
    def __init__(self, model_name, config_names_str, experimental_data_path_a, experimental_data_path_b):
        self.model_name = model_name
        self.num_configs = 14  # 核心修改：统一为14个状态
        self.config_names = [f"State_{i}" for i in range(self.num_configs)]

        self.experimental_data_a = self._load_experimental_data(experimental_data_path_a)
        self.experimental_data_b = self._load_experimental_data(experimental_data_path_b)
        self.parameters = None

        self.default_mechanics_params = {
            'kBT': 4.14, 'lp_s': 0.75, 'lc_s': 0.7, 'lc_d': 0.34, 'E_b': -1.2,
            'E_b_azo_trans': -1.0, 'E_b_azo_cis': -0.1, 'di_DNA': 2, 'n_D1': 10,
            'n_D2': 10, 'n_S1': 4, 'n_gray': 10, 'n_hairpin_1': 8, 'n_hairpin_2': 8,
            'n_azo_1': 3, 'n_azo_2': 3, 'n_T_hairpin_1': 3, 'n_T_hairpin_2': 2,
            'n_track_1': 15, 'n_track_2': 55
        }

        self.default_kinetics_params = {
            'k0': 0.000008, 'k_mig': 0.05, 'drt_z': 0.5, 'drt_s': 0.05,
            'dE_TYE': -1.55, 'p_unbind_track': 0.09507
        }

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
        self.parameters = {}
        self.parameters.update(self.default_mechanics_params)
        self.parameters.update(self.default_kinetics_params)
        if params_dict:
            for k, v in params_dict.items():
                self.parameters[k] = v

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
        E_config_t_base[0] = E_zipper_foot  # 对应MATLAB中的E_config_t(1)
        E_config_t_base[1] = E_shear_foot  # 对应MATLAB中的E_config_t(2)
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
                else:  # n_hairpin_open >= n_hairpin_1 + n_hairpin_2
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
        # State 3
        track_dist_3 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[2], f_config_t_base[2], E_config_c_base[2], f_config_c_base[2] = calculate_double_feet_energy(
            track_dist_3, E_zipper_foot, E_zipper_foot)

        # State 4
        track_dist_4 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[3], f_config_t_base[3], E_config_c_base[3], f_config_c_base[3] = calculate_double_feet_energy(
            track_dist_4, E_shear_foot, E_shear_foot)

        # State 5
        track_dist_5 = (n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[4], f_config_t_base[4], E_config_c_base[4], f_config_c_base[4] = calculate_double_feet_energy(
            track_dist_5, E_zipper_foot, E_shear_foot)

        # State 6
        track_dist_6 = (2 * n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[5], f_config_t_base[5], E_config_c_base[5], f_config_c_base[5] = calculate_double_feet_energy(
            track_dist_6, E_zipper_foot, E_shear_foot)

        # --- 3. 将6个基本能量映射到14个最终状态 ---
        E_config_t_final = np.zeros(self.num_configs)
        E_config_c_final = np.zeros(self.num_configs)
        f_config_t_final = np.zeros(self.num_configs)
        f_config_c_final = np.zeros(self.num_configs)

        # 能量映射
        E_config_t_final[0:3] = E_config_t_base[0]  # States 1,2,3 use base energy 1
        E_config_t_final[3:6] = E_config_t_base[1]  # States 4,5,6 use base energy 2
        E_config_t_final[6:8] = E_config_t_base[2]  # States 7,8 use base energy 3
        E_config_t_final[8:10] = E_config_t_base[3]  # States 9,10 use base energy 4
        E_config_t_final[10:12] = E_config_t_base[4]  # States 11,12 use base energy 5
        E_config_t_final[12:14] = E_config_t_base[5]  # States 13,14 use base energy 6

        E_config_c_final[:] = E_config_t_final  # Cis 能量映射与 Trans 相同

        # 力映射
        f_config_t_final[0:3] = f_config_t_base[0]
        f_config_t_final[3:6] = f_config_t_base[1]
        f_config_t_final[6:8] = f_config_t_base[2]
        f_config_t_final[8:10] = f_config_t_base[3]
        f_config_t_final[10:12] = f_config_t_base[4]
        f_config_t_final[12:14] = f_config_t_base[5]

        f_config_c_final[:] = f_config_t_final  # Cis 力映射与 Trans 相同

        # --- 4. 应用 dE_TYE 能量偏移 ---
        offset_indices = [0, 3, 6, 8, 10, 12]  # MATLAB 的 1,4,7,9,11,13 对应 Python 的 0-based index
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
        Calculates the 14x14 transition rate matrices (k_trans, k_cis)
        and the final ODE matrices (K_matrix_trans, K_matrix_cis),
        perfectly matching the logic from the MATLAB script.
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

        k_trans = np.zeros((self.num_configs, self.num_configs))
        k_cis = np.zeros((self.num_configs, self.num_configs))

        # 定义一个安全的指数函数防止计算溢出
        def safe_exp(val):
            # The value is E_diff / kBT. E_diff is typically within -20 to 20 kBT.
            # Clipping at a larger range like -100 to 100 is safe.
            return math.exp(np.clip(val, -100, 100))

        try:
            # =====================================================================
            #            完整翻译 MATLAB 的 k_trans 矩阵计算
            #  注意: MATLAB k_trans(i, j) 对应 Python k_trans[i-1, j-1]
            # =====================================================================

            # --- single-single transitions ---
            k_trans[3, 0] = k_mig  # k_trans(4,1)
            k_trans[4, 1] = k_mig  # k_trans(5,2)
            k_trans[5, 2] = k_mig  # k_trans(6,3)
            k_trans[0, 3] = k_trans[3, 0] * safe_exp((E_config_t[0] - E_config_t[3]) / kBT)
            k_trans[1, 4] = k_trans[4, 1] * safe_exp((E_config_t[1] - E_config_t[4]) / kBT)
            k_trans[2, 5] = k_trans[5, 2] * safe_exp((E_config_t[2] - E_config_t[5]) / kBT)

            # --- single-double transitions ---
            k_trans[6, 0] = k0 * safe_exp(f_config_t[6] * drt_z / kBT)  # k_trans(7,1)
            k_trans[10, 0] = k0 * safe_exp(f_config_t[10] * drt_s / kBT)  # k_trans(11,1)
            k_trans[0, 6] = k_trans[6, 0] * safe_exp((E_config_t[0] - E_config_t[6]) / kBT)
            k_trans[0, 10] = k_trans[10, 0] * safe_exp((E_config_t[0] - E_config_t[10]) / kBT)

            k_trans[6, 1] = k0 * safe_exp(f_config_t[6] * drt_z / kBT)  # k_trans(7,2)
            k_trans[7, 1] = k0 * safe_exp(f_config_t[7] * drt_z / kBT)  # k_trans(8,2)
            k_trans[11, 1] = k0 * safe_exp(f_config_t[11] * drt_s / kBT)  # k_trans(12,2)
            k_trans[12, 1] = k0 * safe_exp(f_config_t[12] * drt_s / kBT)  # k_trans(13,2)
            k_trans[1, 6] = k_trans[6, 1] * safe_exp((E_config_t[1] - E_config_t[6]) / kBT)
            k_trans[1, 7] = k_trans[7, 1] * safe_exp((E_config_t[1] - E_config_t[7]) / kBT)
            k_trans[1, 11] = k_trans[11, 1] * safe_exp((E_config_t[1] - E_config_t[11]) / kBT)
            k_trans[1, 12] = k_trans[12, 1] * safe_exp((E_config_t[1] - E_config_t[12]) / kBT)

            k_trans[7, 2] = k0 * safe_exp(f_config_t[7] * drt_z / kBT)  # k_trans(8,3)
            k_trans[13, 2] = k0 * safe_exp(f_config_t[13] * drt_s / kBT)  # k_trans(14,3)
            k_trans[2, 7] = k_trans[7, 2] * safe_exp((E_config_t[2] - E_config_t[7]) / kBT)
            k_trans[2, 13] = k_trans[13, 2] * safe_exp((E_config_t[2] - E_config_t[13]) / kBT)

            k_trans[8, 3] = k0 * safe_exp(f_config_t[8] * drt_s / kBT)  # k_trans(9,4)
            k_trans[12, 3] = k0 * safe_exp(f_config_t[12] * drt_z / kBT)  # k_trans(13,4)
            k_trans[3, 8] = k_trans[8, 3] * safe_exp((E_config_t[3] - E_config_t[8]) / kBT)
            k_trans[3, 12] = k_trans[12, 3] * safe_exp((E_config_t[3] - E_config_t[12]) / kBT)

            k_trans[8, 4] = k0 * safe_exp(f_config_t[8] * drt_s / kBT)  # k_trans(9,5)
            k_trans[9, 4] = k0 * safe_exp(f_config_t[9] * drt_s / kBT)  # k_trans(10,5)
            k_trans[10, 4] = k0 * safe_exp(f_config_t[10] * drt_z / kBT)  # k_trans(11,5)
            k_trans[13, 4] = k0 * safe_exp(f_config_t[13] * drt_z / kBT)  # k_trans(14,5)
            k_trans[4, 8] = k_trans[8, 4] * safe_exp((E_config_t[4] - E_config_t[8]) / kBT)
            k_trans[4, 9] = k_trans[9, 4] * safe_exp((E_config_t[4] - E_config_t[9]) / kBT)
            k_trans[4, 10] = k_trans[10, 4] * safe_exp((E_config_t[4] - E_config_t[10]) / kBT)
            k_trans[4, 13] = k_trans[13, 4] * safe_exp((E_config_t[4] - E_config_t[13]) / kBT)

            k_trans[9, 5] = k0 * safe_exp(f_config_t[9] * drt_s / kBT)  # k_trans(10,6)
            k_trans[11, 5] = k0 * safe_exp(f_config_t[11] * drt_z / kBT)  # k_trans(12,6)
            k_trans[5, 9] = k_trans[9, 5] * safe_exp((E_config_t[5] - E_config_t[9]) / kBT)
            k_trans[5, 11] = k_trans[11, 5] * safe_exp((E_config_t[5] - E_config_t[11]) / kBT)

            # --- double-double transitions ---
            k_trans[6, 10] = k_mig;
            k_trans[12, 6] = k_mig  # k_trans(7,11), k_trans(13,7)
            k_trans[10, 6] = k_trans[6, 10] * safe_exp((E_config_t[10] - E_config_t[6]) / kBT)
            k_trans[6, 12] = k_trans[12, 6] * safe_exp((E_config_t[6] - E_config_t[12]) / kBT)

            k_trans[7, 11] = k_mig;
            k_trans[13, 7] = k_mig  # k_trans(8,12), k_trans(14,8)
            k_trans[11, 7] = k_trans[7, 11] * safe_exp((E_config_t[11] - E_config_t[7]) / kBT)
            k_trans[7, 13] = k_trans[13, 7] * safe_exp((E_config_t[7] - E_config_t[13]) / kBT)

            k_trans[8, 10] = k_mig;
            k_trans[12, 8] = k_mig  # k_trans(9,11), k_trans(13,9)
            k_trans[10, 8] = k_trans[8, 10] * safe_exp((E_config_t[10] - E_config_t[8]) / kBT)
            k_trans[8, 12] = k_trans[12, 8] * safe_exp((E_config_t[8] - E_config_t[12]) / kBT)

            k_trans[9, 11] = k_mig;
            k_trans[13, 9] = k_mig  # k_trans(10,12), k_trans(14,10)
            k_trans[11, 9] = k_trans[9, 11] * safe_exp((E_config_t[11] - E_config_t[9]) / kBT)
            k_trans[9, 13] = k_trans[13, 9] * safe_exp((E_config_t[9] - E_config_t[13]) / kBT)

            # =====================================================================
            #            完整翻译 MATLAB 的 k_cis 矩阵计算
            # =====================================================================

            # --- single-single transitions ---
            k_cis[3, 0] = k_mig;
            k_cis[4, 1] = k_mig;
            k_cis[5, 2] = k_mig
            k_cis[0, 3] = k_cis[3, 0] * safe_exp((E_config_c[0] - E_config_c[3]) / kBT)
            k_cis[1, 4] = k_cis[4, 1] * safe_exp((E_config_c[1] - E_config_c[4]) / kBT)
            k_cis[2, 5] = k_cis[5, 2] * safe_exp((E_config_c[2] - E_config_c[5]) / kBT)

            # --- single-double transitions ---
            k_cis[6, 0] = k0 * safe_exp(f_config_c[6] * drt_z / kBT)
            k_cis[10, 0] = k0 * safe_exp(f_config_c[10] * drt_s / kBT)
            k_cis[0, 6] = k_cis[6, 0] * safe_exp((E_config_c[0] - E_config_c[6]) / kBT)
            k_cis[0, 10] = k_cis[10, 0] * safe_exp((E_config_c[0] - E_config_c[10]) / kBT)

            k_cis[6, 1] = k0 * safe_exp(f_config_c[6] * drt_z / kBT)
            k_cis[7, 1] = k0 * safe_exp(f_config_c[7] * drt_z / kBT)
            k_cis[11, 1] = k0 * safe_exp(f_config_c[11] * drt_s / kBT)
            k_cis[12, 1] = k0 * safe_exp(f_config_c[12] * drt_s / kBT)
            k_cis[1, 6] = k_cis[6, 1] * safe_exp((E_config_c[1] - E_config_c[6]) / kBT)
            k_cis[1, 7] = k_cis[7, 1] * safe_exp((E_config_c[1] - E_config_c[7]) / kBT)
            k_cis[1, 11] = k_cis[11, 1] * safe_exp((E_config_c[1] - E_config_c[11]) / kBT)
            k_cis[1, 12] = k_cis[12, 1] * safe_exp((E_config_c[1] - E_config_c[12]) / kBT)

            k_cis[7, 2] = k0 * safe_exp(f_config_c[7] * drt_z / kBT)
            k_cis[13, 2] = k0 * safe_exp(f_config_c[13] * drt_s / kBT)
            k_cis[2, 7] = k_cis[7, 2] * safe_exp((E_config_c[2] - E_config_c[7]) / kBT)
            k_cis[2, 13] = k_cis[13, 2] * safe_exp((E_config_c[2] - E_config_c[13]) / kBT)

            k_cis[8, 3] = k0 * safe_exp(f_config_c[8] * drt_s / kBT)
            k_cis[12, 3] = k0 * safe_exp(f_config_c[12] * drt_z / kBT)
            k_cis[3, 8] = k_cis[8, 3] * safe_exp((E_config_c[3] - E_config_c[8]) / kBT)
            k_cis[3, 12] = k_cis[12, 3] * safe_exp((E_config_c[3] - E_config_c[12]) / kBT)

            k_cis[8, 4] = k0 * safe_exp(f_config_c[8] * drt_s / kBT)
            k_cis[9, 4] = k0 * safe_exp(f_config_c[9] * drt_s / kBT)
            k_cis[10, 4] = k0 * safe_exp(f_config_c[10] * drt_z / kBT)
            k_cis[13, 4] = k0 * safe_exp(f_config_c[13] * drt_z / kBT)
            k_cis[4, 8] = k_cis[8, 4] * safe_exp((E_config_c[4] - E_config_c[8]) / kBT)
            k_cis[4, 9] = k_cis[9, 4] * safe_exp((E_config_c[4] - E_config_c[9]) / kBT)
            k_cis[4, 10] = k_cis[10, 4] * safe_exp((E_config_c[4] - E_config_c[10]) / kBT)
            k_cis[4, 13] = k_cis[13, 4] * safe_exp((E_config_c[4] - E_config_c[13]) / kBT)

            k_cis[9, 5] = k0 * safe_exp(f_config_c[9] * drt_s / kBT)
            k_cis[11, 5] = k0 * safe_exp(f_config_c[11] * drt_z / kBT)
            k_cis[5, 9] = k_cis[9, 5] * safe_exp((E_config_c[5] - E_config_c[9]) / kBT)
            k_cis[5, 11] = k_cis[11, 5] * safe_exp((E_config_c[5] - E_config_c[11]) / kBT)

            # --- double-double transitions ---
            k_cis[6, 10] = k_mig;
            k_cis[12, 6] = k_mig
            k_cis[10, 6] = k_cis[6, 10] * safe_exp((E_config_c[10] - E_config_c[6]) / kBT)
            k_cis[6, 12] = k_cis[12, 6] * safe_exp((E_config_c[6] - E_config_c[12]) / kBT)

            k_cis[7, 11] = k_mig;
            k_cis[13, 7] = k_mig
            k_cis[11, 7] = k_cis[7, 11] * safe_exp((E_config_c[11] - E_config_c[7]) / kBT)
            k_cis[7, 13] = k_cis[13, 7] * safe_exp((E_config_c[7] - E_config_c[13]) / kBT)

            k_cis[8, 10] = k_mig;
            k_cis[12, 8] = k_mig
            k_cis[10, 8] = k_cis[8, 10] * safe_exp((E_config_c[10] - E_config_c[8]) / kBT)
            k_cis[8, 12] = k_cis[12, 8] * safe_exp((E_config_c[8] - E_config_c[12]) / kBT)

            k_cis[9, 11] = k_mig;
            k_cis[13, 9] = k_mig
            k_cis[11, 9] = k_cis[9, 11] * safe_exp((E_config_c[11] - E_config_c[9]) / kBT)
            k_cis[9, 13] = k_cis[13, 9] * safe_exp((E_config_c[9] - E_config_c[13]) / kBT)

        except OverflowError:
            # In case np.clip is not enough, this provides a fallback
            print("Warning: Overflow in rate calculation. Some rates might be incorrect.")

        # --- 最后，构建用于ODE求解器的矩阵 K (也称为 A 或 Q 矩阵) ---
        # 主方程 dP/dt = K * P, 其中:
        # K[i, j] = k_{j, i}  (j -> i 的速率)
        # K[i, i] = - sum_l(k_{i, l}) (从 i -> 所有 l 的速率总和的负值)
        K_matrix_trans = np.zeros((self.num_configs, self.num_configs))
        K_matrix_cis = np.zeros((self.num_configs, self.num_configs))

        # 填充非对角线元素 (注意转置：K的(i,j)元素是k矩阵的(j,i)元素)
        for i in range(self.num_configs):
            for j in range(self.num_configs):
                if i != j:
                    K_matrix_trans[i, j] = k_trans[j, i]
                    K_matrix_cis[i, j] = k_cis[j, i]

        # 填充对角线元素 (离出速率总和的负值)
        for i in range(self.num_configs):
            K_matrix_trans[i, i] = -np.sum(k_trans[i, :])
            K_matrix_cis[i, i] = -np.sum(k_cis[i, :])

        # 清理最终结果，防止非法值
        K_matrix_trans = self._sanitize_array(K_matrix_trans, nan_replacement=0.0)
        K_matrix_cis = self._sanitize_array(K_matrix_cis, nan_replacement=0.0)

        return K_matrix_trans, K_matrix_cis

    def _ode_system(self, t, P, k_matrix):
        return np.dot(k_matrix, P)

    def simulate(self, initial_P, sim_time_step, sim_total_time, light_schedule=None):
        """
        Runs the full kinetic simulation over time.

        Args:
            initial_P (list or np.array): A 14-element vector of initial state probabilities.
            sim_time_step (float): The time step for the output data (in seconds).
            sim_total_time (float): The total simulation time (in seconds).
            light_schedule (list, optional): A list of tuples defining the light sequence,
                                             e.g., [(600, 'vis'), (1200, 'uv')].
                                             If None, defaults to 10-min alternating light.

        Returns:
            pd.DataFrame: A DataFrame containing the time evolution of the 14 state probabilities.
        """
        # 1. 首先计算出模型所需的能量和速率矩阵
        E_config_t, f_config_t, E_config_c, f_config_c = self._calculate_free_energies()
        k_trans_matrix, k_cis_matrix = self._calculate_transition_rates(E_config_t, f_config_t, E_config_c, f_config_c)

        # 2. 验证并设置初始概率分布
        current_P = np.array(initial_P, dtype=float)
        if current_P.size != self.num_configs or abs(np.sum(current_P) - 1.0) > 1e-6:
            print(f"Warning: Invalid initial_P provided. It must be a {self.num_configs}-element array summing to 1.")
            print("Defaulting to 100% probability in State 0.")
            current_P = np.zeros(self.num_configs, dtype=float)
            current_P[0] = 1.0

        # 3. 如果未提供光照方案，则创建与MATLAB一致的默认方案
        if light_schedule is None:
            light_schedule = []
            interval = 600  # 10 minutes in seconds
            num_intervals = math.ceil(sim_total_time / interval)
            for i in range(num_intervals):
                end_time = (i + 1) * interval
                light_type = 'vis' if (i % 2) == 0 else 'uv'
                light_schedule.append((min(end_time, sim_total_time), light_type))

        # 4. 分段进行ODE求解
        all_results = []
        current_time = 0.0

        for segment_end_time, light in light_schedule:
            if current_time >= sim_total_time:
                break

            t_start = current_time
            # 确保段的结束时间不超过总模拟时间
            t_end = min(segment_end_time, sim_total_time)

            # 根据光照类型选择正确的速率矩阵
            k_matrix = k_trans_matrix if light.lower() == 'vis' else k_cis_matrix

            # 定义该段的求解时间点
            t_eval = np.arange(t_start, t_end, sim_time_step)
            if t_eval.size == 0 or t_eval[-1] < t_end:
                t_eval = np.append(t_eval, t_end)

            # 使用scipy的solve_ivp求解器
            solution = solve_ivp(
                fun=self._ode_system,
                t_span=[t_start, t_end],
                y0=current_P,
                method='Radau',  # 'Radau' 是一种适合刚性系统的求解器
                t_eval=t_eval,
                args=(k_matrix,)
            )

            # 存储该段的结果 (转置 .y 使其 shape 为 [n_points, n_states])
            # 去掉第一个点，除非是第一个段，以避免时间点重复
            result_segment = solution.y.T
            if len(all_results) > 0:
                all_results.append(result_segment[1:, :])
            else:
                all_results.append(result_segment)

            # 更新下一段的初始条件和起始时间
            current_P = solution.y[:, -1]
            current_time = t_end

        # 5. 整合结果并创建DataFrame
        if not all_results:
            print("Warning: Simulation produced no results.")
            return pd.DataFrame(columns=['Time'] + [f'P_{i}' for i in range(self.num_configs)])

        final_P_history = np.vstack(all_results)
        final_time_points = np.arange(0, final_P_history.shape[0] * sim_time_step, sim_time_step)

        # 确保时间和概率矩阵的长度一致
        max_len = min(len(final_time_points), final_P_history.shape[0])

        sim_df = pd.DataFrame(final_P_history[:max_len, :], columns=[f'P_{i}' for i in range(self.num_configs)])
        sim_df['Time'] = final_time_points[:max_len]

        # 将Time列放到第一列
        sim_df = sim_df[['Time'] + [f'P_{i}' for i in range(self.num_configs)]]

        return sim_df

    # ==================== MODIFIED: 全面修改奖励函数以包含归一化 ====================
    def evaluate_model(self, simulated_data_df, reward_flag=0):
        if reward_flag == 0:
            # 初始检查仍然保留，但我们会添加更具体的检查
            if self.experimental_data_a is None and self.experimental_data_b is None:
                print("Error: Both experimental datasets failed to load. Cannot calculate reward.")
                return -1000.0

            p_unbind_track = self._safe_float(self.parameters.get('p_unbind_track', 0.09507))

            datasets = {
                'a': self.experimental_data_a,
                'b': self.experimental_data_b
            }

            total_nmse = 0
            num_signals = 0

            for name, exp_df in datasets.items():
                # ==================== 核心修改：增加对每个数据集的None值检查 ====================
                # 在尝试访问 exp_df 的任何内容之前，先检查它是否为 None
                if exp_df is None:
                    print(f"Warning: Dataset '{name}' was not loaded successfully. Skipping its evaluation.")
                    continue  # 使用 continue 跳过本次循环
                # =================================================================================

                try:
                    # 1. 提取和清理实验数据
                    exp_time = exp_df['Time'].values
                    exp_fam = exp_df['FAM/FAM T (+)'].values
                    exp_tye = exp_df['TYE/TYE T (-)'].values
                    exp_cy5 = exp_df['CY5/CY5 T (m)'].values

                    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
                    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

                    if len(exp_time) == 0:
                        print(f"Warning: No valid data rows in dataset '{name}' after cleaning NaNs.")
                        continue

                    # 2. 信号映射与插值
                    sim_time = simulated_data_df['Time'].values

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

                    # 3. 计算MSE
                    mse_fam = mean_squared_error(exp_fam, interp_fam)
                    mse_tye = mean_squared_error(exp_tye, interp_tye)
                    mse_cy5 = mean_squared_error(exp_cy5, interp_cy5)

                    # 4. 计算方差并进行归一化
                    var_fam = np.var(exp_fam) + 1e-9
                    var_tye = np.var(exp_tye) + 1e-9
                    var_cy5 = np.var(exp_cy5) + 1e-9

                    nmse_fam = mse_fam / var_fam
                    nmse_tye = mse_tye / var_tye
                    nmse_cy5 = mse_cy5 / var_cy5

                    total_nmse += (nmse_fam + nmse_tye + nmse_cy5)
                    num_signals += 3

                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not process dataset '{name}'. Check column names. Error: {e}")
                    return -1000.0
                except Exception as e:
                    print(f"An unexpected error occurred during evaluation of dataset '{name}': {e}")
                    return -1000.0

            if num_signals == 0:
                print("Error: No signals could be processed from any dataset.")
                return -1000.0

            # 5. 计算最终奖励
            average_nmse = total_nmse / num_signals
            reward = -average_nmse

            if not np.isfinite(reward):
                return -1000.0
            return float(reward)
        else:
            return 0.0
