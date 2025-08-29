# coding=gb2312
# kinetics/nanorobot_solver.py

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d  # 用于插值实验数据
from sklearn.metrics import mean_squared_error  # 用于计算拟合误差
import math  # 用于数学函数，如exp


class NanorobotSolver:
    def __init__(self, model_name, config_names_str, experimental_data_path_a, experimental_data_path_b):
        self.model_name = model_name
        self.config_names = [name.strip() for name in config_names_str.split(",")]
        self.num_configs = len(self.config_names)
        self.experimental_data_a = self._load_experimental_data(experimental_data_path_a)
        self.experimental_data_b = self._load_experimental_data(experimental_data_path_b)
        self.parameters = None  # 将由MLP生成并设置，是一个字典

        # 默认值（守护用）
        self.default_mechanics_params = {
            'kBT': 4.14,
            'lp_s': 0.75,
            'lc_s': 0.7,
            'lc_d': 0.34,
            'E_b': -1.2,
            'E_b_azo_trans': -1.0,
            'E_b_azo_cis': -0.1,
            'di_DNA': 2,
            'n_D1': 10,
            'n_D2': 10,
            'n_S1': 4,
            'n_gray': 10,
            'n_hairpin_1': 8,
            'n_hairpin_2': 8,
            'n_azo_1': 3,
            'n_azo_2': 3,
            'n_T_hairpin_1': 3,
            'n_T_hairpin_2': 2,
            'n_track_1': 15,
            'n_track_2': 55
        }

        self.default_kinetics_params = {
            'k0': 0.000008,
            'k_mig': 0.05,
            'drt_z': 0.5,
            'drt_s': 0.05,
            'dE_TYE': -1.55
        }

        self.transition_matrix_template = self._define_transition_matrix_template()

    # -------------------- 安全工具函数 --------------------
    @staticmethod
    def _safe_int(val, default=0, min_val=None, max_val=None):
        """把 val 尽量转换为 int；遇到 NaN/inf/None/负值/不合理值则返回 default 或做裁剪。"""
        try:
            if val is None:
                return int(default)
            if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                return int(default)
            iv = int(val)
        except Exception:
            try:
                iv = int(float(val))
            except Exception:
                return int(default)
        if min_val is not None:
            iv = max(iv, min_val)
        if max_val is not None:
            iv = min(iv, max_val)
        return iv

    @staticmethod
    def _safe_float(val, default=0.0, min_val=None, max_val=None):
        """把 val 尽量转换为 float；遇到 NaN/inf/None 则返回 default 或做裁剪。"""
        try:
            if val is None:
                fv = float(default)
            else:
                fv = float(val)
                if np.isnan(fv) or np.isinf(fv):
                    return float(default)
        except Exception:
            return float(default)
        if min_val is not None:
            fv = max(fv, min_val)
        if max_val is not None:
            fv = min(fv, max_val)
        return fv

    @staticmethod
    def _sanitize_array(arr, nan_replacement=0.0, min_val=None, max_val=None):
        """将数组中 NaN/inf 替换并做裁剪。"""
        arr = np.array(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=nan_replacement, posinf=max_val if max_val is not None else 1e12,
                            neginf=min_val if min_val is not None else -1e12)
        if min_val is not None or max_val is not None:
            if min_val is None:
                arr = np.minimum(arr, max_val)
            elif max_val is None:
                arr = np.maximum(arr, min_val)
            else:
                arr = np.clip(arr, min_val, max_val)
        return arr

    # -------------------- 文件加载 --------------------
    def _load_experimental_data(self, path):
        try:
            data = pd.read_csv(path)
            if data.columns[0] == 'Unnamed: 0':
                data.rename(columns={data.columns[0]: 'Time'}, inplace=True)
            if 'Time' not in data.columns or len(data.columns) < 2:
                raise ValueError("Invalid experimental data format. Expected 'Time' column and at least one data column.")
            print(f"Successfully loaded experimental data from {path}")
            return data
        except FileNotFoundError:
            print(f"Error: Experimental data file not found at {path}")
            return None
        except Exception as e:
            print(f"Error loading experimental data: {e}")
            return None

    def _define_transition_matrix_template(self):
        template = np.zeros((self.num_configs, self.num_configs), dtype=bool)
        # (原模板保持)
        template[2, 0] = True
        template[3, 1] = True
        template[0, 2] = True
        template[1, 3] = True
        template[6, 0] = True
        template[4, 0] = True
        template[0, 6] = True
        template[0, 4] = True
        template[4, 1] = True
        template[7, 1] = True
        template[1, 4] = True
        template[1, 7] = True
        template[7, 2] = True
        template[5, 2] = True
        template[2, 7] = True
        template[2, 5] = True
        template[5, 3] = True
        template[6, 3] = True
        template[3, 5] = True
        template[3, 6] = True
        template[4, 6] = True
        template[7, 4] = True
        template[6, 4] = True
        template[4, 7] = True
        template[5, 6] = True
        template[7, 5] = True
        template[6, 5] = True
        template[5, 7] = True
        np.fill_diagonal(template, False)
        print("Defined transition matrix template based on MATLAB kinetics.")
        return template

    # -------------------- 参数设置 --------------------
    def set_parameters(self, params_dict):
        self.parameters = {}
        self.parameters.update(self.default_mechanics_params)
        self.parameters.update(self.default_kinetics_params)
        if params_dict:
            # 合并并兜底（避免 None/NaN）
            for k, v in params_dict.items():
                if v is None:
                    continue
                # 小技巧：如果值是可转换为 float，就转并替换
                try:
                    fv = float(v)
                    if np.isnan(fv) or np.isinf(fv):
                        continue
                    self.parameters[k] = fv
                except Exception:
                    # 非数值类型直接保存（有些参数可能是 bool/str）
                    self.parameters[k] = v

        self.kBT = self._safe_float(self.parameters.get('kBT', self.default_mechanics_params['kBT']),
                                    default=self.default_mechanics_params['kBT'])
        # print("Parameters set for NanorobotSolver.")

    # -------------------- 自由能计算（增加容错） --------------------
    def _calculate_free_energies(self):
        if self.parameters is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        p = self.parameters
        kBT = self._safe_float(p.get("kBT", self.default_mechanics_params["kBT"]), default=4.14, min_val=0.001)
        lp_s = self._safe_float(p.get("lp_s", self.default_mechanics_params["lp_s"]), min_val=1e-6)
        lc_s = self._safe_float(p.get("lc_s", self.default_mechanics_params["lc_s"]), min_val=1e-6)
        lc_d = self._safe_float(p.get("lc_d", self.default_mechanics_params["lc_d"]), min_val=1e-6)
        E_b = self._safe_float(p.get('E_b', self.default_mechanics_params['E_b']))
        E_b_azo_trans = self._safe_float(p.get('E_b_azo_trans', self.default_mechanics_params['E_b_azo_trans']))
        E_b_azo_cis = self._safe_float(p.get('E_b_azo_cis', self.default_mechanics_params['E_b_azo_cis']))

        n_D1 = self._safe_int(p.get('n_D1', self.default_mechanics_params['n_D1']), default=10, min_val=0, max_val=1000)
        n_D2 = self._safe_int(p.get('n_D2', self.default_mechanics_params['n_D2']), default=10, min_val=0, max_val=1000)
        n_gray = self._safe_int(p.get('n_gray', self.default_mechanics_params['n_gray']), default=10, min_val=0)
        n_hairpin_1 = self._safe_int(p.get('n_hairpin_1', self.default_mechanics_params['n_hairpin_1']), default=8, min_val=1)
        n_hairpin_2 = self._safe_int(p.get('n_hairpin_2', self.default_mechanics_params['n_hairpin_2']), default=8, min_val=1)
        n_T_hairpin_1 = self._safe_int(p.get('n_T_hairpin_1', self.default_mechanics_params['n_T_hairpin_1']), default=3, min_val=0)
        n_T_hairpin_2 = self._safe_int(p.get('n_T_hairpin_2', self.default_mechanics_params['n_T_hairpin_2']), default=2, min_val=0)
        n_track_1 = self._safe_int(p.get('n_track_1', self.default_mechanics_params['n_track_1']), default=15, min_val=1)
        n_track_2 = self._safe_int(p.get('n_track_2', self.default_mechanics_params['n_track_2']), default=55, min_val=1)

        # 初始化
        E_config_t = np.zeros(6)
        E_config_c = np.zeros(6)
        f_config_t = np.zeros(6)
        f_config_c = np.zeros(6)

        # shearing foot 最小化（保留原逻辑，但安全地处理除零和非法值）
        E_shear_foot = 100
        for i in range(n_D2 + 1):
            n_D2_detach = i
            E_b_shear = E_b * (n_D1 + n_D2 - n_D2_detach)
            denom = (lc_s * (2 * n_D2_detach + n_D1))
            if denom == 0:
                x = 0.0
            else:
                x = (n_track_1 * lc_d) / denom
            if x < 1 and denom != 0:
                try:
                    E_shear = E_b_shear + (lc_s * (2 * n_D2_detach + n_D1)) * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                except Exception:
                    E_shear = 1000
            else:
                E_shear = 1000
            if E_shear_foot > E_shear:
                E_shear_foot = E_shear

        E_zipper_foot = E_b * (n_D1 + n_D2)

        E_config_t[0] = E_zipper_foot
        E_config_t[1] = E_shear_foot
        E_config_c[0] = E_zipper_foot
        E_config_c[1] = E_shear_foot

        def calculate_double_feet_energy(track_distance, E_foot1, E_foot2):
            E_state_min_t = 1000
            f_state_min_t = 0.0
            E_state_min_c = 1000
            f_state_min_c = 0.0

            max_iter = max(1, n_hairpin_1 + n_hairpin_2)
            for i in range(1, max_iter + 1):
                n_hairpin_open = i
                if n_hairpin_open < n_hairpin_1:
                    x_numerator = track_distance
                    x_denominator = (n_hairpin_open) * 2 * lc_s
                    n_chain = n_hairpin_open
                elif n_hairpin_1 <= n_hairpin_open < n_hairpin_1 + n_hairpin_2:
                    x_numerator = track_distance
                    x_denominator = (n_hairpin_open + n_T_hairpin_1) * 2 * lc_s
                    n_chain = n_hairpin_open + n_T_hairpin_1
                else:
                    x_numerator = track_distance
                    x_denominator = (n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2) * 2 * lc_s
                    n_chain = n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2

                if x_denominator == 0:
                    x = 0.0
                else:
                    x = x_numerator / x_denominator

                if x < 1:
                    try:
                        E_neck = 2 * ((n_chain * 2 * lc_s) / lp_s) * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                        f_state = 2 * kBT / lp_s * (x - 0.25 + (1 - x) ** -2 / 4)
                    except Exception:
                        E_neck = 1000
                        f_state = 1000
                else:
                    E_neck = 1000
                    f_state = 1000

                E_state_t = E_neck + E_foot1 + E_foot2 - 2 * n_hairpin_open * E_b_azo_trans
                E_state_c = E_neck + E_foot1 + E_foot2 - 2 * n_hairpin_open * E_b_azo_cis

                if E_state_min_t > E_state_t:
                    E_state_min_t = E_state_t
                    f_state_min_t = f_state
                if E_state_min_c > E_state_c:
                    E_state_min_c = E_state_c
                    f_state_min_c = f_state

            return E_state_min_t, f_state_min_t, E_state_min_c, f_state_min_c

        track_dist_3 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t[2], f_config_t[2], E_config_c[2], f_config_c[2] = calculate_double_feet_energy(track_dist_3,
                                                                                                  E_zipper_foot,
                                                                                                  E_zipper_foot)

        track_dist_4 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t[3], f_config_t[3], E_config_c[3], f_config_c[3] = calculate_double_feet_energy(track_dist_4,
                                                                                                  E_shear_foot,
                                                                                                  E_shear_foot)

        track_dist_5 = (n_track_2 - 2 * n_gray) * lc_d
        E_config_t[4], f_config_t[4], E_config_c[4], f_config_c[4] = calculate_double_feet_energy(track_dist_5,
                                                                                                  E_zipper_foot,
                                                                                                  E_shear_foot)

        track_dist_6 = (2 * n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t[5], f_config_t[5], E_config_c[5], f_config_c[5] = calculate_double_feet_energy(track_dist_6,
                                                                                                  E_zipper_foot,
                                                                                                  E_shear_foot)

        # map to 8 states
        E_config_t_final = np.zeros(8)
        E_config_c_final = np.zeros(8)
        f_config_t_final = np.zeros(8)
        f_config_c_final = np.zeros(8)

        E_config_t_final[0:2] = E_config_t[0]
        E_config_t_final[2:4] = E_config_t[1]
        E_config_t_final[4] = E_config_t[2]
        E_config_t_final[5] = E_config_t[3]
        E_config_t_final[6] = E_config_t[4]
        E_config_t_final[7] = E_config_t[5]

        E_config_c_final[0:2] = E_config_c[0]
        E_config_c_final[2:4] = E_config_c[1]
        E_config_c_final[4] = E_config_c[2]
        E_config_c_final[5] = E_config_c[3]
        E_config_c_final[6] = E_config_c[4]
        E_config_c_final[7] = E_config_c[5]

        f_config_t_final[0:2] = f_config_t[0]
        f_config_t_final[2:4] = f_config_t[1]
        f_config_t_final[4] = f_config_t[2]
        f_config_t_final[5] = f_config_t[3]
        f_config_t_final[6] = f_config_t[4]
        f_config_t_final[7] = f_config_t[5]

        f_config_c_final[0:2] = f_config_c[0]
        f_config_c_final[2:4] = f_config_c[1]
        f_config_c_final[4] = f_config_c[2]
        f_config_c_final[5] = f_config_c[3]
        f_config_c_final[6] = f_config_c[4]
        f_config_c_final[7] = f_config_c[5]

        # 最终清理（将可能的 NaN/inf 替换）
        E_config_t_final = self._sanitize_array(E_config_t_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        E_config_c_final = self._sanitize_array(E_config_c_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        f_config_t_final = self._sanitize_array(f_config_t_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)
        f_config_c_final = self._sanitize_array(f_config_c_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)

        return E_config_t_final, f_config_t_final, E_config_c_final, f_config_c_final

    # -------------------- 速率矩阵计算（保留原逻辑并做防护） --------------------
    def _calculate_transition_rates(self, E_config_t, f_config_t, E_config_c, f_config_c):
        if self.parameters is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        p = self.parameters
        k0 = self._safe_float(p.get("k0", self.default_kinetics_params["k0"]), min_val=1e-12)
        k_mig = self._safe_float(p.get("k_mig", self.default_kinetics_params["k_mig"]), min_val=0.0)
        drt_z = self._safe_float(p.get("drt_z", self.default_kinetics_params["drt_z"]), min_val=1e-12)
        drt_s = self._safe_float(p.get("drt_s", self.default_kinetics_params["drt_s"]), min_val=1e-12)
        kBT = self._safe_float(p.get("kBT", self.default_mechanics_params["kBT"]), min_val=1e-12)

        E_config_t = self._sanitize_array(E_config_t, nan_replacement=1e6)
        E_config_c = self._sanitize_array(E_config_c, nan_replacement=1e6)
        f_config_t = self._sanitize_array(f_config_t, nan_replacement=0.0)
        f_config_c = self._sanitize_array(f_config_c, nan_replacement=0.0)

        k_trans = np.zeros((8, 8))
        k_cis = np.zeros((8, 8))

        # 使用 try/except 避免 math.exp 的 overflow
        try:
            k_trans[2, 0] = k_mig
            k_trans[3, 1] = k_mig
            k_trans[0, 2] = k_trans[2, 0] * math.exp(np.clip(E_config_t[0] - E_config_t[2], -100, 100))
            k_trans[1, 3] = k_trans[3, 1] * math.exp(np.clip(E_config_t[1] - E_config_t[3], -100, 100))

            k_trans[6, 0] = k0 * math.exp(np.clip(f_config_t[6] * drt_s / kBT, -100, 100))
            k_trans[4, 0] = k0 * math.exp(np.clip(f_config_t[4] * drt_z / kBT, -100, 100))
            k_trans[0, 6] = k_trans[6, 0] * math.exp(np.clip(E_config_t[0] - E_config_t[6], -100, 100))
            k_trans[0, 4] = k_trans[4, 0] * math.exp(np.clip(E_config_t[0] - E_config_t[4], -100, 100))

            k_trans[4, 1] = k0 * math.exp(np.clip(f_config_t[4] * drt_z / kBT, -100, 100))
            k_trans[7, 1] = k0 * math.exp(np.clip(f_config_t[7] * drt_s / kBT, -100, 100))
            k_trans[1, 4] = k_trans[4, 1] * math.exp(np.clip(E_config_t[1] - E_config_t[4], -100, 100))
            k_trans[1, 7] = k_trans[7, 1] * math.exp(np.clip(E_config_t[1] - E_config_t[7], -100, 100))

            k_trans[7, 2] = k0 * math.exp(np.clip(f_config_t[7] * drt_z / kBT, -100, 100))
            k_trans[5, 2] = k0 * math.exp(np.clip(f_config_t[5] * drt_s / kBT, -100, 100))
            k_trans[2, 7] = k_trans[7, 2] * math.exp(np.clip(E_config_t[2] - E_config_t[7], -100, 100))
            k_trans[2, 5] = k_trans[5, 2] * math.exp(np.clip(E_config_t[2] - E_config_t[5], -100, 100))

            k_trans[5, 3] = k0 * math.exp(np.clip(f_config_t[5] * drt_z / kBT, -100, 100))
            k_trans[6, 3] = k0 * math.exp(np.clip(f_config_t[6] * drt_s / kBT, -100, 100))
            k_trans[3, 5] = k_trans[5, 3] * math.exp(np.clip(E_config_t[3] - E_config_t[5], -100, 100))
            k_trans[3, 6] = k_trans[6, 3] * math.exp(np.clip(E_config_t[3] - E_config_t[6], -100, 100))

            k_trans[4, 6] = k_mig
            k_trans[7, 4] = k_mig
            k_trans[6, 4] = k_trans[4, 6] * math.exp(np.clip(E_config_t[4] - E_config_t[6], -100, 100))
            k_trans[4, 7] = k_trans[7, 4] * math.exp(np.clip(E_config_t[4] - E_config_t[7], -100, 100))

            k_trans[5, 6] = k_mig
            k_trans[7, 5] = k_mig
            k_trans[6, 5] = k_trans[5, 6] * math.exp(np.clip(E_config_t[5] - E_config_t[6], -100, 100))
            k_trans[5, 7] = k_trans[7, 5] * math.exp(np.clip(E_config_t[5] - E_config_t[7], -100, 100))
        except Exception:
            # 若发生任何数值错误，返回一个封顶的速率矩阵（多数为 0）
            pass

        try:
            k_cis[2, 0] = k_mig
            k_cis[3, 1] = k_mig
            k_cis[0, 2] = k_cis[2, 0] * math.exp(np.clip(E_config_c[0] - E_config_c[2], -100, 100))
            k_cis[1, 3] = k_cis[3, 1] * math.exp(np.clip(E_config_c[1] - E_config_c[3], -100, 100))

            k_cis[6, 0] = k0 * math.exp(np.clip(f_config_c[6] * drt_s / kBT, -100, 100))
            k_cis[4, 0] = k0 * math.exp(np.clip(f_config_c[4] * drt_z / kBT, -100, 100))
            k_cis[0, 6] = k_cis[6, 0] * math.exp(np.clip(E_config_c[0] - E_config_c[6], -100, 100))
            k_cis[0, 4] = k_cis[4, 0] * math.exp(np.clip(E_config_c[0] - E_config_c[4], -100, 100))

            k_cis[4, 1] = k0 * math.exp(np.clip(f_config_c[4] * drt_z / kBT, -100, 100))
            k_cis[7, 1] = k0 * math.exp(np.clip(f_config_c[7] * drt_s / kBT, -100, 100))
            k_cis[1, 4] = k_cis[4, 1] * math.exp(np.clip(E_config_c[1] - E_config_c[4], -100, 100))
            k_cis[1, 7] = k_cis[7, 1] * math.exp(np.clip(E_config_c[1] - E_config_c[7], -100, 100))

            k_cis[7, 2] = k0 * math.exp(np.clip(f_config_c[7] * drt_z / kBT, -100, 100))
            k_cis[5, 2] = k0 * math.exp(np.clip(f_config_c[5] * drt_s / kBT, -100, 100))
            k_cis[2, 7] = k_cis[7, 2] * math.exp(np.clip(E_config_c[2] - E_config_c[7], -100, 100))
            k_cis[2, 5] = k_cis[5, 2] * math.exp(np.clip(E_config_c[2] - E_config_c[5], -100, 100))

            k_cis[5, 3] = k0 * math.exp(np.clip(f_config_c[5] * drt_z / kBT, -100, 100))
            k_cis[6, 3] = k0 * math.exp(np.clip(f_config_c[6] * drt_s / kBT, -100, 100))
            k_cis[3, 5] = k_cis[5, 3] * math.exp(np.clip(E_config_c[3] - E_config_c[5], -100, 100))
            k_cis[3, 6] = k_cis[6, 3] * math.exp(np.clip(E_config_c[3] - E_config_c[6], -100, 100))

            k_cis[4, 6] = k_mig
            k_cis[7, 4] = k_mig
            k_cis[6, 4] = k_cis[4, 6] * math.exp(np.clip(E_config_c[4] - E_config_c[6], -100, 100))
            k_cis[4, 7] = k_cis[7, 4] * math.exp(np.clip(E_config_c[4] - E_config_c[7], -100, 100))

            k_cis[5, 6] = k_mig
            k_cis[7, 5] = k_mig
            k_cis[6, 5] = k_cis[5, 6] * math.exp(np.clip(E_config_c[5] - E_config_c[6], -100, 100))
            k_cis[5, 7] = k_cis[7, 5] * math.exp(np.clip(E_config_c[5] - E_config_c[7], -100, 100))
        except Exception:
            pass

        np.fill_diagonal(k_trans, 0)
        np.fill_diagonal(k_cis, 0)

        for i in range(self.num_configs):
            row_sum_t = np.sum(k_trans[i, :])
            row_sum_c = np.sum(k_cis[i, :])
            # 将对角线设为 -sum(others)；若 sum 非有限则设为 0
            if np.isfinite(row_sum_t):
                k_trans[i, i] = -row_sum_t
            else:
                k_trans[i, i] = 0.0
            if np.isfinite(row_sum_c):
                k_cis[i, i] = -row_sum_c
            else:
                k_cis[i, i] = 0.0

        # 清理 NaN/inf
        k_trans = self._sanitize_array(k_trans, nan_replacement=0.0, min_val=-1e12, max_val=1e12)
        k_cis = self._sanitize_array(k_cis, nan_replacement=0.0, min_val=-1e12, max_val=1e12)

        return k_trans, k_cis

    # -------------------- ODE 系统 --------------------
    def _ode_system(self, t, P, k_matrix):
        return np.dot(k_matrix, P)

    # -------------------- 模拟（添加求解器输出清理） --------------------
    def simulate(self, initial_P, sim_time_step, sim_total_time, light_schedule=None):
        E_config_t, f_config_t, E_config_c, f_config_c = self._calculate_free_energies()
        k_trans, k_cis = self._calculate_transition_rates(E_config_t, f_config_t, E_config_c, f_config_c)

        time_points = np.arange(0, sim_total_time + sim_time_step, sim_time_step)
        P_history = []
        # initial_P 兜底
        current_P = np.array(initial_P, dtype=float)
        if current_P.size != self.num_configs:
            # 若长度不对则初始化为第一个构型全部占据
            current_P = np.zeros(self.num_configs, dtype=float)
            current_P[0] = 1.0
        # 清理 initial_P
        current_P = np.nan_to_num(current_P, nan=0.0, posinf=1e6, neginf=0.0)
        current_P = np.clip(current_P, 0.0, None)

        if light_schedule is None:
            sol = solve_ivp(self._ode_system, [0, sim_total_time], current_P, args=(k_trans,),
                            t_eval=time_points, method='Radau')
            if not sol.success:
                print("Warning: ODE solver failed for trans interval; solver message:", getattr(sol, 'message', None))
            sol_y = np.nan_to_num(sol.y, nan=0.0, posinf=1e6, neginf=0.0)
            sol_y = np.clip(sol_y, 0.0, 1e6)
            P_history = sol_y.T
        else:
            # 保持原有分段逻辑，但对每段结果做 sanitize
            current_time = 0.0
            # 简化且稳健地处理光照表：按光照段顺序求解
            for (start_t, end_t, light_state) in light_schedule:
                start_t = float(start_t)
                end_t = float(end_t)
                if start_t > current_time:
                    # 先处理 gap 段（使用 trans 作为默认）
                    gap_t = np.arange(current_time, start_t + sim_time_step, sim_time_step)
                    if len(gap_t) > 1:
                        sol = solve_ivp(self._ode_system, [current_time, start_t], current_P,
                                        args=(k_trans,), t_eval=gap_t, method='Radau')
                        sol_y = np.nan_to_num(sol.y, nan=0.0, posinf=1e6, neginf=0.0)
                        sol_y = np.clip(sol_y, 0.0, 1e6)
                        P_history.extend(sol_y.T[1:].tolist())
                        current_P = sol_y.T[-1]

                # 当前光照段
                k_matrix = k_trans if (str(light_state).lower() == 'trans') else k_cis
                seg_t = np.arange(start_t, end_t + sim_time_step, sim_time_step)
                if len(seg_t) > 1:
                    sol = solve_ivp(self._ode_system, [start_t, end_t], current_P, args=(k_matrix,), t_eval=seg_t,
                                    method='Radau')
                    sol_y = np.nan_to_num(sol.y, nan=0.0, posinf=1e6, neginf=0.0)
                    sol_y = np.clip(sol_y, 0.0, 1e6)
                    P_history.extend(sol_y.T[1:].tolist())
                    current_P = sol_y.T[-1]
                current_time = end_t

            # 处理最后一段
            if current_time < sim_total_time:
                tail_t = np.arange(current_time, sim_total_time + sim_time_step, sim_time_step)
                if len(tail_t) > 1:
                    k_matrix = k_trans if (light_schedule[-1][2].lower() == 'trans') else k_cis
                    sol = solve_ivp(self._ode_system, [current_time, sim_total_time], current_P,
                                    args=(k_matrix,), t_eval=tail_t, method='Radau')
                    sol_y = np.nan_to_num(sol.y, nan=0.0, posinf=1e6, neginf=0.0)
                    sol_y = np.clip(sol_y, 0.0, 1e6)
                    P_history.extend(sol_y.T[1:].tolist())

            P_history = np.array(P_history)
            if P_history.shape[0] != len(time_points):
                # 插值补齐
                try:
                    sim_time_points = np.linspace(0, sim_total_time, P_history.shape[0])
                    interpolator = interp1d(sim_time_points, P_history.T, kind='linear', fill_value='extrapolate')
                    P_history = interpolator(time_points).T
                except Exception:
                    # 若插值失败，返回全 0（并警告）
                    P_history = np.zeros((len(time_points), self.num_configs))

        # 最终 sanitize
        P_history = np.nan_to_num(P_history, nan=0.0, posinf=1e6, neginf=0.0)
        P_history = np.clip(P_history, 0.0, 1e6)

        sim_df = pd.DataFrame(P_history, columns=[f'P_{i}' for i in range(self.num_configs)])
        sim_df['Time'] = time_points[:sim_df.shape[0]]
        return sim_df

    # -------------------- 评估（兜底处理 NaN/inf） --------------------
    def evaluate_model(self, simulated_data_df, reward_flag=0):
        if reward_flag == 0:
            if self.experimental_data_a is None or self.experimental_data_b is None:
                print("Experimental data not loaded. Cannot calculate fitting reward.")
                return -1000.0

            # 取实验数据并过滤 NaN
            try:
                exp_time_a = self.experimental_data_a['Time'].values
                exp_fam_a = self.experimental_data_a['FAM/FAM T (+)']
                exp_tye_a = self.experimental_data_a['TYE/TYE T (-)']
                exp_cy5_a = self.experimental_data_a['CY5/CY5 T (m)']

                exp_time_b = self.experimental_data_b['Time'].values
                exp_fam_b = self.experimental_data_b['FAM/FAM T (+)']
                exp_tye_b = self.experimental_data_b['TYE/TYE T (-)']
                exp_cy5_b = self.experimental_data_b['CY5/CY5 T (m)']
            except Exception as e:
                print("Error reading experimental columns:", e)
                return -1000.0

            mask_a = ~np.isnan(exp_time_a) & ~np.isnan(exp_fam_a) & ~np.isnan(exp_tye_a) & ~np.isnan(exp_cy5_a)
            exp_time_a = exp_time_a[mask_a]
            exp_fam_a = exp_fam_a[mask_a]
            exp_tye_a = exp_tye_a[mask_a]
            exp_cy5_a = exp_cy5_a[mask_a]

            mask_b = ~np.isnan(exp_time_b) & ~np.isnan(exp_fam_b) & ~np.isnan(exp_tye_b) & ~np.isnan(exp_cy5_b)
            exp_time_b = exp_time_b[mask_b]
            exp_fam_b = exp_fam_b[mask_b]
            exp_tye_b = exp_tye_b[mask_b]
            exp_cy5_b = exp_cy5_b[mask_b]

            sim_time = simulated_data_df['Time'].values

            # 映射（原样）
            sim_fam_a = simulated_data_df['P_0'] + simulated_data_df['P_1']
            sim_tye_a = simulated_data_df['P_2'] + simulated_data_df['P_3']
            sim_cy5_a = simulated_data_df['P_4'] + simulated_data_df['P_5'] + simulated_data_df['P_6'] + simulated_data_df['P_7']

            # 确保没有 NaN/inf 并做插值
            sim_fam_a = np.nan_to_num(sim_fam_a.values, nan=0.0, posinf=1e6, neginf=0.0)
            sim_tye_a = np.nan_to_num(sim_tye_a.values, nan=0.0, posinf=1e6, neginf=0.0)
            sim_cy5_a = np.nan_to_num(sim_cy5_a.values, nan=0.0, posinf=1e6, neginf=0.0)

            try:
                interp_fam_a = interp1d(sim_time, sim_fam_a, kind='linear', fill_value='extrapolate')
                interp_tye_a = interp1d(sim_time, sim_tye_a, kind='linear', fill_value='extrapolate')
                interp_cy5_a = interp1d(sim_time, sim_cy5_a, kind='linear', fill_value='extrapolate')

                sim_fam_a_interp = interp_fam_a(exp_time_a)
                sim_tye_a_interp = interp_tye_a(exp_time_a)
                sim_cy5_a_interp = interp_cy5_a(exp_time_a)
            except Exception:
                return -1000.0

            mse_fam_a = mean_squared_error(exp_fam_a, sim_fam_a_interp)
            mse_tye_a = mean_squared_error(exp_tye_a, sim_tye_a_interp)
            mse_cy5_a = mean_squared_error(exp_cy5_a, sim_cy5_a_interp)

            # Fig3b
            sim_fam_b = simulated_data_df['P_0'] + simulated_data_df['P_1']
            sim_tye_b = simulated_data_df['P_2'] + simulated_data_df['P_3']
            sim_cy5_b = simulated_data_df['P_4'] + simulated_data_df['P_5'] + simulated_data_df['P_6'] + simulated_data_df['P_7']

            sim_fam_b = np.nan_to_num(sim_fam_b.values, nan=0.0, posinf=1e6, neginf=0.0)
            sim_tye_b = np.nan_to_num(sim_tye_b.values, nan=0.0, posinf=1e6, neginf=0.0)
            sim_cy5_b = np.nan_to_num(sim_cy5_b.values, nan=0.0, posinf=1e6, neginf=0.0)

            try:
                interp_fam_b = interp1d(sim_time, sim_fam_b, kind='linear', fill_value='extrapolate')
                interp_tye_b = interp1d(sim_time, sim_tye_b, kind='linear', fill_value='extrapolate')
                interp_cy5_b = interp1d(sim_time, sim_cy5_b, kind='linear', fill_value='extrapolate')

                sim_fam_b_interp = interp_fam_b(exp_time_b)
                sim_tye_b_interp = interp_tye_b(exp_time_b)
                sim_cy5_b_interp = interp_cy5_b(exp_time_b)
            except Exception:
                return -1000.0

            mse_fam_b = mean_squared_error(exp_fam_b, sim_fam_b_interp)
            mse_tye_b = mean_squared_error(exp_tye_b, sim_tye_b_interp)
            mse_cy5_b = mean_squared_error(exp_cy5_b, sim_cy5_b_interp)

            total_mse = mse_fam_a + mse_tye_a + mse_cy5_a + mse_fam_b + mse_tye_b + mse_cy5_b
            reward = -total_mse

            if not np.isfinite(reward):
                return -1000.0
            return float(reward)
        else:
            return 0.0