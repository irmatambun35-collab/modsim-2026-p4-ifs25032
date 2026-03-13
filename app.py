# ==============================================================================
# MODSIM 2026 - PRAKTIKUM 4: CONTINUOUS SIMULATION
# STUDY CASE: SISTEM DISTRIBUSI AIR ASRAMA (WATER TANK DISTRIBUTION)
# ==============================================================================
# Penulis: Mahasiswa Modsim
# Deskripsi: Simulasi kontinu dinamika tangki air menggunakan persamaan diferensial
#            untuk menganalisis pengisian, pengosongan, dan optimasi ukuran tangki.
# ==============================================================================

# ==============================================================================
# 1. IMPOR LIBRARY & KONFIGURASI AWAL
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import warnings
import sys
import os

# Setup plotting style untuk Matplotlib
plt.style.use('default')
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# Cek ketersediaan Streamlit untuk kompatibilitas dual-mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

print("="*70)
print("SYSTEM DISTRIBUTION WATER TANK SIMULATION ENGINE")
print("="*70)

# ==============================================================================
# 2. KELAS KONFIGURASI SISTEM (CONFIGURATION)
# ==============================================================================

@dataclass
class WaterTankConfig:
    """
    Kelas konfigurasi untuk parameter sistem tangki air asrama.
    Menyimpan semua parameter fisik, operasional, dan simulasi.
    """
    # --- Parameter Geometri Tangki ---
    tank_radius: float = 2.0          # meter (jari-jari tangki)
    tank_height: float = 3.0          # meter (tinggi total tangki)
    initial_water_level: float = 0.5  # meter (ketinggian air awal)
    max_water_level: float = 2.8      # meter (batas aman atas)
    min_water_level: float = 0.3      # meter (batas aman bawah)
    tank_material_mass: float = 50.0  # kg (massa tangki kosong)
    
    # --- Parameter Inlet (Pengisian) ---
    inlet_pipe_diameter: float = 0.1  # meter (diameter pipa masuk)
    inlet_pump_power: float = 1500.0  # Watt (daya pompa pengisi)
    inlet_pressure: float = 200000.0  # Pascal (tekanan pompa)
    inlet_valve_open: bool = True     # status katup masuk
    inlet_flow_max: float = 0.05      # m³/detik (kapasitas maksimal pipa)
    pump_efficiency: float = 0.75     # efisiensi pompa
    
    # --- Parameter Outlet (Pengosongan) ---
    outlet_pipe_diameter: float = 0.08 # meter (diameter pipa keluar)
    outlet_flow_coefficient: float = 0.6 # koefisien debit
    outlet_demand_base: float = 0.02  # m³/detik (kebutuhan dasar asrama)
    outlet_demand_peak: float = 0.08  # m³/detik (kebutuhan puncak)
    outlet_valve_open: bool = True    # status katup keluar
    gravity_drain: bool = True        # apakah menggunakan gravitasi
    
    # --- Parameter Fisik Fluida ---
    water_density: float = 1000.0     # kg/m³
    gravity: float = 9.81             # m/s²
    friction_coefficient: float = 0.02 # koefisien gesekan pipa
    
    # --- Parameter Kontrol Otomatis ---
    auto_fill_threshold: float = 0.8  # 80% kapasitas (pompa nyala)
    auto_stop_threshold: float = 0.95 # 95% kapasitas (pompa mati)
    emergency_empty_threshold: float = 0.1 # 10% (alarm kosong)
    
    # --- Parameter Simulasi ---
    simulation_time: float = 24.0     # jam (durasi simulasi)
    time_step: float = 60.0           # detik (langkah waktu)
    demand_pattern: str = 'daily'     # 'constant', 'daily', 'peak'
    
    # --- Atribut Terhitung (Derived Attributes) ---
    tank_area: float = field(init=False, default=None)
    max_volume: float = field(init=False, default=None)
    initial_volume: float = field(init=False, default=None)
    
    def __post_init__(self):
        """
        Validasi konfigurasi dan hitung atribut turunan setelah inisialisasi.
        Memastikan konsistensi parameter fisik.
        """
        # Hitung luas penampang tangki
        self.tank_area = np.pi * self.tank_radius ** 2
        
        # Hitung volume maksimal dan awal
        self.max_volume = self.tank_area * self.tank_height
        self.initial_volume = self.tank_area * self.initial_water_level
        
        # Validasi batas fisik
        if self.initial_water_level > self.tank_height:
            print(f"Peringatan: Level air awal ({self.initial_water_level}m) melebihi tinggi tangki!")
            self.initial_water_level = self.tank_height * 0.9
            
        if self.min_water_level >= self.max_water_level:
            print("Peringatan: Batas minimum >= batas maksimum! Disesuaikan.")
            self.min_water_level = self.max_water_level * 0.2
            
        if self.inlet_flow_max <= 0:
            raise ValueError("Flow rate inlet harus positif")
            
    def copy(self) -> 'WaterTankConfig':
        """
        Membuat salinan mendalam dari konfigurasi untuk analisis sensitivitas.
        Returns:
            WaterTankConfig: Instance konfigurasi baru.
        """
        params = {k: v for k, v in self.__dict__.items() 
                  if k not in ['tank_area', 'max_volume', 'initial_volume']}
        return WaterTankConfig(**params)
    
    def update_parameter(self, parameter_name: str, value: Any):
        """
        Update satu parameter secara dinamis dan hitung ulang atribut turunan.
        Args:
            parameter_name: Nama parameter yang akan diubah.
            value: Nilai baru parameter.
        """
        if parameter_name in self.__annotations__:
            setattr(self, parameter_name, value)
            self.__post_init__()
        else:
            raise ValueError(f"Parameter '{parameter_name}' tidak valid dalam konfigurasi")
    
    def get_tank_capacity_liters(self) -> float:
        """Mengembalikan kapasitas tangki dalam liter."""
        return self.max_volume * 1000.0
    
    def get_current_fill_percentage(self) -> float:
        """Mengembalikan persentase pengisian saat ini."""
        return (self.initial_water_level / self.tank_height) * 100.0

print("[OK] Kelas Konfigurasi Sistem Berhasil Dimuat")

# ==============================================================================
# 3. MODEL FISIKA & HIDROLIKA (PHYSICS MODEL)
# ==============================================================================

class HydraulicModel:
    """
    Model fisika untuk sistem distribusi air tangki.
    Menghitung aliran, tekanan, head loss, dan konsumsi energi.
    """
    
    def __init__(self, config: WaterTankConfig):
        """
        Inisialisasi model hidrolika dengan konfigurasi tertentu.
        Args:
            config: Objek konfigurasi sistem tangki.
        """
        self.config = config
    
    def calculate_tank_cross_section(self) -> float:
        """
        Hitung luas penampang tangki silinder.
        Returns:
            float: Luas penampang dalam m².
        """
        return np.pi * self.config.tank_radius ** 2
    
    def calculate_inlet_flow(self, water_level: float, 
                             pump_on: bool = True) -> float:
        """
        Hitung laju aliran masuk (Q_in) berdasarkan status pompa dan tekanan.
        Menggunakan persamaan Bernoulli sederhana dengan efisiensi.
        
        Args:
            water_level: Ketinggian air saat ini (m).
            pump_on: Status aktif/tidaknya pompa.
            
        Returns:
            float: Debit aliran masuk (m³/detik).
        """
        if not pump_on or not self.config.inlet_valve_open:
            return 0.0
        
        # Hitung head tekanan pompa
        pump_head = self.config.inlet_pressure / (
            self.config.water_density * self.config.gravity)
        
        # Head efektif melawan gravitasi air di tangki
        effective_head = pump_head - water_level
        
        if effective_head <= 0:
            # Tekanan pompa tidak cukup mengangkat air setinggi ini
            return 0.0
        
        # Luas penampang pipa inlet
        pipe_area = np.pi * (self.config.inlet_pipe_diameter / 2) ** 2
        
        # Kecepatan aliran teoritis (Torricelli modified)
        flow_velocity = np.sqrt(2 * self.config.gravity * effective_head)
        
        # Debit aktual dengan koefisien debit
        flow_rate = pipe_area * flow_velocity * self.config.outlet_flow_coefficient
        
        # Batasi oleh kapasitas maksimal pipa
        return min(flow_rate, self.config.inlet_flow_max)
    
    def calculate_outlet_flow(self, water_level: float, 
                              time_hour: float = 0.0,
                              valve_open: bool = True) -> float:
        """
        Hitung laju aliran keluar (Q_out) berdasarkan demand dan gravitasi.
        Demand dapat bervariasi berdasarkan waktu (pola harian).
        
        Args:
            water_level: Ketinggian air saat ini (m).
            time_hour: Waktu simulasi dalam jam (untuk pola demand).
            valve_open: Status katup outlet.
            
        Returns:
            float: Debit aliran keluar (m³/detik).
        """
        if not valve_open or not self.config.outlet_valve_open:
            return 0.0
        
        if water_level <= 0.01: # Hampir kosong
            return 0.0
        
        # 1. Hitung Demand Berdasarkan Pola Waktu
        base_demand = self.config.outlet_demand_base
        peak_demand = self.config.outlet_demand_peak
        
        if self.config.demand_pattern == 'constant':
            current_demand = base_demand
        elif self.config.demand_pattern == 'daily':
            # Simulasi pola harian: Pagi (6-9), Siang (12-14), Malam (18-21)
            hour_mod = time_hour % 24.0
            if (6 <= hour_mod <= 9) or (18 <= hour_mod <= 21):
                current_demand = peak_demand
            elif (12 <= hour_mod <= 14):
                current_demand = peak_demand * 0.7
            else:
                current_demand = base_demand * 0.5
        else: # peak
            current_demand = peak_demand
            
        # 2. Hitung Aliran Gravitasi (Jika pipa bawah terbuka)
        gravity_flow = 0.0
        if self.config.gravity_drain:
            pipe_area_out = np.pi * (self.config.outlet_pipe_diameter / 2) ** 2
            velocity_out = np.sqrt(2 * self.config.gravity * water_level)
            gravity_flow = pipe_area_out * velocity_out * self.config.outlet_flow_coefficient
        
        # Ambil nilai terbesar antara demand pengguna atau aliran gravitasi
        # Karena sistem harus memenuhi demand, jika gravitasi kurang, pompa boost mungkin perlu
        # Tapi di sini kita asumsikan outlet valve mengatur sesuai demand
        return max(current_demand, gravity_flow * 0.5)
    
    def calculate_pump_power_consumption(self, flow_rate: float, 
                                         height_diff: float) -> float:
        """
        Hitung daya listrik yang dikonsumsi pompa.
        P = (rho * g * Q * H) / efficiency
        
        Args:
            flow_rate: Debit aliran (m³/detik).
            height_diff: Ketinggian angkat (m).
            
        Returns:
            float: Daya dalam Watt.
        """
        if flow_rate <= 0:
            return 0.0
        
        hydraulic_power = (self.config.water_density * self.config.gravity * 
                          flow_rate * height_diff)
        
        electrical_power = hydraulic_power / self.config.pump_efficiency
        return electrical_power
    
    def calculate_energy_kwh(self, power_watt: float, time_seconds: float) -> float:
        """
        Konsumsi daya ke energi listrik (kWh).
        """
        return (power_watt * time_seconds) / 3600000.0
    
    def check_overflow(self, water_level: float) -> bool:
        """Cek kondisi meluap."""
        return water_level >= self.config.tank_height
    
    def check_empty(self, water_level: float) -> bool:
        """Cek kondisi kosong kritis."""
        return water_level <= self.config.emergency_empty_threshold
    
    def get_control_signal(self, water_level: float) -> bool:
        """
        Logika kontrol otomatis pompa (On/Off).
        Returns:
            bool: True jika pompa harus NYALA.
        """
        fill_pct = water_level / self.config.tank_height
        
        # Histeresis kontrol
        if fill_pct <= self.config.auto_fill_threshold:
            return True
        elif fill_pct >= self.config.auto_stop_threshold:
            return False
        else:
            # Pertahankan status sebelumnya (dihandle di simulator)
            return None 

print("[OK] Model Fisika & Hidrolika Berhasil Dimuat")

# ==============================================================================
# 4. SISTEM PERSAMAAN DIFERENSIAL (ODE SYSTEM)
# ==============================================================================

class TankDifferentialEquations:
    """
    Definisi sistem persamaan diferensial biasa (ODE) untuk dinamika tangki.
    State variables: [height, cumulative_inflow, cumulative_outflow, energy]
    """
    
    def __init__(self, hydraulic_model: HydraulicModel):
        """
        Inisialisasi sistem ODE.
        Args:
            hydraulic_model: Instance model fisika.
        """
        self.physics = hydraulic_model
        self.config = hydraulic_model.config
        self.pump_status = True # Status awal pompa
    
    def system_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Fungsi utama sistem ODE yang dipanggil oleh solver.
        dy/dt = f(t, y)
        
        State Vector y:
        y[0] = h (water level in meters)
        y[1] = V_in (cumulative inflow volume in m³)
        y[2] = V_out (cumulative outflow volume in m³)
        y[3] = E (cumulative energy consumption in kWh)
        
        Args:
            t: Waktu saat ini (detik).
            y: Vector state saat ini.
            
        Returns:
            np.ndarray: Turunan state (dy/dt).
        """
        h, V_in, V_out, E = y
        
        # Konversi waktu ke jam untuk pola demand
        time_hour = t / 3600.0
        
        # 1. Update Kontrol Pompa Otomatis
        control_signal = self.physics.get_control_signal(h)
        if control_signal is not None:
            self.pump_status = control_signal
        
        # 2. Hitung Aliran Masuk (Q_in)
        q_in = self.physics.calculate_inlet_flow(h, pump_on=self.pump_status)
        
        # 3. Hitung Aliran Keluar (Q_out)
        q_out = self.physics.calculate_outlet_flow(h, time_hour=time_hour)
        
        # 4. Batasan Fisik (Tidak boleh negatif atau overflow saat kalkulasi dt)
        if h <= 0.01 and q_out > q_in:
            q_out = q_in # Cegah level negatif
        
        if h >= self.config.tank_height and q_in > q_out:
            q_in = q_out # Cegah overflow (spillway assumed)
        
        # 5. Persamaan Diferensial Utama
        tank_area = self.physics.calculate_tank_cross_section()
        
        # dh/dt = (Q_in - Q_out) / Area
        dh_dt = (q_in - q_out) / tank_area
        
        # Limit laju perubahan agar stabil secara numerik
        dh_dt = np.clip(dh_dt, -0.05, 0.05)
        
        # dV_in/dt = Q_in
        dV_in_dt = q_in
        
        # dV_out/dt = Q_out
        dV_out_dt = q_out
        
        # dE/dt = Power / 3600000 (kWh per detik)
        if q_in > 0:
            # Head lift rata-rata (asumsi pompa angkat dari ground ke tengah tangki)
            lift_height = self.config.tank_height - (h / 2.0)
            power = self.physics.calculate_pump_power_consumption(q_in, lift_height)
            dE_dt = power / 3600000.0
        else:
            dE_dt = 0.0
            
        return np.array([dh_dt, dV_in_dt, dV_out_dt, dE_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """
        Mendapatkan kondisi awal sistem untuk solver.
        Returns:
            np.ndarray: Vector state awal [h0, Vin0, Vout0, E0].
        """
        return np.array([
            self.config.initial_water_level,    # h0
            0.0,                                # Vin0
            0.0,                                # Vout0
            0.0                                 # E0
        ])

print("[OK] Sistem Persamaan Diferensial Berhasil Dimuat")

# ==============================================================================
# 5. SIMULATOR UTAMA (MAIN SIMULATOR)
# ==============================================================================

class WaterTankSimulator:
    """
    Kelas utama untuk menjalankan simulasi sistem tangki air.
    Mengintegrasikan konfigurasi, fisika, ODE, dan analisis hasil.
    """
    
    def __init__(self, config: WaterTankConfig):
        """
        Inisialisasi simulator.
        Args:
            config: Konfigurasi sistem tangki.
        """
        self.config = config
        self.physics = HydraulicModel(config)
        self.equations = TankDifferentialEquations(self.physics)
        
        # Storage untuk hasil simulasi
        self.time_history: np.ndarray = None
        self.level_history: np.ndarray = None
        self.inflow_history: np.ndarray = None
        self.outflow_history: np.ndarray = None
        self.energy_history: np.ndarray = None
        self.results: Dict = None
        self.simulation_status: str = "Pending"
    
    def run_simulation(self, verbose: bool = True) -> Dict:
        """
        Menjalankan simulasi numerik menggunakan scipy.solve_ivp.
        Args:
            verbose: Jika True, cetak progress ke console.
        Returns:
            Dict: Dictionary berisi metrik kinerja simulasi.
        """
        if verbose:
            print(f"\nMemulai Simulasi Tangki Air...")
            print(f"Durasi: {self.config.simulation_time} Jam")
            print(f"Langkah Waktu: {self.config.time_step} Detik")
        
        self.simulation_status = "Running"
        
        # Setup waktu simulasi (konversi ke detik)
        t_max = self.config.simulation_time * 3600.0
        t_span = (0, t_max)
        t_eval = np.arange(0, t_max, self.config.time_step)
        
        # Kondisi awal
        y0 = self.equations.get_initial_conditions()
        
        try:
            # Solve ODE System menggunakan metode RK45
            solution = solve_ivp(
                fun=self.equations.system_equations,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                dense_output=True,
                events=None # Bisa ditambah event detection untuk overflow
            )
            
            # Simpan hasil ke atribut kelas
            self.time_history = solution.t / 3600.0  # Konversi ke Jam
            self.level_history = solution.y[0]
            self.inflow_history = solution.y[1]
            self.outflow_history = solution.y[2]
            self.energy_history = solution.y[3]
            
            # Hitung metrik kinerja
            self.results = self._calculate_performance_metrics()
            self.simulation_status = "Completed"
            
            if verbose:
                print("Simulasi Selesai Berhasil.")
                
        except Exception as e:
            self.simulation_status = "Failed"
            if verbose:
                print(f"Error saat simulasi: {str(e)}")
            raise e
            
        return self.results
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Menghitung metrik kinerja sistem berdasarkan hasil simulasi.
        Returns:
            Dict: Metrik seperti waktu pengisian, efisiensi, dll.
        """
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu sebelum menghitung metrik")
        
        # 1. Metrik Waktu
        time_to_full = self._get_time_to_level(self.config.max_water_level)
        time_to_empty = self._get_time_to_level(self.config.min_water_level)
        
        # 2. Metrik Level Air
        max_level = np.max(self.level_history)
        min_level = np.min(self.level_history)
        avg_level = np.mean(self.level_history)
        final_level = self.level_history[-1]
        
        # 3. Metrik Volume
        total_inflow = self.inflow_history[-1]
        total_outflow = self.outflow_history[-1]
        net_volume = total_inflow - total_outflow
        
        # 4. Metrik Energi
        total_energy = self.energy_history[-1]
        specific_energy = total_energy / (total_outflow + 1e-6) # kWh per m³
        
        # 5. Metrik Keandalan
        overflow_count = np.sum(self.level_history >= self.config.tank_height)
        empty_count = np.sum(self.level_history <= self.config.emergency_empty_threshold)
        reliability = 100.0 * (1.0 - (overflow_count + empty_count) / len(self.level_history))
        
        return {
            'time_to_full_hours': time_to_full,
            'time_to_empty_hours': time_to_empty,
            'max_water_level_m': max_level,
            'min_water_level_m': min_level,
            'avg_water_level_m': avg_level,
            'final_water_level_m': final_level,
            'total_inflow_m3': total_inflow,
            'total_outflow_m3': total_outflow,
            'net_volume_change_m3': net_volume,
            'total_energy_kwh': total_energy,
            'specific_energy_kwh_m3': specific_energy,
            'overflow_events': overflow_count,
            'empty_events': empty_count,
            'system_reliability_pct': reliability,
            'simulation_status': self.simulation_status
        }
    
    def _get_time_to_level(self, target_level: float) -> Optional[float]:
        """
        Mencari waktu pertama kali mencapai level tertentu.
        Args:
            target_level: Level air target (meter).
        Returns:
            float: Waktu dalam jam, atau None jika tidak tercapai.
        """
        indices = np.where(self.level_history >= target_level)[0]
        if len(indices) > 0:
            return self.time_history[indices[0]]
        return None
    
    def get_data_dataframe(self) -> pd.DataFrame:
        """
        Mengembalikan hasil simulasi dalam format DataFrame pandas.
        Returns:
            pd.DataFrame: Tabel data waktu, level, inflow, outflow, energi.
        """
        if self.time_history is None:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'Waktu_Jam': self.time_history,
            'Level_Air_m': self.level_history,
            'Volume_Masuk_m3': self.inflow_history,
            'Volume_Keluar_m3': self.outflow_history,
            'Energi_kWh': self.energy_history
        })
        return df

print("[OK] Simulator Utama Berhasil Dimuat")

# ==============================================================================
# 6. VISUALISASI MATPLOTLIB (STATIC PLOTS)
# ==============================================================================

class Visualization:
    """
    Kelas untuk membuat visualisasi statis menggunakan Matplotlib.
    Cocok untuk laporan dan analisis offline.
    """
    
    @staticmethod
    def plot_water_level_profile(simulator: WaterTankSimulator, 
                                 ax: plt.Axes = None) -> plt.Axes:
        """
        Plot profil ketinggian air terhadap waktu.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        time = simulator.time_history
        level = simulator.level_history
        config = simulator.config
        
        # Plot level air
        ax.plot(time, level, 'b-', linewidth=2.5, label='Level Air', alpha=0.8)
        
        # Garis batas
        ax.axhline(y=config.tank_height, color='r', linestyle='--', 
                   label=f'Maksimum ({config.tank_height}m)', alpha=0.7)
        ax.axhline(y=config.max_water_level, color='orange', linestyle=':', 
                   label=f'Batas Aman Atas', alpha=0.7)
        ax.axhline(y=config.min_water_level, color='green', linestyle=':', 
                   label=f'Batas Aman Bawah', alpha=0.7)
        ax.axhline(y=config.emergency_empty_threshold, color='red', linestyle='-.', 
                   label=f'Batas Kritis', alpha=0.5)
        
        # Area aman
        ax.fill_between(time, config.min_water_level, config.max_water_level,
                        color='green', alpha=0.1, label='Zona Operasi Aman')
        
        ax.set_xlabel('Waktu (Jam)', fontsize=12)
        ax.set_ylabel('Ketinggian Air (Meter)', fontsize=12)
        ax.set_title('Profil Ketinggian Air dalam Tangki', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_flow_rates(simulator: WaterTankSimulator, 
                        ax: plt.Axes = None) -> plt.Axes:
        """
        Plot laju aliran masuk dan keluar.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        time = simulator.time_history
        # Approximate flow rates from cumulative volume (derivative)
        dt = np.diff(time) * 3600 # seconds
        q_in = np.diff(simulator.inflow_history) / dt
        q_out = np.diff(simulator.outflow_history) / dt
        time_mid = time[:-1] + (time[1]-time[0])/2
        
        ax.plot(time_mid, q_in*1000, 'b-', label='Debit Masuk (L/s)', alpha=0.7)
        ax.plot(time_mid, q_out*1000, 'r-', label='Debit Keluar (L/s)', alpha=0.7)
        
        ax.set_xlabel('Waktu (Jam)')
        ax.set_ylabel('Debit (Liter/detik)')
        ax.set_title('Dinamika Debit Aliran')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_energy_consumption(simulator: WaterTankSimulator,
                                ax: plt.Axes = None) -> plt.Axes:
        """
        Plot akumulasi konsumsi energi.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(simulator.time_history, simulator.energy_history, 
                'g-', linewidth=2, label='Energi Kumulatif')
        
        ax.set_xlabel('Waktu (Jam)')
        ax.set_ylabel('Energi (kWh)')
        ax.set_title('Konsumsi Energi Pompa')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Anotasi efisiensi
        if simulator.results:
            eff = simulator.results['system_reliability_pct']
            ax.text(0.05, 0.95, f'Reliabilitas: {eff:.1f}%', 
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
        return ax
    
    @staticmethod
    def plot_comprehensive_dashboard(simulator: WaterTankSimulator):
        """
        Membuat dashboard lengkap 3 subplot dalam satu figure.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))
        
        Visualization.plot_water_level_profile(simulator, axes[0])
        Visualization.plot_flow_rates(simulator, axes[1])
        Visualization.plot_energy_consumption(simulator, axes[2])
        
        plt.tight_layout()
        return fig

print("[OK] Modul Visualisasi Matplotlib Berhasil Dimuat")

# ==============================================================================
# 7. VISUALISASI PLOTLY (INTERACTIVE PLOTS)
# ==============================================================================

class PlotlyVisualization:
    """
    Kelas untuk visualisasi interaktif menggunakan Plotly.
    Digunakan untuk aplikasi Streamlit.
    """
    
    @staticmethod
    def plot_interactive_level(simulator: WaterTankConfig) -> go.Figure:
        """
        Membuat figure interaktif untuk level air.
        """
        # Note: Need simulator instance, fixing signature
        pass

    @staticmethod
    def plot_interactive_level(simulator: WaterTankSimulator) -> go.Figure:
        """
        Membuat figure interaktif untuk level air.
        """
        fig = go.Figure()
        
        config = simulator.config
        time = simulator.time_history
        level = simulator.level_history
        
        # Trace Level Air
        fig.add_trace(go.Scatter(
            x=time, y=level,
            mode='lines',
            name='Level Air',
            line=dict(color='blue', width=3),
            hovertemplate='Waktu: %{x:.2f} Jam<br>Level: %{y:.2f} m<extra></extra>'
        ))
        
        # Garis Batas
        fig.add_hline(y=config.tank_height, line_dash="dash", line_color="red",
                      annotation_text="Tinggi Maksimum")
        fig.add_hline(y=config.max_water_level, line_dash="dot", line_color="orange",
                      annotation_text="Batas Aman Atas")
        fig.add_hline(y=config.min_water_level, line_dash="dot", line_color="green",
                      annotation_text="Batas Aman Bawah")
        
        # Layout
        fig.update_layout(
            title=dict(text='Monitoring Level Air Tangki', font=dict(size=20)),
            xaxis_title="Waktu (Jam)",
            yaxis_title="Ketinggian (Meter)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_quality_dashboard(simulator: WaterTankSimulator) -> go.Figure:
        """
        Dashboard interaktif dengan subplot.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Level Air', 'Akumulasi Volume', 
                            'Konsumsi Energi', 'Statistik Kinerja'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        time = simulator.time_history
        
        # 1. Level Air
        fig.add_trace(
            go.Scatter(x=time, y=simulator.level_history, name='Level', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=simulator.config.max_water_level, line_dash="dash", line_color='green', row=1, col=1)
        
        # 2. Volume Kumulatif
        fig.add_trace(
            go.Scatter(x=time, y=simulator.inflow_history, name='Masuk', line=dict(color='cyan')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=simulator.outflow_history, name='Keluar', line=dict(color='red')),
            row=1, col=2
        )
        
        # 3. Energi
        fig.add_trace(
            go.Scatter(x=time, y=simulator.energy_history, name='Energi', line=dict(color='orange')),
            row=2, col=1
        )
        
        # 4. Metrics Bar Chart
        if simulator.results:
            metrics = ['total_inflow_m3', 'total_outflow_m3', 'total_energy_kwh']
            labels = ['Total Inflow', 'Total Outflow', 'Energy (kWh)']
            values = [simulator.results[m] for m in metrics]
            
            fig.add_trace(
                go.Bar(x=labels, y=values, name='Metrik', marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, template="plotly_white")
        fig.update_xaxes(title_text="Waktu (Jam)", row=2, col=1)
        fig.update_yaxes(title_text="Meter", row=1, col=1)
        
        return fig

print("[OK] Modul Visualisasi Plotly Berhasil Dimuat")

# ==============================================================================
# 8. ANALISIS SENSITIVITAS (SENSITIVITY ANALYSIS)
# ==============================================================================

class SensitivityAnalysis:
    """
    Kelas untuk melakukan analisis sensitivitas parameter sistem.
    Membantu menentukan parameter kritis dan ukuran optimal.
    """
    
    @staticmethod
    def analyze_single_parameter(base_config: WaterTankConfig,
                                 parameter_name: str,
                                 values: List[float]) -> Dict:
        """
        Analisis sensitivitas untuk satu parameter terhadap metrik tertentu.
        """
        results = []
        
        for val in values:
            # Copy config agar tidak mengubah base
            config = base_config.copy()
            try:
                config.update_parameter(parameter_name, val)
                
                # Run sim
                sim = WaterTankSimulator(config)
                sim.run_simulation(verbose=False)
                
                results.append({
                    'value': val,
                    'simulator': sim,
                    'metrics': sim.results
                })
            except Exception as e:
                print(f"Error pada value {val}: {e}")
                
        return {
            'parameter': parameter_name,
            'results': results
        }
    
    @staticmethod
    def find_optimal_tank_size(base_config: WaterTankConfig,
                               target_reliability: float = 95.0) -> Dict:
        """
        Mencari ukuran tangki optimal (radius) untuk mencapai reliabilitas target.
        """
        radius_values = np.arange(1.0, 5.0, 0.5)
        best_radius = None
        best_reliability = 0
        
        data = []
        
        for r in radius_values:
            config = base_config.copy()
            config.tank_radius = r
            config.__post_init__() # Recalculate area
            
            sim = WaterTankSimulator(config)
            sim.run_simulation(verbose=False)
            
            rel = sim.results['system_reliability_pct']
            data.append({'Radius': r, 'Reliability': rel, 'Volume': config.max_volume})
            
            if rel >= target_reliability and rel > best_reliability:
                best_reliability = rel
                best_radius = r
                
        return {
            'optimal_radius': best_radius,
            'achieved_reliability': best_reliability,
            'data': pd.DataFrame(data)
        }

print("[OK] Modul Analisis Sensitivitas Berhasil Dimuat")

# ==============================================================================
# 9. APLIKASI STREAMLIT (WEB INTERFACE)
# ==============================================================================

if STREAMLIT_AVAILABLE:
    
    def create_sidebar_inputs() -> WaterTankConfig:
        """
        Membuat sidebar input untuk parameter simulasi di Streamlit.
        """
        st.sidebar.title("⚙️ Parameter Sistem")
        
        st.sidebar.subheader("Geometri Tangki")
        radius = st.sidebar.slider("Jari-jari Tangki (m)", 1.0, 5.0, 2.0, 0.1)
        height = st.sidebar.slider("Tinggi Tangki (m)", 2.0, 10.0, 3.0, 0.5)
        init_level = st.sidebar.slider("Level Awal (m)", 0.0, height, 0.5, 0.1)
        
        st.sidebar.subheader("Pompa & Inlet")
        pump_power = st.sidebar.slider("Daya Pompa (Watt)", 500, 5000, 1500, 100)
        inlet_flow = st.sidebar.slider("Debit Maks Inlet (m³/s)", 0.01, 0.2, 0.05, 0.01)
        
        st.sidebar.subheader("Demand & Outlet")
        demand_base = st.sidebar.slider("Demand Dasar (m³/s)", 0.005, 0.05, 0.02, 0.005)
        demand_peak = st.sidebar.slider("Demand Puncak (m³/s)", 0.05, 0.2, 0.08, 0.01)
        pattern = st.sidebar.selectbox("Pola Demand", ['constant', 'daily', 'peak'])
        
        st.sidebar.subheader("Simulasi")
        sim_time = st.sidebar.slider("Durasi Simulasi (Jam)", 1, 48, 24, 1)
        
        config = WaterTankConfig(
            tank_radius=radius,
            tank_height=height,
            initial_water_level=init_level,
            inlet_pump_power=pump_power,
            inlet_flow_max=inlet_flow,
            outlet_demand_base=demand_base,
            outlet_demand_peak=demand_peak,
            demand_pattern=pattern,
            simulation_time=float(sim_time)
        )
        
        return config

    def display_metrics_dashboard(results: Dict):
        """
        Menampilkan metrik kunci dalam bentuk cards.
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Reliabilitas Sistem", f"{results['system_reliability_pct']:.1f}%")
            st.metric("Level Akhir", f"{results['final_water_level_m']:.2f} m")
        with col2:
            st.metric("Total Air Masuk", f"{results['total_inflow_m3']:.2f} m³")
            st.metric("Total Air Keluar", f"{results['total_outflow_m3']:.2f} m³")
        with col3:
            st.metric("Konsumsi Energi", f"{results['total_energy_kwh']:.2f} kWh")
            st.metric("Energi Spesifik", f"{results['specific_energy_kwh_m3']:.3f} kWh/m³")
        with col4:
            st.metric("Event Overflow", f"{results['overflow_events']}")
            st.metric("Event Kosong", f"{results['empty_events']}")

    def run_streamlit_app():
        """
        Fungsi utama aplikasi Streamlit.
        """
        st.set_page_config(
            page_title="Simulasi Tangki Air Asrama",
            page_icon="💧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("💧 Simulasi Sistem Distribusi Air Asrama")
        st.markdown("""
        Aplikasi ini memodelkan dinamika tangki air menggunakan **Continuous Simulation**.
        Gunakan sidebar untuk mengubah parameter dan analisis kinerja sistem distribusi air.
        """)
        
        # Input
        config = create_sidebar_inputs()
        
        # Run Simulation
        with st.spinner('Menjalankan simulasi hidrolika...'):
            simulator = WaterTankSimulator(config)
            try:
                results = simulator.run_simulation(verbose=False)
                st.success("Simulasi Berhasil Diselesaikan!")
            except Exception as e:
                st.error(f"Terjadi kesalahan simulasi: {e}")
                return
        
        # Metrics
        display_metrics_dashboard(results)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualisasi", "🔍 Sensitivitas", "📊 Data", "📋 Laporan"])
        
        with tab1:
            st.subheader("Grafik Interaksi Sistem")
            if PLOTLY_AVAILABLE:
                fig = PlotlyVisualization.plot_quality_dashboard(simulator)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly tidak tersedia. Install dengan: pip install plotly")
                
            st.subheader("Profil Level Air Detail")
            if PLOTLY_AVAILABLE:
                fig_level = PlotlyVisualization.plot_interactive_level(simulator)
                st.plotly_chart(fig_level, use_container_width=True)
        
        with tab2:
            st.subheader("Analisis Sensitivitas Ukuran Tangki")
            st.write("Mencari ukuran tangki optimal untuk reliabilitas > 95%")
            
            if st.button("Jalankan Analisis Optimasi"):
                with st.spinner('Menghitung optimasi...'):
                    opt_result = SensitivityAnalysis.find_optimal_tank_size(config)
                    
                    if opt_result['optimal_radius']:
                        st.success(f"Radius Optimal: {opt_result['optimal_radius']} meter")
                        st.write(f"Reliabilitas Tercapai: {opt_result['achieved_reliability']:.1f}%")
                        
                        if PLOTLY_AVAILABLE:
                            fig_opt = go.Figure()
                            fig_opt.add_trace(go.Scatter(
                                x=opt_result['data']['Radius'],
                                y=opt_result['data']['Reliability'],
                                mode='lines+markers',
                                name='Reliabilitas'
                            ))
                            fig_opt.add_hline(y=95, line_dash="dash", color="green", annotation_text="Target 95%")
                            fig_opt.update_layout(title="Reliabilitas vs Ukuran Tangki", xaxis_title="Radius (m)", yaxis_title="Reliabilitas (%)")
                            st.plotly_chart(fig_opt, use_container_width=True)
                    else:
                        st.warning("Tidak ditemukan konfigurasi yang memenuhi target reliabilitas 95% dalam range pencarian.")
        
        with tab3:
            st.subheader("Data Mentah Simulasi")
            df = simulator.get_data_dataframe()
            st.dataframe(df.style.format({
                'Waktu_Jam': '{:.2f}',
                'Level_Air_m': '{:.3f}',
                'Volume_Masuk_m3': '{:.2f}',
                'Volume_Keluar_m3': '{:.2f}',
                'Energi_kWh': '{:.3f}'
            }), use_container_width=True, height=400)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "hasil_simulasi_tangki.csv", "text/csv")
        
        with tab4:
            st.subheader("Kesimpulan Studi Kasus")
            st.write("""
            Berdasarkan simulasi yang dilakukan, berikut adalah jawaban atas permasalahan studi kasus:
            1. **Waktu Pengisian**: Tergantung pada daya pompa dan ukuran tangki. Dapat dilihat dari grafik level naik.
            2. **Waktu Pengosongan**: Tergantung pada pola demand penghuni asrama.
            3. **Profil Ketinggian**: Fluktuatif mengikuti pola demand harian (pagi, siang, malam).
            4. **Simultaneous**: Sistem bekerja dinamis, pompa otomatis nyala saat level rendah.
            5. **Ukuran Optimal**: Dapat ditentukan melalui tab Sensitivitas untuk meminimalkan biaya energi dan memaksimalkan reliabilitas.
            """)
            
            st.info(f"Kapasitas Tangki Saat Ini: {config.get_tank_capacity_liters():.0f} Liter")

    # Jalankan Streamlit jika script dijalankan langsung
    if __name__ == "__main__":
        # Cek apakah dijalankan via streamlit cli atau python biasa
        import inspect
        caller_name = inspect.stack()[1].function if len(inspect.stack()) > 1 else ""
        
        # Jika ingin menjalankan mode console tanpa streamlit untuk testing
        if not STREAMLIT_AVAILABLE or 'streamlit' not in sys.modules:
            print("\nMode Console: Menjalankan simulasi demo...")
            config = WaterTankConfig(simulation_time=5.0) # Simulasi cepat 5 jam
            sim = WaterTankSimulator(config)
            res = sim.run_simulation()
            print("\nHasil Metrik:")
            for k, v in res.items():
                print(f"{k}: {v}")
            
            # Plot Matplotlib sederhana
            fig = Visualization.plot_comprehensive_dashboard(sim)
            plt.savefig("laporan_simulasi_tangki.png")
            print("\nGrafik tersimpan sebagai 'laporan_simulasi_tangki.png'")
        else:
            # Jalankan App Streamlit
            run_streamlit_app()

else:
    # Fallback jika streamlit tidak ada, jalankan mode script biasa
    if __name__ == "__main__":
        print("\nStreamlit tidak terdeteksi. Menjalankan mode script standar...")
        config = WaterTankConfig(simulation_time=12.0)
        sim = WaterTankSimulator(config)
        res = sim.run_simulation()
        
        print("\n=== HASIL STUDI KASUS DISTRIBUSI AIR ===")
        print(f"1. Waktu untuk mengisi penuh (estimasi): {res['time_to_full_hours']} Jam")
        print(f"2. Waktu untuk mencapai batas kosong: {res['time_to_empty_hours']} Jam")
        print(f"3. Reliabilitas Sistem: {res['system_reliability_pct']:.2f}%")
        print(f"4. Total Energi: {res['total_energy_kwh']:.2f} kWh")
        
        # Generate Plot
        fig = Visualization.plot_comprehensive_dashboard(sim)
        plt.show()

# ==============================================================================
# 10. DOKUMENTASI & PENUTUP
# ==============================================================================
"""
PANDUAN PENGGUNAAN KODE INI:

1. Untuk Jupyter Notebook / Analisis Offline:
   - Import kelas WaterTankConfig, WaterTankSimulator, Visualization.
   - Buat instance config, lalu simulator.
   - Jalankan sim.run_simulation().
   - Gunakan Visualization.plot_comprehensive_dashboard(sim).

2. Untuk Aplikasi Web Interaktif:
   - Pastikan streamlit terinstall: pip install streamlit plotly
   - Simpan file ini sebagai app.py
   - Jalankan: streamlit run app.py

3. Menjawab Studi Kasus (Section 2.1):
   - Waktu Isi/Penuh: Lihat metrik 'time_to_full_hours'.
   - Waktu Kosong: Lihat metrik 'time_to_empty_hours'.
   - Profil Air: Lihat grafik 'Level Air'.
   - Simultaneous: Terjadi otomatis dalam model ODE (dh/dt = Qin - Qout).
   - Ukuran Optimal: Gunakan fitur SensitivityAnalysis.find_optimal_tank_size.

CATATAN TEKNIS:
- Solver menggunakan RK45 (Runge-Kutta) untuk akurasi tinggi.
- Model mencakup histeresis kontrol pompa untuk mencegah cycling.
- Demand air dimodelkan dengan pola harian realistis.
"""
# ==============================================================================
# END OF CODE
# ==============================================================================
