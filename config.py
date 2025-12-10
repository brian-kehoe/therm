# config.py
CALC_VERSION = "v1.21.2"
TARIFF_PROFILE_ID = "Multi_Band_Smart_Tariff"

# Thresholds
MIN_HEAT_FREQ = 15.0          # Hz
MIN_POWER_W = 50.0            # W
BASELINE_JSON_PATH = "sensor_heartbeat_baseline_seasonal.json"

THRESHOLDS = {
    "short_cycle_min": 20,
    "very_short_cycle_min": 10,
    "flow_limit_tolerance": 2.0,
    "flow_limit_min_duration": 15,
    "hdd_base_temp": 18.0,
    "high_night_share": 0.50,
    "short_cycling_ratio_high": 0.3,
    "flow_over_43c_pct_high": 20,
    "dhw_scop_low": 2.2,
    "night_share_elec_low": 0.30,
    "heating_during_dhw_power_threshold": 120,    # W - Indoor Power level indicating heating during DHW via power proxy
    "heating_during_dhw_detection_pct": 0.15,     # Fraction of DHW run with heating evidence to trigger warning
    # Run detection guards
    "min_heating_run_minutes_with_no_zones": 8,
    "min_heating_run_heat_kwh_with_no_zones": 0.25,
}

NIGHT_HOURS = [2, 3, 4, 5]

# ==========================================
# ENGINE STATE / ACTIVITY THRESHOLDS
# ==========================================

ENGINE_STATE_THRESHOLDS = {
    # Frequency-based detection (primary when available)
    "freq_on_hz": 10.0,
    "freq_off_hz": 5.0,   # optional if you later add hysteresis

    # Flow-based detection (fallback if no freq/power)
    "flow_on_lpm": 2.0,
    "flow_off_lpm": 1.0,

    # Temperature-based detection
    "delta_on_C": 1.0,          # minimum ΔT to count as active heating
    "delta_coast_min_C": 0.5,   # minimum ΔT for coast-down

    # Optional: Power-based detection (when available)
    "power_on_W": 300.0,

    # Coast-down lookback window (minutes)
    "coast_down_window_min": 5,

    # Minimum run duration (filter out noise runs)
    "minimum_run_duration_min": 5,
}


# ==========================================
# PHYSICS & HYDRAULIC CONSTANTS
# ==========================================
SPECIFIC_HEAT_CAPACITY = 65.0  # 3.9 kJ/kg·K × 1000 / 60 s = 65 W/(L/min·°C)

# Calculation Thresholds (Gatekeepers)
PHYSICS_THRESHOLDS = {
    "min_flow_rate_lpm": 3.0,       # Minimum flow to count as "moving water"
    "min_freq_for_delta_t": 5.0,    # Compressor Hz required to trust Delta T
    "min_freq_for_heat": 7.5,       # Compressor Hz required to calculate Heat Output
    "max_valid_delta_t": 15.0,      # Sanity check: Delta T shouldn't exceed this
    "min_valid_delta_t": 0.2        # Ignore tiny fluctuations
}

# ==========================================
# SENSOR FALLBACKS (Primary -> Backup)
# ==========================================
SENSOR_FALLBACKS = {
    'OutdoorTemp': 'OutdoorTemp_OWM',
    'Outdoor_Humidity': 'Outdoor_Humidity_OWM',
    'Wind_Speed': 'Wind_Speed_OWM'
}

# ==========================================
# SENSOR GROUPS (Visual Display Order)
# ==========================================
SENSOR_GROUPS = {
    "⚡ Power & Energy": ['Power', 'Indoor_Power', 'Heat'],
    "💧 Hydraulics": ['DeltaT', 'FlowRate', 'FlowTemp', 'ReturnTemp', 'Pump_Primary', 'Pump_Secondary', 'ValveMode'],
    "🌤️ Environment (Primary)": ['OutdoorTemp', 'Solar_Rad', 'Wind_Speed', 'Outdoor_Humidity'],
    "🌤️ Environment (Backup)": ['OutdoorTemp_OWM', 'Wind_Speed_OWM', 'Outdoor_Humidity_OWM', 'UV_Index_OWM'],
    "⚙️ System State": ['Heat_Pump_Active', 'DHW_Mode', 'Immersion_Mode', 'Quiet_Mode', 'DHW_Temp'],
    "🏠 Zones": ['Zone_UFH', 'Zone_DS', 'Zone_US'],
    "ℹ️ Events": ['Defrost']
}

# ==========================================
# SENSOR EXPECTATION MODES (Data Quality)
# ==========================================
SENSOR_EXPECTATION_MODE = {
    # --- POWER & ENERGY ---
    'Power': 'system',
    'Indoor_Power': 'system',
    'Heat': 'heating_active',
    
    # --- HYDRAULICS ---
    'FlowTemp': 'system',
    'ReturnTemp': 'system',
    'FlowRate': 'heating_active',
    'DeltaT': 'heating_active',
    'COP_Raw': 'heating_active',
    'DHW_Temp': 'dhw_active',

    # --- ENVIRONMENT (Primary) ---
    'OutdoorTemp': 'system',
    'Outdoor_Humidity': 'system',
    'Solar_Rad': 'system',
    'Wind_Speed': 'system',
    
    # --- ENVIRONMENT (Backup - Sparse) ---
    'OutdoorTemp_OWM': 'system_slow',
    'Outdoor_Humidity_OWM': 'system_slow',
    'Wind_Speed_OWM': 'system_slow',
    'UV_Index_OWM': 'system_slow',

    # --- ROOM TEMPERATURES (generic placeholders; user-configured) ---
    'Room_1': 'system',
    'Room_2': 'system',
    'Room_3': 'system',
    'Room_4': 'system',
    'Room_5': 'system',
    'Room_6': 'system',
    'Room_7': 'system',
    'Room_8': 'system',

    # --- EVENTS / BINARY STATE (Neutral Grey Scoring) ---
    'Immersion_Mode': 'event_only',
    'Quiet_Mode': 'event_only',
    'Defrost': 'event_only',

    # --- SYSTEM STATE (Scored) ---
    'Zone_1': 'system',
    'Zone_2': 'system',
    'Zone_3': 'system',
    'Pump_Primary': 'system',
    'Pump_Secondary': 'system',
    'Heat_Pump_Active': 'system',
    'ValveMode': 'system',
    'DHW_Mode': 'system',
}

# ==========================================
# SENSOR ROLES (Heartbeat Learning)
# ==========================================
SENSOR_ROLES = {
    # Core periodic sensors (with prefix for HA, without for Grafana)
    'sensor.heat_pump_power_ch1': 'core_periodic',
    'heat_pump_power_ch1': 'core_periodic',
    'sensor.heat_pump_heat_output': 'core_periodic',
    'heat_pump_heat_output': 'core_periodic',
    'sensor.heat_pump_flow_temperature': 'core_periodic',
    'heat_pump_flow_temperature': 'core_periodic',
    'sensor.heat_pump_return_temperature': 'core_periodic',
    'heat_pump_return_temperature': 'core_periodic',
    'sensor.heat_pump_flow_rate': 'core_periodic',
    'heat_pump_flow_rate': 'core_periodic',
    'sensor.heat_pump_indoor_power': 'core_periodic',
    'heat_pump_indoor_power': 'core_periodic',
    'sensor.heat_pump_compressor_frequency': 'core_periodic',
    'heat_pump_compressor_frequency': 'core_periodic',
    'sensor.heat_pump_flow_delta': 'core_periodic',
    'heat_pump_flow_delta': 'core_periodic',
    'sensor.heat_pump_outdoor_temperature': 'core_periodic',
    'heat_pump_outdoor_temperature': 'core_periodic',
    'sensor.weather_solar_radiation': 'core_periodic',
    'weather_solar_radiation': 'core_periodic',
    'sensor.weather_wind_speed': 'core_periodic',
    'weather_wind_speed': 'core_periodic',
    'sensor.weather_humidity': 'core_periodic',
    'weather_humidity': 'core_periodic',

    # Room temperature sensors (generic placeholders; user-configured)
    'sensor.room_1_temperature': 'room_temp',
    'room_1_temperature': 'room_temp',
    'sensor.room_2_temperature': 'room_temp',
    'room_2_temperature': 'room_temp',
    'sensor.room_3_temperature': 'room_temp',
    'room_3_temperature': 'room_temp',
    'sensor.room_4_temperature': 'room_temp',
    'room_4_temperature': 'room_temp',
    'sensor.room_5_temperature': 'room_temp',
    'room_5_temperature': 'room_temp',
    'sensor.room_6_temperature': 'room_temp',
    'room_6_temperature': 'room_temp',
    'sensor.room_7_temperature': 'room_temp',
    'room_7_temperature': 'room_temp',
    'sensor.room_8_temperature': 'room_temp',
    'room_8_temperature': 'room_temp',
    
    # Backup OpenWeather Sensors (SPARSE)
    'sensor.openweathermap_temperature': 'weather_sparse',
    'sensor.openweathermap_humidity': 'weather_sparse',
    'sensor.openweathermap_wind_speed': 'weather_sparse',
    'sensor.openweathermap_uv_index': 'weather_sparse',
    # Unprefixed OpenWeather sensors (Grafana compatibility)
    'openweathermap_temperature': 'weather_sparse',
    'openweathermap_humidity': 'weather_sparse',
    'openweathermap_wind_speed': 'weather_sparse',
    'openweathermap_uv_index': 'weather_sparse',

    # Binary state sensors (generic placeholders)
    'binary_sensor.zone_1': 'binary_state',
    'zone_1': 'binary_state',
    'binary_sensor.zone_2': 'binary_state',
    'zone_2': 'binary_state',
    'binary_sensor.zone_3': 'binary_state',
    'zone_3': 'binary_state',
    'binary_sensor.primary_pump': 'binary_state',
    'primary_pump': 'binary_state',
    'binary_sensor.secondary_pump': 'binary_state',
    'secondary_pump': 'binary_state',
    'binary_sensor.heat_pump_in_operation': 'binary_state',
    'heat_pump_in_operation': 'binary_state',
    'sensor.heat_pump_immersion_heater_mode_value': 'binary_state',
    'heat_pump_immersion_heater_mode_value': 'binary_state',
    'switch.quiet_mode': 'binary_state',
    'quiet_mode': 'binary_state',
    'sensor.heat_pump_3way_valve_position_value': 'binary_state',
    'heat_pump_3way_valve_position_value': 'binary_state',
    'sensor.heat_pump_hot_water_mode_value': 'binary_state',
    'heat_pump_hot_water_mode_value': 'binary_state',
    'sensor.heat_pump_hot_water_status_value': 'binary_state',
    'heat_pump_hot_water_status_value': 'binary_state',
    'sensor.heat_pump_hot_water_temperature': 'core_periodic',
    'heat_pump_hot_water_temperature': 'core_periodic',
    'sensor.heat_pump_defrost_status': 'rare_event',
    'heat_pump_defrost_status': 'rare_event',

    # Mapped/internal column names (used after schema mapping)
    'Zone_1': 'binary_state',
    'Zone_2': 'binary_state',
    'Zone_3': 'binary_state',
    'Zone_4': 'binary_state',
    'Room_1': 'room_temp',
    'Room_2': 'room_temp',
    'Room_3': 'room_temp',
    'Room_4': 'room_temp',
    'Room_5': 'room_temp',
    'Room_6': 'room_temp',
    'Room_7': 'room_temp',
    'Power': 'core_periodic',
    'FlowTemp': 'core_periodic',
    'ReturnTemp': 'core_periodic',
    'FlowRate': 'core_periodic',
    'OutdoorTemp': 'core_periodic',
    'Freq': 'core_periodic',
    'Indoor_Power': 'core_periodic',
    'DHW_Temp': 'core_periodic',
    'DHW_Active': 'binary_state',
    'DHW_Mode': 'binary_state',
    'ValveMode': 'binary_state',
    'Defrost': 'rare_event',
}

# ==========================================
# ENTITY MAPPING (Raw -> Friendly)
# ==========================================
ENTITY_MAP = {
    'sensor.heat_pump_power_ch1': 'Power',
    'sensor.heat_pump_heat_output': 'Heat',
    'sensor.heat_pump_indoor_power': 'Indoor_Power',
    'sensor.heat_pump_immersion_heater_mode_value': 'Immersion_Mode',
    'sensor.heat_pump_flow_delta': 'DeltaT',
    'sensor.heat_pump_compressor_frequency': 'Freq',
    'sensor.heat_pump_flow_rate': 'FlowRate',
    'sensor.heat_pump_cop': 'COP_Raw',
    'binary_sensor.heat_pump_in_operation': 'Heat_Pump_Active', 
    'sensor.heat_pump_flow_temperature': 'FlowTemp',
    'sensor.heat_pump_return_temperature': 'ReturnTemp',
    'sensor.heat_pump_outdoor_temperature': 'OutdoorTemp',
    'sensor.heat_pump_hot_water_temperature': 'DHW_Temp',
    'sensor.heat_pump_hot_water_mode_value': 'DHW_Mode',
    'sensor.room_1_temperature': 'Room_1',
    'sensor.room_2_temperature': 'Room_2',
    'sensor.room_3_temperature': 'Room_3',
    'sensor.room_4_temperature': 'Room_4',
    'sensor.room_5_temperature': 'Room_5',
    'sensor.room_6_temperature': 'Room_6',
    'sensor.room_7_temperature': 'Room_7',
    'sensor.room_8_temperature': 'Room_8',
    'binary_sensor.zone_1': 'Zone_1',
    'binary_sensor.zone_2': 'Zone_2',
    'binary_sensor.zone_3': 'Zone_3',
    'switch.quiet_mode': 'Quiet_Mode',
    'binary_sensor.primary_pump': 'Pump_Primary',
    'binary_sensor.secondary_pump': 'Pump_Secondary',
    'sensor.heat_pump_defrost_status': 'Defrost',
    'sensor.heat_pump_3way_valve_position_value': 'ValveMode',

    # --- OPENWEATHER BACKUPS ---
    'sensor.openweathermap_temperature': 'OutdoorTemp_OWM',
    'sensor.openweathermap_humidity': 'Outdoor_Humidity_OWM',
    'sensor.openweathermap_wind_speed': 'Wind_Speed_OWM',
    'sensor.openweathermap_uv_index': 'UV_Index_OWM',
}

ZONE_TO_ROOM_MAP = {
    'Zone_1': ['Room_1', 'Room_2'],
    'Zone_2': ['Room_3', 'Room_4'],
    'Zone_3': ['Room_5', 'Room_6'],
}

TARIFF_STRUCTURE = [
    {
        "valid_from": "2023-01-01",
        "name": "Flat Default Tariff",
        "rules": [
            {"name": "All Day", "start": "00:00", "end": "24:00", "rate": 0.30},
        ],
    }
]

CONFIG_HISTORY = [
    {"start": "2023-01-01", "config_tag": "baseline_v1", "change_note": "Initial commissioning."},
    {"start": "2025-11-28", "config_tag": "Pump & Hydraulic Fix", "change_note": "Primary/Secondary Pumps to Constant Speed II."},
    {"start": "2025-11-30", "config_tag": "Data Fix & DHW Tuning", "change_note": "HA Helpers updated. DHW Target 50C."},
    {"start": "2025-12-01", "config_tag": "Sequential Schedule", "change_note": "DHW Night 02:00-03:00, Heating U/F 03:00-06:00."},
    {"start": "2025-12-04", "config_tag": "Sequential Schedule", "change_note": "DHW Day 13:00-13:59, Heating D/S Thermostat 13:00-14:00 @ 17.5C."}
]

AI_SYSTEM_CONTEXT = """
SYSTEM CONTEXT FOR AI ANALYSIS (HEAT PUMP PHYSICS & SETTINGS):
- Heat Pump Model: Samsung EHS Mono Gen 6 (AE080RXYDEG/EU).
- Controller: Joule Kodiak Control Board (SmartPlumb System).
- Control Logic: "Single Master Curve". The system generates water based on the Radiator Requirement.
- Efficiency Goal: Weather Compensation (Low Flow Temp at mild Outdoor Temp).
- Anomaly Detection: System Limit (43°C), Short Cycling, Cost Inefficiency, DHW Drag.
- Property: 175sqm Detached A1 Rated House. High Thermal Mass.
- Operational Strategy: "Super-Heating" the fabric during Night Rate (02:00-06:00).
- Flow Limiting Context: "Virtual_FTlim" = minutes where Actual Flow runs >2 degC below Target during heating. High totals point to flow restriction (air/pump/valves), aggressive curve, or mixing.

KNOWN SYSTEM BEHAVIOUR (HEATING DURING DHW):
- CONTROL LIMITATION: Heating zone pumps (UFH and Rads) can run during DHW production.
- PHYSICAL CONSEQUENCE:
  1. Hydraulic Mixing.
  2. Elevated Return Temperatures.
  3. Efficiency Penalty (< 2.0 COP).
- DETECTION RULE: 'heating_during_dhw_detected' = TRUE if any heating zone pump is ON for >15% of DHW duration.
- INTERPRETATION RULE: If detected, attribute low efficiency to this hydraulic crossover.
"""
