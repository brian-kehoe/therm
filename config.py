# config.py
CALC_VERSION = "v1.21.0"
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
    "ghost_power_threshold": 120    # W - Indoor Power level indicating zone pumps active
}

NIGHT_HOURS = [2, 3, 4, 5]

SENSOR_EXPECTATION_MODE = {
    'Power': 'system', 'Indoor_Power': 'system', 'FlowTemp': 'system', 'ReturnTemp': 'system',
    'Heat': 'heating_active', 'COP_Raw': 'heating_active', 'FlowRate': 'heating_active', 'DeltaT': 'heating_active',
    'DHW_Temp': 'dhw_active',
    'OutdoorTemp': 'system', 'Outdoor_Humidity': 'system', 'Solar_Rad': 'system', 'Wind_Speed': 'system',
    'Room_Hallway': 'system', 'Room_Ecowitt_Hub': 'system', 'Room_Living': 'system', 'Room_Playroom': 'system',
    'Room_MainBed': 'system', 'Room_Oisin': 'system', 'Room_Caoimhe': 'system',
    'DHW_Mode': 'system', 'Immersion_Mode': 'system', 'HP_Binary_Status': 'system', 'Zone_UFH': 'system',
    'Zone_DS': 'system', 'Zone_US': 'system', 'Pump_Primary': 'system', 'Pump_Secondary': 'system',
    'Quiet_Mode': 'system', 'Defrost': 'event_only',
}

SENSOR_ROLES = {
    'sensor.heat_pump_power_ch1': 'core_periodic',
    'sensor.heat_pump_heat_output': 'core_periodic',
    'sensor.heat_pump_flow_temperature': 'core_periodic',
    'sensor.heat_pump_return_temperature': 'core_periodic',
    'sensor.heat_pump_flow_rate': 'core_periodic',
    'sensor.heat_pump_indoor_power': 'core_periodic',
    'sensor.heat_pump_compressor_frequency': 'core_periodic',
    'sensor.heat_pump_flow_delta': 'core_periodic',
    'sensor.heat_pump_outdoor_temperature': 'core_periodic',
    'sensor.ecowitt_weather_solar_radiation': 'core_periodic',
    'sensor.ecowitt_weather_wind_speed': 'core_periodic',
    'sensor.ecowitt_weather_humidity': 'core_periodic',
    'sensor.living_room_temp_humidity_temperature': 'room_temp',
    'sensor.playroom_temp_humidity_sensor_temperature': 'room_temp',
    'sensor.main_bedroom_temperature': 'room_temp',
    'sensor.oisin_s_bedroom_temp_humidity_sensor_temperature': 'room_temp',
    'sensor.caoimhe_s_room_temp_humidity_sensor_temperature': 'room_temp',
    'sensor.ecowitt_weather_indoor_temperature': 'room_temp',
    'sensor.heat_pump_temp_humidity_sensor_temperature': 'room_temp',
    'binary_sensor.underfloor_pump': 'binary_state',
    'binary_sensor.downstairs_radiator_pump': 'binary_state',
    'binary_sensor.upstairs_radiator_pump': 'binary_state',
    'binary_sensor.primary_pump': 'binary_state',
    'binary_sensor.secondary_pump': 'binary_state',
    'binary_sensor.heat_pump_in_operation': 'binary_state',
    'sensor.heat_pump_immersion_heater_mode_value': 'binary_state',
    'switch.quiet_mode': 'binary_state',
    'sensor.heat_pump_3way_valve_position_value': 'binary_state',
    'sensor.heat_pump_hot_water_mode_value': 'binary_state',
    'sensor.heat_pump_defrost_status': 'rare_event',
}

ENTITY_MAP = {
    'sensor.heat_pump_power_ch1': 'Power',
    'sensor.heat_pump_heat_output': 'Heat',
    'sensor.heat_pump_indoor_power': 'Indoor_Power',
    'sensor.heat_pump_immersion_heater_mode_value': 'Immersion_Mode',
    'sensor.heat_pump_flow_delta': 'DeltaT',
    'sensor.heat_pump_compressor_frequency': 'Freq',
    'sensor.heat_pump_flow_rate': 'FlowRate',
    'sensor.heat_pump_cop': 'COP_Raw',
    'binary_sensor.heat_pump_in_operation': 'HP_Binary_Status',
    'sensor.heat_pump_flow_temperature': 'FlowTemp',
    'sensor.heat_pump_return_temperature': 'ReturnTemp',
    'sensor.heat_pump_outdoor_temperature': 'OutdoorTemp',
    'sensor.heat_pump_hot_water_temperature': 'DHW_Temp',
    'sensor.heat_pump_hot_water_mode_value': 'DHW_Mode',
    'sensor.heat_pump_temp_humidity_sensor_temperature': 'Room_Hallway',
    'sensor.ecowitt_weather_solar_radiation': 'Solar_Rad',    
    'sensor.ecowitt_weather_wind_speed': 'Wind_Speed',        
    'sensor.ecowitt_weather_humidity': 'Outdoor_Humidity',    
    'sensor.ecowitt_weather_indoor_temperature': 'Room_Ecowitt_Hub',
    'sensor.living_room_temp_humidity_temperature': 'Room_Living',
    'sensor.playroom_temp_humidity_sensor_temperature': 'Room_Playroom',
    'sensor.main_bedroom_temperature': 'Room_MainBed',
    'sensor.oisin_s_bedroom_temp_humidity_sensor_temperature': 'Room_Oisin',
    'sensor.caoimhe_s_room_temp_humidity_sensor_temperature': 'Room_Caoimhe',
    'binary_sensor.underfloor_pump': 'Zone_UFH',
    'binary_sensor.downstairs_radiator_pump': 'Zone_DS',
    'binary_sensor.upstairs_radiator_pump': 'Zone_US',
    'switch.quiet_mode': 'Quiet_Mode',
    'binary_sensor.primary_pump': 'Pump_Primary',
    'binary_sensor.secondary_pump': 'Pump_Secondary',
    'sensor.heat_pump_defrost_status': 'Defrost',
    'sensor.heat_pump_3way_valve_position_value': 'ValveMode',
}

ZONE_TO_ROOM_MAP = {
    'Zone_UFH': ['Room_Ecowitt_Hub'], 
    'Zone_DS':  ['Room_Living', 'Room_Playroom', 'Room_Hallway'],
    'Zone_US':  ['Room_MainBed', 'Room_Oisin', 'Room_Caoimhe']
}


# Unified Sensor Groupings for Data Quality Tabs
SENSOR_GROUPS = {
    "⚡ Power & Energy": ['Power', 'Indoor_Power', 'Heat'],
    "💧 Hydraulics": ['FlowRate', 'FlowTemp', 'ReturnTemp', 'DeltaT', 'Pump_Primary', 'Pump_Secondary', 'ValveMode'],
    "🌤️ Environment": ['OutdoorTemp', 'Solar_Rad', 'Wind_Speed', 'Outdoor_Humidity'],
    "⚙️ System State": ['HP_Binary_Status', 'DHW_Mode', 'Immersion_Mode', 'Quiet_Mode', 'DHW_Temp'],
    "🏠 Zones": ['Zone_UFH', 'Zone_DS', 'Zone_US'],
    "ℹ️ Events": ['Defrost']
}

TARIFF_STRUCTURE = [
    {
        "valid_from": "2023-01-01",
        "name": "Standard Dual Rate",
        "rules": [
            {"name": "Night", "start": "02:00", "end": "06:00", "rate": 0.08},
            {"name": "Day",    "start": "06:00", "end": "02:00", "rate": 0.33},
        ]
    },
    {
        "valid_from": "2026-01-19",
        "name": "Multi-Band Smart Tariff",
        "rules": [
            {"name": "Night",       "start": "00:00", "end": "02:00", "rate": 0.20},
            {"name": "Night (EV)",  "start": "02:00", "end": "05:00", "rate": 0.075},
            {"name": "Night",       "start": "05:00", "end": "08:00", "rate": 0.20},
            {"name": "Day",         "start": "08:00", "end": "17:00", "rate": 0.33},
            {"name": "Peak",        "start": "17:00", "end": "19:00", "rate": 0.45},
            {"name": "Day",         "start": "19:00", "end": "23:00", "rate": 0.33},
            {"name": "Night",       "start": "23:00", "end": "00:00", "rate": 0.20} 
        ]
    }
]

CONFIG_HISTORY = [
    {"start": "2023-01-01", "config_tag": "baseline_v1", "change_note": "Initial commissioning."},
    {"start": "2025-11-28", "config_tag": "Pump & Hydraulic Fix", "change_note": "Primary/Secondary Pumps to Constant Speed II."},
    {"start": "2025-11-30", "config_tag": "Data Fix & DHW Tuning", "change_note": "HA Helpers updated. DHW Target 50C."},
    {"start": "2025-12-01", "config_tag": "Sequential Schedule", "change_note": "DHW 02:00-03:00, Heating 03:00-06:00."}
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
"""