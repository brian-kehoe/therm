# schema_defs.py

# ==========================================
# PART 1: CONSTANTS & PREFIXES
# ==========================================
ROOM_SENSOR_PREFIX = "Room_"
ZONE_SENSOR_PREFIX = "Zone_"

# ==========================================
# PART 2: UI MAPPING DEFINITIONS
# ==========================================

# 1. Essential Sensors
REQUIRED_SENSORS = {
    "Power": {
        "label": "Heat Pump Power (Elec)",
        "unit": "W",
        "required": True,
        "description": "Total electrical power consumption of the heat pump unit."
    },
    "FlowTemp": {
        "label": "Flow Temperature",
        "unit": "degC",
        "required": True,
        "description": "Temperature of water leaving the heat pump."
    },
    "ReturnTemp": {
        "label": "Return Temperature",
        "unit": "degC",
        "required": True,
        "description": "Temperature of water returning to the heat pump."
    },
}

# 2. Recommended Sensors (High Priority)
# These were causing the duplicate error because they were aliased.
# Now they are distinct.
RECOMMENDED_SENSORS = {
    "FlowRate": {
        "label": "Flow Rate",
        "unit": "L/min",
        "required": False,
        "description": "Water flow rate in the primary circuit. Required for accurate Heat output and COP. If not mapped, the app runs in 'Power & Temps only' mode with no energy or COP metrics."
    },
    "OutdoorTemp": {
        "label": "Outdoor Temperature",
        "unit": "degC",
        "required": False,
        "description": "External ambient air temperature."
    },
}

# 3. Optional Sensors (Advanced / DHW)
OPTIONAL_SENSORS = {
    # DHW Sensors
    "DHW_Temp": {
        "label": "DHW Tank Temperature",
        "unit": "degC",
        "required": False,
        "description": "Current temperature of the hot water cylinder."
    },
    "DHW_Mode": {
        "label": "DHW Mode",
        "unit": "Text/Binary",
        "required": False,
        "description": "DHW Mode e.g. Eco, Standard, Power."
    },
    "ValveMode": {
        "label": "3-Way Valve Position",
        "unit": "Text",
        "required": False,
        "description": "Position of the diverter valve (e.g. 'Heating', 'DHW')."
    },
    "DHW_Active": {
        "label": "DHW Active Status",
        "unit": "0/1",
        "required": False,
        "description": "Binary sensor specifically for DHW status (1=On)."
    },

    # Advanced Diagnostics
    "Indoor_Power": {
        "label": "Indoor Unit Power (optional)",
        "unit": "W",
        "required": False,
        "description": "Indoor unit / compressor cabinet power draw, used as a proxy for immersion and heating during DHW."
    },

    # NEW: direct heat-output sensor
    "Heat": {
        "label": "Heat Output (optional)",
        "unit": "W",
        "required": False,
        "description": "Thermal output of the heat pump in Watts (space + DHW). Use if your system already exposes a heat output sensor."
    },
    "Freq": {
        "label": "Compressor Frequency",
        "unit": "Hz",
        "required": False,
        "description": "Operating frequency of the compressor."
    },
    "Defrost": {
        "label": "Defrost Status",
        "unit": "0/1",
        "required": False,
        "description": "Binary sensor indicating if defrost cycle is active."
    },
}

# 4. Zone Sensors
ZONE_SENSORS = {
    "Zone_1": {
        "label": "Zone 1 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 1."
    },
    "Zone_2": {
        "label": "Zone 2 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 2."
    },
    "Zone_3": {
        "label": "Zone 3 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 3."
    },
    "Zone_4": {
        "label": "Zone 4 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 4."
    },
}

# 5. Room Sensors
ROOM_SENSORS = {
    "Room_1": {
        "label": "Room 1 Temperature",
        "unit": "degC",
        "required": False,
        "description": "Temperature sensor for Room 1."
    },
    "Room_2": {
        "label": "Room 2 Temperature",
        "unit": "degC",
        "required": False,
        "description": "Temperature sensor for Room 2."
    },
    "Room_3": {
        "label": "Room 3 Temperature",
        "unit": "degC",
        "required": False,
        "description": "Temperature sensor for Room 3."
    },
    "Room_4": {
        "label": "Room 4 Temperature",
        "unit": "degC",
        "required": False,
        "description": "Temperature sensor for Room 4."
    },
    "Room_5": {
        "label": "Room 5 Temperature",
        "unit": "degC",
        "required": False,
        "description": "Temperature sensor for Room 5."
    },
}

# 6. Environmental Sensors
ENVIRONMENTAL_SENSORS = {
    "Solar_Rad": {
        "label": "Solar Radiation",
        "unit": "W/m2",
        "required": False,
        "description": "Solar irradiance sensor."
    },
    "Wind_Speed": {
        "label": "Wind Speed",
        "unit": "m/s",
        "required": False,
        "description": "External wind speed sensor."
    },
    "Outdoor_Humidity": {
        "label": "Outdoor Humidity",
        "unit": "%",
        "required": False,
        "description": "External relative humidity."
    },
}

# 7. AI Context Prompts
AI_CONTEXT_PROMPTS = {
    "hp_model": {
        "label": "Heat Pump Model",
        "placeholder": "e.g., Samsung Gen 6, Daikin Altherma 3...",
        "help": "Helps the AI identify model-specific quirks (e.g., defrost behavior)."
    },
    "property_context": {
        "label": "Property Context",
        "placeholder": "e.g., 1990s detached, underfloor heating downstairs...",
        "help": "Provides context for heat loss and thermal retention."
    },
    "operational_goals": {
        "label": "Operational Goals",
        "placeholder": "e.g., Prioritize comfort over cost, minimize cycling...",
        "help": "The AI will judge performance against these specific goals."
    }
}

# ==========================================
# PART 3: UNIT CONVERSIONS & VALIDATION
# ==========================================
# Supported alternative units by sensor (UI dropdowns)
ALT_UNIT_OPTIONS = {
    # Wind speed commonly reported in m/s, km/h, or mph.
    "Wind_Speed": ["m/s", "km/h", "mph"],
}

# Conversion functions to internal base units
# Base units (identity) are included so the conversion table is complete.
UNIT_CONVERSIONS = {
    "degC": lambda x: x,
    "W": lambda x: x,
    "W/m2": lambda x: x,
    "%": lambda x: x,
    "Hz": lambda x: x,
    "0/1": lambda x: x,
    "Text": lambda x: x,
    # Flow/volume helpers
    "L/min": lambda x: x,
    # Wind speed conversions
    "m/s": lambda x: x,
    "km/h": lambda x: x * 0.2777777778,
    "mph": lambda x: x * 0.44704,
}

# Validation ranges for selected sensors (optional; extend as needed)
VALIDATION_RULES = {
    "OutdoorTemp": {"type": "numeric", "min": -40, "max": 55},
    "Outdoor_Humidity": {"type": "numeric", "min": 0, "max": 100},
    "Wind_Speed": {"type": "numeric", "min": 0, "max": 60},
}

# ==========================================
# PART 4: HELPER FUNCTIONS
# ==========================================

def get_unit_options(sensor_key):
    """
    Returns a list of valid units for a given sensor key.
    Used by mapping_ui to populate the unit dropdown.
    """
    all_definitions = {}
    all_definitions.update(REQUIRED_SENSORS)
    all_definitions.update(RECOMMENDED_SENSORS)
    all_definitions.update(OPTIONAL_SENSORS)
    all_definitions.update(ZONE_SENSORS)
    all_definitions.update(ROOM_SENSORS)
    all_definitions.update(ENVIRONMENTAL_SENSORS)

    definition = all_definitions.get(sensor_key)
    if not definition:
        return []

    default_unit = definition.get("unit")
    opts = ALT_UNIT_OPTIONS.get(sensor_key, [])
    if default_unit:
        opts = [default_unit] + opts

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in opts:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

# ==========================================
# PART 5: ANALYSIS FEATURE FLAGS
# ==========================================

REQUIRED_COLUMNS = {
    "core": ["Power", "FlowTemp", "ReturnTemp"],
    "hydraulics": ["FlowRate"],
    "weather": ["OutdoorTemp"],
    "dhw_analysis": ["DHW_Temp"]
}

def check_feature_availability(df, user_mapping=None):
    availability = {}
    columns_present = set(df.columns)
    for feature, requirements in REQUIRED_COLUMNS.items():
        is_available = all(req in columns_present for req in requirements)
        availability[feature] = is_available
    return availability

def get_missing_columns(df, feature_name):
    if feature_name not in REQUIRED_COLUMNS:
        return []
    requirements = REQUIRED_COLUMNS[feature_name]
    return [req for req in requirements if req not in df.columns]
