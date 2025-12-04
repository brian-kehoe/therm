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
        "unit": "°C", 
        "required": True,
        "description": "Temperature of water leaving the heat pump."
    },
    "ReturnTemp": {
        "label": "Return Temperature", 
        "unit": "°C", 
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
        "unit": "°C", 
        "required": False,
        "description": "External ambient air temperature."
    },
}

# 3. Optional Sensors (Advanced / DHW)
OPTIONAL_SENSORS = {
    # DHW Sensors
    "DHW_Temp": {
        "label": "DHW Tank Temperature", 
        "unit": "A�C", 
        "required": False,
        "description": "Current temperature of the hot water cylinder."
    },
    "DHW_Mode": {
        "label": "DHW Active Mode (Signal)", 
        "unit": "Text/Binary", 
        "required": False,
        "description": "State or Mode sensor indicating if DHW is currently active (e.g. 'DHW', 'On')."
    },
    "ValveMode": {
        "label": "3-Way Valve Position", 
        "unit": "Text", 
        "required": False,
        "description": "Position of the diverter valve (e.g. 'Heating', 'DHW')."
    },
    "DHW_Active": {
        "label": "DHW Active (Binary)", 
        "unit": "0/1", 
        "required": False,
        "description": "Binary sensor specifically for DHW status (1=On)."
    },
    
    # Advanced Diagnostics
    "Indoor_Power": {
        "label": "Indoor Unit Power (optional)", 
        "unit": "W", 
        "required": False,
        "description": "Indoor unit / compressor cabinet power draw, used as a proxy for immersion & ghost pumping."
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
}# 4. Zone Sensors
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
        "unit": "°C", 
        "required": False,
        "description": "Temperature sensor for Room 1."
    },
    "Room_2": {
        "label": "Room 2 Temperature", 
        "unit": "°C", 
        "required": False,
        "description": "Temperature sensor for Room 2."
    },
    "Room_3": {
        "label": "Room 3 Temperature", 
        "unit": "°C", 
        "required": False,
        "description": "Temperature sensor for Room 3."
    },
    "Room_4": {
        "label": "Room 4 Temperature", 
        "unit": "°C", 
        "required": False,
        "description": "Temperature sensor for Room 4."
    },
    "Room_5": {
        "label": "Room 5 Temperature", 
        "unit": "°C", 
        "required": False,
        "description": "Temperature sensor for Room 5."
    },
}

# 6. Environmental Sensors
ENVIRONMENTAL_SENSORS = {
    "Solar_Rad": {
        "label": "Solar Radiation", 
        "unit": "W/m²", 
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
    "tariff_structure": {
        "label": "Tariff / Cost Structure",
        "placeholder": "e.g., Night rate 2am-5am @ 0.08, Day @ 0.35...",
        "help": "Used to calculate cost efficiency accurately."
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
# PART 3: HELPER FUNCTIONS
# ==========================================

def get_unit_options(sensor_key):
    """
    Returns a list of valid units for a given sensor key.
    Used by mapping_ui to populate the unit dropdown.
    """
    # 1. Aggregate all definitions
    all_definitions = {}
    all_definitions.update(REQUIRED_SENSORS)
    all_definitions.update(RECOMMENDED_SENSORS) # Added this to fix lookup for FlowRate
    all_definitions.update(OPTIONAL_SENSORS)
    all_definitions.update(ZONE_SENSORS)
    all_definitions.update(ROOM_SENSORS)
    all_definitions.update(ENVIRONMENTAL_SENSORS)
    
    # 2. Find the sensor definition
    definition = all_definitions.get(sensor_key)
    
    if not definition:
        return []
        
    default_unit = definition.get("unit")
    
    # 3. Return options
    if default_unit:
        return [default_unit]
    return []


# ==========================================
# PART 4: ANALYSIS FEATURE FLAGS
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


