# config_manager.py
"""
Manages configuration profiles and export logic.
"""
import json
from datetime import datetime

# Keys we explicitly carry over from a loaded profile when re-exporting
PRESERVED_KEYS = {
    "therm_version",
    "profile_name",
    "mapping",
    "units",
    "ai_context",
    "config_history",
    "rooms_per_zone",
    "tariff_structure",
    "thresholds",
    "physics_thresholds",
    "currency",
}
from schema_defs import REQUIRED_SENSORS, check_feature_availability

def validate_config(config):
    """
    Validates that a configuration meets minimum requirements.
    """
    errors = []
    if "mapping" not in config:
        return False, ["Missing 'mapping' section"]
    
    mapped_sensors = set(config["mapping"].keys())
    required_sensors = set(REQUIRED_SENSORS.keys())
    
    missing = required_sensors - mapped_sensors
    if missing:
        errors.append(f"Missing required sensors: {', '.join(missing)}")
    
    return len(errors) == 0, errors

def export_config_for_sharing(config):
    """
    Creates a clean version of the config for download.
    """
    # Preserve editable fields from the in-memory config (including updated profile_name)
    export_data = {}
    for k in PRESERVED_KEYS:
        # Default list-like structures to an empty list, others to an empty dict.
        # This prevents custom tariffs (lists) from being saved as empty dicts on error.
        if k in ["config_history", "tariff_structure"]:
            export_data[k] = config.get(k, [])
        else:
            export_data[k] = config.get(k, {})

    # Fill defaults where absent
    export_data["therm_version"] = export_data.get("therm_version") or "2.0"
    export_data["profile_name"] = export_data.get("profile_name") or "My Profile"
    export_data["mapping"] = export_data.get("mapping") or {}
    export_data["units"] = export_data.get("units") or {}
    export_data["ai_context"] = export_data.get("ai_context") or {}
    export_data["config_history"] = export_data.get("config_history") or []
    export_data["rooms_per_zone"] = export_data.get("rooms_per_zone") or {}

    export_data["exported_at"] = datetime.now().isoformat()
    return export_data
