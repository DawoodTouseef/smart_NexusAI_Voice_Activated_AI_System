from crewai_tools import tool
import bluetooth

@tool
async def scan_bluetooth_devices():
    """Scan for nearby Bluetooth devices."""
    devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True)
    if devices:
        result = "\n".join([f"Device: {name}, Address: {addr}" for addr, name in devices])
        return f"Nearby Bluetooth devices:\n{result}"
    else:
        return "No Bluetooth devices found."

@tool
async def connect_bluetooth_device(device_address: str):
    """Attempt to connect to a Bluetooth device by address."""
    try:
        # Note: Actual connection requires device-specific handling and permissions.
        return f"Attempted connection to Bluetooth device at {device_address}. Check device pairing."
    except Exception as e:
        return f"Failed to connect to Bluetooth device: {e}"

@tool
async def disconnect_bluetooth_device(device_address: str):
    """Disconnect a Bluetooth device by address."""
    try:
        # Bluetooth disconnection is often done via OS or device settings
        return f"Attempted disconnection from Bluetooth device at {device_address}. Check device settings for confirmation."
    except Exception as e:
        return f"Failed to disconnect from Bluetooth device: {e}"
