from Crewai.tools.base_tool import BaseTool
import pywifi
from pywifi import const
import time


class scan_wifi_networks(BaseTool):
    def __init__(self):
        super().__init__(
            name="Scan Wifi ",
            description="Scan for available Wi-Fi networks."
        )
    async def _run(self):
        """Scan for available Wi-Fi networks."""
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(3)  # Allow time for scanning
        scan_results = iface.scan_results()
        networks = [network.ssid for network in scan_results if network.ssid]
        return f"Available Wi-Fi networks: {', '.join(networks)}" if networks else "No Wi-Fi networks found."


class connect_to_wifi(BaseTool):
    def __init__(self):
        super().__init__(
            name="Connect Wifi",
            description='Connect to a specified Wi-Fi network.'
        )
    async def _run(self,ssid: str, password: str):
        """Connect to a specified Wi-Fi network."""
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.disconnect()
        time.sleep(1)

        profile = pywifi.Profile()
        profile.ssid = ssid
        profile.auth = const.AUTH_ALG_OPEN
        profile.akm.append(const.AKM_TYPE_WPA2PSK)
        profile.cipher = const.CIPHER_TYPE_CCMP
        profile.key = password

        iface.remove_all_network_profiles()
        tmp_profile = iface.add_network_profile(profile)
        iface.connect(tmp_profile)
        time.sleep(5)  # Wait for connection

        if iface.status() == const.IFACE_CONNECTED:
            return f"Connected to Wi-Fi network '{ssid}'."
        else:
            return f"Failed to connect to Wi-Fi network '{ssid}'."


class disconnect_wifi(BaseTool):
    def __init__(self):
        super().__init__(
            name="Disconnect Wifi",
            description="Disconnect from the current Wi-Fi network."
        )
    async def _run(self):
        """Disconnect from the current Wi-Fi network."""
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.disconnect()
        time.sleep(1)
        if iface.status() == const.IFACE_DISCONNECTED:
            return "Disconnected from Wi-Fi."
        else:
            return "Failed to disconnect from Wi-Fi."
