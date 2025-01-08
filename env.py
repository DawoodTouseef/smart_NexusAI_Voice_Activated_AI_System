import json
from pathlib import Path
import os
import logging
import sys

from datetime import datetime
from typing import TypeVar,Generic
import docker
from docker.errors import DockerException, APIError

####################################
# Load .env file
####################################

JARVIS_DIR = Path(__file__).parent  # the path containing this file
CACHE=Path().home() / ".cache"
CACHE_DIR=CACHE/"jarvis"

try:
    from dotenv import find_dotenv, load_dotenv

    # Ensure JARVIS_DIR is properly defined
    JARVIS_DIR = Path(__file__).parent
    load_dotenv(find_dotenv(str(JARVIS_DIR / ".env")))
except ImportError:
    print("dotenv not installed, skipping...")

####################################
# LOGGING
####################################

log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

# Get global log level from environment or use default
GLOBAL_LOG_LEVEL = os.environ.get("GLOBAL_LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL in log_levels:
    logging.basicConfig(
        stream=sys.stdout,
        level=GLOBAL_LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Requires Python 3.8+
    )
else:
    GLOBAL_LOG_LEVEL = "INFO"
    logging.basicConfig(
        stream=sys.stdout,
        level=GLOBAL_LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

log = logging.getLogger(__name__)
log.info(f"GLOBAL_LOG_LEVEL: {GLOBAL_LOG_LEVEL}")

# Example use of log_sources
log_sources = [
    "SEARCH",
    "CONFIG",
]

# Log initialization of sources
for source in log_sources:
    logging.getLogger(source).setLevel(GLOBAL_LOG_LEVEL)
    log.info(f"Log level set for source {source}: {GLOBAL_LOG_LEVEL}")


SRC_LOG_LEVELS = {}

for source in log_sources:
    log_env_var = source + "_LOG_LEVEL"
    SRC_LOG_LEVELS[source] = os.environ.get(log_env_var, "").upper()
    if SRC_LOG_LEVELS[source] not in log_levels:
        SRC_LOG_LEVELS[source] = GLOBAL_LOG_LEVEL
    log.info(f"{log_env_var}: {SRC_LOG_LEVELS[source]}")

log.setLevel(SRC_LOG_LEVELS["CONFIG"])
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR,exist_ok=True)
    log.info(f"Created cache directory: {CACHE_DIR}")

def is_docker_installed():
    """
    Check if Docker is installed by attempting to create a client.
    """
    try:
        print("Checking the Docker.")
        client = docker.from_env()
        client.ping()
        return True
    except DockerException:
        return False

def get_container_id(container_name):
    """
    Check if a container with the given name exists and return its ID.
    """
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)  # Include stopped containers
        for container in containers:
            if container.name == container_name:
                return container
        return None
    except DockerException as e:
        print(f"Error connecting to Docker: {e}")
        return None


def create_nextcloud_container():
    """
    Pull the Nextcloud image and create a new container.
    Equivalent to: docker run -d -p 8080:80 nextcloud
    """
    try:
        client = docker.from_env()
        print("Pulling the 'nextcloud' image...")
        client.images.pull("nextcloud")  # Pull the Nextcloud image from Docker Hub

        print("Creating and starting the 'nextcloud' container...")
        container = client.containers.run(
            "nextcloud",  # Docker image
            name="nextcloud",  # Container name
            detach=True,  # Run in the background (equivalent to -d)
            ports={"80/tcp": 8001},  # Map port 80 in the container to port 8001 on the host
        )
        print(f"'nextcloud' container created and running with ID: {container.id}")
        return container
    except APIError as e:
        print(f"Error while creating the Nextcloud container: {e}")
        return None
    except DockerException as e:
        print(f"Error connecting to Docker: {e}")
        return None


def get_config():
    config_dir = os.path.join(CACHE_DIR, "jarvis", "config")
    config_path = os.path.join(config_dir, "config.json")

    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Create an empty config file if it doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({}, f)
            log.info(f"Created config file at: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)
    return data


CONFIG_DATA = get_config()


def save_to_config(config):
    config_dir = os.path.join(CACHE_DIR, "config")
    config_path = os.path.join(config_dir, "config.json")

    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)  # Added indent for better readability
            log.info(f"Saved configuration to {config_path}")
    except Exception as e:
        log.exception("Failed to save configuration")
        raise e

def get_config_value(config_path: str):
    path_parts = config_path.split(".")
    cur_config = CONFIG_DATA
    for key in path_parts:
        if key in cur_config:
            cur_config = cur_config[key]
        else:
            return None
    return cur_config


PERSISTENT_CONFIG_REGISTRY = []


def save_config(config):
    global CONFIG_DATA
    global PERSISTENT_CONFIG_REGISTRY
    try:
        save_to_config(config)
        CONFIG_DATA = config

        # Trigger updates on all registered PersistentConfig entries
        for config_item in PERSISTENT_CONFIG_REGISTRY:
            config_item.update()
    except Exception as e:
        log.exception("Error saving config:")
        return False

    return True

T = TypeVar("T")


class PersistentConfig(Generic[T]):
    def __init__(self, env_name: str, config_path: str, env_value: T):
        self.env_name = env_name
        self.config_path = config_path
        self.env_value = env_value
        self.config_value = get_config_value(config_path)

        # Load from config if available; otherwise, use env_value
        if self.config_value is not None:
            log.info(f"'{env_name}' loaded from the latest database entry")
            self.value = self.config_value
        elif self.env_value:
            log.info(f"'{env_name}' loaded from environment variable")
            self.value = self.env_value
            self.save()  # Save the environment value to config file
        else:
            log.warning(f"'{env_name}' is not set in config or environment")
            self.value = None

        PERSISTENT_CONFIG_REGISTRY.append(self)
        self.value=str(self.value)
    def __str__(self):
        return str(self.value)

    @property
    def __dict__(self):
        raise TypeError(
            "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
        )

    def __getattribute__(self, item):
        if item == "__dict__":
            raise TypeError(
                "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
            )
        return super().__getattribute__(item)

    def update(self):
        new_value = get_config_value(self.config_path)
        if new_value is not None:
            self.value = new_value
            log.info(f"Updated {self.env_name} to new value {self.value}")

    def save(self):
        log.info(f"Saving '{self.env_name}' to the database")
        path_parts = self.config_path.split(".")
        sub_config = CONFIG_DATA
        for key in path_parts[:-1]:
            if key not in sub_config:
                sub_config[key] = {}
            sub_config = sub_config[key]
        sub_config[path_parts[-1]] = self.value
        save_to_config(CONFIG_DATA)  # Save the entire CONFIG_DATA to update config file
        self.config_value = self.value


class AppConfig:
    _state: dict[str, PersistentConfig]

    def __init__(self):
        super().__setattr__("_state", {})

    def __setattr__(self, key, value):
        if isinstance(value, PersistentConfig):
            self._state[key] = value
        else:
            self._state[key].value = value
            self._state[key].save()

    def __getattr__(self, key):
        return self._state[key].value


GOOGLE_PSE_API_KEY = PersistentConfig(
    "GOOGLE_PSE_API_KEY",
    "rag.web.search.google_pse_api_key",
    os.getenv("GOOGLE_PSE_API_KEY", ""),
)

####################################
# Bing Search
####################################

BING_SUBSCRIPTION_KEY = PersistentConfig(
    "BING_SUBSCRIPTION_KEY",
    "rag.web.search.bing_subscription_key",
    os.getenv("BING_SUBSCRIPTION_KEY", ""),
)
BING_SEARCH_URL = PersistentConfig(
    "BING_SEARCH_URL",
    "rag.web.search.bing_search_url",
    os.getenv("BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search"),
)
BING_SEARCH_MAX_RESULTS = PersistentConfig(
    "BING_SEARCH_MAX_RESULTS",
    "rag.web.search.bing_search_max_results",
    os.getenv("BING_SEARCH_MAX_RESULTS", 10),
)
####################################
# Brave Search
####################################

BRAVE_SEARCH_URL = PersistentConfig(
    "BRAVE_SEARCH_URL",
    "rag.web.search.brave_search_url",
    os.getenv("BRAVE_SEARCH_URL", "https://search.brave.com/search"),
)
BRAVE_SEARCH_MAX_RESULTS = PersistentConfig(
    "BRAVE_SEARCH_MAX_RESULTS",
    "rag.web.search.brave_search_max_results",
    os.getenv("BRAVE_SEARCH_MAX_RESULTS", 10),
)
BRAVE_SEARCH_MAX_PAGES = PersistentConfig(
    "BRAVE_SEARCH_MAX_PAGES",
    "rag.web.search.brave_search_max_pages",
    os.getenv("BRAVE_SEARCH_MAX_PAGES", 10),
)
BRAVE_API_KEY=PersistentConfig(
    "BRAVE_API_KEY",
    "rag.web.search.brave_api_key",
    os.getenv("BRAVE_API_KEY", ""),
)
BRAVE_SAFE_SEARCH_RESULTS=PersistentConfig(
    "BRAVE_SAVE_SEARCH_RESULTS",
    "rag.web.search.brave_save_search_results",
    os.getenv("BRAVE_SAVE_SEARCH_RESULTS", "True"),
)
BRAVE_COUNTRY=PersistentConfig(
    "BRAVE_COUNTRY",
    "rag.web.search.brave_country",
    os.getenv("BRAVE_COUNTRY", "IN"),
)
BRAVE_SEARCH_LANG=PersistentConfig(
    "BRAVE_SEARCH_LANG",
    "rag.web.search.brave_search_lang",
    os.getenv("BRAVE_SEARCH_LANG", "en"),
)
BRAVE_UI_LANG=PersistentConfig(
    "BRAVE_UI_LANG",
    "rag.web.search.brave_ui_lang",
    os.getenv("BRAVE_UI_LANG", "en-US"),
)
BRAVE_OFFSET=PersistentConfig(
    "BRAVE_OFFSET",
    "rag.web.search.brave_offset",
    os.getenv("BRAVE_OFFSET", "0"),
)
BRAVE_FRESHNESS=PersistentConfig(
    "BRAVE_FRESHNESS",
    "rag.web.search.brave_freshness",
    os.getenv("BRAVE_FRESHNESS", f"{datetime.now().year-2}-01-01to{datetime.now().date()}"),
)
BRAVE_SPELLCHECK=PersistentConfig(
    "BRAVE_SPELLCHECK",
    "rag.web.search.brave_spellcheck",
    os.getenv("BRAVE_SPELLCHECK", "True"),
)

####################################
# GitHub
####################################
GITHUB_APP_ID = PersistentConfig(
    "GITHUB_APP_ID",
    "rag.web.search.github_app_id",
    os.getenv("GITHUB_APP_ID", ""),
)
GITHUB_APP_PRIVATE_KEY = PersistentConfig(
    "GITHUB_APP_PRIVATE_KEY",
    "rag.web.search.github_app_private_key",
    os.getenv("GITHUB_APP_PRIVATE_KEY", ""),
)
GITHUB_REPOSITORY=PersistentConfig(
    "GITHUB_REPOSITORY",
    "rag.web.search.github_repository",
    os.getenv("GITHUB_REPOSITORY", ""),
)
####################################
# NextCloud
####################################
if is_docker_installed():
    print("docker is Installed")
    container_name = "nextcloud"
    container = get_container_id(container_name)
    if container:
        if container.status == "running":
            print(f"Container '{container_name}' is already running with ID: {container.id}")
        else:
            print(f"Container '{container_name}' exists but is not running. Starting it...")
            container.start()
            print(f"Container '{container_name}' started with ID: {container.id}")
    else:
        print(f"Container '{container_name}' does not exist. Creating it now...")
        container=create_nextcloud_container()
    if container is not None:
        NEXTCLOUD_CONTAINER_ID = PersistentConfig(
            "NEXTCLOUD_CONTAINER_ID",
            "cloud.nextcloud.container_id",
            os.getenv("NEXTCLOUD_DOMAIN", container.id)
        )
        NEXTCLOUD_DOMAIN=PersistentConfig(
            "NEXTCLOUD_DOMAIN",
            "cloud.nextcloud.nextcloud_domain",
            os.getenv("NEXTCLOUD_DOMAIN","http://localhost:8001/")
        )
        NEXTCLOUD_USERNAME=PersistentConfig(
            "NEXTCLOUD_USERNAME",
            "cloud.nextcloud.nextcloud_username",
            os.getenv("NEXTCLOUD_USERNAME","admin")
        )
        NEXTCLOUD_PASSWORD=PersistentConfig(
            "NEXTCLOUD_PASSWORD",
            "cloud.nextcloud.nextcloud_password",
            os.getenv("NEXTCLOUD_PASSWORD","admin")
        )
else:
    NEXTCLOUD_CONTAINER_ID = PersistentConfig(
        "NEXTCLOUD_CONTAINER_ID",
        "cloud.nextcloud.container_id",
        os.getenv("NEXTCLOUD_DOMAIN", "")
    )
    NEXTCLOUD_DOMAIN = PersistentConfig(
        "NEXTCLOUD_DOMAIN",
        "cloud.nextcloud.nextcloud_domain",
        os.getenv("NEXTCLOUD_DOMAIN", "http://localhost:8001/")
    )
    NEXTCLOUD_USERNAME = PersistentConfig(
        "NEXTCLOUD_USERNAME",
        "cloud.nextcloud.nextcloud_username",
        os.getenv("NEXTCLOUD_USERNAME", "admin")
    )
    NEXTCLOUD_PASSWORD = PersistentConfig(
        "NEXTCLOUD_PASSWORD",
        "cloud.nextcloud.nextcloud_password",
        os.getenv("NEXTCLOUD_PASSWORD", "admin"))
    print("Docker is not Installed.")
####################################
# Sec API
####################################
SEC_API_API_KEY=PersistentConfig(
    "SEC_API_API_KEY",
    "rag.sec_api",
    os.environ.get('SEC_API_API_KEY',""),
)

####################################
# Nexus AI
####################################

NEXUSAI_API_KEY=PersistentConfig(
    "NEXUSAI_API_KEY",
    "nexusai.api_key",
    os.environ.get("NEXUSAI_API_KEY","")
)
NEXUSAI_API_BASE_URL=PersistentConfig(
    "NEXUSAI_BASE_URL",
    "nexusai.base_url",
    os.environ.get("NEXUSAI_BASE_URL","http://localhost:8080/")
)

####################################
# Speech Recognition
####################################

STT_PROVIDER=PersistentConfig(
    "STT_PROVIDER",
    "audio.stt.provider",
    os.environ.get("STT_PROVIDER","nexusai"),
)
STT_MODEL=PersistentConfig(
    "STT_MODEL",
    "audio.stt.model",
    os.environ.get("STT_MODEL","base"),
)
STT_BASE_URL=PersistentConfig(
    "STT_BASE_URL",
    "audio.stt.base_url",
    os.environ.get("STT_BASE_URL","http://localhost:8080/")
)
STT_LANGUAGE=PersistentConfig(
    "STT_LANGUAGE",
    "audio.stt.language",
    os.environ.get("STT_LANGUAGE","en-US")
)
STT_AUDIO_FORMAT=PersistentConfig(
    "STT_AUDIO_FORMAT",
    "audio.stt.audio_format",
    os.environ.get("STT_AUDIO_FORMAT","wav")
)
STT_TIMEOUT=PersistentConfig(
    "STT_TIMEOUT",
    "audio.stt.timeout",
    os.environ.get("STT_TIMEOUT","30")
)
STT_RETRY_ATTEMPTS=PersistentConfig(
    "STT_RETRY_ATTEMPTS",
    "audio.stt.retry_attempts",
    os.environ.get("STT_RETRY_ATTEMPTS","3")
)
STT_RETRY_DELAY=PersistentConfig(
    "STT_RETRY_DELAY",
    "audio.stt.retry_delay",
    os.environ.get("STT_RETRY_DELAY","2")
)
STT_SUPPORTED_LANGUAGES=PersistentConfig(
    "STT_SUPPORTED_LANGUAGES",
    "audio.stt.supported_languages",
    os.environ.get("STT_SUPPORTED_LANGUAGES","en-US,es-ES,fr-FR")
)
STT_RATE_LIMIT=PersistentConfig(
    "STT_RATE_LIMIT",
    "audio.stt.rate_limit",
    os.environ.get("STT_RATE_LIMIT","5")
)
####################################
# Microsoft
####################################

MICROSOFT_CLIENT_ID = PersistentConfig(
    "MICROSOFT_CLIENT_ID",
    "microsoft.client_id",
    os.environ.get("MICROSOFT_CLIENT_ID")
)
MICROSOFT_CLIENT_SECRET = PersistentConfig(
    "MICROSOFT_CLIENT_SECRET",
    "microsoft.client_secret",
    os.environ.get("MICROSOFT_CLIENT_SECRET")
)
MICROSOFT_TENANT_ID = PersistentConfig(
    "MICROSOFT_TENANT_ID",
    "microsoft.tenant_id",
    os.environ.get("MICROSOFT_TENANT_ID")
)
####################################
# AUTOGENSTUDIO
####################################
AGENTS=PersistentConfig(
    "AGENTS_YML",
    "agents.yml",
    os.environ.get("AGENTS_YML",os.path.join(CACHE_DIR,"agents"))
)
####################################
# OpenWake Word
####################################
POSITIVE_REFERENCE_CLIPS=PersistentConfig(
    "POSITIVE_DIR",
    "audio.wakeword.positive",
    os.environ.get("POSITIVE_DIR","")
)
NEGATIVE_REFERENCE_CLIPS=PersistentConfig(
    "NEGATIVE_DIR",
    "audio.wakeword.negative",
    os.environ.get("NEGATIVE_DIR","")
)
WAKE_WORD=PersistentConfig(
    "WAKE_WORD",
    "audio.wakeword.word",
    os.environ.get("WAKE_WORD","JARVIS")
)
####################################
# Home Assistant
####################################
HA_BASE_URL=PersistentConfig(
    "HA_BASE_URL",
    "automation.home-assistant.base_url",
    os.environ.get("HA_BASE_URL","")
)
HA_API_KEY=PersistentConfig(
    "HA_API_KEY",
    "automation.home-assistant.api-key",
    os.environ.get("HA_API_KEY","")
)
####################################
# HuggingFace
####################################
HF_TOKEN=PersistentConfig(
    "HF_TOKEN",
    "hf.api-key",
    os.environ.get("HF_TOKEN","")
)
####################################
# GROQ
####################################
GROQ_APIKEY=PersistentConfig(
    "GROQ_API_KEY",
    "groq.api_key",
    os.environ.get("HF_TOKEN","")
)
####################################
# User Information
####################################
USER_NAME=PersistentConfig(
    "USER_NAME",
    "users.user_name",
    os.environ.get("USER_NAME","Tony stark")
)