from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meal-orchestrator")
except PackageNotFoundError:
    __version__ = "0.0.0"

APP_NAME = "Meal Orchestrator"
USER_AGENT = f"meal-orchestrator/{__version__}"
