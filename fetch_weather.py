from dotenv import load_dotenv
import os
from pyowm import OWM

# Load the .env file
load_dotenv()

API_KEY = os.getenv("OWM_API_KEY")
if not API_KEY:
    raise ValueError("OWM_API_KEY not found in environment")

owm = OWM(API_KEY)
mgr = owm.weather_manager()

obs = mgr.weather_at_place("North Chili,US")
w = obs.weather
print("Temperature:", w.temperature('fahrenheit')["temp"])

