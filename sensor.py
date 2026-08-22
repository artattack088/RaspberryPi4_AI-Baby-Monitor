import smbus2
import bme280
import time

port = 1
address = 0x77
bus = smbus2.SMBus(port)

calibration_params = bme280.load_calibration_params(bus, address)

def read_sensor():
    try:
        data = bme280.sample(bus, address, calibration_params)
        return {
            "temperature": round(data.temperature, 2),
            "humidity": round(data.humidity, 2),
            "pressure": round(data.pressure, 2),
            "status": "online"
        }
    except Exception as e:
        return {"status": "sensor_error", "message": "Check wires"}

if __name__ == "__main__":
    while True:
        reading = read_sensor()
        print(f"🌡️  Temp: {reading['temperature']}°C  💧 Humidity: {reading['humidity']}%  🔵 Pressure: {reading['pressure']}hPa")
        time.sleep(2)
