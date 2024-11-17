import airsim

client = airsim.MultirotorClient("127.0.0.1")
client.confirmConnection()
print("Connected to AirSim!")
