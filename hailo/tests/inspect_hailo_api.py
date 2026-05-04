import inspect
import hailo_platform as hp

print("hailo_platform imported")
print("Available main names:")
for name in dir(hp):
    if not name.startswith("_"):
        print(name)

print("\nHEF class:")
print(hp.HEF)

print("\nVDevice class:")
print(getattr(hp, "VDevice", None))
print("\nConfigureParams:")
print(getattr(hp, "ConfigureParams", None))
