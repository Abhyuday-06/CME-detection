import requests

# NASA OMNI hourly data includes both solar wind AND Kp in one dataset
# Try CGI interface for bulk download
# Parameters: proton speed, proton density, Kp index
# Time range: 2024-05-01 to 2026-01-31

# OMNI2 hourly data via OMNIWeb
# Activity=retrieve, dataset=omni_min (1-min or hourly), variables selected by number
# Variables we need from OMNI2 hourly:
#   28 = Flow speed (km/s)
#   24 = Proton density (N/cm^3)
#   38 = Kp*10

# Method 1: Direct text file download
base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
params = {
    "activity": "retrieve",
    "res": "hour",  # hourly resolution
    "spacecraft": "omni2",
    "start_date": "20240501",
    "end_date": "20260131",
    "vars": "24,28,38",  # density(24), speed(28), Kp*10(38)
}

print("Trying NASA OMNI CGI download...")
r = requests.post(base, data=params, timeout=60)
print(f"Status: {r.status_code}, Length: {len(r.text)}")

# Check content
lines = r.text.strip().split("\n")
print(f"Total lines: {len(lines)}")

# Find where data starts (skip HTML header)
data_start = -1
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and stripped[0].isdigit() and len(stripped.split()) >= 5:
        data_start = i
        break

if data_start >= 0:
    print(f"Data starts at line {data_start}")
    print(f"First data: {lines[data_start]}")
    print(f"Last data:  {lines[-1]}")
    # Count valid data lines
    count = sum(1 for l in lines[data_start:] if l.strip() and l.strip()[0].isdigit())
    print(f"Data lines: {count}")
else:
    print("No numeric data found. HTML response:")
    for line in lines[:30]:
        print(f"  {line[:120]}")
