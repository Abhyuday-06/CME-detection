import requests

# Try OMNI with form-based approach matching their web interface
base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Based on the OMNI web form, the correct parameter format is:
data = (
    "activity=retrieve&"
    "res=hour&"
    "spacecraft=omni2&"
    "start_date=20240501&"
    "end_date=20260105&"
    "vars=24&vars=28&vars=38"
)

print("Trying OMNI with form-encoded params...")
r = requests.post(base, data=data, 
                  headers={"Content-Type": "application/x-www-form-urlencoded"},
                  timeout=60)
print(f"Status: {r.status_code}, Length: {len(r.text)}")
lines = r.text.strip().split("\n")
print(f"Lines: {len(lines)}")
for line in lines[:20]:
    print(f"  {line[:120]}")

print("\n---\nTrying alternative: direct text file download from OMNI FTP-like")
# OMNI2 hourly data is also at:
# https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
for year in [2024, 2025]:
    url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
    try:
        r = requests.get(url, timeout=30)
        print(f"\nomni2_{year}.dat: status={r.status_code}, length={len(r.text)}")
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            print(f"  Lines: {len(lines)}")
            print(f"  First: {lines[0][:100]}")
            # OMNI2 format: Year DOY Hour ... col28=speed col24=density col38=Kp*10
            # Columns are fixed-width
    except Exception as e:
        print(f"  Error: {e}")
