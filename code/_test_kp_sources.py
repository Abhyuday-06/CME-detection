import requests

# 1. NOAA 1 month
r = requests.get("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json", timeout=15)
data = r.json()
print(f"NOAA 1m: {len(data)} records")
print(f"  First: {data[0]['time_tag']}")
print(f"  Last:  {data[-1]['time_tag']}")

# 2. Try ISGI WDC format
for year in [2024, 2025, 2026]:
    url = f"https://isgi.unistra.fr/data/kp/kp{year}.wdc"
    try:
        r = requests.get(url, timeout=15)
        print(f"\nISGI {year}: status={r.status_code}, length={len(r.text)}")
        if r.status_code == 200 and len(r.text) > 100:
            lines = r.text.strip().split("\n")
            print(f"  Lines: {len(lines)}")
            print(f"  Sample: {lines[0][:80]}")
    except Exception as e:
        print(f"ISGI {year}: {e}")

# 3. Try NASA OMNI data (has Kp + solar wind combined!)
url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
print(f"\nNASA OMNI endpoint exists for historical solar wind + Kp")

# 4. Try NOAA FTP-style endpoint for older data
for endpoint in [
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
]:
    try:
        r = requests.get(endpoint, timeout=10)
        data = r.json()
        print(f"\n{endpoint.split('/')[-1]}: {len(data)} rows")
        if len(data) > 1:
            print(f"  First data: {data[1]}")
            print(f"  Last data:  {data[-1]}")
    except Exception as e:
        print(f"  Error: {e}")
