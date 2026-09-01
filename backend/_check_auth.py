import requests

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124"

r = session.post(
    "https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip",
    data={
        "file": "data_odometry_velodyne.zip",
        "email": "25261999.rohan@gdgu.org",
        "submit": "Request Download Link",
    },
    timeout=30,
)
print(f"Status: {r.status_code}")
print(r.text[:3000])
