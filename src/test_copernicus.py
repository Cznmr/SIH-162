import requests
import json

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

params = {
    "$filter": (
        "Collection/Name eq 'CLMS' "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'datasetIdentifier' "
        "and att/OData.CSC.StringAttribute/Value eq 'lcm_global_10m_yearly_v1'"
        ") "
        "and ContentDate/Start ge 2026-01-01T00:00:00.000Z "
        "and ContentDate/Start lt 2027-01-01T00:00:00.000Z "
        "and OData.CSC.Intersects("
        "area=geography'SRID=4326;"
        "POLYGON((77.0 15.5,81.6 15.5,81.6 20.2,77.0 20.2,77.0 15.5))'"
        ")"
    ),
    "$top": "50",
    "$expand": "Attributes",
    "$select": "Id,Name,ContentDate,GeoFootprint,S3Path,Online"
}

response = requests.get(url, params=params, timeout=60)

print("Status:", response.status_code)
print("URL:", response.url)
print()

data = response.json()

print("Number of products found:", len(data.get("value", [])))
print()

for product in data.get("value", []):
    print("NAME:", product.get("Name"))
    print("ID:", product.get("Id"))
    print("DATE:", product.get("ContentDate"))
    print("ONLINE:", product.get("Online"))
    print("S3:", product.get("S3Path"))
    print("FOOTPRINT:", product.get("GeoFootprint"))
    print("-" * 80)