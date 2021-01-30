"""

To test the API 

"""

import os
import requests

BASE = "http://0.0.0.0:5000/"

#response = requests.post(BASE, {"uploaded_pdf": "Laporan_Tugas_Akhir_Tes5.pdf"})
response = requests.post(BASE, {"uploaded_pdf":"MONLEB-2018.pdf"})
print(response.json())
