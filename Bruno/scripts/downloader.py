import requests
import os
from time import sleep

# Diretório onde salvar os arquivos
download_dir = os.getcwd()
os.makedirs(download_dir, exist_ok=True)

# Intervalo de arquivos
start_id = 364
end_id = 427
base_url = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"

for i in range(start_id, end_id + 1):
    file_name = f"{i}_P.zip"
    url = f"{base_url}{file_name}"
    dest_path = os.path.join(download_dir, file_name)
    
    try:
        print(f"⬇️ Baixando {file_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        sleep(1)  # evita sobrecarga no servidor
    except Exception as e:
        print(f"⚠️ Erro ao baixar {file_name}: {e}")

print("✅ Todos os downloads concluídos.")
