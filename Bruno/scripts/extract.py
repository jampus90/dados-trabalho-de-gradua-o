import os
import zipfile

# Pasta onde estão os arquivos .zip
source_dir = os.getcwd()  # ou caminho específico, ex: "/home/bruno/Desktop/TCC/Dataset"
# Nova pasta para os arquivos extraídos
extract_dir = os.path.join(source_dir, "extracted_dataset")
os.makedirs(extract_dir, exist_ok=True)

# Percorre todos os arquivos .zip
for file_name in os.listdir(source_dir):
  if file_name.endswith(".zip"):
    zip_path = os.path.join(source_dir, file_name)
    print(f"📦 Extraindo {file_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(os.path.join(extract_dir, file_name.replace(".zip","")))


print("✅ Todos os arquivos foram extraídos.")
