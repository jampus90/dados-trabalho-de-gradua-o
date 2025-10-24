from pathlib import Path
import zipfile

PATH = Path('C:/Users/Alexandre/Documents/FATEC/TG/Código/arquivos rar')
DESTINY_PATH = Path('C:/Users/Alexandre/Documents/FATEC/TG/Código/dados-trabalho-de-gradua-o/Alexandre')

def read_multiple_files(zip_path, destiny_path):
    zip_files = list(zip_path.glob('*.zip'))

    for file in zip_files:
        with zipfile.ZipFile(file,'r') as zf:
            inner_files = zf.infolist()

            for file_info in inner_files:
                if file_info.is_dir():
                    continue

                lower_file_name = file_info.filename.lower()

                if 'transcript' in lower_file_name and lower_file_name.endswith('.csv'):
                    original_name = file_info.filename
                    final_file_name = Path(original_name).name

                    full_destiny_file_path = destiny_path / final_file_name
                    
                    if full_destiny_file_path.exists():
                        print(f"  Aviso: {full_destiny_file_path.name} já existe e será SOBRESCRITO.")

                    file_info.filename = final_file_name

                    zf.extract(file_info, path=destiny_path)

                    file_info.filename = original_name



read_multiple_files(PATH, DESTINY_PATH)