import pandas as pd
from pathlib import Path

# Diretórios
root_dir = Path('/home/bruno/Desktop/TCC/Dataset/extracted_dataset')
cleaned_dir = Path('cleaned_transcripts')
cleaned_dir.mkdir(parents=True, exist_ok=True)

# Intervalo de pacientes que você quer processar
start_id = 364
end_id = 427

# Percorre cada pasta de entrevista
for interview_folder in root_dir.iterdir():
    if interview_folder.is_dir():
        # Extrai o número da pasta, assumindo formato "{id}_P"
        try:
            patient_id = int(interview_folder.name.split('_')[0])
        except ValueError:
            continue  # pula se não conseguir extrair número
        
        if not (start_id <= patient_id <= end_id):
            continue  # pula pacientes fora do intervalo
        
        # Procura pelo arquivo *_TRANSCRIPT.csv
        transcript_files = list(interview_folder.glob('*_TRANSCRIPT.csv'))
        if not transcript_files:
            continue  # pula se não achar transcript
        transcript_file = transcript_files[0]
        
        # Lê o CSV
        df = pd.read_csv(transcript_file, sep='\t')
        
        # Mantém apenas as falas do paciente
        df_cleaned = df[df['speaker'] == 'Participant'][['value']]
        df_cleaned['value'] = df_cleaned['value'].str.strip()
        df_cleaned = df_cleaned[df_cleaned['value'] != '']
        
        # Salva o CSV limpo
        new_csv_path = cleaned_dir / f'{patient_id}_cleaned.csv'
        df_cleaned.to_csv(new_csv_path, index=False)
        print(f"✅ Limpou e salvou: {new_csv_path.name}")

print("🎯 Todos os transcripts do intervalo 364–427 foram limpos e salvos em cleaned_transcripts/")
