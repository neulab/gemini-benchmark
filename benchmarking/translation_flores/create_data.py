
import pandas as pd
import os
import ast
import tqdm

def process_tsv_file(tsv_file, lang_code, target_root_dir):
    try:
        df = pd.read_csv(tsv_file, sep='\t')
    except Exception as e:
        print(f"Error reading file {tsv_file}: {e}")
        return

    input_dir = os.path.join(target_root_dir, lang_code, 'inputs')
    ref_dir = os.path.join(target_root_dir, lang_code, 'refs')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    for index, row in df.iterrows():
        try:
            # Safely evaluate the string representation of the list of dictionaries
            messages = ast.literal_eval(row['messages'])
            content = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")

            input_path = os.path.join(input_dir, f's{index}.txt')
            ref_path = os.path.join(ref_dir, f's{index}.txt')

            with open(input_path, 'w', encoding='utf-8') as input_file, open(ref_path, 'w', encoding='utf-8') as ref_file:
                input_file.write(content)
                ref_file.write(row['label'])
        except ValueError as e:
            print(f"Error processing file {tsv_file}, row {index}: {e}")

def process_all_tsv_files(source_dir, target_root_dir):
    for file_name in tqdm.tqdm(os.listdir(source_dir)):
        if file_name.endswith('.tsv'):
            lang_code = file_name.split('.')[0]
            tsv_file_path = os.path.join(source_dir, file_name)
            process_tsv_file(tsv_file_path, lang_code, target_root_dir)


# Replace with the path to your source directory containing TSV files
source_dir = './aligned_system_outputs/tt-zero'

# Replace with your target directory path where the dataset directories should be created
target_root_dir = './main_texts/tt-zero'

process_all_tsv_files(source_dir, target_root_dir)

