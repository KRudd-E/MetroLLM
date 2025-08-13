import pandas as pd
import os
import re
from openpyxl import load_workbook
from src.utils.utils import tc_app_query, tc_def_query, tc_comp_query

class TextClass:
    def __init__(self, config, src):
        self.config = config
        if src.lower() == 'applications'  : self.applicationsDB_Gen(config['applicationsDB'])
        if src.lower() == 'definitions'   : self.definitionsDB_Gen(config['definitionsDB'])
        if src.lower() == 'companies'     : self.companiesDB_Gen(config['companiesDB'])

    def applicationsDB_Gen(self, config):
        tc_app_query()

        excel_path = os.path.join(os.getcwd() + config['xlsx_dir'])
        pure_xlsx_df = pd.read_excel(
            excel_path,
            sheet_name=config['sheet_name'],
            index_col=None
        )
        pure_xlsx_df.rename(columns={'File': 'Name'}, inplace=True)
        pure_xlsx_df = pure_xlsx_df[['Name', 'Date', 'Sector', 'Task', 'User', 'Location', 'Level']]
        
        wb = load_workbook(excel_path)
        ws = wb[config['sheet_name']] if config['sheet_name'] in wb.sheetnames else wb.active
        assert ws is not None, "Worksheet not found in the workbook."
        
        db_base = os.path.join(os.getcwd() + config['db_dir'])

        # Add new columns for text data
        pure_xlsx_df['id'] = None
        pure_xlsx_df['Subdir'] = None
        pure_xlsx_df['File'] = None
        pure_xlsx_df['Text'] = None
        pure_xlsx_df['Text_len'] = None

        for i, row in enumerate(ws.iter_rows(min_row=2, min_col=2, max_col=2), start=0):
            cell = row[0]
            try:
                pdf_path = cell.hyperlink.target.replace('%20', ' ') #type: ignore
                file_name = (pdf_path.split('\\')[-1])[:-4] + '.txt'
                subfolder_name = pdf_path.split('\\')[-2]
                subfolder_path = os.path.join(db_base + subfolder_name)

                if not os.path.exists(subfolder_path):
                    print(f"Missing subfolder: {subfolder_path}")
                    continue

                with open(os.path.join(subfolder_path, file_name), 'r', encoding='utf-8') as file:
                    txt = file.read()
                    txt = self.format_txt(txt)

                    pure_xlsx_df.at[i, 'id'] = i + 1
                    pure_xlsx_df.at[i, 'Text'] = txt
                    pure_xlsx_df.at[i, 'Text_len'] = len(txt)
                    pure_xlsx_df.at[i, 'Subdir'] = subfolder_name
                    pure_xlsx_df.at[i, 'File'] = file_name

            except (AttributeError, FileNotFoundError) as e:
                print(f"{cell.coordinate} - {e}")
                continue

        df = pure_xlsx_df[['id', 'Name', 'Subdir', 'File', 'Task', 'User', 'Location', 'Level', 'Text_len', 'Text']]

        # ensure id is integer
        df['id'].astype(int)
        
        # drop rows specified in config based on id column
        if 'rows_to_remove' in config and isinstance(config['rows_to_remove'], list):
            df = df[~df['id'].isin(config['rows_to_remove'])]

        df.to_csv(os.path.join(os.getcwd() + config['output_dir']), index=False, encoding='utf-8', quoting="", escapechar='\\')
        
    
    
    
    def definitionsDB_Gen(self, config):
        tc_def_query()
        print("definitionsDB_Gen not implemented yet.")
        pass
    
    
    def companiesDB_Gen(self, config):
        tc_comp_query()
        print("companiesDB_Gen not implemented yet.")
        pass
    
    
    
    @staticmethod
    def format_txt(text: str) -> str:
        # Remove common unwanted phrases and symbols
        text = re.sub(r'(Copyright\s*©?|visit us @|Visit us @|©|®)', '', text)
        text = text.replace('""', '"')

        # Collapse all whitespace to single spaces early
        text = ' '.join(text.split())

        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Remove large blocks of periods (both with and without spaces)
        text = re.sub(r'\.{3,}', ' ', text)               # ... style
        text = re.sub(r'(?:\s*\.\s*){3,}', ' ', text)     # . . . . style

        # Remove large blocks of underscores
        text = re.sub(r'_{3,}', ' ', text)

        # Remove badly encoded or stray symbols (but keep common punctuation)
        text = re.sub(r'[^\w\s,.!?;:\'\"-]', ' ', text)

        # Remove long numeric sequences
        text = re.sub(r'\b\d{6,}\b', '', text)

        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)

        # Fix spaced-out letters: e.g., "I s t h i s" -> "Is this"
        def fix_spaced_words(match):
            return match.group(0).replace(' ', '')

        # Pattern: sequences of single letters with spaces between them
        text = re.sub(r'(?<=\b)(?:[A-Za-z]\s){2,}[A-Za-z](?=\b)', fix_spaced_words, text)

        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
