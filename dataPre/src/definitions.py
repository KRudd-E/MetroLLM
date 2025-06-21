"""
This module defines the Definitions class, which reformats the data into a useful format for modelDev.
"""
from docx import Document
import pandas as pd
import time


class Definitions_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Definitions_Reformat initialized with config:\n {self.config}')

    def run(self):
        doc = Document(self.config['data_path'])
        data = []

        current_letter = ''
        name = ''
        definition = ''
        counter = 1

        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            style = para.style.name

            # Heading 1: grouping letter
            if 'Heading 1' in style:
                current_letter = para.text.strip().upper()
                counter = 1

            # Heading 2: name
            elif 'Heading 2' in style:
                name = para.text.strip()

            # Normal: definition
            elif 'Normal' in style:
                definition = para.text.strip()
                if current_letter and name:
                    if current_letter == '123':
                        entry_id = f"{1}{counter:03d}"
                    else:
                        entry_id = f"{current_letter}{counter:03d}"
                    data.append({
                        'id': entry_id,
                        'name': name,
                        'definition': definition
                    })
                    counter += 1
                    name = ''
                    definition = ''

        df = pd.DataFrame(data).set_index('id')

        output_path = self.config.get('output_path')

        df.to_csv(output_path, encoding='utf-8-sig')
        if self.config.get('debug', True):
            print(f'Data reformatted and saved to {output_path}')