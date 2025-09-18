from docx import Document
import pandas as pd
import re


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
        collecting_definition = False

        #** Iterate over paragraphs **#
        for para in doc.paragraphs:
            if not para.text.strip():
                continue

            style: str = para.style.name # type: ignore

            #** Heading 1: grouping letter **#
            if 'Heading 1' in style:
                if name and definition:
                    cleaned_def = self.clean_definition(definition)
                    entry_id = f"{1}{counter:07d}" if current_letter == '123' else f"{current_letter}{counter:07d}"
                    data.append({
                        'id': entry_id,
                        'name': name.strip(),
                        'definition': cleaned_def
                    })
                    counter += 1
                    name = ''
                    definition = ''
                current_letter = para.text.strip().upper()
                counter = 1
                collecting_definition = False

            #** Heading 2: Definition name **#
            elif 'Heading 2' in style:
                if name and definition:
                    cleaned_def = self.clean_definition(definition)
                    entry_id = f"{1}{counter:07d}" if current_letter == '123' else f"{current_letter}{counter:07d}"
                    data.append({
                        'id': entry_id,
                        'name': name.strip(),
                        'definition': cleaned_def
                    })
                    counter += 1
                name = para.text.strip()
                definition = ''
                collecting_definition = True

            #** Normal text: Definition content **#
            elif collecting_definition:
                definition += ' ' + para.text.strip()

        # Handle final entry
        if name and definition:
            cleaned_def = self.clean_definition(definition)
            entry_id = f"{1}{counter:07d}" if current_letter == '123' else f"{current_letter}{counter:07d}"
            data.append({
                'id': entry_id,
                'name': name.strip(),
                'definition': cleaned_def
            })

        #** Save to CSV **#
        df = pd.DataFrame(data).set_index('id')
        output_path = self.config.get('output_path')

        df.to_csv(output_path, encoding='utf-8-sig')
        if self.config.get('debug', True):
            print(f'Data reformatted and saved to {output_path}')
            
            
    def clean_definition(self, text):
        # Replace +word+[word] → word
        text = re.sub(r"\+([^\[\]+]+)\+\[\1\]", r"\1", text)
        # Remove [anything]
        text = re.sub(r"\[[^\[\]]*\]", "", text)
        # Remove standalone +...+ markup
        text = re.sub(r"\+([^\+]+)\+", r"\1", text)
        return text.strip()