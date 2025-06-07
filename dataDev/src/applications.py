"""
This module defines the Applications class, which reformats the data into a useful format for modelDev.
"""

import pypdf
import os
from tqdm import tqdm
import fitz

class Applications_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Applications_Reformat initialized with config:\n {self.config}')

    def run(self):

        # Get the list of subfolders in the data path
        subfolder_directories = sorted([x[0] for x in os.walk(os.getcwd() + self.config['data_path'])][1:])
        subfolder_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['data_path']) if x[1] != []][0])
        if self.config.get('debug', True):
            print(f">>Found {len(subfolder_directories)} subfolders in {self.config['data_path']}")

        # Iterate through each subfolder and respective PDF files
        text_errors, img_errors = 0, 0 # Error counters
        for subfolder_dir, subfolder_name in tqdm(zip(subfolder_directories, subfolder_names)):
            
            # Get all PDF files in the current subfolder
            pdf_files = [f for f in os.listdir(subfolder_dir) if f.endswith('.pdf')]
            if self.config.get('debug', True):
                 tqdm.write(f">>Processing {len(pdf_files):03} PDF files in {subfolder_name}.")
            
            # Create output directories if they do not exist
            if not os.path.exists(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}"):
                os.makedirs(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}")
            
            # Process each PDF file in the subfolder
            for pdf_file in pdf_files:
                
                # pdf-txt
                if self.config.get('pdf-txt', True):
                    try:    
                        reader = pypdf.PdfReader(subfolder_dir + '/' + pdf_file)
                        nf = open(f'{os.getcwd()}{self.config["output_path"]}/{subfolder_name}/{pdf_file[:-4]}.txt', 'w')
                        nf.write('\n'.join([page.extract_text() for page in reader.pages]))
                        nf.close()
                    except Exception as e:
                        text_errors += 1
                
                # pdf-img
                if self.config.get('pdf-img', True):
                    try:    
                        doc = fitz.open(subfolder_dir + '/' + pdf_file)
                        for i, page in enumerate(doc):
                            images = page.get_images(full=True)
                            for img_index, img in enumerate(images):
                                pix = fitz.Pixmap(doc, img[0])
                                if pix.alpha:  # has transparency
                                    pix = fitz.Pixmap(fitz.csRGB, pix)  # remove alpha
                                pix.save(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}/{pdf_file[:-4]}_{i+1}-{img_index+1}.png")
                        doc.close()
                    except Exception as e:
                        img_errors += 1
            
        # Print summary of errors
        if self.config.get('debug', True):
            print(f">>Finished processing {len(subfolder_directories)} subfolders.")
            if text_errors > 0:
                print(f">>Text extraction errors: {text_errors}")
            if img_errors > 0:
                print(f">>Image extraction errors: {img_errors}")
            if text_errors == 0 and img_errors == 0:
                print(">>No errors encountered during processing.")
