""" applications.py
Handles the reformatting of application PDF files into text and image formats.
"""

import re, io, os, uuid, fitz
import pandas as pd
from tqdm import tqdm
from pytesseract import image_to_string
from PIL import Image


class Applications_Reformat:
    def __init__(self, config):
        self.config = config
        
        #** Error counters **#
        self.TEXT_ERRORS = 0
        self.IMAGE_ERRORS = 0
        self.FILE_SKIPS = 0
        self.BAD_TEXT_FILES = 0
        self.OCR_ERRORS = 0
        self.SHORT_TEXT_FILES = []
        
        fitz.TOOLS.mupdf_display_errors(False)


    def run(self):

        df = pd.DataFrame(columns=['id','file_name','subfolder','text'])

        #** Get subdirectories **#
        subdirs = sorted([x[0] for x in os.walk(os.getcwd() + self.config['data_path'])][1:])
        subdir_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['data_path']) if x[1] != []][0])
        
        if self.config['starting_subfolder']:
            subdir_names, subdirs = self.starting_subfolder_manager(self.config['starting_subfolder'], subdir_names, subdirs)

        #** Iterate through subdirectories **#
        for subfolder_dir, subfolder_name in tqdm(zip(subdirs, subdir_names), total=len(subdirs), desc="Processing subfolders", dynamic_ncols=True):
            
            pdf_files = [f for f in os.listdir(subfolder_dir) if f.endswith('.pdf')]
            os.makedirs(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}", exist_ok=True)
            
            #** Iterate through PDF files **#
            for pdf_file in pdf_files:
                tqdm.write(f"Processing {pdf_file} in {subfolder_name}...")

                if os.path.exists(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}/{pdf_file[:-4]}.txt") and self.config.get('skip_existing', True):
                    self.FILE_SKIPS += 1
                    continue
                
                doc = fitz.open(subfolder_dir + '/' + pdf_file) 

                #** Extract images and/or text **#
                if self.config.get('pdf-img', True):
                    self.get_images_from_pdf(self, pdf_file, subfolder_name, doc)
                
                if self.config.get('pdf-txt', True):
                    text = self.get_text_from_pdf(pdf_file, subfolder_name, doc)
                    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                    df = pd.concat([df, pd.DataFrame([[str(uuid.uuid4()), pdf_file[:-4], subfolder_name, text]], 
                                                     columns=['id','file_name','subfolder','text'])], ignore_index=True)
                
                doc.close()
                
        
        #** Save dataframe and print summary **#
        df.to_csv(os.path.join(os.getcwd() + self.config['csv_output_path']), index=False, quotechar='"', escapechar='\\')
        
        self.print_summary(len(subdirs))
 


    def get_images_from_pdf(self, page, pdf_file, subfolder_name, doc):
        """ Extracts images from a PDF file and saves them in the specified output directory.
        """
        for page in doc:
            try:
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    pix = fitz.Pixmap(doc, img[0])
                    try:
                        pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB if not already
                    except:
                        pass
                    pix.save(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}/{pdf_file[:-4]}_{page.number+1}-{img_index+1}.png")
            except Exception as e:
                tqdm.write(f"Error extracting images from {pdf_file} on page {page.number+1}: {e}")
                self.IMAGE_ERRORS += 1


    def get_text_from_pdf(self, pdf_file, subfolder_name, doc, text=''):
        """ Extracts text from a PDF file, using OCR if necessary, and saves it in the specified output directory.
        An x-y coordinate based method is used to group text blocks into columns for better reading order.
        """
        
        vertical_tolerance = 12  
        column_tolerance = 50    
        all_text = ''
        
        #** Iterate pages **#
        for index, page in enumerate(doc):
            page_text = ''

            try:
                blocks = page.get_text('blocks')
                text_blocks = [b for b in blocks if b[6] == 0]

                if not text_blocks:
                    continue

                #** Group blocks into columns and sort **#
                text_blocks.sort(key=lambda b: -b[3])
                columns = []
                for block in text_blocks:
                    x0 = block[0]
                    assigned = False
                    for col in columns:
                        if abs(col['x'] - x0) < column_tolerance:
                            col['blocks'].append(block)
                            assigned = True
                            break
                    if not assigned:
                        columns.append({'x': x0, 'blocks': [block]})

                columns.sort(key=lambda c: c['x'])

                for col in columns:
                    col['blocks'].sort(key=lambda b: b[3])
                    for block in col['blocks']:
                        page_text += block[4].strip() + '.\n\n'

            except Exception as e:
                tqdm.write(f"Error extracting text from {pdf_file} on page {index+1}: {e}")
                self.TEXT_ERRORS += 1

            #** OCR if poor text quality or no text **#
            if self.is_bad_text(page_text) | len(page_text.strip()) < 10:
                tqdm.write(f"Bad text detected, trying OCR...")
                try:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = image_to_string(img)
                    self.BAD_TEXT_FILES += 1
                    tqdm.write(f"OCR successful for {pdf_file} on page {index+1}. Length: {len(page_text)} characters.")
                
                except Exception as e:
                    tqdm.write(f"OCR failed for {pdf_file} on page {index+1}: {e}")
                    self.OCR_ERRORS += 1

            all_text += page_text
        
        if len(all_text.strip()) < 2:
            self.SHORT_TEXT_FILES.append(pdf_file)
            
    
        
        #** Save text **#
        if self.config.get('allow_empty_text_files', True) or all_text.strip():
            output_path = os.path.join(os.getcwd() + self.config['output_path'], subfolder_name)
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f"{pdf_file[:-4]}.txt"), 'w', encoding='utf-8') as nf:
                nf.write(all_text)
        
        return all_text


    @staticmethod
    def is_bad_text(text: str) -> bool:
        """ Indicate if text contains a high proportion of non-standard characters.
        """
        clean = re.sub(r'[\s\w,.!?;:\'"\(\)\[\]\{\}-]', '', text)  # non-standard chars
        garbage_ratio = len(clean) / max(len(text), 1)
        return garbage_ratio > 0.2  # i.e., true/false


    @staticmethod
    def starting_subfolder_manager(starting_subfolder: str, subdir_names: list, subdirs: list) -> tuple:
        """ Adjusts subdirs and subdir_names to start from the specified starting_subfolder.
        """
        if starting_subfolder not in subdir_names:
            raise ValueError(f"Starting subfolder '{starting_subfolder}' not found in subdir_names\n Check config.yaml and data.")
        sbf_idx = subdir_names.index(starting_subfolder)
        subdirs = subdirs[sbf_idx:]
        subdir_names = subdir_names[sbf_idx:]    
        return subdir_names, subdirs


    def print_summary(self, len_subdirs: int) -> None:
        """ Prints a summary of the processing results.
        """
        print(f"\nFinished processing {len_subdirs} sub-folders.")
        print(f"\nBad text files: {self.BAD_TEXT_FILES}")
        if self.config.get('pdf-txt', True):
            print(f"\nText extraction errors: {self.TEXT_ERRORS}")
            print(f"\nOCR errors: {self.OCR_ERRORS}")
        if self.config.get('pdf-txt', True):
            print(f"\nShort text files (less than 2 characters): {self.SHORT_TEXT_FILES}")
        if self.config.get('pdf-img', True):
            print(f"\nImage extraction errors: {self.IMAGE_ERRORS}")
        if self.config.get('skip_existing', True):
            print(f"\nFiles skipped (already processed): {self.FILE_SKIPS}")
