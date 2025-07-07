import pypdf
import os
from tqdm import tqdm
import fitz
import logging

class Applications_Reformat:
    def __init__(self, config):
        self.config = config
        if self.config.get('debug', True):
            print(f'Applications_Reformat initialized with config:\n {self.config}')
        else:
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            fitz.TOOLS.mupdf_display_errors(False)

    def run(self):

        # Get the list of subfolders in the data path
        subfolder_directories = sorted([x[0] for x in os.walk(os.getcwd() + self.config['data_path'])][1:])
        subfolder_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['data_path']) if x[1] != []][0])
        if self.config.get('debug', True):
            print(f"Found {len(subfolder_directories)} subfolders in {self.config['data_path']}")

        # Iterate through each subfolder and respective PDF files
        text_errors, img_errors = 0, 0 # Error counters
        for subfolder_dir, subfolder_name in tqdm(zip(subfolder_directories, subfolder_names), 
                                                  total=len(subfolder_directories),
                                                    desc="Processing subfolders",
                                                    dynamic_ncols=True,
                                                    colour='blue'):
            
            # Get all PDF files in the current subfolder
            pdf_files = [f for f in os.listdir(subfolder_dir) if f.endswith('.pdf')]
            if self.config.get('debug', True):
                 tqdm.write(f"Processing {len(pdf_files):03} PDF files in {subfolder_name}.")
            
            # Create output directories if they do not exist
            if not os.path.exists(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}"):
                os.makedirs(f"{os.getcwd()}{self.config['output_path']}/{subfolder_name}")
            
            # Process each PDF file in the subfolder
            for pdf_file in pdf_files:
                
                doc = fitz.open(subfolder_dir + '/' + pdf_file)
                text = '' 

                for page in doc: 
                    # Extract images
                    if self.config.get('pdf-img', True):
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
                            img_errors += 1
                    
                    # Extract text
                    if self.config.get('pdf-txt', True):
                        try:
                            blocks = page.get_text('blocks')
                            blocks.sort(key=lambda b: (b[1], b[0]))  # top-down, left-right

                            for block in blocks:
                                if block[6] == 0:
                                    text += block[4].strip() + '\n\n'  # Add spacing between paragraphs

                        except Exception as e:
                            tqdm.write(f"Error extracting text from {pdf_file} on page {page.number+1}: {e}")
                            text_errors += 1

                # Save text after processing all pages
                if text:
                    with open(f'{os.getcwd()}{self.config["output_path"]}/{subfolder_name}/{pdf_file[:-4]}.txt', 'w') as nf:
                        nf.write(text)

                doc.close()
                
            
        print(f"Finished processing {len(subfolder_directories)} subfolders.")
        print(f"Text extraction errors: {text_errors}")
        print(f"Image extraction errors: {img_errors}")
