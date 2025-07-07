import os
from tqdm import tqdm
import fitz
import logging

class Applications_Reformat:
    def __init__(self, config):
        self.config = config
        fitz.TOOLS.mupdf_display_errors(False)

    def run(self):

        # Get the list of subfolders in the data path
        subdirs = sorted([x[0] for x in os.walk(os.getcwd() + self.config['data_path'])][1:])
        subdir_names = sorted([x[1] for x in os.walk(os.getcwd() + self.config['data_path']) if x[1] != []][0])

        if self.config['starting_subfolder']:
            subdir_names, subdirs = self.starting_subfolder_manager(self.config['starting_subfolder'], subdir_names, subdirs)

        # Iterate through each subfolder and respective PDF files
        text_errors, img_errors = 0, 0 # Error counters
        for subfolder_dir, subfolder_name in tqdm(zip(subdirs, subdir_names), 
                                                  total=len(subdirs),
                                                    desc="Processing subfolders",
                                                    dynamic_ncols=True,
                                                    colour='blue'):
            
            # Get all PDF files in the current subfolder
            pdf_files = [f for f in os.listdir(subfolder_dir) if f.endswith('.pdf')]

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
                
            
        print(f"Finished processing {len(subdirs)} subfolders.")
        print(f"Text extraction errors: {text_errors}")
        print(f"Image extraction errors: {img_errors}")



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