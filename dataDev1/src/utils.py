def get_config(config_path="dataDev1/config.yaml"):
    """Load configuration settings from a YAML file
    """
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Script that runs dataDev1")
    parser.add_argument('--data_source', '-ds', type=str, required=True, help='Source of the data: one of Applications, Companies, or Definitions')
    data_source = parser.parse_args().data_source
    if data_source.lower() not in ['applications', 'a', 'companies', 'c', 'definitions', 'd']:
        raise ValueError("Invalid data source (ds). Choose from: Applications, Companies, or Definitions.")
    if data_source.lower() == 'a': data_source = 'Applications'
    elif data_source.lower() == 'c': data_source = 'Companies'
    elif data_source.lower() == 'd': data_source = 'Definitions'
    else: data_source = data_source.capitalize()
    return data_source

def dataDev1_query(data_source) -> None:
    while True:
        user_input = input(f"\n         ********************* >>>>> -- DataDev1 -- <<<<< *********************          \n\nThis script will transform original data into decomposed data suitable for further development.\nYou have opted to decompose the **{data_source}** database.\nDo you wish to continue? (Y/N): ").strip().lower()
        if user_input == 'y': break
        elif user_input == 'n': exit()
        else: print("\nInvalid input. Please enter Y or N.")
          
def applicationsDB_initialisation_query():
    """Prompt user for applications DB processing confirmation.
    """
    while True:
        user_input = input("\nThis will decompose .pdf files into its contstituent parts via .txt and .png files.\nDo you wish to continue processing the applications database? (Y/N): ").strip().lower()
        if user_input == 'y': break
        elif user_input == 'n': exit()
        else: print("\nInvalid input. Please enter Y or N.")

def companiesDB_initialisation_query():
    """Prompt user for companies DB processing confirmation.
    """
    while True:
        user_input = input("This will retrieve data from copmany websites and save to .txt files.\nDo you wish to continue processing the companies database? (Y/N): ").strip().lower()
        if user_input == 'y': break
        elif user_input == 'n': exit()
        else: print("\nInvalid input. Please enter Y or N.")

def definitionsDB_initialisation_query():
    """Prompt user for definitions DB processing confirmation.
    """
    while True:
        user_input = input("This will decompose a word document, producing a .csv with relevant information.\nDo you wish to continue processing the definitions database? (Y/N): ").strip().lower()
        if user_input == 'y': break
        elif user_input == 'n': exit()
        else: print("\nInvalid input. Please enter Y or N.")

