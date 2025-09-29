# %%
# Coded by 2023200190, with the assistance of Doubao AI
# "docxtpl" is exactly much better than "python-docx" to complete this task. This is what I learn after being tortured by "python-docx".

# Import the required packages
import pandas as pd
from docxtpl import DocxTemplate
import os

# %%
# Read the data from two excel list
df_universities = pd.read_excel("universities_30.xlsx")
df_information = pd.read_excel("areas_journals_skills.xlsx")

# %%
def application_generator(df_universities,df_information,template):
    '''
    This function is used to generate application letter with specific format of information
    '''
    
    # Store the data in a dictionary, which contains all of the substitution for one peice of letter
    # Then store those dictionaries in to a list, which contains 90 items
    information_bank = []
    for university in df_universities["UNIVERSITIES"]:
        for number in range(0,(len(df_information["column0"])-1)):
            template_fill = {"university_names":university,"interested_research_areas":df_information["column0"][number],"top_journals":df_information["column1"][number],"skills":df_information["column1"][3]}
            information_bank.append(template_fill)
    script_dir = os.path.dirname(os.path.abspath(os.getcwd())) # Locate the path of the work folder
    subfolder_name = "application_letters_90_1" # Name the subfolder to store our letters
    subfolder_path = os.path.join(script_dir, subfolder_name) # join the full name of the path
    os.makedirs(subfolder_path, exist_ok=True) # Determin whether the folder is already existed or not, if not then create a new one
    for bundle in information_bank:
        template.render(bundle)
        university_name = bundle["university_names"] # Give each letter a name with university
        interested_research_area = bundle["interested_research_areas"] # Give each letter a name with area
        output_filename = f"{university_name}_{interested_research_area}_application_letter.docx" # Generate the full name
        save_path = os.path.join(subfolder_path, output_filename) # Generate the full path
        template.save(save_path) # Create one letter for every dictionary
    print(f"The application letters are saved at: {subfolder_path}") # When the whole generation is finished, print where the folder is

# %%
# Read the data from two excel list and template
df_universities = pd.read_excel("universities_30.xlsx")
df_information = pd.read_excel("areas_journals_skills.xlsx")
template = DocxTemplate("application_letter_template _doxtpl.docx")

# Use the function to generate application letter
application_generator(df_universities,df_information,template)



