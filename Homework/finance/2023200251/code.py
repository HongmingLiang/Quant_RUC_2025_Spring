from docxtpl import DocxTemplate
from docx2pdf import convert
import pandas as pd
import os

df = pd.read_excel(r"C:\Users\lldlu\Desktop\excel list.xlsx")
df2 = pd.read_excel(r"C:\Users\lldlu\Desktop\university.xlsx")

University = list(df2["University"][0:30])
Major = list(df['Major'][0:3])
Journal= list(df['Journal'][0:10])
Skill = list(df['Skill'][0:5])

target_folder = r"C:\Users\lldlu\Desktop\words"

for i in range(len(Major)):
    for j in range(len(University)):
        template = DocxTemplate(r"C:\Users\lldlu\Desktop\template.docx")
        content = {
            'university': University[j],
            'research_area': Major[i],
            'journal1': Journal[3*i],
            'journal2': Journal[3*i+1],
            'journal3': Journal[3*i+2],
            'skill1': Skill[0],
            'skill2': Skill[1],
            'skill3': Skill[2],
            'skill4': Skill[3],
            'skill5': Skill[4]
        }
        template.render(content)
        file_name = f"{University[j]}_{Major[i]}.docx"
        output_path = os.path.join(target_folder, file_name)
        template.save(output_path)

pdf_folder = r"C:\Users\lldlu\Desktop\pdfs"

try:
    convert(target_folder, pdf_folder)
    print("Convert successfully")
except Exception as e:
    print(f"An error occurred: {e}")
