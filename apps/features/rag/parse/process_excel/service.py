import pandas as pd


class ExcelParser:
    def __call__(self, filename, **kwargs):
        file_format = filename.split('.')[-1]
        engine = 'openpyxl' if file_format == 'xlsx' else 'xlrd'
        excel_file = pd.ExcelFile(filename, engine=engine)
        sheet_names = excel_file.sheet_names
        blocks = []
        for sheet_name in sheet_names:
            blocks.append({
                'type': "title",
                'text': sheet_name
            })
            df = pd.read_excel(filename, sheet_name=sheet_name, engine=engine).fillna('')
            df2md = df.to_markdown()
            blocks.append({
                'type': "table",
                'text': df2md
            })
        return blocks


if __name__ == '__main__':
    excel_parser = ExcelParser()
    excel_parser(r'C:\Users\27970\Desktop\tro_ai_prompts_pl.xls')
