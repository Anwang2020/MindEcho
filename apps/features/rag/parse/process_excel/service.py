import pandas as pd


class ExcelParser:
    def __call__(self, filename, **kwargs):
        file_format = filename.split('.')[-1]
        engine = 'openpyxl' if file_format == 'xlsx' else 'xlrd'
        blocks = []
        with pd.ExcelFile(filename, engine=engine) as excel_file:
            for sheet_name in excel_file.sheet_names:
                blocks.append({
                    'type': "title",
                    'text': sheet_name
                })
                df = pd.read_excel(excel_file, sheet_name=sheet_name).fillna('')
                blocks.append({
                    'type': "table",
                    'text': df.to_markdown()
                })
        return blocks


if __name__ == '__main__':
    excel_parser = ExcelParser()
    excel_parser(r'C:\Users\27970\Desktop\tro_ai_prompts_pl.xls')
