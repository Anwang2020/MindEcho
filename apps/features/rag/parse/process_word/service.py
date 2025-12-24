from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from spire.doc import Document as DocDocument


class WordParser:

    def __call__(self, file_path):
        if not file_path.endswith(".docx"):
            doc = DocDocument()
            doc.LoadFromFile(file_path)
            file_content = doc.GetText().strip().replace("\r\n", "\n")
            file_content = file_content.replace(
                "Evaluation Warning: The document was created with Spire.Doc for Python.", "")
            return [{"type": "paragraph", "text": file_content}]
        doc = Document(file_path)

        blocks = self.parse_blocks(doc)

        return blocks

    @staticmethod
    def iter_block_items(doc):
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    def parse_blocks(self, doc):
        blocks = []

        for item in self.iter_block_items(doc):

            # ---- 段落 ----
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue

                style = item.style.name if item.style else ""

                if style.startswith("Heading"):
                    level = int(style.replace("Heading", "").strip())
                    blocks.append({
                        "type": "title",
                        "level": level,
                        "text": text
                    })
                else:
                    blocks.append({
                        "type": "paragraph",
                        "text": text
                    })

            # ---- 表格 ----
            elif isinstance(item, Table):
                md_table = self.table_to_markdown(item)
                if md_table:
                    blocks.append({
                        "type": "table",
                        "text": md_table
                    })

        return blocks

    @staticmethod
    def table_to_markdown(table):
        rows = []
        max_cols = 0

        for row in table.rows:
            cells = [
                cell.text.strip().replace("\n", " ").replace("|", "\\|")
                for cell in row.cells
            ]
            rows.append(cells)
            max_cols = max(max_cols, len(cells))

        if not rows:
            return ""

        for row in rows:
            row.extend([""] * (max_cols - len(row)))

        md = []
        md.append("| " + " | ".join(rows[0]) + " |")
        md.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in rows[1:]:
            md.append("| " + " | ".join(row) + " |")

        return "\n".join(md)
