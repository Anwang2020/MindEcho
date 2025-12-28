from fastapi import APIRouter
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pathlib import Path
import os
import json

from .service import rag_workflow

router = APIRouter(tags=["rag"], prefix="/rag")
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
TABLE_REGISTRY_PATH = BASE_DIR / 'table_registry.json'


@router.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    file_paths = []
    response = 'upload failed'
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file_paths.append(file_path)
            with open(file_path, "wb") as f:
                f.write(await file.read())

        response = await rag_workflow.preprocess(file_paths)

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        return {"message": response}


@router.get("/validate")
def validate_description():
    with open(TABLE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        description = {v: k.split('_')[:-1][0] for k, v in registry[0].items()}
        return description


@router.post("/update")
def validate_description(update_dict: dict):
    with open(TABLE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)[0]
    description = {v: k for k, v in registry.items()}
    response = {}
    for update_item in update_dict:
        try:
            if update_item in description:
                registry_key = description[update_item]
                embedding_dim = registry_key.split('_')[-1]
                registry.pop(registry_key)
                registry[update_dict[update_item] + '_' + embedding_dim] = update_item
                response.update({update_item: {"success": 'ture', "message": update_dict[update_item]}})
            else:
                response.update({update_item: {"success": 'false', "message": 'no such id'}})
            with open(TABLE_REGISTRY_PATH, 'w', encoding='utf-8') as f:
                json.dump([registry], f, ensure_ascii=False, indent=4)
        except Exception as e:
            response.update({update_item: {"success": 'false', "message": str(e)}})
    return response


@router.post("/delete")
def validate_description(delete_list: List[str]):
    from lancedb import connect
    db = connect(f"{BASE_DIR}/embedding_db")
    with open(TABLE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)[0]
    description = {v: k for k, v in registry.items()}
    response = {}
    try:
        for delete_item in delete_list:
            if delete_item not in description:
                response.update({"success": 'false', "message": 'no such id'})
                continue
            registry_key = description[delete_item]
            registry.pop(registry_key)
            response.update({"success": 'ture', "message": f'delete {delete_item}'})
            db.drop_table(delete_item)
        with open(TABLE_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump([registry], f, ensure_ascii=False, indent=4)
    except Exception as e:
        response.update({"success": 'false', "message": str(e)})
    return response

# if __name__ == '__main__':
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8001)
