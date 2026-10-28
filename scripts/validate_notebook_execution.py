"""Ejecuta notebooks de confianza en proceso, sin requerir sockets de Jupyter."""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    notebook=nbformat.read(path,as_version=4)
    shell=InteractiveShell(user_ns={})
    count=0
    for cell in notebook.cells:
        if cell.cell_type!='code': continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path.name}: celda {count}') from (result.error_before_exec or result.error_in_exec)
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    nbformat.validate(notebook);nbformat.write(notebook,path)
    print('OK',path.name,flush=True)
