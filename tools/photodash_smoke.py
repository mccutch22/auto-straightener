"""Local representative-photo regression; originals are never overwritten."""
import sys, json, time, base64, ctypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.photodash import process
from PIL import Image,ImageDraw
from io import BytesIO

root=Path(__file__).resolve().parents[3]
output=Path(__file__).resolve().parents[1]/'evaluation'/'photodash-2026-09-29'
output.mkdir(parents=True,exist_ok=True)
files=list((root/'public/property/4302-mcalister-park').glob('*.jpg'))[:24]
files+=list((root/'work/thurmont-exposure-audit').glob('318e9508*-0.jpg'))[:1]
records=[];tiles=[]
for i,path in enumerate(files):
    start=time.monotonic()
    try:
        result=process(path.read_bytes());encoded=result.pop('imageBase64',None)
        if encoded:(output/f'{i+1:02d}-after.jpg').write_bytes(base64.b64decode(encoded))
        before=Image.open(path).convert('RGB');after=Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB') if encoded else before.copy()
        tile=Image.new('RGB',(800,330),'#eeeeee')
        for n,picture in enumerate([before,after]):
            picture.thumbnail((395,292));tile.paste(picture,(n*400+(395-picture.width)//2,(292-picture.height)//2))
        ImageDraw.Draw(tile).text((8,302),f'{i+1}. {path.name} | {result["outcome"]} {result["mode"]} | crop {result["cropFraction"]:.1%}',fill='black')
        tiles.append(tile);records.append(dict(file=str(path),seconds=round(time.monotonic()-start,2),**result))
        print(f'{i+1}/{len(files)} {result["outcome"]} {result["mode"]}',flush=True)
    except Exception as e:records.append(dict(file=str(path),error=str(e)));print('ERROR',path.name,str(e),flush=True)
for page in range((len(tiles)+7)//8):
    sheet=Image.new('RGB',(1600,1320),'white')
    for j,tile in enumerate(tiles[page*8:page*8+8]):sheet.paste(tile,((j%2)*800,(j//2)*330))
    sheet.save(output/f'contact-{page+1}.jpg')
(output/'results.json').write_text(json.dumps(records,indent=2))
print(json.dumps(dict(total=len(records),corrected=sum(r.get('outcome')=='corrected' for r in records),errors=sum('error'in r for r in records),seconds=sum(r.get('seconds',0) for r in records))))
