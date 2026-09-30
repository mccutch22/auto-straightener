import unittest
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.manual import adjust, manual_geometry


class ManualTests(unittest.TestCase):
    def test_manual_bypasses_line_detection_and_never_exposes_padding(self):
        for width,height in [(1200,900),(900,1200)]:
            source=np.full((height,width,3),255,np.uint8)
            for rotation,vertical,horizontal in [(0,60,0),(0,-60,0),(3,20,-30),(-15,-100,100),(15,100,-100)]:
                with self.subTest(size=(width,height),values=(rotation,vertical,horizontal)):
                    result=adjust(source,rotation,vertical,horizontal)
                    self.assertEqual(result.applied_mode,'manual')
                    self.assertEqual(result.corrected_bgr.min(),255)
                    h,w=result.corrected_bgr.shape[:2]
                    self.assertEqual(w*height,h*width)
                    self.assertLess(w,width)
                    matrix,_,_=manual_geometry(width,height,rotation,vertical,horizontal)
                    inv=np.linalg.inv(matrix)
                    for x,y in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
                        p=inv@[x,y,1];sx,sy=p[:2]/p[2]
                        self.assertGreaterEqual(sx,4);self.assertGreaterEqual(sy,4)
                        self.assertLessEqual(sx,width-5);self.assertLessEqual(sy,height-5)

    def test_zero_and_invalid_parameters(self):
        source=np.full((600,800,3),123,np.uint8)
        result=adjust(source,0,0,0)
        self.assertEqual(result.applied_mode,'none');self.assertTrue(np.array_equal(source,result.corrected_bgr))
        for values in [(16,0,0),(0,101,0),(0,0,-101),(float('nan'),0,0)]:
            with self.assertRaises(ValueError):adjust(source,*values)

    def test_private_endpoint(self):
        client=TestClient(app);b=BytesIO();Image.new('RGB',(800,600),'white').save(b,format='PNG')
        with patch('app.main.API_TOKEN','secret'):
            self.assertEqual(client.post('/v1/adjust?rotation=2',content=b.getvalue()).status_code,401)
            self.assertEqual(client.post('/v1/adjust?rotation=99',content=b.getvalue(),headers={'Authorization':'Bearer secret'}).status_code,422)
            r=client.post('/v1/adjust?vertical=30&rotation=-1',content=b.getvalue(),headers={'Authorization':'Bearer secret'})
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['mode'],'manual');self.assertIn('imageBase64',r.json())
