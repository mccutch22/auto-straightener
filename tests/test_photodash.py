import unittest
from io import BytesIO
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.main import require_auth, health
from app.photodash import process
from app.straighten import auto_straighten_verticals, _warp_and_crop, _rotation_homography
from test_straighten import architectural_grid


class PhotoDashTests(unittest.TestCase):
    def test_handheld_preset_corrects_stronger_skew_in_both_directions(self):
        for sign in [-1, 1]:
            with self.subTest(sign=sign):
                image = np.full((900, 1200, 3), 240, np.uint8)
                for x in range(120, 1200, 120):
                    cv2.line(image, (round(600+(x-600)*(1-sign*.13)), 40),
                             (round(600+(x-600)*(1+sign*.13)), 860), (20,20,20), 7)
                baseline = auto_straighten_verticals(image, max_dimension=1400)
                self.assertEqual(baseline.applied_mode, 'none')
                result = process(cv2.imencode('.png', image)[1].tobytes())
                self.assertEqual(result['mode'], 'perspective', result['warnings'])
                validation = result['geometry']['validation']
                self.assertLess(validation['candidate_output_vertical_error_deg'],
                                validation['initial_vertical_error_deg'])
                self.assertLessEqual(result['cropFraction'], .234)
                self.assertEqual(result['width'] * 900, result['height'] * 1200)

    def test_handheld_preset_preserves_ambiguous_single_line(self):
        image = np.full((800, 800, 3), 230, np.uint8)
        cv2.line(image, (360, 60), (430, 740), (20,20,20), 6)
        result = process(cv2.imencode('.png', image)[1].tobytes())
        self.assertEqual(result['outcome'], 'unchanged', result['warnings'])
        self.assertNotIn('imageBase64', result)

    def test_binary_http_auth_busy_and_success(self):
        client=TestClient(app);buffer=BytesIO();Image.new('RGB',(800,600),'white').save(buffer,format='JPEG')
        with patch('app.main.API_TOKEN','secret'):
            self.assertEqual(client.post('/v1/straighten',content=buffer.getvalue()).status_code,401)
            response=client.post('/v1/straighten',content=buffer.getvalue(),headers={'Authorization':'Bearer secret'})
            self.assertEqual(response.status_code,200);self.assertEqual(response.json()['outcome'],'unchanged')
            from app.photodash import busy
            busy.acquire()
            try:self.assertEqual(client.post('/v1/straighten',content=buffer.getvalue(),headers={'Authorization':'Bearer secret'}).status_code,429)
            finally:busy.release()
    def test_requires_token_even_when_unconfigured(self):
        for configured, supplied in [(None, None), ('secret', None), ('secret', 'Bearer wrong')]:
            with patch('app.main.API_TOKEN', configured), self.assertRaises(HTTPException):
                require_auth(supplied)
        with patch('app.main.API_TOKEN', 'secret'):
            require_auth('Bearer secret')

    def test_corrupt_and_oversized_images_rejected(self):
        with self.assertRaises(HTTPException):
            process(b'not a jpeg')
        buffer=BytesIO();Image.new('RGB',(6000,5000)).save(buffer,format='PNG')
        with self.assertRaises(HTTPException):
            process(buffer.getvalue())

    def test_no_op_does_not_reencode_and_normalizes_orientation(self):
        buffer=BytesIO();photo=Image.new('RGB',(600,400),'white');exif=photo.getexif();exif[274]=6;photo.save(buffer,format='JPEG',exif=exif)
        result=process(buffer.getvalue())
        self.assertEqual(result['outcome'],'unchanged')
        self.assertNotIn('imageBase64',result)
        self.assertEqual((result['width'],result['height']),(400,600))

    def test_both_rotation_directions_and_portrait(self):
        for width,height in [(1200,900),(900,1200)]:
            for angle in [-3,3]:
                with self.subTest(width=width,angle=angle):
                    image=architectural_grid(width,height)
                    distorted=cv2.warpAffine(image,cv2.getRotationMatrix2D((width/2,height/2),angle,1),(width,height))
                    result=auto_straighten_verticals(distorted,mode='level',max_dimension=900)
                    self.assertEqual(result.applied_mode,'level',result.debug)
                    self.assertLess(result.correction_angle_deg*angle,0)
                    self.assertLess(abs(result.correction_angle_deg+angle),.3)
                    h,w=result.corrected_bgr.shape[:2];self.assertEqual(w*height,h*width)

    def test_mpo_uses_primary_image_not_auxiliary_frame(self):
        buffer=BytesIO();Image.new('RGB',(800,600),'white').save(buffer,format='MPO',save_all=True,append_images=[Image.new('RGB',(400,300),'black')])
        result=process(buffer.getvalue());self.assertEqual((result['width'],result['height']),(800,600))

    def test_crop_contains_only_real_source_pixels(self):
        for angle in [-5,5]:
            image=np.full((900,1200,3),255,np.uint8)
            output,loss,debug=_warp_and_crop(image,_rotation_homography(1200,900,angle),'crop')
            self.assertEqual(output.min(),255)
            h,w=output.shape[:2];self.assertEqual(w*900,h*1200)
            inverse=np.linalg.inv(np.array(debug['effective_homography']))
            corners=cv2.perspectiveTransform(np.array([[[0.,0.],[w-1.,0.],[w-1.,h-1.],[0.,h-1.]]]),inverse)[0]
            self.assertGreaterEqual(corners.min(),4)
            self.assertLessEqual(corners[:,0].max(),1195)
            self.assertLessEqual(corners[:,1].max(),895)
            self.assertGreater(loss,0)

    def test_perspective_both_directions(self):
        for sign in [-1,1]:
            image=np.full((900,1200,3),240,np.uint8)
            for x in range(120,1200,120):
                cv2.line(image,(round(600+(x-600)*(1-sign*.09)),40),(round(600+(x-600)*(1+sign*.09)),860),(20,20,20),7)
            result=auto_straighten_verticals(image,minimum_confidence=.25,max_dimension=900)
            self.assertTrue(result.perspective_applied,result.debug)
            self.assertLess(result.debug['validation']['candidate_output_vertical_error_deg'],result.debug['validation']['initial_vertical_error_deg'])

if __name__=='__main__':unittest.main()
