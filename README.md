# d2r_nerf_test
nia_가공데이터_테스트를 위한 레포지토리입니다. 

- 01/27 Update - Origin [Page](https://github.com/summeryoo/d2r_nerf_test)

x_focalLength_inPixels y_focalLength_inPixels를 사용하겠다고 하였지만
width의 여백이 너무 많아 비효율적이여서 height와 동일하게끔 width를 height 크기만큼 
crop 하였고(1920->1080) 따라서 focal도 y_focal 값 하나만 사용하였습니다. 이점염두해주세요!

- 02/03 Update

crop과 resize한걸 고려해서 x_focalLength_inPixels y_focalLength_inPixels를 그만큼 resize했습니다.

NeRF_notebook_colmap.ipynb의 경우 focal length를 하나만 사용한 원래 코드를 따라간 colmap pose 기반의 코드

NeRF_notebook_d2r.ipynb의 경우 focal length를 x_focalLength_inPixels y_focalLength_inPixels를 고려해서 변경하고
camera pose를 colmap과 align될 수 있게 수정한 코드가 포함되어 있습니다.