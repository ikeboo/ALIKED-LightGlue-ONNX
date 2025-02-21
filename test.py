import cv2
from aliked_lightglue_onnx import AlikedLightGlueONNX

# load onnx model
aliked_path = "onnx/aliked-n16rot-top2k-640.onnx"
lightglue_path = "onnx/lightglue_for_aliked.onnx"
model = AlikedLightGlueONNX(aliked_path,lightglue_path)

# load images
img0 = cv2.imread("assets/st_pauls_cathedral/1.jpg")
img1 = cv2.imread("assets/st_pauls_cathedral/3.jpg")

# infer
m_kpts0, m_kpts1, score = model(img0,img1)

# calculate transformed image
ret,transformed_img = model.transform_image(img0,img1,
                                            m_kpts0,m_kpts1)

# draw matches
matches_img = model.draw_matches(img0,img1,m_kpts0,m_kpts1,score)

# visualize matches and transformed image
model.show_result(img0,img1,m_kpts0,m_kpts1,score)