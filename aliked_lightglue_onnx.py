from typing import Tuple
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

class AlikedLightGlueONNX:
    def __init__(self, 
                 aliked_path: str,
                 lightglue_path:str, 
                 score_thresh: float = 0.2):
        """
        Args:
            aliked_path: str path to the aliked onnx model
            lightglue_path: str path to the lightglue onnx model
            score_thresh: float score threshold for filtering matches
        """
        self.score_thresh = score_thresh
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
        self.aliked = ort.InferenceSession(aliked_path,providers=providers)
        self.input_name = self.aliked.get_inputs()[0].name
        self.input_shape = self.aliked.get_inputs()[0].shape[2:]
        self.lightglue = ort.InferenceSession(lightglue_path,providers=providers)
        self.templates = {} # name -> (kpt,desc,score,scale)

    def __call__(self,image0,image1)->Tuple[np.ndarray]:
        """
        Args:
            image0: str|np.ndarray shape (H,W,3) BGR, if str, it should be the name of the template
            image1: np.ndarray shape (H,W,3) BGR
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
            scores: np.ndarray shape (K,) scores of the matches
        """
        # kpt extraction from image0
        # if image0 is a string, use infered result from templates
        if type(image0) == str:
            kpt0,desc0,score0,scale0 = self.templates.get(image0,(None,None,None,None))
            if kpt0 is None:
                raise ValueError(f"Template {image0} not found")
        else:
            scale0,tensor0 = self.preprocess(image0)
            kpt0,desc0,score0 = self.aliked.run(None, {self.input_name: tensor0})

        # kpt extraction from image1
        scale1,tensor1 = self.preprocess(image1)
        kpt1,desc1,score1 = self.aliked.run(None, {self.input_name: tensor1})

        # find matches
        m_kpts0, m_kpts1,scores  = self.find_matches_from_kpts(kpt0,kpt1,desc0,desc1,scale0,scale1)
        return m_kpts0, m_kpts1, scores
    
    def register_template(self, name:str,image:np.ndarray)->None:
        """
        register a template image to be used in the future
        Args:
            name: str template name
            image: np.ndarray shape (H,W,3) BGR
        """
        scale,tensor = self.preprocess(image)
        kpt,desc,score = self.aliked.run(None, {self.input_name: tensor})
        self.templates[name]=(kpt,desc,score,scale)
    
    def padding(self, image:np.ndarray)->Tuple[float,np.ndarray]:
        """
        resize the image keeping the aspect ratio
        and pad it to the input shape (letterbox process)
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            padded: np.ndarray shape (H',W',3) BGR
        """
        size=self.input_shape
        h, w = image.shape[:2]
        resize_scale = min(size[0] /h, size[1] / w)
        scale = 1 / max(h, w)
        h_new, w_new = int(h * resize_scale), int(w * resize_scale)
        image = cv2.resize(image, (w_new, h_new))

        padded = np.zeros((size[0], size[1], 3), dtype=np.float32)
        padded[:h_new, :w_new] = image

        return scale,padded.astype("uint8")

    def preprocess(self, image)->Tuple[float,np.ndarray]:
        """
        process the image to the format that the model accepts
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            tensor: np.ndarray shape (1,3,H,W) RGB
        """
        # resize the image and pad it
        scale,padded_image = self.padding(image)
        # convert to RGB
        padded_image_rgb = padded_image[:, :, ::-1]
        # normalize the image
        tensor = padded_image_rgb / 255.0
        # HWC -> CHW
        tensor = tensor.transpose(2, 0, 1)
        # add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        return scale,tensor.astype(np.float32)
    
    def find_matches_from_kpts(self,kpts0,kpts1,desc0,desc1,scales0, scales1):
        """
        find matches between two sets of keypoints
        Args:
            kpts0: np.ndarray shape (N,2) normalized coordinates of keypoints
            kpts1: np.ndarray shape (M,2) normalized coordinates of keypoints
            desc0: np.ndarray shape (N,D)
            desc1: np.ndarray shape (M,D)
            scales0: float
            scales1: float
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
        """
        # find matches
        matches0, mscores0 = self.lightglue.run(
            None,
            {
                "kpts0": kpts0[None,:], # add batch dimension
                "kpts1": kpts1[None,:],
                "desc0": desc0[None,:],
                "desc1": desc1[None,:],
            },
        )
        # filter matches by score
        matches0 = matches0[mscores0>self.score_thresh]
        # postprocess the matches
        m_kpts0, m_kpts1 = self.postprocess(
            kpts0, kpts1, matches0, scales0, scales1
        )
        scores = mscores0[mscores0>self.score_thresh]
        return m_kpts0, m_kpts1, scores
    
    @staticmethod
    def postprocess(kpts0, kpts1, matches, scales0, scales1):
        """
        postprocess the matches
        Args:
            kpts0: np.ndarray shape (N,2) normalized coordinates of keypoints
            kpts1: np.ndarray shape (M,2) normalized coordinates of keypoints
            matches: np.ndarray shape (K,2) indices of matches
            scales0: float
            scales1: float
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
        """
        # denormalize the keypoints
        kpts0 = (kpts0 + 1) / scales0 / 2 
        kpts1 = (kpts1 + 1) / scales1 / 2
        # create match indices
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        return m_kpts0, m_kpts1
    
    @staticmethod
    def transform_image(ref_img,target_img,ref_points,target_points):
        """
        affine transformation of the target image to the reference image
        Args:
            ref_img: np.ndarray shape (H,W,3) BGR
            target_img:  np.ndarray shape (H,W,3) BGR
            ref_points: matched keypoints in reference image, shape (N,2)
            target_points: matched keypoints in target image, shape (N,2)
        Returns:
            success: bool
            transformed_image: np.ndarray shape (H,W,3) BGR
        """
        
        ref_points = np.array(ref_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        
        # calculate homography
        # M, _ = cv2.estimateAffinePartial2D(ref_points,target_points, method=cv2.RANSAC)
        M, mask = cv2.findHomography(ref_points, target_points, cv2.RANSAC, 5.0)
        
        if M is not None:
            # affine transformation
            # transformed_image = cv2.warpAffine(target_img, M, (ref_img.shape[1], ref_img.shape[0]))
            transformed_image = cv2.warpPerspective(target_img, np.linalg.inv(M),  (ref_img.shape[1], ref_img.shape[0]))
            return True,transformed_image
        else:
            print("Failed to find homography")
            return False,target_img
        
    @staticmethod    
    def draw_matches(ref_img,target_img,ref_points,target_points,scores):
        """
        draw the matched pairs in image
        Args:
            ref_img: np.ndarray shape (H,W,3) BGR
            target_img:  np.ndarray shape (H,W,3) BGR
            ref_points: matched keypoints in reference image, shape (N,2)
            target_points: matched keypoints in target image, shape (N,2)
            scores: np.ndarray shape (N,) scores of the matches
        Returns:
            None
        """
        marged_width = ref_img.shape[1] + target_img.shape[1]
        marged_height = max(ref_img.shape[0], target_img.shape[0])
        matches_img = np.zeros((marged_height, marged_width, 3), dtype=np.uint8)
        matches_img[:ref_img.shape[0], :ref_img.shape[1]] = ref_img
        matches_img[:target_img.shape[0], ref_img.shape[1]:] = target_img
        for ref_point, target_point,score in zip(ref_points, target_points, scores):
            ref_point = ref_point.astype(int)
            target_point = target_point.astype(int)
            target_point[0] += ref_img.shape[1]
            color = (0, int(255 * score), int(255 * (1 - score)))
            cv2.line(matches_img, tuple(ref_point), tuple(target_point),color, 1)
        return matches_img
    
    def show_result(self,img0,img1,m_kpts0,m_kpts1,score):
        """
        visualize the matches and transformed image
        Args:
            img0: np.ndarray shape (H,W,3) BGR
            img1:  np.ndarray shape (H,W,3) BGR
            m_kpts0: matched keypoints in image0, shape (N,2)
            m_kpts1: matched keypoints in image1, shape (N,2)
            score: np.ndarray shape (N,) scores of the matches
        Returns:
            None
        """
        ret,transformed_img = self.transform_image(img0,img1,m_kpts0,m_kpts1)
        matches_img = self.draw_matches(img0,img1,m_kpts0,m_kpts1,score)

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[matches_img.shape[1]/matches_img.shape[0], 
                                                transformed_img.shape[1]/transformed_img.shape[0]])
        a0 = fig.add_subplot(gs[0])
        a1 = fig.add_subplot(gs[1])
        
        a0.imshow(matches_img[..., ::-1])
        a0.set_title("Matches")
        a0.axis("off")
        a1.imshow(transformed_img[..., ::-1])
        a1.set_title("Transformed image")
        a1.axis("off")
        plt.show()

        
if __name__ == "__main__":
    aliked_path = "onnx/aliked-n16rot-top2k-640.onnx"
    lightglue_path = "onnx/lightglue_for_aliked.onnx"
    model = AlikedLightGlueONNX(aliked_path,lightglue_path,score_thresh=0.1)
    img0 = cv2.imread("assets/st_pauls_cathedral/1.jpg")
    img1 = cv2.imread("assets/st_pauls_cathedral/3.jpg")
    # m_kpts0, m_kpts1, score = model(img0,img1)
    model.register_template("0",img0)
    m_kpts0, m_kpts1, score = model("0",img1)
    # ret,transformed_img = model.transform_image(img0,img1,m_kpts0,m_kpts1)
    # matches_img = model.draw_matches(img0,img1,m_kpts0,m_kpts1,score)
    model.show_result(img0,img1,m_kpts0,m_kpts1,score)

