import numpy as np
import cv2
import pangolin
import OpenGL.GL as gl

class Frame(object):
    def __init__(self):
        self.img =None
        self.img_idx = None


        # Feature Extraction Method
        self.orb = cv2.ORB_create(3000)

        self.lk_params = dict( 
                    winSize = (15, 15), 
                    maxLevel = 2, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                10, 0.03)) 
    
        # Keypoints Extraction
        self.keypoints_2d = []
        self.descriptors = []
        self.imgs = []
        self.orb_features = []
        self.keypoints_3d = []
        
        self.K = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
                           [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
                           [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        # CameraPose 
        self.poses = []
        

    def vo(self, img):
        self.imgs.append(img)
        if len(self.keypoints_2d) < 1 :
            keypoints = self.keyframe_features()
        else:

            key_element = self.keypoints_2d[-1]
            
            # KeyFrame Selection
            if len(key_element) < 1000:
                self.keypoints_2d.pop()
                keypoints = self.keyframe_features()
            
            # Non-KeyFrame
            else:
                keypoints = self.feature_tracking()
        # return self.keypoints_2d

    def keyframe_features(self):
        current_kpts, curr_orb_desc = self.orb.detectAndCompute(self.imgs[-1], None)
        current_kpts_np = np.float32([(p.pt[0], p.pt[1]) for p in current_kpts])
        self.keypoints_2d.append(current_kpts_np)
        return current_kpts_np
    
    def feature_tracking(self):
        prev_frame = self.imgs[-2]
        cur_frame = self.imgs[-1]
        current_kpts_np, st, err = cv2.calcOpticalFlowPyrLK(np.mean(prev_frame, axis =2).astype(np.uint8), 
                                                   np.mean(cur_frame, axis = 2).astype(np.uint8), 
                                                   self.keypoints_2d[-1], None, **self.lk_params)
        

        self.keypoints_2d[-1] = np.expand_dims(self.keypoints_2d[-1], axis =1)
        current_kpts_np = np.expand_dims(current_kpts_np, axis =1)
        good_prev_kps = self.keypoints_2d[-1][st == 1]
        
        current_kpts_np = current_kpts_np[st == 1]
        kRansacProb = 0.999
        kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
        normalized_prev_kpts = self.normalize_points(good_prev_kps, self.K)
        normalized_curr_kpts = self.normalize_points(current_kpts_np, self.K)
        E, mask = cv2.findEssentialMat(normalized_curr_kpts, normalized_prev_kpts, method=cv2.RANSAC, focal =1, pp =(0, 0), prob=kRansacProb, threshold=kRansacThresholdNormalized)
        _, R, t, mask_pose = cv2.recoverPose(E, normalized_curr_kpts, normalized_prev_kpts)
        t = t.reshape(3, )
        current_kpts_np = current_kpts_np[mask.ravel() == 1]
        good_prev_kps = good_prev_kps[mask.ravel() == 1]
        
        T = np.eye(4)
        if len(self.poses) < 1:
            initial_pose = np.eye(4)
            self.poses.append(initial_pose)    
            R = np.dot(np.eye(3), R)
            t = np.zeros(3).reshape(3, ) + np.dot(np.eye(3), t)
            T[:3, :3]=R
            T[:3, 3]=t
        else:
            prev_pose = self.poses[-1]
            R = np.dot(prev_pose[:3, :3], R)
            t = prev_pose[:3, 3] + np.dot(prev_pose[:3, :3], t)
            T[:3, :3]=R
            T[:3, 3]=t    
        self.poses.append(T)
        
        # Visualization For The Feature Tracking
        for prev_p, cur_p in zip(good_prev_kps, current_kpts_np):
            cv2.circle(cur_frame, (int(prev_p[0]), int(prev_p[1])), 1, (255, 0, 0), 1)
            cv2.line(cur_frame, (int(cur_p[0]), int(cur_p[1])), (int(prev_p[0]), int(prev_p[1])), (0, 255, 0))
        cv2.imshow("Visualization for the tracking : ", cur_frame)
        # current_kpts_np = np.expand_dims(current_kpts_np, axis =1)
        self.keypoints_2d.append(current_kpts_np)
        
        # self.keypoints_2d.append(curr_kpts_after_match_np)
        return None

    def normalize_points(self, keypoints, K):
        """
        Normalize 2D keypoints using the intrinsic camera matrix.
        :param keypoints: Nx2 array of x, y coordinates
        :param K: 3x3 Camera intrinsic matrix
        :return: Nx2 array of normalized coordinates
        """
        # Convert to homogeneous coordinates by adding a row of 1s
        ones = np.ones((keypoints.shape[0], 1))
        homogeneous_points = np.hstack([keypoints, ones])
        # Apply the inverse of the intrinsic matrix
        normalized_points = np.linalg.inv(K) @ homogeneous_points.T
        # Convert back from homogeneous coordinates
        normalized_points = normalized_points[:2] / normalized_points[2]
        return normalized_points.T


def drawPlane(num_divs=200, div_size=10):
    # Plane parallel to x-z at origin with normal -y
    minx = -num_divs * div_size
    minz = -num_divs * div_size
    maxx = num_divs * div_size
    maxz = num_divs * div_size
    gl.glColor3f(0.7, 0.7, 0.7)
    gl.glBegin(gl.GL_LINES)
    for n in range(2 * num_divs + 1):
        gl.glVertex3f(minx + div_size * n, 0, minz)
        gl.glVertex3f(minx + div_size * n, 0, maxz)
        gl.glVertex3f(minx, 0, minz + div_size * n)
        gl.glVertex3f(maxx, 0, minz + div_size * n)
    gl.glEnd()


if __name__ == "__main__":

    video_path = "/home/wondong/code/kadif_research/depth_slam/D3VO/data/video.mp4"

    cap = cv2.VideoCapture(video_path)


    test = Frame()
    # Visualization with Pangolin
    h, w = 1024, 1024
    kUiWidth = 180  # Width of the UI panel

    # Initialization for the Pangolin Visualization
    pangolin.CreateWindowAndBind('Map Viewer', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Camera setup
    viewpoint_x, viewpoint_y, viewpoint_z = 0, -40, -80
    viewpoint_f = 1000
    proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w//2, h//2, 0.1, 5000)
    look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
    scam = pangolin.OpenGlRenderState(proj, look_view)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, kUiWidth/w, 1.0, -w/h)
    dcam.SetHandler(pangolin.Handler3D(scam))

    # Panel for UI elements
    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(1.0, 0.0, 0.0, kUiWidth / float(w))
    
    checkboxFollow = pangolin.VarBool('ui.Follow', value=True, toggle=True)
    checkboxCams = pangolin.VarBool('ui.Draw Cameras', value=True, toggle=True)
    checkboxCovisibility = pangolin.VarBool('ui.Draw Covisibility', value=True, toggle=True)
    checkboxSpanningTree = pangolin.VarBool('ui.Draw Tree', value=True, toggle=True)
    checkboxGrid = pangolin.VarBool('ui.Grid', value=True, toggle=True)
    checkboxPause = pangolin.VarBool('ui.Pause', value=False, toggle=True)
    int_slider = pangolin.VarInt('ui.Point Size', value=2, min=1, max=10)
    img_idx = 0
    while True:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        if checkboxGrid.Get():
            drawPlane()

        retavl, img = cap.read()
        
        test.vo(img)
        pose_list = test.poses
        points_list = test.keypoints_3d        

        
        if len(pose_list) >0:
            poses_array = np.stack(pose_list, axis=0)  # Stack all poses to create a 3D array

            print("Current Camera Pose : \n", pose_list[-1])
            
            if checkboxCams.Get():

                if len(pose_list) > 2:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(poses_array[:-1])
                if len(pose_list) >= 1:
                    gl.glColor3f(1.0, 0.0, 0.0)
                    pangolin.DrawCameras(poses_array[-1:])
        

                    
        pangolin.FinishFrame()

        if cv2.waitKey(1) == "q":
            break