import numpy as np
import cv2
import g2o
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
        
        self.K = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
                           [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
                           [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        # CameraPose 
        self.poses = []

        # 3d Keypoints
        self.keypoints_3d = []
        self.drawing_keypoints = []
        self.optim_keypoints_2d = []
        self.optim_keypoints_3d = []



        
        

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
                self.keyframe = False
                
            # Non-KeyFrame
            else:
                keypoints = self.feature_tracking()
                self.keyframe = True
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





        ########################################################################################################
        ########################################################################################################
        ########################################################################################################
        previous_pose = np.eye(4)
        previous_pose = previous_pose[:-1, :]
        current_pose = np.hstack([R, t.reshape(3, 1)])

        # OpenCV Triangulation
        pts4d = cv2.triangulatePoints(previous_pose, current_pose, normalized_curr_kpts.T, normalized_prev_kpts.T).T
        # pts4d = cv2.triangulatePoints(previous_pose, current_pose, current_kpts_np.T, good_prev_kps.T).T
        # pts4d /= pts4d[:, 3:]
        # kpts_3d = pts4d[:, :-1]
        

        pts4d /= pts4d[:, 3:]
        
        keypoints_ = []
        keypoints_2d = []
        for i, (p, cp, pp, npp) in enumerate(zip(pts4d, current_kpts_np, good_prev_kps, normalized_prev_kpts)):
            pl1 = np.dot(previous_pose, p)
            pl2 = np.dot(current_pose, p)            
            if pl1[2] < 0 or pl2[2] < 0:
                continue    
            
            keypoints_.append(p)

            # keypoints_2d.append(npp)
            keypoints_2d.append(cp)
            # keypoints_2d.append(pp)
            
        
        keypoints_np = np.array(keypoints_)
        keypoints_2d_np = np.array(keypoints_2d)
        
        ########################################################################################################
        ########################################################################################################
        ########################################################################################################

        
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

######################################################################3
        current_status = self.poses[-1]
        current_status = current_status[:-1, :]

        
        check = np.dot(current_status, keypoints_np.T)
        check = check.T
        # check = check[:, :-1]
        self.drawing_keypoints.append(check)



        # if len(keypoints_2d_np) != 3000:
        self.optim_keypoints_2d.append(keypoints_2d_np)
        self.optim_keypoints_3d.append(keypoints_np[:, :-1])



        # Visualization For The Feature Tracking
        for prev_p, cur_p in zip(good_prev_kps, current_kpts_np):
            cv2.circle(cur_frame, (int(prev_p[0]), int(prev_p[1])), 1, (255, 0, 0), 1)
            cv2.line(cur_frame, (int(cur_p[0]), int(cur_p[1])), (int(prev_p[0]), int(prev_p[1])), (0, 255, 0))
        cv2.imshow("Visualization for the tracking : ", cur_frame)
        # current_kpts_np = np.expand_dims(current_kpts_np, axis =1)
        self.keypoints_2d.append(current_kpts_np)


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
    
    def triangulate(self, pose1, pose2, pts1, pts2):
        ret = np.zeros((pts1.shape[0], 4))
        
        for i, p in enumerate(zip(pts1, pts2)):
            A = np.zeros((4,4))
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]
            _, _, vt = np.linalg.svd(A)
            ret[i] = vt[3]
        return ret

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





    # Backend Process
    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    # add normalized camera
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)         
    cam.set_id(0)                       
    opt.add_parameter(cam)   

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))




    graph_idx = 0
    se3_list = [] 
    pose_graph_index=[]
    points_graph_index=[]

    while True:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        if checkboxGrid.Get():
            drawPlane()

        retavl, img = cap.read()
        
        test.vo(img)
        pose_list = test.poses
        points_list = test.drawing_keypoints   
        
        
        
        popints_3d_list = test.optim_keypoints_3d
        points_2d_list = test.optim_keypoints_2d
        if len(points_2d_list) > 1:
            points_2d = points_2d_list[-1]

        # TODO : Graph Optimization Code
        for pose in pose_list:
            se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_estimate(se3)
            v_se3.set_id(graph_idx)
            pose_graph_index.append(graph_idx)

            if pose_graph_index == 0:
                v_se3.set_fixed(pose_graph_index)
            graph_idx += 1
            opt.add_vertex(v_se3)
            est = v_se3.estimate()
            se3_list.append(v_se3)
            

        # for (points, points2d) in zip(popints_3d_list, points_2d_list):
        for (points, points2d) in zip(points_list, points_2d_list):
            for (point, point2d) in zip(points, points2d):
                pt = g2o.VertexSBAPointXYZ()
                pt.set_id(graph_idx)
                graph_idx += 1
                pt.set_estimate(point)
                pt.set_marginalized(True)
                opt.add_vertex(pt)

                # if len(se3_list) > 1:
                for i in range(len(se3_list)):
                    edge = g2o.EdgeProjectXYZ2UV()
                    edge.set_parameter_id(0, 0)
                    edge.set_vertex(0, pt)
                    edge.set_vertex(1, se3_list[i])
                    edge.set_measurement(point2d)
                    edge.set_information(np.eye(2))
                    edge.set_robust_kernel(robust_kernel)
                    opt.add_edge(edge)

        if (img_idx+1) % 10 == 0: 
            opt.set_verbose(True)
            opt.initialize_optimization()
            opt.optimize(1)
            

        optimized_poses = []
        for v_se3 in se3_list:
            se3 = v_se3.estimate()
            pose = np.eye(4)
            pose[0:3, 0:3] = se3.rotation().matrix()
            pose[0:3, 3] = se3.translation()
            # print(pose)
            optimized_poses.append(pose)

        # Retrieve optimized 3D points
        optimized_points = []
        for v in opt.vertices().values():
            if isinstance(v, g2o.VertexSBAPointXYZ):
                optimized_points.append(v.estimate())


        if len(optimized_poses) >0:
            optimized_poses_array = np.stack(optimized_poses, axis=0)  # Stack all poses to create a 3D array
            # print(optimized_poses_array)
            if checkboxCams.Get():

                if len(pose_list) > 2:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(optimized_poses_array[:-1])
                if len(pose_list) >= 1:
                    gl.glColor3f(1.0, 0.0, 0.0)
                    pangolin.DrawCameras(optimized_poses_array[-1:])

                if len(optimized_points) > 1:                    
                    current_points = np.array(optimized_points)
                    
                    # gl.glPointSize(2)
                    # gl.glColor3f(0.0, 0.0, 0.0)
                    # pangolin.DrawPoints(current_points)
                    
                    gl.glPointSize(2)
                    gl.glColor3f(1.0, 0.0, 0.0)
                    pangolin.DrawPoints(current_points)
                

        pangolin.FinishFrame()

        if cv2.waitKey(1) == "q":
            break