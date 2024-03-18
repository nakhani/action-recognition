import cv2 , os , glob , pickle
import numpy as np 

class MoSIFT():

    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.formats = (".mp4",".3gp",".avi")
        self.vidobj = None
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def vidread(self,path_to_video):
        if path_to_video.lower().endswith(self.formats):
            print("Format is checked , file is a valid video format.")
            try:
                self.vidobj = cv2.VideoCapture(path_to_video)
                self.width = int(self.vidobj.get(3))
                self.height = int(self.vidobj.get(4))
                # print(self.width,self.height)
            except:
                print("something is wrong with video file !")
        else:
            self.vidobj = None
            print("Wrong video format !")

    def retFrames(self):
        if self.vidobj :
            ret , frame = self.vidobj.read()
            if ret:
                return ret ,frame
            else: return False , None
        else: return False , None

    def dosift(self , path_to_video):
        self.vidread(path_to_video)
        frames = []
        while(True):
            ret , frame = self.retFrames()
            if not ret:
                print('End of Frames !')
                break
            gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            kps = self.sift.detect(gray)
            img=cv2.drawKeypoints(gray,kps,frame)
            frames.append(img)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
        out = cv2.VideoWriter(os.path.join('Output','sift_key_points.avi'), fourcc, 15, (self.width, self.height))
        print("SIFT algorithm is applied to {} frames of the video. saving as : Output/sift_key_points.avi ".format(len(frames)))
        for frame in frames:
            out.write(frame)
        out.release()
        del out

    def do_OF(self,path_to_video):
        self.vidread(path_to_video)
        frames = []
        ret , frame = self.retFrames()
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        while(True):
            ret, frame = self.retFrames()
            if not ret:
                print('End of Frames !')
                break
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            prvs = next
            frames.append(bgr)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
        out = cv2.VideoWriter(os.path.join('Output','optical_flow.avi'), fourcc, 15, (self.width, self.height))
        print("Optical flow algorithm is applied to {} frames of the video. saving as : Output/optical_flow.avi ".format(len(frames)))
        for frame in frames:
            out.write(frame)
        out.release()
        del out

    def get_keypoints(self,frame):
        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kps = self.sift.detect(gray)
        pts = cv2.KeyPoint_convert(kps)
        return pts

    def get_flow(self,pre,nex):
        nex = cv2.cvtColor(nex, cv2.COLOR_BGR2GRAY)
        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(pre, nex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def cluster(self,pts,K):
        if len(pts) >= 8 :
            return cv2.kmeans(pts,K,None,self.criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        else: return None

    def flowavg(self,pre,nex):
        nei = [-1,0,1]
        pts = self.get_keypoints(pre)
        flow = self.get_flow(pre,nex)
        if len(pts) >= 8:
            clusters = self.cluster(pts,1) # Has no use since K = 1 ! if K > 1 you should loop over clusters !
            imgcntr = np.array([self.height/2 , self.width/2])
            displacementvec = imgcntr - clusters[2][0]
            displacementvec = displacementvec.astype("int")
            score = []
            pts = pts.astype("int")
            for pt in pts:
                temp = 0
                cnt = 0
                for i in nei:
                    for j in nei:
                        try:
                            temp += np.sqrt(flow[pt[1]+i,pt[0]+j,0]**2 + flow[pt[1]+i,pt[0]+j,1]**2)
                            cnt += 1
                        except:
                            continue
                score.append(np.append((pt + displacementvec)/(2*imgcntr) ,temp/cnt))
            return np.array(score)
        else: return None

    def create_dataset(self,adrs):
        subdirs = [x for x in os.listdir(adrs)]
        for label in subdirs:
            data = []
            vids = glob.glob(os.path.join(adrs,label,"*"))
            for vid in vids:
                self.vidread(vid)
                while True:
                    _ , pre = self.retFrames()
                    ret , nex = self.retFrames()
                    if not ret:
                        break
                    out = mosift.flowavg(pre,nex)
                    if not out is None:
                        data.append(out)
            print("Saving as pickle ...")
            with open("./dataset/lists/{}.pkl".format(label) , "wb") as f:
                pickle.dump(data,f)



if __name__ == "__main__":
    mosift = MoSIFT()
    file = input("give full address to the file :")
    mosift.dosift(file)
    mosift.do_OF(file)
    mosift.create_dataset("./dataset/videos/")