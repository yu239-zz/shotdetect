#!/usr/bin/python

import cv2
import numpy as np
import os
import sys

'''
  A simple yet effective python implementation for video shot detection of abrupt transition
  based on python OpenCV
'''

__hist_size__ = 64          # how many bins for each R,G,B histogram
__min_duration__ = 7        # if a shot has length less than this, merge it with others

class shotDetector:
    def __init__(self, video_path=None, min_duration=__min_duration__, output_dir=None):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = output_dir

    def run(self, video_path=None):
        if video_path is not None:
            self.video_path = video_path    
        assert (self.video_path is not None), "you should must the video path!"

        self.shots = []
        cap = cv2.VideoCapture(self.video_path)
        hists = []
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            if self.output_dir is not None:
                frames.append(frame)
            # compute RGB histogram for each frame
            chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0,256]) \
                          for c in range(3)]
            chists = np.array([chist/float(sum(chist)) for chist in chists])
            hists.append(chists.flatten())
        # compute hist chisquare distances
        scores = [cv2.compareHist(pair[0],pair[1],cv2.cv.CV_COMP_CHISQR) \
                      for pair in zip(hists[1:], hists[:-1])]
        # compute automatic threshold
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + 4*std_score

        # decide shot boundaries
        prev_i = 0
        prev_score = scores[0]
        for i, score in enumerate(scores[1:]):
            if score>=threshold and abs(score-prev_score)>=threshold:
                self.shots.append((prev_i, i+2))
                prev_i = i + 2
            prev_score = score
        video_length = len(hists)
        self.shots.append((prev_i, video_length))
        assert video_length>=self.min_duration, "duration error"
        
        self.merge_short_shots()
        
        # save key frames
        if self.output_dir is not None:
            os.system("mkdir -p %s" % self.output_dir)
            for shot in self.shots:
                cv2.imwrite("%s/frame-%d.jpg" % (self.output_dir,shot[0]), frames[shot[0]])
        print "key frames written to %s" % self.output_dir

    def merge_short_shots(self):
        # merge short shots
        while True:
            durations = [shot[1]-shot[0] for shot in self.shots]
            shortest = min(durations)
            # no need to merge
            if shortest >= self.min_duration:
                break
            idx = durations.index(shortest)
            left_half = self.shots[:idx]
            right_half = self.shots[idx+1:]
            shot = self.shots[idx]

            # can only merge left
            if idx == len(self.shots)-1:
                left = True                
            # can only merge right
            elif idx == 0:
                left = False                
            else:
                # otherwise merge the shorter one
                if durations[idx-1] < durations[idx+1]:
                    left = True
                else:
                    left = False
            if left:
                self.shots = left_half[:-1] + [(left_half[-1][0],shot[1])] + right_half
            else:
                self.shots = left_half + [(shot[0],right_half[0][1])] + right_half[1:]

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "usage: ./shotdetect.py <video-path> [<key-frames-output-dir>]"
        sys.exit()
    video_path = sys.argv[1]
    key_frames_dir = None if len(sys.argv)<3 else sys.argv[2]
    detector = shotDetector(video_path, output_dir=key_frames_dir)
    detector.run()
    print detector.shots
