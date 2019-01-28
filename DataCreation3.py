#coding=utf-8


from VideoMatExtraction import VideoMatExtraction
from PaddingBackgroundCreation import PaddingBackgroundCreation
import os
import cv2
"""
  This version saves the data as it should be

  Example:
  from DataCreation import DataCreation
  create = DataCreation(ipath='/data/DataBases/SURREAL/SURREAL/data/cmu/train/run0/01_01/', opath='/data/HectorSanchez/Deep-Image-Matting/data/',
                        bg_path='/data/HectorSanchez/Deep-Image-Matting')
  create.create_data()
"""
class DataCreation(object):
  def __init__(self,ipath, apath, opath, bg_path, count=0):
    self.count = 0 # Counter for the number of the file
    self.bg_path = bg_path #Where the bg images are
    if self.bg_path[-1]!='/':
      self.bg_path = self.bg_path+'/'

    self.ipath = ipath
    if self.ipath[-1]!='/':
      self.ipath = self.ipath+'/'

    self.apath= apath
    if self.apath[-1]!='/':
      self.apath=self.apath+'/'

    self.opath = opath
    if self.opath[-1]!='/':
      self.opath = self.opath+'/'

    self.videos = [] # Names of videos
    self.mats = [] #Names of mat files

    self.fgs = []
    self.alphas = []

    self.__read_path()

  def __read_path(self):
    """
       Read the files in the path directory, saving only those 
       being a mp4 file, or a segment mat file.
    """

    # files = sorted(os.listdir(self.ipath))
    #
    # for file in files:
    #   if '.mp4' in file:
    #     self.videos.append(file)
    #   elif 'segm.mat' in file:
    #     self.mats.append(file)
    ifiles=sorted(os.listdir(self.ipath))
    afiles=sorted(os.listdir(self.apath))

    for ifile,afile in zip(ifiles,afiles):
      self.fgs.append(os.path.join(self.ipath,ifile))
      self.alphas.append(os.path.join(self.apath,afile))

  def create_data(self):

    img_path = self.opath+'eps1280/'
    seg_path = self.opath+'alpha1280/'
    bg_path = self.opath+'bg/'

    
    # Checks if final output dir exists
    if not os.path.exists(img_path):
      os.makedirs(img_path)
    if not os.path.exists(seg_path):
      os.makedirs(seg_path)
    if not os.path.exists(bg_path):
      os.makedirs(bg_path)

    length=len(self.alphas)
    for i,(segment,frame) in enumerate(zip(self.alphas,self.fgs)):
      print("still remains {0} ...".format(length-i))
      group_dir_eps = img_path + str(self.count) + '/'
      # Checks if group dir exists
      if not os.path.exists(group_dir_eps):
        os.makedirs(group_dir_eps)

      group_dir_alpha = seg_path + str(self.count) + '/'
      if not os.path.exists(group_dir_alpha):
        os.makedirs(group_dir_alpha)

      group_dir_bg = bg_path + str(self.count) + '/'
      if not os.path.exists(group_dir_bg):
        os.makedirs(group_dir_bg)
      # print("----hello----")
      # print(self.count)
      # print("还差{0}张图像...".format(length-i))
      # Resize to width = 640
      segment=cv2.imread(segment)
      segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
      frame=cv2.imread(frame)
      segment = self.__adjust_size(segment)
      frame = self.__adjust_size(frame)

      alpha = self.__create_alpha(segment)

      save_paths = [group_dir_eps, group_dir_alpha, group_dir_bg]
      create = PaddingBackgroundCreation(img=frame, alpha=alpha, bgs_path=self.bg_path, save_paths=save_paths)

      create.create_data()
      # cv2.imwrite(img_path+str(i)+'.png',frame)
      # cv2.imwrite(seg_path+str(i)+'.png',alpha)
      self.count += 1
  def __create_alpha(self,segment):
    alpha = (segment>0)*255
    alpha = alpha.astype(float)
    return(alpha)

  def __adjust_size(self,img):
    """
      Adjust the image so the biggest side is 640px
    """
    rows = img.shape[0]
    cols = img.shape[1]

    if cols>rows:
      new_cols = 640
      new_rows = int((rows*new_cols)/cols)
    else:
      new_rows = 640
      new_cols = int((cols*new_rows)/rows)

    new_img = cv2.resize(img,(new_cols, new_rows), interpolation = cv2.INTER_CUBIC)

    return(new_img)


if __name__=="__main__":
  # ipath="D:/DeepImageMatting_Data/Training_set/Adobe-licensed images/fg"
  # apath="D:/DeepImageMatting_Data/Training_set/Adobe-licensed images/alpha"
  # opath="D:/DeepImageMatting_Data/Training_set/Adobe-licensed images/merge_structure"
  # bpath="D:/train2014"

  ipath = "D:/data_portrait/fg/"
  apath = "D:/data_portrait/alpha/"
  opath = "D:/PersonMerge/"
  bpath = "D:/train2014"
  create = DataCreation(ipath=ipath,apath=apath,opath=opath,bg_path=bpath)
  create.create_data()
