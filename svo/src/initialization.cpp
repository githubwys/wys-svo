// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  detectFeatures(frame_ref, px_ref_, f_ref_);//先检测FAST特征点和边缘特征
  if(px_ref_.size() < 100)
  {
    // 第一帧图像需要100+特征，否则在纹理更丰富的环境中继续尝试。
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  //px_cur_ 当前帧的2D点  // px_ref_ 参考帧（前一帧）的2D点
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

// processSecondFrame()用于跟第一张进行三角初始化
// 从第一张图像开始，就用光流法持续跟踪特征点，
// 把特征像素点转换成在相机坐标系下的深度归一化的点，并进行畸变校正,再让模变成1,映射到单位球面上面。
// 如果匹配点的数量大于阈值，并且视差的中位数大于阈值。
// 如果视差的方差大的话，选择计算E矩阵，如果视差的方差小的话，选择计算H矩阵。
// 如果计算完H或E后，还有足够的内点，就认为这帧是合适的用来三角化的帧。根据H或E恢复出来的位姿和地图点，进行尺度变换，把深度的中值调为1。
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  // 光流法返回视差
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  // 返回的有视差的点少于50个，视为失败
  if(disparities_.size() < Config::initMinTracked())// config.cpp : (("init_min_tracked", 50)),
    return FAILURE;

  double disparity = vk::getMedian(disparities_);// 并且视差的中位数
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");//SVO_INFO_STREAM:cerr：输出到标准错误的ostream对象，常用于程序错误信息；
  // 视差中位数不能小于initMinDisparity=50，才有关键帧
  if(disparity < Config::initMinDisparity())// (("svo/init_min_disparity", 50.0)),
    return NO_KEYFRAME;

  // 计算单应矩阵
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  if(inliers_.size() < Config::initMinInliers()) //判断H阵中，两帧之间剩下的内点数量是否足够 ("svo/init_min_inliers", 40)
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  // 重新缩放贴图，使平均场景深度等于指定的比例
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);// 计算中位数
  double scale = Config::mapScale()/scene_depth_median;
  // 计算当前帧（第二个关键帧）的pose： 与第一个关键帧的Transform由computeHomography()得到，参数：T_cur_from_ref_
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  // 对于每个内插层，在两个框架中创建三维点并添加特征
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    // isInFrame()判断是不是inliers_点在frame范围中？
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0)); // 特征结构的实例化
      frame_cur->addFeature(ftr_cur); // 当前frame帧frame_cur添加特征ftr_cur
      new_point->addFrameRef(ftr_cur); // 点point添加当前帧的依赖，注明点在哪一帧当中

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));  // 特征结构的实例化
      frame_ref_->addFeature(ftr_ref); // 前一frame帧frame_ref_添加特征ftr_ref
      new_point->addFrameRef(ftr_ref); // 点point添加前一帧的依赖，注明点在哪一帧当中
    }
  }
  return SUCCESS;//
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

void detectFeatures(//先检测FAST特征点和边缘特征
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(  // 初始化Fast角点检测器
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  // 调用detector的函数detect()
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // 初始化seed，用于深度滤波器
  // reverse() 链表的逆序
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}

void trackKlt( // opencv光流法跟踪  　Kanade-Lucas-Tomasi方法
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  // cv::calcOpticalFlowPyrLK() 
  // Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
  // 使用金字塔的迭代Lucas Kanade方法计算稀疏特征的光流
  // 
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],// 初始图像；最终图像
                           px_ref, px_cur, // 需要跟踪的点，点的新位置
                           status, error, // status长度为points个数，值为0/1，表示对应点在第二张图像中是否发现 //error删除变化剧烈的点
                           cv::Size2i(klt_win_size, klt_win_size),//定义了金字塔尺寸
                           4, // 定义了金字塔层数 
                           termcrit, // 终止迭代条件 
                           cv::OPTFLOW_USE_INITIAL_FLOW); // =4：函数调用之前，数组B已包含特征点的初始坐标值。（flag对boolkeeping控制）
  
  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    if(!status[i])// 删除找不到对应点的光流点，status=0则删除
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    // c2f() //将像素坐标（c）转换为帧的单位球体坐标（f）。
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());//norm()它返回二范数
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

void computeHomography( // 初始化：计算单应矩阵
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold, // 限差
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }
  // 初始化Homography类
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  // 计算单应矩阵
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  // 由单应矩阵计算两帧之间特征点的内点
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
