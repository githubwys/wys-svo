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

#ifndef SVO_FEATURE_H_
#define SVO_FEATURE_H_

#include <svo/frame.h>

namespace svo {

/// A salient image region that is tracked across frames.
// 跨帧跟踪的显著图像区域。
struct Feature // 特征类
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType {
    CORNER,
    EDGELET
  };

  // 类型可以是角或边
  FeatureType type;     //!< Type can be corner or edgelet.
  // 指针指向特征被检测出来的帧
  Frame* frame;         //!< Pointer to frame in which the feature was detected.
  // 特征在0级图像金字塔的像素坐标
  Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
  // 特征的单位方向向量
  Vector3d f;           //!< Unit-bearing vector of the feature.
  // 特征被提取出来的图像金字塔层数
  int level;            //!< Image pyramid level where feature was extracted.
  // 指向与特征相关的3D点的指针
  Point* point;         //!< Pointer to 3D point which corresponds to the feature.
  // 归一化的，边缘特征的主梯度方向
  Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.

  Feature(Frame* _frame, const Vector2d& _px, int _level) : // 特征结构的初始化
    type(CORNER),
    frame(_frame),
    px(_px),
    f(frame->cam_->cam2world(px)),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {}

  Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) :  // 特征结构的初始化
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(NULL),
    grad(1.0,0.0)
  {}

  Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :  // 特征结构的初始化
    type(CORNER),
    frame(_frame),
    px(_px),
    f(_f),
    level(_level),
    point(_point),
    grad(1.0,0.0)
  {}
};

} // namespace svo

#endif // SVO_FEATURE_H_
