// Copyright 2024 Lihan Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "small_gicp_relocalization/small_gicp_relocalization.hpp"

#include "pcl/common/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include "small_gicp/pcl/pcl_registration.hpp"
#include "small_gicp/util/downsampling_omp.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

namespace small_gicp_relocalization
{

SmallGicpRelocalizationNode::SmallGicpRelocalizationNode(const rclcpp::NodeOptions & options)
: Node("small_gicp_relocalization", options),
  result_t_(Eigen::Isometry3d::Identity()),
  previous_result_t_(Eigen::Isometry3d::Identity())
{
  this->declare_parameter("pub_prior_pcd", false);
  this->declare_parameter("num_threads", 4);
  this->declare_parameter("num_neighbors", 20);
  this->declare_parameter("global_leaf_size", 0.25);
  this->declare_parameter("registered_leaf_size", 0.25);
  this->declare_parameter("max_dist_sq", 1.0);
  this->declare_parameter("map_frame", "map");
  this->declare_parameter("odom_frame", "odom");
  this->declare_parameter("base_frame", "");
  this->declare_parameter("lidar_frame", "");
  this->declare_parameter("prior_pcd_file", "");
  this->declare_parameter("enable_relocalization", true);
  this->declare_parameter("relocalization_x_range", 5.0);
  this->declare_parameter("relocalization_y_range", 5.0);
  this->declare_parameter("relocalization_yaw_range_deg", 90.0);
  this->declare_parameter("relocalization_x_step", 1.0);
  this->declare_parameter("relocalization_y_step", 1.0);
  this->declare_parameter("relocalization_yaw_step_deg", 10.0);

  this->get_parameter("pub_prior_pcd", pub_prior_pcd_);
  this->get_parameter("num_threads", num_threads_);
  this->get_parameter("num_neighbors", num_neighbors_);
  this->get_parameter("global_leaf_size", global_leaf_size_);
  this->get_parameter("registered_leaf_size", registered_leaf_size_);
  this->get_parameter("max_dist_sq", max_dist_sq_);
  this->get_parameter("map_frame", map_frame_);
  this->get_parameter("odom_frame", odom_frame_);
  this->get_parameter("base_frame", base_frame_);
  this->get_parameter("lidar_frame", lidar_frame_);
  this->get_parameter("prior_pcd_file", prior_pcd_file_);
  this->get_parameter("enable_relocalization", enable_relocalization_);
  this->get_parameter("relocalization_x_range", relocalization_x_range_);
  this->get_parameter("relocalization_y_range", relocalization_y_range_);
  this->get_parameter("relocalization_yaw_range_deg", relocalization_yaw_range_deg_);
  this->get_parameter("relocalization_x_step", relocalization_x_step_);
  this->get_parameter("relocalization_y_step", relocalization_y_step_);
  this->get_parameter("relocalization_yaw_step_deg", relocalization_yaw_step_deg_);
  
  // ---------------------
  // Initialize processing objects
  // ---------------------
  registered_scan_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  global_map_      = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  register_        = std::make_shared<
    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP>>();

  tf_buffer_      = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_    = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  // ---------------------
  // Load map
  // ---------------------
  loadGlobalMap(prior_pcd_file_);

  // ---------------------
  // Publish map PCD if needed
  // ---------------------
  if (pub_prior_pcd_) {
    prior_pcd_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("prior_pcd", 1);
    prior_pub_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),  // 1Hz
      std::bind(&SmallGicpRelocalizationNode::publishPriorPcd, this));
  }

  // Downsample points and convert them into pcl::PointCloud<pcl::PointCovariance>
  target_ = small_gicp::voxelgrid_sampling_omp<
    pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(
    *global_map_, global_leaf_size_);

  // Estimate covariances of points
  small_gicp::estimate_covariances_omp(*target_, num_neighbors_, num_threads_);

  // Create KdTree for target
  target_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(
    target_, small_gicp::KdTreeBuilderOMP(num_threads_));

  // ---------------------
  // Subscribe to point cloud (registered)
  // ---------------------
  pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "cloud_registered", 10,
    std::bind(&SmallGicpRelocalizationNode::registeredPcdCallback, this, std::placeholders::_1));

  // ---------------------
  // Subscribe to initial pose
  // ---------------------
  initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "initialpose", 10,
    std::bind(&SmallGicpRelocalizationNode::initialPoseCallback, this, std::placeholders::_1));

  // ---------------------
  // Timer: Periodically attempt registration
  // ---------------------
  register_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(500),  // 2 Hz
    std::bind(&SmallGicpRelocalizationNode::performRegistration, this));

  // ---------------------
  // Timer: Publish TF
  // ---------------------
  transform_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),  // 20 Hz
    std::bind(&SmallGicpRelocalizationNode::publishTransform, this));
}

// --------------------------------------------------
// Single registration function, returns registration result
// --------------------------------------------------
small_gicp::RegistrationResult SmallGicpRelocalizationNode::alignOnce(
  const pcl::PointCloud<pcl::PointCovariance> & target,
  const pcl::PointCloud<pcl::PointCovariance> & source,
  const Eigen::Isometry3d & initial_guess)
{
  register_->reduction.num_threads = num_threads_;
  register_->rejector.max_dist_sq  = max_dist_sq_;

  // Call GICP alignment
  auto result = register_->align(target, source, *target_tree_, initial_guess);
  return result;
}

// --------------------------------------------------
// Publish map PCD
// --------------------------------------------------
void SmallGicpRelocalizationNode::publishPriorPcd()
{
  if (!global_map_ || global_map_->empty()) {
    return;
  }
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*global_map_, msg);
  msg.header.stamp = this->now();
  msg.header.frame_id = map_frame_;
  prior_pcd_pub_->publish(msg);
}

// --------------------------------------------------
// Load map and transform (map<-odom<-lidar)
// --------------------------------------------------
void SmallGicpRelocalizationNode::loadGlobalMap(const std::string & file_name)
{
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_name, *global_map_) == -1) {
    RCLCPP_ERROR(this->get_logger(), "Couldn't read PCD file: %s", file_name.c_str());
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Loaded global map with %zu points", global_map_->points.size());

  // NOTE: Transform global pcd_map (based on `lidar_odom` frame) to the `odom` frame
  Eigen::Affine3d odom_to_lidar_odom;
  while (true) {
    try {
      auto tf_stamped = tf_buffer_->lookupTransform(
        base_frame_, lidar_frame_, this->now(), rclcpp::Duration::from_seconds(1.0));
      odom_to_lidar_odom = tf2::transformToEigen(tf_stamped.transform);
      RCLCPP_INFO_STREAM(
        this->get_logger(), "odom_to_lidar_odom: translation = "
                              << odom_to_lidar_odom.translation().transpose() << ", rpy = "
                              << odom_to_lidar_odom.rotation().eulerAngles(0, 1, 2).transpose());
      break;
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s Retrying...", ex.what());
      rclcpp::sleep_for(std::chrono::seconds(1));
    }
  }
  pcl::transformPointCloud(*global_map_, *global_map_, odom_to_lidar_odom);
}

// --------------------------------------------------
// Point cloud subscription callback
// --------------------------------------------------
void SmallGicpRelocalizationNode::registeredPcdCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  last_scan_time_ = msg->header.stamp;

  pcl::fromROSMsg(*msg, *registered_scan_);

  // Downsample Registered points and convert them into pcl::PointCloud<pcl::PointCovariance>.
  source_ = small_gicp::voxelgrid_sampling_omp<
    pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(
    *registered_scan_, registered_leaf_size_);

  // Estimate point covariances
  small_gicp::estimate_covariances_omp(*source_, num_neighbors_, num_threads_);

  // Create KdTree for source.
  source_tree_ = std::make_shared<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(
    source_, small_gicp::KdTreeBuilderOMP(num_threads_));
}

// --------------------------------------------------
// Timer: Perform regular registration
// --------------------------------------------------
void SmallGicpRelocalizationNode::performRegistration()
{
  if (!source_ || !source_tree_) {
    return;
  }

  // First perform regular GICP registration
  auto result = alignOnce(*target_, *source_, previous_result_t_);
  if (result.converged) {
    // Registration successful, update result_t_
    result_t_        = result.T_target_source;
    previous_result_t_ = result.T_target_source;
  } else {
    // Registration not converged, try relocalization if enabled
    RCLCPP_WARN(this->get_logger(), "GICP did not converge in normal matching.");
    if (enable_relocalization_) {
      performRelocalization();
    }
  }
}

// --------------------------------------------------
// When registration fails, perform relocalization
// --------------------------------------------------
void SmallGicpRelocalizationNode::performRelocalization()
{
  RCLCPP_INFO(this->get_logger(), "Start re-localization...");

  // Sample around previous_result_t_
  const double center_x   = previous_result_t_.translation().x();
  const double center_y   = previous_result_t_.translation().y();
  
  // Extract yaw (Z rotation) from rotation matrix using eulerAngles(0,1,2).z()
  Eigen::Vector3d eulers   = previous_result_t_.rotation().eulerAngles(0, 1, 2);
  double center_yaw        = eulers.z();  // In radians

  bool any_converged = false;
  double min_error   = std::numeric_limits<double>::max();
  small_gicp::RegistrationResult best_result;

  // Calculate yaw sampling step in radians
  double yaw_range_rad     = relocalization_yaw_range_deg_ * M_PI / 180.0;
  double yaw_step_rad      = relocalization_yaw_step_deg_  * M_PI / 180.0;

  // Sample within ranges:
  // x: [center_x - x_range, center_x + x_range]
  // y: [center_y - y_range, center_y + y_range]
  // yaw: [center_yaw - yaw_range_rad, center_yaw + yaw_range_rad]
  for (double dx = -relocalization_x_range_; dx <= relocalization_x_range_; dx += relocalization_x_step_) {
    for (double dy = -relocalization_y_range_; dy <= relocalization_y_range_; dy += relocalization_y_step_) {
      for (double dyaw = -yaw_range_rad; dyaw <= yaw_range_rad; dyaw += yaw_step_rad) {
        Eigen::Isometry3d guess = Eigen::Isometry3d::Identity();
        // Translation
        guess.translation().x() = center_x + dx;
        guess.translation().y() = center_y + dy;
        guess.translation().z() = previous_result_t_.translation().z();  // Keep z unchanged or define custom

        // Rotation: around Z axis
        Eigen::AngleAxisd rot_z(center_yaw + dyaw, Eigen::Vector3d::UnitZ());
        guess.linear() = rot_z.toRotationMatrix();

        // Perform complete GICP
        auto result = alignOnce(*target_, *source_, guess);
        if (result.converged) {
          // Record error 
          double error = result.error;
          if (error < min_error) {
            min_error   = error;
            best_result  = result;
            any_converged = true;
          }
        }
      }
    }
  }

  if (!any_converged) {
    RCLCPP_WARN(this->get_logger(), "Relocalization failed: no candidate converged.");
    return;
  }

  // Found optimal solution, update
  RCLCPP_INFO(this->get_logger(), "Relocalization success, min_error=%.4f", min_error);
  result_t_         = best_result.T_target_source;
  previous_result_t_ = best_result.T_target_source;
}

// --------------------------------------------------
// Publish TF
// --------------------------------------------------
void SmallGicpRelocalizationNode::publishTransform()
{
  if (result_t_.matrix().isZero()) {
    return;
  }

  geometry_msgs::msg::TransformStamped transform_stamped;
  // `+ 0.1` means transform into future. according to https://robotics.stackexchange.com/a/96615
  transform_stamped.header.stamp = last_scan_time_ + rclcpp::Duration::from_seconds(0.1);
  transform_stamped.header.frame_id = map_frame_;
  transform_stamped.child_frame_id  = odom_frame_;

  const Eigen::Vector3d translation = result_t_.translation();
  const Eigen::Quaterniond rotation(result_t_.rotation());

  transform_stamped.transform.translation.x = translation.x();
  transform_stamped.transform.translation.y = translation.y();
  transform_stamped.transform.translation.z = translation.z();
  transform_stamped.transform.rotation.x = rotation.x();
  transform_stamped.transform.rotation.y = rotation.y();
  transform_stamped.transform.rotation.z = rotation.z();
  transform_stamped.transform.rotation.w = rotation.w();

  tf_broadcaster_->sendTransform(transform_stamped);
}

// --------------------------------------------------
// Initial pose callback
// --------------------------------------------------
void SmallGicpRelocalizationNode::initialPoseCallback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  RCLCPP_INFO(
    this->get_logger(), "Received initial pose: [x: %f, y: %f, z: %f]", msg->pose.pose.position.x,
    msg->pose.pose.position.y, msg->pose.pose.position.z);

  Eigen::Isometry3d initial_pose;
  initial_pose.translation() << msg->pose.pose.position.x, msg->pose.pose.position.y,
    msg->pose.pose.position.z;
  initial_pose.linear() = Eigen::Quaterniond(
                            msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z)
                            .toRotationMatrix();

  previous_result_t_ = initial_pose;
  result_t_          = initial_pose;
}

}  // namespace small_gicp_relocalization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(small_gicp_relocalization::SmallGicpRelocalizationNode)