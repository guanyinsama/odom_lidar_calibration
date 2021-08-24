/*******************************************************
 * Copyright (C) 2019, SLAM Group, Megvii-R
 *******************************************************/

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <fstream>//zdf 
#include <math.h>
#include <chrono>

#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/io/io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include "camera_models/include/Camera.h"
#include "camera_models/include/CameraFactory.h"
#include "calc_cam_pose/calcCamPose.h"
#include "agv_msgs/AgvStatus.h"
#include "solveQyx.h"

using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

typedef boost::shared_ptr<nav_msgs::Odometry const> OdomConstPtr;

// 初始化点云可视化界面
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_ptr;
std::queue<OdomConstPtr> odo_buf;
std::queue<sensor_msgs::PointCloud2::ConstPtr> lidar_buf;
std::mutex m_buf;
bool hasImg = false;//zdf
std::vector<data_selection::odo_data> odoDatas;
std::vector<data_selection::cam_data> camDatas;
int turns = 0;
//record the first frame calculated successfully
bool fisrt_frame = true;
Eigen::Matrix3d Rwc0;
Eigen::Vector3d twc0;
//decide if the frequent is decreased
bool halfFreq = false;
int frame_index = 0;

void wheel_callback(const sensor_msgs::JointState &odo_msg)
{
    double time = odo_msg.header.stamp.toSec();
//    Eigen::Vector3d linear = {odo_msg->twist.twist.linear.x,
//                              odo_msg->twist.twist.linear.y,
//                              odo_msg->twist.twist.linear.z};
//    Eigen::Vector3d angular = {odo_msg->twist.twist.angular.x,
//                               odo_msg->twist.twist.angular.y,
//                               odo_msg->twist.twist.angular.z};
    data_selection::odo_data odo_tmp;

    odo_tmp.time = time;
//    odo_tmp.v_left = linear[0] / 0.1 - angular[2] * 0.56 / (2 * 0.1);// linear velcity of x axis
//    odo_tmp.v_right = linear[0] / 0.1 + angular[2] * 0.56 / (2 * 0.1);// angular velcity of z axis
    // odo_tmp.v_left = odo_msg.leftspeed;
    // odo_tmp.v_right = odo_msg.rightspeed;
    odo_tmp.v_left = odo_msg.velocity.data()[0];
    odo_tmp.v_right = odo_msg.velocity.data()[1];
    std::cout << odo_tmp.v_left << std::endl;
    std::cout << odo_tmp.v_right << std::endl;
    odoDatas.push_back(odo_tmp);
}

void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg) {
    if (!halfFreq) {
        m_buf.lock();
        lidar_buf.push(lidar_msg);
        cout << "lidar_msg->data.size1()" << endl;
        cout << lidar_msg->data.size() << endl;
        m_buf.unlock();
    } else {
        frame_index++;
        if (frame_index % 2 == 0) {
            m_buf.lock();
            lidar_buf.push(lidar_msg);
            cout << "lidar_msg->data.size()" << endl;
            cout << lidar_msg->data.size() << endl;
            m_buf.unlock();
        }
    }
}

void viewTransformedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &source_frame,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr &target_frame, Eigen::Matrix4d tranform) {

    viewer_ptr = boost::shared_ptr<pcl::visualization::PCLVisualizer>(
            new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_ptr->setBackgroundColor(0, 0, 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_frame, *transformed_source_cloud, tranform);

    //对目标点云着色（红色）并可视化
    viewer_ptr->removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            target_color(target_frame, 255, 0, 0);
    viewer_ptr->addPointCloud<pcl::PointXYZ>(target_frame, target_color, "target cloud");
    viewer_ptr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, "target cloud");
    //对转换后的目标点云着色（绿色）并可视化
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            output_color(transformed_source_cloud, 0, 255, 0);
    viewer_ptr->addPointCloud<pcl::PointXYZ>(transformed_source_cloud, output_color, "output cloud");
    viewer_ptr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, "output cloud");
//    //对转换后的目标点云着色（蓝色）并可视化
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
//            source_color(transformed_source_cloud, 0, 0, 255);
//    viewer_ptr->addPointCloud<pcl::PointXYZ>(source_frame, source_color, "source cloud");
//    viewer_ptr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
//                                                 1, "source cloud");
    // 启动可视化
    viewer_ptr->addCoordinateSystem(1.0);
    viewer_ptr->initCameraParameters();
    //等待直到可视化窗口关闭。

    viewer_ptr->spinOnce(4000);
    // boost::this_thread::sleep(boost::posix_time::microseconds(1));
}

void viewCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &source_frame) {
    //对目标点云着色（红色）并可视化
    viewer_ptr = boost::shared_ptr<pcl::visualization::PCLVisualizer>(
            new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_ptr->setBackgroundColor(0, 0, 0);
    viewer_ptr->removeAllPointClouds();
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
//            target_color(source_frame, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            target_color(source_frame, 255, 0, 0);

    viewer_ptr->addPointCloud<pcl::PointXYZ>(source_frame, target_color, "sample cloud");
    viewer_ptr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 2, "sample cloud");
    // 启动可视化
    viewer_ptr->addCoordinateSystem(1.0);
    viewer_ptr->initCameraParameters();
    //等待直到可视化窗口关闭。
//    while (!viewer_ptr->wasStopped())
//    {
//        viewer_ptr->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
    viewer_ptr->spinOnce(4000);
    // boost::this_thread::sleep(boost::posix_time::microseconds(1));
}

void removeGround(pcl::PointCloud<pcl::PointXYZ>::Ptr &origin_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr &result_ptr) {
    //进行地面滤波算法，将场景中的地面点予以滤出，防止这些点影响场景特征，从而导致匹配出错；
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    Eigen::Vector3f ground_vec;
    ground_vec << 0, 0, 1;
    seg.setAxis(ground_vec);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.3);
    seg.setInputCloud(origin_ptr);
    seg.setMaxIterations(200);//最大迭代次数为100
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(origin_ptr);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*result_ptr);
}

void ndt(pcl::PointCloud<pcl::PointXYZ>::Ptr &source_frame,
         pcl::PointCloud<pcl::PointXYZ>::Ptr &target_frame, Eigen::Matrix4d &raw_transform,
         Eigen::Matrix4d &result_transform) {
    //将惯导数据作为初值赋值给NDT算法，计算激光雷达的旋转平移矩阵；
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    // 设置NDT参数
    // 设置终止转换误差条件；
    ndt.setTransformationEpsilon(0.0001);
    // 设置最大迭代步长；
    ndt.setStepSize(0.005);
    // 设置NDT网格结构分辨率；
    ndt.setResolution(1.0);
    // 设置最大迭代步数；
    ndt.setMaximumIterations(100);
    // 设置输入点云；
    ndt.setInputSource(source_frame);
    // 设置目标点云；
    ndt.setInputTarget(target_frame);
    // output_cloud为配准后的输出点云；
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*output_cloud, raw_transform.cast<float>());

    result_transform = ndt.getFinalTransformation().cast<double>();
}

double icp(pcl::PointCloud<pcl::PointXYZ>::Ptr &source_frame,
           pcl::PointCloud<pcl::PointXYZ>::Ptr &target_frame, Eigen::Matrix4d &raw_transform,
           Eigen::Matrix4d &result_transform) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_frame);
    icp.setInputTarget(target_frame);
    icp.setMaxCorrespondenceDistance(10);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(0.0001);
    icp.setMaximumIterations(3000);

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*output_cloud, raw_transform.cast<float>());
    result_transform = icp.getFinalTransformation().cast<double>();

    return icp.getFitnessScore();
}

void removeNaN(pcl::PointCloud<pcl::PointXYZ>::Ptr &source_frame) {
    std::vector<int> indices_src;
    pcl::removeNaNFromPointCloud(*source_frame, *source_frame, indices_src);
}

void getRoughtresult(pcl::PointCloud<pcl::PointXYZ>::Ptr &last_frame,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &current_frame,
                     Eigen::Matrix4d &result_transform) {
    //下采样滤波
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setLeafSize(0.012, 0.012, 0.012);
    voxel_grid.setInputCloud(last_frame);
    PointCloud::Ptr cloud_src(new PointCloud);
    voxel_grid.filter(*cloud_src);
    std::cout << "down size *cloud_import from " << last_frame->size() << "to" << cloud_src->size() << endl;
    //pcl::io::savePCDFileASCII("bunny_src_down.pcd", *cloud_src);
    //计算表面法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_src;
    ne_src.setInputCloud(cloud_src);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree<pcl::PointXYZ>());
    ne_src.setSearchMethod(tree_src);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud<pcl::Normal>);
    ne_src.setRadiusSearch(0.02);
    ne_src.compute(*cloud_src_normals);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
    voxel_grid_2.setLeafSize(0.01, 0.01, 0.01);
    voxel_grid_2.setInputCloud(current_frame);
    PointCloud::Ptr cloud_tgt(new PointCloud);
    voxel_grid_2.filter(*cloud_tgt);
    std::cout << "down size *cloud_tgt_o.pcd from " << current_frame->size() << "to" << cloud_tgt->size() << endl;
    //pcl::io::savePCDFileASCII("bunny_tgt_down.pcd", *cloud_tgt);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_tgt;
    ne_tgt.setInputCloud(cloud_tgt);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree<pcl::PointXYZ>());
    ne_tgt.setSearchMethod(tree_tgt);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud<pcl::Normal>);
    //ne_tgt.setKSearch(20);
    ne_tgt.setRadiusSearch(0.02);
    ne_tgt.compute(*cloud_tgt_normals);

    //计算FPFH
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
    fpfh_src.setInputCloud(cloud_src);
    fpfh_src.setInputNormals(cloud_src_normals);
    pcl::search::KdTree<PointT>::Ptr tree_src_fpfh(new pcl::search::KdTree<PointT>);
    fpfh_src.setSearchMethod(tree_src_fpfh);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_src.setRadiusSearch(0.05);
    fpfh_src.compute(*fpfhs_src);
    std::cout << "compute *cloud_src fpfh" << endl;

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_tgt;
    fpfh_tgt.setInputCloud(cloud_tgt);
    fpfh_tgt.setInputNormals(cloud_tgt_normals);
    pcl::search::KdTree<PointT>::Ptr tree_tgt_fpfh(new pcl::search::KdTree<PointT>);
    fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_tgt.setRadiusSearch(0.05);
    fpfh_tgt.compute(*fpfhs_tgt);
    std::cout << "compute *cloud_tgt fpfh" << endl;

    //SAC配准
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
    scia.setInputSource(cloud_src);
    scia.setInputTarget(cloud_tgt);
    scia.setSourceFeatures(fpfhs_src);
    scia.setTargetFeatures(fpfhs_tgt);
    //scia.setMinSampleDistance(1);
    //scia.setNumberOfSamples(2);
    //scia.setCorrespondenceRandomness(20);
    PointCloud::Ptr sac_result(new PointCloud);
    scia.align(*sac_result);
    std::cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
    result_transform = scia.getFinalTransformation().cast<double>();
    // std::cout << result_transform << endl;
    //pcl::io::savePCDFileASCII("junjie_transformed_sac.pcd", *sac_result);
};

double getLidarCurrent2LastRt(pcl::PointCloud<pcl::PointXYZ>::Ptr &last_frame,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &current_frame,
                              Eigen::Matrix4d &result_transform) {
    Eigen::Matrix4d ndt_result_transform;
    removeNaN(current_frame);
    removeNaN(last_frame);
    Eigen::Matrix4d result_transform_rough;
    getRoughtresult(last_frame, current_frame, result_transform_rough);
    // cout<<result_transform_rough<<endl;
    ndt(current_frame, last_frame, result_transform_rough, ndt_result_transform);
    // cout<<ndt_result_transform<<endl;
    return icp(current_frame, last_frame, ndt_result_transform, result_transform);
}

// extract images with same timestamp from two topics
void calc_process() {
    Eigen::Matrix3d Rwl;
    Eigen::Vector3d twl;
    double t_last = 0.0;//time of last image
    bool first = true; //judge if last frame was calculated successfully

    std::cout << std::endl << "images counts waiting for processing: " << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame(new pcl::PointCloud<pcl::PointXYZ>());
    while (1) {
        std_msgs::Header header;
        double time = 0;
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame_with_ground(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_frame(new pcl::PointCloud<pcl::PointXYZ>());
        // cout<<"lidar_buf"<<lidar_buf.size()<<endl;
        // cout<<"odo_buf"<<odo_buf.size()<<endl;
        m_buf.lock();
        if (!lidar_buf.empty()) {
            cout<<"lidar_buf________"<<lidar_buf.size()<<endl;
            time = lidar_buf.front()->header.stamp.toSec();
            header = lidar_buf.front()->header;
            pcl::fromROSMsg(*lidar_buf.front(), *current_frame_with_ground);
            cout<<"去NAN前点云的数量:"<<current_frame_with_ground->size()<<endl;
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*current_frame_with_ground, *current_frame_with_ground, indices);
            cout<<"去NAN后去地前点云的数量:"<<current_frame_with_ground->size()<<endl;
            //viewCloud(current_frame_with_ground);
            removeGround(current_frame_with_ground,current_frame);
            cout<<"去地后点云的数量:"<<current_frame->size()<<endl;
            // viewCloud(current_frame);
            lidar_buf.pop();
        }

        m_buf.unlock();
        // cout<<"hasImg:"<<hasImg<<endl;

        if (!current_frame->empty()) {
            cout<<"turn:"<<turns<<endl;
            if(turns>10){
                hasImg = true;
            }
            if (fisrt_frame) {
                fisrt_frame = false;
                continue;
            }
            //judge if the last frame was calculated successfully
            if (!first) {
                //Todo zdf
                // Eigen::Vector3d eulerAngle_wc = Rwc.eulerAngles(2,1,0);//ZYX
                // Eigen::Vector3d eulerAngle_wl = Rwl.eulerAngles(2,1,0);
                // double theta_y = eulerAngle_wc[1] - eulerAngle_wl[1];

                Eigen::Matrix4d Tlc;
                getLidarCurrent2LastRt(last_frame, current_frame, Tlc);
                cout<<"lidar transform:"<<Tlc<<endl;
                //viewTransformedCloud(current_frame, last_frame, Tlc);
                Eigen::Matrix3d Rlc = Tlc.block<3, 3>(0, 0);
                Eigen::Vector3d tlc = Tlc.block<3, 1>(0, 3);

                Eigen::Quaterniond q_cl(Rlc);

                Eigen::AngleAxisd rotation_vector(q_cl);
                Eigen::Vector3d axis = rotation_vector.axis();
                double deltaTheta_cl = rotation_vector.angle();
                if (axis(2) < 0) {
                    deltaTheta_cl *= -1;
                    axis *= -1;
                }

                //Eigen::Vector3d tcl = -Rwc.inverse() * (twc - twl);
//                Eigen::Vector3d tlc = -Rwl.inverse() * (twl - twc);

                data_selection::cam_data cam_tmp;
                cam_tmp.start_t = t_last;
                cam_tmp.end_t = time;
                //cam_tmp.theta_y = theta_y;
                cam_tmp.deltaTheta = deltaTheta_cl; // cam_tmp.deltaTheta is deltaTheta_lc
                cam_tmp.axis = axis;
                cam_tmp.Rcl = Rlc;
                cam_tmp.tlc = tlc;
                camDatas.push_back(cam_tmp);
                turns=turns+1;
            }
            t_last = time;
            last_frame = current_frame;
            first = false;

        } else {
            if (hasImg) {
                SolveQyx cSolveQyx;

                std::cout << "============ calibrating... ===============" << std::endl;
                data_selection ds;
                std::vector<data_selection::sync_data> sync_result;
                ds.selectData(odoDatas, camDatas, sync_result);

                //first estimate the Ryx and correct tlc of camera
                Eigen::Matrix3d Ryx;
                std::cout << "============ calibrating...Ryx ===============" << std::endl;
                cSolveQyx.estimateRyx(sync_result, Ryx);
                std::cout << "============ calibrating...correctCamera ===============" << std::endl;
                cSolveQyx.correctCamera(sync_result, camDatas, Ryx);

                //calibrate r_L  r_R  axle  lx  ly  yaw
                cSolver cSolve;
                cSolver::calib_result paras;//radius_l,radius_r,axle,l[3]
                std::cout << "============ calibrating...paras ===============" << std::endl;
                cSolve.calib(sync_result, 4, paras); // by svd

                //secondly estimate the Ryx
                std::cout << "============ calibrating...Ryx ===============" << std::endl;
                cSolveQyx.estimateRyx(sync_result, Ryx);

                //refine all the extrinal parameters
                std::cout << "============ calibrating...all paras ===============" << std::endl;
                cSolveQyx.refineExPara(sync_result, paras, Ryx);

                break;
            }
        }

    }

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "vins_fusion");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    cout << "start" << endl;
    std::string config_file = argv[1];
    cout << config_file << endl;
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    std::string WHEEL_TOPIC, LIDAR_TOPIC;
    fsSettings["wheel_topic"] >> WHEEL_TOPIC;
    fsSettings["lidar_topic"] >> LIDAR_TOPIC;
    cout << WHEEL_TOPIC << endl;
    cout << LIDAR_TOPIC << endl;

//    CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(config_file);

    //the following three rows are to run the calibrating project through playing bag package
    ros::Subscriber sub_imu = n.subscribe(WHEEL_TOPIC, 500, wheel_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img = n.subscribe(LIDAR_TOPIC, 200, lidar_callback);
    std::thread calc_thread = std::thread{calc_process};


    ros::spin();
    return 0;
}
